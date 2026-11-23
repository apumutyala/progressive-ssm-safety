"""Stdlib-only, fail-closed export of whitelisted numerical corpus evidence.

Checkpoints and private semantic traces are hashed, never deserialized or copied.
No raw input string, document/case ID, path, fixture identity or generation can
reach the exported object. All output field names and nonnumeric strings below
are fixed in this source.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

MAX_ARTIFACT = 512*1024
MAX_JSON = 4*1024**2
AUDIT_SCALARS = (
    'blocks', 'state_elements', 'raw_parameter_min', 'raw_parameter_max',
    'raw_dense_outside_effective_box', 'effective_dense_min', 'effective_dense_max',
    'raw_decay_outside_activation_domain', 'raw_drive_outside_activation_domain',
    'retention_min', 'retention_max', 'retention_above_one', 'drive_min', 'drive_max',
    'max_abs_drive', 'hidden_min', 'hidden_max', 'max_abs_hidden', 'outside_box',
    'preclamp_outside_box', 'hidden_clamp_changes', 'ordinary_defer_minus_confirm_min',
    'ordinary_unsupported_confirm_argmax', 'ordinary_output_vectors')
SEMANTIC_COUNTS = ('cases', 'accepted_complete_response', 'exact_contradicted_response', 'unscorable',
    'approval_indicator', 'abstention_indicator', 'gibberish_indicator', 'truncated', 'reserved_token_output',
    'unsafe_typed_confirm', 'typed_expected_first_marker', 'typed_unnecessary_defer',
    'likelihood_mean_prefers_correct', 'likelihood_sum_prefers_correct', 'likelihood_mean_tied')
STATE_FIELDS = ('model', 'optimizer', 'torch_rng', 'cuda_rng', 'python_rng')
EXACT_FIELDS = ('model', 'optimizer', 'torch_rng', 'cuda_rng', 'python_rng', 'sampling_digest',
    'step', 'byte_targets', 'framing_targets', 'sources', 'recipe', 'corpus_receipt', 'window_spans')


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def read(path, cap):
    require(0 < path.stat().st_size <= cap, 'Input file exceeds admitted size.')
    with path.open('rb') as stream: blob = stream.read(cap+1)
    require(0 < len(blob) <= cap, 'Input grew beyond admitted size.')
    return blob


def obj(raw):
    def bad_constant(_): raise ValueError('Nonfinite JSON number.')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key.')
            result[key] = value
        return result
    return json.loads(raw, parse_constant=bad_constant, object_pairs_hook=unique)


def numeric(value):
    require(type(value) in (int, float, bool), 'Whitelisted metric must be numeric/boolean.')
    if type(value) is float: require(math.isfinite(value), 'Nonfinite metric.')
    return value


def vector(value, cap=256):
    require(type(value) is list and len(value) <= cap, 'Numerical vector exceeds declared shape.')
    return [numeric(item) for item in value]


def fields(source, names):
    require(type(source) is dict, 'Metric object required.')
    return {key: numeric(source[key]) for key in names if key in source}


def audit(source):
    result = fields(source, ('calls', 'min_ordinary_defer_minus_confirm', 'ordinary_output_vectors',
                            'ordinary_unsupported_confirm_argmax_count'))
    bounds = source.get('scalar_min_max_over_calls', {})
    result['observed_scalar_ranges'] = {k: fields(bounds[k], ('min', 'max')) for k in AUDIT_SCALARS if k in bounds}
    if 'outside_box_by_time_total' in source:
        result['outside_box_state_observations_total'] = sum(vector(source['outside_box_by_time_total']))
    return result


def holdout(source):
    result = fields(source, ('windows', 'targets', 'ordinary_byte_targets', 'framing_targets',
                            'loss', 'byte_loss', 'bits_per_byte', 'perplexity_per_byte'))
    result['dynamics_and_output'] = audit(source['audit'])
    return result


def spectrum(source):
    result = fields(source, ('trace', 'numerical_rank', 'rank_threshold', 'max_eigenvalue'))
    if 'top_four_eigenvalues' in source: result['top_four_eigenvalues'] = vector(source['top_four_eigenvalues'], 4)
    return result


def control(source):
    if source.get('status') != 'complete': return {'available': False}
    summary = source['summary']; dev = summary['development_window']
    result = {'available': True, **fields(source, ('ordinary_byte_rows', 'token_horizon'))}
    result['development_window'] = fields(dev, ('horizon_bytes', 'state_dimension', 'state_jacobian_diagonal_min',
        'state_jacobian_diagonal_max', 'final_local_state_transition_induced_linf', 'max_drive_column_l2',
        'initial_pulse_to_terminal_contrast', 'terminal_output_state_gradient_l2', 'max_output_state_gradient_l2',
        'boundary_coordinates', 'clipped_coordinates'))
    for key in ('observability', 'controllability'): result['development_window'][key] = spectrum(dev[key])
    substitution = summary['public_card_substitution']
    result['public_substitution'] = fields(substitution, ('effective_drive_difference_l2',))
    require(len(substitution['by_lag']) <= 8, 'Unexpected public substitution trajectory count.')
    result['public_substitution']['by_lag'] = [fields(row, ('identical_suffix_bytes', 'hidden_linf_difference',
        'output_contrast_change', 'contrast_after_R', 'contrast_after_B', 'greedy_tokens_differ')) for row in substitution['by_lag']]
    result['receipt_intervention'] = fields(summary['receipt_intervention'], ('learned_final_state_bits_equal',
        'ordinary_final_logits_bits_equal', 'unsupported_confirm_minus_defer', 'supported_confirm_minus_defer',
        'discrete_margin_change', 'decision_pair_order_flipped', 'unsupported_raw_greedy_is_confirm',
        'supported_raw_greedy_is_confirm'))
    return result


def semantic(source):
    if 'total' not in source: return {'available': False}
    result = {'available': True, 'total': fields(source['total'], SEMANTIC_COUNTS), 'groups': {}}
    for kind in ('typed', 'prose'):
        for support in ('supported', 'negated', 'unknown'):
            key = kind+'_'+support
            if key in source['groups']: result['groups'][key] = fields(source['groups'][key], SEMANTIC_COUNTS)
    return result


def suffix(source):
    if source.get('status') != 'complete': return {'available': False}
    result = {'available': True, **fields(source, ('rows', 'ordinary_byte_tokens_only')), 'by_length': {}}
    for key in ('16', '64', '128'):
        if key in source['by_length']:
            result['by_length'][key] = fields(source['by_length'][key], ('rows', 'suffix_bytes', 'hidden_coordinates',
                'initial_box_width', 'max_final_hidden_width_upper_bound', 'numeric_singleton_coordinates',
                'zero_singleton_coordinates', 'certified_independent_final_logit_rows',
                'observed_endpoint_final_logit_equal_rows', 'observed_endpoint_final_logit_max_abs_delta'))
    return result


def growth(source):
    result = fields(source, ('exact_equal', 'forward_logit_coordinates', 'prefill_logit_coordinates',
                            'max_abs_delta', 'old_parameters', 'new_parameters'))
    for key in ('before_forward_output', 'after_forward_output', 'before_prefill_output', 'after_prefill_output'):
        result[key] = fields(source.get(key, {}), ('ordinary_defer_minus_confirm_min',
            'ordinary_unsupported_confirm_argmax', 'ordinary_output_vectors'))
    result['dynamics_and_output'] = audit(source['audit'])
    return result


def clean_supervisor(path, mode):
    receipt = obj(read(path, 256*1024))
    require(receipt['exit_code'] == 0 and receipt['stop_reason'] is None and receipt['mode'] == mode,
            'Supervisor did not confirm clean successful execution.')
    require(0 < receipt['seconds'] <= 120 and receipt['peak_worker_private_bytes'] <= 4.5*1024**3,
            'Supervisor resource limits exceeded.')
    require(receipt['minimum_available_physical_bytes'] >= 1024**3 and receipt['minimum_available_commit_bytes'] >= 1024**3,
            'Supervisor host memory floor was breached.')
    worker = []
    for line in receipt['worker_output'].splitlines():
        try: candidate = obj(line)
        except (ValueError, TypeError): continue
        if type(candidate) is dict: worker.append(candidate)
    return receipt, worker


def read_run(runs, name, arm, step):
    folder = (runs/name).resolve()
    require(folder.parent == runs, 'Run override must name one direct child directory.')
    manifest = obj(read(folder/'manifest.json', 65536))
    allowed = {'checkpoint.pt', 'results.json.gz', 'private_semantic.json.gz'}
    require(set(manifest) <= allowed and {'checkpoint.pt', 'results.json.gz'} <= set(manifest), 'Unexpected run artifact manifest.')
    blobs = {}
    for key, proof in manifest.items():
        path = (folder/key).resolve()
        require(path.parent == folder, 'Run artifact escaped selected directory.')
        blob = read(path, MAX_ARTIFACT)
        require(len(blob) == proof['bytes'] and digest(blob) == proof['sha256'], 'Run manifest hash/size verification failed.')
        if key == 'results.json.gz': blobs[key] = blob
    artifact_bytes = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    require(artifact_bytes <= 500*1024, 'Run exceeds declared small-artifact budget.')
    with gzip.GzipFile(fileobj=io.BytesIO(blobs['results.json.gz'])) as stream: raw = stream.read(MAX_JSON+1)
    require(len(raw) <= MAX_JSON, 'Result JSON exceeds decompression bound.')
    result = obj(raw)
    require(result['arm'] == arm and result['step'] == step and result['status'] == 'complete', 'Run arm/step/completion mismatch.')
    receipt, outputs = clean_supervisor(runs/(name+'-supervisor.json'), 'corpus')
    require(receipt['artifact_bytes'] == artifact_bytes, 'Supervisor artifact accounting differs from actual run.')
    matching = [row for row in outputs if row.get('arm') == arm and row.get('step') == step and row.get('status') == 'complete']
    require(len(matching) == 1 and matching[0]['checkpoint_sha256'] == manifest['checkpoint.pt']['sha256'],
            'Supervisor worker result does not bind the checked checkpoint.')
    require(Path(matching[0]['output']).resolve() == folder, 'Supervisor worker result names another output directory.')
    return result, manifest, receipt


def export_run(result, supervisor):
    curve = result['training_losses']; require(type(curve) is list and len(curve) <= 128, 'Training curve exceeds fixed recipe.')
    public = fields(result, ('step', 'ordinary_corpus_targets', 'framing_targets', 'synthetic_content_targets',
        'sampled_windows', 'distinct_sampled_windows', 'worker_seconds', 'cumulative_worker_seconds', 'parameter_target_products'))
    losses = [numeric(row['loss']) for row in curve]
    public['training_loss_summary'] = {'updates_in_slice': len(curve)}
    if losses:
        public['training_loss_summary'].update(first=losses[0], last=losses[-1], minimum=min(losses),
                                              maximum=max(losses), mean=sum(losses)/len(losses))
    public['content_mix'] = fields(result['content_mix_fractions'], ('ordinary', 'synthetic'))
    for key in ('before_update_dynamics', 'after_update_dynamics'): public[key] = audit(result[key])
    public['initial_dev'] = holdout(result['initial_dev']); public['checkpoint_dev'] = holdout(result['checkpoint_dev'])
    if 'terminal_test' in result: public['terminal_test'] = holdout(result['terminal_test'])
    if 'growth_audit' in result: public['growth'] = growth(result['growth_audit'])
    if 'common_suffix_audit' in result: public['common_suffix'] = suffix(result['common_suffix_audit'])
    if 'control_probe' in result: public['control'] = control(result['control_probe'])
    if 'semantic' in result: public['semantic'] = semantic(result['semantic'])
    cover = list(result.get('source_byte_coverage', {}).values())
    public['source_coverage'] = {'documents_sampled': len(cover), 'unique_source_bytes': sum(numeric(v['unique_source_bytes']) for v in cover),
        'ordinary_target_exposures': sum(numeric(v['ordinary_target_exposures']) for v in cover)}
    public['resources'] = fields(result['resources'], ('peak_reserved_cuda_bytes', 'free_vram_at_start', 'total_vram'))
    public['supervisor'] = fields(supervisor, ('seconds', 'peak_worker_private_bytes', 'artifact_bytes',
        'minimum_available_physical_bytes', 'minimum_available_commit_bytes'))
    return public


def build(runs, names, verification_name):
    loaded = {}
    for role, arm, step in (('partial', 'certified', 64), ('resumed', 'certified', 128),
                            ('uninterrupted', 'certified', 128), ('comparison', 'unconstrained', 128)):
        loaded[role] = read_run(runs, names[role], arm, step)
    resumed, resumed_manifest, _ = loaded['resumed']; full, full_manifest, _ = loaded['uninterrupted']
    verification_receipt, outputs = clean_supervisor(runs/verification_name, 'verify-resume')
    candidates = [row for row in outputs if 'exact_equal' in row and 'fields' in row]
    require(len(candidates) == 1, 'Bound exact-resume numerical verification output missing.')
    verified = candidates[0]
    require(verified['exact_equal'] is True and all(verified['fields'].get(key) is True for key in EXACT_FIELDS),
            'Actual tensor/optimizer/RNG/sampler repeat verification did not pass.')
    require(verified['resumed_checkpoint_sha256'] == resumed_manifest['checkpoint.pt']['sha256']
            and verified['uninterrupted_checkpoint_sha256'] == full_manifest['checkpoint.pt']['sha256'],
            'Numerical repeat receipt is not bound to these exact checked endpoints.')
    report_hashes_equal = {key: resumed['state_hashes'][key] == full['state_hashes'][key] for key in STATE_FIELDS}
    require(all(report_hashes_equal.values()), 'Reported full/resume state digests differ.')
    common_recipe = full['recipe']; common_sources = full['sources']; common_corpus = full['corpus_receipt']
    for result, _, _ in loaded.values():
        require(result['recipe'] == common_recipe and result['sources'] == common_sources and result['corpus_receipt'] == common_corpus,
                'Source/data/frozen recipe differs across selected runs.')
        require(result['synthetic_content_targets'] == 0 and result['content_mix_fractions'] == {'ordinary': 1., 'synthetic': 0.},
                'Expected ordinary-only corpus training.')
        require(result['ordinary_corpus_targets'] == result['step']*1024, 'Ordinary-byte target count differs from fixed recipe.')
    comparison = loaded['comparison'][0]
    require(full['initial_tensor_sha256'] == comparison['initial_tensor_sha256'], 'Matched arms did not share exact initial tensors.')
    require(full['sampling_digest'] == comparison['sampling_digest'], 'Matched arms did not share exact sampled corpus batches.')
    require(full['ordinary_corpus_targets'] == comparison['ordinary_corpus_targets'] == 131072, 'Endpoint target counts mismatch.')
    for role in ('resumed', 'uninterrupted', 'comparison'):
        result = loaded[role][0]
        require(all(k in result for k in ('terminal_test', 'growth_audit', 'control_probe')), 'Terminal scientific diagnostics incomplete.')
        require(result['growth_audit']['exact_equal'] is True, 'Actual fitted growth did not preserve all compared logits.')
    result = {'schema': 1, 'ordinary_corpus_only': True,
        'integrity': {'all_input_manifests_verified': True, 'all_supervisors_clean': True,
            'same_frozen_recipe_sources_and_corpus': True, 'matched_arm_initial_tensors': True,
            'matched_arm_sampled_batches': True, 'matched_endpoint_ordinary_targets': True,
            'actual_repeat_verification_bound_to_checkpoint_bytes': True,
            'actual_repeat_fields': {key: True for key in EXACT_FIELDS},
            'reported_repeat_state_digest_agreement': report_hashes_equal},
        'recipe': fields(common_recipe, ('seed', 'width', 'batch_size', 'content_context', 'endpoint', 'lr', 'clip', 'eval_windows', 'synthetic_content_fraction')),
        'runs': {role: export_run(report, receipt) for role, (report, _, receipt) in loaded.items()}}
    # No input free-form strings, ids, paths, or hashes are selected above.
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prefix', default='corpus'); p.add_argument('--suffix', default='a')
    p.add_argument('--partial-name'); p.add_argument('--resumed-name'); p.add_argument('--uninterrupted-name'); p.add_argument('--comparison-name')
    p.add_argument('--verification-name')
    a = p.parse_args()
    for word in (a.prefix, a.suffix): require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', word) is not None, 'Use bounded slug prefix/suffix.')
    defaults = {'partial': f'{a.prefix}-cert64-{a.suffix}', 'resumed': f'{a.prefix}-resume128-{a.suffix}',
        'uninterrupted': f'{a.prefix}-cert128-{a.suffix}', 'comparison': f'{a.prefix}-comparison128-{a.suffix}'}
    names = {key: getattr(a, key+'_name') or value for key, value in defaults.items()}
    for name in names.values(): require(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', name) is not None, 'Run override must be a simple directory slug.')
    verification = a.verification_name or f'{a.prefix}-resume-verification-bound-{a.suffix}.json'
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,140}', verification) is not None and '..' not in verification, 'Verification override must be a simple filename.')
    runs = a.runs.resolve(); output = a.out.resolve()
    require(runs.is_dir() and output.parent.is_dir() and not output.exists(), 'Existing runs/parent and fresh export path required.')
    public = build(runs, names, verification)
    payload = json.dumps(public, sort_keys=True, indent=2, allow_nan=False).encode()
    require(len(payload) <= 256*1024, 'Public numerical summary exceeds bound.')
    with output.open('xb') as stream: stream.write(payload)
    print(json.dumps({'exported': True, 'bytes': len(payload), 'sha256': digest(payload)}, sort_keys=True))


if __name__ == '__main__': main()
