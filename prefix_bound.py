"""CPU endpoint verifier for forgetting learned state after a fixed suffix.

THEOREM (conditional on the numerical backend): each coordinate of this model's
learned recurrence is a monotone FP32 map of its previous coordinate. Tokens
fix nonnegative retention and a fixed injection. Rounded multiplication by a
nonnegative finite value, addition of a fixed finite value, and clamp preserve
numeric order under conventional IEEE round-to-nearest, without NaNs. Thus
running the actual block on all-minus-one and all-plus-one states sandwiches
every admitted learned initial state in [-1,1]^width. Different blocks have
independent state transitions, so the argument applies to their product.

If all final endpoints are the same NONZERO FP32 value, each intermediate
admissible initial state gives that same final bit pattern. The nonlinear mix
and vocabulary head depend only on this final state at that position. Under
deterministic row-local arithmetic, the final full raw logit vector is therefore
independent of unknown prior learned state, with identical protected premises.
This is a property of the learned recurrence, not of a reserved decision token.

Both endpoints use Model._score -> SelectiveBlock.forward, including the existing
sequence shape, elementwise operations, and head path. No analytic replacement,
different batch packing, scan, float64 recurrence, or tolerance establishes
coalescence. Widths alone are computed in float64 from the observed FP32 values.
Matching endpoint final logits is a sanity check, not an alternative theorem.

SIGNED ZERO: ordinary numeric monotonicity does not distinguish +0 from -0.
The verifier compares FP32 bits and conservatively declines certification if
any singleton coordinate is zero, even if both endpoint zero signs match.
This avoids assuming an unproved total-order property for signed zeros.

The certificate is local to a fixed suffix, fixed model, fixed batch/sequence
shape and backend, and identical full protected-premise trajectory. It is not
exact prefill/decode parity, a bound on unknown protected context, semantic
safety, general forgetting, or evidence of useful language behavior. All states
must be finite and within the proved invariant box. Same-shaped deterministic
row-local kernels, no autocast/TF32/quantization, no faults and no NaNs are
assumptions. A different backend needs a new receipt. Long-retention modes may
never coalesce within the admitted suffix budget; that is an inconclusive
certificate and must be reported with the maximum final hidden interval width.

No GPU is selected. Run this diagnostic through the reviewed numerical-worker
supervisor. The optional CLI accepts only a digest-identified trusted local
checkpoint, uses weights_only=True, and does not install or download anything.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch


def same_fp32_bits(left, right):
    """Elementwise equality including signed-zero bits; NaNs are rejected later."""
    if left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape:
        raise ValueError("bit comparison needs same-shaped float32 tensors")
    return left.contiguous().view(torch.int32) == right.contiguous().view(torch.int32)


@torch.no_grad()
def verify_suffix(model, tokens, protected_premises=None):
    """Return a JSON-ready conditional certificate or an inconclusive width bound.

    protected_premises is the identical internal [B,T,2] Boolean trajectory,
    not raw events/receipts and not a claim that the metadata is authoritative.
    None explicitly fixes both premises false at every position.
    """
    if tokens.device.type != "cpu" or tokens.dtype != torch.int64:
        raise ValueError("this bounded verifier admits CPU int64 tokens only")
    if tokens.ndim != 2 or not 1 <= tokens.size(0) <= 8 or not 1 <= tokens.size(1) <= 512:
        raise ValueError("suffix shape must be [batch<=8, 1..512 tokens]")
    if not hasattr(model, "_score") or not hasattr(model, "blocks"):
        raise TypeError("expected the reviewed diagonal selective Model family")
    if not 1 <= len(model.blocks) <= 8 or any(block.width > 128 for block in model.blocks):
        raise ValueError("diagnostic admits at most eight blocks of width <=128")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("CPU-only diagnostic; model must already be on CPU")
    if tokens.min().item() < 0 or tokens.max().item() >= model.vocab_size:
        raise ValueError("suffix token is outside the configured vocabulary")
    model.validate_parameters()
    if protected_premises is None:
        protected_premises = torch.zeros((*tokens.shape, 2), dtype=torch.bool)
    if (protected_premises.dtype != torch.bool or protected_premises.device != tokens.device
            or protected_premises.shape != (*tokens.shape, 2)):
        raise ValueError("protected_premises must be CPU bool [batch,time,2]")
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("this diagnostic requires highest FP32 matmul precision")
    low_initial = tuple(torch.full((tokens.size(0), block.width), -1.0) for block in model.blocks)
    high_initial = tuple(torch.full((tokens.size(0), block.width), 1.0) for block in model.blocks)
    low_logits, low = model._score(tokens, low_initial, protected_premises)
    high_logits, high = model._score(tokens, high_initial, protected_premises)
    if not all(torch.isfinite(t).all().item() for t in (*low, *high, low_logits, high_logits)):
        raise FloatingPointError("nonfinite endpoint execution; certificate rejected")
    blocks = []
    for index, (lower, upper) in enumerate(zip(low, high)):
        if not ((lower <= upper).all() and (lower >= -1).all() and (upper <= 1).all()).item():
            raise RuntimeError("observed endpoint ordering/invariant failure")
        equal_bits = same_fp32_bits(lower, upper)
        zero_singleton = (lower == 0) & (upper == 0)
        blocks.append({
            "block": index, "coordinates": lower.numel(),
            "equal_bit_coordinates": int(equal_bits.sum()),
            "unequal_bit_coordinates": int((~equal_bits).sum()),
            "numerically_coalesced_coordinates": int((lower == upper).sum()),
            "zero_singleton_coordinates": int(zero_singleton.sum()),
            "signed_zero_disagreements": int((zero_singleton & ~equal_bits).sum()),
            "max_final_hidden_width": float((upper.double() - lower.double()).max()),
            "coalesced_nonzero_everywhere": bool((equal_bits & ~zero_singleton).all()),
        })
    hidden_coalesced = all(row["coalesced_nonzero_everywhere"] for row in blocks)
    final_logit_bits_equal = bool(same_fp32_bits(low_logits[:, -1], high_logits[:, -1]).all())
    certified = hidden_coalesced and final_logit_bits_equal
    if hidden_coalesced and not final_logit_bits_equal:
        reason = "Endpoint hidden states coalesced but readout bits differed; backend assumption failed."
    elif any(row["zero_singleton_coordinates"] for row in blocks):
        reason = "Inconclusive: zero singleton(s) require a separate signed-zero proof."
    elif not hidden_coalesced:
        reason = "Inconclusive: at least one learned hidden interval has not coalesced exactly."
    else:
        reason = "All learned-state endpoint intervals coalesced to identical nonzero FP32 values."
    controls = {model.confirm_id, model.defer_id}
    controls.update(getattr(model, attr) for attr in ("query_a_id", "query_b_id") if hasattr(model, attr))
    weight_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        weight_digest.update(name.encode())
        weight_digest.update(value.detach().contiguous().numpy().tobytes())
    return {
        "schema": 1, "certified_prior_learned_state_independent_final_logits": certified,
        "reason": reason, "backend": "cpu", "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "model_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent / "model.py").read_bytes()).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_parameter_sha256": weight_digest.hexdigest(),
        "batch_shape": list(tokens.shape), "suffix_ids": tokens.tolist(),
        "ordinary_suffix_excludes_model_decision_and_query_ids": not any(
            bool((tokens == token).any()) for token in controls),
        "protected_premises": protected_premises.tolist(), "blocks": blocks,
        "max_final_hidden_width": max(row["max_final_hidden_width"] for row in blocks),
        "endpoint_final_logits_equal_bits": final_logit_bits_equal,
        "endpoint_final_logits_max_abs_difference": float(
            (low_logits[:, -1].double() - high_logits[:, -1].double()).abs().max()),
        "scope": "Final next-token full vocabulary logits only; fixed suffix and identical protected premises.",
        "assumptions": [
            "Existing diagonal transition; all prior learned states in [-1,1].",
            "Conventional monotone FP32 rounding, no NaNs/faults, deterministic row-local kernels.",
            "Same fixed model, batch/sequence shape, eager math path and CPU backend.",
            "No claim about unknown protected context, token meaning, or semantic safety.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Expected digest from the trusted run manifest")
    parser.add_argument("--suffix", action="append", required=True,
                        help="Ordinary text; repeat for a separate fixed-shape diagnostic")
    args = parser.parse_args()
    if len(args.suffix) > 4 or any(len(text) > 4096 for text in args.suffix):
        parser.error("at most four suffixes, each at most 4096 characters")
    if args.checkpoint.stat().st_size > 4 * 1024**2:
        parser.error("checkpoint exceeds the admitted four MiB size")
    observed = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if observed != args.sha256:
        parser.error("checkpoint digest mismatch")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        parser.error("unsupported checkpoint format")
    task = saved["training_config"]["task"]
    if task not in ("planning", "interval"):
        parser.error("unsupported checkpoint task")
    root = Path(__file__).resolve().parent
    for name in ("model.py", "interval_model.py") if task == "interval" else ("model.py",):
        if saved["source_sha256"].get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest():
            parser.error(f"checkpoint's {name} source pin differs from this verifier's model")
    corpus = importlib.import_module("interval_data" if task == "interval" else "data")
    if task == "interval":
        cls = importlib.import_module("interval_model").IntervalModel
        model = cls(len(corpus.VOCAB), saved["width"], corpus.CONFIRM, corpus.DEFER,
                    corpus.QUERY_A, corpus.QUERY_B)
    else:
        cls = importlib.import_module("model").Model
        model = cls(len(corpus.VOCAB), saved["width"], corpus.CONFIRM, corpus.DEFER)
    if not 1 <= saved["blocks"] <= 8:
        parser.error("checkpoint block count outside diagnostic bounds")
    for _ in range(saved["blocks"] - 1):
        model.grow()
    model.load_state_dict(saved["model"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model.eval()
    reports = []
    for suffix in args.suffix:
        ids = corpus.encode(suffix)
        if not ids or corpus.UNK in ids:
            parser.error("suffix must use nonempty known vocabulary")
        report = verify_suffix(model, torch.tensor([ids], dtype=torch.long))
        if not report["ordinary_suffix_excludes_model_decision_and_query_ids"]:
            parser.error("CLI suffix must exclude decision and query tokens")
        report["suffix_text"] = suffix
        reports.append(report)
    print(json.dumps({"checkpoint_sha256": observed, "reports": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
