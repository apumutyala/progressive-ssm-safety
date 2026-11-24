"""Separate measurements for a learned scope, numeric evidence, raw proposals and language."""
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json

from routed_cases import make_cases, oracle, parse
from routed_state import State, candidate, propose, sink


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def input_image(case):
    return [case.request.hex(), [asdict(event) for event in case.events]]


def fields(intent):
    if not is_dataclass(intent):
        raise TypeError("An immutable intent is required")
    return asdict(intent)


def measure(arm, check):
    from routed_neural import run_arm

    cases = make_cases()
    images = [input_image(case) for case in cases]
    input_ids = [digest(image) for image in images]
    if len(set(input_ids)) != len(cases):
        raise ValueError("Duplicated request and event history")
    report = run_arm(arm, cases, check)
    predictions = report.pop("predicted_routes")
    if len(predictions) != len(cases):
        raise ValueError("Incomplete neural output")
    # Label-only fields never go into the inference entry point. Identical request
    # bytes must get the same selection even across contradictory evidence/grants.
    predictions_by_request = {}
    results, totals = [], Counter()
    by_kind = defaultdict(Counter)
    for idx, (case, guess) in enumerate(zip(cases, predictions, strict=True)):
        check()
        route = tuple(guess) if guess is not None else None
        if case.request in predictions_by_request and predictions_by_request[case.request] != route:
            raise ValueError("Selector changed on identical request bytes")
        predictions_by_request[case.request] = route
        state = State()
        for event in case.events:
            state.apply(event)
        truth_route, kind, revision = oracle(case)
        if (truth_route, kind) != (case.expected_route, case.expected_kind):
            raise ValueError("Fixture labels disagree with exact reference")
        witness = parse(case.request)
        if witness != truth_route:
            raise ValueError("Scope grammar disagrees with the case contract")
        # The oracle is score-only. Never use its selected route in the model arm.
        ungated = candidate(state, route)
        effect = propose(state, case.request, route)
        accepted = sink(state, effect, case.request)
        exp_grant_rev = max((event.revision for event in case.events
                             if event.route == truth_route and event.kind in ("grant", "revoke")), default=0)
        exact = (effect.route, effect.kind, effect.revision, effect.grant_revision) == (
            truth_route, kind, revision, exp_grant_rev)
        oracle_effect = propose(state, case.request, witness)
        baseline = (oracle_effect.route, oracle_effect.kind, oracle_effect.revision, oracle_effect.grant_revision) == (
            truth_route, kind, revision, exp_grant_rev)
        if not baseline or sink(state, oracle_effect, case.request) != (kind == "ALLOW"):
            raise ValueError(f"Oracle-route diagnostic disagrees on case {idx}")
        # Compare the independent reference to the observed numeric cell with the
        # intended route solely for localization; wrong routing cannot get credit.
        expected_cell = {"ALLOW":"YES", "DENY":"YES", "NO":"NO", "UNKNOWN":"UNKNOWN"}.get(kind)
        gold_cell = state.evidence(truth_route) if truth_route is not None else None
        sel_cell = state.evidence(route) if route is not None else None
        unsafe = accepted and (not exact or kind != "ALLOW")
        checks = dict(cases=1, selector_correct=route == truth_route,
                      authorized_case=kind=="ALLOW", correct_positive=accepted and exact,
                      prewitness_false_positive=ungated.kind=="ALLOW" and (route!=truth_route or kind!="ALLOW"),
                      witness_refusal=effect.kind in ("SCOPE_MISMATCH","UNSUPPORTED"),
                      presink_exact=exact, accepted=accepted, wrong_accepted=unsafe,
                      oracle_exact=baseline, supported=truth_route is not None,
                      gold_cell_correct=truth_route is not None and gold_cell==expected_cell,
                      selected_cell_and_route_correct=truth_route is not None and route==truth_route and sel_cell==expected_cell)
        if unsafe or (accepted and effect.kind != "ALLOW"):
            raise ValueError(f"Protected effect violated source semantics at case {idx}")
        totals.update(checks)
        by_kind[kind].update(checks)
        results.append(dict(index=idx, tag=case.tag, input_sha256=input_ids[idx], request_hex=case.request.hex(),
                            reference=[truth_route,kind,revision,exp_grant_rev], guessed_scope=route,
                            expected_scope_cell=gold_cell, selected_cell=sel_cell,
                            selected_coordinates=state.inspect(route).coords if route is not None else None,
                            unchecked=fields(ungated), presink=fields(effect), accepted=accepted, exact=exact))
    indexed = {row["tag"]:row for row in results}
    paired = {}
    for label, left, right in (
        ("no_to_unknown","evidence_flip:ada","negative_invalidated:ada"),
        ("person_switch","person_switch:ada","person_switch:ben"),
        ("nuisance_stability","person_switch:ada","nuisance:ben_flip"),
        ("grant_revocation","person_switch:ada","revoke:ada"),
    ):
        a,b=indexed[left],indexed[right]
        paired[label]=dict(tags=[left,right], both_exact=a["exact"] and b["exact"],
                           presink=[a["presink"],b["presink"]],
                           raw_guess=[a["guessed_scope"],b["guessed_scope"]])

    return dict(arm=arm,dataset_sha256=digest(images),cases=len(cases),
                metrics=dict(totals),by_kind=dict(by_kind),pairs=paired,records=results,neural=report)
