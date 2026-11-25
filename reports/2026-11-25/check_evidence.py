"""Audit the headline outcome counts in the Nov 25 evidence snapshot."""
import json
from pathlib import Path


def main():
    data = json.loads(Path(__file__).with_name("evidence.json").read_text(encoding="utf-8"))
    assert data["schema"] == 1
    for name, arm in data["typed_plan"].items():
        outcomes = arm["outcomes"]
        assert all(len(row) == 32 and set(row) <= set("CDPS") for row in outcomes.values())
        strict = sum(all(bit == "C" for bit in trio) for trio in
                     zip(outcomes["base"], outcomes["A_edit"], outcomes["B_edit"]))
        assert strict == arm["strict"]
        assert outcomes["withdraw"] == outcomes["conflict"] == "D" * 32
        print(f"typed plan {name}: {strict}/32")
    for seed, score in data["typed_skills"].items():
        assert score["old_before"] == 24 and score["new_at_insertion"] == 0
        for name, arm in score["arms"].items():
            assert arm["old"] == 24 and arm["new"] == (0 if name == "narrow" else 24)
            assert arm["exact_final"] == (name == "isolated")
            if name == "isolated":
                assert arm["swap_old"] == arm["swap_new"] == 0
            print(f"typed skill {seed} {name}: old {arm['old']}/24, new {arm['new']}/24")
    for name, arm in data["language_growth"].items():
        for key, size in (("exact_bits", 96), ("eos_bits", 96),
                          ("both_outrank_foil_bits", 48), ("query_changed_bits", 48)):
            assert len(arm[key]) == size and set(arm[key]) <= {"0", "1"}
        exact = arm["exact_bits"].count("1")
        pairs = arm["both_outrank_foil_bits"].count("1")
        assert exact == pairs == arm["eos_bits"].count("1") == 0
        assert arm["query_changed_bits"].count("1") == data["query_pair_report"][name]["any_raw_change"]
        print(f"language {name}: {exact}/96 exact, {pairs}/48 pairs, {arm['public_nats']:.4f} nats/token")


if __name__ == "__main__":
    main()
