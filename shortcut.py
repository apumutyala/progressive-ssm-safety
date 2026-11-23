"""How far can an answer template get using only query identity and the decision bit?"""
from collections import Counter, defaultdict
import json
import interval_data


def benchmark():
    train = [r for r in interval_data.examples("train") if r["kind"] == "decision"]
    test = [r for r in interval_data.examples("test") if r["kind"] == "decision"]
    lookup = defaultdict(Counter)
    for row in train:
        key = row["query_plan"], row["supported"]
        lookup[key][row["reference"]] += 1
    # Fit only to training examples; choose lexicographically on tied counts.
    winners = {k: min(c, key=lambda answer: (-c[answer], answer)) for k,c in lookup.items()}
    predictions = []
    for row in test:
        answer = winners[row["query_plan"], row["supported"]]
        predictions.append({"id":row["id"], "prediction":answer, "exact":row["reference"] == answer})
    return {"fit_examples": len(train), "test_examples": len(test),
            "information_used": ["query plan", "trusted feasibility bit"],
            "exact": sum(r["exact"] for r in predictions), "predictions":predictions}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
