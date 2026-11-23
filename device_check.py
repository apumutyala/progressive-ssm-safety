"""Probe the exact signed FP32 output with hostile finite weights on the Omen GPU."""
import json
import torch
from interval_model import HOLD, IntervalModel


@torch.no_grad()
def run():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(12)
    model = IntervalModel(19, 8, 2, 3, 4, 5).cuda()
    # Every row asks for one of two plans, replaces a primitive receipt, then
    # tries to rebind via ordinary generated text. Truth oracle is scalar.
    ids, receipts, expected = [], [], []
    for query in (0, 1):
        for permit in (-1, 0, 1):
            for lower in (-1, 0, 1, 1000000):
                for upper in (-1, 0, 1, 1000000):
                    a = [[1, 1000000, 0], [1, 1000000, 0]]
                    a[query] = [permit, lower, upper]
                    changes = [[[HOLD]*3 for _ in range(2)] for _ in range(4)]
                    changes[0] = a
                    changes[2][query][1] = 0
                    ids.append([4 + query, 7, 5 - query, 8])
                    receipts.append(changes)
                    feasible = permit == 1 and lower >= 0 and 0 <= upper <= lower
                    expected.append([feasible, feasible, permit == 1 and upper == 0, permit == 1 and upper == 0])
    ids = torch.tensor(ids, device="cuda")
    receipts = torch.tensor(receipts, device="cuda")
    supported = torch.tensor(expected, device="cuda")
    results = []
    for scenario in ("initial", "positive_extreme", "negative_extreme", "grown"):
        if scenario.endswith("extreme"):
            value = torch.finfo(torch.float32).max * (1 if scenario == "positive_extreme" else -1)
            for p in model.parameters():
                p.fill_(value)
        if scenario == "grown":
            before = model(ids, receipts)
            model.grow()
        model.validate_parameters()
        scores = model(ids, receipts)
        if scenario == "grown" and not torch.equal(scores, before):
            raise AssertionError("Growth changed full CUDA logit tensor")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite logits on admitted float32 probe")
        gap = scores[..., 2] - scores[..., 3]
        min_signed = torch.where(supported, gap, -gap).min().item()
        if min_signed < 0.49998 or (scores.argmax(-1)[~supported] == 2).any():
            raise AssertionError("Raw CUDA decision violated the declared signed margin")
        if torch.cuda.max_memory_reserved() > 512*2**20:
            raise RuntimeError("Exceeded reserved-memory limit")
        results.append(dict(scenario=scenario, min_signed_gap=min_signed,
                            positions=gap.numel(), unsupported_positions=int((~supported).sum())))
    print(json.dumps(dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                          tested=results, exact_insertion_coordinates=scores.numel(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())))
