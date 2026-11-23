"""Conditional FP32 bound on influence of prior hidden state for a fixed suffix.

Each token fixes its own positive retention and injection per coordinate in
CorpusBlock. For finite operands, rounded multiplication by a positive number,
addition of a fixed number and clipping preserve numeric order. Applying the
actual recurrence to both corners of its invariant box therefore encloses every
admissible final hidden state. This uses the executed operations, including
rounding at each step. It does not substitute the real-arithmetic gate product.

A nonzero numeric singleton has one FP32 representation. Zero is excluded from
exact certification because numeric order alone does not distinguish +/-0.
With all coordinates fixed to nonzero singletons, the pointwise time-local head
receives identical features and the protected state stays identical. Under
fixed, deterministic row-local kernels, final raw logits and future decoding on
identical inputs are then independent of prior learned state. The theorem is
conditional on the unmodified eager family, conventional monotone finite FP32
arithmetic, the actual suffix/shape, immutable parameters and backend. It does
not estimate memory of a typical prefix, forgotten semantic content, or output
variation when the box has positive width. Endpoints can be unreachable.
"""
from dataclasses import replace
import math

import torch

from corpus_model import BYTE_OFFSET, VOCAB_SIZE, CorpusBlock, CorpusModel


def _fp32_units(bits):
    """Decode a finite binary32 bit pattern into exact integer units of 2**-149."""
    bits &= 0xFFFFFFFF
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    if exponent == 255:
        raise FloatingPointError("Nonfinite endpoint state")
    units = fraction if exponent == 0 else ((1 << 23) + fraction) << (exponent - 1)
    return -units if bits >> 31 else units


@torch.no_grad()
def verify_suffix(model, tokens):
    """Probe one batch of ordinary byte-token suffixes, emitting aggregates only.

    Start both runs with the ordinary episode's six unknown integer records
    and its unbound first query. Byte tokens cannot modify either. The hidden
    coordinates range independently over [-1,1] for each block and batch row.
    """
    if type(model) is not CorpusModel or model.arm != "certified" or not all(
        type(block) is CorpusBlock for block in model.blocks
    ):
        raise ValueError("Requires the reviewed certified byte model and blocks")
    if (tokens.dtype != torch.int64 or tokens.ndim != 2 or not 1 <= tokens.size(0) <= 8
            or not 1 <= tokens.size(1) <= 128 or tokens.device.type not in ("cpu", "cuda")):
        raise ValueError("Expected int64 ordinary suffix IDs of shape [1..8, 1..128]")
    if any(p.device != tokens.device for p in model.parameters()):
        raise ValueError("Tokens and fitted model must already be on the same device")
    if ((tokens < BYTE_OFFSET) | (tokens >= VOCAB_SIZE)).any().item():
        raise ValueError("The diagnostic admits only ordinary byte IDs")
    if not torch.are_deterministic_algorithms_enabled() or torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Requires deterministic eager execution and highest FP32 matmul precision")
    model.validate_parameters()
    initial = model.initial_state(tokens)

    def corner(value):
        hidden = tuple(torch.full_like(h, value) for h in initial.hidden)
        logits, state = model._run(tokens, replace(initial, hidden=hidden), events=None)
        if not torch.equal(state.records, initial.records) or not torch.equal(state.query, initial.query):
            raise RuntimeError("Ordinary suffix changed the fixed protected state")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite endpoint logits")
        return torch.cat(state.hidden, dim=-1), logits[:, -1, :].contiguous()

    low, logits_low = corner(-1.0)
    high, logits_high = corner(1.0)
    logit_bits_equal = (logits_low.view(torch.int32) == logits_high.view(torch.int32)).all(dim=1).tolist()
    # Report observed hidden endpoints using their stored bits, including
    # subnormals, independently of hardware denormal comparison/conversion modes.
    minimum, maximum = -(1 << 149), 1 << 149
    width_units = singleton_count = zero_count = independent_rows = 0
    for row, (lower, upper) in enumerate(zip(low.view(torch.int32).tolist(), high.view(torch.int32).tolist())):
        independent = True
        for a, b in zip(lower, upper):
            left, right = _fp32_units(a), _fp32_units(b)
            if not minimum <= left <= right <= maximum:
                raise RuntimeError("Invalid endpoint ordering or invariant-box check")
            width_units = max(width_units, right - left)
            singleton_count += left == right
            zero_count += left == right == 0
            independent &= left == right and left != 0
        if independent and not logit_bits_equal[row]:
            raise RuntimeError("Endpoint logits differed despite certified nonzero singleton state")
        independent_rows += independent
    # Convert the exact integer diameter once. Round outward if the binary64
    # integer conversion rounded downward (scaling by 2**-149 is exact).
    rounded = float(width_units)
    if int(rounded) < width_units:
        rounded = math.nextafter(rounded, math.inf)
    upper_width = math.ldexp(rounded, -149)
    return {
        "rows": tokens.size(0),
        "suffix_bytes": tokens.size(1),
        "hidden_coordinates": low.numel(),
        "initial_box_width": 2.0,
        "max_final_hidden_width_upper_bound": upper_width,
        "numeric_singleton_coordinates": singleton_count,
        "zero_singleton_coordinates": zero_count,
        "certified_independent_final_logit_rows": independent_rows,
        "observed_endpoint_final_logit_equal_rows": sum(logit_bits_equal),
        "observed_endpoint_final_logit_max_abs_delta": (logits_low.double() - logits_high.double()).abs().max().item(),
    }
