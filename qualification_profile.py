"""Tiny synchronized profile of the exact method used for prefill, decode and training."""
import time


def profile(arm, device, check):
    import torch
    import torch.nn.functional as F
    from qualification_model import make_model
    from qualification_data import ordinary_records, records

    def measured(fn):
        check()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        check()
        return output, elapsed

    torch.manual_seed(12)
    model = make_model(arm, width=32, depth=2, max_context=1024).to(device)
    prose = max(ordinary_records(), key=lambda r: len(r['ids']))
    evidence = records('train')[0]
    batches = [('prose64', list(prose['ids'][:65]), list(prose['loss_mask'][:64])),
               ('prose128', list(prose['ids'][:129]), list(prose['loss_mask'][:128])),
               ('evidence', list(evidence['ids']), list(evidence['loss_mask']))]
    rows = []
    for name, ids, mask in batches:
        if len(ids)>1024 or len(ids)<3 or len(mask)!=len(ids)-1 or not any(mask):
            raise ValueError('Invalid profile record or context')
        x = torch.tensor([ids[:-1]], device=device)
        target = torch.tensor([ids[1:]], device=device)
        use = torch.tensor([mask], dtype=torch.bool, device=device)
        target = target.masked_fill(~use, -100)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        def backward():
            model.zero_grad(set_to_none=True)
            scores = model(x)
            loss = F.cross_entropy(scores.flatten(0, 1), target.flatten(), ignore_index=-100)
            loss.backward()
            return loss.detach()
        # Warm once for this exact shape. No optimizer or parameter updates.
        loss, cold = measured(backward)
        loss, elapsed = measured(backward)
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                          for p in model.parameters()):
            raise ValueError('Nonfinite profile objective/gradient')
        model.eval()
        with torch.no_grad():
            (all_logits, state), prefill = measured(lambda: model.prefill(x))
            first = all_logits[:, -1].argmax(-1)
            (next_logits, _), decode = measured(lambda: model.decode_step(first, state))
            full = model(torch.cat([x, first[:,None]], dim=1))[:,-1]
            max_diff = (next_logits-full).abs().max().item()
            if not torch.allclose(next_logits, full, atol=3e-5, rtol=1e-5):
                raise ValueError('Cached decode and full recomputation disagree')
        model.train()
        rows.append(dict(record=name, input_positions=x.numel(), targets=sum(mask),
                         cold_forward_backward_s=cold, warm_forward_backward_s=elapsed,
                         supervised_targets_per_second=sum(mask)/elapsed, prefill_s=prefill, one_decode_s=decode,
                         max_abs_cache_difference=max_diff, loss=loss.item(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
    return dict(arm=arm, device=device, width=32, depth=2,
                parameters=sum(p.numel() for p in model.parameters()), rows=rows)
