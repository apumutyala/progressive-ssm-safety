"""CPU-only learned-state interventions for the existing two-block SSM/GRU.

No loading, training, CUDA calls, files or launches. Call only from the admitted
single numerical worker. Cases have id/prompt/reference strings and optionally
an aligned signed-int64 events array for BOS+prompt bytes. State replacement
changes ONLY the chosen learned hidden tensors. Recipient records, query latch,
suffix inputs and parameters are preserved. Mixed per-block states can be
off-manifold: an intervention is not a naturally reachable semantic world.

First-divergent-answer-byte scores teacher-force ONLY the candidates' common
answer prefix, using the complete264-token vocabulary with lower-ID tie breaks.
That diagnostic is distinct from the separately returned raw greedy outputs.
Equal replies need an explicit contrasting candidate; no meaningless zero-token
or negative-index 'margin' is manufactured. The split is an unchanged byte
suffix. A caller must validate its semantic placement after varying facts/query;
automatic maximal common suffix detection does not understand fact boundaries.
"""
from __future__ import annotations

from dataclasses import is_dataclass, replace
import hashlib
import random


def _need(value,message):
    if not value:raise ValueError(message)


def _bytes(text,label):
    _need(type(text) is str and 0<len(text.encode("utf-8"))<=2048,label+" must be bounded nonempty UTF8")
    return text.encode("utf-8")


def _bits_equal(left,right):
    # A singleton Long view can be "contiguous" while retaining a nonunit
    # element stride, so dtype-view(uint8) is not valid for every query tensor.
    # NumPy's C-order byte packing preserves stored FP32 signed-zero bits and
    # integer bytes without dtype reinterpretation or any device transfer.
    return (left.shape==right.shape and left.dtype==right.dtype and
            left.device.type==right.device.type=="cpu" and
            left.detach().numpy().tobytes(order="C")==right.detach().numpy().tobytes(order="C"))


def _authority_equal(left,right):
    return _bits_equal(left.records,right.records) and _bits_equal(left.query,right.query)


def _patch_hidden(recipient,donor,blocks):
    """No mutation/parameter changes; all non-hidden dataclass fields retained."""
    _need(is_dataclass(recipient) and is_dataclass(donor),"expected dataclass inference states")
    _need(len(recipient.hidden)==len(donor.hidden)==2,"expected exactly two hidden blocks")
    indices=tuple(blocks)
    _need(len(set(indices))==len(indices) and all(type(index) is int and 0<=index<2 for index in indices),"invalid patch blocks")
    hidden=[]
    for index,(target,source) in enumerate(zip(recipient.hidden,donor.hidden)):
        _need(target.shape==source.shape and target.dtype==source.dtype and target.device.type==source.device.type=="cpu",
              "donor/recipient hidden layout differs")
        hidden.append((source if index in indices else target).detach().clone())
    patched=replace(recipient,hidden=tuple(hidden))
    _need(patched.records is recipient.records and patched.query is recipient.query and _authority_equal(patched,recipient),
          "intervention altered protected state")
    return patched


def _state_difference(left,right):
    result=[]
    for index,(a,b) in enumerate(zip(left.hidden,right.hidden)):
        difference=(a.detach().double()-b.detach().double()).abs()
        result.append({"block":index,"coordinates":a.numel(),"unequal_coordinates":int((a!=b).sum()),
                       "max_abs_difference":float(difference.max()),"mean_abs_difference":float(difference.mean()),
                       "l2_difference":float(difference.square().sum().sqrt())})
    return result


def _candidate_pair(left,right,contrast):
    if contrast is None:
        _need(left["reference"]!=right["reference"],"equal/sham references require an explicit contrasting foil")
        candidates=(left["reference"],right["reference"])
    elif type(contrast) is str:
        candidates=(left["reference"],contrast)
    else:
        _need(type(contrast) in (tuple,list) and len(contrast)==2,"contrast must be two ordered reference strings")
        candidates=tuple(contrast)
    encoded=[]
    for value in candidates:
        raw=_bytes(value,"candidate reference")
        _need(len(raw)<=192,"candidate reply cap")
        encoded.append([8+byte for byte in raw]+[3])
    _need(candidates[0]!=candidates[1],"candidate references must differ")
    _need(all(case["reference"] in candidates for case in (left,right)),"each true reference must be one contrast candidate")
    position=next(index for index,(a,b) in enumerate(zip(*encoded)) if a!=b)
    return candidates,encoded[0][:position],encoded[0][position],encoded[1][position],position


def trace_pair(model,left,right,*,contrast=None,split_suffix=None,max_new_tokens=96,atol=1e-5,rtol=1e-5):
    """Read-only CPU trace; return original/counterfactual scores, checks and IDs.

    contrast=(positive_reference,foil_reference) gives a common signed axis.
    Each side's signed_correct_minus_foil flips this axis when its truth is the
    second candidate. For a sham with equal replies supply the true reply and a
    distinct foil. split_suffix may be str/bytes; otherwise use the maximal
    common raw-byte suffix. No automatic semantic boundary claim is made.
    """
    import torch
    from corpus_model import CorpusModel,ByteTokenizer

    _need(isinstance(model,CorpusModel) and model.arm=="certified" and len(model.blocks)==2 and
          1<=model.width<=64 and model.vocab_size==264,"expected reviewed two-block bounded SSM or GRU family")
    _need(all(parameter.device.type=="cpu" and parameter.dtype==torch.float32 for parameter in model.parameters()),
          "CPU FP32 parameters required; no automatic device transfer")
    _need(getattr(model,"_audit",None) is None,"do not intervene during another active audit")
    _need(type(max_new_tokens) is int and 1<=max_new_tokens<=128 and 0<=atol<=1e-3 and 0<=rtol<=1e-3,"bounded inference/tolerance required")
    for case in (left,right):
        _need(type(case) is dict and type(case.get("id")) is str,"case id/prompt/reference required")
        _bytes(case.get("prompt"),"prompt");_bytes(case.get("reference"),"reference")
    raw_left,raw_right=left["prompt"].encode(),right["prompt"].encode()
    if split_suffix is None:
        length=0
        for a,b in zip(reversed(raw_left),reversed(raw_right)):
            if a!=b:break
            length+=1
        _need(length>0,"pair has no nonempty unchanged suffix")
        suffix=raw_left[-length:];rule="maximal_common_byte_suffix_not_semantically_validated"
    else:
        suffix=split_suffix.encode() if type(split_suffix) is str else split_suffix
        _need(type(suffix) is bytes and suffix and raw_left.endswith(suffix) and raw_right.endswith(suffix),
              "declared suffix must be a nonempty exact suffix of both prompts")
        rule="caller_declared_shared_suffix_semantic_placement_is_external"
    candidates,answer_prefix,positive_id,foil_id,divergence=_candidate_pair(left,right,contrast)
    model.validate_parameters()  # Failure must leave all original mode flags intact.
    modes=[(module,module.training) for module in model.modules()]
    state_before={name:value.detach().clone() for name,value in model.state_dict().items()}
    rng_before=torch.get_rng_state().clone();python_before=random.getstate()

    def finite(value,label):
        _need(value.device.type=="cpu" and bool(torch.isfinite(value).all()),"nonfinite/non-CPU "+label)

    def token(value):return torch.tensor([value],dtype=torch.int64,device="cpu")

    def inputs(case,raw):
        ids=[2]+[8+byte for byte in raw]
        supplied=case.get("events")
        events=None
        if supplied is not None:
            if isinstance(supplied,torch.Tensor):
                _need(supplied.device.type=="cpu" and supplied.dtype==torch.int64,"case events must already be CPU int64")
                events=supplied.detach().clone()
            else:events=torch.tensor(supplied,dtype=torch.int64,device="cpu")
            if events.ndim==3:events=events.unsqueeze(0)
            _need(tuple(events.shape)==(1,len(ids),2,3),"events must align to BOS+prompt bytes")
        return ids,events

    def resume(state,ids,events=None,logits=None):
        for index,value in enumerate(ids):
            logits,state=model.decode_step(token(value),state,None if events is None else events[:,index])
            finite(logits,"incremental logits")
        return logits,state

    def metric(logits,case):
        vector=logits[0];finite(vector,"decision vector")
        axis=torch.arange(264,device="cpu")
        def rank(value):
            return 1+int((vector>vector[value]).sum())+int(((vector==vector[value])&(axis<value)).sum())
        sign=1 if case["reference"]==candidates[0] else -1
        margin=float(vector[positive_id]-vector[foil_id])
        return {"positive_minus_foil":margin,"signed_correct_minus_foil":sign*margin,
                "positive_logit":float(vector[positive_id]),"foil_logit":float(vector[foil_id]),
                "positive_rank":rank(positive_id),"foil_rank":rank(foil_id),
                "correct_rank":rank(positive_id if sign==1 else foil_id),
                "raw_argmax_id":int(vector.argmax()),"correct_token_is_raw_argmax":int(vector.argmax())==(positive_id if sign==1 else foil_id)}

    def generate(logits,state,recipient,donor=None):
        ids=[]
        for index in range(max_new_tokens):
            finite(logits,"free-generation logits")
            selected=int(logits[0].argmax());ids.append(selected)
            if selected==3 or index+1==max_new_tokens:break
            logits,state=model.decode_step(token(selected),state)
        raw=bytes(value-8 for value in ids if value>=8)
        ordinary=all(value>=8 or value==3 for value in ids)
        text=raw.decode("utf-8",errors="backslashreplace")
        return {"raw_ids":ids,"rendered":ByteTokenizer.render(ids),"ordinary_byte_text":text,
                "eos_terminated":ids[-1]==3,"capped":ids[-1]!=3,"matches_recipient_reference":ordinary and text==recipient["reference"],
                "matches_donor_reference":None if donor is None else ordinary and text==donor["reference"]}

    def from_cut(case,cut_state,suffix_ids,suffix_events,cut_logits,generation=True,donor=None):
        endpoint,state=resume(cut_state,suffix_ids,suffix_events,cut_logits)
        prompt_state=state
        decision,decision_state=resume(state,answer_prefix,logits=endpoint)
        result={"decision":metric(decision,case)}
        if generation:result["generation"]=generate(endpoint,prompt_state,case,donor)
        return result,endpoint,prompt_state,decision_state

    try:
        model.eval()
        with torch.no_grad():
            contexts=[]
            for case,raw in ((left,raw_left),(right,raw_right)):
                ids,events=inputs(case,raw)
                prefix_length=len(ids)-len(suffix)
                prefix=torch.tensor([ids[:prefix_length]],dtype=torch.int64,device="cpu")
                cut_logits,cut_state=model.prefill(prefix,None if events is None else events[:,:prefix_length])
                finite(cut_logits,"cut logits")
                for hidden in cut_state.hidden:
                    finite(hidden,"learned state");_need(bool((hidden.abs()<=1).all()),"hidden outside admitted box")
                suffix_events=None if events is None else events[:,prefix_length:]
                measured,endpoint,end_state,decision_state=from_cut(case,cut_state,ids[prefix_length:],suffix_events,cut_logits[:,-1])
                full,_=model.prefill(torch.tensor([ids],dtype=torch.int64,device="cpu"),events)
                finite(full,"full prefill logits")
                delta=float((endpoint-full[:,-1]).abs().max())
                parity=bool(torch.allclose(endpoint,full[:,-1],atol=atol,rtol=rtol))
                _need(parity,"split/resumed path differs from full prefill beyond declared tolerance")
                contexts.append({"case":case,"cut":cut_state,"cut_logits":cut_logits[:,-1],"suffix":ids[prefix_length:],
                    "events":suffix_events,"result":measured,"end":end_state,"decision":decision_state,
                    "endpoint":endpoint,"prefix_length":prefix_length,"parity_delta":delta})
            patched={};single={};checks={}
            ordinary_ids=[index for index in range(264) if index!=4]
            for index,name in ((0,"left"),(1,"right")):
                recipient,donor=contexts[index],contexts[1-index]
                identity=_patch_hidden(recipient["cut"],recipient["cut"],(0,1))
                noop=_patch_hidden(recipient["cut"],donor["cut"],())
                for label,state in (("identity",identity),("noop",noop)):
                    _,logits,protected,_=from_cut(recipient["case"],state,recipient["suffix"],recipient["events"],recipient["cut_logits"],generation=False)
                    checks[name+"_"+label+"_max_abs_delta"]=float((logits-recipient["endpoint"]).abs().max())
                    _need(_bits_equal(logits,recipient["endpoint"]) and _authority_equal(protected,recipient["end"]),
                          "identity/no-op intervention changed output/protected state")
                all_state=_patch_hidden(recipient["cut"],donor["cut"],(0,1))
                result,logits,protected,_=from_cut(recipient["case"],all_state,recipient["suffix"],recipient["events"],
                    recipient["cut_logits"],donor=donor["case"])
                _need(_authority_equal(protected,recipient["end"]),"patch altered recipient's final protected state")
                reverse=bool(torch.allclose(logits[:,ordinary_ids],donor["endpoint"][:,ordinary_ids],atol=atol,rtol=rtol))
                _need(reverse,"full hidden swap failed ordinary-logit reverse-direction equivalence")
                checks[name+"_from_donor_nonconfirm_swap_close"]=reverse
                result["margin_shift_vs_original"]=result["decision"]["positive_minus_foil"]-recipient["result"]["decision"]["positive_minus_foil"]
                patched[name+"_from_"+("right" if index==0 else "left")]=result
                block_results=[]
                for block in range(2):
                    partial=_patch_hidden(recipient["cut"],donor["cut"],(block,))
                    result,_,protected,_=from_cut(recipient["case"],partial,recipient["suffix"],recipient["events"],recipient["cut_logits"],generation=False)
                    _need(_authority_equal(protected,recipient["end"]),"single-block patch changed authority")
                    block_results.append({"block":block,**result,"margin_shift_vs_original":result["decision"]["positive_minus_foil"]-
                                          recipient["result"]["decision"]["positive_minus_foil"]})
                single[name+"_from_donor"]=block_results
            result={"schema":1,"backend":"CPU_FP32","pair_ids":[left["id"],right["id"]],
                "sham_equal_reference":left["reference"]==right["reference"],
                "split":{"rule":rule,"shared_suffix_bytes":len(suffix),"shared_suffix_text":suffix.decode("utf-8",errors="backslashreplace"),
                         "left_prefix_input_tokens":contexts[0]["prefix_length"],"right_prefix_input_tokens":contexts[1]["prefix_length"]},
                "comparison":{"positive_reference":candidates[0],"foil_reference":candidates[1],
                    "common_answer_prefix_ids":answer_prefix,"first_divergent_answer_index":divergence,
                    "positive_id":positive_id,"foil_id":foil_id,"full_vocabulary":264,"tie_break":"lower_token_id",
                    "teacher_forced_common_prefix_diagnostic":True},
                "state_difference":{"at_split":_state_difference(contexts[0]["cut"],contexts[1]["cut"]),
                    "after_shared_suffix":_state_difference(contexts[0]["end"],contexts[1]["end"]),
                    "at_divergent_answer_position":_state_difference(contexts[0]["decision"],contexts[1]["decision"])},
                "checks":{**checks,"left_full_prefill_max_abs_delta":contexts[0]["parity_delta"],
                    "right_full_prefill_max_abs_delta":contexts[1]["parity_delta"],"atol":atol,"rtol":rtol,
                    "recipient_authority_and_latch_preserved":True,"parameters_and_rng_unchanged":True},
                "original":{"left":contexts[0]["result"],"right":contexts[1]["result"]},"patched":patched,"single_block_patches":single,
                "limitations":["State differences alone do not prove readout use or useful semantic grounding.",
                    "Teacher-forced common reply prefix and raw free response are separate measurements.",
                    "Single-block mixed states can be off-manifold; individual effects need not add.",
                    "Automatic suffix detection is syntactic, not a proof that all varying facts were consumed.",
                    "No arbitrary-prose safety, general alignment or GPU-path claim."]}
    finally:
        for module,flag in modes:module.training=flag
        rng_unchanged=torch.equal(rng_before,torch.get_rng_state()) and random.getstate()==python_before
        torch.set_rng_state(rng_before);random.setstate(python_before)
        unchanged=set(state_before)==set(model.state_dict()) and all(_bits_equal(value,model.state_dict()[name]) for name,value in state_before.items())
        _need(unchanged and rng_unchanged,"read-only trace changed model bytes or inference RNG")
    return result
