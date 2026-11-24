"""Small deterministic CPU controls; execute only in the admitted numerical slot.

No checkpoint/data files or CUDA APIs. The positive control fits only ordinary
Y/N output-head coefficients on a fixed real two-block SSM: capacity/causality,
not evidence of useful English behavior or alignment. Core inference is readonly.
"""
import dataclasses
import unittest

import torch

from corpus_model import make_model
from comparator import make_gru_model
from causal_trace_core import _bits_equal,_candidate_pair,_patch_hidden,trace_pair


class CausalTraceTests(unittest.TestCase):
    def cases(self):
        return ({"id":"unit-left","prompt":"x.","reference":"Y"},
                {"id":"unit-right","prompt":"y.","reference":"N"})

    def model(self,learn=True):
        torch.random.default_generator.manual_seed(914)
        model=make_model("certified",width=1);model.grow(silent_head=False)
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            for block in model.blocks:
                block.drive.weight[8+ord("x")]=1
                block.drive.weight[8+ord("y")]=-1
                block.decay.weight.fill_(1.)
                block.decay.weight[8+ord("Y")]=-16
                block.decay.weight[8+ord("N")]=-16
            model.blocks[0].head.bias[3]=.01
        if learn:
            optimizer=torch.optim.SGD([block.head.weight for block in model.blocks],lr=1.)
            inputs=torch.tensor([[2,8+ord("x"),8+ord(".")],[2,8+ord("y"),8+ord(".")]])
            targets=torch.tensor([8+ord("Y"),8+ord("N")])
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                logits=model(inputs)[:,-1]
                torch.nn.functional.cross_entropy(logits,targets).backward();optimizer.step()
            model.validate_parameters()
        return model

    def test_normal_learned_hidden_positive_control_and_reverse_intervention(self):
        model=self.model();left,right=self.cases()
        result=trace_pair(model,left,right,max_new_tokens=4)
        original=result["original"]
        self.assertGreater(original["left"]["decision"]["positive_minus_foil"],0)
        self.assertLess(original["right"]["decision"]["positive_minus_foil"],0)
        self.assertGreater(result["state_difference"]["at_split"][0]["max_abs_difference"],0)
        self.assertLess(result["patched"]["left_from_right"]["margin_shift_vs_original"],0)
        self.assertGreater(result["patched"]["right_from_left"]["margin_shift_vs_original"],0)
        self.assertEqual(result["original"]["left"]["generation"]["raw_ids"][0],8+ord("Y"))
        self.assertEqual(result["patched"]["left_from_right"]["generation"]["raw_ids"][0],8+ord("N"))

    def test_state_ignored_negative_control_has_state_difference_but_zero_effect(self):
        model=self.model(learn=False);left,right=self.cases()
        result=trace_pair(model,left,right,max_new_tokens=2)
        self.assertGreater(result["state_difference"]["at_split"][0]["max_abs_difference"],0)
        self.assertEqual(result["patched"]["left_from_right"]["margin_shift_vs_original"],0)
        self.assertEqual(result["patched"]["right_from_left"]["margin_shift_vs_original"],0)
        self.assertEqual(result["original"]["left"]["decision"]["positive_minus_foil"],0)

    def test_common_answer_prefix_equal_sham_and_prefix_vs_eos_candidates(self):
        left,right=self.cases();right["reference"]=left["reference"]
        with self.assertRaises(ValueError):trace_pair(self.model(False),left,right,max_new_tokens=1)
        result=trace_pair(self.model(False),left,right,contrast=("Y","Yes"),max_new_tokens=1)
        self.assertTrue(result["sham_equal_reference"])
        self.assertEqual(result["comparison"]["common_answer_prefix_ids"],[8+ord("Y")])
        self.assertEqual(result["comparison"]["positive_id"],3)
        self.assertEqual(result["comparison"]["foil_id"],8+ord("e"))

    def test_identity_noop_modes_and_parameter_bytes_preserved(self):
        model=self.model(False);model.train();model.blocks[0].eval()
        flags=[module.training for module in model.modules()]
        before={name:value.detach().clone() for name,value in model.state_dict().items()}
        left,right=self.cases();result=trace_pair(model,left,right,split_suffix=".",max_new_tokens=1)
        self.assertEqual(flags,[module.training for module in model.modules()])
        for name,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[name]))
        self.assertEqual(result["checks"]["left_identity_max_abs_delta"],0)
        self.assertEqual(result["checks"]["right_noop_max_abs_delta"],0)
        self.assertTrue(result["checks"]["recipient_authority_and_latch_preserved"])

    def test_direct_patch_preserves_nontrivial_authority_and_latch(self):
        model=self.model(False)
        a=torch.tensor([[6]]);events=torch.tensor([[[[1,4,2],[-1,-1,-1]]]])
        _,recipient=model.prefill(a,events);_,donor=model.prefill(torch.tensor([[7]]))
        patched=_patch_hidden(recipient,donor,(0,))
        self.assertIs(patched.records,recipient.records)
        self.assertIs(patched.query,recipient.query)
        self.assertEqual(patched.query.item(),0)
        self.assertTrue(torch.equal(patched.records,recipient.records))

    def test_actual_two_block_gru_cached_full_parity_and_patch_identity(self):
        torch.random.default_generator.manual_seed(712)
        model=make_gru_model(width=2,blocks=2)
        left,right=self.cases()
        result=trace_pair(model,left,right,max_new_tokens=2)
        self.assertTrue(result["checks"]["left_from_donor_nonconfirm_swap_close"])
        self.assertEqual(len(result["single_block_patches"]["left_from_donor"]),2)
        self.assertLessEqual(result["checks"]["left_full_prefill_max_abs_delta"],1e-5)

    def test_invalid_suffix_and_equal_contrast_rejected_without_state_mutation(self):
        left,right=self.cases()
        with self.assertRaises(ValueError):trace_pair(self.model(False),left,right,split_suffix="wrong")
        with self.assertRaises(ValueError):trace_pair(self.model(False),left,right,contrast=("Y","Y"))

    def test_singleton_stride_byte_comparison_preserves_signed_zero(self):
        singleton=torch.tensor([7,9,11,13],dtype=torch.int64)[::2][:1]
        self.assertNotEqual(singleton.stride()[-1],1)
        self.assertTrue(_bits_equal(singleton,torch.tensor([7],dtype=torch.int64)))
        self.assertFalse(_bits_equal(torch.tensor([0.]),torch.tensor([-0.])))

    def test_parameter_validation_failure_preserves_original_modes(self):
        model=self.model(False);model.train();model.blocks[0].eval()
        flags=[module.training for module in model.modules()]
        with torch.no_grad():model.blocks[0].head.weight[0,0]=float("nan")
        left,right=self.cases()
        with self.assertRaises(FloatingPointError):trace_pair(model,left,right,max_new_tokens=1)
        self.assertEqual(flags,[module.training for module in model.modules()])


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
