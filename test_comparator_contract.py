"""Stdlib structural/count checks only; no Torch import or model construction."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def load_counter_module():
    spec = importlib.util.spec_from_file_location("_comparator_contract_only", ROOT / "comparator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComparatorContractTests(unittest.TestCase):
    def test_import_and_accounting_do_not_import_torch_or_model(self):
        before = set(sys.modules)
        module = load_counter_module()
        module.analytic_comparison()
        new = set(sys.modules)-before
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in new))
        self.assertTrue(new.isdisjoint({"model", "corpus_model", "interval_model"}))

    def test_independently_count_all_trainable_gru_and_ssm_tensors(self):
        module = load_counter_module()
        v, s, h = 264, 16, 18
        ssm = v*s + v*s + s*s+s + (2*s)*v+v
        gru = v*h + (3*h)*h+3*h + (3*h)*h+3*h + h*h+h + (2*h)*v+v
        self.assertEqual((ssm, gru), (17432, 16914))
        self.assertEqual(module.parameter_count("ssm", s, 2), 2*ssm)
        self.assertEqual(module.parameter_count("gru", h, 2), 2*gru)
        result = module.analytic_comparison()
        self.assertEqual(result["gru_minus_ssm_parameters"], -1036)
        self.assertEqual((result["ssm_dense_macs_per_token"], result["gru_dense_macs_per_token"]),
                         (17408, 23544))

    def test_width_choice_precedes_results_and_is_nearest_in_declared_family(self):
        module = load_counter_module()
        independent = lambda h: 264*h + 6*h*h+6*h + h*h+h + 528*h+264
        best = min(range(1, 257), key=lambda h: (abs(independent(h)-17432), h))
        self.assertEqual(best, 18)
        self.assertEqual(module.nearest_gru_width(), best)
        self.assertLessEqual(module.nearest_gru_width(8192), 8192)
        for invalid in (True, 0, -1, 1.5, "18"):
            with self.assertRaises(ValueError):
                module.parameter_count("gru", invalid)

    def test_shared_readout_and_authentication_are_inherited_without_override(self):
        tree = ast.parse((ROOT / "comparator.py").read_text(encoding="utf-8"))
        runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_runtime_classes")
        model = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == "GRUCorpusModel")
        self.assertEqual([ast.unparse(base) for base in model.bases], ["CorpusModel"])
        own_methods = {node.name for node in model.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(own_methods, {"__init__", "grow"})
        # No copied readout, input framing, authentication, or decoding logic.
        self.assertTrue(own_methods.isdisjoint({"_score", "_run", "_records", "_queries", "initial_state",
                                              "prefill", "decode_step", "forward", "validate_parameters"}))

    def test_runtime_dependencies_are_lazy_and_no_training_or_data_code_is_added(self):
        tree = ast.parse((ROOT / "comparator.py").read_text(encoding="utf-8"))
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual([(node.module, [alias.name for alias in node.names]) for node in top_imports],
                         [("functools", ["lru_cache"])])
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertTrue(attributes.isdisjoint({"save", "load", "backward", "step", "manual_seed", "multinomial"}))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue(names.isdisjoint({"open", "Adam", "Corpus", "DataLoader"}))

    def test_gru_has_real_trainable_gates_and_uses_no_premise_labels(self):
        tree = ast.parse((ROOT / "comparator.py").read_text(encoding="utf-8"))
        runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_runtime_classes")
        block = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == "GRUBlock")
        forward = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        attrs = {node.attr for node in ast.walk(forward) if isinstance(node, ast.Attribute)}
        self.assertTrue({"input_gates", "hidden_gates", "sigmoid", "tanh", "chunk", "clamp"} <= attrs)
        self.assertTrue(attrs.isdisjoint({"records", "query", "evidence", "confirm_id", "defer_id"}))
        arguments = forward.args
        self.assertEqual([a.arg for a in arguments.args], ["self", "tokens", "previous"])
        self.assertEqual([a.arg for a in arguments.kwonlyargs], ["certified", "audit"])


if __name__ == "__main__":
    unittest.main()
