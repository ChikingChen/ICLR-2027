import importlib.util
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "pred" / "model_wrappers.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("model_wrappers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelWrappersTest(unittest.TestCase):
    """验证本地模型 wrapper 的轻量兼容逻辑。"""

    def test_chatglm_config_gets_num_hidden_layers_alias(self):
        module = _load_module()
        config = SimpleNamespace(num_layers=40)

        module.ensure_num_hidden_layers_alias(config)

        self.assertEqual(config.num_hidden_layers, 40)

    def test_existing_num_hidden_layers_is_not_overwritten(self):
        module = _load_module()
        config = SimpleNamespace(num_layers=40, num_hidden_layers=32)

        module.ensure_num_hidden_layers_alias(config)

        self.assertEqual(config.num_hidden_layers, 32)

    def test_attention_layers_are_summarized_by_top_scores(self):
        module = _load_module()

        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {10: "A", 11: "B", 12: "C"}[token_ids[0]]

        attentions = (
            torch.tensor([[[[0.1, 0.3, 0.6]], [[0.2, 0.2, 0.6]]]]),
        )

        layers = module.summarize_attention_layers(
            attentions=attentions,
            token_ids=[10, 11, 12],
            tokenizer=FakeTokenizer(),
            top_k=2,
        )

        self.assertEqual(layers[0]["layer"], 0)
        self.assertAlmostEqual(layers[0]["sum"], 1.0, places=6)
        self.assertEqual([item["position"] for item in layers[0]["top_tokens"]], [2, 1])
        self.assertEqual([item["token"] for item in layers[0]["top_tokens"]], ["C", "B"])

    def test_generation_ppl_stats_use_generated_token_scores(self):
        module = _load_module()
        scores = (
            torch.log(torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float32)),
            torch.log(torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)),
        )
        generated_token_ids = torch.tensor([[1, 0]])

        stats = module.compute_generation_ppl_stats(scores, generated_token_ids)

        expected_logprob_sum = torch.log(torch.tensor(0.6)) + torch.log(torch.tensor(0.7))
        expected_nll = -float(expected_logprob_sum.item()) / 2
        self.assertAlmostEqual(stats["generation_logprob_sum"], float(expected_logprob_sum.item()), places=6)
        self.assertEqual(stats["generation_token_count"], 2)
        self.assertAlmostEqual(stats["generation_nll"], expected_nll, places=6)
        self.assertAlmostEqual(stats["generation_ppl"], float(torch.exp(torch.tensor(expected_nll)).item()), places=6)

    def test_generation_ppl_stats_handle_empty_generation(self):
        module = _load_module()

        stats = module.compute_generation_ppl_stats((), torch.empty((1, 0), dtype=torch.long))

        self.assertEqual(stats["generation_logprob_sum"], 0.0)
        self.assertEqual(stats["generation_token_count"], 0)
        self.assertIsNone(stats["generation_nll"])
        self.assertIsNone(stats["generation_ppl"])


if __name__ == "__main__":
    unittest.main()
