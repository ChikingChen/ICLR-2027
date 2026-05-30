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

    def test_generation_ppl_stats_can_include_token_details(self):
        module = _load_module()

        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {0: "<zero>", 1: "甲"}[int(token_ids[0])]

        scores = (
            torch.log(torch.tensor([[0.25, 0.75]], dtype=torch.float32)),
            torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float32)),
        )
        generated_token_ids = torch.tensor([[1, 0]])

        stats = module.compute_generation_ppl_stats(
            scores,
            generated_token_ids,
            tokenizer=FakeTokenizer(),
            include_token_details=True,
        )

        token_details = stats["generation_tokens"]
        self.assertEqual(len(token_details), 2)
        self.assertEqual(token_details[0]["position"], 0)
        self.assertEqual(token_details[0]["token_id"], 1)
        self.assertEqual(token_details[0]["token"], "甲")
        self.assertAlmostEqual(token_details[0]["logprob"], float(torch.log(torch.tensor(0.75)).item()), places=6)
        self.assertAlmostEqual(token_details[0]["nll"], -float(torch.log(torch.tensor(0.75)).item()), places=6)
        self.assertAlmostEqual(token_details[0]["ppl"], float(torch.exp(-torch.log(torch.tensor(0.75))).item()), places=6)
        self.assertEqual(token_details[1]["position"], 1)
        self.assertEqual(token_details[1]["token_id"], 0)
        self.assertEqual(token_details[1]["token"], "<zero>")

    def test_generation_ppl_stats_handle_empty_generation(self):
        module = _load_module()

        stats = module.compute_generation_ppl_stats((), torch.empty((1, 0), dtype=torch.long))

        self.assertEqual(stats["generation_logprob_sum"], 0.0)
        self.assertEqual(stats["generation_token_count"], 0)
        self.assertIsNone(stats["generation_nll"])
        self.assertIsNone(stats["generation_ppl"])

    def test_generation_timing_summary_averages_decode_by_generated_tokens(self):
        module = _load_module()

        summary = module.summarize_generation_timing(
            prefill_forward_ms=[10.0],
            decode_forward_ms=[2.0, 4.0],
            generated_token_count=3,
            timer_backend="cuda_event",
        )

        self.assertEqual(summary["timer_backend"], "cuda_event")
        self.assertEqual(summary["prefill_forward_ms"], 10.0)
        self.assertEqual(summary["decode_forward_ms_total"], 6.0)
        self.assertEqual(summary["decode_steps"], 2)
        self.assertEqual(summary["generated_token_count"], 3)
        self.assertAlmostEqual(summary["decode_forward_ms_per_token_avg"], 2.0)

    def test_attention_timing_collector_summarizes_prefill_and_decode(self):
        module = _load_module()

        collector = module.AttentionTimingCollector()
        collector.record(
            "prefill",
            {
                "timer_backend": "cuda_event_attention_ops",
                "attention_kernel_ms": 3.0,
                "attention_kernel_event_count": 2,
            },
        )
        collector.record(
            "decode",
            {
                "timer_backend": "cuda_event_attention_ops",
                "attention_kernel_ms": 1.0,
                "attention_kernel_event_count": 1,
            },
        )
        collector.record(
            "decode",
            {
                "timer_backend": "cuda_event_attention_ops",
                "attention_kernel_ms": 2.0,
                "attention_kernel_event_count": 1,
            },
        )

        summary = collector.summary(generated_token_count=3, input_tokens=128)

        self.assertEqual(summary["timer_backend"], "cuda_event_attention_ops")
        self.assertEqual(summary["input_tokens"], 128)
        self.assertEqual(summary["generated_token_count"], 3)
        self.assertAlmostEqual(summary["prefill_attention_kernel_ms"], 3.0)
        self.assertAlmostEqual(summary["decode_attention_kernel_ms_total"], 3.0)
        self.assertAlmostEqual(summary["decode_attention_kernel_ms_per_token_avg"], 1.0)
        self.assertEqual(summary["attention_kernel_event_count"], 4)

    def test_mask_bos_inputs_for_generation_zeroes_bos_and_preserves_positions(self):
        module = _load_module()

        class FakeTokenizer:
            bos_token_id = 128000

            def decode(self, token_ids, skip_special_tokens=False):
                return "<|begin_of_text|>" if token_ids == [128000] else str(token_ids[0])

        inputs = {
            "input_ids": torch.tensor([[128000, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

        masked_inputs, ablation = module.mask_bos_inputs_for_generation(inputs, FakeTokenizer())

        self.assertEqual(masked_inputs["attention_mask"].tolist(), [[0, 1, 1]])
        self.assertEqual(masked_inputs["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertTrue(ablation["enabled"])
        self.assertEqual(ablation["token_position"], 0)
        self.assertEqual(ablation["token_id"], 128000)
        self.assertEqual(ablation["token_text"], "<|begin_of_text|>")

    def test_mask_bos_inputs_for_generation_rejects_missing_bos(self):
        module = _load_module()

        class FakeTokenizer:
            bos_token_id = 128000

        inputs = {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "expected BOS token id 128000 at position 0"):
            module.mask_bos_inputs_for_generation(inputs, FakeTokenizer())

    def test_generate_inputs_drop_position_ids_when_bos_is_masked(self):
        module = _load_module()
        wrapper = object.__new__(module.HuggingFaceModel)
        wrapper.mask_bos_token = True
        inputs = {
            "input_ids": torch.tensor([[128000, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }

        generate_inputs = wrapper._inputs_for_generate(inputs)

        self.assertNotIn("position_ids", generate_inputs)
        self.assertIn("position_ids", inputs)

    def test_attention_profile_generation_kwargs_drop_score_retention(self):
        module = _load_module()
        wrapper = object.__new__(module.HuggingFaceModel)
        wrapper.generation_kwargs = {
            "max_new_tokens": 8,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_attentions": True,
        }

        profile_kwargs = wrapper._generation_kwargs_for_attention_profile()

        self.assertEqual(profile_kwargs["max_new_tokens"], 8)
        self.assertEqual(profile_kwargs["num_logits_to_keep"], 1)
        self.assertNotIn("return_dict_in_generate", profile_kwargs)
        self.assertNotIn("output_scores", profile_kwargs)
        self.assertNotIn("output_attentions", profile_kwargs)

    def test_generate_for_attention_profile_retries_without_num_logits_to_keep(self):
        module = _load_module()
        wrapper = object.__new__(module.HuggingFaceModel)
        wrapper.mask_bos_token = False
        wrapper.generation_kwargs = {
            "max_new_tokens": 1,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                if "num_logits_to_keep" in kwargs:
                    raise ValueError("unused model_kwargs: ['num_logits_to_keep']")
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

        fake_model = FakeModel()
        wrapper.model = fake_model

        output = wrapper._generate_for_attention_profile(
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
        )

        self.assertEqual(output.tolist(), [[1, 2, 3]])
        self.assertEqual(len(fake_model.calls), 2)
        self.assertIn("num_logits_to_keep", fake_model.calls[0])
        self.assertNotIn("num_logits_to_keep", fake_model.calls[1])
        self.assertNotIn("output_scores", fake_model.calls[0])
        self.assertNotIn("return_dict_in_generate", fake_model.calls[0])

    def test_masked_bos_prepare_uses_cache_position_for_position_ids(self):
        module = _load_module()
        wrapper = object.__new__(module.HuggingFaceModel)
        wrapper.mask_bos_token = True

        class FakeModel:
            def prepare_inputs_for_generation(
                self,
                input_ids,
                past_key_values=None,
                attention_mask=None,
                inputs_embeds=None,
                cache_position=None,
                **kwargs,
            ):
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": torch.full_like(input_ids, 99),
                }

        wrapper.model = FakeModel()

        with wrapper._preserve_positions_for_masked_bos_generation():
            prefill_inputs = wrapper.model.prepare_inputs_for_generation(
                torch.tensor([[128000, 11, 12]], dtype=torch.long),
                attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
                cache_position=torch.tensor([0, 1, 2], dtype=torch.long),
            )
            decode_inputs = wrapper.model.prepare_inputs_for_generation(
                torch.tensor([[13]], dtype=torch.long),
                past_key_values=object(),
                attention_mask=torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
                cache_position=torch.tensor([3], dtype=torch.long),
            )

        self.assertEqual(prefill_inputs["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(decode_inputs["position_ids"].tolist(), [[3]])

    def test_attention_kernel_summary_filters_attention_events(self):
        module = _load_module()

        events = [
            SimpleNamespace(name="flash_attn_fwd", device_time_total=3000.0, cuda_time_total=0.0),
            SimpleNamespace(name="aten::add", device_time_total=7000.0, cuda_time_total=0.0),
            SimpleNamespace(name="scaled_dot_product_attention", device_time_total=2000.0, cuda_time_total=0.0),
        ]

        summary = module.summarize_attention_kernel_events(events)

        self.assertEqual(summary["attention_kernel_event_count"], 2)
        self.assertAlmostEqual(summary["attention_kernel_ms"], 5.0)
        self.assertNotIn("warning", summary)

    def test_attention_kernel_summary_warns_when_no_attention_event_matches(self):
        module = _load_module()

        summary = module.summarize_attention_kernel_events(
            [SimpleNamespace(name="aten::add", device_time_total=3000.0, cuda_time_total=0.0)]
        )

        self.assertEqual(summary["attention_kernel_event_count"], 0)
        self.assertIsNone(summary["attention_kernel_ms"])
        self.assertIn("warning", summary)

    def test_cuda_event_attention_timer_records_wrapped_global_call(self):
        module = _load_module()

        def fake_attention_op(x):
            return x + 1

        namespace = {"fake_attention_op": fake_attention_op}
        timer = module.CudaEventAttentionTimer()

        with timer:
            timer.patch_function(namespace, "fake_attention_op", "fake_attention_op")
            output = namespace["fake_attention_op"](torch.tensor([1.0]))

        summary = timer.summary()
        self.assertEqual(output.tolist(), [2.0])
        self.assertEqual(summary["attention_kernel_event_count"], 1)
        self.assertIsNotNone(summary["attention_kernel_ms"])
        self.assertGreaterEqual(summary["attention_kernel_ms"], 0.0)
        self.assertIn(
            summary["timer_backend"],
            {"cuda_event_attention_ops", "perf_counter_attention_ops"},
        )

    def test_cuda_event_attention_timer_restores_wrapped_global_call(self):
        module = _load_module()

        def fake_attention_op(x):
            return x + 1

        namespace = {"fake_attention_op": fake_attention_op}
        timer = module.CudaEventAttentionTimer()

        with timer:
            self.assertTrue(timer.patch_function(namespace, "fake_attention_op", "fake_attention_op"))
            self.assertIsNot(namespace["fake_attention_op"], fake_attention_op)

        self.assertIs(namespace["fake_attention_op"], fake_attention_op)


if __name__ == "__main__":
    unittest.main()
