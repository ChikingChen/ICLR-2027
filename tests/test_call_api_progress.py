import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path


CALL_API_PATH = Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "pred" / "call_api.py"


def _load_call_api_module():
    old_argv = sys.argv[:]
    sys.argv = [
        "call_api.py",
        "--data_dir",
        "/tmp/ruler-data",
        "--save_dir",
        "/tmp/ruler-pred",
        "--task",
        "niah_single_1",
    ]
    try:
        spec = importlib.util.spec_from_file_location("call_api_under_test", CALL_API_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


class CallApiProgressTest(unittest.TestCase):
    """验证 `call_api.py` 保留 batch 进度日志开关。"""

    def test_source_contains_optional_batch_progress_logs(self):
        source = CALL_API_PATH.read_text(encoding="utf-8")

        self.assertIn("--log_batch_progress", source)
        self.assertIn("--max_retries", source)
        self.assertIn("--log_attention_scores", source)
        self.assertIn("--attention_top_k", source)
        self.assertIn("--log_generation_ppl", source)
        self.assertIn("--log_generation_token_ppl", source)
        self.assertIn("--log_prefill_decode_timing", source)
        self.assertIn("--profile_attention_kernels", source)
        self.assertIn("--attention_profile_sample_offset", source)
        self.assertIn("--mask_bos_token", source)
        self.assertIn("[BATCH_START]", source)
        self.assertIn("[BATCH_DONE]", source)
        self.assertIn("[BATCH_FAILED]", source)

    def test_validate_runtime_args_rejects_non_singleton_batch_size(self):
        module = _load_call_api_module()

        parsed = module.parser.parse_args(
            [
                "--data_dir",
                "/tmp/ruler-data",
                "--save_dir",
                "/tmp/ruler-pred",
                "--task",
                "niah_single_1",
                "--batch_size",
                "2",
            ]
        )

        with self.assertRaisesRegex(ValueError, "batch_size.*1"):
            module.validate_runtime_args(parsed)

    def test_validate_runtime_args_rejects_mask_bos_token_for_non_hf_backend(self):
        module = _load_call_api_module()

        parsed = module.parser.parse_args(
            [
                "--data_dir",
                "/tmp/ruler-data",
                "--save_dir",
                "/tmp/ruler-pred",
                "--task",
                "niah_single_1",
                "--server_type",
                "openai",
                "--mask_bos_token",
            ]
        )

        with self.assertRaisesRegex(ValueError, "mask_bos_token.*hf"):
            module.validate_runtime_args(parsed)

    def test_attention_profile_sample_is_first_input_record(self):
        module = _load_call_api_module()

        sample = module.select_attention_profile_sample(
            [
                {"index": 41, "input": "第一条"},
                {"index": 42, "input": "第二条"},
            ],
            sample_offset=0,
        )

        self.assertEqual(sample["sample_line_no"], 0)
        self.assertEqual(sample["sample"]["index"], 41)
        with self.assertRaisesRegex(ValueError, "第 0 行"):
            module.select_attention_profile_sample([{"index": 41, "input": "第一条"}], sample_offset=1)

    def test_process_batch_with_retries_returns_after_transient_failure(self):
        module = _load_call_api_module()

        class FlakyLLM:
            def __init__(self):
                self.calls = 0

            def process_batch(self, prompts):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first failure")
                return [{"text": prompts[0]}]

        llm = FlakyLLM()
        batch_meta = {
            "batch_no": 1,
            "total_batches": 1,
            "size": 1,
            "index_start": 7,
            "index_end": 7,
        }
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = module.process_batch_with_retries(
                llm=llm,
                input_list=["ok"],
                batch_meta=batch_meta,
                max_retries=2,
                task_name="niah_single_1",
            )

        self.assertEqual(result, [{"text": "ok"}])
        self.assertEqual(llm.calls, 2)

    def test_process_batch_with_retries_raises_after_retry_limit(self):
        module = _load_call_api_module()

        class FailingLLM:
            def __init__(self):
                self.calls = 0

            def process_batch(self, prompts):
                self.calls += 1
                raise RuntimeError("always fails")

        llm = FailingLLM()
        batch_meta = {
            "batch_no": 3,
            "total_batches": 9,
            "size": 1,
            "index_start": 11,
            "index_end": 11,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                module.process_batch_with_retries(
                    llm=llm,
                    input_list=["bad"],
                    batch_meta=batch_meta,
                    max_retries=2,
                    task_name="niah_single_1",
                )

        self.assertEqual(llm.calls, 2)
        self.assertIn("[BATCH_FAILED]", output.getvalue())

    def test_attention_markdown_is_readable(self):
        module = _load_call_api_module()

        summary = {
            "mode": "first_generated_token",
            "prompt_tokens": 3,
            "generated_token_id": 99,
            "generated_token_text": "答案",
            "layers": [
                {
                    "layer": 0,
                    "sum": 1.0,
                    "top_tokens": [
                        {"rank": 1, "position": 2, "score": 0.75, "token": "needle"},
                        {"rank": 2, "position": 1, "score": 0.25, "token": "context"},
                    ],
                }
            ],
        }

        markdown = module.format_attention_markdown(
            task_name="niah_single_1",
            sample_index=42,
            summary=summary,
        )

        self.assertIn("## 样本 index=42", markdown)
        self.assertIn("| 层 | 归一化和 | Top 注意力 token |", markdown)
        self.assertIn("needle", markdown)
        self.assertIn("0.750000", markdown)

    def test_prediction_record_keeps_generation_ppl_fields(self):
        module = _load_call_api_module()

        record = module.build_prediction_record(
            pred={
                "text": ["answer"],
                "generation_logprob_sum": -1.2,
                "generation_token_count": 3,
                "generation_nll": 0.4,
                "generation_ppl": 1.4918246976412703,
            },
            index=7,
            input_text="prompt",
            outputs=["answer"],
            others={},
            truncation=-1,
            length=4096,
        )

        self.assertEqual(record["pred"], "answer")
        self.assertEqual(record["generation_logprob_sum"], -1.2)
        self.assertEqual(record["generation_token_count"], 3)
        self.assertEqual(record["generation_nll"], 0.4)
        self.assertEqual(record["generation_ppl"], 1.4918246976412703)

    def test_generation_token_record_keeps_per_token_ppl_details(self):
        module = _load_call_api_module()

        record = module.build_generation_token_record(
            pred={
                "generation_tokens": [
                    {
                        "position": 0,
                        "token_id": 198,
                        "token": "\\n",
                        "logprob": -0.2,
                        "nll": 0.2,
                        "ppl": 1.2214027581601699,
                    }
                ]
            },
            index=7,
            task="vt",
        )

        self.assertEqual(record["index"], 7)
        self.assertEqual(record["task"], "vt")
        self.assertEqual(record["generation_token_count"], 1)
        self.assertEqual(record["tokens"][0]["token_id"], 198)
        self.assertEqual(record["tokens"][0]["ppl"], 1.2214027581601699)
        json.dumps(record, ensure_ascii=False)

    def test_generation_timing_record_keeps_sample_line_number_and_index(self):
        module = _load_call_api_module()

        record = module.build_generation_timing_record(
            pred={
                "generation_timing": {
                    "timer_backend": "cuda_event",
                    "input_tokens": 128,
                    "generated_token_count": 2,
                    "prefill_forward_ms": 10.0,
                    "decode_forward_ms_total": 4.0,
                    "decode_forward_ms_per_token_avg": 2.0,
                    "decode_steps": 1,
                }
            },
            index=7,
            task="vt",
            sample_line_no=3,
        )

        self.assertEqual(record["record_type"], "sample_timing")
        self.assertEqual(record["task"], "vt")
        self.assertEqual(record["sample_line_no"], 3)
        self.assertEqual(record["sample_index"], 7)
        self.assertEqual(record["prefill_forward_ms"], 10.0)
        self.assertEqual(record["decode_forward_ms_per_token_avg"], 2.0)

    def test_attention_profile_record_documents_first_sample_policy(self):
        module = _load_call_api_module()

        record = module.build_attention_profile_record(
            profile={
                "timer_backend": "torch_profiler",
                "input_tokens": 128,
                "generated_token_count": 2,
                "prefill_attention_kernel_ms": 3.0,
                "decode_attention_kernel_ms_total": 1.5,
                "decode_attention_kernel_ms_per_token_avg": 0.75,
                "attention_kernel_event_count": 6,
            },
            task="vt",
            sample={"index": 7},
            sample_line_no=0,
            input_file=Path("/tmp/validation.jsonl"),
        )

        self.assertEqual(record["record_type"], "attention_profile")
        self.assertEqual(record["profile_sample_policy"], "first_input_record")
        self.assertEqual(record["sample_line_no"], 0)
        self.assertEqual(record["sample_index"], 7)
        self.assertEqual(record["input_file"], "/tmp/validation.jsonl")
        self.assertEqual(record["prefill_attention_kernel_ms"], 3.0)
        json.dumps(record, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
