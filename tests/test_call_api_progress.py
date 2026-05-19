import contextlib
import importlib.util
import io
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
        self.assertIn("[BATCH_START]", source)
        self.assertIn("[BATCH_DONE]", source)
        self.assertIn("[BATCH_FAILED]", source)

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


if __name__ == "__main__":
    unittest.main()
