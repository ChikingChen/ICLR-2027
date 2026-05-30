import importlib.util
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "eval" / "collect_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_results", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


class CollectResultsTest(unittest.TestCase):
    """验证 RULER 汇总脚本的明细、聚合和 csv 输出。"""

    def _make_fixture(self, root):
        output_root = root / "local_eval"
        data_root = root / "parquet_data" / "synthetic"
        for task in ["niah_single_1", "qa_1"]:
            _write_jsonl(
                data_root / "4096" / "data" / task / "validation.jsonl",
                [
                    {"index": 0, "input": "问题0", "outputs": ["答案0"], "others": {}},
                    {"index": 1, "input": "问题1", "outputs": ["答案1"], "others": {}},
                ],
            )
        _write_jsonl(
            output_root / "model-a" / "synthetic" / "4096" / "pred" / "niah_single_1.jsonl",
            [
                {
                    "index": 0,
                    "input": "问题0",
                    "outputs": ["答案0"],
                    "others": {},
                    "pred": "答案0",
                    "generation_logprob_sum": -1.5,
                    "generation_token_count": 3,
                },
                {
                    "index": 1,
                    "input": "问题1",
                    "outputs": ["答案1"],
                    "others": {},
                    "pred": "答案1",
                    "generation_logprob_sum": -2.5,
                    "generation_token_count": 5,
                },
            ],
        )
        _write_jsonl(
            output_root / "model-a" / "synthetic" / "4096" / "pred" / "niah_single_1.generation_timing.jsonl",
            [
                {
                    "record_type": "sample_timing",
                    "task": "niah_single_1",
                    "sample_line_no": 0,
                    "sample_index": 0,
                    "timer_backend": "cuda_event",
                    "input_tokens": 128,
                    "generated_token_count": 2,
                    "prefill_forward_ms": 10.0,
                    "decode_forward_ms_total": 4.0,
                    "decode_forward_ms_per_token_avg": 2.0,
                    "decode_steps": 1,
                },
                {
                    "record_type": "sample_timing",
                    "task": "niah_single_1",
                    "sample_line_no": 1,
                    "sample_index": 1,
                    "timer_backend": "cuda_event",
                    "input_tokens": 130,
                    "generated_token_count": 3,
                    "prefill_forward_ms": 20.0,
                    "decode_forward_ms_total": 6.0,
                    "decode_forward_ms_per_token_avg": 2.0,
                    "decode_steps": 2,
                },
                {
                    "record_type": "attention_profile",
                    "task": "niah_single_1",
                    "profile_sample_policy": "all_input_records",
                    "sample_line_no": 0,
                    "sample_index": 0,
                    "timer_backend": "cuda_event_attention_ops",
                    "input_tokens": 128,
                    "generated_token_count": 2,
                    "prefill_attention_kernel_ms": 3.0,
                    "decode_attention_kernel_ms_total": 1.5,
                    "decode_attention_kernel_ms_per_token_avg": 0.75,
                    "attention_kernel_event_count": 6,
                },
                {
                    "record_type": "attention_profile",
                    "task": "niah_single_1",
                    "profile_sample_policy": "all_input_records",
                    "sample_line_no": 1,
                    "sample_index": 1,
                    "timer_backend": "cuda_event_attention_ops",
                    "input_tokens": 130,
                    "generated_token_count": 3,
                    "prefill_attention_kernel_ms": 9.0,
                    "decode_attention_kernel_ms_total": 7.5,
                    "decode_attention_kernel_ms_per_token_avg": 2.5,
                    "attention_kernel_event_count": 4,
                },
            ],
        )
        timing_file = root / "timing.jsonl"
        _write_jsonl(
            timing_file,
            [
                {
                    "model": "model-a",
                    "length": 4096,
                    "task": "niah_single_1",
                    "gpu": 2,
                    "started_at": 100.0,
                    "ended_at": 112.5,
                    "elapsed_seconds": 12.5,
                }
            ],
        )
        return output_root, data_root, timing_file

    def test_detail_rows_are_unique_by_model_length_task(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))

            workbook = module.collect_results(
                output_root=output_root,
                data_root=data_root,
                benchmark="synthetic",
                models=["model-a"],
                lengths=[4096],
                tasks=["niah_single_1", "qa_1"],
                timing_file=timing_file,
            )

            detail = workbook["detail"]
            keys = [(row["model"], row["length"], row["task"]) for row in detail]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(detail), 2)

    def test_missing_prediction_file_sets_missing_status(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))

            workbook = module.collect_results(
                output_root=output_root,
                data_root=data_root,
                benchmark="synthetic",
                models=["model-a"],
                lengths=[4096],
                tasks=["niah_single_1", "qa_1"],
                timing_file=timing_file,
            )

            detail = workbook["detail"]
            status_by_task = {row["task"]: row["status"] for row in detail}
            self.assertEqual(status_by_task["niah_single_1"], "completed")
            self.assertEqual(status_by_task["qa_1"], "missing")
            completed_row = next(row for row in detail if row["task"] == "niah_single_1")
            self.assertEqual(completed_row["gpu"], 2)
            self.assertTrue(completed_row["pred_file"].endswith("niah_single_1.jsonl"))
            self.assertTrue(completed_row["log_file"].endswith("niah_single_1.log"))

    def test_summary_by_model_has_elapsed_and_wall_time(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))

            workbook = module.collect_results(
                output_root=output_root,
                data_root=data_root,
                benchmark="synthetic",
                models=["model-a"],
                lengths=[4096],
                tasks=["niah_single_1", "qa_1"],
                timing_file=timing_file,
            )

            summary = workbook["summary_by_model"][0]
            self.assertEqual(summary["total_task_elapsed_seconds"], 12.5)
            self.assertEqual(summary["wall_time_seconds"], 12.5)

    def test_csv_writes_detail_and_summary_files(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))
            output_file = Path(tmp_dir) / "results.csv"

            module.main(
                [
                    "--output-root",
                    str(output_root),
                    "--data-root",
                    str(data_root),
                    "--models",
                    "model-a",
                    "--seq-lengths",
                    "4096",
                    "--tasks",
                    "niah_single_1,qa_1",
                    "--timing-file",
                    str(timing_file),
                    "--output-file",
                    str(output_file),
                ]
            )

            detail_rows = _read_csv(output_file)
            self.assertEqual(detail_rows[0]["model"], "model-a")
            self.assertEqual(detail_rows[0]["task"], "niah_single_1")
            self.assertEqual(detail_rows[0]["status"], "completed")

            summary_file = Path(tmp_dir) / "results_summary_by_model_and_length.csv"
            self.assertTrue(summary_file.exists())
            summary_rows = _read_csv(summary_file)
            self.assertEqual(summary_rows[0]["model"], "model-a")
            self.assertEqual(summary_rows[0]["length"], "4096")

            expected_files = {
                "results_summary_by_model.csv",
                "results_summary_by_model_and_length.csv",
                "results_summary_by_task.csv",
                "results_run_info.csv",
            }
            self.assertTrue(expected_files.issubset({path.name for path in Path(tmp_dir).glob("*.csv")}))

    def test_xlsx_output_file_is_rejected(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))
            output_file = Path(tmp_dir) / "results.xlsx"

            with self.assertRaisesRegex(ValueError, r"\.csv"):
                module.main(
                    [
                        "--output-root",
                        str(output_root),
                        "--data-root",
                        str(data_root),
                        "--models",
                        "model-a",
                        "--seq-lengths",
                        "4096",
                        "--tasks",
                        "niah_single_1,qa_1",
                        "--timing-file",
                        str(timing_file),
                        "--output-file",
                        str(output_file),
                    ]
                )

    def test_xlsx_output_file_is_rejected_before_data_discovery(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_file = root / "results.xlsx"

            with self.assertRaisesRegex(ValueError, r"\.csv"):
                module.main(
                    [
                        "--output-root",
                        str(root / "missing_local_eval"),
                        "--data-root",
                        str(root / "missing_data_root"),
                        "--models",
                        "model-a",
                        "--output-file",
                        str(output_file),
                    ]
                )

    def test_generation_ppl_is_aggregated_from_logprob_and_tokens(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))

            workbook = module.collect_results(
                output_root=output_root,
                data_root=data_root,
                benchmark="synthetic",
                models=["model-a"],
                lengths=[4096],
                tasks=["niah_single_1"],
                timing_file=timing_file,
            )

            detail_row = workbook["detail"][0]
            self.assertAlmostEqual(detail_row["generation_nll"], 0.5)
            self.assertAlmostEqual(detail_row["generation_ppl"], math.exp(0.5))
            summary_row = workbook["summary_by_model"][0]
            self.assertAlmostEqual(summary_row["avg_generation_nll"], 0.5)
            self.assertAlmostEqual(summary_row["avg_generation_ppl"], math.exp(0.5))

    def test_generation_timing_sidecar_is_aggregated(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))

            workbook = module.collect_results(
                output_root=output_root,
                data_root=data_root,
                benchmark="synthetic",
                models=["model-a"],
                lengths=[4096],
                tasks=["niah_single_1"],
                timing_file=timing_file,
            )

            detail_row = workbook["detail"][0]
            self.assertEqual(detail_row["sample_timing_records"], 2)
            self.assertEqual(detail_row["attention_profile_sample_line_no"], 0)
            self.assertEqual(detail_row["attention_profile_sample_index"], 0)
            self.assertEqual(detail_row["attention_profile_records"], 2)
            self.assertAlmostEqual(detail_row["prefill_attention_kernel_ms_total"], 12.0)
            self.assertAlmostEqual(detail_row["attention_profile_generated_token_count_total"], 5.0)
            self.assertAlmostEqual(detail_row["prefill_forward_ms_total"], 30.0)
            self.assertAlmostEqual(detail_row["decode_forward_ms_total"], 10.0)
            self.assertAlmostEqual(detail_row["decode_forward_ms_per_token_avg"], 2.0)
            self.assertAlmostEqual(detail_row["prefill_attention_kernel_ms"], 6.0)
            self.assertAlmostEqual(detail_row["decode_attention_kernel_ms_total"], 9.0)
            self.assertAlmostEqual(detail_row["decode_attention_kernel_ms_per_token_avg"], 1.8)
            self.assertEqual(detail_row["attention_kernel_event_count"], 10)

            summary_row = workbook["summary_by_model"][0]
            self.assertAlmostEqual(summary_row["total_prefill_forward_ms"], 30.0)
            self.assertAlmostEqual(summary_row["total_decode_forward_ms"], 10.0)
            self.assertAlmostEqual(summary_row["avg_decode_forward_ms_per_token"], 2.0)
            self.assertEqual(summary_row["attention_profiled_tasks"], 1)
            self.assertEqual(summary_row["attention_profile_records"], 2)
            self.assertAlmostEqual(summary_row["avg_prefill_attention_kernel_ms"], 6.0)
            self.assertAlmostEqual(summary_row["avg_decode_attention_kernel_ms_per_token"], 1.8)


if __name__ == "__main__":
    unittest.main()
