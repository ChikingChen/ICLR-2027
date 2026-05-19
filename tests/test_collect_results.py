import importlib.util
import json
import math
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "eval" / "collect_results.py"
NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


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


def _shared_strings(xlsx_path):
    with zipfile.ZipFile(xlsx_path) as archive:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("main:si", NS):
        values.append("".join(node.text or "" for node in item.findall(".//main:t", NS)))
    return values


def _sheet_names(xlsx_path):
    with zipfile.ZipFile(xlsx_path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return [sheet.attrib["name"] for sheet in root.findall("main:sheets/main:sheet", NS)]


class CollectResultsTest(unittest.TestCase):
    """验证 RULER 汇总脚本的明细、聚合和 xlsx 输出。"""

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

    def test_xlsx_contains_summary_by_model_and_length_sheet(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root, data_root, timing_file = self._make_fixture(Path(tmp_dir))
            output_file = Path(tmp_dir) / "results.xlsx"

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

            self.assertIn("summary_by_model_and_length", _sheet_names(output_file))
            self.assertIn("summary_by_model_and_length", _shared_strings(output_file))

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


if __name__ == "__main__":
    unittest.main()
