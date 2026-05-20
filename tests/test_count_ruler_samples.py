import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "count_ruler_samples.py"


def load_module():
    spec = importlib.util.spec_from_file_location("count_ruler_samples", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n",
        encoding="utf-8",
    )


class CountRulerSamplesTest(unittest.TestCase):
    """验证 RULER 输入样本数统计脚本。"""

    def test_parse_lengths_accepts_suffixes_and_numbers(self):
        module = load_module()

        self.assertEqual(module.parse_lengths("4k,8192,64k"), [4096, 8192, 65536])

    def test_collect_counts_marks_missing_files(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            write_jsonl(
                data_root / "4096" / "data" / "niah_single_1" / "validation.jsonl",
                [{"index": 0}, {"index": 1}],
            )

            rows = module.collect_counts(
                data_root=data_root,
                lengths=[4096, 8192],
                tasks=["niah_single_1", "qa_1"],
                subset="validation",
            )

            by_task = {row["task"]: row for row in rows}
            self.assertEqual(by_task["niah_single_1"]["counts"][4096], 2)
            self.assertIsNone(by_task["niah_single_1"]["counts"][8192])
            self.assertIsNone(by_task["qa_1"]["counts"][4096])

    def test_table_output_uses_length_labels_and_missing_marker(self):
        module = load_module()
        rows = [
            {"task": "niah_single_1", "counts": {4096: 2, 8192: None}, "total": 2},
            {"task": "qa_1", "counts": {4096: None, 8192: 3}, "total": 3},
        ]

        table = module.format_table(rows, lengths=[4096, 8192])

        self.assertIn("task", table)
        self.assertIn("4k", table)
        self.assertIn("8k", table)
        self.assertIn("MISSING", table)
        self.assertIn("niah_single_1", table)

    def test_csv_output_is_machine_readable(self):
        module = load_module()
        rows = [
            {"task": "niah_single_1", "counts": {4096: 2, 8192: None}, "total": 2},
        ]

        csv_text = module.format_csv(rows, lengths=[4096, 8192])

        self.assertEqual(csv_text.splitlines()[0], "task,4k,8k,total")
        self.assertEqual(csv_text.splitlines()[1], "niah_single_1,2,MISSING,2")


if __name__ == "__main__":
    unittest.main()
