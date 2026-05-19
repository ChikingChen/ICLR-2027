import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "RULER" / "scripts" / "data" / "prepare_parquet.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_parquet", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareParquetTest(unittest.TestCase):
    def test_parse_dataset_dir_keeps_task_prefix_with_underscores(self):
        module = load_module()

        self.assertEqual(
            module.parse_dataset_dir_name("niah_multikey_1_128k"),
            ("niah_multikey_1", "128k", 131072),
        )
        self.assertEqual(
            module.parse_dataset_dir_name("qa_2_1M"),
            ("qa_2", "1M", 1048576),
        )

    def test_parse_length_filter_accepts_suffixes_and_numbers(self):
        module = load_module()

        self.assertEqual(module.parse_length_filter("4k,131072,1M"), {4096, 131072, 1048576})

    def test_normalize_record_converts_answers_to_outputs(self):
        module = load_module()
        record = {
            "index": 7,
            "input": "问题",
            "answers": ["答案甲", "答案乙"],
            "length": 4096,
            "predictions": {"old": "ignored"},
        }

        self.assertEqual(
            module.normalize_record(record, task="niah_single_1", length_suffix="4k"),
            {
                "index": 7,
                "input": "问题",
                "outputs": ["答案甲", "答案乙"],
                "others": {},
                "truncation": -1,
                "length": 4096,
            },
        )

    def test_write_jsonl_sorts_limits_and_writes_utf8_records(self):
        module = load_module()
        records = [
            {"index": 2, "input": "乙", "answers": ["B"], "length": 4096},
            {"index": 1, "input": "甲", "answers": ["A"], "length": 4096},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "validation.jsonl"
            written = module.write_jsonl(
                records,
                output_file,
                task="niah_single_1",
                length_suffix="4k",
                max_samples=1,
            )

            self.assertEqual(written, 1)
            lines = output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["index"], 1)
            self.assertIn("甲", lines[0])


if __name__ == "__main__":
    unittest.main()
