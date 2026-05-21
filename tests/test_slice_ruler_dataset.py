import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "slice_ruler_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("slice_ruler_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for index in range(count):
            record = {
                "index": index,
                "input": f"问题 {index}",
                "outputs": [f"答案 {index}"],
                "others": {},
                "truncation": -1,
                "length": 4096,
            }
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


class SliceRulerDatasetTest(unittest.TestCase):
    """验证 RULER jsonl 子数据集的固定随机划分行为。"""

    def test_slice_file_uses_fixed_random_positions_and_preserves_order(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_file = tmp_path / "source.jsonl"
            target_file = tmp_path / "target.jsonl"
            write_jsonl(source_file, 10)
            original_text = source_file.read_text(encoding="utf-8")

            result = module.slice_jsonl_file(
                source_file=source_file,
                target_file=target_file,
                sample_count=3,
                seed=0,
            )

            records = [
                json.loads(line)
                for line in target_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["index"] for record in records], [0, 6, 9])
            self.assertEqual(result["selected_line_numbers"], [1, 7, 10])
            self.assertEqual(result["selected_indices"], [0, 6, 9])
            self.assertEqual(source_file.read_text(encoding="utf-8"), original_text)

    def test_slice_dataset_writes_runner_compatible_layout_and_report(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "source" / "synthetic"
            target_root = tmp_path / "target" / "synthetic"
            first_source = source_root / "4096" / "data" / "niah_single_1" / "validation.jsonl"
            second_source = source_root / "8192" / "data" / "qa_2" / "validation.jsonl"
            write_jsonl(first_source, 10)
            write_jsonl(second_source, 10)

            report = module.slice_dataset(
                source_root=source_root,
                target_root=target_root,
                lengths=[4096, 8192],
                tasks=["niah_single_1", "qa_2"],
                subset="validation",
                sample_count=3,
                seed=0,
                overwrite=True,
            )

            self.assertEqual(report["source_root"], str(source_root))
            self.assertEqual(report["target_root"], str(target_root))
            self.assertEqual(report["lengths"], [4096, 8192])
            self.assertEqual(report["tasks"], ["niah_single_1", "qa_2"])
            self.assertEqual(len(report["sliced"]), 2)
            self.assertEqual(report["missing"], 2)
            target_file = target_root / "4096" / "data" / "niah_single_1" / "validation.jsonl"
            report_file = target_root / "subset_report.json"
            self.assertTrue(target_file.exists())
            self.assertTrue(report_file.exists())
            self.assertEqual(len(target_file.read_text(encoding="utf-8").splitlines()), 3)

    def test_slice_file_rejects_short_sources(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_file = tmp_path / "source.jsonl"
            target_file = tmp_path / "target.jsonl"
            write_jsonl(source_file, 2)

            with self.assertRaisesRegex(ValueError, "样本数不足"):
                module.slice_jsonl_file(
                    source_file=source_file,
                    target_file=target_file,
                    sample_count=3,
                    seed=0,
                )
            self.assertFalse(target_file.exists())


if __name__ == "__main__":
    unittest.main()
