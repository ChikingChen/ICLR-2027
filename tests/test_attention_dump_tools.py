import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DUMP_TOOL_PATH = ROOT / "tools" / "dump_llama_attention.py"
INSPECT_TOOL_PATH = ROOT / "tools" / "inspect_attention_dump.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionDumpToolsTest(unittest.TestCase):
    """验证独立注意力导出和查看工具的数据格式。"""

    def test_dump_helpers_write_metadata_tokens_and_summary(self):
        dump_tool = _load_module("dump_llama_attention", DUMP_TOOL_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            prompt_tokens = [
                {"position": 0, "source": "prompt", "token_id": 11, "token_text": "A"},
                {"position": 1, "source": "prompt", "token_id": 12, "token_text": "B"},
            ]
            generated_tokens = [
                {
                    "generated_index": 0,
                    "position": 2,
                    "source": "generated",
                    "token_id": 13,
                    "token_text": "C",
                    "attention_file": "token_0000.npy",
                }
            ]
            metadata = dump_tool.build_metadata(
                model_path=Path("models/Meta-Llama-3.1-8B"),
                data_file=Path("sample.jsonl"),
                sample_offset=0,
                sample_index=7,
                prompt_text="prompt",
                prompt_token_count=2,
                generated_token_count=1,
                num_layers=2,
                num_heads=2,
                dtype="float32",
                attention_files=[
                    {
                        "generated_index": 0,
                        "file": "token_0000.npy",
                        "shape": [2, 2, 3],
                        "max_sum_error": 0.0,
                    }
                ],
            )

            dump_tool.write_json(output_dir / "metadata.json", metadata)
            dump_tool.write_jsonl(output_dir / "prompt_tokens.jsonl", prompt_tokens)
            dump_tool.write_jsonl(output_dir / "generated_tokens.jsonl", generated_tokens)
            dump_tool.write_summary(output_dir / "summary.md", metadata)

            loaded = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["sample_index"], 7)
            self.assertEqual(loaded["attention_files"][0]["shape"], [2, 2, 3])
            self.assertIn("完整 Attention 导出摘要", (output_dir / "summary.md").read_text(encoding="utf-8"))

    def test_attention_array_validation_requires_layer_head_position_shape_and_unit_sum(self):
        dump_tool = _load_module("dump_llama_attention", DUMP_TOOL_PATH)

        attention = np.array(
            [
                [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],
                [[0.4, 0.4, 0.2], [0.25, 0.25, 0.5]],
            ],
            dtype=np.float32,
        )

        summary = dump_tool.summarize_attention_array(attention)

        self.assertEqual(summary["shape"], [2, 2, 3])
        self.assertAlmostEqual(summary["max_sum_error"], 0.0, places=6)

    def test_inspector_prints_complete_attention_table(self):
        inspect_tool = _load_module("inspect_attention_dump", INSPECT_TOOL_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir)
            (dump_dir / "metadata.json").write_text(
                json.dumps({"prompt_token_count": 2}, ensure_ascii=False),
                encoding="utf-8",
            )
            (dump_dir / "prompt_tokens.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"position": 0, "source": "prompt", "token_id": 11, "token_text": "A"}),
                        json.dumps({"position": 1, "source": "prompt", "token_id": 12, "token_text": "B"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (dump_dir / "generated_tokens.jsonl").write_text(
                json.dumps(
                    {
                        "generated_index": 0,
                        "position": 2,
                        "source": "generated",
                        "token_id": 13,
                        "token_text": "C",
                        "attention_file": "token_0000.npy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            np.save(
                dump_dir / "token_0000.npy",
                np.array(
                    [
                        [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],
                        [[0.4, 0.4, 0.2], [0.25, 0.25, 0.5]],
                    ],
                    dtype=np.float32,
                ),
            )

            rows = inspect_tool.build_attention_rows(
                dump_dir=dump_dir,
                generated_token=0,
                layer=1,
                head=0,
            )
            table = inspect_tool.format_table(rows)

            self.assertEqual(len(rows), 3)
            self.assertIn("position | source", table)
            self.assertIn("generated", table)
            self.assertIn("0.400000", table)

    def test_inspector_can_sort_rows_by_attention_descending(self):
        inspect_tool = _load_module("inspect_attention_dump", INSPECT_TOOL_PATH)

        rows = [
            {"position": 0, "source": "prompt", "token_id": 11, "token_text": "A", "attention": 0.2},
            {"position": 1, "source": "prompt", "token_id": 12, "token_text": "B", "attention": 0.8},
            {"position": 2, "source": "generated", "token_id": 13, "token_text": "C", "attention": 0.5},
        ]

        sorted_rows = inspect_tool.sort_attention_rows(rows, sort_by="attention", descending=True)

        self.assertEqual([row["position"] for row in sorted_rows], [1, 2, 0])
        self.assertEqual([row["position"] for row in rows], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
