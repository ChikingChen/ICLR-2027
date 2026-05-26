import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DUMP_TOOL_PATH = ROOT / "tools" / "dump_llama_attention.py"
INSPECT_TOOL_PATH = ROOT / "tools" / "inspect_attention_dump.py"
BOS_SUMMARY_TOOL_PATH = ROOT / "tools" / "summarize_bos_attention.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionDumpToolsTest(unittest.TestCase):
    """验证独立注意力导出和查看工具的数据格式。"""

    def write_bos_summary_dump(self, dump_dir: Path) -> None:
        (dump_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_index": 7,
                    "generated_token_count": 2,
                    "num_layers": 2,
                    "num_heads": 2,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (dump_dir / "prompt_tokens.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "position": 0,
                            "source": "prompt",
                            "token_id": 128000,
                            "token_text": "<|begin_of_text|>",
                        }
                    ),
                    json.dumps({"position": 1, "source": "prompt", "token_id": 11, "token_text": "A"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (dump_dir / "generated_tokens.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "generated_index": 0,
                            "position": 2,
                            "source": "generated",
                            "token_id": 12,
                            "token_text": "B",
                            "attention_file": "token_0000.npy",
                        }
                    ),
                    json.dumps(
                        {
                            "generated_index": 1,
                            "position": 3,
                            "source": "generated",
                            "token_id": 13,
                            "token_text": "C",
                            "attention_file": "token_0001.npy",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        np.save(
            dump_dir / "token_0000.npy",
            np.array(
                [
                    [[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]],
                    [[0.4, 0.4, 0.2], [0.0, 0.5, 0.5]],
                ],
                dtype=np.float32,
            ),
        )
        np.save(
            dump_dir / "token_0001.npy",
            np.array(
                [
                    [[0.25, 0.25, 0.25, 0.25], [0.8, 0.1, 0.1, 0.0]],
                    [[0.05, 0.9, 0.03, 0.02], [0.2, 0.2, 0.2, 0.4]],
                ],
                dtype=np.float32,
            ),
        )

    def test_apply_bos_attention_mask_zeroes_bos_and_preserves_positions(self):
        dump_tool = _load_module("dump_llama_attention", DUMP_TOOL_PATH)

        self.assertTrue(hasattr(dump_tool, "apply_bos_attention_mask"))

        class FakeTokenizer:
            bos_token_id = 128000

            def decode(self, token_ids, skip_special_tokens=False):
                return "<|begin_of_text|>" if token_ids == [128000] else str(token_ids[0])

        inputs = {
            "input_ids": torch.tensor([[128000, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

        masked_inputs, ablation = dump_tool.apply_bos_attention_mask(inputs, FakeTokenizer())

        self.assertEqual(masked_inputs["attention_mask"].tolist(), [[0, 1, 1]])
        self.assertEqual(masked_inputs["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertTrue(ablation["enabled"])
        self.assertEqual(ablation["token_position"], 0)
        self.assertEqual(ablation["token_id"], 128000)
        self.assertEqual(ablation["token_text"], "<|begin_of_text|>")

    def test_apply_bos_attention_mask_rejects_missing_bos_at_position_zero(self):
        dump_tool = _load_module("dump_llama_attention", DUMP_TOOL_PATH)

        class FakeTokenizer:
            bos_token_id = 128000

        inputs = {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "expected BOS token id 128000 at position 0"):
            dump_tool.apply_bos_attention_mask(inputs, FakeTokenizer())

    def test_prompt_token_records_include_attention_mask_state(self):
        dump_tool = _load_module("dump_llama_attention", DUMP_TOOL_PATH)

        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {128000: "<|begin_of_text|>", 11: "A"}[token_ids[0]]

        rows = dump_tool.build_token_records(
            FakeTokenizer(),
            [128000, 11],
            source="prompt",
            attention_mask_values=[0, 1],
        )

        self.assertEqual(rows[0]["attention_mask"], 0)
        self.assertTrue(rows[0]["masked"])
        self.assertEqual(rows[1]["attention_mask"], 1)
        self.assertFalse(rows[1]["masked"])

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
                generated_text="answer",
                attention_mask_ablation={
                    "enabled": True,
                    "mode": "bos_token",
                    "token_position": 0,
                    "token_id": 128000,
                    "token_text": "<|begin_of_text|>",
                    "semantics": "token kept in input_ids; attention_mask is 0 during generation and replay",
                },
            )

            dump_tool.write_json(output_dir / "metadata.json", metadata)
            dump_tool.write_jsonl(output_dir / "prompt_tokens.jsonl", prompt_tokens)
            dump_tool.write_jsonl(output_dir / "generated_tokens.jsonl", generated_tokens)
            dump_tool.write_summary(output_dir / "summary.md", metadata)

            loaded = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["sample_index"], 7)
            self.assertEqual(loaded["attention_files"][0]["shape"], [2, 2, 3])
            self.assertEqual(loaded["generated_text"], "answer")
            self.assertTrue(loaded["attention_mask_ablation"]["enabled"])
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

    def test_bos_summary_counts_top1_and_mean_attention_across_generated_layer_heads(self):
        summary_tool = _load_module("summarize_bos_attention", BOS_SUMMARY_TOOL_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir)
            self.write_bos_summary_dump(dump_dir)

            summary = summary_tool.summarize_bos_attention(dump_dir)

            self.assertEqual(summary["sample_index"], 7)
            self.assertEqual(summary["generated_token_count"], 2)
            self.assertEqual(summary["num_layers"], 2)
            self.assertEqual(summary["num_heads"], 2)
            self.assertEqual(summary["total_units"], 8)
            self.assertEqual(summary["bos_top1_count"], 4)
            self.assertAlmostEqual(summary["bos_top1_ratio"], 0.5)
            self.assertAlmostEqual(summary["bos_attention_mean"], 0.3)
            self.assertAlmostEqual(summary["bos_attention_min"], 0.0)
            self.assertAlmostEqual(summary["bos_attention_max"], 0.8)

    def test_bos_summary_rejects_missing_bos_token_text(self):
        summary_tool = _load_module("summarize_bos_attention", BOS_SUMMARY_TOOL_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir)
            (dump_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (dump_dir / "prompt_tokens.jsonl").write_text(
                json.dumps({"position": 0, "source": "prompt", "token_id": 11, "token_text": "A"}) + "\n",
                encoding="utf-8",
            )
            (dump_dir / "generated_tokens.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not find token_text '<\\|begin_of_text\\|>'"):
                summary_tool.summarize_bos_attention(dump_dir)

    def test_bos_summary_cli_prints_human_readable_totals(self):
        summary_tool = _load_module("summarize_bos_attention", BOS_SUMMARY_TOOL_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir)
            self.write_bos_summary_dump(dump_dir)
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = summary_tool.main(["--dump-dir", str(dump_dir)])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("bos_top1_count: 4", output)
            self.assertIn("bos_top1_ratio: 0.50000000", output)
            self.assertIn("bos_attention_mean: 0.30000001", output)


if __name__ == "__main__":
    unittest.main()
