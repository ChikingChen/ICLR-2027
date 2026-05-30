import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "compare_pooling_attention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_pooling_attention", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoolingAttentionCompareTest(unittest.TestCase):
    """验证 pooling token 和细粒度 token attention 的对照格式。"""

    def test_apply_bos_attention_mask_zeroes_bos_for_pooling_tool(self):
        tool = _load_module()

        self.assertTrue(hasattr(tool, "apply_bos_attention_mask"))

        class FakeTokenizer:
            bos_token_id = 128000

            def decode(self, token_ids, skip_special_tokens=False):
                return "<|begin_of_text|>" if token_ids == [128000] else str(token_ids[0])

        inputs = {
            "input_ids": torch.tensor([[128000, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

        masked_inputs, ablation = tool.apply_bos_attention_mask(inputs, FakeTokenizer())

        self.assertEqual(masked_inputs["attention_mask"].tolist(), [[0, 1, 1]])
        self.assertEqual(masked_inputs["position_ids"].tolist(), [[0, 1, 2]])
        self.assertTrue(ablation["enabled"])
        self.assertEqual(ablation["token_text"], "<|begin_of_text|>")

    def test_build_parser_accepts_mask_bos_token_flag(self):
        tool = _load_module()

        args = tool.build_parser().parse_args(
            [
                "--model-path",
                "models/Llama-3.1-8B",
                "--data-file",
                "sample.jsonl",
                "--output-dir",
                "out",
                "--mask-bos-token",
            ]
        )

        self.assertTrue(args.mask_bos_token)

    def test_build_parser_accepts_true_pooling_attention_flag(self):
        tool = _load_module()

        args = tool.build_parser().parse_args(
            [
                "--model-path",
                "models/Llama-3.1-8B",
                "--data-file",
                "sample.jsonl",
                "--output-dir",
                "out",
                "--true-pooling-attention",
            ]
        )

        self.assertTrue(args.true_pooling_attention)

    def test_build_prompt_blocks_splits_tokens_into_fixed_size_ranges(self):
        tool = _load_module()

        blocks = tool.build_prompt_blocks(token_count=5, block_size=2)

        self.assertEqual(
            blocks,
            [
                {"block_id": 0, "start_position": 0, "end_position": 1, "token_count": 2},
                {"block_id": 1, "start_position": 2, "end_position": 3, "token_count": 2},
                {"block_id": 2, "start_position": 4, "end_position": 4, "token_count": 1},
            ],
        )

    def test_compare_blocks_reports_max_avg_pooling_and_fine_token_scores(self):
        tool = _load_module()
        attention = np.array(
            [
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.20, 0.10, 0.50, 0.20],
                ],
                [
                    [0.05, 0.25, 0.20, 0.50],
                    [0.15, 0.35, 0.25, 0.25],
                ],
            ],
            dtype=np.float32,
        )
        token_records = [
            {"position": 0, "source": "prompt", "token_id": 10, "token_text": "A"},
            {"position": 1, "source": "prompt", "token_id": 11, "token_text": "B"},
            {"position": 2, "source": "prompt", "token_id": 12, "token_text": "C"},
            {"position": 3, "source": "prompt", "token_id": 13, "token_text": "D"},
        ]

        comparison = tool.build_pooling_comparison(
            attention=attention,
            prompt_tokens=token_records,
            block_size=2,
            top_k_blocks=1,
        )

        pooling_tokens = comparison["pooling_tokens"]
        fine_tokens = comparison["fine_tokens"]
        summary = comparison["pooling_vs_fine_summary"]

        self.assertEqual(len(pooling_tokens), 2)
        self.assertAlmostEqual(fine_tokens[0]["full_attention_mean"], 0.125)
        self.assertAlmostEqual(fine_tokens[1]["full_attention_mean"], 0.225)
        self.assertAlmostEqual(fine_tokens[2]["full_attention_mean"], 0.3125)
        self.assertAlmostEqual(fine_tokens[3]["full_attention_mean"], 0.3375)

        self.assertAlmostEqual(pooling_tokens[0]["pooling_score_max"], 0.225)
        self.assertAlmostEqual(pooling_tokens[0]["pooling_score_avg"], 0.175)
        self.assertAlmostEqual(pooling_tokens[0]["full_attention_sum"], 0.35)
        self.assertAlmostEqual(pooling_tokens[1]["pooling_score_max"], 0.3375)
        self.assertAlmostEqual(pooling_tokens[1]["pooling_score_avg"], 0.325)
        self.assertTrue(pooling_tokens[1]["selected_by_max"])
        self.assertTrue(pooling_tokens[1]["selected_by_avg"])

        self.assertEqual(summary[0]["block_id"], 0)
        self.assertEqual([token["position"] for token in summary[0]["tokens"]], [0, 1])
        self.assertEqual(summary[0]["tokens"][0]["token_text"], "A")

    def test_build_pooling_attention_csv_rows_reports_layer_head_block_comparison(self):
        tool = _load_module()
        attention = np.array(
            [
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.20, 0.10, 0.50, 0.20],
                ],
                [
                    [0.05, 0.25, 0.20, 0.50],
                    [0.15, 0.35, 0.25, 0.25],
                ],
            ],
            dtype=np.float32,
        )
        token_records = [
            {"position": 0, "source": "prompt", "token_id": 10, "token_text": "A"},
            {"position": 1, "source": "prompt", "token_id": 11, "token_text": "B"},
            {"position": 2, "source": "prompt", "token_id": 12, "token_text": "C"},
            {"position": 3, "source": "prompt", "token_id": 13, "token_text": "D"},
        ]

        rows = tool.build_pooling_attention_csv_rows(
            attention=attention,
            prompt_tokens=token_records,
            block_size=2,
            metadata={
                "sample_index": 7,
                "query_generated_index": 0,
                "query_token_id": 25,
                "query_token_text": ":",
            },
        )

        self.assertEqual(len(rows), 8)
        first = rows[0]
        self.assertEqual(first["sample_index"], 7)
        self.assertEqual(first["query_generated_index"], 0)
        self.assertEqual(first["query_token_text"], ":")
        self.assertEqual(first["layer"], 0)
        self.assertEqual(first["head"], 0)
        self.assertEqual(first["block_id"], 0)
        self.assertEqual(first["start_position"], 0)
        self.assertEqual(first["end_position"], 1)
        self.assertEqual(first["token_count"], 2)
        self.assertEqual(first["token_positions"], "[0, 1]")
        self.assertEqual(first["token_texts"], "[\"A\", \"B\"]")
        self.assertEqual(first["fine_attention_values"], "[0.1, 0.2]")
        self.assertAlmostEqual(first["fine_attention_sum"], 0.3)
        self.assertAlmostEqual(first["avg_pooling_attention"], 0.15)
        self.assertAlmostEqual(first["max_pooling_attention"], 0.2)
        self.assertAlmostEqual(first["avg_minus_sum"], -0.15)
        self.assertAlmostEqual(first["max_minus_sum"], -0.1)
        self.assertFalse(first["avg_equals_sum"])
        self.assertFalse(first["max_equals_sum"])
        self.assertAlmostEqual(first["avg_over_sum"], 0.5)
        self.assertAlmostEqual(first["max_over_sum"], 2.0 / 3.0)

    def test_build_true_pooling_attention_csv_rows_recomputes_softmax_over_blocks(self):
        tool = _load_module()
        dense_attention = np.array([[[0.2, 0.3, 0.5]]], dtype=np.float32)
        query_states = np.array([[[1.0, 0.0]]], dtype=np.float32)
        key_states = np.array([[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]], dtype=np.float32)

        rows = tool.build_true_pooling_attention_csv_rows(
            dense_attention=dense_attention,
            query_states=query_states,
            key_states=key_states,
            block_size=2,
            query_generated_index=0,
        )

        self.assertEqual(tool.TRUE_POOLING_ATTENTION_CSV_FIELDS, list(rows[0].keys()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["query_generated_index"], 0)
        self.assertEqual(rows[0]["token_range"], "1-2")
        self.assertEqual(rows[1]["token_range"], "3-3")
        self.assertAlmostEqual(rows[0]["dense_attention_sum"], 0.5)
        self.assertAlmostEqual(rows[1]["dense_attention_sum"], 0.5)

        avg_logits = np.array([0.5, 2.0], dtype=np.float64) / np.sqrt(2.0)
        avg_expected = np.exp(avg_logits - avg_logits.max())
        avg_expected = avg_expected / avg_expected.sum()
        max_logits = np.array([1.0, 2.0], dtype=np.float64) / np.sqrt(2.0)
        max_expected = np.exp(max_logits - max_logits.max())
        max_expected = max_expected / max_expected.sum()

        self.assertAlmostEqual(rows[0]["avg_pooling_attention"], avg_expected[0])
        self.assertAlmostEqual(rows[1]["avg_pooling_attention"], avg_expected[1])
        self.assertAlmostEqual(rows[0]["max_pooling_attention"], max_expected[0])
        self.assertAlmostEqual(rows[1]["max_pooling_attention"], max_expected[1])
        self.assertEqual(rows[0]["layer"], 0)
        self.assertEqual(rows[0]["head"], 0)

    def test_summarize_true_pooling_rows_reports_column_sum_errors(self):
        tool = _load_module()
        dense_attention = np.array([[[0.2, 0.3, 0.5]]], dtype=np.float32)
        query_states = np.array([[[1.0, 0.0]]], dtype=np.float32)
        key_states = np.array([[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]], dtype=np.float32)
        rows = tool.build_true_pooling_attention_csv_rows(
            dense_attention=dense_attention,
            query_states=query_states,
            key_states=key_states,
            block_size=2,
            query_generated_index=0,
        )

        summary = tool.summarize_true_pooling_attention_rows(rows)

        self.assertEqual(summary["row_count"], 2)
        self.assertAlmostEqual(summary["dense_attention_sum_max_error"], 0.0)
        self.assertAlmostEqual(summary["avg_pooling_attention_sum_max_error"], 0.0)
        self.assertAlmostEqual(summary["max_pooling_attention_sum_max_error"], 0.0)

    def test_write_outputs_persists_pooling_and_fine_attention_files(self):
        tool = _load_module()
        comparison = {
            "pooling_tokens": [
                {
                    "block_id": 0,
                    "start_position": 0,
                    "end_position": 1,
                    "token_count": 2,
                    "pooling_score_max": 0.2,
                    "pooling_score_avg": 0.15,
                    "full_attention_sum": 0.3,
                    "full_attention_mean": 0.15,
                    "full_attention_max": 0.2,
                    "rank_by_max": 1,
                    "rank_by_avg": 1,
                    "selected_by_max": True,
                    "selected_by_avg": True,
                }
            ],
            "fine_tokens": [
                {
                    "position": 0,
                    "block_id": 0,
                    "token_id": 10,
                    "token_text": "A",
                    "full_attention_mean": 0.1,
                    "full_attention_max": 0.2,
                    "full_attention_min": 0.0,
                }
            ],
            "pooling_vs_fine_summary": [
                {
                    "block_id": 0,
                    "start_position": 0,
                    "end_position": 1,
                    "pooling_score_max": 0.2,
                    "pooling_score_avg": 0.15,
                    "full_attention_sum": 0.3,
                    "tokens": [
                        {
                            "position": 0,
                            "token_id": 10,
                            "token_text": "A",
                            "full_attention_mean": 0.1,
                            "full_attention_max": 0.2,
                        }
                    ],
                }
            ],
        }
        metadata = {"model_path": "models/Llama-3.1-8B", "block_size": 2}
        attention = np.array([[[0.1, 0.2]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            tool.write_outputs(output_dir, metadata, comparison, attention)

            pooling_row = json.loads((output_dir / "pooling_tokens.jsonl").read_text(encoding="utf-8"))
            fine_row = json.loads((output_dir / "fine_tokens.jsonl").read_text(encoding="utf-8"))
            summary_row = json.loads(
                (output_dir / "pooling_vs_fine_summary.jsonl").read_text(encoding="utf-8")
            )

            self.assertEqual(pooling_row["pooling_score_max"], 0.2)
            self.assertEqual(fine_row["full_attention_mean"], 0.1)
            self.assertEqual(summary_row["tokens"][0]["position"], 0)
            self.assertTrue((output_dir / "attention_detail.npz").exists())
            self.assertIn("Pooling Token", (output_dir / "summary.md").read_text(encoding="utf-8"))

    def test_write_outputs_persists_pooling_attention_comparison_csv(self):
        tool = _load_module()
        comparison = {
            "pooling_tokens": [
                {
                    "block_id": 0,
                    "start_position": 0,
                    "end_position": 1,
                    "token_count": 2,
                    "pooling_score_max": 0.2,
                    "pooling_score_avg": 0.15,
                    "full_attention_sum": 0.3,
                    "full_attention_mean": 0.15,
                    "full_attention_max": 0.2,
                    "rank_by_max": 1,
                    "rank_by_avg": 1,
                    "selected_by_max": True,
                    "selected_by_avg": True,
                }
            ],
            "fine_tokens": [
                {
                    "position": 0,
                    "block_id": 0,
                    "token_id": 10,
                    "token_text": "A",
                    "full_attention_mean": 0.1,
                    "full_attention_max": 0.2,
                    "full_attention_min": 0.0,
                },
                {
                    "position": 1,
                    "block_id": 0,
                    "token_id": 11,
                    "token_text": "B",
                    "full_attention_mean": 0.2,
                    "full_attention_max": 0.3,
                    "full_attention_min": 0.0,
                },
            ],
            "pooling_vs_fine_summary": [
                {
                    "block_id": 0,
                    "start_position": 0,
                    "end_position": 1,
                    "pooling_score_max": 0.2,
                    "pooling_score_avg": 0.15,
                    "full_attention_sum": 0.3,
                    "tokens": [
                        {
                            "position": 0,
                            "token_id": 10,
                            "token_text": "A",
                            "full_attention_mean": 0.1,
                            "full_attention_max": 0.2,
                        },
                        {
                            "position": 1,
                            "token_id": 11,
                            "token_text": "B",
                            "full_attention_mean": 0.2,
                            "full_attention_max": 0.3,
                        },
                    ],
                }
            ],
        }
        metadata = {
            "model_path": "models/Llama-3.1-8B",
            "block_size": 2,
            "sample_index": 7,
            "query_generated_index": 0,
            "query_token_id": 25,
            "query_token_text": ":",
        }
        attention = np.array([[[0.1, 0.2]]], dtype=np.float32)
        tokens = [
            {"position": 0, "source": "prompt", "token_id": 10, "token_text": "A"},
            {"position": 1, "source": "prompt", "token_id": 11, "token_text": "B"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            tool.write_outputs(output_dir, metadata, comparison, attention, tokens=tokens)

            with (output_dir / "pooling_attention_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["token_positions"], "[0, 1]")
            self.assertEqual(rows[0]["token_texts"], "[\"A\", \"B\"]")
            self.assertEqual(rows[0]["fine_attention_values"], "[0.1, 0.2]")
            self.assertEqual(float(rows[0]["fine_attention_sum"]), 0.3)
            self.assertEqual(float(rows[0]["avg_pooling_attention"]), 0.15)
            self.assertEqual(float(rows[0]["max_pooling_attention"]), 0.2)
            self.assertEqual(rows[0]["avg_equals_sum"], "False")
            self.assertEqual(rows[0]["max_equals_sum"], "False")

    def test_write_outputs_persists_true_pooling_attention_csv(self):
        tool = _load_module()
        comparison = {
            "pooling_tokens": [],
            "fine_tokens": [],
            "pooling_vs_fine_summary": [],
        }
        metadata = {
            "model_path": "models/Llama-3.1-8B",
            "block_size": 2,
            "query_generated_index": 0,
        }
        attention = np.array([[[1.0]]], dtype=np.float32)
        true_rows = [
            {
                "query_generated_index": 0,
                "token_range": "1-1",
                "dense_attention_sum": 1.0,
                "avg_pooling_attention": 1.0,
                "max_pooling_attention": 1.0,
                "layer": 0,
                "head": 0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            tool.write_outputs(
                output_dir,
                metadata,
                comparison,
                attention,
                true_pooling_rows=true_rows,
            )

            with (output_dir / "true_pooling_attention_7col_block2.csv").open(
                newline="", encoding="utf-8"
            ) as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(rows[0]["query_generated_index"], "0")
            self.assertEqual(rows[0]["token_range"], "1-1")
            self.assertEqual(float(rows[0]["dense_attention_sum"]), 1.0)


if __name__ == "__main__":
    unittest.main()
