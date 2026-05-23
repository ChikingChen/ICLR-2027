import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "compare_pooling_attention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_pooling_attention", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoolingAttentionCompareTest(unittest.TestCase):
    """验证 pooling token 和细粒度 token attention 的对照格式。"""

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


if __name__ == "__main__":
    unittest.main()
