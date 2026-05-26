#!/usr/bin/env python
"""统计完整 attention dump 中 BOS token 的 attention sink 强度。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


DEFAULT_BOS_TOKEN_TEXT = "<|begin_of_text|>"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_token_position(dump_dir: Path, token_text: str) -> int:
    for row in read_jsonl(dump_dir / "prompt_tokens.jsonl"):
        if row.get("token_text") == token_text:
            return int(row["position"])
    raise ValueError(
        f"Could not find token_text '{token_text}' in prompt_tokens.jsonl; "
        "use --bos-position to specify it explicitly."
    )


def load_generated_records(dump_dir: Path) -> List[Dict[str, Any]]:
    records = read_jsonl(dump_dir / "generated_tokens.jsonl")
    if not records:
        raise ValueError("No generated token records found in generated_tokens.jsonl.")
    return sorted(records, key=lambda row: int(row["generated_index"]))


def summarize_bos_attention(
    dump_dir: Path,
    bos_token_text: str = DEFAULT_BOS_TOKEN_TEXT,
    bos_position: int | None = None,
) -> Dict[str, Any]:
    dump_dir = Path(dump_dir)
    metadata = read_json(dump_dir / "metadata.json")
    target_position = find_token_position(dump_dir, bos_token_text) if bos_position is None else bos_position
    generated_records = load_generated_records(dump_dir)

    total_units = 0
    bos_top1_count = 0
    bos_attention_sum = 0.0
    bos_attention_min: float | None = None
    bos_attention_max: float | None = None
    num_layers: int | None = None
    num_heads: int | None = None

    for record in generated_records:
        attention_file = record.get("attention_file")
        if not attention_file:
            raise ValueError(f"generated_index {record.get('generated_index')} is missing attention_file.")
        attention = np.load(dump_dir / str(attention_file), mmap_mode="r")
        if attention.ndim != 3:
            raise ValueError(f"attention file {attention_file} must be 3D, got shape {attention.shape}.")
        if target_position < 0 or target_position >= attention.shape[2]:
            raise IndexError(
                f"BOS position {target_position} is outside key length {attention.shape[2]} "
                f"for attention file {attention_file}."
            )

        current_layers, current_heads, _ = attention.shape
        if num_layers is None:
            num_layers = int(current_layers)
            num_heads = int(current_heads)
        elif current_layers != num_layers or current_heads != num_heads:
            raise ValueError(
                f"attention file {attention_file} has layer/head shape "
                f"{current_layers}x{current_heads}, expected {num_layers}x{num_heads}."
            )

        bos_values = attention[:, :, target_position].astype(np.float64, copy=False)
        max_values = attention.max(axis=2)
        total_units += int(bos_values.size)
        bos_top1_count += int((bos_values == max_values).sum())
        bos_attention_sum += float(bos_values.sum(dtype=np.float64))
        current_min = float(bos_values.min())
        current_max = float(bos_values.max())
        bos_attention_min = current_min if bos_attention_min is None else min(bos_attention_min, current_min)
        bos_attention_max = current_max if bos_attention_max is None else max(bos_attention_max, current_max)

    bos_attention_mean = bos_attention_sum / total_units
    return {
        "dump_dir": str(dump_dir),
        "sample_index": metadata.get("sample_index"),
        "bos_token_text": bos_token_text,
        "bos_position": int(target_position),
        "generated_token_count": len(generated_records),
        "num_layers": int(num_layers or 0),
        "num_heads": int(num_heads or 0),
        "total_units": int(total_units),
        "bos_top1_count": int(bos_top1_count),
        "bos_top1_ratio": float(bos_top1_count / total_units),
        "bos_attention_mean": float(bos_attention_mean),
        "bos_attention_min": float(bos_attention_min if bos_attention_min is not None else 0.0),
        "bos_attention_max": float(bos_attention_max if bos_attention_max is not None else 0.0),
    }


def format_summary(summary: Dict[str, Any]) -> str:
    lines = [
        f"dump_dir: {summary['dump_dir']}",
        f"sample_index: {summary['sample_index']}",
        f"bos_token_text: {summary['bos_token_text']}",
        f"bos_position: {summary['bos_position']}",
        f"generated_token_count: {summary['generated_token_count']}",
        f"num_layers: {summary['num_layers']}",
        f"num_heads: {summary['num_heads']}",
        f"total_units: {summary['total_units']}",
        f"bos_top1_count: {summary['bos_top1_count']}",
        f"bos_top1_ratio: {summary['bos_top1_ratio']:.8f}",
        f"bos_attention_mean: {summary['bos_attention_mean']:.8f}",
        f"bos_attention_min: {summary['bos_attention_min']:.8f}",
        f"bos_attention_max: {summary['bos_attention_max']:.8f}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计完整 attention dump 中 <|begin_of_text|> token 的 top-1 次数和平均 attention。"
    )
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--bos-token-text", default=DEFAULT_BOS_TOKEN_TEXT)
    parser.add_argument("--bos-position", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_bos_attention(
        dump_dir=args.dump_dir,
        bos_token_text=args.bos_token_text,
        bos_position=args.bos_position,
    )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
