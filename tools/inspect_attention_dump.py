#!/usr/bin/env python
"""查看完整 attention dump 中某个 token/layer/head 的分布。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def read_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 jsonl 文件。"""

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_position_tokens(dump_dir: Path) -> Dict[int, Dict[str, Any]]:
    """加载 prompt 和 generated token 的 position 映射。"""

    tokens: Dict[int, Dict[str, Any]] = {}
    for file_name in ("prompt_tokens.jsonl", "generated_tokens.jsonl"):
        for row in read_jsonl(dump_dir / file_name):
            tokens[int(row["position"])] = row
    return tokens


def load_generated_record(dump_dir: Path, generated_token: int) -> Dict[str, Any]:
    """读取指定生成 token 的元数据。"""

    for row in read_jsonl(dump_dir / "generated_tokens.jsonl"):
        if int(row["generated_index"]) == generated_token:
            return row
    raise IndexError(f"找不到 generated token：{generated_token}")


def load_attention_vector(dump_dir: Path, generated_token: int, layer: int, head: int) -> np.ndarray:
    """加载指定 token/layer/head 的 attention 向量。"""

    generated_record = load_generated_record(dump_dir, generated_token)
    attention = np.load(dump_dir / generated_record["attention_file"])
    if attention.ndim != 3:
        raise ValueError(f"attention 文件必须是三维数组，实际为 {attention.shape}")
    if layer < 0 or layer >= attention.shape[0]:
        raise IndexError(f"layer 超出范围：{layer}，可用范围 0-{attention.shape[0] - 1}")
    if head < 0 or head >= attention.shape[1]:
        raise IndexError(f"head 超出范围：{head}，可用范围 0-{attention.shape[1] - 1}")
    return attention[layer, head, :]


def build_attention_rows(dump_dir: Path, generated_token: int, layer: int, head: int) -> List[Dict[str, Any]]:
    """构造完整 attention 表格行。"""

    vector = load_attention_vector(dump_dir, generated_token, layer, head)
    tokens = load_position_tokens(dump_dir)
    rows: List[Dict[str, Any]] = []
    for position, weight in enumerate(vector.tolist()):
        token = tokens.get(
            position,
            {
                "position": position,
                "source": "unknown",
                "token_id": None,
                "token_text": "<missing>",
            },
        )
        rows.append(
            {
                "position": position,
                "source": token.get("source", "unknown"),
                "token_id": token.get("token_id"),
                "token_text": token.get("token_text", ""),
                "attention": float(weight),
            }
        )
    return rows


def sort_attention_rows(rows: List[Dict[str, Any]], sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    """按指定列返回排序后的 attention 行，不修改原始行列表。"""

    if sort_by == "position":
        key = lambda row: int(row["position"])
    elif sort_by == "attention":
        key = lambda row: float(row["attention"])
    else:
        raise ValueError(f"不支持的排序字段：{sort_by}")
    return sorted(rows, key=key, reverse=descending)


def format_table(rows: List[Dict[str, Any]]) -> str:
    """把 attention 行格式化成固定列宽表格。"""

    headers = ["position", "source", "token_id", "token_text", "attention"]
    widths = {
        "position": max(len(headers[0]), *(len(str(row["position"])) for row in rows)),
        "source": max(len(headers[1]), *(len(str(row["source"])) for row in rows)),
        "token_id": max(len(headers[2]), *(len(str(row["token_id"])) for row in rows)),
        "token_text": max(len(headers[3]), *(min(len(str(row["token_text"])), 60) for row in rows)),
        "attention": len("attention"),
    }
    lines = [
        " | ".join(
            [
                headers[0].ljust(widths["position"]),
                headers[1].ljust(widths["source"]),
                headers[2].ljust(widths["token_id"]),
                headers[3].ljust(widths["token_text"]),
                headers[4].rjust(widths["attention"]),
            ]
        )
    ]
    lines.append(
        "-+-".join(
            [
                "-" * widths["position"],
                "-" * widths["source"],
                "-" * widths["token_id"],
                "-" * widths["token_text"],
                "-" * widths["attention"],
            ]
        )
    )
    for row in rows:
        token_text = str(row["token_text"])
        if len(token_text) > 60:
            token_text = token_text[:57] + "..."
        lines.append(
            " | ".join(
                [
                    str(row["position"]).ljust(widths["position"]),
                    str(row["source"]).ljust(widths["source"]),
                    str(row["token_id"]).ljust(widths["token_id"]),
                    token_text.ljust(widths["token_text"]),
                    f"{row['attention']:.6f}".rjust(widths["attention"]),
                ]
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="查看完整 attention dump 中某个 token/layer/head 的分布。")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--generated-token", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument(
        "--sort-by",
        choices=("position", "attention"),
        default="position",
        help="表格排序字段，默认按 token 位置排序。",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="按指定字段降序输出；配合 --sort-by attention 可让注意力从大到小显示。",
    )
    return parser


def main() -> int:
    """命令行入口。"""

    args = build_parser().parse_args()
    metadata = read_json(args.dump_dir / "metadata.json")
    rows = build_attention_rows(
        dump_dir=args.dump_dir,
        generated_token=args.generated_token,
        layer=args.layer,
        head=args.head,
    )
    total = sum(row["attention"] for row in rows)
    sorted_rows = sort_attention_rows(rows, sort_by=args.sort_by, descending=args.descending)
    print(f"dump_dir: {args.dump_dir}")
    print(f"sample_index: {metadata.get('sample_index')}")
    print(f"generated_token: {args.generated_token}")
    print(f"layer: {args.layer}")
    print(f"head: {args.head}")
    print(f"sort_by: {args.sort_by}")
    print(f"descending: {args.descending}")
    print(f"attention_sum: {total:.8f}")
    print()
    print(format_table(sorted_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
