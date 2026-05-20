#!/usr/bin/env python3
"""统计 RULER 转换后 jsonl 输入中各任务的样本数。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "RULER" / "benchmark_root" / "parquet_data" / "synthetic"
DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536]
SYNTHETIC_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def parse_length_value(value: str) -> int:
    """把 `4k`、`64k` 或 `4096` 这类长度写法转换为 token 数。"""

    normalized = value.strip()
    if not normalized:
        raise ValueError("长度不能为空")

    suffix = normalized[-1].lower()
    number_text = normalized[:-1]
    if suffix == "k":
        if not number_text.isdigit():
            raise ValueError(f"不支持的长度写法：{value}")
        return int(number_text) * 1024
    if suffix == "m":
        if not number_text.isdigit():
            raise ValueError(f"不支持的长度写法：{value}")
        return int(number_text) * 1024 * 1024
    if normalized.isdigit():
        length = int(normalized)
        if length <= 0:
            raise ValueError(f"长度必须是正整数：{value}")
        return length
    raise ValueError(f"不支持的长度写法：{value}")


def parse_lengths(raw_value: str) -> List[int]:
    """解析逗号分隔长度列表，并保留用户给定顺序。"""

    lengths = [parse_length_value(part) for part in raw_value.split(",") if part.strip()]
    if not lengths:
        raise ValueError("--lengths 至少需要一个长度")
    return lengths


def parse_tasks(raw_value: str) -> List[str]:
    """解析任务列表；`all` 表示使用全部 synthetic 任务。"""

    if raw_value.strip().lower() == "all":
        return list(SYNTHETIC_TASKS)
    tasks = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not tasks:
        raise ValueError("--tasks 至少需要一个任务名")
    return tasks


def length_label(length: int) -> str:
    """把 token 数转换为适合表头展示的短标签。"""

    if length % (1024 * 1024) == 0:
        return f"{length // (1024 * 1024)}M"
    if length % 1024 == 0:
        return f"{length // 1024}k"
    return str(length)


def count_jsonl_records(path: Path) -> Optional[int]:
    """统计 jsonl 文件非空行数；文件缺失时返回 None。"""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for line in file_obj if line.strip())


def sample_file(data_root: Path, length: int, task: str, subset: str) -> Path:
    """返回某个长度和任务对应的 RULER 输入 jsonl 路径。"""

    return data_root / str(length) / "data" / task / f"{subset}.jsonl"


def collect_counts(
    data_root: Path,
    lengths: Sequence[int],
    tasks: Sequence[str],
    subset: str,
) -> List[Dict[str, object]]:
    """收集每个任务在各长度上的样本数。"""

    rows: List[Dict[str, object]] = []
    for task in tasks:
        counts: Dict[int, Optional[int]] = {}
        total = 0
        for length in lengths:
            count = count_jsonl_records(sample_file(data_root, length, task, subset))
            counts[length] = count
            if count is not None:
                total += count
        rows.append({"task": task, "counts": counts, "total": total})
    return rows


def _cell_text(value: Optional[int]) -> str:
    """把内部计数值转换为展示文本。"""

    if value is None:
        return "MISSING"
    return str(value)


def format_table(rows: Sequence[Dict[str, object]], lengths: Sequence[int]) -> str:
    """把统计结果格式化为终端可读表格。"""

    headers = ["task", *[length_label(length) for length in lengths], "total"]
    table_rows: List[List[str]] = []
    for row in rows:
        counts = row["counts"]
        assert isinstance(counts, dict)
        table_rows.append(
            [
                str(row["task"]),
                *[_cell_text(counts[length]) for length in lengths],
                str(row["total"]),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(table_row[index]) for table_row in table_rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for table_row in table_rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(table_row))
        )
    return "\n".join(lines)


def format_csv(rows: Sequence[Dict[str, object]], lengths: Sequence[int]) -> str:
    """把统计结果格式化为 CSV 文本。"""

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["task", *[length_label(length) for length in lengths], "total"])
    for row in rows:
        counts = row["counts"]
        assert isinstance(counts, dict)
        writer.writerow(
            [
                row["task"],
                *[_cell_text(counts[length]) for length in lengths],
                row["total"],
            ]
        )
    return output.getvalue().rstrip("\n")


def format_json(rows: Sequence[Dict[str, object]], lengths: Sequence[int]) -> str:
    """把统计结果格式化为 JSON 文本，长度键使用短标签。"""

    payload = []
    for row in rows:
        counts = row["counts"]
        assert isinstance(counts, dict)
        payload.append(
            {
                "task": row["task"],
                "counts": {length_label(length): counts[length] for length in lengths},
                "total": row["total"],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="统计 RULER 4k 到 64k 输入样本数。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
        help="逗号分隔长度列表，支持 4k、8192、64k 这类写法。",
    )
    parser.add_argument("--tasks", default="all", help="逗号分隔任务列表，或 all。")
    parser.add_argument("--subset", default="validation", help="要统计的数据 split。")
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="输出格式。",
    )
    return parser


def render(rows: Sequence[Dict[str, object]], lengths: Sequence[int], output_format: str) -> str:
    """按用户指定格式渲染统计结果。"""

    if output_format == "table":
        return format_table(rows, lengths)
    if output_format == "csv":
        return format_csv(rows, lengths)
    if output_format == "json":
        return format_json(rows, lengths)
    raise ValueError(f"不支持的输出格式：{output_format}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        lengths = parse_lengths(args.lengths)
        tasks = parse_tasks(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))

    rows = collect_counts(
        data_root=args.data_root,
        lengths=lengths,
        tasks=tasks,
        subset=args.subset,
    )
    print(render(rows, lengths, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
