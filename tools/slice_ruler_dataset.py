#!/usr/bin/env python
"""
从已经转换好的 RULER jsonl 数据中复制出一个固定随机子数据集。

脚本只读取源数据目录，按 RULER runner 兼容的目录结构写入新的数据根目录。
默认用于从 4k 到 64k 的 13 个 synthetic 任务中，每个任务长度抽取 20 条。
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RULER_ROOT = PROJECT_ROOT / "RULER"
DEFAULT_SOURCE_ROOT = RULER_ROOT / "benchmark_root" / "parquet_data" / "synthetic"
DEFAULT_TARGET_ROOT = RULER_ROOT / "benchmark_root" / "parquet_data_20" / "synthetic"
DEFAULT_LENGTHS = (4096, 8192, 16384, 32768, 65536)
DEFAULT_TASKS = (
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
)


def parse_int_csv(raw_value: str, option_name: str) -> List[int]:
    """解析逗号分隔的正整数列表。"""

    values: List[int] = []
    for part in raw_value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{option_name} 只能包含整数：{raw_value}") from exc
        if value <= 0:
            raise ValueError(f"{option_name} 只能包含正整数：{raw_value}")
        values.append(value)
    if not values:
        raise ValueError(f"{option_name} 至少需要一个整数")
    return values


def parse_task_csv(raw_value: str) -> List[str]:
    """解析逗号分隔的任务名列表，支持 all 表示默认 13 个任务。"""

    if raw_value.strip().lower() == "all":
        return list(DEFAULT_TASKS)

    tasks = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not tasks:
        raise ValueError("--tasks 至少需要一个任务名")

    unknown = [task for task in tasks if task not in DEFAULT_TASKS]
    if unknown:
        raise ValueError(f"未知任务：{', '.join(unknown)}")
    return tasks


def parse_lengths(raw_value: str) -> List[int]:
    """解析长度列表，支持 all 表示默认 4k 到 64k。"""

    if raw_value.strip().lower() == "all":
        return list(DEFAULT_LENGTHS)
    return parse_int_csv(raw_value, "--lengths")


def read_jsonl_lines(path: Path) -> List[str]:
    """读取 jsonl 文件的非空行，保留每行原始文本。"""

    with path.open("r", encoding="utf-8") as file_obj:
        return [line.rstrip("\n") for line in file_obj if line.strip()]


def selected_indices_from_lines(lines: Sequence[str]) -> List[Any]:
    """从已经选中的 jsonl 行中提取样本 index 字段。"""

    indices: List[Any] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            indices.append(None)
            continue
        indices.append(record.get("index"))
    return indices


def slice_jsonl_file(
    source_file: Path,
    target_file: Path,
    sample_count: int,
    seed: int,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """从单个 jsonl 文件中固定随机抽样并复制到目标文件。"""

    if sample_count <= 0:
        raise ValueError("sample_count 必须是正整数")
    if not source_file.exists():
        raise FileNotFoundError(f"找不到源文件：{source_file}")
    if target_file.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在：{target_file}")

    lines = read_jsonl_lines(source_file)
    if len(lines) < sample_count:
        raise ValueError(
            f"样本数不足：{source_file} 只有 {len(lines)} 条，无法抽取 {sample_count} 条"
        )

    selected_offsets = sorted(random.Random(seed).sample(range(len(lines)), sample_count))
    selected_lines = [lines[offset] for offset in selected_offsets]

    target_file.parent.mkdir(parents=True, exist_ok=True)
    with target_file.open("w", encoding="utf-8") as file_obj:
        for line in selected_lines:
            file_obj.write(line + "\n")

    return {
        "source_file": str(source_file),
        "target_file": str(target_file),
        "source_lines": len(lines),
        "written": len(selected_lines),
        "seed": seed,
        "selected_offsets": selected_offsets,
        "selected_line_numbers": [offset + 1 for offset in selected_offsets],
        "selected_indices": selected_indices_from_lines(selected_lines),
    }


def source_file_for(source_root: Path, length: int, task: str, subset: str) -> Path:
    """返回源数据中某个长度任务的 jsonl 文件路径。"""

    return source_root / str(length) / "data" / task / f"{subset}.jsonl"


def target_file_for(target_root: Path, length: int, task: str, subset: str) -> Path:
    """返回子数据集中某个长度任务的 jsonl 文件路径。"""

    return target_root / str(length) / "data" / task / f"{subset}.jsonl"


def write_report(report_file: Path, report: Dict[str, Any]) -> None:
    """写出子数据集划分报告。"""

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def slice_dataset(
    source_root: Path,
    target_root: Path,
    lengths: Sequence[int],
    tasks: Sequence[str],
    subset: str,
    sample_count: int,
    seed: int,
    overwrite: bool = True,
    report_file: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """按长度和任务矩阵复制出 RULER 子数据集，并返回报告内容。"""

    if sample_count <= 0:
        raise ValueError("--samples 必须是正整数")
    if not source_root.exists():
        raise FileNotFoundError(f"找不到源数据根目录：{source_root}")

    sliced: List[Dict[str, Any]] = []
    missing_items: List[Dict[str, Any]] = []
    for length in lengths:
        for task in tasks:
            source_file = source_file_for(source_root, length, task, subset)
            target_file = target_file_for(target_root, length, task, subset)
            if not source_file.exists():
                missing_items.append(
                    {
                        "length": length,
                        "task": task,
                        "source_file": str(source_file),
                    }
                )
                continue
            sliced.append(
                slice_jsonl_file(
                    source_file=source_file,
                    target_file=target_file,
                    sample_count=sample_count,
                    seed=seed,
                    overwrite=overwrite,
                )
            )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "subset": subset,
        "lengths": list(lengths),
        "tasks": list(tasks),
        "sample_count": sample_count,
        "seed": seed,
        "sliced": sliced,
        "missing": len(missing_items),
        "missing_items": missing_items,
    }
    selected_report_file = report_file or (target_root / "subset_report.json")
    write_report(selected_report_file, report)

    if strict and missing_items:
        missing_text = ", ".join(f"{item['task']}:{item['length']}" for item in missing_items)
        raise FileNotFoundError(f"部分源文件缺失：{missing_text}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="从 RULER jsonl 输入中复制出固定随机子数据集。"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="源 RULER synthetic 数据根目录。",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="子数据集写入根目录。",
    )
    parser.add_argument(
        "--lengths",
        default="all",
        help="逗号分隔长度列表；默认 all 表示 4096,8192,16384,32768,65536。",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="逗号分隔任务列表；默认 all 表示 13 个 synthetic 任务。",
    )
    parser.add_argument("--subset", default="validation", help="要复制的数据 split。")
    parser.add_argument("--samples", type=int, default=20, help="每个任务长度抽取的样本数。")
    parser.add_argument("--seed", type=int, default=0, help="固定随机抽样种子。")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="目标文件已存在时退出，而不是覆盖写入。",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="允许部分长度任务源文件缺失，仅在报告中记录。",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="划分报告路径；默认写入 target-root/subset_report.json。",
    )
    return parser


def main() -> None:
    """执行 RULER 子数据集划分。"""

    args = build_parser().parse_args()
    lengths = parse_lengths(args.lengths)
    tasks = parse_task_csv(args.tasks)
    report = slice_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        lengths=lengths,
        tasks=tasks,
        subset=args.subset,
        sample_count=args.samples,
        seed=args.seed,
        overwrite=not args.no_overwrite,
        report_file=args.report_file,
        strict=not args.allow_missing,
    )

    print(
        f"已写出子数据集：{args.target_root}，"
        f"完成 {len(report['sliced'])} 个任务文件，每个 {args.samples} 条。"
    )
    print(f"划分报告：{args.report_file or (args.target_root / 'subset_report.json')}")


if __name__ == "__main__":
    main()
