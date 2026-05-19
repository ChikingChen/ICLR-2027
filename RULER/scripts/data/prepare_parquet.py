"""
将下载好的 RULER parquet 数据转换为原版评测流程可读取的 jsonl。

该脚本只做离线格式转换，不重新生成 prompt，也不重新套用模型模板。
转换后的目录结构与 `pred/call_api.py` 的读取约定保持一致。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_PATH = Path(__file__).resolve()
RULER_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = RULER_ROOT.parent
DEFAULT_PARQUET_DATA_DIR = PROJECT_ROOT / "benchmark" / "RULER-llama3-1M"
DEFAULT_SAVE_DIR = RULER_ROOT / "benchmark_root" / "parquet_data" / "synthetic"
REQUIRED_FIELDS = ("index", "input", "answers", "length")


def parse_length_value(value: str) -> int:
    """把 `4k`、`1M` 或 `4096` 这类长度写法转换成 token 数。"""

    normalized = value.strip()
    if not normalized:
        raise ValueError("长度不能为空")

    suffix = normalized[-1].lower()
    number_text = normalized[:-1]
    if suffix == "k":
        if not number_text.isdigit():
            raise ValueError(f"不支持的长度后缀：{value}")
        return int(number_text) * 1024
    if suffix == "m":
        if not number_text.isdigit():
            raise ValueError(f"不支持的长度后缀：{value}")
        return int(number_text) * 1024 * 1024
    if normalized.isdigit():
        length = int(normalized)
        if length <= 0:
            raise ValueError(f"长度必须是正整数：{value}")
        return length
    raise ValueError(f"不支持的长度写法：{value}")


def canonical_length_suffix(length: int) -> str:
    """把 token 数转换成本地数据目录使用的长度后缀。"""

    if length <= 0:
        raise ValueError(f"长度必须是正整数：{length}")
    if length % (1024 * 1024) == 0:
        return f"{length // (1024 * 1024)}M"
    if length % 1024 == 0:
        return f"{length // 1024}k"
    return str(length)


def parse_length_filter(raw: Optional[str]) -> Optional[Set[int]]:
    """解析 `--lengths` 参数；未传时表示不过滤长度。"""

    if raw is None or raw.strip() == "":
        return None
    return {parse_length_value(part) for part in raw.split(",") if part.strip()}


def parse_task_filter(raw: Optional[str]) -> Optional[Set[str]]:
    """解析 `--tasks` 参数；未传时表示不过滤任务。"""

    if raw is None or raw.strip() == "":
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def parse_dataset_dir_name(name: str) -> Tuple[str, str, int]:
    """从 `<task>_<length>` 目录名中解析任务名和长度。"""

    if "_" not in name:
        raise ValueError(f"目录名不符合 <task>_<length> 格式：{name}")
    task, length_suffix = name.rsplit("_", 1)
    if not task:
        raise ValueError(f"目录名缺少任务名：{name}")
    length = parse_length_value(length_suffix)
    return task, canonical_length_suffix(length), length


def read_parquet_records(parquet_files: Sequence[Path]) -> List[Dict[str, Any]]:
    """读取一个任务长度目录下的所有 parquet 分片。"""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境缺少 pyarrow，请使用 dl-a800 环境运行。") from exc

    records: List[Dict[str, Any]] = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        records.extend(table.to_pylist())
    return records


def validate_record(record: Dict[str, Any], task: str, length_suffix: str) -> None:
    """检查 parquet 行是否包含转换所需字段。"""

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        index = record.get("index", "<unknown>")
        raise ValueError(
            f"{task}_{length_suffix} 的样本 {index} 缺少必要字段：{', '.join(missing)}"
        )
    if not isinstance(record["answers"], list):
        raise ValueError(
            f"{task}_{length_suffix} 的样本 {record['index']} 字段 answers 不是 list"
        )


def normalize_record(record: Dict[str, Any], task: str, length_suffix: str) -> Dict[str, Any]:
    """把 parquet 行转换成 `call_api.py` 需要的 jsonl 行。"""

    validate_record(record, task, length_suffix)
    return {
        "index": record["index"],
        "input": record["input"],
        "outputs": record["answers"],
        "others": {},
        "truncation": -1,
        "length": record["length"],
    }


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    output_file: Path,
    task: str,
    length_suffix: str,
    max_samples: Optional[int] = None,
) -> int:
    """按 index 排序并写出 jsonl，返回实际写入样本数。"""

    normalized_records = [
        normalize_record(record, task=task, length_suffix=length_suffix) for record in records
    ]
    normalized_records.sort(key=lambda item: item["index"])
    if max_samples is not None:
        normalized_records = normalized_records[:max_samples]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_obj:
        for record in normalized_records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(normalized_records)


def discover_dataset_dirs(
    parquet_data_dir: Path,
    task_filter: Optional[Set[str]],
    length_filter: Optional[Set[int]],
) -> Tuple[List[Tuple[Path, str, str, int]], List[Dict[str, Any]]]:
    """发现需要转换的 parquet 任务目录。"""

    if not parquet_data_dir.exists():
        raise FileNotFoundError(f"找不到 parquet 数据目录：{parquet_data_dir}")

    selected: List[Tuple[Path, str, str, int]] = []
    ignored: List[Dict[str, Any]] = []
    for child in sorted(parquet_data_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            task, length_suffix, length = parse_dataset_dir_name(child.name)
        except ValueError as exc:
            ignored.append({"path": str(child), "reason": str(exc)})
            continue
        if task_filter is not None and task not in task_filter:
            continue
        if length_filter is not None and length not in length_filter:
            continue
        selected.append((child, task, length_suffix, length))
    return selected, ignored


def convert_dataset_dir(
    dataset_dir: Path,
    save_dir: Path,
    task: str,
    length_suffix: str,
    length: int,
    subset: str,
    max_samples: Optional[int],
    skip_existing: bool,
) -> Dict[str, Any]:
    """转换单个 `<task>_<length>` parquet 目录。"""

    output_file = save_dir / str(length) / "data" / task / f"{subset}.jsonl"
    if skip_existing and output_file.exists():
        return {
            "task": task,
            "length_suffix": length_suffix,
            "length": length,
            "status": "skipped",
            "output_file": str(output_file),
            "written": 0,
        }

    parquet_files = sorted(dataset_dir.glob(f"{subset}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"找不到 parquet 分片：{dataset_dir}/{subset}-*.parquet")

    records = read_parquet_records(parquet_files)
    written = write_jsonl(
        records,
        output_file,
        task=task,
        length_suffix=length_suffix,
        max_samples=max_samples,
    )
    return {
        "task": task,
        "length_suffix": length_suffix,
        "length": length,
        "status": "converted",
        "parquet_files": [str(path) for path in parquet_files],
        "output_file": str(output_file),
        "written": written,
    }


def write_report(report_file: Path, report: Dict[str, Any]) -> None:
    """写出转换汇总报告。"""

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="将本地 RULER parquet 数据一次性转换为原版评测流程使用的 jsonl。"
    )
    parser.add_argument(
        "--parquet_data_dir",
        type=Path,
        default=DEFAULT_PARQUET_DATA_DIR,
        help="RULER parquet benchmark 根目录。",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="转换后 jsonl 的保存根目录。",
    )
    parser.add_argument("--subset", type=str, default="validation", help="要转换的数据 split。")
    parser.add_argument(
        "--tasks",
        type=str,
        help="逗号分隔的任务过滤列表，例如 niah_single_1,qa_2；不传则转换全部任务。",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        help="逗号分隔的长度过滤列表，例如 4k,128k,1M 或 4096,131072；不传则转换全部长度。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="每个任务长度最多写出多少条样本；不传则写出全部样本。",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果目标 jsonl 已存在，则跳过而不是覆盖。",
    )
    parser.add_argument(
        "--report_file",
        type=Path,
        help="转换报告路径；默认写入 save_dir/conversion_report.json。",
    )
    return parser


def main() -> None:
    """执行 parquet 到 jsonl 的批量转换。"""

    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples 必须是正整数")

    task_filter = parse_task_filter(args.tasks)
    length_filter = parse_length_filter(args.lengths)
    report_file = args.report_file or (args.save_dir / "conversion_report.json")

    dataset_dirs, ignored = discover_dataset_dirs(
        args.parquet_data_dir,
        task_filter=task_filter,
        length_filter=length_filter,
    )

    converted: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for dataset_dir, task, length_suffix, length in dataset_dirs:
        try:
            result = convert_dataset_dir(
                dataset_dir=dataset_dir,
                save_dir=args.save_dir,
                task=task,
                length_suffix=length_suffix,
                length=length,
                subset=args.subset,
                max_samples=args.max_samples,
                skip_existing=args.skip_existing,
            )
            converted.append(result)
            print(
                f"{result['status']}: {task}_{length_suffix} -> {result['output_file']} "
                f"({result['written']} 条)"
            )
        except Exception as exc:
            failed.append(
                {
                    "task": task,
                    "length_suffix": length_suffix,
                    "length": length,
                    "path": str(dataset_dir),
                    "error": str(exc),
                }
            )

    report = {
        "parquet_data_dir": str(args.parquet_data_dir),
        "save_dir": str(args.save_dir),
        "subset": args.subset,
        "converted": converted,
        "failed": failed,
        "ignored": ignored,
    }
    write_report(report_file, report)
    print(f"转换报告已写入：{report_file}")

    if failed:
        failed_items = ", ".join(f"{item['task']}_{item['length_suffix']}" for item in failed)
        raise RuntimeError(f"部分 parquet 数据转换失败：{failed_items}")


if __name__ == "__main__":
    main()
