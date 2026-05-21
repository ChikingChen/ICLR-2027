#!/usr/bin/env python3
"""汇总本地 RULER 预测、评分、耗时和生成概率到一组 csv 文件。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


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

DETAIL_COLUMNS = [
    "model",
    "length",
    "task",
    "status",
    "score",
    "nulls",
    "samples",
    "pred_lines",
    "generation_logprob_sum",
    "generation_token_count",
    "generation_nll",
    "generation_ppl",
    "sample_timing_records",
    "timing_generated_token_count_total",
    "prefill_forward_ms_total",
    "decode_forward_ms_total",
    "decode_forward_ms_per_token_avg",
    "attention_profile_sample_line_no",
    "attention_profile_sample_index",
    "prefill_attention_kernel_ms",
    "decode_attention_kernel_ms_total",
    "decode_attention_kernel_ms_per_token_avg",
    "attention_kernel_event_count",
    "elapsed_seconds",
    "gpu",
    "pred_file",
    "log_file",
]

SUMMARY_COLUMNS = [
    "model",
    "completed_tasks",
    "total_tasks",
    "completion_rate",
    "avg_score",
    "avg_generation_ppl",
    "avg_generation_nll",
    "total_task_elapsed_seconds",
    "wall_time_seconds",
    "avg_elapsed_seconds",
    "total_prefill_forward_ms",
    "total_decode_forward_ms",
    "avg_decode_forward_ms_per_token",
    "attention_profiled_tasks",
    "avg_prefill_attention_kernel_ms",
    "avg_decode_attention_kernel_ms_per_token",
    "total_samples",
    "total_pred_lines",
]

SUMMARY_BY_LENGTH_COLUMNS = [
    "model",
    "length",
    "completed_tasks",
    "total_tasks",
    "completion_rate",
    "avg_score",
    "avg_generation_ppl",
    "avg_generation_nll",
    "total_task_elapsed_seconds",
    "wall_time_seconds",
    "avg_elapsed_seconds",
    "total_prefill_forward_ms",
    "total_decode_forward_ms",
    "avg_decode_forward_ms_per_token",
    "attention_profiled_tasks",
    "avg_prefill_attention_kernel_ms",
    "avg_decode_attention_kernel_ms_per_token",
    "total_samples",
    "total_pred_lines",
]

SUMMARY_BY_TASK_COLUMNS = [
    "task",
    "completed_runs",
    "avg_score",
    "best_score",
    "best_model",
    "best_length",
    "avg_generation_ppl",
    "avg_generation_nll",
    "avg_elapsed_seconds",
    "total_prefill_forward_ms",
    "total_decode_forward_ms",
    "avg_decode_forward_ms_per_token",
    "attention_profiled_tasks",
    "avg_prefill_attention_kernel_ms",
    "avg_decode_attention_kernel_ms_per_token",
    "total_samples",
]

RUN_INFO_COLUMNS = ["key", "value"]


class TimingRecord:
    """保存 runner timing jsonl 中一个任务的耗时字段。"""

    def __init__(
        self,
        elapsed_seconds: Optional[float] = None,
        gpu: Optional[int] = None,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
    ) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.gpu = gpu
        self.started_at = started_at
        self.ended_at = ended_at


def default_output_root() -> Path:
    """返回相对当前脚本稳定的默认预测根目录。"""

    return Path(__file__).resolve().parents[2] / "benchmark_root" / "local_eval"


def default_data_root() -> Path:
    """返回相对当前脚本稳定的默认输入数据根目录。"""

    return Path(__file__).resolve().parents[2] / "benchmark_root" / "parquet_data" / "synthetic"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="统一汇总 RULER 本地预测结果为 csv。")
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--benchmark", default="synthetic")
    parser.add_argument("--models", default=None, help="逗号分隔模型过滤；不传则自动发现。")
    parser.add_argument("--seq-lengths", default="all", help="逗号分隔长度过滤或 all。")
    parser.add_argument("--tasks", default="all", help="逗号分隔任务过滤或 all。")
    parser.add_argument("--timing-file", type=Path, default=None, help="runner 结构化 timing jsonl。")
    parser.add_argument("--output-file", type=Path, default=None, help="csv 主输出路径。")
    return parser


def parse_csv(value: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔字符串，保留原始顺序并去掉空项。"""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def parse_lengths(value: str, data_root: Path) -> List[int]:
    """解析长度过滤；all 时从数据根目录自动发现数字目录。"""

    if value == "all":
        return sorted(int(path.name) for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())
    lengths = []
    for part in parse_csv(value) or []:
        lengths.append(int(part))
    return lengths


def resolve_tasks(value: str, benchmark: str) -> List[str]:
    """解析任务过滤；synthetic 的 all 使用固定 13 任务顺序。"""

    if value == "all":
        if benchmark != "synthetic":
            raise ValueError("当前只内置 synthetic benchmark 的 all 任务列表")
        return list(SYNTHETIC_TASKS)
    tasks = parse_csv(value) or []
    unknown = [task for task in tasks if benchmark == "synthetic" and task not in SYNTHETIC_TASKS]
    if unknown:
        raise ValueError(f"未知 synthetic 任务: {','.join(unknown)}")
    return tasks


def discover_models(output_root: Path, benchmark: str) -> List[str]:
    """从 output_root 下自动发现包含 benchmark 目录的模型名。"""

    if not output_root.exists():
        return []
    return sorted(path.name for path in output_root.iterdir() if (path / benchmark).is_dir())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 jsonl 文件；不存在时返回空列表。"""

    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def count_jsonl_lines(path: Path) -> int:
    """统计 jsonl 非空行数；文件不存在时返回 0。"""

    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_task_configs(benchmark: str) -> Dict[str, Dict[str, Any]]:
    """加载 RULER 原生任务配置和 metric 函数。"""

    eval_dir = Path(__file__).resolve().parent
    constants_file = eval_dir / benchmark / "constants.py"
    yaml_file = eval_dir.parent / f"{benchmark}.yaml"
    if not constants_file.exists() or not yaml_file.exists():
        raise FileNotFoundError(f"找不到 {benchmark} 的 constants.py 或 yaml 配置")

    spec = importlib.util.spec_from_file_location(f"{benchmark}_constants", constants_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    customized = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    configs: Dict[str, Dict[str, Any]] = {}
    for task_name, config in customized.items():
        merged = dict(config)
        merged.update(module.TASKS[config["task"]])
        configs[task_name] = merged
    return configs


def postprocess_pred(predict_str: str) -> str:
    """对齐 evaluate.py 的预测后处理逻辑。"""

    predict_str = predict_str.strip()
    return re.sub(r"[\x00-\x1f]", "\n", predict_str).strip()


def score_predictions(rows: Sequence[Mapping[str, Any]], task_config: Mapping[str, Any]) -> Tuple[Optional[float], str]:
    """使用 RULER synthetic metric 计算分数和空预测数量。"""

    predicts = [postprocess_pred(str(row.get("pred", ""))) for row in rows]
    references = [row.get("outputs", [row.get("output", "")]) for row in rows]
    nulls = f"{sum(len(pred) == 0 for pred in predicts)}/{len(predicts)}"
    if not predicts:
        return None, nulls
    if references and references[0] and references[0][0] is not None:
        return task_config["metric_fn"](predicts, references), nulls
    return 0.0, nulls


def aggregate_generation_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """聚合预测文件中的生成 logprob 和 token 数并计算 NLL/PPL。"""

    logprob_sum = 0.0
    token_count = 0.0
    has_logprob = False
    has_token_count = False
    for row in rows:
        if row.get("generation_logprob_sum") is not None:
            logprob_sum += float(row["generation_logprob_sum"])
            has_logprob = True
        if row.get("generation_token_count") is not None:
            token_count += float(row["generation_token_count"])
            has_token_count = True

    if not has_logprob or not has_token_count or token_count <= 0:
        return {
            "generation_logprob_sum": None,
            "generation_token_count": None if not has_token_count else token_count,
            "generation_nll": None,
            "generation_ppl": None,
        }
    nll = -logprob_sum / token_count
    return {
        "generation_logprob_sum": logprob_sum,
        "generation_token_count": token_count,
        "generation_nll": nll,
        "generation_ppl": math.exp(nll),
    }


def aggregate_generation_timing(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """聚合单个任务的 generation_timing sidecar。"""

    sample_rows = [row for row in rows if row.get("record_type") == "sample_timing"]
    profile_rows = [row for row in rows if row.get("record_type") == "attention_profile"]
    prefill_total = sum_optional(row.get("prefill_forward_ms") for row in sample_rows)
    decode_total = sum_optional(row.get("decode_forward_ms_total") for row in sample_rows)
    generated_token_total = sum_optional(row.get("generated_token_count") for row in sample_rows)
    decode_per_token = None
    if decode_total is not None and generated_token_total is not None and generated_token_total > 0:
        decode_per_token = decode_total / generated_token_total

    profile = profile_rows[0] if profile_rows else {}
    return {
        "sample_timing_records": len(sample_rows),
        "timing_generated_token_count_total": generated_token_total,
        "prefill_forward_ms_total": prefill_total,
        "decode_forward_ms_total": decode_total,
        "decode_forward_ms_per_token_avg": decode_per_token,
        "attention_profile_sample_line_no": profile.get("sample_line_no"),
        "attention_profile_sample_index": profile.get("sample_index"),
        "prefill_attention_kernel_ms": _float_or_none(profile.get("prefill_attention_kernel_ms")),
        "decode_attention_kernel_ms_total": _float_or_none(profile.get("decode_attention_kernel_ms_total")),
        "decode_attention_kernel_ms_per_token_avg": _float_or_none(profile.get("decode_attention_kernel_ms_per_token_avg")),
        "attention_kernel_event_count": profile.get("attention_kernel_event_count"),
    }


def _float_or_none(value: Any) -> Optional[float]:
    """把 timing 字段转成 float；空值返回 None。"""

    if value is None or value == "":
        return None
    return float(value)


def load_timing(timing_file: Optional[Path]) -> Dict[Tuple[str, int, str], TimingRecord]:
    """读取 runner timing jsonl，按 model、length、task 建索引。"""

    if timing_file is None or not timing_file.exists():
        return {}
    timing: Dict[Tuple[str, int, str], TimingRecord] = {}
    for row in read_jsonl(timing_file):
        model = row.get("model")
        task = row.get("task")
        length = row.get("length", row.get("seq_length"))
        if model is None or task is None or length is None:
            continue
        elapsed = row.get("task_elapsed_seconds", row.get("elapsed_seconds"))
        raw_gpu = row.get("gpu")
        timing[(str(model), int(length), str(task))] = TimingRecord(
            elapsed_seconds=_float_or_none(elapsed),
            gpu=None if raw_gpu is None or raw_gpu == "" else int(raw_gpu),
            started_at=_float_or_none(row.get("started_at")),
            ended_at=_float_or_none(row.get("ended_at")),
        )
    return timing


def build_detail_rows(
    output_root: Path,
    data_root: Path,
    benchmark: str,
    models: Sequence[str],
    lengths: Sequence[int],
    tasks: Sequence[str],
    timing: Mapping[Tuple[str, int, str], TimingRecord],
) -> List[Dict[str, Any]]:
    """生成 model、length、task 粒度的明细行。"""

    task_configs = load_task_configs(benchmark)
    detail = []
    for model in models:
        for length in lengths:
            for task in tasks:
                input_file = data_root / str(length) / "data" / task / "validation.jsonl"
                pred_file = output_root / model / benchmark / str(length) / "pred" / f"{task}.jsonl"
                log_file = output_root / model / benchmark / str(length) / "logs" / f"{task}.log"
                samples = count_jsonl_lines(input_file)
                pred_rows = read_jsonl(pred_file)
                pred_lines = len(pred_rows)
                generation_timing_rows = read_jsonl(pred_file.with_suffix(".generation_timing.jsonl"))
                score = None
                nulls = ""
                status = "missing"
                if pred_file.exists():
                    status = "failed" if pred_lines < samples else "completed"
                    if task in task_configs:
                        score, nulls = score_predictions(pred_rows, task_configs[task])
                    else:
                        status = "failed"
                generation = aggregate_generation_stats(pred_rows)
                generation_timing = aggregate_generation_timing(generation_timing_rows)
                timing_record = timing.get((model, length, task), TimingRecord())
                detail.append(
                    {
                        "model": model,
                        "length": length,
                        "task": task,
                        "status": status,
                        "samples": samples,
                        "pred_lines": pred_lines,
                        "score": score,
                        "nulls": nulls,
                        "elapsed_seconds": timing_record.elapsed_seconds,
                        "gpu": timing_record.gpu,
                        "started_at": timing_record.started_at,
                        "ended_at": timing_record.ended_at,
                        "pred_file": str(pred_file),
                        "log_file": str(log_file),
                        **generation,
                        **generation_timing,
                    }
                )
    return detail


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    """计算非空数值平均值。"""

    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def sum_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    """计算非空数值总和；全部为空则返回 None。"""

    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean)


def max_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    """计算非空数值最大值；全部为空则返回 None。"""

    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return max(clean)


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """聚合一组 detail 行的通用统计字段。"""

    completed = [row for row in rows if row["status"] == "completed"]
    total_tasks = len(rows)
    started_values = [float(row["started_at"]) for row in rows if row.get("started_at") is not None]
    ended_values = [float(row["ended_at"]) for row in rows if row.get("ended_at") is not None]
    wall_time_seconds = None
    if started_values and ended_values:
        wall_time_seconds = max(ended_values) - min(started_values)
    total_decode_ms = sum_optional(row.get("decode_forward_ms_total") for row in rows)
    total_timing_tokens = sum_optional(row.get("timing_generated_token_count_total") for row in rows)
    avg_decode_ms_per_token = None
    if total_decode_ms is not None and total_timing_tokens is not None and total_timing_tokens > 0:
        avg_decode_ms_per_token = total_decode_ms / total_timing_tokens
    return {
        "completed_tasks": len(completed),
        "total_tasks": total_tasks,
        "completion_rate": len(completed) / total_tasks if total_tasks else None,
        "avg_score": average(row.get("score") for row in completed),
        "avg_generation_ppl": average(row.get("generation_ppl") for row in completed),
        "avg_generation_nll": average(row.get("generation_nll") for row in completed),
        "total_task_elapsed_seconds": sum_optional(row.get("elapsed_seconds") for row in rows),
        "wall_time_seconds": wall_time_seconds,
        "avg_elapsed_seconds": average(row.get("elapsed_seconds") for row in rows),
        "total_prefill_forward_ms": sum_optional(row.get("prefill_forward_ms_total") for row in rows),
        "total_decode_forward_ms": total_decode_ms,
        "avg_decode_forward_ms_per_token": avg_decode_ms_per_token,
        "attention_profiled_tasks": sum(1 for row in rows if row.get("attention_profile_sample_line_no") is not None),
        "avg_prefill_attention_kernel_ms": average(row.get("prefill_attention_kernel_ms") for row in rows),
        "avg_decode_attention_kernel_ms_per_token": average(row.get("decode_attention_kernel_ms_per_token_avg") for row in rows),
        "total_samples": sum(int(row.get("samples") or 0) for row in rows),
        "total_pred_lines": sum(int(row.get("pred_lines") or 0) for row in rows),
    }


def summary_by_model(detail: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """按模型聚合汇总行。"""

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail:
        groups[str(row["model"])].append(row)
    return [{"model": model, **summarize_group(rows)} for model, rows in sorted(groups.items())]


def summary_by_model_and_length(detail: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """按模型和长度聚合汇总行。"""

    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail:
        groups[(str(row["model"]), int(row["length"]))].append(row)
    return [
        {"model": model, "length": length, **summarize_group(rows)}
        for (model, length), rows in sorted(groups.items())
    ]


def summary_by_task(detail: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """按任务聚合不同模型和长度的表现。"""

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail:
        groups[str(row["task"])].append(row)

    summary = []
    for task, rows in sorted(groups.items()):
        completed = [row for row in rows if row["status"] == "completed"]
        scored = [row for row in completed if row.get("score") is not None]
        best = max(scored, key=lambda row: float(row["score"])) if scored else None
        group_summary = summarize_group(rows)
        summary.append(
            {
                "task": task,
                "completed_runs": len(completed),
                "avg_score": average(row.get("score") for row in completed),
                "best_score": None if best is None else best.get("score"),
                "best_model": "" if best is None else best.get("model"),
                "best_length": "" if best is None else best.get("length"),
                "avg_generation_ppl": average(row.get("generation_ppl") for row in completed),
                "avg_generation_nll": average(row.get("generation_nll") for row in completed),
                "avg_elapsed_seconds": average(row.get("elapsed_seconds") for row in rows),
                "total_prefill_forward_ms": group_summary["total_prefill_forward_ms"],
                "total_decode_forward_ms": group_summary["total_decode_forward_ms"],
                "avg_decode_forward_ms_per_token": group_summary["avg_decode_forward_ms_per_token"],
                "attention_profiled_tasks": group_summary["attention_profiled_tasks"],
                "avg_prefill_attention_kernel_ms": group_summary["avg_prefill_attention_kernel_ms"],
                "avg_decode_attention_kernel_ms_per_token": group_summary["avg_decode_attention_kernel_ms_per_token"],
                "total_samples": sum(int(row.get("samples") or 0) for row in rows),
            }
        )
    return summary


def build_run_info(
    output_root: Path,
    data_root: Path,
    benchmark: str,
    models: Sequence[str],
    lengths: Sequence[int],
    tasks: Sequence[str],
) -> List[Dict[str, str]]:
    """生成运行元信息 sheet。"""

    return [
        {"key": "generated_at", "value": dt.datetime.now(dt.timezone.utc).isoformat()},
        {"key": "output_root", "value": str(output_root)},
        {"key": "data_root", "value": str(data_root)},
        {"key": "benchmark", "value": benchmark},
        {"key": "models", "value": ",".join(models)},
        {"key": "lengths", "value": ",".join(str(length) for length in lengths)},
        {"key": "tasks", "value": ",".join(tasks)},
        {"key": "score_definition", "value": "复用 RULER eval/synthetic/constants.py 的任务 metric_fn。"},
        {"key": "ppl_definition", "value": "generation_nll=-sum(generation_logprob_sum)/sum(generation_token_count); generation_ppl=exp(nll)。"},
        {"key": "time_definition", "value": "total_task_elapsed_seconds 为任务 elapsed_seconds 求和，wall_time_seconds 为同组 max(ended_at)-min(started_at)。"},
        {"key": "notes", "value": "缺少预测文件记为 missing；预测行数少于输入样本数记为 failed。"},
    ]


def collect_results(
    output_root: Path,
    data_root: Path,
    benchmark: str,
    models: Sequence[str],
    lengths: Sequence[int],
    tasks: Sequence[str],
    timing_file: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """收集明细和四类汇总数据，返回按 sheet 名分组的行字典。"""

    timing = load_timing(timing_file)
    detail = build_detail_rows(output_root, data_root, benchmark, models, lengths, tasks, timing)
    return {
        "detail": detail,
        "summary_by_model": summary_by_model(detail),
        "summary_by_model_and_length": summary_by_model_and_length(detail),
        "summary_by_task": summary_by_task(detail),
        "run_info": build_run_info(output_root, data_root, benchmark, models, lengths, tasks),
    }


def rows_to_matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> List[List[Any]]:
    """按固定列顺序把字典行转换成二维表。"""

    return [list(columns)] + [[row.get(column, "") for column in columns] for row in rows]


def csv_file_for_sheet(output_file: Path, sheet_name: str) -> Path:
    """返回某张汇总表对应的 csv 文件路径。"""

    if sheet_name == "detail":
        return output_file
    return output_file.with_name(f"{output_file.stem}_{sheet_name}.csv")


def validate_csv_output_file(output_file: Path) -> None:
    """确认输出路径使用 csv 后缀，避免误生成 xlsx 文件。"""

    if output_file.suffix.lower() != ".csv":
        raise ValueError(f"--output-file 必须使用 .csv 后缀：{output_file}")


def write_csv(output_file: Path, workbook: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Path]:
    """用标准库把五张汇总表写成一组 csv 文件，返回实际写出的路径。"""

    sheet_columns = {
        "detail": DETAIL_COLUMNS,
        "summary_by_model": SUMMARY_COLUMNS,
        "summary_by_model_and_length": SUMMARY_BY_LENGTH_COLUMNS,
        "summary_by_task": SUMMARY_BY_TASK_COLUMNS,
        "run_info": RUN_INFO_COLUMNS,
    }
    validate_csv_output_file(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written_files: List[Path] = []
    for sheet_name, columns in sheet_columns.items():
        csv_file = csv_file_for_sheet(output_file, sheet_name)
        with csv_file.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerows(rows_to_matrix(workbook[sheet_name], columns))
        written_files.append(csv_file)
    return written_files


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""

    args = build_parser().parse_args(argv)
    output_file = args.output_file or (args.output_root / "ruler_results.csv")
    validate_csv_output_file(output_file)
    models = parse_csv(args.models) or discover_models(args.output_root, args.benchmark)
    lengths = parse_lengths(args.seq_lengths, args.data_root)
    tasks = resolve_tasks(args.tasks, args.benchmark)
    workbook = collect_results(
        output_root=args.output_root,
        data_root=args.data_root,
        benchmark=args.benchmark,
        models=models,
        lengths=lengths,
        tasks=tasks,
        timing_file=args.timing_file,
    )
    written_files = write_csv(output_file, workbook)
    print(f"Saved RULER results to {', '.join(str(path) for path in written_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
