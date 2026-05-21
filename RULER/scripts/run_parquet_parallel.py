#!/usr/bin/env python
"""
并行调度已经转换好的 parquet-jsonl RULER 任务。

这个脚本不直接加载模型，而是为每个模型、长度和任务组合启动一个
`pred/call_api.py` 子进程。每个子进程只暴露一张 GPU，从而避免
Hugging Face `device_map="auto"` 默认把同一个模型切到所有可见 GPU 上。
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, NamedTuple, Optional, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
RULER_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_ROOT = RULER_ROOT / "benchmark_root" / "parquet_data" / "synthetic"
DEFAULT_OUTPUT_ROOT = RULER_ROOT / "benchmark_root" / "local_eval"
DEFAULT_REPORT_NAME = "ruler_results.csv"
DEFAULT_TIMING_NAME = "ruler_timing.jsonl"
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


class ModelSpec(NamedTuple):
    """表示一个命令行传入的模型别名和权重路径。"""

    name: str
    path: Path


class Job(NamedTuple):
    """表示一个待执行的模型、长度和任务组合。"""

    model: ModelSpec
    seq_length: int
    task: str


class RunnerConfig(NamedTuple):
    """保存 runner 构造 `call_api.py` 命令所需的稳定配置。"""

    scripts_dir: Path
    data_root: Path
    output_root: Path
    benchmark: str
    server_type: str
    subset: str
    python: str
    model_python: Dict[str, str]
    temperature: float
    top_k: int
    top_p: float
    random_seed: int
    stop_words: str
    batch_size: int
    poll_interval: float
    log_batch_progress: bool
    log_attention_scores: bool
    attention_top_k: int
    log_generation_ppl: bool
    overwrite_existing: bool
    timing_file: Path
    auto_evaluate: bool
    report_file: Path


class RunningJob(NamedTuple):
    """记录一个已经启动但尚未结束的子进程。"""

    job: Job
    gpu: int
    process: subprocess.Popen
    log_handle: Any
    output_thread: threading.Thread
    started_at: float
    total_samples: int


def parse_model_specs(raw_models: Sequence[str]) -> List[ModelSpec]:
    """解析重复传入的 `--model NAME=PATH` 参数。"""

    models: List[ModelSpec] = []
    for raw_model in raw_models:
        if "=" not in raw_model:
            raise ValueError(f"模型参数必须使用 NAME=PATH 格式：{raw_model}")
        name, raw_path = raw_model.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"模型参数必须同时包含名称和路径：{raw_model}")
        models.append(ModelSpec(name=name, path=Path(raw_path)))
    if not models:
        raise ValueError("至少需要传入一个 --model NAME=PATH")
    return models


def parse_model_python_specs(
    raw_model_pythons: Sequence[str],
    models: Sequence[ModelSpec],
) -> Dict[str, str]:
    """解析重复传入的 `--model-python NAME=PYTHON` 参数并校验模型别名。"""

    model_names = {model.name for model in models}
    mapping: Dict[str, str] = {}
    for raw_model_python in raw_model_pythons:
        if "=" not in raw_model_python:
            raise ValueError(f"模型 Python 参数必须使用 NAME=PYTHON 格式：{raw_model_python}")
        name, python = raw_model_python.split("=", 1)
        name = name.strip()
        python = python.strip()
        if not name or not python:
            raise ValueError(f"模型 Python 参数必须同时包含名称和 Python：{raw_model_python}")
        if name not in model_names:
            raise ValueError(f"--model-python 引用了未知模型别名：{name}")
        mapping[name] = python
    return mapping


def parse_int_csv(raw_value: str, option_name: str, allow_zero: bool = True) -> List[int]:
    """解析逗号分隔的整数列表，可按参数决定是否允许 0。"""

    values: List[int] = []
    for part in raw_value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{option_name} 只能包含整数：{raw_value}") from exc
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{option_name} 包含不合法的整数：{raw_value}")
        values.append(value)
    if not values:
        raise ValueError(f"{option_name} 至少需要一个整数")
    return values


def resolve_seq_lengths(raw_value: str, data_root: Path) -> List[int]:
    """解析 `--seq-lengths`，支持 `all` 或逗号分隔的数字长度。"""

    if raw_value.strip().lower() != "all":
        return parse_int_csv(raw_value, "--seq-lengths", allow_zero=False)

    if not data_root.exists():
        raise FileNotFoundError(f"找不到数据根目录：{data_root}")

    lengths: List[int] = []
    for child in data_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            lengths.append(int(child.name))
    if not lengths:
        raise ValueError(f"数据根目录下没有可用的数字长度目录：{data_root}")
    return sorted(lengths)


def resolve_tasks(raw_value: str) -> List[str]:
    """解析 `--tasks`，支持 `all` 或逗号分隔的任务名。"""

    if raw_value.strip().lower() == "all":
        return list(DEFAULT_TASKS)

    tasks = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not tasks:
        raise ValueError("--tasks 至少需要一个任务名")

    unknown = [task for task in tasks if task not in DEFAULT_TASKS]
    if unknown:
        raise ValueError(f"未知任务：{', '.join(unknown)}")
    return tasks


def build_jobs(
    models: Sequence[ModelSpec],
    seq_lengths: Sequence[int],
    tasks: Sequence[str],
) -> List[Job]:
    """按长度、任务、模型展开任务矩阵，让不同模型尽早交错运行。"""

    jobs: List[Job] = []
    for seq_length in seq_lengths:
        for task in tasks:
            for model in models:
                jobs.append(Job(model=model, seq_length=seq_length, task=task))
    return jobs


def data_dir_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务传给 `call_api.py --data_dir` 的目录。"""

    return config.data_root / str(job.seq_length) / "data"


def save_dir_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务预测 jsonl 的保存目录。"""

    return config.output_root / job.model.name / config.benchmark / str(job.seq_length) / "pred"


def log_dir_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务日志文件的保存目录。"""

    return config.output_root / job.model.name / config.benchmark / str(job.seq_length) / "logs"


def prediction_file_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务的预测 jsonl 文件路径。"""

    return save_dir_for(job, config) / f"{job.task}.jsonl"


def log_file_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务的子进程日志文件路径。"""

    return log_dir_for(job, config) / f"{job.task}.log"


def overwrite_files_for(job: Job, config: RunnerConfig) -> List[Path]:
    """返回覆盖重跑时需要删除的旧输出文件。"""

    pred_file = prediction_file_for(job, config)
    return [
        pred_file,
        pred_file.with_suffix(".attention.jsonl"),
        pred_file.with_suffix(".attention.md"),
        log_file_for(job, config),
    ]


def delete_existing_outputs(job: Job, config: RunnerConfig) -> List[Path]:
    """删除单个任务的旧预测、注意力摘要和日志文件，返回实际删除的路径。"""

    removed: List[Path] = []
    for path in overwrite_files_for(job, config):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def task_file_for(job: Job, config: RunnerConfig) -> Path:
    """返回某个任务的输入 jsonl 文件路径。"""

    return data_dir_for(job, config) / job.task / f"{config.subset}.jsonl"


def count_jsonl_lines(path: Path) -> int:
    """统计 jsonl 文件行数；文件不存在时返回 0。"""

    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for _ in file_obj)


def build_call_api_command(job: Job, config: RunnerConfig) -> List[str]:
    """构造单个任务调用 `pred/call_api.py` 的完整命令。"""

    python = config.model_python.get(job.model.name, config.python)
    command = [
        python,
        "pred/call_api.py",
        "--data_dir",
        str(data_dir_for(job, config)),
        "--save_dir",
        str(save_dir_for(job, config)),
        "--benchmark",
        config.benchmark,
        "--task",
        job.task,
        "--subset",
        config.subset,
        "--server_type",
        config.server_type,
        "--model_name_or_path",
        str(job.model.path),
        "--temperature",
        str(config.temperature),
        "--top_k",
        str(config.top_k),
        "--top_p",
        str(config.top_p),
        "--random_seed",
        str(config.random_seed),
        "--batch_size",
        str(config.batch_size),
    ]
    if config.stop_words:
        command.extend(["--stop_words", config.stop_words])
    if config.log_batch_progress:
        command.append("--log_batch_progress")
    if config.log_attention_scores:
        command.append("--log_attention_scores")
        command.extend(["--attention_top_k", str(config.attention_top_k)])
    if config.log_generation_ppl:
        command.append("--log_generation_ppl")
    return command


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="把已经转换好的 RULER parquet-jsonl 数据并行送入 pred/call_api.py。"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="模型别名和路径，格式为 NAME=PATH；可以重复传入多个模型。",
    )
    parser.add_argument(
        "--seq-lengths",
        default="all",
        help="逗号分隔的长度列表，例如 4096,1048576；也可以传 all。",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="逗号分隔的任务列表，例如 niah_single_1,qa_2；也可以传 all。",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="逗号分隔的物理 GPU 编号，例如 0,1,2,3。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="最多同时运行多少个子进程；默认等于 GPU 数量。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="prepare_parquet.py 转换后的 synthetic 数据根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="预测结果和日志保存根目录。",
    )
    parser.add_argument("--benchmark", default="synthetic", help="RULER benchmark 名称。")
    parser.add_argument("--subset", default="validation", help="要读取的数据 split。")
    parser.add_argument("--server-type", default="hf", help="传给 call_api.py 的 server_type。")
    parser.add_argument("--python", default=sys.executable, help="用于启动 call_api.py 的 Python。")
    parser.add_argument(
        "--model-python",
        action="append",
        default=[],
        help="模型专属 Python，格式为 NAME=PYTHON；可以重复传入多个模型。",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--stop-words", default="", help="逗号分隔的停止词。")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="监视运行中任务的刷新间隔，单位秒。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果预测文件已经存在，则跳过对应任务。",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="启动任务前删除已有预测、注意力摘要和日志文件，强制重新生成。",
    )
    parser.add_argument(
        "--log-batch-progress",
        action="store_true",
        help="把 batch 级进度日志传给 call_api.py。",
    )
    parser.add_argument(
        "--log-attention-scores",
        action="store_true",
        help="让 Hugging Face 子进程输出逐层注意力摘要。",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=8,
        help="每层注意力摘要保留分数最高的 token 数量。",
    )
    parser.add_argument(
        "--log-generation-ppl",
        action="store_true",
        help="让 Hugging Face 子进程在生成阶段记录生成 token 的 PPL 字段。",
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        help="结构化任务耗时 jsonl；默认写到 output-root/ruler_timing.jsonl。",
    )
    parser.add_argument(
        "--auto-evaluate",
        action="store_true",
        help="全部子任务结束后自动调用 eval/collect_results.py 生成统一 csv 汇总。",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="自动汇总 csv 主输出路径；默认写到 output-root/ruler_results.csv。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的任务和命令，不启动推理。",
    )
    return parser


def build_config(args: argparse.Namespace, scripts_dir: Optional[Path] = None) -> RunnerConfig:
    """把 argparse 结果转换成 runner 内部配置。"""

    selected_scripts_dir = scripts_dir if scripts_dir is not None else SCRIPTS_DIR
    models = parse_model_specs(args.model)
    model_python = parse_model_python_specs(args.model_python, models)
    if args.skip_existing and args.overwrite_existing:
        raise ValueError("--skip-existing 和 --overwrite-existing 不能同时使用")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须是正整数")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval 必须是正数")
    if args.attention_top_k <= 0:
        raise ValueError("--attention-top-k 必须是正整数")
    timing_file = args.timing_file if args.timing_file is not None else args.output_root / DEFAULT_TIMING_NAME
    report_file = args.report_file if args.report_file is not None else args.output_root / DEFAULT_REPORT_NAME
    if report_file.suffix.lower() != ".csv":
        raise ValueError(f"--report-file 必须使用 .csv 后缀：{report_file}")

    return RunnerConfig(
        scripts_dir=selected_scripts_dir,
        data_root=args.data_root,
        output_root=args.output_root,
        benchmark=args.benchmark,
        server_type=args.server_type,
        subset=args.subset,
        python=args.python,
        model_python=model_python,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        random_seed=args.random_seed,
        stop_words=args.stop_words,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        log_batch_progress=args.log_batch_progress,
        log_attention_scores=args.log_attention_scores,
        attention_top_k=args.attention_top_k,
        log_generation_ppl=args.log_generation_ppl,
        overwrite_existing=args.overwrite_existing,
        timing_file=timing_file,
        auto_evaluate=args.auto_evaluate,
        report_file=report_file,
    )


def format_job(job: Job) -> str:
    """把任务信息格式化成适合终端显示的短文本。"""

    return f"model={job.model.name} length={job.seq_length} task={job.task}"


def shell_join(command: Sequence[str]) -> str:
    """把命令列表格式化成可复制的 shell 文本。"""

    return " ".join(shlex.quote(part) for part in command)


def is_batch_progress_line(line: str) -> bool:
    """判断子进程输出是否是需要回显到 runner 终端的 batch 进度行。"""

    return "[BATCH_START]" in line or "[BATCH_DONE]" in line or "[BATCH_FAILED]" in line


def stream_child_output(
    process: subprocess.Popen,
    log_handle: Any,
    job: Job,
    gpu: int,
) -> None:
    """把子进程输出写入日志，并把 batch 进度行同步回显到终端。"""

    if process.stdout is None:
        return
    for line in process.stdout:
        log_handle.write(line)
        if is_batch_progress_line(line):
            print(f"[BATCH] gpu={gpu} {format_job(job)} {line.rstrip()}", flush=True)


def filter_existing_jobs(
    jobs: Iterable[Job],
    config: RunnerConfig,
    skip_existing: bool,
) -> List[Job]:
    """根据 `--skip-existing` 过滤已经有预测文件的任务。"""

    selected: List[Job] = []
    for job in jobs:
        pred_file = prediction_file_for(job, config)
        if skip_existing and pred_file.exists():
            print(f"[SKIP] {format_job(job)} pred={pred_file}", flush=True)
            continue
        selected.append(job)
    return selected


def print_dry_run(jobs: Sequence[Job], gpus: Sequence[int], config: RunnerConfig) -> None:
    """打印任务矩阵和命令，不启动任何子进程。"""

    print(f"[DRY-RUN] jobs={len(jobs)} gpus={','.join(str(gpu) for gpu in gpus)}")
    if not jobs:
        return
    for idx, job in enumerate(jobs):
        gpu = gpus[idx % len(gpus)]
        command = build_call_api_command(job, config)
        print(f"[DRY-RUN] gpu={gpu} {format_job(job)}")
        print(f"  CUDA_VISIBLE_DEVICES={gpu} {shell_join(command)}")


def write_timing_record(
    config: RunnerConfig,
    job: Job,
    gpu: int,
    started_at: float,
    ended_at: float,
    elapsed_seconds: float,
    pred_lines: int,
    total_samples: int,
    exit_code: int,
) -> None:
    """追加写入单个任务的结构化耗时记录，供后续汇总脚本读取。"""

    config.timing_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": job.model.name,
        "length": job.seq_length,
        "task": job.task,
        "gpu": gpu,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "pred_lines": pred_lines,
        "total_samples": total_samples,
        "exit_code": exit_code,
    }
    with config.timing_file.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_timing_file(config: RunnerConfig) -> None:
    """初始化本次 runner 的耗时记录文件，避免混入上一次运行。"""

    config.timing_file.parent.mkdir(parents=True, exist_ok=True)
    config.timing_file.write_text("", encoding="utf-8")


def build_collect_results_command(
    config: RunnerConfig,
    models: Sequence[str],
    seq_lengths: Sequence[int],
    tasks: Sequence[str],
) -> List[str]:
    """构造自动汇总脚本命令。"""

    return [
        config.python,
        "eval/collect_results.py",
        "--output-root",
        str(config.output_root),
        "--data-root",
        str(config.data_root),
        "--benchmark",
        config.benchmark,
        "--models",
        ",".join(models),
        "--seq-lengths",
        ",".join(str(length) for length in seq_lengths),
        "--tasks",
        ",".join(tasks),
        "--timing-file",
        str(config.timing_file),
        "--output-file",
        str(config.report_file),
    ]


def run_collect_results(
    config: RunnerConfig,
    models: Sequence[str],
    seq_lengths: Sequence[int],
    tasks: Sequence[str],
) -> int:
    """调用统一汇总脚本，返回其退出码。"""

    command = build_collect_results_command(
        config=config,
        models=models,
        seq_lengths=seq_lengths,
        tasks=tasks,
    )
    print(f"[EVALUATE] {shell_join(command)}", flush=True)
    completed = subprocess.run(command, cwd=config.scripts_dir)
    return completed.returncode


def launch_job(job: Job, gpu: int, config: RunnerConfig) -> RunningJob:
    """在指定 GPU 上启动一个 `call_api.py` 子进程。"""

    save_dir_for(job, config).mkdir(parents=True, exist_ok=True)
    log_dir_for(job, config).mkdir(parents=True, exist_ok=True)
    if config.overwrite_existing:
        removed = delete_existing_outputs(job, config)
        if removed:
            removed_text = ",".join(str(path) for path in removed)
            print(f"[OVERWRITE] {format_job(job)} removed={removed_text}", flush=True)
    log_file = log_file_for(job, config)
    command = build_call_api_command(job, config)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = log_file.open("w", encoding="utf-8", buffering=1)
    log_handle.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {shell_join(command)}\n")
    process = subprocess.Popen(
        command,
        cwd=config.scripts_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_thread = threading.Thread(
        target=stream_child_output,
        kwargs=dict(process=process, log_handle=log_handle, job=job, gpu=gpu),
        daemon=True,
    )
    output_thread.start()
    total_samples = count_jsonl_lines(task_file_for(job, config))
    print(f"[START] gpu={gpu} pid={process.pid} {format_job(job)} log={log_file}", flush=True)
    return RunningJob(
        job=job,
        gpu=gpu,
        process=process,
        log_handle=log_handle,
        output_thread=output_thread,
        started_at=time.time(),
        total_samples=total_samples,
    )


def print_running_status(running_jobs: Sequence[RunningJob], config: RunnerConfig) -> None:
    """打印当前正在运行任务的预测行数进度。"""

    for running in running_jobs:
        pred_file = prediction_file_for(running.job, config)
        completed = count_jsonl_lines(pred_file)
        total = running.total_samples
        if total > 0:
            progress = f"{completed}/{total}"
        else:
            progress = f"{completed}/?"
        elapsed = time.time() - running.started_at
        print(
            f"[RUNNING] gpu={running.gpu} pid={running.process.pid} "
            f"{format_job(running.job)} progress={progress} elapsed={elapsed:.1f}s",
            flush=True,
        )


def run_scheduler(
    jobs: Sequence[Job],
    gpus: Sequence[int],
    config: RunnerConfig,
    max_workers: Optional[int],
) -> int:
    """按 GPU 空闲情况动态调度任务，返回进程退出码。"""

    if not gpus:
        raise ValueError("至少需要一个 GPU 编号")
    if max_workers is None:
        worker_count = len(gpus)
    else:
        if max_workers <= 0:
            raise ValueError("--max-workers 必须是正整数")
        worker_count = min(max_workers, len(gpus))

    pending: Deque[Job] = deque(jobs)
    available_gpus: Deque[int] = deque(gpus[:worker_count])
    running: Dict[int, RunningJob] = {}
    failed_jobs: List[RunningJob] = []
    completed_count = 0
    total_jobs = len(jobs)

    print(f"[QUEUE] jobs={total_jobs} workers={worker_count} gpus={list(available_gpus)}", flush=True)
    while pending or running:
        while pending and available_gpus and len(running) < worker_count:
            gpu = available_gpus.popleft()
            job = pending.popleft()
            running_job = launch_job(job, gpu, config)
            running[running_job.process.pid] = running_job

        if running:
            time.sleep(config.poll_interval)

        finished_pids: List[int] = []
        for pid, running_job in list(running.items()):
            return_code = running_job.process.poll()
            if return_code is None:
                continue

            running_job.output_thread.join()
            running_job.log_handle.close()
            finished_pids.append(pid)
            completed_count += 1
            available_gpus.append(running_job.gpu)
            ended_at = time.time()
            elapsed = ended_at - running_job.started_at
            pred_lines = count_jsonl_lines(prediction_file_for(running_job.job, config))
            write_timing_record(
                config=config,
                job=running_job.job,
                gpu=running_job.gpu,
                started_at=running_job.started_at,
                ended_at=ended_at,
                elapsed_seconds=elapsed,
                pred_lines=pred_lines,
                total_samples=running_job.total_samples,
                exit_code=return_code,
            )

            if return_code == 0:
                print(
                    f"[DONE] {completed_count}/{total_jobs} gpu={running_job.gpu} "
                    f"return_code=0 {format_job(running_job.job)} "
                    f"pred_lines={pred_lines} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            else:
                failed_jobs.append(running_job)
                print(
                    f"[FAILED] {completed_count}/{total_jobs} gpu={running_job.gpu} "
                    f"return_code={return_code} {format_job(running_job.job)} "
                    f"log={log_file_for(running_job.job, config)}",
                    flush=True,
                )

        for pid in finished_pids:
            del running[pid]

        if running:
            print_running_status(list(running.values()), config)

    if failed_jobs:
        print(f"[SUMMARY] failed={len(failed_jobs)} total={total_jobs}", flush=True)
        return 1
    print(f"[SUMMARY] failed=0 total={total_jobs}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    models = parse_model_specs(args.model)
    seq_lengths = resolve_seq_lengths(args.seq_lengths, config.data_root)
    tasks = resolve_tasks(args.tasks)
    gpus = parse_int_csv(args.gpus, "--gpus")
    jobs = build_jobs(models, seq_lengths, tasks)
    jobs = filter_existing_jobs(jobs, config, skip_existing=args.skip_existing)

    if args.dry_run:
        print_dry_run(jobs, gpus, config)
        return 0
    reset_timing_file(config)
    scheduler_code = run_scheduler(jobs, gpus, config, max_workers=args.max_workers)
    if config.auto_evaluate:
        evaluate_code = run_collect_results(
            config=config,
            models=[model.name for model in models],
            seq_lengths=seq_lengths,
            tasks=tasks,
        )
        if scheduler_code == 0:
            return evaluate_code
    return scheduler_code


if __name__ == "__main__":
    raise SystemExit(main())
