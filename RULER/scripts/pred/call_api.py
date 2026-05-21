# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback
from contextlib import ExitStack
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--log_batch_progress", action="store_true", help="输出每个 batch 的开始和结束进度。")
parser.add_argument("--max_retries", type=int, default=3, help="单个 batch 推理失败后的最大重试次数。")
parser.add_argument("--log_attention_scores", action="store_true", help="输出 Hugging Face 模型逐层注意力摘要。")
parser.add_argument("--attention_top_k", type=int, default=8, help="每层注意力摘要保留分数最高的 token 数量。")
parser.add_argument("--log_generation_ppl", action="store_true", help="输出 Hugging Face 生成答案 token 的 PPL。")
parser.add_argument("--log_generation_token_ppl", action="store_true", help="额外输出 Hugging Face 每个生成 token 的 PPL 明细。")
parser.add_argument("--log_prefill_decode_timing", action="store_true", help="输出 Hugging Face prefill/decode forward 阶段耗时。")
parser.add_argument("--profile_attention_kernels", action="store_true", help="对每个任务第 0 行样本额外统计严格 attention CUDA kernel 时间。")
parser.add_argument("--attention_profile_sample_offset", type=int, default=0, help="固定为 0，只 profile 输入 jsonl 第 0 行样本。")


def validate_runtime_args(parsed_args):
    """集中校验运行参数，保证本地评测口径稳定。"""

    if parsed_args.batch_size != 1:
        raise ValueError("--batch_size 当前固定为 1，请不要传入其他值。")
    if parsed_args.max_retries <= 0:
        raise ValueError("--max_retries 必须是正整数")
    if parsed_args.attention_top_k <= 0:
        raise ValueError("--attention_top_k 必须是正整数")
    if parsed_args.log_attention_scores and parsed_args.server_type != "hf":
        raise ValueError("--log_attention_scores 目前只支持 --server_type hf")
    if (parsed_args.log_generation_ppl or parsed_args.log_generation_token_ppl) and parsed_args.server_type != "hf":
        raise ValueError("--log_generation_ppl 和 --log_generation_token_ppl 目前只支持 --server_type hf")
    if (parsed_args.log_prefill_decode_timing or parsed_args.profile_attention_kernels) and parsed_args.server_type != "hf":
        raise ValueError("--log_prefill_decode_timing 和 --profile_attention_kernels 目前只支持 --server_type hf")
    if parsed_args.profile_attention_kernels and parsed_args.log_attention_scores:
        raise ValueError("--profile_attention_kernels 需要保持 flash attention 路径，不能和 --log_attention_scores 同时使用。")
    if parsed_args.attention_profile_sample_offset != 0:
        raise ValueError("--attention_profile_sample_offset 当前固定为 0，也就是只采输入 jsonl 第 0 行。")
    if parsed_args.log_generation_token_ppl:
        parsed_args.log_generation_ppl = True
    parsed_args.stop_words = list(filter(None, parsed_args.stop_words.split(',')))
    if parsed_args.server_type == 'hf' or parsed_args.server_type == 'gemini':
        parsed_args.threads = 1
    return parsed_args


args = validate_runtime_args(parser.parse_args())


def get_llm(tokens_to_generate):
    if args.server_type == 'trtllm':
        from client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'sglang':
        from client_wrappers import SGLClient
        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'hf':
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
            log_attention_scores=args.log_attention_scores,
            attention_top_k=args.attention_top_k,
            log_generation_ppl=args.log_generation_ppl,
            log_generation_token_ppl=args.log_generation_token_ppl,
            log_prefill_decode_timing=args.log_prefill_decode_timing,
            profile_attention_kernels=args.profile_attention_kernels,
        )
    
    elif args.server_type == 'mamba':
        from model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
        
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def format_batch_failure_message(task_name, batch_meta, attempt, max_retries, error):
    """格式化 batch 失败日志，方便 runner 从子进程输出中识别失败位置。"""

    return (
        f"[BATCH_FAILED] task={task_name} "
        f"batch={batch_meta['batch_no']}/{batch_meta['total_batches']} "
        f"size={batch_meta['size']} "
        f"index_range={batch_meta['index_start']}-{batch_meta['index_end']} "
        f"attempt={attempt}/{max_retries} "
        f"error={type(error).__name__}: {error}"
    )


def process_batch_with_retries(llm, input_list, batch_meta, max_retries, task_name):
    """有限重试执行一次 batch 推理；超过次数后把异常抛回主流程。"""

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return llm.process_batch(prompts=input_list)
        except Exception as error:
            last_error = error
            traceback.print_exc()
            sys.stderr.flush()
            print(
                format_batch_failure_message(
                    task_name=task_name,
                    batch_meta=batch_meta,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=error,
                ),
                flush=True,
            )

    raise RuntimeError(
        f"batch 推理连续失败 {max_retries} 次："
        f"task={task_name} batch={batch_meta['batch_no']}/{batch_meta['total_batches']}"
    ) from last_error


def raise_worker_errors(worker_errors, worker_error_lock):
    """如果任一推理线程失败，则在主线程重新抛出异常并终止当前任务。"""

    with worker_error_lock:
        errors = list(worker_errors)
    if errors:
        raise RuntimeError(f"推理线程失败，停止写入当前任务：{errors[0]}") from errors[0]


def escape_markdown_cell(value) -> str:
    """转义写入 Markdown 表格单元格的文本。"""

    return str(value).replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


def format_attention_markdown(task_name, sample_index, summary) -> str:
    """把单条样本的逐层注意力摘要格式化成容易阅读的 Markdown。"""

    generated_text = escape_markdown_cell(summary.get("generated_token_text", ""))
    generated_id = summary.get("generated_token_id")
    lines = [
        f"## 样本 index={sample_index}",
        "",
        f"- 任务：`{task_name}`",
        f"- 注意力模式：`{summary.get('mode', 'unknown')}`",
        f"- prompt token 数：{summary.get('prompt_tokens', 'unknown')}",
        f"- 观察的生成 token：`{generated_text}` (id={generated_id})",
    ]
    if summary.get("warning"):
        lines.append(f"- 警告：{escape_markdown_cell(summary['warning'])}")

    lines.extend(
        [
            "",
            "| 层 | 归一化和 | Top 注意力 token |",
            "|---:|---:|---|",
        ]
    )
    layers = summary.get("layers") or []
    if not layers:
        lines.append("| - | - | 无可用注意力输出 |")
        return "\n".join(lines)

    for layer in layers:
        token_parts = []
        for item in layer.get("top_tokens", []):
            token = escape_markdown_cell(item.get("token", ""))
            token_parts.append(
                f"{item.get('rank')}. pos={item.get('position')} "
                f"score={item.get('score', 0.0):.6f} token=`{token}`"
            )
        lines.append(
            f"| {layer.get('layer')} | {layer.get('sum', 0.0):.6f} | {'; '.join(token_parts)} |"
        )
    return "\n".join(lines)


def build_prediction_record(pred, index, input_text, outputs, others, truncation, length):
    """把模型返回结果转换成预测 jsonl 记录，并保留可选生成统计字段。"""

    if isinstance(pred['text'], str):
        pred_text = pred['text']
    elif len(pred['text']) > 0:
        pred_text = pred['text'][0]
    else:
        pred_text = ''

    record = {
        'index': index,
        'pred': pred_text,
        'input': input_text,
        'outputs': outputs,
        'others': others,
        'truncation': truncation,
        'length': length,
    }
    for field in (
        "generation_logprob_sum",
        "generation_token_count",
        "generation_nll",
        "generation_ppl",
    ):
        if isinstance(pred, dict) and field in pred:
            record[field] = pred[field]
    return record


def build_generation_token_record(pred, index, task):
    """把模型返回的逐 token 生成 PPL 明细转换成 sidecar jsonl 记录。"""

    tokens = []
    if isinstance(pred, dict):
        tokens = pred.get("generation_tokens") or []
    return {
        "index": index,
        "task": task,
        "generation_token_count": len(tokens),
        "tokens": tokens,
    }


def build_generation_timing_record(pred, index, task, sample_line_no):
    """把模型返回的 prefill/decode timing 转换成 sidecar jsonl 记录。"""

    timing = {}
    if isinstance(pred, dict):
        timing = pred.get("generation_timing") or {}
    record = {
        "record_type": "sample_timing",
        "task": task,
        "sample_line_no": sample_line_no,
        "sample_index": index,
    }
    record.update(timing)
    return record


def select_attention_profile_sample(data, sample_offset=0):
    """选择严格 profiler 样本；当前固定只允许输入 jsonl 第 0 行。"""

    if sample_offset != 0:
        raise ValueError("attention profiler 当前固定只采输入 jsonl 第 0 行。")
    if not data:
        return None
    return {
        "sample_line_no": 0,
        "sample": data[0],
    }


def build_attention_profile_record(profile, task, sample, sample_line_no, input_file):
    """把第 0 行样本的严格 attention profiler 结果转换成 sidecar jsonl 记录。"""

    record = {
        "record_type": "attention_profile",
        "task": task,
        "profile_sample_policy": "first_input_record",
        "sample_line_no": sample_line_no,
        "sample_index": sample.get("index"),
        "input_file": str(input_file),
    }
    record.update(profile)
    return record


def timing_sidecar_has_attention_profile(path):
    """判断 timing sidecar 是否已经存在 attention_profile 记录，避免续跑重复追加。"""

    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") == "attention_profile":
                return True
    return False


def main():
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    all_data = []
    for sample_line_no, sample in enumerate(read_manifest(task_file)):
        sample = dict(sample)
        sample["sample_line_no"] = sample_line_no
        all_data.append(sample)
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in all_data if sample['index'] not in pred_index]
    else:
        data = list(all_data)

    # Load api
    llm = get_llm(config['tokens_to_generate'])

    def get_output(
        idx_list,
        index_list,
        input_list,
        outputs_list,
        others_list,
        truncation_list,
        length_list,
        sample_line_no_list,
        batch_meta,
    ):
        nonlocal llm

        try:
            pred_list = process_batch_with_retries(
                llm=llm,
                input_list=input_list,
                batch_meta=batch_meta,
                max_retries=args.max_retries,
                task_name=args.task,
            )
        except Exception as error:
            with worker_error_lock:
                worker_errors.append(error)
            return

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list, sample_line_no_list)

        for pred, idx, index, input, outputs, others, truncation, length, sample_line_no in zipped_iter:
            outputs_parallel[idx] = build_prediction_record(
                pred=pred,
                index=index,
                input_text=input,
                outputs=outputs,
                others=others,
                truncation=truncation,
                length=length,
            )
            if isinstance(pred, dict) and pred.get("attention") is not None:
                attention_parallel[idx] = {
                    "index": index,
                    "task": args.task,
                    "attention": pred["attention"],
                }
            if isinstance(pred, dict) and pred.get("generation_tokens") is not None:
                generation_token_parallel[idx] = build_generation_token_record(
                    pred=pred,
                    index=index,
                    task=args.task,
                )
            if isinstance(pred, dict) and pred.get("generation_timing") is not None:
                generation_timing_parallel[idx] = build_generation_timing_record(
                    pred=pred,
                    index=index,
                    task=args.task,
                    sample_line_no=sample_line_no,
                )

        if args.log_batch_progress:
            elapsed = time.time() - batch_meta['started_at']
            print(
                f"[BATCH_DONE] task={args.task} "
                f"batch={batch_meta['batch_no']}/{batch_meta['total_batches']} "
                f"size={batch_meta['size']} "
                f"index_range={batch_meta['index_start']}-{batch_meta['index_end']} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

    threads = []
    worker_errors = []
    worker_error_lock = threading.Lock()
    outputs_parallel = [{} for _ in range(len(data))]
    attention_parallel = [{} for _ in range(len(data))]
    generation_token_parallel = [{} for _ in range(len(data))]
    generation_timing_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    attention_jsonl_file = pred_file.with_suffix(".attention.jsonl")
    attention_markdown_file = pred_file.with_suffix(".attention.md")
    generation_token_jsonl_file = pred_file.with_suffix(".generation_tokens.jsonl")
    generation_timing_jsonl_file = pred_file.with_suffix(".generation_timing.jsonl")
    write_attention_header = (
        args.log_attention_scores
        and (not attention_markdown_file.exists() or attention_markdown_file.stat().st_size == 0)
    )

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with ExitStack() as stack:
        fout = stack.enter_context(open(pred_file, 'at', encoding="utf-8", buffering=1))
        attention_jsonl = None
        attention_markdown = None
        generation_token_jsonl = None
        generation_timing_jsonl = None
        if args.log_attention_scores:
            attention_jsonl = stack.enter_context(open(attention_jsonl_file, 'at', encoding="utf-8", buffering=1))
            attention_markdown = stack.enter_context(open(attention_markdown_file, 'at', encoding="utf-8", buffering=1))
            if write_attention_header:
                attention_markdown.write(f"# {args.task} 注意力摘要\n\n")
            print(f"Attention summaries to {attention_markdown_file} and {attention_jsonl_file}", flush=True)
        if args.log_generation_token_ppl:
            generation_token_jsonl = stack.enter_context(open(generation_token_jsonl_file, 'at', encoding="utf-8", buffering=1))
            print(f"生成 token PPL 明细写入 {generation_token_jsonl_file}", flush=True)
        if args.log_prefill_decode_timing or args.profile_attention_kernels:
            generation_timing_jsonl = stack.enter_context(open(generation_timing_jsonl_file, 'at', encoding="utf-8", buffering=1))
            print(f"生成阶段 timing 明细写入 {generation_timing_jsonl_file}", flush=True)

        if args.profile_attention_kernels and not timing_sidecar_has_attention_profile(generation_timing_jsonl_file):
            profile_sample = select_attention_profile_sample(
                all_data,
                sample_offset=args.attention_profile_sample_offset,
            )
            if profile_sample is not None:
                sample = profile_sample["sample"]
                try:
                    profile = llm.profile_attention_kernels_for_prompt(sample["input"])
                except Exception as error:
                    profile = {
                        "timer_backend": "torch_profiler",
                        "input_tokens": None,
                        "generated_token_count": None,
                        "prefill_attention_kernel_ms": None,
                        "decode_attention_kernel_ms_total": None,
                        "decode_attention_kernel_ms_per_token_avg": None,
                        "attention_kernel_event_count": 0,
                        "warning": f"{type(error).__name__}: {error}",
                    }
                generation_timing_jsonl.write(
                    json.dumps(
                        build_attention_profile_record(
                            profile=profile,
                            task=args.task,
                            sample=sample,
                            sample_line_no=profile_sample["sample_line_no"],
                            input_file=task_file,
                        ),
                        ensure_ascii=False,
                    )
                    + '\n'
                )

        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            index_list = [data_point['index'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered
            batch_meta = {
                'batch_no': batch_idx + 1,
                'total_batches': len(batched_data),
                'size': len(batch),
                'index_start': index_list[0],
                'index_end': index_list[-1],
                'started_at': time.time(),
            }
            if args.log_batch_progress:
                print(
                    f"[BATCH_START] task={args.task} "
                    f"batch={batch_meta['batch_no']}/{batch_meta['total_batches']} "
                    f"size={batch_meta['size']} "
                    f"index_range={batch_meta['index_start']}-{batch_meta['index_end']}",
                    flush=True,
                )

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=index_list,
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                    sample_line_no_list=[data_point.get('sample_line_no') for data_point in batch],
                    batch_meta=batch_meta,
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []
                raise_worker_errors(worker_errors, worker_error_lock)

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')
                    if attention_jsonl is not None and len(attention_parallel[idx]) > 0:
                        attention_record = attention_parallel[idx]
                        attention_jsonl.write(json.dumps(attention_record, ensure_ascii=False) + '\n')
                        attention_markdown.write(
                            format_attention_markdown(
                                task_name=args.task,
                                sample_index=attention_record["index"],
                                summary=attention_record["attention"],
                            )
                            + "\n\n"
                        )
                    if generation_token_jsonl is not None and len(generation_token_parallel[idx]) > 0:
                        generation_token_jsonl.write(json.dumps(generation_token_parallel[idx], ensure_ascii=False) + '\n')
                    if generation_timing_jsonl is not None and len(generation_timing_parallel[idx]) > 0:
                        generation_timing_jsonl.write(json.dumps(generation_timing_parallel[idx], ensure_ascii=False) + '\n')

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
