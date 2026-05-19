#!/usr/bin/env python
"""导出 Llama 单样本完整生成 attention。"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def read_jsonl_sample(path: Path, sample_offset: int) -> Dict[str, Any]:
    """读取 jsonl 中指定偏移位置的样本。"""

    if sample_offset < 0:
        raise ValueError("--sample-offset 必须大于或等于 0")
    with path.open("r", encoding="utf-8") as file_obj:
        for offset, line in enumerate(file_obj):
            if offset == sample_offset:
                return json.loads(line)
    raise IndexError(f"样本偏移超出文件范围：{sample_offset}")


def write_json(path: Path, value: Dict[str, Any]) -> None:
    """写入带缩进的 JSON 文件。"""

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写入 jsonl 文件。"""

    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_token_text(text: str) -> str:
    """把 token 文本整理成单行，方便后续表格查看。"""

    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text if text else "<empty>"


def decode_token(tokenizer, token_id: int) -> str:
    """解码单个 token id。"""

    try:
        return clean_token_text(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(token_id)


def build_token_records(tokenizer, token_ids: Sequence[int], source: str, start_position: int = 0) -> List[Dict[str, Any]]:
    """构造 token 元数据记录。"""

    rows: List[Dict[str, Any]] = []
    for local_index, token_id in enumerate(token_ids):
        row: Dict[str, Any] = {
            "position": start_position + local_index,
            "source": source,
            "token_id": int(token_id),
            "token_text": decode_token(tokenizer, int(token_id)),
        }
        if source == "generated":
            row["generated_index"] = local_index
            row["attention_file"] = attention_filename(local_index)
        rows.append(row)
    return rows


def attention_filename(generated_index: int) -> str:
    """返回某个生成 token 对应的 attention 文件名。"""

    return f"token_{generated_index:04d}.npy"


def summarize_attention_array(attention: np.ndarray) -> Dict[str, Any]:
    """统计 attention 数组形状和归一化误差。"""

    if attention.ndim != 3:
        raise ValueError(f"attention 数组必须是三维 [layer, head, key_position]，实际为 {attention.shape}")
    sums = attention.sum(axis=-1, dtype=np.float64)
    max_sum_error = float(np.max(np.abs(sums - 1.0))) if sums.size else 0.0
    return {
        "shape": [int(dim) for dim in attention.shape],
        "max_sum_error": max_sum_error,
        "min_sum": float(np.min(sums)) if sums.size else 0.0,
        "max_sum": float(np.max(sums)) if sums.size else 0.0,
    }


def build_metadata(
    model_path: Path,
    data_file: Path,
    sample_offset: int,
    sample_index: Any,
    prompt_text: str,
    prompt_token_count: int,
    generated_token_count: int,
    num_layers: int,
    num_heads: int,
    dtype: str,
    attention_files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构造导出目录的元数据。"""

    return {
        "model_path": str(model_path),
        "data_file": str(data_file),
        "sample_offset": sample_offset,
        "sample_index": sample_index,
        "prompt_preview": clean_token_text(prompt_text[:200]),
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dtype": dtype,
        "attention_semantics": "每个 token_XXXX.npy 保存第 XXXX 个生成 token 作为 query 时，对 prompt 和已生成上下文的 attention。",
        "attention_shape": "[num_layers, num_heads, key_length]",
        "attention_files": attention_files,
    }


def write_summary(path: Path, metadata: Dict[str, Any]) -> None:
    """写入人可读导出摘要。"""

    lines = [
        "# 完整 Attention 导出摘要",
        "",
        f"- 模型：`{metadata['model_path']}`",
        f"- 数据：`{metadata['data_file']}`",
        f"- 样本 offset：{metadata['sample_offset']}",
        f"- 样本 index：{metadata['sample_index']}",
        f"- prompt token 数：{metadata['prompt_token_count']}",
        f"- generated token 数：{metadata['generated_token_count']}",
        f"- 层数：{metadata['num_layers']}",
        f"- head 数：{metadata['num_heads']}",
        f"- dtype：`{metadata['dtype']}`",
        f"- attention shape：`{metadata['attention_shape']}`",
        "",
        "| 生成 token | 文件 | shape | 最大归一化误差 |",
        "|---:|---|---|---:|",
    ]
    for file_info in metadata["attention_files"]:
        lines.append(
            f"| {file_info['generated_index']} | `{file_info['file']}` | "
            f"`{file_info['shape']}` | {file_info['max_sum_error']:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备输出目录，避免误覆盖已有结果。"""

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空，如需覆盖请加 --overwrite：{output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def get_input_device(model) -> torch.device:
    """返回适合放置输入张量的设备。"""

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_llama_model(model_path: Path):
    """加载本地 Llama 模型和 tokenizer。"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def stack_step_attentions(attentions: Sequence[torch.Tensor], dtype: str) -> np.ndarray:
    """把单步输出的逐层 attention 堆叠为 [layer, head, key_position]。"""

    layers = []
    for attention in attentions:
        if attention is None:
            continue
        vector = attention.detach()[0, :, -1, :].float().cpu()
        layers.append(vector)
    if not layers:
        raise RuntimeError("模型没有返回 attention 张量，请确认 eager attention 和 output_attentions=True 可用。")
    stacked = torch.stack(layers, dim=0).numpy()
    if dtype == "float16":
        return stacked.astype(np.float16)
    return stacked.astype(np.float32)


def generate_tokens(model, inputs: Dict[str, torch.Tensor], max_new_tokens: int) -> torch.Tensor:
    """先正常生成完整回答，返回新生成 token ids。"""

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    sequence = generated.sequences[0]
    prompt_length = inputs["input_ids"].shape[1]
    return sequence[prompt_length:].detach().cpu()


def dump_generated_attention(
    model,
    inputs: Dict[str, torch.Tensor],
    generated_token_ids: torch.Tensor,
    output_dir: Path,
    dtype: str,
) -> List[Dict[str, Any]]:
    """逐个生成 token replay，并保存完整 head attention。"""

    attention_files: List[Dict[str, Any]] = []
    if generated_token_ids.numel() == 0:
        return attention_files

    with torch.no_grad():
        prefill = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_key_values = prefill.past_key_values
        if past_key_values is None:
            raise RuntimeError("模型没有返回 past_key_values，无法高效 replay 生成 token。")

        attention_mask = inputs.get("attention_mask")
        for generated_index, token_id in enumerate(generated_token_ids.tolist()):
            current_token = torch.tensor([[int(token_id)]], device=get_input_device(model), dtype=torch.long)
            if attention_mask is not None:
                next_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            outputs = model(
                input_ids=current_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            attention = stack_step_attentions(outputs.attentions, dtype=dtype)
            file_name = attention_filename(generated_index)
            np.save(output_dir / file_name, attention)
            file_info = {
                "generated_index": generated_index,
                "file": file_name,
                **summarize_attention_array(attention),
            }
            attention_files.append(file_info)
            past_key_values = outputs.past_key_values
            del outputs, attention

    return attention_files


def run_dump(args: argparse.Namespace) -> None:
    """执行完整 attention 导出。"""

    prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    sample = read_jsonl_sample(args.data_file, args.sample_offset)
    prompt_text = sample["input"]
    sample_index = sample.get("index", args.sample_offset)

    tokenizer, model = load_llama_model(args.model_path)
    device = get_input_device(model)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_token_ids = inputs["input_ids"][0].detach().cpu().tolist()
    generated_token_ids = generate_tokens(model, inputs, max_new_tokens=args.max_new_tokens)

    prompt_tokens = build_token_records(tokenizer, prompt_token_ids, source="prompt")
    generated_tokens = build_token_records(
        tokenizer,
        generated_token_ids.tolist(),
        source="generated",
        start_position=len(prompt_token_ids),
    )
    attention_files = dump_generated_attention(
        model=model,
        inputs=inputs,
        generated_token_ids=generated_token_ids,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )

    first_shape = attention_files[0]["shape"] if attention_files else [0, 0, 0]
    metadata = build_metadata(
        model_path=args.model_path,
        data_file=args.data_file,
        sample_offset=args.sample_offset,
        sample_index=sample_index,
        prompt_text=prompt_text,
        prompt_token_count=len(prompt_token_ids),
        generated_token_count=int(generated_token_ids.numel()),
        num_layers=first_shape[0],
        num_heads=first_shape[1],
        dtype=args.dtype,
        attention_files=attention_files,
    )

    write_json(args.output_dir / "metadata.json", metadata)
    write_jsonl(args.output_dir / "prompt_tokens.jsonl", prompt_tokens)
    write_jsonl(args.output_dir / "generated_tokens.jsonl", generated_tokens)
    write_summary(args.output_dir / "summary.md", metadata)
    print(f"Attention dump saved to {args.output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="导出 Llama 单样本完整生成 attention。")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的输出目录。")
    return parser


def main() -> int:
    """命令行入口。"""

    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须是正整数")
    run_dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
