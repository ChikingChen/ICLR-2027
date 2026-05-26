#!/usr/bin/env python
"""对比 pooling token 分数和细粒度 full attention 分数。"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

try:
    from attention_mask_utils import apply_bos_attention_mask, disabled_attention_mask_ablation
except ImportError:
    from tools.attention_mask_utils import apply_bos_attention_mask, disabled_attention_mask_ablation


POOLING_ATTENTION_CSV_FIELDS = [
    "sample_index",
    "query_generated_index",
    "query_token_id",
    "query_token_text",
    "layer",
    "head",
    "block_id",
    "start_position",
    "end_position",
    "token_count",
    "token_positions",
    "token_ids",
    "token_texts",
    "fine_attention_values",
    "fine_attention_sum",
    "avg_pooling_attention",
    "max_pooling_attention",
    "avg_minus_sum",
    "max_minus_sum",
    "avg_equals_sum",
    "max_equals_sum",
    "avg_over_sum",
    "max_over_sum",
]


def read_jsonl_sample(path: Path, sample_offset: int) -> Dict[str, Any]:
    """读取 jsonl 文件中指定 offset 的样本。"""

    if sample_offset < 0:
        raise ValueError("--sample-offset 必须大于或等于 0")
    with path.open("r", encoding="utf-8") as file_obj:
        for offset, line in enumerate(file_obj):
            if offset == sample_offset:
                return json.loads(line)
    raise IndexError(f"样本 offset 超出范围：{sample_offset}")


def write_json(path: Path, value: Dict[str, Any]) -> None:
    """写入带缩进的 JSON 文件。"""

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写入 jsonl 文件。"""

    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """写入 CSV 文件。"""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_float(value: float) -> float:
    """把 numpy/float32 值整理成适合 CSV 和 JSON 展示的浮点数。"""

    return float(f"{float(value):.8g}")


def json_cell(value: Any) -> str:
    """把列表值编码成一个稳定的 CSV 单元格。"""

    return json.dumps(value, ensure_ascii=False)


def clean_token_text(text: str) -> str:
    """把 token 文本整理成适合单行输出的形式。"""

    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text if text else "<empty>"


def decode_token(tokenizer, token_id: int) -> str:
    """解码单个 token id，失败时退回到 id 字符串。"""

    try:
        return clean_token_text(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(token_id)


def build_token_records(
    tokenizer,
    token_ids: Sequence[int],
    source: str,
    start_position: int = 0,
    attention_mask_values: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """构造 prompt 或 generated token 的元数据记录。"""

    rows: List[Dict[str, Any]] = []
    for local_index, token_id in enumerate(token_ids):
        row: Dict[str, Any] = {
            "position": start_position + local_index,
            "source": source,
            "token_id": int(token_id),
            "token_text": decode_token(tokenizer, int(token_id)),
        }
        if attention_mask_values is not None:
            mask_value = int(attention_mask_values[local_index])
            row["attention_mask"] = mask_value
            row["masked"] = mask_value == 0
        if source == "generated":
            row["generated_index"] = local_index
        rows.append(row)
    return rows


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备输出目录，并在显式允许时覆盖旧结果。"""

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空，如需覆盖请传 --overwrite：{output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def get_input_device(model) -> torch.device:
    """返回适合放置 input tensor 的模型设备。"""

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_llama_model(model_path: Path):
    """加载本地 Llama 模型和 tokenizer，并优先启用 eager attention。"""

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


def generate_tokens(model, inputs: Dict[str, torch.Tensor], max_new_tokens: int) -> torch.Tensor:
    """先正常生成回答，返回新增生成 token ids。"""

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


def stack_step_attentions(attentions: Sequence[torch.Tensor], dtype: str) -> np.ndarray:
    """把单步输出的逐层 attention 堆叠成 [layer, head, key_position]。"""

    layers = []
    for attention in attentions:
        if attention is None:
            continue
        layers.append(attention.detach()[0, :, -1, :].float().cpu())
    if not layers:
        raise RuntimeError("模型没有返回 attention 张量，请确认 eager attention 可用")
    stacked = torch.stack(layers, dim=0).numpy()
    if dtype == "float16":
        return stacked.astype(np.float16)
    return stacked.astype(np.float32)


def replay_query_attention(
    model,
    inputs: Dict[str, torch.Tensor],
    generated_token_ids: torch.Tensor,
    query_generated_index: int,
    dtype: str,
) -> np.ndarray:
    """用 KV cache replay 到指定生成 token，并捕获该 token 的完整 attention。"""

    if query_generated_index < 0:
        raise ValueError("--query-generated-index 必须大于或等于 0")
    if query_generated_index >= int(generated_token_ids.numel()):
        raise IndexError(
            f"query generated index {query_generated_index} 超出生成 token 数量 "
            f"{int(generated_token_ids.numel())}"
        )

    with torch.no_grad():
        prefill = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_key_values = prefill.past_key_values
        if past_key_values is None:
            raise RuntimeError("模型没有返回 past_key_values，无法 replay query attention")

        attention_mask = inputs.get("attention_mask")
        query_attention = None
        for generated_index, token_id in enumerate(generated_token_ids[: query_generated_index + 1].tolist()):
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
                position_ids=torch.tensor(
                    [[inputs["input_ids"].shape[1] + generated_index]],
                    device=get_input_device(model),
                    dtype=torch.long,
                )
                if "position_ids" in inputs
                else None,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=generated_index == query_generated_index,
                return_dict=True,
            )
            if generated_index == query_generated_index:
                query_attention = stack_step_attentions(outputs.attentions, dtype=dtype)
            past_key_values = outputs.past_key_values

    if query_attention is None:
        raise RuntimeError("未捕获到 query attention")
    return query_attention


def build_prompt_blocks(token_count: int, block_size: int) -> List[Dict[str, int]]:
    """按固定 block size 把 prompt token 切成 pooling token 覆盖范围。"""

    if token_count < 0:
        raise ValueError("token_count 必须大于或等于 0")
    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")

    blocks: List[Dict[str, int]] = []
    for start in range(0, token_count, block_size):
        end = min(start + block_size, token_count) - 1
        blocks.append(
            {
                "block_id": len(blocks),
                "start_position": start,
                "end_position": end,
                "token_count": end - start + 1,
            }
        )
    return blocks


def rank_block_scores(rows: List[Dict[str, Any]], score_key: str, rank_key: str, selected_key: str, top_k: int) -> None:
    """按指定分数给 block 排名，并标记 top-k 是否选中。"""

    ranked = sorted(rows, key=lambda row: (-float(row[score_key]), int(row["block_id"])))
    selected_ids = {row["block_id"] for row in ranked[: max(0, min(top_k, len(ranked)))]}
    for rank, row in enumerate(ranked, start=1):
        row[rank_key] = rank
        row[selected_key] = row["block_id"] in selected_ids


def _layer_head_values(values: np.ndarray, include_details: bool) -> Any:
    """按开关决定是否把每层每 head 的明细写入 jsonl。"""

    if not include_details:
        return None
    return values.astype(float).tolist()


def build_pooling_comparison(
    attention: np.ndarray,
    prompt_tokens: Sequence[Dict[str, Any]],
    block_size: int,
    top_k_blocks: int,
    include_layer_head_details: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """构造 pooling token 和细粒度 token 的 attention 对照表。"""

    if attention.ndim != 3:
        raise ValueError(f"attention 必须是 [layer, head, key_position]，实际为 {attention.shape}")
    if top_k_blocks <= 0:
        raise ValueError("top_k_blocks 必须大于 0")

    prompt_length = len(prompt_tokens)
    if prompt_length > attention.shape[-1]:
        raise ValueError(
            f"prompt token 数 {prompt_length} 超过 attention key 长度 {attention.shape[-1]}"
        )

    prompt_attention = attention[:, :, :prompt_length].astype(np.float64, copy=False)
    mean_attention = prompt_attention.mean(axis=(0, 1))
    blocks = build_prompt_blocks(prompt_length, block_size)

    fine_tokens: List[Dict[str, Any]] = []
    for token_record in prompt_tokens:
        position = int(token_record["position"])
        values = prompt_attention[:, :, position]
        row = {
            "position": position,
            "block_id": position // block_size,
            "token_id": int(token_record["token_id"]),
            "token_text": token_record["token_text"],
            "full_attention_mean": float(mean_attention[position]),
            "full_attention_max": float(values.max()),
            "full_attention_min": float(values.min()),
        }
        details = _layer_head_values(values, include_layer_head_details)
        if details is not None:
            row["full_attention_by_layer_head"] = details
        fine_tokens.append(row)

    pooling_tokens: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for block in blocks:
        start = block["start_position"]
        end_exclusive = block["end_position"] + 1
        block_vector = mean_attention[start:end_exclusive]
        block_values = prompt_attention[:, :, start:end_exclusive]

        pooling_row: Dict[str, Any] = {
            **block,
            "pooling_score_max": float(block_vector.max()) if block_vector.size else 0.0,
            "pooling_score_avg": float(block_vector.mean()) if block_vector.size else 0.0,
            "full_attention_sum": float(block_vector.sum()) if block_vector.size else 0.0,
            "full_attention_mean": float(block_vector.mean()) if block_vector.size else 0.0,
            "full_attention_max": float(block_vector.max()) if block_vector.size else 0.0,
        }
        if include_layer_head_details:
            pooling_row["pooling_score_max_by_layer_head"] = block_values.max(axis=2).astype(float).tolist()
            pooling_row["pooling_score_avg_by_layer_head"] = block_values.mean(axis=2).astype(float).tolist()
            pooling_row["full_attention_sum_by_layer_head"] = block_values.sum(axis=2).astype(float).tolist()

        token_rows = [
            {
                "position": fine_token["position"],
                "token_id": fine_token["token_id"],
                "token_text": fine_token["token_text"],
                "full_attention_mean": fine_token["full_attention_mean"],
                "full_attention_max": fine_token["full_attention_max"],
                "full_attention_min": fine_token["full_attention_min"],
            }
            for fine_token in fine_tokens[start:end_exclusive]
        ]
        summary_rows.append(
            {
                **block,
                "pooling_score_max": pooling_row["pooling_score_max"],
                "pooling_score_avg": pooling_row["pooling_score_avg"],
                "full_attention_sum": pooling_row["full_attention_sum"],
                "full_attention_mean": pooling_row["full_attention_mean"],
                "full_attention_max": pooling_row["full_attention_max"],
                "tokens": token_rows,
            }
        )
        pooling_tokens.append(pooling_row)

    rank_block_scores(pooling_tokens, "pooling_score_max", "rank_by_max", "selected_by_max", top_k_blocks)
    rank_block_scores(pooling_tokens, "pooling_score_avg", "rank_by_avg", "selected_by_avg", top_k_blocks)

    ranks = {row["block_id"]: row for row in pooling_tokens}
    for row in summary_rows:
        ranked_row = ranks[row["block_id"]]
        row["rank_by_max"] = ranked_row["rank_by_max"]
        row["rank_by_avg"] = ranked_row["rank_by_avg"]
        row["selected_by_max"] = ranked_row["selected_by_max"]
        row["selected_by_avg"] = ranked_row["selected_by_avg"]

    return {
        "pooling_tokens": pooling_tokens,
        "fine_tokens": fine_tokens,
        "pooling_vs_fine_summary": summary_rows,
    }


def build_pooling_attention_csv_rows(
    attention: np.ndarray,
    prompt_tokens: Sequence[Dict[str, Any]],
    block_size: int,
    metadata: Dict[str, Any] | None = None,
    tolerance: float = 1e-8,
) -> List[Dict[str, Any]]:
    """按 layer/head/block 展开 pooling 分数和细粒度 attention 之和的 CSV 行。"""

    if attention.ndim != 3:
        raise ValueError(f"attention 必须是 [layer, head, key_position]，实际为 {attention.shape}")
    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")

    prompt_length = len(prompt_tokens)
    if prompt_length > attention.shape[-1]:
        raise ValueError(
            f"prompt token 数 {prompt_length} 超过 attention key 长度 {attention.shape[-1]}"
        )

    metadata = metadata or {}
    prompt_attention = attention[:, :, :prompt_length].astype(np.float64, copy=False)
    blocks = build_prompt_blocks(prompt_length, block_size)
    rows: List[Dict[str, Any]] = []

    for layer in range(prompt_attention.shape[0]):
        for head in range(prompt_attention.shape[1]):
            for block in blocks:
                start = block["start_position"]
                end_exclusive = block["end_position"] + 1
                block_tokens = prompt_tokens[start:end_exclusive]
                raw_values = prompt_attention[layer, head, start:end_exclusive]
                values = [compact_float(value) for value in raw_values.tolist()]
                fine_sum = compact_float(sum(values))
                avg_pooling = compact_float(fine_sum / len(values)) if values else 0.0
                max_pooling = compact_float(max(values)) if values else 0.0
                avg_minus_sum = compact_float(avg_pooling - fine_sum)
                max_minus_sum = compact_float(max_pooling - fine_sum)
                avg_equals_sum = abs(avg_pooling - fine_sum) <= tolerance
                max_equals_sum = abs(max_pooling - fine_sum) <= tolerance

                rows.append(
                    {
                        "sample_index": metadata.get("sample_index", ""),
                        "query_generated_index": metadata.get("query_generated_index", ""),
                        "query_token_id": metadata.get("query_token_id", ""),
                        "query_token_text": metadata.get("query_token_text", ""),
                        "layer": layer,
                        "head": head,
                        "block_id": block["block_id"],
                        "start_position": start,
                        "end_position": block["end_position"],
                        "token_count": block["token_count"],
                        "token_positions": json_cell([int(token["position"]) for token in block_tokens]),
                        "token_ids": json_cell([int(token["token_id"]) for token in block_tokens]),
                        "token_texts": json_cell([token["token_text"] for token in block_tokens]),
                        "fine_attention_values": json_cell(values),
                        "fine_attention_sum": fine_sum,
                        "avg_pooling_attention": avg_pooling,
                        "max_pooling_attention": max_pooling,
                        "avg_minus_sum": avg_minus_sum,
                        "max_minus_sum": max_minus_sum,
                        "avg_equals_sum": avg_equals_sum,
                        "max_equals_sum": max_equals_sum,
                        "avg_over_sum": compact_float(avg_pooling / fine_sum) if fine_sum else "",
                        "max_over_sum": compact_float(max_pooling / fine_sum) if fine_sum else "",
                    }
                )

    return rows


def format_top_block_table(rows: Sequence[Dict[str, Any]], selected_key: str, rank_key: str) -> List[str]:
    """把选中的 top block 格式化成 Markdown 表格。"""

    selected = sorted((row for row in rows if row[selected_key]), key=lambda row: int(row[rank_key]))
    lines = [
        "| rank | block | token range | max score | avg score | full sum |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row[rank_key]} | {row['block_id']} | "
            f"{row['start_position']}-{row['end_position']} | "
            f"{row['pooling_score_max']:.8f} | {row['pooling_score_avg']:.8f} | "
            f"{row['full_attention_sum']:.8f} |"
        )
    return lines


def write_summary_markdown(path: Path, metadata: Dict[str, Any], comparison: Dict[str, List[Dict[str, Any]]]) -> None:
    """写入人可读的 pooling 与细粒度 attention 对照摘要。"""

    top_token_count = int(metadata.get("top_k_blocks", 8))
    pooling_rows = comparison["pooling_tokens"]
    fine_rows = sorted(
        comparison["fine_tokens"],
        key=lambda row: (-float(row["full_attention_mean"]), int(row["position"])),
    )[:top_token_count]

    lines = [
        "# Pooling Token Attention 对照",
        "",
        f"- 模型：`{metadata.get('model_path', '')}`",
        f"- 数据：`{metadata.get('data_file', '')}`",
        f"- 样本 offset：{metadata.get('sample_offset', '')}",
        f"- 样本 index：{metadata.get('sample_index', '')}",
        f"- query generated index：{metadata.get('query_generated_index', '')}",
        f"- query token：`{metadata.get('query_token_text', '')}` ({metadata.get('query_token_id', '')})",
        f"- prompt token 数：{metadata.get('prompt_token_count', '')}",
        f"- block size：{metadata.get('block_size', '')}",
        f"- BOS attention mask：`{metadata.get('attention_mask_ablation', {}).get('enabled', False)}`",
        f"- 生成文本预览：`{metadata.get('generated_text_preview', '')}`",
        "",
        "## Max Pooling Top Blocks",
        "",
        *format_top_block_table(pooling_rows, "selected_by_max", "rank_by_max"),
        "",
        "## Avg Pooling Top Blocks",
        "",
        *format_top_block_table(pooling_rows, "selected_by_avg", "rank_by_avg"),
        "",
        "## Top Fine-Grained Tokens",
        "",
        "| rank | position | block | token | attention mean | attention max |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(fine_rows, start=1):
        lines.append(
            f"| {rank} | {row['position']} | {row['block_id']} | `{row['token_text']}` | "
            f"{row['full_attention_mean']:.8f} | {row['full_attention_max']:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    metadata: Dict[str, Any],
    comparison: Dict[str, List[Dict[str, Any]]],
    attention: np.ndarray,
    tokens: Sequence[Dict[str, Any]] | None = None,
) -> None:
    """写出 metadata、pooling 分数、细粒度分数、完整 attention 和摘要文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", metadata)
    if tokens is not None:
        write_jsonl(output_dir / "tokens.jsonl", tokens)
    write_jsonl(output_dir / "pooling_tokens.jsonl", comparison["pooling_tokens"])
    write_jsonl(output_dir / "fine_tokens.jsonl", comparison["fine_tokens"])
    write_jsonl(output_dir / "pooling_vs_fine_summary.jsonl", comparison["pooling_vs_fine_summary"])
    prompt_tokens = (
        [token for token in tokens if token.get("source") == "prompt"]
        if tokens is not None
        else comparison["fine_tokens"]
    )
    csv_rows = build_pooling_attention_csv_rows(
        attention=attention,
        prompt_tokens=prompt_tokens,
        block_size=int(metadata["block_size"]),
        metadata=metadata,
    )
    write_csv(output_dir / "pooling_attention_comparison.csv", csv_rows, POOLING_ATTENTION_CSV_FIELDS)
    np.savez_compressed(output_dir / "attention_detail.npz", attention=attention)
    write_summary_markdown(output_dir / "summary.md", metadata, comparison)


def build_metadata(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    prompt_token_count: int,
    generated_tokens: Sequence[Dict[str, Any]],
    attention: np.ndarray,
    generated_text: str = "",
    attention_mask_ablation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """构造本次对照实验的元数据。"""

    query_index = int(args.query_generated_index)
    query_token = generated_tokens[query_index]
    prompt_attention = attention[:, :, :prompt_token_count].astype(np.float64, copy=False)
    prompt_mass_by_layer_head = prompt_attention.sum(axis=2)
    return {
        "model_path": str(args.model_path),
        "data_file": str(args.data_file),
        "sample_offset": int(args.sample_offset),
        "sample_index": sample.get("index", args.sample_offset),
        "block_size": int(args.block_size),
        "top_k_blocks": int(args.top_k_blocks),
        "max_new_tokens": int(args.max_new_tokens),
        "query_generated_index": query_index,
        "query_token_id": int(query_token["token_id"]),
        "query_token_text": query_token["token_text"],
        "generated_text": generated_text,
        "generated_text_preview": clean_token_text(generated_text[:200]),
        "attention_mask_ablation": attention_mask_ablation or disabled_attention_mask_ablation(),
        "prompt_token_count": int(prompt_token_count),
        "generated_token_count": int(len(generated_tokens)),
        "attention_shape": [int(dim) for dim in attention.shape],
        "attention_semantics": (
            "attention_detail.npz 保存指定生成 query token 的 attention，形状为 "
            "[num_layers, num_heads, key_position]。pooling 分数只比较 prompt blocks。"
        ),
        "pooling_attention_csv_semantics": (
            "pooling_attention_comparison.csv 按 layer/head/block 展开；fine_attention_sum "
            "是不 pooling 时 block 内普通 token attention 之和，avg_pooling_attention 是均值，"
            "max_pooling_attention 是最大值。"
        ),
        "prompt_attention_mass_mean": float(prompt_mass_by_layer_head.mean()),
        "prompt_attention_mass_min": float(prompt_mass_by_layer_head.min()),
        "prompt_attention_mass_max": float(prompt_mass_by_layer_head.max()),
    }


def run_compare(args: argparse.Namespace) -> None:
    """执行单样本 pooling token 与 full attention 对照实验。"""

    prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    sample = read_jsonl_sample(args.data_file, args.sample_offset)
    prompt_text = sample["input"]

    tokenizer, model = load_llama_model(args.model_path)
    device = get_input_device(model)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if args.mask_bos_token:
        inputs, attention_mask_ablation = apply_bos_attention_mask(inputs, tokenizer)
    else:
        attention_mask_ablation = disabled_attention_mask_ablation()

    prompt_token_ids = inputs["input_ids"][0].detach().cpu().tolist()
    generated_token_ids = generate_tokens(model, inputs, max_new_tokens=args.max_new_tokens)
    if int(generated_token_ids.numel()) == 0:
        raise RuntimeError("模型没有生成新 token，无法对比 pooling attention")
    generated_text = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)

    prompt_attention_mask = inputs.get("attention_mask")
    prompt_tokens = build_token_records(
        tokenizer,
        prompt_token_ids,
        source="prompt",
        attention_mask_values=None
        if prompt_attention_mask is None
        else prompt_attention_mask[0].detach().cpu().tolist(),
    )
    generated_tokens = build_token_records(
        tokenizer,
        generated_token_ids.tolist(),
        source="generated",
        start_position=len(prompt_token_ids),
    )
    attention = replay_query_attention(
        model=model,
        inputs=inputs,
        generated_token_ids=generated_token_ids,
        query_generated_index=args.query_generated_index,
        dtype=args.dtype,
    )
    comparison = build_pooling_comparison(
        attention=attention,
        prompt_tokens=prompt_tokens,
        block_size=args.block_size,
        top_k_blocks=args.top_k_blocks,
        include_layer_head_details=not args.omit_layer_head_details,
    )
    metadata = build_metadata(
        args=args,
        sample=sample,
        prompt_token_count=len(prompt_token_ids),
        generated_tokens=generated_tokens,
        attention=attention,
        generated_text=generated_text,
        attention_mask_ablation=attention_mask_ablation,
    )
    write_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        comparison=comparison,
        attention=attention,
        tokens=[*prompt_tokens, *generated_tokens],
    )
    print(f"Pooling attention 对照结果已保存到 {args.output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="对比 max/avg pooling-token block 分数和细粒度 full attention。"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--query-generated-index", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k-blocks", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--omit-layer-head-details",
        action="store_true",
        help="不在 jsonl 中写入每层每 head 明细；仍会写出 attention_detail.npz。",
    )
    parser.add_argument(
        "--mask-bos-token",
        action="store_true",
        help="保留 BOS token，但将其 attention_mask 置 0，用于测试遮住 <|begin_of_text|> 的影响。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖非空输出目录。")
    return parser


def main() -> int:
    """命令行入口。"""

    args = build_parser().parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size 必须大于 0")
    if args.top_k_blocks <= 0:
        raise ValueError("--top-k-blocks 必须大于 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.query_generated_index < 0:
        raise ValueError("--query-generated-index 必须大于或等于 0")
    run_compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
