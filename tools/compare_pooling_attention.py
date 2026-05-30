#!/usr/bin/env python
"""对比 pooling token 分数和细粒度 full attention 分数。"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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

TRUE_POOLING_ATTENTION_CSV_FIELDS = [
    "query_generated_index",
    "token_range",
    "dense_attention_sum",
    "avg_pooling_attention",
    "max_pooling_attention",
    "layer",
    "head",
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


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """对一维 logits 做数值稳定 softmax。"""

    if values.ndim != 1:
        raise ValueError(f"softmax 输入必须是一维数组，实际为 {values.shape}")
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    shifted = values.astype(np.float64, copy=False) - float(np.max(values))
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total == 0.0:
        raise ValueError("softmax logits 产生了 0 分母")
    return exp_values / total


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


def repeat_key_value_heads(key_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """把 GQA/MQA key heads repeat 到 query head 数。"""

    if num_key_value_groups == 1:
        return key_states
    batch, num_key_value_heads, sequence_length, head_dim = key_states.shape
    key_states = key_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        sequence_length,
        head_dim,
    )
    return key_states.reshape(batch, num_key_value_heads * num_key_value_groups, sequence_length, head_dim)


def cache_key_for_layer(past_key_values: Any, layer_idx: int) -> torch.Tensor:
    """从 Transformers cache 或 tuple cache 中取某一层的 key tensor。"""

    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx]
    return past_key_values[layer_idx][0]


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


def find_llama_attention_modules(model) -> List[Any]:
    """返回带 q/k projection 的 Llama attention 模块。"""

    modules = []
    for module in model.modules():
        if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "head_dim", "num_key_value_groups")):
            modules.append(module)
    if not modules:
        raise RuntimeError("未找到可捕获 Q/K 的 Llama attention 模块")

    def layer_sort_key(item: Tuple[int, Any]) -> int:
        fallback, module = item
        layer_idx = getattr(module, "layer_idx", None)
        return fallback if layer_idx is None else int(layer_idx)

    return [module for _, module in sorted(enumerate(modules), key=layer_sort_key)]


def compute_query_states_from_capture(capture: Dict[str, Any]) -> torch.Tensor:
    """用目标 forward 的 attention 输入重算 RoPE 后 query states。"""

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    module = capture["module"]
    hidden_states = capture["hidden_states"]
    batch_size, query_length, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(batch_size, query_length, -1, module.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, query_length, -1, module.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, query_length, -1, module.head_dim).transpose(1, 2)

    position_embeddings = capture.get("position_embeddings")
    if position_embeddings is None:
        position_ids = capture.get("position_ids")
        cos, sin = module.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states[:, :, -1, :]


def collect_query_and_key_states(
    captures: Dict[int, Dict[str, Any]],
    past_key_values: Any,
    key_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """从目标 query forward 捕获 query states，并从 cache 中取全部可见 key states。"""

    if not captures:
        raise RuntimeError("未捕获到 attention 输入，无法计算真实 pooling attention")

    query_layers = []
    key_layers = []
    for layer_idx in sorted(captures):
        capture = captures[layer_idx]
        module = capture["module"]
        query_states = compute_query_states_from_capture(capture)
        key_states = cache_key_for_layer(past_key_values, layer_idx)
        key_states = key_states[:, :, :key_length, :]
        key_states = repeat_key_value_heads(key_states, int(module.num_key_value_groups))
        query_layers.append(query_states[0].float().detach().cpu())
        key_layers.append(key_states[0].float().detach().cpu())

    query_array = torch.stack(query_layers, dim=0).numpy()
    key_array = torch.stack(key_layers, dim=0).numpy()
    return query_array.astype(np.float32), key_array.astype(np.float32)


def register_attention_input_hooks(model) -> Tuple[Dict[int, Dict[str, Any]], List[Any]]:
    """注册 forward pre-hook，用于捕获每层目标 query 的 attention 输入。"""

    captures: Dict[int, Dict[str, Any]] = {}
    handles = []

    def capture_input(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            position_embeddings = tuple(item.detach() for item in position_embeddings)
        layer_idx = int(getattr(module, "layer_idx", len(captures)))
        captures[layer_idx] = {
            "module": module,
            "hidden_states": hidden_states.detach(),
            "position_ids": None
            if kwargs.get("position_ids") is None
            else kwargs["position_ids"].detach(),
            "position_embeddings": position_embeddings,
        }

    for module in find_llama_attention_modules(model):
        handles.append(module.register_forward_pre_hook(capture_input, with_kwargs=True))
    return captures, handles


def remove_hooks(handles: Sequence[Any]) -> None:
    """移除一组 PyTorch hook。"""

    for handle in handles:
        handle.remove()


def replay_query_attention(
    model,
    inputs: Dict[str, torch.Tensor],
    generated_token_ids: torch.Tensor,
    query_generated_index: int,
    dtype: str,
) -> np.ndarray:
    """用 KV cache replay 到指定生成 token，并捕获该 token 的完整 attention。"""

    result = replay_query_attention_and_states(
        model=model,
        inputs=inputs,
        generated_token_ids=generated_token_ids,
        query_generated_index=query_generated_index,
        dtype=dtype,
        capture_qk=False,
    )
    return result["attention"]


def replay_query_attention_and_states(
    model,
    inputs: Dict[str, torch.Tensor],
    generated_token_ids: torch.Tensor,
    query_generated_index: int,
    dtype: str,
    capture_qk: bool = False,
) -> Dict[str, Any]:
    """replay 到指定生成 token，并可额外捕获真实 pooling 所需 Q/K states。"""

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
        query_states = None
        key_states = None
        for generated_index, token_id in enumerate(generated_token_ids[: query_generated_index + 1].tolist()):
            current_token = torch.tensor([[int(token_id)]], device=get_input_device(model), dtype=torch.long)
            if attention_mask is not None:
                next_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, next_mask], dim=1)

            captures: Dict[int, Dict[str, Any]] = {}
            handles: List[Any] = []
            if capture_qk and generated_index == query_generated_index:
                captures, handles = register_attention_input_hooks(model)
            try:
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
            finally:
                if handles:
                    remove_hooks(handles)
            if generated_index == query_generated_index:
                query_attention = stack_step_attentions(outputs.attentions, dtype=dtype)
                if capture_qk:
                    key_length = int(query_attention.shape[-1])
                    query_states, key_states = collect_query_and_key_states(
                        captures=captures,
                        past_key_values=outputs.past_key_values,
                        key_length=key_length,
                    )
            past_key_values = outputs.past_key_values

    if query_attention is None:
        raise RuntimeError("未捕获到 query attention")
    result: Dict[str, Any] = {"attention": query_attention}
    if capture_qk:
        if query_states is None or key_states is None:
            raise RuntimeError("未捕获到真实 pooling 所需的 Q/K states")
        result["query_states"] = query_states
        result["key_states"] = key_states
    return result


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


def validate_true_pooling_shapes(
    dense_attention: np.ndarray,
    query_states: np.ndarray,
    key_states: np.ndarray,
) -> None:
    """校验真实 pooling attention 的 Q/K/dense attention 形状。"""

    if dense_attention.ndim != 3:
        raise ValueError(f"dense_attention 必须是 [layer, head, key_position]，实际为 {dense_attention.shape}")
    if query_states.ndim != 3:
        raise ValueError(f"query_states 必须是 [layer, head, head_dim]，实际为 {query_states.shape}")
    if key_states.ndim != 4:
        raise ValueError(f"key_states 必须是 [layer, head, key_position, head_dim]，实际为 {key_states.shape}")
    if dense_attention.shape[:2] != query_states.shape[:2]:
        raise ValueError("dense_attention 和 query_states 的 layer/head 维度不一致")
    if dense_attention.shape != key_states.shape[:3]:
        raise ValueError("dense_attention 和 key_states 的 layer/head/key_position 维度不一致")
    if query_states.shape[2] != key_states.shape[3]:
        raise ValueError("query_states 和 key_states 的 head_dim 不一致")


def build_true_pooling_attention_csv_rows(
    dense_attention: np.ndarray,
    query_states: np.ndarray,
    key_states: np.ndarray,
    block_size: int,
    query_generated_index: int,
) -> List[Dict[str, Any]]:
    """按真实 QK pooling 重新计算 block attention，并输出 7 列 CSV 行。"""

    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    validate_true_pooling_shapes(dense_attention, query_states, key_states)

    dense_attention = dense_attention.astype(np.float64, copy=False)
    query_states = query_states.astype(np.float64, copy=False)
    key_states = key_states.astype(np.float64, copy=False)
    key_length = dense_attention.shape[-1]
    blocks = build_prompt_blocks(key_length, block_size)
    head_dim = int(query_states.shape[-1])
    rows: List[Dict[str, Any]] = []

    for layer in range(dense_attention.shape[0]):
        for head in range(dense_attention.shape[1]):
            avg_logits = []
            max_logits = []
            for block in blocks:
                start = block["start_position"]
                end_exclusive = block["end_position"] + 1
                block_keys = key_states[layer, head, start:end_exclusive, :]
                avg_key = block_keys.mean(axis=0)
                max_key = block_keys.max(axis=0)
                avg_logits.append(float(np.dot(query_states[layer, head], avg_key) / math.sqrt(head_dim)))
                max_logits.append(float(np.dot(query_states[layer, head], max_key) / math.sqrt(head_dim)))

            avg_attention = stable_softmax(np.array(avg_logits, dtype=np.float64))
            max_attention = stable_softmax(np.array(max_logits, dtype=np.float64))

            for block_index, block in enumerate(blocks):
                start = block["start_position"]
                end_exclusive = block["end_position"] + 1
                rows.append(
                    {
                        "query_generated_index": int(query_generated_index),
                        "token_range": f"{start + 1}-{block['end_position'] + 1}",
                        "dense_attention_sum": compact_float(
                            dense_attention[layer, head, start:end_exclusive].sum()
                        ),
                        "avg_pooling_attention": compact_float(avg_attention[block_index]),
                        "max_pooling_attention": compact_float(max_attention[block_index]),
                        "layer": int(layer),
                        "head": int(head),
                    }
                )

    return rows


def summarize_true_pooling_attention_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """统计真实 pooling CSV 各 attention 列在 query/layer/head 下的归一化误差。"""

    grouped: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for row in rows:
        key = (int(row["query_generated_index"]), int(row["layer"]), int(row["head"]))
        values = grouped.setdefault(
            key,
            {
                "dense_attention_sum": 0.0,
                "avg_pooling_attention": 0.0,
                "max_pooling_attention": 0.0,
            },
        )
        values["dense_attention_sum"] += float(row["dense_attention_sum"])
        values["avg_pooling_attention"] += float(row["avg_pooling_attention"])
        values["max_pooling_attention"] += float(row["max_pooling_attention"])

    def max_error(column: str) -> float:
        if not grouped:
            return 0.0
        return max(abs(values[column] - 1.0) for values in grouped.values())

    return {
        "row_count": int(len(rows)),
        "group_count": int(len(grouped)),
        "dense_attention_sum_max_error": float(max_error("dense_attention_sum")),
        "avg_pooling_attention_sum_max_error": float(max_error("avg_pooling_attention")),
        "max_pooling_attention_sum_max_error": float(max_error("max_pooling_attention")),
    }


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
    true_pooling_rows: Sequence[Dict[str, Any]] | None = None,
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
    if true_pooling_rows is not None:
        true_pooling_file = output_dir / f"true_pooling_attention_7col_block{int(metadata['block_size'])}.csv"
        write_csv(true_pooling_file, true_pooling_rows, TRUE_POOLING_ATTENTION_CSV_FIELDS)
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
    replay_result = replay_query_attention_and_states(
        model=model,
        inputs=inputs,
        generated_token_ids=generated_token_ids,
        query_generated_index=args.query_generated_index,
        dtype=args.dtype,
        capture_qk=args.true_pooling_attention,
    )
    attention = replay_result["attention"]
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
    metadata["key_token_count"] = int(attention.shape[-1])
    true_pooling_rows = None
    if args.true_pooling_attention:
        true_pooling_rows = build_true_pooling_attention_csv_rows(
            dense_attention=attention,
            query_states=replay_result["query_states"],
            key_states=replay_result["key_states"],
            block_size=args.block_size,
            query_generated_index=args.query_generated_index,
        )
        true_pooling_summary = summarize_true_pooling_attention_rows(true_pooling_rows)
        metadata["true_pooling_attention_csv"] = (
            f"true_pooling_attention_7col_block{int(args.block_size)}.csv"
        )
        metadata["true_pooling_attention_csv_fields"] = TRUE_POOLING_ATTENTION_CSV_FIELDS
        metadata["true_pooling_attention_csv_semantics"] = (
            "按 query_generated_index/layer/head 展开；token_range 覆盖当前 query 可见的全部 key tokens，"
            "即 prompt tokens 加已生成到当前 query 的 generated tokens。dense_attention_sum 是原始 dense "
            "attention 在该 block 内的和；avg_pooling_attention 是 block 内 K 向量平均后重新 QK softmax "
            "得到的 block attention；max_pooling_attention 是 block 内 K 向量逐维 max 后重新 QK softmax "
            "得到的 block attention。"
        )
        metadata["true_pooling_attention_summary"] = true_pooling_summary
    write_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        comparison=comparison,
        attention=attention,
        tokens=[*prompt_tokens, *generated_tokens],
        true_pooling_rows=true_pooling_rows,
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
    parser.add_argument(
        "--true-pooling-attention",
        action="store_true",
        help="额外用目标 query 的 Q/K states 对全部可见 key tokens 做真实 avg-key/max-key pooling attention。",
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
