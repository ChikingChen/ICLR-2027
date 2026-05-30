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

import logging
import math
import json
import requests
import time
import torch
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


ATTENTION_KERNEL_NAME_PARTS = (
    "flash_attn",
    "flashattention",
    "scaled_dot_product",
    "fmha",
    "sdpa",
    "attention",
)

ATTENTION_PROFILE_GENERATION_KWARGS_TO_DROP = (
    "return_dict_in_generate",
    "output_scores",
    "output_logits",
    "output_attentions",
    "output_hidden_states",
)


def ensure_num_hidden_layers_alias(config) -> None:
    """为只提供 `num_layers` 的模型配置补充 Transformers 新缓存接口需要的别名。"""

    if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
        config.num_hidden_layers = config.num_layers


def decode_attention_token(tokenizer, token_id: int) -> str:
    """把单个 token id 解码成适合写入表格的一行短文本。"""

    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        text = str(token_id)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text if text else "<empty>"


def mask_bos_inputs_for_generation(inputs, tokenizer) -> tuple:
    """保留 BOS token，但让后续生成不能 attend 到 position 0。"""

    if "input_ids" not in inputs:
        raise ValueError("inputs must contain input_ids")
    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("mask_bos_token only supports batch_size=1 inputs")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("tokenizer does not define bos_token_id")

    first_token_id = int(input_ids[0, 0].item())
    if first_token_id != int(bos_token_id):
        raise ValueError(
            f"expected BOS token id {bos_token_id} at position 0, found {first_token_id}"
        )

    masked_inputs = dict(inputs)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.clone()
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids shape")

    attention_mask[0, 0] = 0
    position_ids = torch.arange(
        input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    masked_inputs["attention_mask"] = attention_mask
    masked_inputs["position_ids"] = position_ids

    return masked_inputs, {
        "enabled": True,
        "mode": "bos_token",
        "token_position": 0,
        "token_id": first_token_id,
        "token_text": decode_attention_token(tokenizer, first_token_id),
        "semantics": "token kept in input_ids; attention_mask is 0 during generation",
    }


def summarize_attention_layers(
    attentions: Sequence[torch.Tensor],
    token_ids: Sequence[int],
    tokenizer,
    top_k: int,
) -> List[dict]:
    """把逐层注意力张量压缩为每层 Top-K token 摘要。"""

    layers: List[dict] = []
    if not attentions:
        return layers

    for layer_idx, attention in enumerate(attentions):
        if attention is None:
            continue
        weights = attention.detach().float().cpu()
        if weights.dim() == 4:
            vector = weights[0, :, -1, :].mean(dim=0)
        elif weights.dim() == 3:
            vector = weights[0, -1, :]
        else:
            continue

        usable_length = min(vector.numel(), len(token_ids))
        if usable_length == 0:
            continue
        vector = vector[:usable_length]
        layer_token_ids = list(token_ids[:usable_length])
        total = float(vector.sum().item())
        if total > 0:
            vector = vector / total
        top_count = min(top_k, vector.numel())
        top_scores, top_positions = torch.topk(vector, k=top_count)

        top_tokens = []
        for rank, (score, position) in enumerate(zip(top_scores.tolist(), top_positions.tolist()), start=1):
            token_id = int(layer_token_ids[position])
            top_tokens.append(
                {
                    "rank": rank,
                    "position": int(position),
                    "token_id": token_id,
                    "token": decode_attention_token(tokenizer, token_id),
                    "score": float(score),
                }
            )

        layers.append(
            {
                "layer": layer_idx,
                "sum": float(vector.sum().item()),
                "top_tokens": top_tokens,
            }
        )

    return layers


def compute_generation_ppl_stats(
    scores: Sequence[torch.Tensor],
    generated_token_ids: torch.Tensor,
    tokenizer=None,
    include_token_details: bool = False,
) -> dict:
    """基于 generate 返回的逐步 scores 计算单条样本的生成 token PPL。"""

    if generated_token_ids.dim() == 2:
        if generated_token_ids.shape[0] != 1:
            raise ValueError("compute_generation_ppl_stats 只接受单条样本的生成 token。")
        generated_token_ids = generated_token_ids[0]

    token_count = min(len(scores), int(generated_token_ids.numel()))
    if token_count == 0:
        stats = {
            "generation_logprob_sum": 0.0,
            "generation_token_count": 0,
            "generation_nll": None,
            "generation_ppl": None,
        }
        if include_token_details:
            stats["generation_tokens"] = []
        return stats

    logprob_sum = 0.0
    token_details = []
    for step_idx in range(token_count):
        step_scores = scores[step_idx]
        if step_scores.dim() == 2:
            step_scores = step_scores[0]
        token_id = int(generated_token_ids[step_idx].item())
        step_logprobs = torch.log_softmax(step_scores.float(), dim=-1)
        token_logprob = float(step_logprobs[token_id].item())
        logprob_sum += token_logprob
        if include_token_details:
            token_text = str(token_id)
            if tokenizer is not None:
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                except Exception:
                    token_text = str(token_id)
            token_nll = -token_logprob
            token_details.append(
                {
                    "position": step_idx,
                    "token_id": token_id,
                    "token": token_text,
                    "logprob": token_logprob,
                    "nll": token_nll,
                    "ppl": math.exp(token_nll),
                }
            )

    nll = -logprob_sum / token_count
    stats = {
        "generation_logprob_sum": logprob_sum,
        "generation_token_count": token_count,
        "generation_nll": nll,
        "generation_ppl": math.exp(nll),
    }
    if include_token_details:
        stats["generation_tokens"] = token_details
    return stats


def compute_batch_generation_ppl_stats(
    scores: Sequence[torch.Tensor],
    generated_token_ids: torch.Tensor,
    tokenizer=None,
    include_token_details: bool = False,
) -> List[dict]:
    """基于 generate scores 为 batch 中每条样本分别计算生成 token PPL。"""

    stats = []
    for sample_idx in range(generated_token_ids.shape[0]):
        sample_scores = [step_scores[sample_idx:sample_idx + 1] for step_scores in scores]
        stats.append(
            compute_generation_ppl_stats(
                sample_scores,
                generated_token_ids[sample_idx:sample_idx + 1],
                tokenizer=tokenizer,
                include_token_details=include_token_details,
            )
        )
    return stats


def summarize_generation_timing(
    prefill_forward_ms: Sequence[float],
    decode_forward_ms: Sequence[float],
    generated_token_count: int,
    timer_backend: str,
) -> dict:
    """把一次生成期间采集到的 prefill/decode forward 耗时聚合成稳定字段。"""

    prefill_total = sum(float(value) for value in prefill_forward_ms)
    decode_total = sum(float(value) for value in decode_forward_ms)
    if generated_token_count > 0:
        decode_per_token = decode_total / generated_token_count
    else:
        decode_per_token = None
    return {
        "timer_backend": timer_backend,
        "prefill_forward_ms": prefill_total if prefill_forward_ms else None,
        "decode_forward_ms_total": decode_total if decode_forward_ms else 0.0,
        "decode_forward_ms_per_token_avg": decode_per_token,
        "decode_steps": len(decode_forward_ms),
        "generated_token_count": int(generated_token_count),
    }


def _event_device_time_us(event: Any) -> float:
    """从 torch profiler event 中提取设备耗时，单位微秒。"""

    for field in ("device_time_total", "cuda_time_total", "self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, field, None)
        if value:
            return float(value)
    return 0.0


def summarize_attention_kernel_events(events: Sequence[Any]) -> dict:
    """统计 profiler 中名称像 attention kernel 的设备事件总耗时。"""

    total_us = 0.0
    event_count = 0
    for event in events:
        name = str(getattr(event, "name", getattr(event, "key", ""))).lower()
        if not any(part in name for part in ATTENTION_KERNEL_NAME_PARTS):
            continue
        device_time_us = _event_device_time_us(event)
        if device_time_us <= 0:
            continue
        total_us += device_time_us
        event_count += 1

    summary = {
        "attention_kernel_event_count": event_count,
        "attention_kernel_ms": total_us / 1000.0 if event_count else None,
    }
    if event_count == 0:
        summary["warning"] = "profiler 没有匹配到 attention 相关 CUDA kernel。"
    return summary


def _first_cuda_device(args: Sequence[Any], kwargs: Dict[str, Any]) -> Optional[torch.device]:
    """从 forward 参数中找到第一个 CUDA tensor 的 device。"""

    values = list(args) + list(kwargs.values())
    while values:
        value = values.pop(0)
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value.device
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (list, tuple)):
            values.extend(value)
    return None


def measure_callable_ms(
    func: Callable,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> tuple:
    """执行 callable 并返回结果、毫秒耗时和计时后端。"""

    device = _first_cuda_device(args, kwargs)
    if torch.cuda.is_available() and device is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        return result, float(start_event.elapsed_time(end_event)), "cuda_event"

    started_at = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return result, elapsed_ms, "perf_counter"


def _callable_globals(func: Callable) -> Optional[Dict[str, Any]]:
    """Return the globals dict behind a function or bound method."""

    raw_func = getattr(func, "__func__", func)
    globals_dict = getattr(raw_func, "__globals__", None)
    if isinstance(globals_dict, dict):
        return globals_dict
    return None


class CudaEventAttentionTimer:
    """Record attention op time by wrapping known attention call sites with CUDA events."""

    def __init__(self, model=None) -> None:
        self.model = model
        self._patches: List[tuple] = []
        self._patched_keys = set()
        self._records: List[dict] = []
        self._recording_depth = 0
        self.timer_backend = (
            "cuda_event_attention_ops"
            if torch.cuda.is_available()
            else "perf_counter_attention_ops"
        )

    def __enter__(self):
        if self.model is not None:
            self.patch_model(self.model)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False

    def patch_function(self, namespace: Dict[str, Any], name: str, label: Optional[str] = None) -> bool:
        """Temporarily wrap `namespace[name]` and record each successful call."""

        if namespace is None or name not in namespace:
            return False
        original = namespace.get(name)
        if not callable(original):
            return False

        key = (id(namespace), name)
        if key in self._patched_keys:
            return False

        call_label = label or name

        @wraps(original)
        def wrapped(*args, **kwargs):
            return self._record_call(call_label, original, args, kwargs)

        self._patched_keys.add(key)
        self._patches.append((namespace, name, original))
        namespace[name] = wrapped
        return True

    def patch_model(self, model) -> int:
        """Patch common HF/GLM attention functions reachable from model modules."""

        patched_count = 0
        patched_count += int(
            self.patch_function(
                torch.nn.functional.__dict__,
                "scaled_dot_product_attention",
                "scaled_dot_product_attention",
            )
        )

        modules = model.modules() if hasattr(model, "modules") else []
        for submodule in modules:
            globals_dict = _callable_globals(getattr(submodule, "forward", None))
            if globals_dict is None:
                continue

            # HF Llama/Qwen FlashAttention classes call this helper. Timing this
            # helper avoids double-counting nested flash_attn_func calls.
            if callable(globals_dict.get("_flash_attention_forward")):
                patched_count += int(
                    self.patch_function(
                        globals_dict,
                        "_flash_attention_forward",
                        "_flash_attention_forward",
                    )
                )
                continue

            for name in ("flash_attn_func", "flash_attn_varlen_func", "scaled_dot_product_attention"):
                patched_count += int(self.patch_function(globals_dict, name, name))

        return patched_count

    def restore(self) -> None:
        """Restore all wrapped functions in reverse patch order."""

        while self._patches:
            namespace, name, original = self._patches.pop()
            namespace[name] = original
        self._patched_keys.clear()

    def _record_call(self, label: str, func: Callable, args: Sequence[Any], kwargs: Dict[str, Any]):
        if self._recording_depth > 0:
            return func(*args, **kwargs)

        device = _first_cuda_device(args, kwargs)
        if torch.cuda.is_available() and device is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            self._recording_depth += 1
            with torch.cuda.device(device):
                start_event.record(torch.cuda.current_stream(device))
            try:
                result = func(*args, **kwargs)
            except Exception:
                raise
            else:
                with torch.cuda.device(device):
                    end_event.record(torch.cuda.current_stream(device))
                self._records.append(
                    {
                        "label": label,
                        "device": device,
                        "start_event": start_event,
                        "end_event": end_event,
                        "backend": "cuda_event_attention_ops",
                    }
                )
                return result
            finally:
                self._recording_depth -= 1

        started_at = time.perf_counter()
        self._recording_depth += 1
        try:
            result = func(*args, **kwargs)
        except Exception:
            raise
        else:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._records.append(
                {
                    "label": label,
                    "elapsed_ms": elapsed_ms,
                    "backend": "perf_counter_attention_ops",
                }
            )
            return result
        finally:
            self._recording_depth -= 1

    def summary(self) -> dict:
        cuda_devices = {}
        for record in self._records:
            if record.get("backend") == "cuda_event_attention_ops":
                device = record.get("device")
                cuda_devices[str(device)] = device

        for device in cuda_devices.values():
            torch.cuda.synchronize(device)

        total_ms = 0.0
        for record in self._records:
            if record.get("backend") == "cuda_event_attention_ops":
                elapsed_ms = float(record["start_event"].elapsed_time(record["end_event"]))
            else:
                elapsed_ms = float(record.get("elapsed_ms", 0.0))
            record["elapsed_ms"] = elapsed_ms
            total_ms += elapsed_ms

        event_count = len(self._records)
        backend = (
            "cuda_event_attention_ops"
            if any(record.get("backend") == "cuda_event_attention_ops" for record in self._records)
            else self.timer_backend
        )
        summary = {
            "timer_backend": backend,
            "attention_kernel_event_count": event_count,
            "attention_kernel_ms": total_ms if event_count else None,
        }
        if event_count == 0:
            summary["warning"] = "CUDA event attention timer 没有捕获到 attention op 调用。"
        return summary


class ForwardTimingCollector:
    """在 Hugging Face `generate()` 期间按 prefill/decode 记录 forward 耗时。"""

    def __init__(self) -> None:
        self.prefill_forward_ms: List[float] = []
        self.decode_forward_ms: List[float] = []
        self.timer_backend = "unknown"

    def classify(self, kwargs: Dict[str, Any]) -> str:
        """根据 KV cache 和调用顺序判断当前 forward 属于 prefill 还是 decode。"""

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and not self.prefill_forward_ms:
            return "prefill"
        return "decode"

    def record(self, phase: str, elapsed_ms: float, timer_backend: str) -> None:
        """记录一次 forward 耗时。"""

        self.timer_backend = timer_backend
        if phase == "prefill":
            self.prefill_forward_ms.append(float(elapsed_ms))
        else:
            self.decode_forward_ms.append(float(elapsed_ms))

    def summary(self, generated_token_count: int) -> dict:
        """返回当前采集到的稳定 timing 字段。"""

        return summarize_generation_timing(
            prefill_forward_ms=self.prefill_forward_ms,
            decode_forward_ms=self.decode_forward_ms,
            generated_token_count=generated_token_count,
            timer_backend=self.timer_backend,
        )


class AttentionTimingCollector:
    """在 Hugging Face `generate()` 期间按 prefill/decode 记录 attention op 耗时。"""

    def __init__(self) -> None:
        self.prefill_attention_ms: List[float] = []
        self.decode_attention_ms: List[float] = []
        self.attention_kernel_event_count = 0
        self.timer_backend = "unknown"
        self.warnings: List[str] = []

    def classify(self, kwargs: Dict[str, Any]) -> str:
        """根据 KV cache 和调用顺序判断当前 forward 属于 prefill 还是 decode。"""

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and not self.prefill_attention_ms:
            return "prefill"
        return "decode"

    def record(self, phase: str, summary: Mapping[str, Any]) -> None:
        """记录一次 forward 内部 attention op 的 CUDA event 汇总。"""

        self.timer_backend = str(summary.get("timer_backend") or self.timer_backend)
        self.attention_kernel_event_count += int(summary.get("attention_kernel_event_count") or 0)
        attention_ms = summary.get("attention_kernel_ms")
        if attention_ms is not None:
            if phase == "prefill":
                self.prefill_attention_ms.append(float(attention_ms))
            else:
                self.decode_attention_ms.append(float(attention_ms))
        if summary.get("warning"):
            warning = str(summary["warning"])
            if warning not in self.warnings:
                self.warnings.append(warning)

    def summary(self, generated_token_count: int, input_tokens: int) -> dict:
        """返回当前样本的 attention timing 字段。"""

        prefill_total = sum(self.prefill_attention_ms) if self.prefill_attention_ms else None
        decode_total = sum(self.decode_attention_ms) if self.decode_attention_ms else 0.0
        decode_per_token = None
        if generated_token_count > 0 and decode_total is not None:
            decode_per_token = decode_total / generated_token_count
        result = {
            "timer_backend": self.timer_backend,
            "input_tokens": int(input_tokens),
            "generated_token_count": int(generated_token_count),
            "prefill_attention_kernel_ms": prefill_total,
            "decode_attention_kernel_ms_total": decode_total,
            "decode_attention_kernel_ms_per_token_avg": decode_per_token,
            "attention_kernel_event_count": self.attention_kernel_event_count,
        }
        if self.warnings:
            result["warning"] = "；".join(self.warnings)
        return result


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.log_attention_scores = bool(generation_kwargs.pop("log_attention_scores", False))
        self.log_generation_ppl = bool(generation_kwargs.pop("log_generation_ppl", False))
        self.log_generation_token_ppl = bool(generation_kwargs.pop("log_generation_token_ppl", False))
        self.log_prefill_decode_timing = bool(generation_kwargs.pop("log_prefill_decode_timing", False))
        self.profile_attention_kernels = bool(generation_kwargs.pop("profile_attention_kernels", False))
        self.mask_bos_token = bool(generation_kwargs.pop("mask_bos_token", False))
        self.log_attn_implementation = bool(generation_kwargs.pop("log_attn_implementation", False))
        if self.log_generation_token_ppl:
            self.log_generation_ppl = True
        self.attention_top_k = int(generation_kwargs.pop("attention_top_k", 8))
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if self.log_attention_scores:
            model_kwargs = {"attn_implementation": "eager"}
        elif 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        if (
            self.log_attention_scores
            or self.log_generation_ppl
            or self.log_prefill_decode_timing
            or self.profile_attention_kernels
            or self.mask_bos_token
        ):
            self.pipeline = None
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    **model_kwargs,
                )
            except (TypeError, ValueError):
                self.model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
        else:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
            except:
                self.pipeline = None
                self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)

        if self.pipeline is not None:
            ensure_num_hidden_layers_alias(self.pipeline.model.config)
            active_model = self.pipeline.model
        else:
            ensure_num_hidden_layers_alias(self.model.config)
            active_model = self.model
        if self.log_attn_implementation:
            print(
                "[ATTN_IMPLEMENTATION] model.config._attn_implementation="
                f"{getattr(active_model.config, '_attn_implementation', None)}",
                flush=True,
            )
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        if self.log_generation_ppl:
            self.generation_kwargs["return_dict_in_generate"] = True
            self.generation_kwargs["output_scores"] = True

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def _prepare_generation_inputs(self, prompts: List[str], padding: bool) -> tuple:
        """Tokenize prompts and optionally apply the BOS attention-mask ablation."""

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=padding).to(self.model.device)
        if not self.mask_bos_token:
            return inputs, None
        return mask_bos_inputs_for_generation(inputs, self.tokenizer)

    def _inputs_for_generate(self, inputs) -> dict:
        """Return generate kwargs, avoiding position_ids validation failures."""

        if not self.mask_bos_token or "position_ids" not in inputs:
            return inputs
        generate_inputs = dict(inputs)
        generate_inputs.pop("position_ids", None)
        return generate_inputs

    @contextmanager
    def _preserve_positions_for_masked_bos_generation(self):
        """Inject position_ids from cache_position while BOS is masked."""

        if not self.mask_bos_token:
            yield
            return

        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_inputs_for_generation(
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            **kwargs,
        ):
            kwargs.pop("position_ids", None)
            model_inputs = original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )
            input_tensor = model_inputs.get("input_ids")
            if input_tensor is None:
                input_tensor = model_inputs.get("inputs_embeds")
            if input_tensor is None:
                return model_inputs

            current_length = int(input_tensor.shape[1])
            if cache_position is None:
                position_ids = torch.arange(
                    current_length,
                    dtype=torch.long,
                    device=input_tensor.device,
                ).unsqueeze(0)
            else:
                position_ids = cache_position.reshape(-1)[-current_length:].to(
                    dtype=torch.long,
                    device=input_tensor.device,
                ).unsqueeze(0)
            model_inputs["position_ids"] = position_ids
            return model_inputs

        self.model.prepare_inputs_for_generation = prepare_inputs_for_generation
        try:
            yield
        finally:
            self.model.prepare_inputs_for_generation = original_prepare

    def _generate_with_runtime_timing(self, inputs) -> tuple:
        """运行 `generate()` 并按需采集 forward 和 attention timing。"""

        forward_collector = ForwardTimingCollector() if self.log_prefill_decode_timing else None
        attention_collector = AttentionTimingCollector() if self.profile_attention_kernels else None
        original_forward = self.model.forward

        def timed_forward(*forward_args, **forward_kwargs):
            classifier = forward_collector or attention_collector
            phase = classifier.classify(forward_kwargs) if classifier is not None else "decode"

            def run_forward():
                if forward_collector is None:
                    return original_forward(*forward_args, **forward_kwargs)
                result, elapsed_ms, timer_backend = measure_callable_ms(
                    original_forward,
                    forward_args,
                    forward_kwargs,
                )
                forward_collector.record(phase, elapsed_ms, timer_backend)
                return result

            if attention_collector is None:
                return run_forward()

            with CudaEventAttentionTimer(self.model) as timer:
                result = run_forward()
            attention_collector.record(phase, timer.summary())
            return result

        self.model.forward = timed_forward
        try:
            with self._preserve_positions_for_masked_bos_generation():
                generated_output = self.model.generate(
                    **self._inputs_for_generate(inputs),
                    **self.generation_kwargs,
                )
        finally:
            self.model.forward = original_forward
        return generated_output, forward_collector, attention_collector

    def _generate_with_forward_timing(self, inputs) -> tuple:
        """运行 `generate()` 并用临时 forward wrapper 采集 prefill/decode 耗时。"""

        generated_output, collector, _ = self._generate_with_runtime_timing(inputs)
        return generated_output, collector

    def _generate_without_timing(self, inputs):
        """运行普通 Hugging Face generate，便于和计时路径共用后处理。"""

        with self._preserve_positions_for_masked_bos_generation():
            return self.model.generate(
                **self._inputs_for_generate(inputs),
                **self.generation_kwargs,
            )

    def _generation_kwargs_for_attention_profile(self) -> dict:
        """Return generation kwargs that avoid retaining score/logit tensors."""

        profile_kwargs = dict(self.generation_kwargs)
        for key in ATTENTION_PROFILE_GENERATION_KWARGS_TO_DROP:
            profile_kwargs.pop(key, None)
        profile_kwargs["num_logits_to_keep"] = 1
        return profile_kwargs

    def _generate_for_attention_profile(self, inputs):
        """Generate only token ids for attention replay, without score retention."""

        profile_kwargs = self._generation_kwargs_for_attention_profile()
        try:
            with self._preserve_positions_for_masked_bos_generation():
                return self.model.generate(
                    **self._inputs_for_generate(inputs),
                    **profile_kwargs,
                )
        except (TypeError, ValueError) as error:
            if "num_logits_to_keep" not in str(error):
                raise
            profile_kwargs.pop("num_logits_to_keep", None)
            with self._preserve_positions_for_masked_bos_generation():
                return self.model.generate(
                    **self._inputs_for_generate(inputs),
                    **profile_kwargs,
                )

    def _model_forward_for_attention_profile(self, model_inputs: Dict[str, Any]):
        """Run model forward with smallest useful logits output when supported."""

        try:
            return self.model(**model_inputs, num_logits_to_keep=1)
        except (TypeError, ValueError) as error:
            if "num_logits_to_keep" not in str(error):
                raise
            return self.model(**model_inputs)

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.log_attention_scores:
            return [self._process_one_with_attention(prompt) for prompt in prompts]

        timing_collector = None
        attention_timing_collector = None
        attention_mask_ablation = None
        if self.pipeline is None:
            if (self.log_prefill_decode_timing or self.profile_attention_kernels) and len(prompts) != 1:
                raise ValueError("prefill/decode timing 和 attention timing 只支持 batch_size=1。")
            inputs, attention_mask_ablation = self._prepare_generation_inputs(prompts, padding=True)
            if self.log_prefill_decode_timing or self.profile_attention_kernels:
                generated_output, timing_collector, attention_timing_collector = self._generate_with_runtime_timing(inputs)
            else:
                generated_output = self._generate_without_timing(inputs)
            if hasattr(generated_output, "sequences"):
                generated_ids = generated_output.sequences
            else:
                generated_ids = generated_output
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []
        ppl_stats = None
        if self.log_generation_ppl:
            input_length = inputs["input_ids"].shape[1]
            generated_token_ids = generated_ids[:, input_length:]
            ppl_stats = compute_batch_generation_ppl_stats(
                generated_output.scores,
                generated_token_ids,
                tokenizer=self.tokenizer,
                include_token_details=self.log_generation_token_ppl,
            )

        for result_idx, (text, prompt) in enumerate(zip(generated_texts, prompts)):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            result = {'text': [text]}
            if ppl_stats is not None:
                result.update(ppl_stats[result_idx])
            if timing_collector is not None:
                input_length = inputs["input_ids"].shape[1]
                generated_token_count = int(generated_ids[result_idx, input_length:].numel())
                timing = timing_collector.summary(generated_token_count=generated_token_count)
                timing["input_tokens"] = int(input_length)
                result["generation_timing"] = timing
            if attention_timing_collector is not None:
                input_length = inputs["input_ids"].shape[1]
                generated_token_count = int(generated_ids[result_idx, input_length:].numel())
                result["attention_profile"] = attention_timing_collector.summary(
                    generated_token_count=generated_token_count,
                    input_tokens=int(input_length),
                )
            if attention_mask_ablation is not None:
                result["attention_mask_ablation"] = attention_mask_ablation
            results.append(result)

        return results

    def _process_one_with_attention(self, prompt: str) -> dict:
        """生成单条样本，并附带首个生成 token 的逐层注意力摘要。"""

        inputs, attention_mask_ablation = self._prepare_generation_inputs([prompt], padding=False)
        input_length = inputs["input_ids"].shape[1]
        timing_collector = None
        if self.log_prefill_decode_timing:
            generated_ids, timing_collector = self._generate_with_forward_timing(inputs)
        else:
            generated_ids = self._generate_without_timing(inputs)
        if hasattr(generated_ids, "sequences"):
            sequences = generated_ids.sequences
        else:
            sequences = generated_ids

        new_token_ids = sequences[0, input_length:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        if self.stop is not None:
            for s in self.stop:
                text = text.split(s)[0]

        attention_summary = self._summarize_first_generated_token_attention(
            inputs=inputs,
            sequence=sequences[0],
            input_length=input_length,
        )
        result = {"text": [text], "attention": attention_summary}
        if attention_mask_ablation is not None:
            result["attention_mask_ablation"] = attention_mask_ablation
        if self.log_generation_ppl:
            result.update(
                compute_generation_ppl_stats(
                    generated_ids.scores,
                    new_token_ids.reshape(1, -1),
                    tokenizer=self.tokenizer,
                    include_token_details=self.log_generation_token_ppl,
                )
            )
        if timing_collector is not None:
            timing = timing_collector.summary(generated_token_count=int(new_token_ids.numel()))
            timing["input_tokens"] = int(input_length)
            result["generation_timing"] = timing
        return result

    def _measure_attention_callable(self, func: Callable) -> tuple:
        """运行一次 callable，并用 CUDA event wrapper 统计 attention op 耗时。"""

        with CudaEventAttentionTimer(self.model) as timer:
            result = func()
        return result, timer.summary()

    def _profile_prefill_attention(self, inputs) -> tuple:
        """Measure full prompt prefill forward attention op time."""

        def run_prefill():
            with torch.no_grad():
                model_inputs = dict(inputs)
                model_inputs.update(
                    {
                        "use_cache": True,
                        "return_dict": True,
                    }
                )
                return self._model_forward_for_attention_profile(model_inputs)

        return self._measure_attention_callable(run_prefill)

    def _profile_decode_attention(self, inputs, past_key_values, generated_token_ids: torch.Tensor) -> dict:
        """Replay decode forward for generated tokens and measure attention op time."""

        if generated_token_ids.numel() <= 1:
            return {
                "attention_kernel_event_count": 0,
                "attention_kernel_ms": 0.0,
                "warning": "生成 token 数不足 2，无法 replay decode forward。",
            }

        decode_token_ids = generated_token_ids[:-1].reshape(-1)
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")
        next_position = None
        if position_ids is not None:
            next_position = int(position_ids[0, -1].item()) + 1

        def run_decode():
            nonlocal past_key_values, attention_mask
            with torch.no_grad():
                for decode_idx, token_id in enumerate(decode_token_ids):
                    step_inputs = {
                        "input_ids": token_id.reshape(1, 1).to(self.model.device),
                        "past_key_values": past_key_values,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    if attention_mask is not None:
                        next_mask = torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        attention_mask = torch.cat([attention_mask, next_mask], dim=1)
                        step_inputs["attention_mask"] = attention_mask
                    if next_position is not None:
                        step_inputs["position_ids"] = torch.tensor(
                            [[next_position + decode_idx]],
                            dtype=position_ids.dtype,
                            device=position_ids.device,
                        )
                    outputs = self._model_forward_for_attention_profile(step_inputs)
                    next_past = getattr(outputs, "past_key_values", None)
                    if next_past is not None:
                        past_key_values = next_past

        _, summary = self._measure_attention_callable(run_decode)
        return summary

    def profile_attention_kernels_for_prompt(self, prompt: str) -> dict:
        """对单条 prompt 额外执行事件计时，返回 prefill/decode attention op 耗时。"""

        if self.pipeline is not None:
            raise RuntimeError("attention kernel profiling 需要直接 Hugging Face model，不能使用 pipeline。")

        inputs, attention_mask_ablation = self._prepare_generation_inputs([prompt], padding=False)
        input_length = int(inputs["input_ids"].shape[1])
        result = {
            "timer_backend": "cuda_event_attention_ops",
            "input_tokens": input_length,
            "generated_token_count": 0,
            "prefill_attention_kernel_ms": None,
            "decode_attention_kernel_ms_total": None,
            "decode_attention_kernel_ms_per_token_avg": None,
            "attention_kernel_event_count": 0,
        }
        if attention_mask_ablation is not None:
            result["attention_mask_ablation"] = attention_mask_ablation
        if not torch.cuda.is_available():
            result["warning"] = "CUDA 不可用，无法统计严格 GPU attention kernel 时间。"
            return result

        with torch.no_grad():
            generated_output = self._generate_for_attention_profile(inputs)
        sequences = generated_output.sequences if hasattr(generated_output, "sequences") else generated_output
        generated_token_ids = sequences[0, input_length:].detach()
        generated_token_count = int(generated_token_ids.numel())
        result["generated_token_count"] = generated_token_count

        prefill_outputs, prefill_summary = self._profile_prefill_attention(inputs)
        past_key_values = getattr(prefill_outputs, "past_key_values", None)
        if past_key_values is None:
            result["warning"] = "prefill 没有返回 past_key_values，无法 profile decode attention。"
        decode_summary = {"attention_kernel_event_count": 0, "attention_kernel_ms": None}
        if past_key_values is not None:
            decode_summary = self._profile_decode_attention(inputs, past_key_values, generated_token_ids)

        prefill_ms = prefill_summary["attention_kernel_ms"]
        decode_ms = decode_summary["attention_kernel_ms"]
        result.update(
            {
                "prefill_attention_kernel_ms": prefill_ms,
                "decode_attention_kernel_ms_total": decode_ms,
                "attention_kernel_event_count": (
                    prefill_summary["attention_kernel_event_count"]
                    + decode_summary["attention_kernel_event_count"]
                ),
            }
        )
        if decode_ms is not None and generated_token_count > 0:
            result["decode_attention_kernel_ms_per_token_avg"] = decode_ms / generated_token_count

        warnings = [
            summary.get("warning")
            for summary in (prefill_summary, decode_summary)
            if summary.get("warning")
        ]
        if result.get("warning"):
            warnings.insert(0, result["warning"])
        if warnings:
            result["warning"] = "；".join(warnings)
        return result

    def _summarize_first_generated_token_attention(self, inputs, sequence: torch.Tensor, input_length: int) -> dict:
        """读取首个生成 token 在每层对上下文 token 的注意力分布。"""

        summary = {
            "mode": "first_generated_token",
            "prompt_tokens": int(input_length),
            "generated_token_id": None,
            "generated_token_text": "",
            "layers": [],
        }
        if sequence.shape[0] <= input_length:
            summary["warning"] = "本次生成没有产生新 token，无法读取生成 token 注意力。"
            return summary

        first_generated_id = sequence[input_length].detach().reshape(1, 1).to(self.model.device)
        generated_token_id = int(first_generated_id.item())
        summary["generated_token_id"] = generated_token_id
        summary["generated_token_text"] = decode_attention_token(self.tokenizer, generated_token_id)

        with torch.no_grad():
            prefill_outputs = self.model(
                **inputs,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_key_values = getattr(prefill_outputs, "past_key_values", None)
            if past_key_values is None:
                summary["warning"] = "模型没有返回 past_key_values，无法低内存读取生成 token 注意力。"
                return summary

            step_inputs = {
                "input_ids": first_generated_id,
                "past_key_values": past_key_values,
                "use_cache": False,
                "output_attentions": True,
                "return_dict": True,
            }
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                next_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                step_inputs["attention_mask"] = torch.cat([attention_mask, next_mask], dim=1)
            if "position_ids" in inputs:
                step_inputs["position_ids"] = torch.tensor(
                    [[input_length]],
                    dtype=inputs["position_ids"].dtype,
                    device=inputs["position_ids"].device,
                )
            attention_outputs = self.model(**step_inputs)

        attentions = getattr(attention_outputs, "attentions", None)
        key_token_ids = torch.cat(
            [inputs["input_ids"][0].detach().cpu(), first_generated_id.detach().cpu().reshape(-1)]
        ).tolist()
        summary["layers"] = summarize_attention_layers(
            attentions=attentions,
            token_ids=key_token_ids,
            tokenizer=self.tokenizer,
            top_k=self.attention_top_k,
        )
        if not summary["layers"]:
            summary["warning"] = "模型没有返回可用的 attention 张量；请确认当前 attention backend 支持 output_attentions。"
        return summary


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
