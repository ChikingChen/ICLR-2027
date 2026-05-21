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
import torch
from typing import Dict, List, Optional, Sequence


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


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.log_attention_scores = bool(generation_kwargs.pop("log_attention_scores", False))
        self.log_generation_ppl = bool(generation_kwargs.pop("log_generation_ppl", False))
        self.log_generation_token_ppl = bool(generation_kwargs.pop("log_generation_token_ppl", False))
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
        
        if self.log_attention_scores or self.log_generation_ppl:
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
        else:
            ensure_num_hidden_layers_alias(self.model.config)
            
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

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.log_attention_scores:
            return [self._process_one_with_attention(prompt) for prompt in prompts]

        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_output = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
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
            results.append(result)

        return results

    def _process_one_with_attention(self, prompt: str) -> dict:
        """生成单条样本，并附带首个生成 token 的逐层注意力摘要。"""

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs,
        )
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
        if self.log_generation_ppl:
            result.update(
                compute_generation_ppl_stats(
                    generated_ids.scores,
                    new_token_ids.reshape(1, -1),
                    tokenizer=self.tokenizer,
                    include_token_details=self.log_generation_token_ppl,
                )
            )
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
