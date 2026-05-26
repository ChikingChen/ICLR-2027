"""Shared helpers for attention-mask ablation experiments."""

from typing import Any, Dict, Tuple

import torch


def decode_token_text(tokenizer: Any, token_id: int) -> str:
    """Decode one token for metadata, falling back to the token id string."""

    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def disabled_attention_mask_ablation() -> Dict[str, Any]:
    """Return metadata for runs without attention-mask ablation."""

    return {
        "enabled": False,
        "mode": None,
        "token_position": None,
        "token_id": None,
        "token_text": None,
        "semantics": "no attention-mask ablation applied",
    }


def apply_bos_attention_mask(inputs: Dict[str, torch.Tensor], tokenizer: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Mask the BOS token at position 0 while preserving token positions."""

    if "input_ids" not in inputs:
        raise ValueError("inputs must contain input_ids")
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("--mask-bos-token only supports batch_size=1 inputs")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("tokenizer does not define bos_token_id")
    first_token_id = int(input_ids[0, 0].item())
    if first_token_id != int(bos_token_id):
        raise ValueError(
            f"--mask-bos-token expected BOS token id {bos_token_id} at position 0, "
            f"found {first_token_id}"
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
        "token_text": decode_token_text(tokenizer, first_token_id),
        "semantics": "token kept in input_ids; attention_mask is 0 during generation and replay",
    }
