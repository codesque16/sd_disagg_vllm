# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    safetensors_weights_iterator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model


def _resolve_draft_weight_folder(model_path: str) -> str:
    """Resolve a local directory that contains draft safetensors/bin weights."""
    if os.path.isdir(model_path):
        return model_path
    try:
        return download_weights_from_hf(
            model_path,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=None,
        )
    except Exception:
        pass
    # Prefer an already-cached HF snapshot when hub list/download is unavailable.
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not resolve draft weight directory for {model_path!r}"
        ) from e


def _iter_draft_weights(model_path: str):
    folder = _resolve_draft_weight_folder(model_path)
    st: list[str] = []
    bins: list[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".safetensors"):
                st.append(p)
            elif f.endswith(".bin"):
                bins.append(p)
    if st:
        yield from safetensors_weights_iterator(st, use_tqdm_on_load=False)
    else:
        for p in bins:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            yield from sd.items()


def _fc_dims_from_draft_config(hf_config) -> tuple[bool, int, int]:
    """Return (use_aux, fc_in, fc_out) for the DFlash aux-HS projector."""
    drafter_config = getattr(hf_config, "eagle_config", {}) or {}
    if hasattr(drafter_config, "to_dict"):
        drafter_config = drafter_config.to_dict()
    dflash_config = getattr(hf_config, "dflash_config", {}) or {}
    if hasattr(dflash_config, "to_dict"):
        dflash_config = dflash_config.to_dict()
    merged = dict(drafter_config)
    merged.update(dflash_config)

    use_aux = bool(merged.get("use_aux_hidden_state", True))
    hidden_size = int(hf_config.hidden_size)
    if not use_aux:
        return False, hidden_size, hidden_size

    num_features = int(hf_config.num_hidden_layers)
    if "target_layer_ids" in merged:
        num_features = len(merged["target_layer_ids"])
    elif "layer_ids" in merged:
        num_features = len(merged["layer_ids"])

    if hasattr(hf_config, "target_hidden_size") and hf_config.target_hidden_size:
        target_h = int(hf_config.target_hidden_size)
    else:
        target_h = hidden_size
    return True, target_h * num_features, hidden_size


class _DFlashFcInner(nn.Module):
    def __init__(self, fc: nn.Linear, use_aux_hidden_state: bool):
        super().__init__()
        self.fc = fc
        self.use_aux_hidden_state = use_aux_hidden_state


class DFlashFcOnlyStub(nn.Module):
    """Verify-side stub: aux-HS ``fc`` projector only (no draft layers / KV)."""

    def __init__(self, fc: nn.Linear | None, use_aux_hidden_state: bool):
        super().__init__()
        if fc is None:
            # Identity path when aux HS is disabled.
            self.model = _DFlashFcInner(
                nn.Linear(1, 1, bias=False), use_aux_hidden_state=False
            )
        else:
            self.model = _DFlashFcInner(fc, use_aux_hidden_state)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.in_features
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlash fc-only stub expects {expected} concatenated aux "
                f"hidden features but received {hidden_states.shape[-1]}."
            )
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result


def load_dflash_fc_only(vllm_config: VllmConfig, device: torch.device) -> nn.Module:
    """Load only the DFlash aux-HS ``fc`` projector for remote-only verify (C1)."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None
    hf_config = draft_model_config.hf_config
    dtype = vllm_config.model_config.dtype

    use_aux, fc_in, fc_out = _fc_dims_from_draft_config(hf_config)
    if not use_aux:
        logger.info(
            "DFlash remote-only: aux HS disabled; verify loads identity stub "
            "(no draft weights)."
        )
        return DFlashFcOnlyStub(None, use_aux_hidden_state=False).to(device)

    fc = nn.Linear(fc_in, fc_out, bias=False)
    fc = fc.to(device=device, dtype=dtype)
    loaded = False
    model_path = draft_model_config.model
    for name, tensor in _iter_draft_weights(model_path):
        # Checkpoints store ``fc.weight``; load_weights on full model remaps to
        # ``model.fc.weight``. Accept either.
        if name.endswith("fc.weight") or name == "fc.weight":
            weight = tensor.to(dtype=dtype, device="cpu")
            if tuple(weight.shape) != (fc_out, fc_in):
                raise ValueError(
                    f"DFlash fc weight shape mismatch: ckpt={tuple(weight.shape)} "
                    f"expected={(fc_out, fc_in)} from draft config "
                    f"(model={model_path!r})."
                )
            with torch.no_grad():
                fc.weight.copy_(weight.to(device=device, dtype=dtype))
            loaded = True
            break
    if not loaded:
        raise RuntimeError(
            f"DFlash remote-only: could not find fc.weight in draft checkpoint "
            f"{model_path!r}."
        )
    logger.info(
        "DFlash remote-only: loaded fc-only projector on verify GPU "
        "(%d -> %d, ~%.2f MiB); draft layers/KV stay on the sink.",
        fc_in,
        fc_out,
        fc.weight.numel() * fc.weight.element_size() / (1024 * 1024),
    )
    return DFlashFcOnlyStub(fc, use_aux_hidden_state=True)
