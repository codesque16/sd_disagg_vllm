# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up gpt-oss / OAI Triton MoE routing kernels before serving.

``_topk_forward`` / ``_sum_bitmatrix_rows`` / ``_combined_routing_*`` specialize
on bitmatrix column strides ``cdiv(n_rows_max, 32) * 32``. Dense ``_dummy_run``
sweeps can miss live scheduled token counts (CUDA-graph padding, mixed
prefill+decode sizes), so those kernels still JIT mid-serve and stall
``execute_context`` with ``posix_spawn`` / ``waitpid`` / ``cuModuleLoadData``.

This warmup calls the same routing subgraph as ``triton_kernel_moe_forward``
for every 32-token stride bucket up to ``max_num_batched_tokens``. No-op when
the model has no OAI Triton experts.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_OAI_TRITON_EXPERT_TYPES = frozenset(
    {
        "OAITritonExperts",
        "OAITritonMxfp4ExpertsMonolithic",
    }
)


@dataclass(frozen=True)
class _RoutingWarmupConfig:
    topk: int
    sm_first: bool
    num_experts: int
    num_local_experts: int
    expert_map: torch.Tensor | None
    device: torch.device
    dtype: torch.dtype


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_routing_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    """One size per bitmatrix stride bucket plus capture / max endpoints.

    ``topk`` allocates bitmatrix storage with column stride
    ``cdiv(n_rows_max, 32) * 32``; that stride is a Triton constexpr, so each
    distinct pad bucket needs its own compile.
    """
    if max_tokens <= 0:
        return []

    sizes: set[int] = {1, max_tokens}
    for pad in range(32, cdiv(max_tokens, 32) * 32 + 1, 32):
        sizes.add(min(pad, max_tokens))
    for size in cudagraph_capture_sizes:
        if 1 <= size <= max_tokens:
            sizes.add(size)
    return _normalize_token_sizes(sizes, max_tokens=max_tokens)


def _oai_triton_fused_experts(obj: object) -> object | None:
    """Return OAI Triton experts object if ``obj`` wraps or is one.

    ``OAITriton*`` experts are plain ABCs, not ``nn.Module``s, so they do not
    appear in ``model.modules()``. They hang off
    ``RoutedExperts.quant_method.moe_kernel.fused_experts``.
    """
    if obj.__class__.__name__ in _OAI_TRITON_EXPERT_TYPES:
        return obj
    quant_method = getattr(obj, "quant_method", None)
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    fused = getattr(moe_kernel, "fused_experts", None)
    if fused is not None and fused.__class__.__name__ in _OAI_TRITON_EXPERT_TYPES:
        return fused
    return None


def _find_oai_triton_experts(
    model: torch.nn.Module,
) -> tuple[object, torch.nn.Module | None] | None:
    """Return ``(experts, host_module)`` for the first OAI Triton MoE layer."""
    for module in model.modules():
        experts = _oai_triton_fused_experts(module)
        if experts is not None:
            return experts, module
    return None


def _expert_map_from_host(host: torch.nn.Module | None) -> torch.Tensor | None:
    if host is None:
        return None
    if callable(getattr(type(host), "expert_map", None)):
        try:
            value = host.expert_map
        except Exception:
            value = None
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(host, "expert_map", None)
    return value if isinstance(value, torch.Tensor) else None


def _routing_config_from_experts(
    experts: object,
    host: torch.nn.Module | None,
) -> _RoutingWarmupConfig | None:
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    moe_config = getattr(experts, "moe_config", None)
    if moe_config is None:
        moe_config = getattr(host, "moe_config", None) if host is not None else None
    if moe_config is None:
        return None

    topk_attr = getattr(experts, "topk", None)
    if topk_attr is None and host is not None:
        topk_attr = getattr(host, "top_k", None)
    topk = int(topk_attr if topk_attr is not None else moe_config.experts_per_token)

    if hasattr(experts, "renormalize"):
        renormalize = bool(experts.renormalize)
    elif host is not None and hasattr(host, "renormalize"):
        renormalize = bool(host.renormalize)
    else:
        renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    dtype = moe_config.router_logits_dtype or moe_config.in_dtype
    if dtype is None:
        dtype = torch.bfloat16

    device = torch.device(moe_config.device)
    if device.type != "cuda":
        return None

    num_experts = int(moe_config.num_experts)
    num_local_experts = int(moe_config.num_local_experts)
    expert_map = None
    if num_local_experts < num_experts:
        expert_map = _expert_map_from_host(host)
        if expert_map is not None:
            expert_map = expert_map.to(device=device)

    return _RoutingWarmupConfig(
        topk=topk,
        sm_first=not renormalize,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        expert_map=expert_map,
        device=device,
        dtype=dtype,
    )


def _warmup_routing_for_size(
    n_tokens: int,
    cfg: _RoutingWarmupConfig,
    *,
    use_legacy: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        make_routing_data,
    )

    gating = torch.randn(
        (n_tokens, cfg.num_experts),
        device=cfg.device,
        dtype=cfg.dtype,
    )

    # Match ``triton_kernel_moe_forward`` branching.
    if use_legacy and cfg.expert_map is None:
        from triton_kernels.routing import routing as fused_routing

        fused_routing(gating, cfg.topk, sm_first=cfg.sm_first)
        return

    from triton_kernels.topk import topk as topk_fn

    logits = gating
    if cfg.sm_first:
        logits = torch.softmax(logits, dim=-1)
    topk_result = topk_fn(logits, cfg.topk, apply_softmax=not cfg.sm_first)
    if isinstance(topk_result, tuple):
        topk_weights, topk_ids_raw, _ = topk_result
    else:
        topk_weights = topk_result.vals
        topk_ids_raw = topk_result.indx

    if cfg.expert_map is not None:
        topk_ids = cfg.expert_map[topk_ids_raw.to(torch.long)]
        make_routing_data(topk_ids, topk_weights, cfg.num_local_experts)
    else:
        topk_ids = topk_ids_raw.to(torch.long)
        make_routing_data(topk_ids, topk_weights, cfg.num_experts)


@instrument(span_name="gpt-oss Triton MoE routing warmup")
def gpt_oss_triton_moe_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    if not has_triton_kernels():
        return

    found = _find_oai_triton_experts(model)
    if found is None:
        return
    experts, host = found

    cfg = _routing_config_from_experts(experts, host)
    if cfg is None:
        return

    token_sizes = _select_routing_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        use_legacy_triton_kernels,
    )

    started = time.perf_counter()
    logger.info(
        "Warming up %d gpt-oss/OAI Triton MoE routing sizes "
        "(topk=%d, experts=%d/%d, max=%d, legacy=%s).",
        len(token_sizes),
        cfg.topk,
        cfg.num_local_experts,
        cfg.num_experts,
        token_sizes[-1],
        use_legacy_triton_kernels,
    )
    with torch.inference_mode():
        for n_tokens in token_sizes:
            _warmup_routing_for_size(
                n_tokens,
                cfg,
                use_legacy=use_legacy_triton_kernels,
            )
        torch.accelerator.synchronize()
    logger.info(
        "gpt-oss/OAI Triton MoE routing warmup finished in %.2f seconds "
        "(%d sizes).",
        time.perf_counter() - started,
        len(token_sizes),
    )
