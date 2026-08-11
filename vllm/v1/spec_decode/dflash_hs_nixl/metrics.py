# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the DFlash HS NIXL sink / draft runner.

Exposes the same scrape keys ``benchmark_random.sh`` uses for verify EngineCore:
``vllm:kv_cache_usage_perc``, ``vllm:kv_cache_block_size_bytes``, and
``vllm:cache_config_info{num_gpu_blocks=...}``.
"""

from __future__ import annotations

from prometheus_client import Gauge, start_http_server

from vllm.logger import init_logger

logger = init_logger(__name__)

_LABELS = ("model_name",)

gauge_kv_cache_usage = Gauge(
    "vllm:kv_cache_usage_perc",
    "Draft/sink KV-cache usage fraction (0..1). Same semantics as EngineCore.",
    labelnames=_LABELS,
)
gauge_kv_cache_block_size_bytes = Gauge(
    "vllm:kv_cache_block_size_bytes",
    "Draft/sink KV-cache bytes per block.",
    labelnames=_LABELS,
)
gauge_kv_cache_usage_bytes = Gauge(
    "vllm:kv_cache_usage_bytes",
    "Draft/sink KV bytes currently in use (used_blocks × bytes_per_block).",
    labelnames=_LABELS,
)
gauge_kv_cache_total_bytes = Gauge(
    "vllm:kv_cache_total_bytes",
    "Draft/sink KV pool bytes allocated (usable_blocks × bytes_per_block).",
    labelnames=_LABELS,
)
gauge_num_seqs = Gauge(
    "vllm:num_requests_running",
    "Draft/sink live sequences currently holding KV.",
    labelnames=_LABELS,
)

# Emulate EngineCore Info as a gauge permanently set to 1 (same as loggers.py).
_cache_config_info: Gauge | None = None

_metrics_started = False
_model_name = "unknown"
_bytes_per_block = 0
_num_gpu_blocks = 0


def start_sink_metrics_server(
    port: int,
    host: str = "0.0.0.0",
    *,
    model_name: str = "",
    bytes_per_block: int = 0,
    num_gpu_blocks: int = 0,
) -> None:
    """Expose draft/sink metrics at ``http://{host}:{port}/metrics``."""
    global _metrics_started, _model_name, _bytes_per_block, _num_gpu_blocks
    global _cache_config_info
    if port <= 0:
        return
    if _metrics_started:
        return
    _model_name = model_name or "unknown"
    _bytes_per_block = int(bytes_per_block)
    _num_gpu_blocks = int(num_gpu_blocks)
    start_http_server(port, addr=host)
    _metrics_started = True

    # Match EngineCore: cache_config_info is a gauge=1 with num_gpu_blocks label.
    _cache_config_info = Gauge(
        "vllm:cache_config_info",
        "Information of the draft/sink CacheConfig",
        labelnames=("num_gpu_blocks", "model_name"),
    )
    _cache_config_info.labels(
        num_gpu_blocks=str(_num_gpu_blocks),
        model_name=_model_name,
    ).set(1)

    observe_draft_kv(usage=0.0, free_blocks=max(_num_gpu_blocks - 1, 0), num_seqs=0)
    logger.info(
        "DFlash HS NIXL sink metrics listening on http://%s:%d/metrics "
        "(blocks=%d bytes/block=%d)",
        host,
        port,
        _num_gpu_blocks,
        _bytes_per_block,
    )


def observe_draft_kv(
    *,
    usage: float,
    free_blocks: int,
    num_seqs: int,
    bytes_per_block: int | None = None,
    num_gpu_blocks: int | None = None,
) -> None:
    """Update KV gauges from the sink draft block pool."""
    if not _metrics_started:
        return
    bpb = int(bytes_per_block if bytes_per_block is not None else _bytes_per_block)
    nblocks = int(num_gpu_blocks if num_gpu_blocks is not None else _num_gpu_blocks)
    usable = max(nblocks - 1, 0)
    used = max(usable - int(free_blocks), 0)
    labels = {"model_name": _model_name}
    gauge_kv_cache_usage.labels(**labels).set(float(usage))
    gauge_kv_cache_block_size_bytes.labels(**labels).set(bpb)
    gauge_kv_cache_usage_bytes.labels(**labels).set(used * bpb)
    gauge_kv_cache_total_bytes.labels(**labels).set(usable * bpb)
    gauge_num_seqs.labels(**labels).set(int(num_seqs))
