# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-side DFlash runner: colocated prepare/forward/sample on GPU1."""

from __future__ import annotations

import gc
import math
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    download_weights_from_hf,
    safetensors_weights_iterator,
)
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    prepare_dflash_inputs,
)
from vllm.v1.worker.utils import prepare_kernel_block_sizes

logger = init_logger(__name__)


def _iter_weights(model_path: str):
    try:
        folder = download_weights_from_hf(
            model_path,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=None,
        )
    except Exception:
        folder = model_path
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


def _copy_embed_lm_head_from_target(
    draft_model: nn.Module, target_model: str, dtype: torch.dtype
) -> None:
    """Load target embed_tokens / lm_head by value when draft shares them."""
    draft_inner = draft_model.model if hasattr(draft_model, "model") else draft_model
    embed = getattr(draft_inner, "embed_tokens", None)
    lm_head = getattr(draft_model, "lm_head", None)
    need_embed = embed is not None
    need_lm = lm_head is not None
    if not need_embed and not need_lm:
        return
    for name, tensor in _iter_weights(target_model):
        if need_embed and (
            name == "model.embed_tokens.weight" or name.endswith("embed_tokens.weight")
        ):
            loader = getattr(embed.weight, "weight_loader", default_weight_loader)
            loader(embed.weight, tensor.to(dtype=dtype))
            need_embed = False
            logger.info("DFlash HS NIXL draft: loaded embed_tokens from target")
        if need_lm and (name == "lm_head.weight" or name.endswith("lm_head.weight")):
            loader = getattr(lm_head.weight, "weight_loader", default_weight_loader)
            loader(lm_head.weight, tensor.to(dtype=dtype))
            need_lm = False
            logger.info("DFlash HS NIXL draft: loaded lm_head from target")
        if not need_embed and not need_lm:
            break


class _BlockPool:
    def __init__(self, num_blocks: int):
        # Reserve block 0 as padding / invalid.
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, 0, -1))

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def usable_blocks(self) -> int:
        return max(self.num_blocks - 1, 0)

    @property
    def usage(self) -> float:
        usable = self.usable_blocks
        if usable <= 0:
            return 0.0
        return (usable - self.free_blocks) / usable

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(
                f"DFlash HS NIXL draft OOM: need {n} blocks, have {len(self._free)}"
            )
        start = len(self._free) - n
        out = self._free[start:]
        del self._free[start:]
        return out

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)


class DFlashDraftRunner:
    """Owns draft weights + KV; runs colocated prepare → forward → sample."""

    def __init__(
        self,
        *,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int | None = None,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",
        block_size: int = 16,
        device: torch.device | None = None,
    ):
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.num_query_per_req = 1 + self.num_speculative_tokens
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len)
        self.block_size = int(block_size)
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        # Set by sink before each SPECulate: _si{N}_vi{V}_ki{K}_n{reqs}
        self._nvtx_suffix: str = ""

        if max_num_batched_tokens is None:
            max_num_batched_tokens = min(
                self.max_num_seqs * self.num_query_per_req * 8, 32768
            )
        self.max_num_batched_tokens = int(max_num_batched_tokens)

        target_model_config = ModelConfig(
            model=target_model,
            runner="generate",
            max_model_len=self.max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        parallel_config = ParallelConfig(tensor_parallel_size=1)
        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        speculative_config = SpeculativeConfig(
            target_model_config=target_model_config,
            target_parallel_config=parallel_config,
            model=draft_model,
            method="dflash",
            num_speculative_tokens=self.num_speculative_tokens,
        )
        draft_hf = speculative_config.draft_model_config.hf_config
        self.vllm_config = VllmConfig(
            model_config=target_model_config,
            cache_config=CacheConfig(
                block_size=self.block_size,
                gpu_memory_utilization=self._gpu_memory_utilization,
                cache_dtype="auto",
                enable_prefix_caching=False,
            ),
            parallel_config=parallel_config,
            scheduler_config=SchedulerConfig(
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_model_len,
                is_encoder_decoder=target_model_config.is_encoder_decoder,
            ),
            device_config=DeviceConfig(device="cuda"),
            load_config=LoadConfig(),
            speculative_config=speculative_config,
            attention_config=AttentionConfig(
                use_non_causal=dflash_has_any_non_causal(draft_hf),
            ),
        )

        self._seqs: dict[str, dict[str, Any]] = {}
        self._pool: _BlockPool | None = None
        self.spec: DFlashSpeculator | None = None
        self.block_tables: BlockTables | None = None
        self.kv_cache_config: KVCacheConfig | None = None
        self.bytes_per_block: int = 0
        self.num_gpu_blocks: int = 0
        self._kv_caches: list[torch.Tensor] = []
        # Durable per-slot GPU state (like verify req_states) — avoid rebuild
        # + full H2D on every SPECulate.
        self._last_sampled: torch.Tensor | None = None
        self._next_prefill: torch.Tensor | None = None
        self._temperature: torch.Tensor | None = None
        self._seeds: torch.Tensor | None = None
        self._positions_buf: torch.Tensor | None = None
        self._query_start_loc_buf: torch.Tensor | None = None
        self._num_sampled_buf: torch.Tensor | None = None
        self._num_rejected_buf: torch.Tensor | None = None
        self._idx_mapping_buf: torch.Tensor | None = None

        with set_current_vllm_config(self.vllm_config):
            self._init_distributed()
            gc.collect()
            torch.cuda.empty_cache()
            self._load_and_bind()
            self._init_durable_buffers()

    def _init_distributed(self) -> None:
        init_method = get_distributed_init_method(get_ip(), get_open_port())
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=init_method,
            local_rank=int(self.device.index or 0),
            backend="nccl",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def _load_and_bind(self) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        assert draft_model_config is not None

        # Load draft weights into the static forward context.
        model = get_model(
            vllm_config=self.vllm_config, model_config=draft_model_config
        )
        _copy_embed_lm_head_from_target(
            model,
            self.vllm_config.model_config.model,
            self.vllm_config.model_config.dtype,
        )
        if hasattr(model, "model") and hasattr(model.model, "_build_fused_kv_buffers"):
            model.model._build_fused_kv_buffers()

        self.spec = DFlashSpeculator(self.vllm_config, self.device)
        # Avoid constructing the verify-side NIXL probe inside the draft process.
        self.spec._hs_nixl_probe = None
        self.spec.model = model
        all_attn = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.spec.draft_attn_layer_names = all_attn
        self.spec._validate_local_argmax_reduction()

        kv_cache_spec = get_kv_cache_spec(self.vllm_config)
        if not kv_cache_spec:
            raise RuntimeError("DFlash draft produced no KV cache specs")
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        if not kv_cache_groups:
            raise RuntimeError("DFlash draft produced no KV cache groups")

        bytes_per_block = sum(
            int(g.kv_cache_spec.page_size_bytes) * len(g.layer_names)
            for g in kv_cache_groups
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        available = int(free_bytes * self._gpu_memory_utilization)
        num_gpu_blocks = max(available // max(bytes_per_block, 1), 8)
        # Cap to something sane for the seq budget.
        max_needed = (
            self.max_num_seqs
            * (math.ceil(self.max_model_len / self.block_size) + 2)
            + 1
        )
        num_gpu_blocks = min(num_gpu_blocks, max_needed)
        logger.info(
            "DFlash HS NIXL draft KV: num_blocks=%d bytes/block=%d available=%.2fGiB",
            num_gpu_blocks,
            bytes_per_block,
            available / (1024**3),
        )
        self.bytes_per_block = int(bytes_per_block)
        self.num_gpu_blocks = int(num_gpu_blocks)

        kv_cache_tensors = [
            KVCacheTensor(
                size=int(group.kv_cache_spec.page_size_bytes) * num_gpu_blocks,
                shared_by=[layer_name],
            )
            for group in kv_cache_groups
            for layer_name in group.layer_names
        ]
        self.kv_cache_config = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )
        self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self._pool = _BlockPool(num_gpu_blocks)

        # set_attn discovers backends; pass empty target groups.
        dummy_model_state = SimpleNamespace()
        target_buffers = InputBuffers(
            max_num_reqs=self.max_num_seqs,
            max_num_tokens=self.max_num_batched_tokens,
            device=self.device,
        )

        # Build BlockTables after we know kernel block sizes from attn init.
        # First pass: init backends via a temporary BlockTables-sized stub.
        # We call set_attn which calls init_attn_backend — needs BlockTables.
        block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
        max_num_blocks_per_group = [
            math.ceil(self.max_model_len / bs) + 2 for bs in block_sizes
        ]
        # Placeholder kernel sizes (= block sizes); corrected after set_attn.
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=list(block_sizes),
        )
        self.spec.set_attn(
            dummy_model_state,  # type: ignore[arg-type]
            self.kv_cache_config,
            self.block_tables,
            target_buffers,
            [],
        )
        kernel_block_sizes = prepare_kernel_block_sizes(
            self.kv_cache_config, self.spec.attn_groups
        )
        # Rebuild BlockTables with correct kernel block sizes if they differ.
        if list(kernel_block_sizes) != list(block_sizes):
            self.block_tables = BlockTables(
                block_sizes=block_sizes,
                max_num_reqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_blocks_per_group=max_num_blocks_per_group,
                device=self.device,
                kernel_block_sizes=kernel_block_sizes,
            )
            self.spec.block_tables = self.block_tables

        self.spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

        init_kv_cache(
            self._kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.spec.attn_groups,
            self.device,
            self.vllm_config.cache_config.cache_dtype,
            self.block_tables.kernel_block_sizes,
            self.vllm_config,
        )
        # Warmup + capture so sink draft matches colocated FULL graph path.
        try:
            self.spec.capture()
        except Exception:
            logger.exception(
                "DFlash HS NIXL draft cudagraph capture failed; using eager"
            )
            self.spec.init_cudagraph_manager(CUDAGraphMode.NONE)
        logger.info(
            "DFlash HS NIXL draft runner ready: H=%d max_tokens=%d K=%d cg=%s",
            self.spec.hidden_size,
            self.spec.max_num_tokens,
            self.num_speculative_tokens,
            getattr(
                getattr(self.spec, "query_cudagraph_manager", None),
                "cudagraph_mode",
                CUDAGraphMode.NONE,
            ),
        )

    @property
    def hidden_states(self) -> torch.Tensor:
        assert self.spec is not None
        return self.spec.hidden_states

    @property
    def hidden_size(self) -> int:
        assert self.spec is not None
        return self.spec.hidden_size

    @property
    def dtype(self) -> torch.dtype:
        assert self.spec is not None
        return self.spec.dtype

    def _init_durable_buffers(self) -> None:
        """Allocate once; SPECulate only patches active slots / token prefix."""
        device = self.device
        self._last_sampled = torch.zeros(
            self.max_num_seqs, dtype=torch.int64, device=device
        )
        self._next_prefill = torch.zeros(
            self.max_num_seqs, dtype=torch.int64, device=device
        )
        self._temperature = torch.zeros(
            self.max_num_seqs, dtype=torch.float32, device=device
        )
        self._seeds = torch.zeros(
            self.max_num_seqs, dtype=torch.int64, device=device
        )
        self._positions_buf = torch.zeros(
            self.max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self._query_start_loc_buf = torch.zeros(
            self.max_num_seqs + 1, dtype=torch.int32, device=device
        )
        self._num_sampled_buf = torch.zeros(
            self.max_num_seqs, dtype=torch.int32, device=device
        )
        self._num_rejected_buf = torch.zeros(
            self.max_num_seqs, dtype=torch.int32, device=device
        )
        self._idx_mapping_buf = torch.zeros(
            self.max_num_seqs, dtype=torch.int32, device=device
        )

    def publish_kv_metrics(self) -> None:
        """Push current draft KV pool usage to the sink Prometheus gauges."""
        if self._pool is None:
            return
        try:
            from vllm.v1.spec_decode.dflash_hs_nixl.metrics import observe_draft_kv

            observe_draft_kv(
                usage=self._pool.usage,
                free_blocks=self._pool.free_blocks,
                num_seqs=len(self._seqs),
                bytes_per_block=self.bytes_per_block,
                num_gpu_blocks=self.num_gpu_blocks,
            )
        except Exception:
            # Metrics must never break the draft path.
            pass

    def free(self, req_ids: list[str]) -> None:
        assert self._pool is not None
        assert self.block_tables is not None
        touched = False
        for rid in req_ids:
            state = self._seqs.pop(rid, None)
            if state is None:
                continue
            blocks = state["blocks"]
            self._pool.free(blocks)
            slot = int(state["slot"])
            for gid in range(len(self.block_tables.block_tables)):
                self.block_tables.block_tables[gid].gpu[slot].zero_()
                self.block_tables.input_block_tables[gid][slot].zero_()
                self.block_tables.num_blocks.np[gid, slot] = 0
            touched = True
            if self._last_sampled is not None:
                self._last_sampled[slot] = 0
                self._next_prefill[slot] = 0
                self._temperature[slot] = 0
                self._seeds[slot] = 0
        if touched:
            self.block_tables.num_blocks.copy_to_uva()
            self.publish_kv_metrics()

    def _ensure_seq(self, req_id: str) -> int:
        if req_id in self._seqs:
            return int(self._seqs[req_id]["slot"])
        used = {int(s["slot"]) for s in self._seqs.values()}
        slot = next(i for i in range(self.max_num_seqs) if i not in used)
        self._seqs[req_id] = {
            "slot": slot,
            "blocks": [],
            "ctx_len": 0,
            "num_blocks": 0,
        }
        return slot

    def _ensure_blocks(self, req_id: str, need_tokens: int) -> bool:
        """Grow CPU-side block list. Returns True if new blocks need a GPU flush."""
        assert self._pool is not None
        assert self.block_tables is not None
        state = self._seqs[req_id]
        need_blocks = math.ceil(need_tokens / self.block_size)
        have = len(state["blocks"])
        if need_blocks <= have:
            return False
        extra = self._pool.allocate(need_blocks - have)
        state["blocks"].extend(extra)
        state["num_blocks"] = len(state["blocks"])
        self.publish_kv_metrics()
        return True

    def _flush_dirty_block_tables(self, dirty_req_ids: list[str]) -> None:
        """Batch-append new block ids via BlockTables (same path as verify).

        Avoids per-req ``tensor(..., device=cuda)`` sync storms that showed up as
        multi-ms ``dflash_sink_pack_blocks`` when many seqs grow on one step.
        """
        if not dirty_req_ids:
            return
        assert self.block_tables is not None
        n_groups = len(self.block_tables.block_tables)
        for rid in dirty_req_ids:
            state = self._seqs[rid]
            slot = int(state["slot"])
            blocks: list[int] = state["blocks"]
            # num_blocks.np is the durable GPU-side count (verify's source of truth).
            start = int(self.block_tables.num_blocks.np[0, slot])
            if len(blocks) <= start:
                continue
            new_ids = blocks[start:]
            # Same physical blocks for every draft KV group.
            new_block_ids = tuple(new_ids for _ in range(n_groups))
            self.block_tables.append_block_ids(slot, new_block_ids, overwrite=False)
            state["num_blocks"] = len(blocks)
        self.block_tables.apply_staged_writes()

    @torch.inference_mode()
    def speculate(
        self,
        *,
        req_ids: list[str],
        num_ctx_tokens: int,
        tensors: dict[str, torch.Tensor],
        wait_hiddens_ready: Any | None = None,
    ) -> torch.Tensor:
        """Run colocated prepare → precompute → generate/sample.

        Context hiddens land in ``self.hidden_states[:num_ctx_tokens]`` via NIXL.
        When ``wait_hiddens_ready`` is set, CPU prep runs first and the callable
        blocks until HS PtoP is done (so prep overlaps the transfer).

        Meta arrives on CPU via ZMQ. Per-req block tables / sampling state are
        durable on GPU (like verify); only new blocks and this-step token meta
        are patched. Draft generate uses FULL cudagraph when captured.
        """
        assert self.spec is not None
        assert self.block_tables is not None
        assert self.kv_cache_config is not None
        assert self._last_sampled is not None
        assert self._positions_buf is not None
        assert self._idx_mapping_buf is not None

        num_reqs = len(req_ids)
        if num_reqs == 0:
            return torch.zeros(
                0, self.num_speculative_tokens, dtype=torch.int64, device=self.device
            )

        device = self.device
        dirty_block_reqs: list[str] = []
        sfx = getattr(self, "_nvtx_suffix", "") or ""
        torch.cuda.nvtx.range_push(f"dflash_sink_prep_cpu{sfx}")
        try:
            def _host_1d(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
                return t.detach().to(dtype=dtype).reshape(-1).contiguous()

            positions_cpu = _host_1d(tensors["positions"], torch.int64)
            query_start_loc_cpu = _host_1d(tensors["query_start_loc"], torch.int32)
            num_sampled_cpu = _host_1d(tensors["num_sampled"], torch.int32)
            num_rejected_cpu = _host_1d(tensors["num_rejected"], torch.int32)
            last_sampled_in = _host_1d(tensors["last_sampled"], torch.int64)
            next_prefill_in = _host_1d(tensors["next_prefill_tokens"], torch.int64)
            temperature_in = _host_1d(tensors["temperature"], torch.float32)
            seeds_in = _host_1d(tensors["seeds"], torch.int64)

            nst = tensors["num_scheduled_tokens"].reshape(-1)[:num_reqs]
            num_scheduled_tokens = (
                nst.detach().cpu().numpy().astype(np.int32)
                if nst.is_cuda
                else np.asarray(nst.numpy(), dtype=np.int32)
            )
            seq_ub = tensors["seq_lens_cpu_upper_bound"].reshape(-1)[:num_reqs]
            if isinstance(seq_ub, torch.Tensor):
                seq_ub_np = (
                    seq_ub.detach().cpu().numpy()
                    if seq_ub.is_cuda
                    else seq_ub.numpy()
                )
            else:
                seq_ub_np = np.asarray(seq_ub)
            seq_lens_cpu_upper_bound = torch.as_tensor(
                seq_ub_np[:num_reqs], dtype=torch.int32
            )

            slots = [self._ensure_seq(rid) for rid in req_ids]
            qsl_np = query_start_loc_cpu[: num_reqs + 1].numpy()
            pos_np = positions_cpu[:num_ctx_tokens].numpy()
            n_rej = num_rejected_cpu[:num_reqs].numpy().astype(np.int64)
            for i, rid in enumerate(req_ids):
                s = int(qsl_np[i])
                e = int(qsl_np[i + 1])
                if e > s:
                    max_ctx_pos = int(pos_np[s:e].max())
                    valid_e = e - int(n_rej[i])
                    if valid_e > s:
                        last_valid = int(pos_np[valid_e - 1])
                    else:
                        last_valid = max(0, int(self._seqs[rid]["ctx_len"]) - 1)
                else:
                    max_ctx_pos = max(0, int(self._seqs[rid]["ctx_len"]) - 1)
                    last_valid = max_ctx_pos
                need = max(last_valid, max_ctx_pos) + 1 + self.num_query_per_req
                if self._ensure_blocks(rid, need):
                    dirty_block_reqs.append(rid)
                self._seqs[rid]["ctx_len"] = max(
                    int(self._seqs[rid]["ctx_len"]), last_valid + 1
                )
        finally:
            torch.cuda.nvtx.range_pop()

        # Only flush slots that grew — one staged H2D like verify model_runner.
        torch.cuda.nvtx.range_push(f"dflash_sink_pack_blocks{sfx}")
        try:
            self._flush_dirty_block_tables(dirty_block_reqs)
        finally:
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"dflash_sink_h2d{sfx}")
        try:
            # Patch durable buffers (slot-indexed like verify req_states).
            assert self._query_start_loc_buf is not None
            assert self._num_sampled_buf is not None
            assert self._num_rejected_buf is not None
            assert self._next_prefill is not None
            assert self._temperature is not None
            assert self._seeds is not None
            assert self._positions_buf is not None
            assert self._idx_mapping_buf is not None
            assert self._last_sampled is not None

            # Host→device via copy_ from CPU tensors (no per-call device=cuda sync).
            self._positions_buf[:num_ctx_tokens].copy_(
                positions_cpu[:num_ctx_tokens], non_blocking=True
            )
            self._query_start_loc_buf[: num_reqs + 1].copy_(
                query_start_loc_cpu[: num_reqs + 1], non_blocking=True
            )
            self._num_sampled_buf[:num_reqs].copy_(
                num_sampled_cpu[:num_reqs], non_blocking=True
            )
            self._num_rejected_buf[:num_reqs].copy_(
                num_rejected_cpu[:num_reqs], non_blocking=True
            )

            slots_np = np.asarray(slots, dtype=np.int32)
            self._idx_mapping_buf[:num_reqs].copy_(
                torch.from_numpy(slots_np), non_blocking=True
            )
            slots_t = self._idx_mapping_buf[:num_reqs].long()
            self._last_sampled.index_copy_(
                0, slots_t, last_sampled_in[:num_reqs].to(device, non_blocking=True)
            )
            self._next_prefill.index_copy_(
                0, slots_t, next_prefill_in[:num_reqs].to(device, non_blocking=True)
            )
            self._temperature.index_copy_(
                0, slots_t, temperature_in[:num_reqs].to(device, non_blocking=True)
            )
            self._seeds.index_copy_(
                0, slots_t, seeds_in[:num_reqs].to(device, non_blocking=True)
            )

            positions = self._positions_buf[:num_ctx_tokens]
            query_start_loc = self._query_start_loc_buf[: num_reqs + 1]
            num_sampled = self._num_sampled_buf[:num_reqs]
            num_rejected = self._num_rejected_buf[:num_reqs]
            idx_mapping = self._idx_mapping_buf[:num_reqs]
            last_sampled = self._last_sampled
            next_prefill = self._next_prefill
            temperature = self._temperature
            seeds = self._seeds

            input_batch = SimpleNamespace(
                num_reqs=num_reqs,
                num_tokens=num_ctx_tokens,
                positions=positions,
                query_start_loc=query_start_loc,
                idx_mapping=idx_mapping,
                num_scheduled_tokens=num_scheduled_tokens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            )

            max_seq_len = int(np.max(seq_ub_np[:num_reqs])) if num_reqs else 0
            self.spec.draft_max_seq_len = min(
                max_seq_len + self.num_query_per_req, self.max_model_len
            )
            self.spec._copy_request_inputs(num_reqs, idx_mapping, temperature, seeds)
        finally:
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"dflash_sink_prepare{sfx}")
        try:
            # Densify durable slot tables → batch-ordered input_block_tables.
            # prepare_dflash indexes block_table by batch row (not slot); same as
            # colocated propose after model_runner.gather_block_tables.
            self.block_tables.gather_block_tables(
                idx_mapping, num_reqs_padded=num_reqs
            )
            for i, gid in enumerate(self.spec.draft_kv_cache_group_ids):
                prepare_dflash_inputs(
                    self.spec.input_buffers,
                    self.block_tables.slot_mappings[gid],
                    self.spec.context_positions,
                    self.spec._context_slot_mappings[i],
                    self.spec.sample_indices,
                    self.spec.sample_pos,
                    self.spec.sample_idx_mapping,
                    input_batch,  # type: ignore[arg-type]
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill,
                    self.block_tables.input_block_tables[gid],
                    self.block_tables.kernel_block_sizes[gid],
                    self.spec.parallel_drafting_token_id,
                    self.spec.num_query_per_req,
                    self.spec.num_speculative_steps,
                    self.spec.max_num_reqs,
                    self.spec.max_num_tokens,
                    self.spec.max_model_len,
                    self.spec.sample_from_anchor,
                )
        finally:
            torch.cuda.nvtx.range_pop()

        if wait_hiddens_ready is not None:
            wait_hiddens_ready()

        torch.cuda.nvtx.range_push(f"dflash_sink_precompute{sfx}")
        try:
            if self.spec._layer_group_idx is not None:
                context_slots: torch.Tensor | list[torch.Tensor | None] | None = [
                    self.spec._context_slot_mappings[gidx][:num_ctx_tokens]
                    for gidx in self.spec._layer_group_idx
                ]
            else:
                context_slots = self.spec._context_slot_mappings[0][:num_ctx_tokens]
            self.spec.model.precompute_and_store_context_kv(
                self.spec.hidden_states[:num_ctx_tokens],
                self.spec.context_positions[:num_ctx_tokens],
                context_slots,
            )
        finally:
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"dflash_sink_generate{sfx}")
        try:
            num_query_tokens = num_reqs * self.spec.num_query_per_req
            batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
                self.spec.query_cudagraph_manager,
                num_reqs,
                num_query_tokens,
                uniform_token_count=self.spec.num_query_per_req,
                dp_size=1,
                dp_rank=0,
                need_eager=False,
            )
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            num_tokens_padded = batch_desc.num_tokens
            draft_attn_metadata = self.spec._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                causal=self.spec._group_causal,
            )
            draft_slot_mappings = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, :num_tokens_padded],
                self.kv_cache_config,
            )
            self.spec._prepare_eplb_forward(num_query_tokens)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.spec.query_cudagraph_manager is not None
                self.spec.query_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self.spec._generate_draft(
                    num_reqs,
                    num_tokens_padded,
                    draft_attn_metadata,
                    draft_slot_mappings,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )
            return self.spec.draft_tokens[:num_reqs].clone()
        finally:
            torch.cuda.nvtx.range_pop()
