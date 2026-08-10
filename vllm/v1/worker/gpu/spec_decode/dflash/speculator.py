# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    load_dflash_fc_only,
    load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _dual_run_check_enabled(speculative_config: Any) -> bool:
    """Env ``VLLM_DFLASH_DUAL_RUN_CHECK`` overrides speculative_config flag."""
    import os

    env = os.environ.get("VLLM_DFLASH_DUAL_RUN_CHECK")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(speculative_config, "disagg_dflash_dual_run_check", False))


class DFlashSpeculator(DraftModelSpeculator):
    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        # Whether the anchor query position is itself a prediction. DFlash default uses
        # the anchor as the bonus token (only mask tokens predict); DSpark samples from
        # the anchor and the N-1 mask token positions. See _prepare_dflash_inputs_kernel
        self.sample_from_anchor = False

        # Context positions for the K/V precompute. Populated by
        # prepare_dflash_inputs, and processed by the model's
        # precompute_and_store_context_kv method. NOT captured by CUDA graphs.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1

        # Milestone-2: verify serves remote draft tokens; local draft skipped.
        self.remote_only = bool(
            getattr(self.speculative_config, "disagg_dflash_remote_only", False)
        )
        # Milestone-4: kick SPECulate without waiting; publish real draft ids
        # when the ZMQ reply arrives (CPU ready signal + GPU buffer write).
        self.async_verify = bool(
            getattr(self.speculative_config, "disagg_dflash_async_verify", False)
        )
        # Set by model_runner.execute_model from SchedulerOutput.schedule_step
        # so kick/poll NVTX ranges correlate with execute_i{N}_*.
        self.last_verify_step: int = 0
        self._nvtx_kick_iter: int = 0

        # Milestone-0/1: optional NIXL HS (+ optional dual-run remote draft).
        self._hs_nixl_probe = None
        # Dual-run only: remote SPECulate recv deferred past next execute launch.
        self._remote_dual_run_deferred = False
        self._deferred_local_draft: torch.Tensor | None = None
        # Async remote-only: one-deep in-flight SPECulate on the socket, plus a
        # FIFO of ready draft batches (catchup before a new kick can land a
        # reply while a prior poll has not yet drained the previous ready set).
        self._async_lock = threading.Lock()
        self._async_pending_req_ids: list[str] | None = None
        self._async_pending_idx_mapping: torch.Tensor | None = None
        self._async_pending_num_reqs: int = 0
        # Each entry: (req_ids, idx_mapping, draft_tokens_gpu, draft_tokens_cpu)
        self._async_ready_queue: list[
            tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        # Pinned staging for non-blocking H2D of remote draft replies.
        self._async_draft_pin: torch.Tensor | None = None
        # Optional mp.Queue / queue.Queue: CPU draft ids for the engine without
        # waiting on the worker RPC thread (set by executor/model_runner).
        self._async_draft_side_queue: Any | None = None
        # Optional install into req_states.draft_tokens (model_runner hook).
        self._async_draft_install_fn: Any | None = None
        addr = self.speculative_config.disagg_dflash_address
        if addr:
            from vllm.v1.spec_decode.dflash_hs_nixl import DFlashHsNixlProbe

            self._hs_nixl_probe = DFlashHsNixlProbe(
                addr,
                max_tokens=self.max_num_tokens,
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                device=device,
            )
            if self.remote_only:
                if self.async_verify:
                    logger.info(
                        "DFlash HS NIXL probe enabled (address=%s); "
                        "remote-only async verify (non-blocking propose)",
                        addr,
                    )
                else:
                    logger.info(
                        "DFlash HS NIXL probe enabled (address=%s); "
                        "remote-only serving (no local draft forward)",
                        addr,
                    )
            else:
                logger.info(
                    "DFlash HS NIXL probe enabled (address=%s); "
                    "local draft still used for serving (dual-run)",
                    addr,
                )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The draft's attention differs from the target's in causality.
        return replace(
            self.vllm_config,
            attention_config=replace(
                self.vllm_config.attention_config,
                use_non_causal=self.requires_non_causal,
            ),
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.remote_only:
            # Draft kernels / graphs live on the sink GPU.
            self.query_cudagraph_manager = None
            return
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "%s draft attention (%s) does not support full CUDA graphs; "
                "running the draft eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        # PIECEWISE cudagraphs are not supported for dflash.
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        if self.remote_only:
            logger.info(
                "Skipping %s CUDA graph capture on verify (remote-only draft).",
                self._speculator_name,
            )
            return
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        # Reset sampling indices to zero to prevent stale values from prior
        # dummy runs from being baked into the captured graph.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        if self.remote_only:
            return load_dflash_fc_only(self.vllm_config, self.device)
        return load_dflash_model(target_model, self.vllm_config)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        if self.remote_only:
            # fc-only stub has no draft attention / KV on the verify GPU.
            self.model_state = model_state
            self.kv_cache_config = kv_cache_config
            self.block_tables = block_tables
            self.target_input_buffers = target_input_buffers
            self.target_attn_groups = target_attn_groups
            self.attn_groups = []
            self.draft_attn_layer_names = set()
            self.draft_kv_cache_group_ids = []
            self.draft_kv_cache_group_id = -1
            self._context_slot_mappings = None
            self._layer_group_idx = None
            self._group_causal = not self.requires_non_causal
            return

        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]

        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to. Models that share a single group
        # leave this as None and share one context slot mapping.
        self._layer_group_idx: list[int] | None = None
        # Per-KV-group causal, falling back to whether the drafter is all-causal.
        self._group_causal: dict[int, bool] | bool = not self.requires_non_causal
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]] for name in layer_names
            ]
            if hasattr(self.model, "get_draft_attn_causal"):
                self._group_causal = {
                    name_to_gid[name]: layer_causal
                    for name, layer_causal in zip(
                        layer_names, self.model.get_draft_attn_causal()
                    )
                }

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        # sample_pos is the predicted token's position Q; verification keys
        # Gumbel by the predecessor (Q-1). sample_draft adds +1, so pass Q-2.
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 2,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int | None = None,
        causal: bool | Mapping[int, bool] = False,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None  # Omitted for DFlash, read from self instead
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor | None:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        # Milestone-2 remote-only: kick SPECulate, block for draft tokens, skip
        # local prepare/forward/sample (draft lives on the sink GPU).
        # Milestone-4 async_verify: kick without wait; returns None until poll.
        if self.remote_only:
            if dummy_run:
                # Memory/cudagraph warmup: no sink round-trip; no local draft.
                return self.draft_tokens[:num_reqs]
            return self._propose_remote_only(
                input_batch=input_batch,
                num_reqs=num_reqs,
                num_target_tokens=num_target_tokens,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                temperature=temperature,
                seeds=seeds,
            )

        # After HS DtoD: kick NIXL(+SPECulate) on a side thread — do not wait.
        # Dual-run: remote wait is deferred until the *next* propose so the
        # worker can prepare/launch the next execute_context first.
        if self._hs_nixl_probe is not None and not dummy_run:
            # Complete previous step's remote recv (after batch N+1 was prepped).
            self.finish_remote_dual_run()
            if not self._hs_nixl_probe._handshook:
                self._hs_nixl_probe.handshake()
            if not hasattr(self, "_hs_ready_event"):
                self._hs_ready_event = torch.cuda.Event()
            self._hs_ready_event.record()
            if self._hs_nixl_probe.draft_enabled:
                self._hs_nixl_probe.begin_speculate(
                    self.hidden_states[:num_target_tokens],
                    req_ids=list(input_batch.req_ids[:num_reqs]),
                    num_speculative_tokens=self.num_speculative_steps,
                    positions=input_batch.positions,
                    # Host mirrors — encode path never .cpu() these.
                    query_start_loc=input_batch.query_start_loc_np,
                    num_sampled=num_sampled,
                    num_rejected=num_rejected,
                    last_sampled=last_sampled,
                    next_prefill_tokens=next_prefill_tokens,
                    temperature=temperature,
                    seeds=seeds,
                    num_scheduled_tokens=input_batch.num_scheduled_tokens,
                    seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                    idx_mapping=input_batch.idx_mapping_np,
                    src_ready_event=self._hs_ready_event,
                )
            else:
                self._hs_nixl_probe.begin_transfer(
                    self.hidden_states[:num_target_tokens],
                    src_ready_event=self._hs_ready_event,
                )

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        # Support multiple draft KV cache groups by preparing inputs once for each
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        # Pre-insert context K/V into the cache. Runs eagerly outside the captured graph
        # because the context shape varies per step. During dummy runs the block tables
        # are placeholders, so we skip the cache write to avoid clobbering real entries.
        # Each layer uses the context slots of its own kv-cache group.
        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][:num_target_tokens]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slots,
        )

        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=self._group_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]

    def _propose_remote_only(
        self,
        *,
        input_batch: InputBatch,
        num_reqs: int,
        num_target_tokens: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor | None:
        """Kick sink SPECulate; sync waits for ZMQ, async returns None."""
        probe = self._hs_nixl_probe
        if probe is None:
            raise RuntimeError(
                "disagg_dflash_remote_only requires disagg_dflash_address "
                "(HS NIXL probe was not created)."
            )
        if not probe._handshook:
            probe.handshake()
        if not probe.draft_enabled:
            raise RuntimeError(
                "disagg_dflash_remote_only requires a sink with draft enabled "
                "(start dflash_hs_nixl_sink with --draft-model)."
            )

        # Socket is 1-deep: must resolve any prior async SPECulate before a new kick.
        # Under wait-for-drafts (no decode-1), soft-skipping the kick deadlocks:
        # the just-sampled batch never gets SPECulate and stays unscheduled forever
        # (common with prefill-while-waiting or batch_queue overlap). Always catch
        # up — non-blocking first, then block — and queue ready drafts for poll.
        vi = int(getattr(self, "last_verify_step", 0) or 0)
        if self.async_verify and self._async_pending_req_ids is not None:
            torch.cuda.nvtx.range_push(
                f"dflash_hs_nixl_async_catchup_wait_vi{vi}"
            )
            try:
                if not self._resolve_async_remote_drafts(blocking=False):
                    self._resolve_async_remote_drafts(blocking=True)
                if self._async_pending_req_ids is not None:
                    raise RuntimeError(
                        "DFlash async remote-only: prior SPECulate still pending "
                        "after catchup; cannot kick a new batch"
                    )
            finally:
                torch.cuda.nvtx.range_pop()

        if not hasattr(self, "_hs_ready_event"):
            self._hs_ready_event = torch.cuda.Event()
        self._hs_ready_event.record()
        req_ids = list(input_batch.req_ids[:num_reqs])
        self._nvtx_kick_iter += 1
        ki = self._nvtx_kick_iter
        # Capture pending mapping *before* begin_speculate: the bg reply
        # callback may run before begin_speculate returns.
        idx_mapping_pending = (
            input_batch.idx_mapping[:num_reqs].clone() if self.async_verify else None
        )
        if self.async_verify:
            with self._async_lock:
                self._async_pending_req_ids = req_ids
                self._async_pending_idx_mapping = idx_mapping_pending
                self._async_pending_num_reqs = num_reqs
            torch.cuda.nvtx.range_push(
                f"dflash_hs_nixl_async_kick_vi{vi}_ki{ki}_n{num_reqs}"
            )
        try:
            probe.begin_speculate(
                self.hidden_states[:num_target_tokens],
                req_ids=req_ids,
                num_speculative_tokens=self.num_speculative_steps,
                positions=input_batch.positions,
                query_start_loc=input_batch.query_start_loc_np,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                temperature=temperature,
                seeds=seeds,
                num_scheduled_tokens=input_batch.num_scheduled_tokens,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                idx_mapping=input_batch.idx_mapping_np,
                src_ready_event=self._hs_ready_event,
                defer_meta_sync=self.async_verify,
                verify_step=vi,
                kick_iter=ki,
                on_speculate_reply=(
                    self._on_async_speculate_reply if self.async_verify else None
                ),
            )
        finally:
            if self.async_verify:
                torch.cuda.nvtx.range_pop()

        if self.async_verify:
            # Non-blocking: drafts arrive via bg recv callback / poll.
            return None

        # CPU wait while GPU0 is idle — expected nsys hole until GPU1 finishes.
        torch.cuda.nvtx.range_push(f"dflash_hs_nixl_remote_wait_vi{vi}_ki{ki}")
        try:
            remote = probe.finish_speculate()
        finally:
            torch.cuda.nvtx.range_pop()
        if remote is None:
            raise RuntimeError("DFlash remote-only: SPECulate reply missing draft tokens")
        if remote.shape[0] != num_reqs or remote.shape[-1] != self.num_speculative_steps:
            raise RuntimeError(
                "DFlash remote-only: unexpected draft_tokens shape "
                f"{tuple(remote.shape)} (expected ({num_reqs}, "
                f"{self.num_speculative_steps}))"
            )
        remote_gpu = remote.to(
            device=self.draft_tokens.device, dtype=self.draft_tokens.dtype
        )
        self.draft_tokens[:num_reqs].copy_(remote_gpu)
        return self.draft_tokens[:num_reqs]

    def _ensure_async_draft_pin(self, num_reqs: int) -> None:
        """Pinned host staging for non-blocking remote-draft H2D.

        Allocate outside InferenceMode so engine-thread poll (not under
        InferenceMode) can ``copy_`` into the buffer after propose/warmup
        created it under InferenceMode.
        """
        k = self.num_speculative_steps
        dtype = self.draft_tokens.dtype
        need = (
            self._async_draft_pin is None
            or self._async_draft_pin.shape[0] < num_reqs
            or self._async_draft_pin.dtype != dtype
        )
        if need:
            with torch.inference_mode(False):
                self._async_draft_pin = torch.empty(
                    (self.max_num_reqs, k), dtype=dtype, pin_memory=True
                )

    def set_async_draft_side_channel(
        self,
        side_queue: Any | None,
        install_fn: Any | None = None,
    ) -> None:
        """Wire engine-visible draft publish + optional req_states install hook."""
        self._async_draft_side_queue = side_queue
        self._async_draft_install_fn = install_fn

    def _on_async_speculate_reply(self, remote: torch.Tensor) -> None:
        """Bg-thread callback: stage drafts, install on GPU, publish CPU to engine."""
        with self._async_lock:
            req_ids = self._async_pending_req_ids
            idx_mapping = self._async_pending_idx_mapping
            if req_ids is None or idx_mapping is None:
                return
            self._stash_async_ready_drafts(
                req_ids=req_ids,
                idx_mapping=idx_mapping,
                remote=remote,
                already_locked=True,
            )
            if not self._async_ready_queue:
                return
            # Prefer immediate install + side-channel publish so the engine can
            # schedule without an RPC poll stuck behind execute_model.
            use_side = self._async_draft_side_queue is not None
            if use_side:
                req_ids, idx_mapping, remote_gpu, remote_cpu = (
                    self._async_ready_queue.pop()
                )
            else:
                req_ids, idx_mapping, remote_gpu, remote_cpu = self._async_ready_queue[
                    -1
                ]

        install_fn = self._async_draft_install_fn
        if install_fn is not None:
            try:
                install_fn(req_ids, idx_mapping, remote_gpu, remote_cpu)
            except Exception:
                logger.exception(
                    "DFlash async: req_states draft install from bg reply failed"
                )

        side_q = self._async_draft_side_queue
        if side_q is not None:
            try:
                side_q.put_nowait((list(req_ids), remote_cpu.tolist()))
            except Exception:
                try:
                    side_q.put((list(req_ids), remote_cpu.tolist()), timeout=0.01)
                except Exception:
                    logger.warning(
                        "DFlash async: side-channel publish failed; "
                        "re-queue for RPC poll"
                    )
                    with self._async_lock:
                        self._async_ready_queue.append(
                            (req_ids, idx_mapping, remote_gpu, remote_cpu)
                        )

    def _stash_async_ready_drafts(
        self,
        *,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        remote: torch.Tensor,
        already_locked: bool = False,
    ) -> None:
        """Validate ZMQ drafts and stage them for model_runner install + CPU publish.

        Keeps a CPU copy for scheduler ``spec_token_ids`` (no DtoH round-trip) and
        enqueues a pinned→GPU H2D with ``non_blocking=True`` so we never
        ``cudaStreamSynchronize`` the default compute stream on the engine thread.
        """

        def _do() -> None:
            num_reqs = len(req_ids)
            if (
                remote.shape[0] != num_reqs
                or remote.shape[-1] != self.num_speculative_steps
            ):
                raise RuntimeError(
                    "DFlash async remote-only: unexpected draft_tokens shape "
                    f"{tuple(remote.shape)} (expected ({num_reqs}, "
                    f"{self.num_speculative_steps}))"
                )
            # Wire decode yields a CPU tensor; own it for the scheduler tolist path.
            if remote.device.type != "cpu":
                remote_cpu = remote.detach().to(
                    dtype=self.draft_tokens.dtype, device="cpu"
                ).contiguous()
            else:
                remote_cpu = remote.detach().to(
                    dtype=self.draft_tokens.dtype
                ).contiguous()

            # Poll may run outside InferenceMode while buffers were first touched
            # under it. Use a per-batch pinned staging buffer when the ready queue
            # is non-empty so a later stash cannot overwrite an in-flight H2D src.
            with torch.inference_mode(False):
                if self._async_ready_queue:
                    pin = torch.empty(
                        remote_cpu.shape,
                        dtype=self.draft_tokens.dtype,
                        pin_memory=True,
                    )
                else:
                    self._ensure_async_draft_pin(num_reqs)
                    assert self._async_draft_pin is not None
                    pin = self._async_draft_pin[:num_reqs]
                pin.copy_(remote_cpu)
                # Owned GPU buffer: later kicks must not alias this tensor.
                remote_gpu = torch.empty(
                    remote_cpu.shape,
                    dtype=self.draft_tokens.dtype,
                    device=self.draft_tokens.device,
                )
                remote_gpu.copy_(pin, non_blocking=True)

            self._async_ready_queue.append(
                (req_ids, idx_mapping, remote_gpu, remote_cpu)
            )
            self._async_pending_req_ids = None
            self._async_pending_idx_mapping = None
            self._async_pending_num_reqs = 0

        if already_locked:
            _do()
        else:
            with self._async_lock:
                _do()

    def _resolve_async_remote_drafts(self, *, blocking: bool) -> bool:
        """Try to complete an in-flight async SPECulate. Returns True if pending cleared."""
        with self._async_lock:
            if self._async_pending_req_ids is None:
                return True
        probe = self._hs_nixl_probe
        if probe is None:
            return False
        if blocking:
            remote = probe.finish_speculate()
        else:
            remote = probe.try_finish_speculate()
        # Bg callback may have stashed while we joined/recv'd.
        with self._async_lock:
            if self._async_pending_req_ids is None:
                return True
        if remote is None:
            # Not ready yet, or drained by FREE while we still tracked pending.
            if blocking:
                with self._async_lock:
                    self._async_pending_req_ids = None
                    self._async_pending_idx_mapping = None
                    self._async_pending_num_reqs = 0
            return False
        with self._async_lock:
            if self._async_pending_req_ids is None:
                return True
            assert self._async_pending_idx_mapping is not None
            self._stash_async_ready_drafts(
                req_ids=self._async_pending_req_ids,
                idx_mapping=self._async_pending_idx_mapping,
                remote=remote,
                already_locked=True,
            )
        return True

    def clear_async_draft_state(self) -> None:
        """Drop in-flight / ready async draft bookkeeping (shutdown / hard reset)."""
        self._async_pending_req_ids = None
        self._async_pending_idx_mapping = None
        self._async_pending_num_reqs = 0
        self._async_ready_queue.clear()

    def recover_async_drafts_after_socket_drain(
        self,
        remote: torch.Tensor | None,
        *,
        exclude_req_ids: set[str] | None = None,
    ) -> None:
        """Stash a SPECulate reply drained by FREE/PROFILE for still-running reqs.

        FREE/PROFILE must clear the 1-deep DEALER socket before their own
        command. Under wait-for-drafts, dropping that reply (or clearing the
        ready queue) deadlocks survivors that were in the drained batch.
        """
        exclude = exclude_req_ids or set()
        if (
            remote is not None
            and self._async_pending_req_ids is not None
            and self._async_pending_idx_mapping is not None
        ):
            req_ids = self._async_pending_req_ids
            idx_mapping = self._async_pending_idx_mapping
            keep = [i for i, r in enumerate(req_ids) if r not in exclude]
            if not keep:
                self._async_pending_req_ids = None
                self._async_pending_idx_mapping = None
                self._async_pending_num_reqs = 0
            elif len(keep) == len(req_ids):
                self._stash_async_ready_drafts(
                    req_ids=req_ids,
                    idx_mapping=idx_mapping,
                    remote=remote,
                )
            else:
                t = torch.tensor(keep, dtype=torch.long)
                self._stash_async_ready_drafts(
                    req_ids=[req_ids[i] for i in keep],
                    idx_mapping=idx_mapping[t],
                    remote=remote[t],
                )
        if exclude:
            self._prune_async_ready_queue(exclude)

    def _prune_async_ready_queue(self, exclude_req_ids: set[str]) -> None:
        if not self._async_ready_queue or not exclude_req_ids:
            return
        pruned: list[tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for req_ids, idx_mapping, draft_gpu, draft_cpu in self._async_ready_queue:
            keep = [i for i, r in enumerate(req_ids) if r not in exclude_req_ids]
            if not keep:
                continue
            if len(keep) == len(req_ids):
                pruned.append((req_ids, idx_mapping, draft_gpu, draft_cpu))
                continue
            t = torch.tensor(keep, dtype=torch.long)
            pruned.append(
                (
                    [req_ids[i] for i in keep],
                    idx_mapping[t],
                    draft_gpu[t],
                    draft_cpu[t],
                )
            )
        self._async_ready_queue = pruned

    def poll_async_remote_drafts(
        self,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Non-blocking take of ready async drafts for GPU install + CPU publish.

        Returns
        ``(req_ids, idx_mapping, draft_tokens_gpu, draft_tokens_cpu)`` or None.
        """
        if not self.async_verify:
            return None
        vi = int(getattr(self, "last_verify_step", 0) or 0)
        torch.cuda.nvtx.range_push(f"dflash_hs_nixl_async_poll_vi{vi}")
        try:
            self._resolve_async_remote_drafts(blocking=False)
            with self._async_lock:
                if not self._async_ready_queue:
                    return None
                return self._async_ready_queue.pop(0)
        finally:
            torch.cuda.nvtx.range_pop()

    def defer_remote_dual_run(self, local_draft_tokens: torch.Tensor) -> None:
        """Mark remote SPECulate in-flight; do not recv yet (dual-run only).

        Serving already has ``local_draft_tokens``. Waiting is deferred to
        ``finish_remote_dual_run`` at the start of the next propose so the
        worker can prepare/launch the next batch first.
        """
        if self.remote_only:
            return
        probe = self._hs_nixl_probe
        if probe is None:
            return
        if probe._bg_thread is None and not getattr(probe, "_speculate_pending", False):
            return
        self._remote_dual_run_deferred = True
        # Clone only if we will compare later (buffer is reused next step).
        if _dual_run_check_enabled(self.speculative_config):
            self._deferred_local_draft = local_draft_tokens.detach().clone()
        else:
            self._deferred_local_draft = None

    def finish_remote_dual_run(
        self, local_draft_tokens: torch.Tensor | None = None
    ) -> None:
        """Recv GPU1 draft tokens for a previously deferred dual-run step.

        Called at the start of the next propose (after next execute was able to
        be prepared/launched). Serving still uses local drafts only.

        This wait is CPU-side (ZMQ recv). A CUDA-stream wait would need NIXL
        WRITE of tokens into a GPU0 buffer + event signaling instead of ZMQ.

        Token match check is off by default (see
        ``disagg_dflash_dual_run_check`` / ``VLLM_DFLASH_DUAL_RUN_CHECK``).
        """
        if self.remote_only:
            return
        probe = self._hs_nixl_probe
        if probe is None:
            return
        pending = bool(getattr(self, "_remote_dual_run_deferred", False)) or (
            probe._bg_thread is not None
            or getattr(probe, "_speculate_pending", False)
        )
        if not pending:
            return
        local = local_draft_tokens
        if local is None:
            local = getattr(self, "_deferred_local_draft", None)
        vi = int(getattr(self, "last_verify_step", 0) or 0)
        torch.cuda.nvtx.range_push(f"dflash_hs_nixl_remote_wait_vi{vi}")
        try:
            try:
                remote = probe.finish_speculate()
            except Exception:
                logger.exception("DFlash dual-run finish_speculate failed")
                return
            if remote is None:
                return
            if local is None:
                return
            if tuple(remote.shape) != tuple(local.shape):
                logger.warning(
                    "DFlash dual-run shape mismatch: remote=%s local=%s "
                    "(serving uses local drafts)",
                    tuple(remote.shape),
                    tuple(local.shape),
                )
                return
            if not _dual_run_check_enabled(self.speculative_config):
                return
            # Expensive: H2D + torch.equal syncs — only when explicitly enabled.
            remote_gpu = remote.to(device=local.device, dtype=local.dtype)
            if not torch.equal(remote_gpu, local):
                n_mismatch = int((remote_gpu != local).any(dim=-1).sum().item())
                logger.warning(
                    "DFlash dual-run mismatch: %d/%d reqs differ "
                    "(serving uses local drafts)",
                    n_mismatch,
                    local.shape[0],
                )
        finally:
            self._remote_dual_run_deferred = False
            self._deferred_local_draft = None
            torch.cuda.nvtx.range_pop()

    def remote_cuda_profile(self, start: bool) -> None:
        """Mirror verify cudaProfilerStart/Stop onto the HS NIXL sink GPU."""
        probe = self._hs_nixl_probe
        if probe is None:
            return
        try:
            # PROFILE may drain an in-flight SPECulate on the DEALER socket.
            drained = probe.profile(start)
            if self.async_verify:
                # Stash drained drafts; do not wipe the ready queue.
                self.recover_async_drafts_after_socket_drain(drained)
        except Exception as e:
            logger.warning("DFlash HS NIXL remote PROFILE failed: %s", e)


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the mask tokens
    # at offsets > 0 are sampled from, each AT its own position.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
