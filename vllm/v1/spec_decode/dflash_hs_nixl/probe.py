# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side NIXL client: WRITE hiddens + optional SPECulate/FREE over ZMQ."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import zmq

from vllm.distributed.nixl_utils import NixlWrapper, is_nixl_available, nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash_hs_nixl.protocol import (
    CMD_FREE,
    CMD_HELLO,
    CMD_PROFILE,
    decode_speculate_response,
    encode_free_request,
    encode_hs_ready,
    encode_profile_request,
    encode_speculate_request,
)

logger = init_logger(__name__)

_VERIFY_AGENT_NAME = "dflash-hs-nixl-verify"
_MEM_TYPE = "VRAM"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data)


def _require_nixl() -> None:
    if not is_nixl_available() or NixlWrapper is None or nixl_agent_config is None:
        raise RuntimeError(
            "disagg_dflash_address requires the nixl (or rixl) package."
        )


def _make_agent(name: str) -> Any:
    _require_nixl()
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        backends=["UCX"],
        capture_telemetry=True,
    )
    return NixlWrapper(name, cfg)


def _wait_xfer_done(agent: Any, handle: Any, *, timeout_s: float = 120.0) -> None:
    t0 = time.perf_counter()
    while True:
        st = agent.check_xfer_state(handle)
        if st == "DONE":
            return
        if st == "ERR":
            raise RuntimeError("DFlash HS NIXL transfer entered ERR state")
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(
                f"DFlash HS NIXL transfer timed out after {timeout_s:.1f}s "
                f"(last state={st!r})"
            )
        time.sleep(0.0001)


class DFlashHsNixlProbe:
    """Handshake once, then blocking NIXL WRITE (+ optional SPECulate) each propose."""

    def __init__(
        self,
        address: str,
        *,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        timeout_ms: int = 600_000,
    ):
        self._address = address
        self._max_tokens = int(max_tokens)
        self._hidden_size = int(hidden_size)
        self._dtype = dtype
        self.device = device
        self._timeout_s = max(timeout_ms / 1000.0, 1.0)
        self._handshook = False
        self.draft_enabled = False
        self._speculate_pending = False
        # Background NIXL+SPECulate send so CPU can continue local draft launch
        # (async scheduling) without waiting on staging/PtoP.
        self._bg_thread: threading.Thread | None = None
        self._bg_error: BaseException | None = None
        # Set when bg thread recv's SPECulate and no on_speculate_reply callback
        # consumed it (sync dual-run / fallback). Guarded by `_reply_lock`.
        self._bg_reply: torch.Tensor | None = None
        self._reply_lock = threading.Lock()
        # True after bg (or callback path) finished the SPECulate reply.
        self._reply_complete = False

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        # HELLO may wait on draft weight load; keep a long timeout.
        hello_timeout_ms = max(int(timeout_ms), 600_000)
        self._sock.setsockopt(zmq.RCVTIMEO, hello_timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, hello_timeout_ms)
        self._sock.connect(address)

        self._agent: Any = None
        self._peer_name: str | None = None
        self._local_buf: torch.Tensor | None = None
        self._local_reg: Any = None
        self._remote_addr = 0
        self._remote_device_id = 0
        # Dedicated stream so staging DtoD + NIXL wait only on HS readiness,
        # not the default-stream target-kernel backlog (Stream 25).
        self._copy_stream: torch.cuda.Stream | None = None
        self._staging_done_event: torch.cuda.Event | None = None
        self._meta_done_event: torch.cuda.Event | None = None
        # Pinned host staging for SPECulate meta DtoH. `.to("cpu", non_blocking=True)`
        # into *unpinned* memory still issues cudaStreamSynchronize; pinned copy_
        # is required for a true async kick.
        self._meta_pin_positions: torch.Tensor | None = None
        self._meta_pin_num_sampled: torch.Tensor | None = None
        self._meta_pin_num_rejected: torch.Tensor | None = None
        self._meta_pin_last_sampled: torch.Tensor | None = None
        self._meta_pin_next_prefill: torch.Tensor | None = None
        self._meta_pin_temperature: torch.Tensor | None = None
        self._meta_pin_seeds: torch.Tensor | None = None

    def handshake(self) -> None:
        if self._handshook:
            return
        self._agent = _make_agent(_VERIFY_AGENT_NAME)
        req = {
            "cmd": CMD_HELLO,
            "agent_metadata": _b64(self._agent.get_agent_metadata()),
            "max_tokens": self._max_tokens,
            "hidden_size": self._hidden_size,
            "dtype": str(self._dtype).removeprefix("torch."),
        }
        try:
            self._sock.send_multipart([json.dumps(req).encode("utf-8")])
            frames = self._sock.recv_multipart()
        except zmq.Again as e:
            raise TimeoutError(
                f"DFlash HS NIXL HELLO timed out waiting for sink at {self._address}. "
                "Is dflash_hs_nixl_sink running?"
            ) from e
        meta = json.loads(frames[0].decode("utf-8"))
        if meta.get("error"):
            raise RuntimeError(f"DFlash HS NIXL HELLO failed: {meta}")
        if meta.get("cmd") != CMD_HELLO:
            raise RuntimeError(f"Unexpected HELLO reply: {meta}")

        self._peer_name = self._agent.add_remote_agent(_unb64(meta["agent_metadata"]))
        if int(meta["max_tokens"]) < self._max_tokens or int(meta["hidden_size"]) != (
            self._hidden_size
        ):
            raise RuntimeError(
                f"Sink staging mismatch: sink max_tokens={meta['max_tokens']} "
                f"H={meta['hidden_size']}, verify max_tokens={self._max_tokens} "
                f"H={self._hidden_size}"
            )
        self._dtype = getattr(torch, str(meta.get("dtype", "bfloat16")), self._dtype)
        self._remote_addr = int(meta["hidden_addr"])
        self._remote_device_id = int(meta["device_id"])
        self.draft_enabled = bool(meta.get("draft_enabled", False))

        # Allocate outside InferenceMode so the bg transfer thread can copy_ into it.
        # Handshake often runs under InferenceMode (propose path); tensors created
        # there become "inference tensors" and reject inplace updates from other modes.
        with torch.inference_mode(False):
            self._local_buf = torch.zeros(
                self._max_tokens,
                self._hidden_size,
                dtype=self._dtype,
                device=self.device,
            )
        self._local_reg = self._agent.register_memory(self._local_buf)
        if not self._local_reg:
            raise RuntimeError("DFlash HS NIXL verify register_memory failed")

        # SPECulate / FREE may take longer than HELLO.
        timeout_ms = int(self._timeout_s * 1000)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)

        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._staging_done_event = torch.cuda.Event()
        self._meta_done_event = torch.cuda.Event()
        self._alloc_meta_pins()

        self._handshook = True
        logger.info(
            "DFlash HS NIXL handshake ok: addr=%s max_tokens=%d H=%d peer=%s "
            "draft_enabled=%s",
            self._address,
            self._max_tokens,
            self._hidden_size,
            self._peer_name,
            self.draft_enabled,
        )

    def _nixl_write_nbytes(self, nbytes: int) -> None:
        """PtoP WRITE of the staged prefix (single initialize_xfer; exact nbytes)."""
        assert self._agent is not None
        assert self._local_buf is not None
        assert self._peer_name is not None
        if nbytes <= 0:
            return
        torch.cuda.nvtx.range_push("dflash_hs_nixl_xfer_setup")
        try:
            local_dev = int(self._local_buf.get_device())
            local_descs = self._agent.get_xfer_descs(
                [(int(self._local_buf.data_ptr()), nbytes, local_dev)],
                mem_type=_MEM_TYPE,
            )
            remote_descs = self._agent.get_xfer_descs(
                [(self._remote_addr, nbytes, self._remote_device_id)],
                mem_type=_MEM_TYPE,
            )
            handle = self._agent.initialize_xfer(
                "WRITE",
                local_descs,
                remote_descs,
                self._peer_name,
                b"",
            )
        finally:
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("dflash_hs_nixl_ptop")
        try:
            st = self._agent.transfer(handle)
            if st == "ERR":
                self._agent.release_xfer_handle(handle)
                raise RuntimeError("DFlash HS NIXL WRITE transfer failed")
            if st != "DONE":
                _wait_xfer_done(self._agent, handle, timeout_s=self._timeout_s)
            self._agent.release_xfer_handle(handle)
        finally:
            torch.cuda.nvtx.range_pop()

    def transfer(
        self,
        hidden_states: torch.Tensor,
        *,
        src_ready_event: torch.cuda.Event | None = None,
    ) -> None:
        """Blocking: staging DtoD + NIXL PtoP WRITE, wait DONE.

        If ``src_ready_event`` is set (recorded after the HS DtoD on the default
        stream), staging runs on a side stream that waits only for that event so
        PtoP follows HS DtoD without draining the target-kernel backlog.
        """
        if not self._handshook:
            self.handshake()
        assert self._agent is not None
        assert self._local_buf is not None
        assert self._peer_name is not None
        assert self._copy_stream is not None
        assert self._staging_done_event is not None

        hiddens = hidden_states.contiguous()
        n_ctx = int(hiddens.shape[0])
        if n_ctx > self._max_tokens:
            raise RuntimeError(
                f"DFlash HS NIXL: {n_ctx} tokens exceeds max_tokens={self._max_tokens}"
            )
        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        if n_ctx == 0:
            return

        torch.cuda.nvtx.range_push("dflash_hs_nixl")
        try:
            if src_ready_event is not None:
                self._copy_stream.wait_event(src_ready_event)
            with torch.cuda.stream(self._copy_stream):
                self._local_buf[:n_ctx].copy_(hiddens, non_blocking=True)
                hiddens.record_stream(self._copy_stream)
                self._local_buf.record_stream(self._copy_stream)
                self._staging_done_event.record(self._copy_stream)
            # Sync only the staging copy stream (not default-stream kernels).
            self._staging_done_event.synchronize()
            self._nixl_write_nbytes(nbytes)
        finally:
            torch.cuda.nvtx.range_pop()

    def _stage_hiddens(
        self,
        hidden_states: torch.Tensor,
        *,
        src_ready_event: torch.cuda.Event | None = None,
    ) -> int:
        """Staging DtoD only; returns nbytes. Caller runs NIXL after/with meta pack."""
        assert self._local_buf is not None
        assert self._copy_stream is not None
        assert self._staging_done_event is not None
        hiddens = hidden_states.contiguous()
        n_ctx = int(hiddens.shape[0])
        if n_ctx > self._max_tokens:
            raise RuntimeError(
                f"DFlash HS NIXL: {n_ctx} tokens exceeds max_tokens={self._max_tokens}"
            )
        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        if n_ctx == 0:
            return 0
        if src_ready_event is not None:
            self._copy_stream.wait_event(src_ready_event)
        with torch.cuda.stream(self._copy_stream):
            self._local_buf[:n_ctx].copy_(hiddens, non_blocking=True)
            hiddens.record_stream(self._copy_stream)
            self._local_buf.record_stream(self._copy_stream)
            self._staging_done_event.record(self._copy_stream)
        self._staging_done_event.synchronize()
        return nbytes

    def _nixl_write(self, nbytes: int) -> None:
        """PtoP WRITE of the already-staged local buffer prefix."""
        self._nixl_write_nbytes(nbytes)

    def transfer_and_speculate(
        self,
        hidden_states: torch.Tensor,
        *,
        req_ids: list[str],
        num_speculative_tokens: int,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_scheduled_tokens: np.ndarray | torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor,
        idx_mapping: torch.Tensor,
        src_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor | None:
        """Blocking NIXL + SPECulate (legacy). Prefer begin/finish for overlap."""
        self.begin_speculate(
            hidden_states,
            req_ids=req_ids,
            num_speculative_tokens=num_speculative_tokens,
            positions=positions,
            query_start_loc=query_start_loc,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            num_scheduled_tokens=num_scheduled_tokens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            idx_mapping=idx_mapping,
            src_ready_event=src_ready_event,
        )
        return self.finish_speculate()

    def _join_bg(self) -> None:
        """Wait for background NIXL/SPECulate-send thread; raise if it failed."""
        t = self._bg_thread
        if t is not None:
            t.join()
            self._bg_thread = None
        err = self._bg_error
        self._bg_error = None
        if err is not None:
            raise RuntimeError(f"DFlash HS NIXL background transfer failed: {err}") from err

    def _alloc_meta_pins(self) -> None:
        """Pinned host buffers for non-blocking SPECulate meta DtoH.

        Allocate outside InferenceMode so bg-thread / non-IM paths can copy_
        into them after handshake ran under InferenceMode.
        """
        mt = self._max_tokens
        # Upper-bound reqs by max_tokens (decode is 1 token/req; prefill is fewer).
        with torch.inference_mode(False):
            self._meta_pin_positions = torch.empty(
                mt, dtype=torch.int64, pin_memory=True
            )
            self._meta_pin_num_sampled = torch.empty(
                mt, dtype=torch.int32, pin_memory=True
            )
            self._meta_pin_num_rejected = torch.empty(
                mt, dtype=torch.int32, pin_memory=True
            )
            self._meta_pin_last_sampled = torch.empty(
                mt, dtype=torch.int64, pin_memory=True
            )
            self._meta_pin_next_prefill = torch.empty(
                mt, dtype=torch.int64, pin_memory=True
            )
            self._meta_pin_temperature = torch.empty(
                mt, dtype=torch.float32, pin_memory=True
            )
            self._meta_pin_seeds = torch.empty(mt, dtype=torch.int64, pin_memory=True)

    def _enqueue_meta_dtoh(
        self,
        *,
        n_ctx: int,
        num_reqs: int,
        positions: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        idx_mapping: torch.Tensor | np.ndarray,
        src_ready_event: torch.cuda.Event | None,
    ) -> dict[str, torch.Tensor]:
        """Enqueue GPU→pinned DtoH on the copy stream; does not synchronize.

        Returns host tensor views into the pinned staging buffers. Valid only
        after ``_meta_done_event`` is synchronized. For async verify this runs
        on the bg transfer thread (never on the propose/kick thread).
        """
        assert self._copy_stream is not None
        assert self._meta_done_event is not None
        assert self._meta_pin_positions is not None
        assert self._meta_pin_num_sampled is not None
        assert self._meta_pin_num_rejected is not None
        assert self._meta_pin_last_sampled is not None
        assert self._meta_pin_next_prefill is not None
        assert self._meta_pin_temperature is not None
        assert self._meta_pin_seeds is not None

        if isinstance(idx_mapping, np.ndarray):
            idx = torch.as_tensor(
                idx_mapping[:num_reqs], dtype=torch.long, device=self.device
            )
        else:
            idx = idx_mapping[:num_reqs].long()

        # Wait only for HS/meta producers, not a full default-stream drain on
        # this thread. Sync of meta_done happens on the bg thread (async) or
        # explicitly after this returns (sync path).
        if src_ready_event is not None:
            self._copy_stream.wait_event(src_ready_event)
        else:
            ready = torch.cuda.Event()
            ready.record()
            self._copy_stream.wait_event(ready)

        def _gather(t: torch.Tensor) -> torch.Tensor:
            g = t.reshape(t.shape[0], -1)[idx, 0] if t.ndim > 1 else t[idx]
            return g.reshape(-1).contiguous()

        pin_pos = self._meta_pin_positions[:n_ctx]
        pin_ns = self._meta_pin_num_sampled[:num_reqs]
        pin_nr = self._meta_pin_num_rejected[:num_reqs]
        pin_ls = self._meta_pin_last_sampled[:num_reqs]
        pin_np = self._meta_pin_next_prefill[:num_reqs]
        pin_temp = self._meta_pin_temperature[:num_reqs]
        pin_seeds = self._meta_pin_seeds[:num_reqs]

        with torch.cuda.stream(self._copy_stream):
            # dtype/device normalize on GPU, then pinned copy_ (true async DtoH).
            pos_src = positions[:n_ctx].detach().to(
                dtype=pin_pos.dtype, device=self.device
            ).contiguous()
            pin_pos.copy_(pos_src, non_blocking=True)
            pos_src.record_stream(self._copy_stream)

            ns_src = (
                num_sampled[:num_reqs]
                .reshape(-1)
                .detach()
                .to(dtype=pin_ns.dtype, device=self.device)
                .contiguous()
            )
            pin_ns.copy_(ns_src, non_blocking=True)
            ns_src.record_stream(self._copy_stream)

            nr_src = (
                num_rejected[:num_reqs]
                .reshape(-1)
                .detach()
                .to(dtype=pin_nr.dtype, device=self.device)
                .contiguous()
            )
            pin_nr.copy_(nr_src, non_blocking=True)
            nr_src.record_stream(self._copy_stream)

            ls_src = _gather(last_sampled).to(dtype=pin_ls.dtype, device=self.device)
            pin_ls.copy_(ls_src, non_blocking=True)
            ls_src.record_stream(self._copy_stream)

            np_src = _gather(next_prefill_tokens).to(
                dtype=pin_np.dtype, device=self.device
            )
            pin_np.copy_(np_src, non_blocking=True)
            np_src.record_stream(self._copy_stream)

            temp_src = _gather(temperature).to(
                dtype=pin_temp.dtype, device=self.device
            )
            pin_temp.copy_(temp_src, non_blocking=True)
            temp_src.record_stream(self._copy_stream)

            seeds_src = _gather(seeds).to(dtype=pin_seeds.dtype, device=self.device)
            pin_seeds.copy_(seeds_src, non_blocking=True)
            seeds_src.record_stream(self._copy_stream)

            self._meta_done_event.record(self._copy_stream)

        return {
            "positions": pin_pos,
            "num_sampled": pin_ns,
            "num_rejected": pin_nr,
            "last_sampled": pin_ls,
            "next_prefill_tokens": pin_np,
            "temperature": pin_temp,
            "seeds": pin_seeds,
        }

    def begin_speculate(
        self,
        hidden_states: torch.Tensor,
        *,
        req_ids: list[str],
        num_speculative_tokens: int,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor | np.ndarray,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_scheduled_tokens: np.ndarray | torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor | np.ndarray,
        idx_mapping: torch.Tensor | np.ndarray,
        src_ready_event: torch.cuda.Event | None = None,
        defer_meta_sync: bool = False,
        verify_step: int = 0,
        kick_iter: int = 0,
        on_speculate_reply: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        """Kick NIXL on a side thread; return immediately.

        Default: pack+send SPECulate meta on this thread (sink CPU-prep ∥ PtoP),
        then bg stage → PtoP → HS_READY.

        With ``defer_meta_sync=True`` (async verify): the propose/kick thread
        only snapshots host-side meta and starts the bg thread — **no CUDA
        enqueue or stream sync**. Bg does meta DtoH → wait → SPECulate send →
        stage/PtoP/HS_READY so ``sample_tokens`` can return and the next
        ``execute_context`` can be submitted while GPU0 still drains.

        When ``on_speculate_reply`` is set (async verify), the bg thread also
        blocking-recvs the SPECulate reply and invokes the callback so drafts
        are stashed without waiting for the worker RPC thread to poll.
        """
        if self._bg_thread is not None or getattr(self, "_speculate_pending", False):
            raise RuntimeError(
                "DFlash HS NIXL: begin_speculate while prior transfer/reply pending"
            )
        if not self._handshook:
            self.handshake()

        with self._reply_lock:
            self._bg_reply = None
            self._reply_complete = False

        hs = hidden_states
        n_ctx = int(hs.shape[0])
        num_reqs = len(req_ids)
        req_ids_list = list(req_ids)
        reply_cb = on_speculate_reply
        # Sync path: host_meta already on pinned buffers after enqueue+sync.
        # Async path: deferred_pack carries host snapshots + GPU tensor refs;
        # bg runs _enqueue_meta_dtoh then encode/send.
        deferred_pack: dict[str, Any] | None = None
        host_meta_ready: dict[str, Any] | None = None
        frames: list[bytes] | None = None
        sent_speculate = False

        def _snapshot_host_meta() -> tuple[np.ndarray, Any, Any]:
            # Already-host fields only (numpy / CPU copies — no CUDA sync).
            if isinstance(num_scheduled_tokens, np.ndarray):
                nst = np.array(num_scheduled_tokens[:num_reqs], copy=True)
            else:
                nst_t = num_scheduled_tokens[:num_reqs]
                if nst_t.device.type != "cpu":
                    raise RuntimeError(
                        "DFlash HS NIXL meta_pack expects num_scheduled_tokens "
                        "on host (ndarray/CPU); GPU path would sync."
                    )
                nst = nst_t.detach().to(dtype=torch.int32).numpy()

            if isinstance(query_start_loc, np.ndarray):
                qsl_host: torch.Tensor | np.ndarray = np.array(
                    query_start_loc[: num_reqs + 1], copy=True
                )
            else:
                qsl_t = query_start_loc[: num_reqs + 1]
                if qsl_t.device.type != "cpu":
                    raise RuntimeError(
                        "DFlash HS NIXL meta_pack expects query_start_loc on host"
                    )
                qsl_host = qsl_t.detach().contiguous()

            if isinstance(seq_lens_cpu_upper_bound, np.ndarray):
                seq_host: torch.Tensor | np.ndarray = np.array(
                    seq_lens_cpu_upper_bound[:num_reqs], copy=True
                )
            else:
                seq_t = seq_lens_cpu_upper_bound[:num_reqs]
                if seq_t.device.type != "cpu":
                    raise RuntimeError(
                        "DFlash HS NIXL meta_pack expects "
                        "seq_lens_cpu_upper_bound on host"
                    )
                seq_host = seq_t.detach().contiguous()
            return nst, qsl_host, seq_host

        if self.draft_enabled:
            if defer_meta_sync:
                # Propose thread: host snapshot only. Any CUDA in meta_pack here
                # historically became cudaStreamSynchronize on the compute stream
                # (~tens of ms) and blocked the next execute_context submit.
                torch.cuda.nvtx.range_push("dflash_hs_nixl_meta_pack")
                try:
                    nst, qsl_host, seq_host = _snapshot_host_meta()
                    deferred_pack = {
                        "req_ids": req_ids_list,
                        "num_ctx_tokens": n_ctx,
                        "num_speculative_tokens": num_speculative_tokens,
                        "query_start_loc": qsl_host,
                        "num_scheduled_tokens": nst,
                        "seq_lens_cpu_upper_bound": seq_host,
                        "positions": positions,
                        "num_sampled": num_sampled,
                        "num_rejected": num_rejected,
                        "last_sampled": last_sampled,
                        "next_prefill_tokens": next_prefill_tokens,
                        "temperature": temperature,
                        "seeds": seeds,
                        "idx_mapping": idx_mapping,
                        "verify_step": int(verify_step),
                        "kick_iter": int(kick_iter),
                    }
                finally:
                    torch.cuda.nvtx.range_pop()
                # Bg will send; treat as in-flight so HS_READY is still emitted.
                sent_speculate = True
            else:
                torch.cuda.nvtx.range_push("dflash_hs_nixl_meta_pack")
                try:
                    nst, qsl_host, seq_host = _snapshot_host_meta()
                    host_meta_ready = self._enqueue_meta_dtoh(
                        n_ctx=n_ctx,
                        num_reqs=num_reqs,
                        positions=positions,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                        last_sampled=last_sampled,
                        next_prefill_tokens=next_prefill_tokens,
                        temperature=temperature,
                        seeds=seeds,
                        idx_mapping=idx_mapping,
                        src_ready_event=src_ready_event,
                    )
                    assert self._meta_done_event is not None
                    self._meta_done_event.synchronize()
                    frames = encode_speculate_request(
                        req_ids=req_ids_list,
                        num_ctx_tokens=n_ctx,
                        num_speculative_tokens=num_speculative_tokens,
                        query_start_loc=qsl_host,
                        num_scheduled_tokens=nst,
                        seq_lens_cpu_upper_bound=seq_host,
                        verify_step=int(verify_step),
                        kick_iter=int(kick_iter),
                        **host_meta_ready,
                    )
                finally:
                    torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_send")
                try:
                    self._sock.send_multipart(frames)
                    self._speculate_pending = True
                    sent_speculate = True
                finally:
                    torch.cuda.nvtx.range_pop()

        def _worker() -> None:
            try:
                torch.cuda.set_device(self.device)
                if deferred_pack is not None:
                    assert self._meta_done_event is not None
                    # Full meta DtoH + wait on bg — not under async_kick.
                    torch.cuda.nvtx.range_push("dflash_hs_nixl_meta_pack")
                    try:
                        host_meta = self._enqueue_meta_dtoh(
                            n_ctx=int(deferred_pack["num_ctx_tokens"]),
                            num_reqs=len(deferred_pack["req_ids"]),
                            positions=deferred_pack["positions"],
                            num_sampled=deferred_pack["num_sampled"],
                            num_rejected=deferred_pack["num_rejected"],
                            last_sampled=deferred_pack["last_sampled"],
                            next_prefill_tokens=deferred_pack["next_prefill_tokens"],
                            temperature=deferred_pack["temperature"],
                            seeds=deferred_pack["seeds"],
                            idx_mapping=deferred_pack["idx_mapping"],
                            src_ready_event=src_ready_event,
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("dflash_hs_nixl_meta_pack_wait")
                    try:
                        self._meta_done_event.synchronize()
                        frames_bg = encode_speculate_request(
                            req_ids=deferred_pack["req_ids"],
                            num_ctx_tokens=deferred_pack["num_ctx_tokens"],
                            num_speculative_tokens=deferred_pack[
                                "num_speculative_tokens"
                            ],
                            query_start_loc=deferred_pack["query_start_loc"],
                            num_scheduled_tokens=deferred_pack["num_scheduled_tokens"],
                            seq_lens_cpu_upper_bound=deferred_pack[
                                "seq_lens_cpu_upper_bound"
                            ],
                            verify_step=int(deferred_pack.get("verify_step", 0)),
                            kick_iter=int(deferred_pack.get("kick_iter", 0)),
                            **host_meta,
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_send")
                    try:
                        self._sock.send_multipart(frames_bg)
                        self._speculate_pending = True
                    finally:
                        torch.cuda.nvtx.range_pop()

                with torch.inference_mode():
                    torch.cuda.nvtx.range_push("dflash_hs_nixl_stage")
                    try:
                        nbytes = self._stage_hiddens(
                            hs, src_ready_event=src_ready_event
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()

                    if nbytes > 0:
                        self._nixl_write_nbytes(nbytes)

                if sent_speculate:
                    torch.cuda.nvtx.range_push("dflash_hs_nixl_hs_ready")
                    try:
                        self._sock.send_multipart(
                            [encode_hs_ready(num_ctx_tokens=n_ctx)]
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()

                    # Async path: recv SPECulate on this thread so the worker
                    # RPC thread (possibly stuck in execute_model) is not the
                    # only place that can take the ZMQ reply.
                    if defer_meta_sync:
                        torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_recv")
                        try:
                            reply_frames = self._sock.recv_multipart()
                            remote = decode_speculate_response(reply_frames)
                        finally:
                            torch.cuda.nvtx.range_pop()
                        with self._reply_lock:
                            self._speculate_pending = False
                            self._reply_complete = True
                            if reply_cb is not None:
                                # Callback stashes/publishes; do not keep a copy.
                                self._bg_reply = None
                            else:
                                self._bg_reply = remote
                        if reply_cb is not None:
                            reply_cb(remote)
            except BaseException as e:
                self._bg_error = e
                if sent_speculate:
                    try:
                        self._sock.send_multipart(
                            [
                                encode_hs_ready(
                                    num_ctx_tokens=n_ctx, error=str(e)
                                )
                            ]
                        )
                    except Exception:
                        pass
                    with self._reply_lock:
                        self._speculate_pending = False
                        self._reply_complete = True
                        self._bg_reply = None

        self._bg_thread = threading.Thread(
            target=_worker, name="dflash-hs-nixl-xfer", daemon=True
        )
        self._bg_thread.start()

    def begin_transfer(
        self,
        hidden_states: torch.Tensor,
        *,
        src_ready_event: torch.cuda.Event | None = None,
    ) -> None:
        """Async HS staging+NIXL only (no SPECulate). Returns immediately."""
        if self._bg_thread is not None or self._speculate_pending:
            raise RuntimeError(
                "DFlash HS NIXL: begin_transfer while prior transfer/reply pending"
            )
        if not self._handshook:
            self.handshake()
        hs = hidden_states

        def _worker() -> None:
            try:
                torch.cuda.set_device(self.device)
                with torch.inference_mode():
                    self.transfer(hs, src_ready_event=src_ready_event)
            except BaseException as e:
                self._bg_error = e

        self._bg_thread = threading.Thread(
            target=_worker, name="dflash-hs-nixl-xfer", daemon=True
        )
        self._bg_thread.start()

    def finish_speculate(self) -> torch.Tensor | None:
        """Join bg send (if any), then wait for SPECulate reply. None if draft off."""
        self._join_bg()
        with self._reply_lock:
            if self._bg_reply is not None:
                remote = self._bg_reply
                self._bg_reply = None
                self._speculate_pending = False
                return remote
            if self._reply_complete:
                # Callback path already consumed the reply.
                self._speculate_pending = False
                return None
        if not self.draft_enabled or not getattr(self, "_speculate_pending", False):
            self._speculate_pending = False
            return None
        torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_wait")
        try:
            reply = self._sock.recv_multipart()
            return decode_speculate_response(reply)
        finally:
            with self._reply_lock:
                self._speculate_pending = False
                self._reply_complete = True
            torch.cuda.nvtx.range_pop()

    def try_finish_speculate(self) -> torch.Tensor | None:
        """Non-blocking SPECulate recv. None if not ready / draft off / no pending.

        Does not join a still-running bg transfer thread (sink only replies after
        HS_READY, so a live bg means the reply cannot be ready yet). When the bg
        thread has exited, joins to surface transfer errors, then takes a
        bg-stashed reply or NOBLOCK recv.
        """
        t = self._bg_thread
        if t is not None and t.is_alive():
            return None
        self._join_bg()
        with self._reply_lock:
            if self._bg_reply is not None:
                remote = self._bg_reply
                self._bg_reply = None
                self._speculate_pending = False
                return remote
            if self._reply_complete:
                self._speculate_pending = False
                return None
        if not self.draft_enabled or not getattr(self, "_speculate_pending", False):
            return None
        torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_try")
        try:
            try:
                reply = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
            with self._reply_lock:
                self._speculate_pending = False
                self._reply_complete = True
            return decode_speculate_response(reply)
        finally:
            torch.cuda.nvtx.range_pop()

    def free(self, req_ids: list[str]) -> torch.Tensor | None:
        """FREE draft-side KV. Returns a SPECulate reply drained to clear the socket.

        Callers (async verify) must stash that reply for still-running reqs — do not
        drop it. Returning None means no in-flight SPECulate was drained.
        """
        if not req_ids:
            return None
        if not self._handshook:
            self.handshake()
        drained: torch.Tensor | None = None
        if not self.draft_enabled:
            self._join_bg()
            return None
        # Drain any in-flight SPECulate so FREE is not interleaved on the socket.
        if self._bg_thread is not None or self._speculate_pending:
            try:
                drained = self.finish_speculate()
            except Exception as e:
                logger.warning("DFlash HS NIXL drain before FREE failed: %s", e)
                self._speculate_pending = False
                self._bg_thread = None
                self._bg_error = None
        self._sock.send_multipart([encode_free_request(list(req_ids))])
        reply = self._sock.recv_multipart()
        meta = json.loads(reply[0].decode("utf-8"))
        if meta.get("error"):
            raise RuntimeError(f"FREE failed: {meta}")
        if meta.get("cmd") != CMD_FREE:
            raise RuntimeError(f"Unexpected FREE reply: {meta}")
        return drained

    def profile(self, start: bool) -> torch.Tensor | None:
        """Mirror verify cudaProfilerStart/Stop onto the sink GPU (nsys API range).

        Combined ``--capture-range=cudaProfilerApi`` only records CUDA contexts
        that call the API; without this, GPU1 shows PtoP (verify-initiated) but
        no draft kernels.

        Returns a SPECulate reply drained to clear the socket (or None). Async
        verify must stash it for still-running requests.
        """
        if not self._handshook:
            self.handshake()
        drained: torch.Tensor | None = None
        # DEALER is in-order: never PROFILE while a SPECulate reply is pending.
        if self._bg_thread is not None or self._speculate_pending:
            try:
                drained = self.finish_speculate()
            except Exception as e:
                logger.warning("DFlash HS NIXL drain before PROFILE failed: %s", e)
                self._speculate_pending = False
                self._bg_thread = None
                self._bg_error = None
        try:
            self._sock.send_multipart([encode_profile_request(start=start)])
            reply = self._sock.recv_multipart()
            meta = json.loads(reply[0].decode("utf-8"))
            if meta.get("error") or not meta.get("ok", True):
                logger.warning("DFlash HS NIXL PROFILE ack failed: %s", meta)
            elif meta.get("cmd") != CMD_PROFILE:
                logger.warning("Unexpected PROFILE reply: %s", meta)
        except zmq.ZMQError as e:
            logger.warning("DFlash HS NIXL PROFILE failed: %s", e)
        return drained

    def close(self) -> None:
        try:
            self._join_bg()
        except Exception as e:
            logger.warning("DFlash HS NIXL bg join on close failed: %s", e)
        try:
            if self._agent is not None and self._local_reg is not None:
                self._agent.deregister_memory(self._local_reg)
        except Exception as e:
            logger.warning("DFlash HS NIXL deregister failed: %s", e)
        self._local_reg = None
        self._agent = None
        try:
            self._sock.close(0)
        except Exception:
            pass
