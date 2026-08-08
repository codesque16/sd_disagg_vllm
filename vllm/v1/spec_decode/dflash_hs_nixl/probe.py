# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side NIXL client: WRITE hiddens + optional SPECulate/FREE over ZMQ."""

from __future__ import annotations

import base64
import json
import threading
import time
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
    ) -> None:
        """Kick NIXL on a side thread; return immediately.

        On this thread: pack+send SPECulate meta (sink can CPU-prep early).
        Bg: stage → PtoP → HS_READY. Sink waits on HS_READY before precompute.
        """
        if self._bg_thread is not None or getattr(self, "_speculate_pending", False):
            raise RuntimeError(
                "DFlash HS NIXL: begin_speculate while prior transfer/reply pending"
            )
        if not self._handshook:
            self.handshake()

        hs = hidden_states
        n_ctx = int(hs.shape[0])
        num_reqs = len(req_ids)
        req_ids_list = list(req_ids)
        sent_speculate = False

        # Pack + send on the caller thread *before* local draft / NIXL so:
        # - SPECulate reaches the sink early (CPU prep ∥ PtoP)
        # - _speculate_pending is set before bg runs (PROFILE can drain safely)
        if self.draft_enabled:
            torch.cuda.nvtx.range_push("dflash_hs_nixl_meta_pack")
            try:
                if isinstance(num_scheduled_tokens, np.ndarray):
                    nst = np.array(num_scheduled_tokens[:num_reqs], copy=True)
                else:
                    nst = (
                        num_scheduled_tokens[:num_reqs]
                        .detach()
                        .to(dtype=torch.int32, device="cpu")
                        .numpy()
                    )

                if isinstance(query_start_loc, np.ndarray):
                    qsl_host: torch.Tensor | np.ndarray = np.array(
                        query_start_loc[: num_reqs + 1], copy=True
                    )
                else:
                    qsl_host = query_start_loc[: num_reqs + 1].detach().cpu()

                if isinstance(seq_lens_cpu_upper_bound, np.ndarray):
                    seq_host: torch.Tensor | np.ndarray = np.array(
                        seq_lens_cpu_upper_bound[:num_reqs], copy=True
                    )
                elif (
                    isinstance(seq_lens_cpu_upper_bound, torch.Tensor)
                    and seq_lens_cpu_upper_bound.device.type == "cpu"
                ):
                    seq_host = (
                        seq_lens_cpu_upper_bound[:num_reqs].detach().contiguous()
                    )
                else:
                    seq_host = seq_lens_cpu_upper_bound[:num_reqs].detach().cpu()

                if isinstance(idx_mapping, np.ndarray):
                    idx = torch.as_tensor(
                        idx_mapping[:num_reqs], dtype=torch.long, device=self.device
                    )
                else:
                    idx = idx_mapping[:num_reqs].long()

                assert self._copy_stream is not None
                assert self._meta_done_event is not None
                ready = torch.cuda.Event()
                ready.record()
                self._copy_stream.wait_event(ready)

                def _gather_host(t: torch.Tensor) -> torch.Tensor:
                    g = (
                        t.reshape(t.shape[0], -1)[idx, 0]
                        if t.ndim > 1
                        else t[idx]
                    )
                    return g.reshape(-1).contiguous()

                with torch.cuda.stream(self._copy_stream):
                    positions_host = (
                        positions[:n_ctx]
                        .detach()
                        .contiguous()
                        .to("cpu", non_blocking=True)
                    )
                    num_sampled_host = (
                        num_sampled[:num_reqs]
                        .reshape(-1)
                        .contiguous()
                        .to("cpu", non_blocking=True)
                    )
                    num_rejected_host = (
                        num_rejected[:num_reqs]
                        .reshape(-1)
                        .contiguous()
                        .to("cpu", non_blocking=True)
                    )
                    last_sampled_host = _gather_host(last_sampled).to(
                        "cpu", non_blocking=True
                    )
                    next_prefill_host = _gather_host(next_prefill_tokens).to(
                        "cpu", non_blocking=True
                    )
                    temperature_host = _gather_host(temperature).to(
                        "cpu", non_blocking=True
                    )
                    seeds_host = _gather_host(seeds).to("cpu", non_blocking=True)
                    self._meta_done_event.record(self._copy_stream)
                # Only wait for meta DtoH (copy stream), not the whole default stream.
                self._meta_done_event.synchronize()

                frames = encode_speculate_request(
                    req_ids=req_ids_list,
                    num_ctx_tokens=n_ctx,
                    num_speculative_tokens=num_speculative_tokens,
                    positions=positions_host,
                    query_start_loc=qsl_host,
                    num_sampled=num_sampled_host,
                    num_rejected=num_rejected_host,
                    last_sampled=last_sampled_host,
                    next_prefill_tokens=next_prefill_host,
                    temperature=temperature_host,
                    seeds=seeds_host,
                    num_scheduled_tokens=nst,
                    seq_lens_cpu_upper_bound=seq_host,
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
                        self._speculate_pending = False

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
        if not self.draft_enabled or not getattr(self, "_speculate_pending", False):
            self._speculate_pending = False
            return None
        torch.cuda.nvtx.range_push("dflash_hs_nixl_speculate_wait")
        try:
            reply = self._sock.recv_multipart()
            return decode_speculate_response(reply)
        finally:
            self._speculate_pending = False
            torch.cuda.nvtx.range_pop()

    def free(self, req_ids: list[str]) -> None:
        if not req_ids:
            return
        if not self._handshook:
            self.handshake()
        if not self.draft_enabled:
            self._join_bg()
            return
        # Drain any in-flight SPECulate so FREE is not interleaved on the socket.
        if self._bg_thread is not None or self._speculate_pending:
            try:
                self.finish_speculate()
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

    def profile(self, start: bool) -> None:
        """Mirror verify cudaProfilerStart/Stop onto the sink GPU (nsys API range).

        Combined ``--capture-range=cudaProfilerApi`` only records CUDA contexts
        that call the API; without this, GPU1 shows PtoP (verify-initiated) but
        no draft kernels.
        """
        if not self._handshook:
            self.handshake()
        # DEALER is in-order: never PROFILE while a SPECulate reply is pending.
        if self._bg_thread is not None or self._speculate_pending:
            try:
                self.finish_speculate()
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
