# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side NIXL client: blocking WRITE of context hiddens to a sink."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import torch
import zmq

from vllm.distributed.nixl_utils import NixlWrapper, is_nixl_available, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)

_VERIFY_AGENT_NAME = "dflash-hs-nixl-verify"
_MEM_TYPE = "VRAM"
_CMD_HELLO = "hello"


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
    """Handshake once, then blocking NIXL WRITE each propose."""

    def __init__(
        self,
        address: str,
        *,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        timeout_ms: int = 180_000,
    ):
        self._address = address
        self._max_tokens = int(max_tokens)
        self._hidden_size = int(hidden_size)
        self._dtype = dtype
        self.device = device
        self._timeout_s = max(timeout_ms / 1000.0, 1.0)
        self._handshook = False

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        hello_timeout_ms = min(int(timeout_ms), 30_000)
        self._sock.setsockopt(zmq.RCVTIMEO, hello_timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, hello_timeout_ms)
        self._sock.connect(address)

        self._agent: Any = None
        self._peer_name: str | None = None
        self._local_buf: torch.Tensor | None = None
        self._local_reg: Any = None
        self._remote_addr = 0
        self._remote_device_id = 0

    def handshake(self) -> None:
        if self._handshook:
            return
        self._agent = _make_agent(_VERIFY_AGENT_NAME)
        req = {
            "cmd": _CMD_HELLO,
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
        if meta.get("cmd") != _CMD_HELLO:
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

        self._local_buf = torch.zeros(
            self._max_tokens,
            self._hidden_size,
            dtype=self._dtype,
            device=self.device,
        )
        self._local_reg = self._agent.register_memory(self._local_buf)
        if not self._local_reg:
            raise RuntimeError("DFlash HS NIXL verify register_memory failed")

        self._handshook = True
        logger.info(
            "DFlash HS NIXL handshake ok: addr=%s max_tokens=%d H=%d peer=%s",
            self._address,
            self._max_tokens,
            self._hidden_size,
            self._peer_name,
        )

    def transfer(self, hidden_states: torch.Tensor) -> None:
        """Blocking: staging DtoD + NIXL PtoP WRITE, wait DONE."""
        if not self._handshook:
            self.handshake()
        assert self._agent is not None
        assert self._local_buf is not None
        assert self._peer_name is not None

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
            # Staging DtoD then NIXL PtoP; no CUDA sync (copy is fire-and-forget
            # w.r.t. the CPU — only NIXL DONE is waited below).
            self._local_buf[:n_ctx].copy_(hiddens)

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
            st = self._agent.transfer(handle)
            if st == "ERR":
                self._agent.release_xfer_handle(handle)
                raise RuntimeError("DFlash HS NIXL WRITE transfer failed")
            if st != "DONE":
                _wait_xfer_done(self._agent, handle, timeout_s=self._timeout_s)
            self._agent.release_xfer_handle(handle)
        finally:
            torch.cuda.nvtx.range_pop()

    def close(self) -> None:
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
