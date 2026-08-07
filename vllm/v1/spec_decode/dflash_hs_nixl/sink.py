# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU sink: register VRAM staging and accept NIXL WRITEs of hiddens."""

from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
from typing import Any

import torch
import zmq

from vllm.distributed.nixl_utils import NixlWrapper, is_nixl_available, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)

_DRAFT_AGENT_NAME = "dflash-hs-nixl-sink"
_MEM_TYPE = "VRAM"
_CMD_HELLO = "hello"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data)


def _require_nixl() -> None:
    if not is_nixl_available() or NixlWrapper is None or nixl_agent_config is None:
        raise RuntimeError("dflash_hs_nixl_sink requires the nixl (or rixl) package.")


def _make_agent(name: str) -> Any:
    _require_nixl()
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        backends=["UCX"],
        capture_telemetry=True,
    )
    return NixlWrapper(name, cfg)


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name, torch.bfloat16)


class HsNixlSink:
    """Owns NIXL-registered staging; serves HELLO over ZMQ."""

    def __init__(self, bind: str, device: torch.device):
        self.bind = bind
        self.device = device
        self._agent = _make_agent(_DRAFT_AGENT_NAME)
        self._staging: torch.Tensor | None = None
        self._reg: Any = None
        self._max_tokens = 0
        self._hidden_size = 0
        self._dtype = torch.bfloat16

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind)
        logger.info("DFlash HS NIXL sink listening on %s (device=%s)", bind, device)

    def _ensure_staging(
        self, max_tokens: int, hidden_size: int, dtype: torch.dtype
    ) -> None:
        if self._staging is not None:
            if max_tokens > self._max_tokens or hidden_size != self._hidden_size:
                raise RuntimeError(
                    f"Staging already allocated "
                    f"(max_tokens={self._max_tokens}, H={self._hidden_size}); "
                    f"cannot resize to max_tokens={max_tokens}, H={hidden_size}"
                )
            return
        self._max_tokens = int(max_tokens)
        self._hidden_size = int(hidden_size)
        self._dtype = dtype
        self._staging = torch.zeros(
            self._max_tokens,
            self._hidden_size,
            dtype=self._dtype,
            device=self.device,
        )
        self._reg = self._agent.register_memory(self._staging)
        if not self._reg:
            raise RuntimeError("DFlash HS NIXL sink register_memory failed")
        logger.info(
            "DFlash HS NIXL sink staging ready: max_tokens=%d H=%d dtype=%s "
            "nbytes=%d device_id=%d",
            self._max_tokens,
            self._hidden_size,
            self._dtype,
            self._staging.numel() * self._staging.element_size(),
            int(self._staging.get_device()),
        )

    def _handle_hello(self, identity: bytes, meta: dict[str, Any]) -> None:
        self._agent.add_remote_agent(_unb64(meta["agent_metadata"]))
        self._ensure_staging(
            int(meta["max_tokens"]),
            int(meta["hidden_size"]),
            _dtype_from_name(str(meta.get("dtype", "bfloat16"))),
        )
        assert self._staging is not None
        reply = {
            "cmd": _CMD_HELLO,
            "agent_metadata": _b64(self._agent.get_agent_metadata()),
            "max_tokens": self._max_tokens,
            "hidden_size": self._hidden_size,
            "dtype": str(self._dtype).removeprefix("torch."),
            "hidden_addr": int(self._staging.data_ptr()),
            "hidden_nbytes": int(
                self._staging.numel() * self._staging.element_size()
            ),
            "device_id": int(self._staging.get_device()),
            "mem_type": _MEM_TYPE,
        }
        self._sock.send_multipart([identity, json.dumps(reply).encode("utf-8")])
        logger.info("DFlash HS NIXL sink HELLO complete")

    def serve(self) -> None:
        while True:
            frames = self._sock.recv_multipart()
            if len(frames) < 2:
                continue
            identity, payload = frames[0], frames[-1]
            try:
                meta = json.loads(payload.decode("utf-8"))
            except Exception as e:
                err = {"error": f"bad json: {e}"}
                self._sock.send_multipart(
                    [identity, json.dumps(err).encode("utf-8")]
                )
                continue
            try:
                if meta.get("cmd") == _CMD_HELLO:
                    self._handle_hello(identity, meta)
                else:
                    err = {"error": f"unknown cmd={meta.get('cmd')!r}"}
                    self._sock.send_multipart(
                        [identity, json.dumps(err).encode("utf-8")]
                    )
            except Exception as e:
                logger.exception("DFlash HS NIXL sink handler failed")
                err = {"error": str(e)}
                self._sock.send_multipart(
                    [identity, json.dumps(err).encode("utf-8")]
                )

    def close(self) -> None:
        try:
            if self._agent is not None and self._reg is not None:
                self._agent.deregister_memory(self._reg)
        except Exception as e:
            logger.warning("DFlash HS NIXL sink deregister failed: %s", e)
        self._reg = None
        try:
            self._sock.close(0)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DFlash hidden-state NIXL sink (Milestone 0 probe)."
    )
    parser.add_argument(
        "--bind",
        type=str,
        required=True,
        help="ZMQ bind address, e.g. tcp://0.0.0.0:50051",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device visible in this process (use CUDA_VISIBLE_DEVICES).",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown sink args: %s", unknown)

    if not torch.cuda.is_available():
        logger.error("CUDA is required for the HS NIXL sink")
        return 1
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    sink = HsNixlSink(args.bind, device)

    def _stop(*_args: object) -> None:
        logger.info("Shutting down DFlash HS NIXL sink")
        sink.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    sink.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
