# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU sink: NIXL HS staging + optional draft prepare/forward/sample."""

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
from vllm.v1.spec_decode.dflash_hs_nixl.protocol import (
    CMD_FREE,
    CMD_HELLO,
    CMD_PROFILE,
    CMD_SPECULATE,
    decode_free_request,
    decode_hs_ready,
    decode_speculate_request,
    encode_free_ack,
    encode_profile_ack,
    encode_speculate_response,
)

logger = init_logger(__name__)

_DRAFT_AGENT_NAME = "dflash-hs-nixl-sink"
_MEM_TYPE = "VRAM"


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
    """Owns NIXL-registered staging; optional draft runner for SPECulate/FREE."""

    def __init__(
        self,
        bind: str,
        device: torch.device,
        draft_runner: Any | None = None,
    ):
        self.bind = bind
        self.device = device
        self._draft_runner = draft_runner
        self._agent = _make_agent(_DRAFT_AGENT_NAME)
        self._staging: torch.Tensor | None = None
        self._reg: Any = None
        self._speculate_iter: int = 0
        self._max_tokens = 0
        self._hidden_size = 0
        self._dtype = torch.bfloat16

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind)
        logger.info(
            "DFlash HS NIXL sink listening on %s (device=%s draft=%s)",
            bind,
            device,
            draft_runner is not None,
        )

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
        if self._draft_runner is not None:
            # Prefer draft HS buffer as NIXL target → no staging→draft DtoD.
            hs = self._draft_runner.hidden_states
            if hs.shape[0] < self._max_tokens or hs.shape[1] != self._hidden_size:
                raise RuntimeError(
                    f"Draft hidden_states shape {tuple(hs.shape)} incompatible with "
                    f"verify max_tokens={self._max_tokens} H={self._hidden_size}"
                )
            if hs.dtype != self._dtype:
                raise RuntimeError(
                    f"Draft dtype {hs.dtype} != verify dtype {self._dtype}"
                )
            self._staging = hs
            logger.info(
                "DFlash HS NIXL sink using draft hidden_states as staging "
                "(no extra DtoD)"
            )
        else:
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
            "cmd": CMD_HELLO,
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
            "draft_enabled": self._draft_runner is not None,
        }
        self._sock.send_multipart([identity, json.dumps(reply).encode("utf-8")])
        logger.info("DFlash HS NIXL sink HELLO complete")

    def _handle_speculate(self, identity: bytes, frames: list[bytes]) -> None:
        if self._draft_runner is None:
            err = {"error": "draft runner not enabled on sink"}
            self._sock.send_multipart([identity, json.dumps(err).encode("utf-8")])
            return
        header, tensors = decode_speculate_request(frames)
        self._speculate_iter += 1
        si = self._speculate_iter
        vi = int(header.get("verify_step", 0) or 0)
        ki = int(header.get("kick_iter", 0) or 0)
        n_reqs = len(header.get("req_ids", []))
        nvtx_suffix = f"_si{si}_vi{vi}_ki{ki}_n{n_reqs}"
        self._draft_runner._nvtx_suffix = nvtx_suffix

        def _wait_hiddens_ready() -> None:
            # Probe sends SPECulate meta first, then PtoP, then HS_READY.
            # CPU prep above overlaps the transfer; block here before touching HS.
            torch.cuda.nvtx.range_push(f"dflash_sink_hs_wait{nvtx_suffix}")
            try:
                ready = self._sock.recv_multipart()
                if len(ready) < 2:
                    raise RuntimeError("HS_READY: short multipart")
                # ROUTER: [identity, payload]
                decode_hs_ready(ready[1])
            finally:
                torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"dflash_hs_nixl_speculate{nvtx_suffix}")
        try:
            draft_tokens = self._draft_runner.speculate(
                req_ids=list(header["req_ids"]),
                num_ctx_tokens=int(header["num_ctx_tokens"]),
                tensors=tensors,
                wait_hiddens_ready=_wait_hiddens_ready,
            )
            reply = encode_speculate_response(draft_tokens)
            self._sock.send_multipart([identity, *reply])
        finally:
            torch.cuda.nvtx.range_pop()

    def _handle_free(self, identity: bytes, payload: bytes) -> None:
        if self._draft_runner is None:
            err = {"error": "draft runner not enabled on sink"}
            self._sock.send_multipart([identity, json.dumps(err).encode("utf-8")])
            return
        req_ids = decode_free_request(payload)
        self._draft_runner.free(req_ids)
        self._sock.send_multipart([identity, encode_free_ack()])

    def _handle_profile(self, identity: bytes, meta: dict[str, Any]) -> None:
        """cudaProfilerStart/Stop so nsys --capture-range=cudaProfilerApi
        records the sink GPU in the same window as verify."""
        start = bool(meta.get("start", False))
        try:
            import torch.cuda.profiler as cuda_profiler

            if start:
                cuda_profiler.start()
            else:
                cuda_profiler.stop()
            logger.info(
                "DFlash HS NIXL sink cuda profiler %s",
                "start" if start else "stop",
            )
            self._sock.send_multipart(
                [identity, encode_profile_ack(start=start, ok=True)]
            )
        except Exception as e:
            logger.warning("DFlash HS NIXL sink cuda profiler failed: %s", e)
            self._sock.send_multipart(
                [identity, encode_profile_ack(start=start, ok=False)]
            )

    def serve(self) -> None:
        while True:
            frames = self._sock.recv_multipart()
            if len(frames) < 2:
                continue
            identity, rest = frames[0], frames[1:]
            payload = rest[0]
            try:
                # Multi-frame SPECulate: JSON header + tensor bodies.
                peek = json.loads(payload.decode("utf-8"))
            except Exception as e:
                err = {"error": f"bad json: {e}"}
                self._sock.send_multipart(
                    [identity, json.dumps(err).encode("utf-8")]
                )
                continue
            try:
                cmd = peek.get("cmd")
                if cmd == CMD_HELLO:
                    self._handle_hello(identity, peek)
                elif cmd == CMD_SPECULATE:
                    self._handle_speculate(identity, rest)
                elif cmd == CMD_FREE:
                    self._handle_free(identity, payload)
                elif cmd == CMD_PROFILE:
                    self._handle_profile(identity, peek)
                else:
                    err = {"error": f"unknown cmd={cmd!r}"}
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
    # So nsys shows "VLLM::DFlashHsNixlSink" instead of generic "python"
    # (draft kernels live under this process, not VLLM::EngineCore).
    from vllm.utils.system_utils import set_process_title

    set_process_title("DFlashHsNixlSink")

    parser = argparse.ArgumentParser(
        description="DFlash HS NIXL sink (+ optional draft forward)."
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
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--block-size", type=int, default=16)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown sink args: %s", unknown)

    if not torch.cuda.is_available():
        logger.error("CUDA is required for the HS NIXL sink")
        return 1
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    draft_runner = None
    want_draft = args.draft_model is not None
    if want_draft:
        if not args.target_model or args.num_speculative_tokens is None:
            logger.error(
                "--draft-model requires --target-model and --num-speculative-tokens"
            )
            return 1
        from vllm.v1.spec_decode.dflash_hs_nixl.draft_runner import DFlashDraftRunner

        logger.info(
            "Loading DFlash draft runner: draft=%s target=%s K=%d",
            args.draft_model,
            args.target_model,
            args.num_speculative_tokens,
        )
        draft_runner = DFlashDraftRunner(
            draft_model=args.draft_model,
            target_model=args.target_model,
            num_speculative_tokens=args.num_speculative_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            block_size=args.block_size,
            device=device,
        )

    sink = HsNixlSink(args.bind, device, draft_runner=draft_runner)

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
