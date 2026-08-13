# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ wire helpers for DFlash HS NIXL SPECulate / FREE (meta + small tensors)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

CMD_HELLO = "hello"
CMD_SPECULATE = "speculate"
CMD_HS_READY = "hs_ready"
CMD_FREE = "free"
CMD_PROFILE = "profile"

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int32: "int32",
    torch.int64: "int64",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(f"Unsupported dtype for DFlash HS NIXL wire: {dtype}")
    return _DTYPE_TO_STR[dtype]


def _as_host_tensor(
    value: torch.Tensor | np.ndarray, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Normalize to a contiguous CPU torch tensor (no CUDA sync)."""
    if isinstance(value, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(value))
        return t.to(dtype=dtype) if dtype is not None else t
    if value.device.type != "cpu":
        raise ValueError(
            "SPECulate encode expects host tensors/ndarray; "
            f"got device={value.device}"
        )
    t = value.detach().contiguous()
    return t.to(dtype=dtype) if dtype is not None else t


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a CPU tensor to raw bytes."""
    t = _as_host_tensor(tensor)
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def bytes_to_tensor(buf: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    if dtype == "bfloat16":
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }[dtype]
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def _tensor_frame_meta(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    t = _as_host_tensor(tensor)
    return {
        "name": name,
        "dtype": _dtype_str(t.dtype),
        "shape": list(t.shape),
        "nbytes": int(t.numel() * t.element_size()),
    }


def encode_speculate_request(
    *,
    req_ids: list[str],
    num_ctx_tokens: int,
    num_speculative_tokens: int,
    positions: torch.Tensor | np.ndarray,
    query_start_loc: torch.Tensor | np.ndarray,
    num_sampled: torch.Tensor | np.ndarray,
    num_rejected: torch.Tensor | np.ndarray,
    last_sampled: torch.Tensor | np.ndarray,
    next_prefill_tokens: torch.Tensor | np.ndarray,
    temperature: torch.Tensor | np.ndarray,
    seeds: torch.Tensor | np.ndarray,
    num_scheduled_tokens: np.ndarray | torch.Tensor,
    seq_lens_cpu_upper_bound: torch.Tensor | np.ndarray,
    verify_step: int = 0,
    kick_iter: int = 0,
) -> list[bytes]:
    """Pack SPECulate meta from host tensors/ndarray. Hiddens stay on NIXL.

    Callers must snapshot GPU meta to host *before* encode (and ideally before
    kicking NIXL / local draft) so this path never issues ``.cpu()``.
    """
    tensors: dict[str, torch.Tensor] = {
        "positions": _as_host_tensor(positions),
        "query_start_loc": _as_host_tensor(query_start_loc, dtype=torch.int32),
        "num_sampled": _as_host_tensor(num_sampled),
        "num_rejected": _as_host_tensor(num_rejected),
        "last_sampled": _as_host_tensor(last_sampled),
        "next_prefill_tokens": _as_host_tensor(next_prefill_tokens),
        "temperature": _as_host_tensor(temperature),
        "seeds": _as_host_tensor(seeds),
        "seq_lens_cpu_upper_bound": _as_host_tensor(seq_lens_cpu_upper_bound),
        "num_scheduled_tokens": _as_host_tensor(
            num_scheduled_tokens, dtype=torch.int32
        ),
    }

    header = {
        "cmd": CMD_SPECULATE,
        "req_ids": list(req_ids),
        "num_ctx_tokens": int(num_ctx_tokens),
        "num_speculative_tokens": int(num_speculative_tokens),
        "num_reqs": len(req_ids),
        "verify_step": int(verify_step),
        "kick_iter": int(kick_iter),
        "tensors": [_tensor_frame_meta(n, t) for n, t in tensors.items()],
    }
    frames = [json.dumps(header).encode("utf-8")]
    for t in tensors.values():
        frames.append(tensor_to_bytes(t))
    return frames


def decode_speculate_request(
    frames: list[bytes],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    header = json.loads(frames[0].decode("utf-8"))
    if header.get("cmd") != CMD_SPECULATE:
        raise ValueError(f"expected speculate, got {header.get('cmd')!r}")
    metas = header["tensors"]
    if len(frames) - 1 != len(metas):
        raise ValueError(
            f"speculate frame count mismatch: header={len(metas)} body={len(frames) - 1}"
        )
    tensors: dict[str, torch.Tensor] = {}
    for meta, buf in zip(metas, frames[1:]):
        if int(meta["nbytes"]) != len(buf):
            raise ValueError(
                f"tensor {meta['name']}: nbytes={meta['nbytes']} got={len(buf)}"
            )
        tensors[meta["name"]] = bytes_to_tensor(buf, meta["dtype"], meta["shape"])
    return header, tensors


def encode_speculate_response(draft_tokens: torch.Tensor) -> list[bytes]:
    # Sink reply may still be on GPU; one DtoH after generate is fine.
    host = draft_tokens.detach().contiguous()
    if host.device.type != "cpu":
        host = host.cpu()
    header = {
        "cmd": CMD_SPECULATE,
        "tensors": [_tensor_frame_meta("draft_tokens", host)],
    }
    return [json.dumps(header).encode("utf-8"), tensor_to_bytes(host)]


def decode_speculate_response(frames: list[bytes]) -> torch.Tensor:
    header = json.loads(frames[0].decode("utf-8"))
    if header.get("error"):
        raise RuntimeError(f"SPECulate failed: {header['error']}")
    if header.get("cmd") != CMD_SPECULATE:
        raise ValueError(f"expected speculate reply, got {header.get('cmd')!r}")
    meta = header["tensors"][0]
    return bytes_to_tensor(frames[1], meta["dtype"], meta["shape"])


def encode_hs_ready(
    *, num_ctx_tokens: int, error: str | None = None
) -> bytes:
    """Tiny post-PtoP signal: HS is valid in the sink staging buffer."""
    payload: dict[str, Any] = {
        "cmd": CMD_HS_READY,
        "num_ctx_tokens": int(num_ctx_tokens),
    }
    if error is not None:
        payload["error"] = error
    return json.dumps(payload).encode("utf-8")


def decode_hs_ready(payload: bytes) -> dict[str, Any]:
    meta = json.loads(payload.decode("utf-8"))
    if meta.get("cmd") != CMD_HS_READY:
        raise ValueError(f"expected hs_ready, got {meta.get('cmd')!r}")
    if meta.get("error"):
        raise RuntimeError(f"HS transfer failed: {meta['error']}")
    return meta


def encode_free_request(req_ids: list[str]) -> bytes:
    return json.dumps({"cmd": CMD_FREE, "req_ids": list(req_ids)}).encode("utf-8")


def decode_free_request(payload: bytes) -> list[str]:
    meta = json.loads(payload.decode("utf-8"))
    if meta.get("cmd") != CMD_FREE:
        raise ValueError(f"expected free, got {meta.get('cmd')!r}")
    return list(meta["req_ids"])


def encode_free_ack() -> bytes:
    return json.dumps({"cmd": CMD_FREE, "ok": True}).encode("utf-8")


def encode_profile_request(*, start: bool) -> bytes:
    return json.dumps({"cmd": CMD_PROFILE, "start": bool(start)}).encode("utf-8")


def encode_profile_ack(*, start: bool, ok: bool = True) -> bytes:
    return json.dumps(
        {"cmd": CMD_PROFILE, "start": bool(start), "ok": bool(ok)}
    ).encode("utf-8")
