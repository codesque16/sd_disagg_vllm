# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure NVTX range helper — push/pop only, no syncs or logic."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def nvtx_range(name: str, *, generic: str | None = None) -> Iterator[None]:
    """Mark a CUDA NVTX range. Does not synchronize or alter execution.

    When ``generic`` is set, pushes an outer span with that stable name and an
    inner span with ``name`` (often iteration-specific) so nsys can both group
    by type and keep per-iteration detail.
    """
    if generic is not None:
        torch.cuda.nvtx.range_push(generic)
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        if generic is not None:
            torch.cuda.nvtx.range_pop()
