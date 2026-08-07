# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entry: ``python -m vllm.entrypoints.dflash_hs_nixl_sink``."""

from vllm.v1.spec_decode.dflash_hs_nixl.sink import main

if __name__ == "__main__":
    raise SystemExit(main())
