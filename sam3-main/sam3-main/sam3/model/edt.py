# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Lightweight stub of the Triton-based EDT module from SAM3.

The original SAM3 repository implements Euclidean Distance Transform (EDT)
using Triton kernels, which are not available on Windows. In this project we
only need SAM3's image model, not EDT, but other modules import
``sam3.model.edt.edt_triton`` at import time.

To keep the package importable on Windows, we provide a minimal stub that
defines ``edt_triton`` and always raises at call time. No existing code in
this campus-surveillance project calls it.
"""

from __future__ import annotations

import torch


def edt_triton(data: torch.Tensor):
    """Stub EDT implementation – always fails at runtime."""

    raise RuntimeError(
        "edt_triton is not supported on this platform because Triton is "
        "unavailable. This stub exists only to make the `sam3` package "
        "importable on Windows."
    )

