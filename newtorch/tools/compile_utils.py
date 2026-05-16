"""torch.compile helpers that skip CUDA-only backends on CPU."""

from __future__ import annotations

from typing import Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable)


def compile_cudagraphs_if_cuda(fn: F) -> F:
    """Apply ``torch.compile(backend='cudagraphs')`` only when CUDA is available."""
    if torch.cuda.is_available():
        return torch.compile(backend="cudagraphs")(fn)  # type: ignore[return-value]
    return fn


def compile_if_cuda(fn: F, *, backend: str = "inductor") -> F:
    """Apply ``torch.compile`` with a generic backend when CUDA is available."""
    if torch.cuda.is_available():
        return torch.compile(backend=backend)(fn)  # type: ignore[return-value]
    return fn
