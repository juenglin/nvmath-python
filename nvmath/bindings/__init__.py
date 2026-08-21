# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore

import importlib

# Load library wrappers on first access (PEP 562). Eager imports here would
# map every wrapper .so (including optional *Mp / NVSHMEM modules) as soon as
# any nvmath.bindings Cython submodule is loaded — for example CuPy's
# `from nvmath.bindings.cycurand cimport ...`.
_REQUIRED = frozenset(
    {
        "cublas",
        "cublasLt",
        "cudss",
        "cufft",
        "curand",
        "cusolver",
        "cusolverDn",
        "cusolverSp",
        "cusparse",
        "cusparseLt",
        "cutensor",
        "nvpl",
    }
)
_OPTIONAL = frozenset({"cufftMp", "nvshmem", "cublasMp", "cusolverMp"})
_SUBMODULES = _REQUIRED | _OPTIONAL


def __getattr__(name):
    if name in _SUBMODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            if name in _OPTIONAL:
                globals()[name] = None
                return None
            raise
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)


__all__ = [
    "cublas",
    "cublasLt",
    "cublasMp",
    "cudss",
    "cufft",
    "cufftMp",
    "curand",
    "cusolver",
    "cusolverDn",
    "cusolverSp",
    "cusolverMp",
    "cusparse",
    "cusparseLt",
    "cutensor",
    "nvpl",
    "nvshmem",
]
