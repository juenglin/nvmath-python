# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore

import importlib
import os

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


def _eager_import_enabled() -> bool:
    # Same opt-in convention as scientific-python's lazy_loader: EAGER_IMPORT
    # set to a truthy value forces the lazy imports to run now, so a broken or
    # missing wrapper fails at import time instead of on first attribute access.
    return os.environ.get("EAGER_IMPORT", "").lower() not in ("", "0", "false")


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


if _eager_import_enabled():
    # Force required wrappers to load now (hard failure if one is broken).
    # Optional wrappers stay best-effort: force-loading them would raise on
    # platforms where their library is absent, so they keep None-on-ImportError.
    for _name in sorted(_REQUIRED):
        __getattr__(_name)
    for _name in sorted(_OPTIONAL):
        try:
            __getattr__(_name)
        except ImportError:
            globals()[_name] = None
    del _name


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
