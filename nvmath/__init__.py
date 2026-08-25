# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Math Python libraries."""

import importlib
import importlib.metadata
import os

# High-level subpackages (fft, linalg, sparse, tensor, bindings) and public
# names from nvmath._utils / nvmath.memory are loaded on first access. Eager
# imports here would pull in every library wrapper whenever a nested module
# such as nvmath.bindings.cycurand is imported (the CuPy bindings-build path).
# Native-library preload (module_init_force_cupy_lib_load) is likewise deferred
# until a high-level subpackage is used; it must not run for cycurand-only
# imports.
#
# We deliberately hand-roll this instead of using scientific-python's
# lazy_loader (SPEC 1), because lazy_loader cannot express two behaviors we
# must preserve:
#
#   1. Optional bindings wrappers (cufftMp, cublasMp, cusolverMp, nvshmem)
#      resolve to None when their library is unavailable. lazy_loader.attach
#      only ever returns a module or raises ImportError.
#
#   2. A host-API subpackage (fft/linalg/sparse/tensor), on first access, must
#      first run module_init_force_cupy_lib_load() -- which dlopens the NVIDIA
#      libraries (libcublas, libcufft, ...) so CuPy can find them (cupy#9127).
#      This preload must NOT run for a bindings-only import such as CuPy's
#      `from nvmath.bindings.cycurand cimport ...`, which needs a single
#      wrapper and none of the host APIs. lazy_loader.attach offers no hook to
#      run a side effect on access of some names but not others.
#
# Both would require wrapping lazy_loader anyway, so a small module-level
# __getattr__ (PEP 562) is the simpler fit. We still adopt lazy_loader's
# EAGER_IMPORT convention so broken imports can be surfaced.
_LAZY_MODULES = frozenset({"bindings", "fft", "linalg", "sparse", "tensor"})
_LAZY_ATTRS = {
    "BaseCUDAMemoryManager": "nvmath.memory",
    "BaseCUDAMemoryManagerAsync": "nvmath.memory",
    "ComputeType": "nvmath._utils",
    "CudaDataType": "nvmath._utils",
    "LibraryPropertyType": "nvmath._utils",
    "MemoryPointer": "nvmath.memory",
}
_PRELOAD_ON = frozenset({"fft", "linalg", "sparse", "tensor"})
_libs_preloaded = False

__all__ = [
    "BaseCUDAMemoryManager",
    "BaseCUDAMemoryManagerAsync",
    "bindings",
    "ComputeType",
    "CudaDataType",
    "fft",
    "LibraryPropertyType",
    "linalg",
    "MemoryPointer",
    "sparse",
    "tensor",
]


def _ensure_libs_preloaded() -> None:
    global _libs_preloaded
    if _libs_preloaded:
        return
    _libs_preloaded = True
    from nvmath._utils import module_init_force_cupy_lib_load

    module_init_force_cupy_lib_load()


def _eager_import_enabled() -> bool:
    # See lazy_loader's EAGER_IMPORT convention: a truthy value forces the lazy
    # imports to run now, so missing/broken submodules fail at import time.
    return os.environ.get("EAGER_IMPORT", "").lower() not in ("", "0", "false")


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        if name in _PRELOAD_ON:
            _ensure_libs_preloaded()
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    origin = _LAZY_ATTRS.get(name)
    if origin is not None:
        value = getattr(importlib.import_module(origin), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES | _LAZY_ATTRS.keys())


if _eager_import_enabled():
    for _name in sorted(_LAZY_MODULES | _LAZY_ATTRS.keys()):
        __getattr__(_name)
    del _name


__version__ = importlib.metadata.version("nvmath-python")
