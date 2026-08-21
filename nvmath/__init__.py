# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Math Python libraries."""

import importlib
import importlib.metadata

# High-level subpackages (fft, linalg, sparse, tensor, bindings) and public
# names from nvmath._utils / nvmath.memory are loaded on first access. Eager
# imports here would pull in every library wrapper whenever a nested module
# such as nvmath.bindings.cycurand is imported (the CuPy bindings-build path).
# Native-library preload (module_init_force_cupy_lib_load) is likewise deferred
# until a high-level subpackage is used; it must not run for cycurand-only
# imports.
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


__version__ = importlib.metadata.version("nvmath-python")
