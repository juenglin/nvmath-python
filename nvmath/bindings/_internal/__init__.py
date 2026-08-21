# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore

import importlib


def __getattr__(name):
    # Submodules were previously imported as a side effect of eagerly loading
    # every nvmath.bindings wrapper. Keep getattr() working now that wrappers
    # are loaded on demand (see nvmath._utils.module_init_force_cupy_lib_load).
    try:
        mod = importlib.import_module(f"{__name__}.{name}")
    except ImportError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = mod
    return mod
