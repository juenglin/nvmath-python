# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath._utils import module_init_force_cupy_lib_load

module_init_force_cupy_lib_load()

from . import advanced, ust  # noqa: E402
from .generic import ComputeType, ExecutionCUDA, Matmul, MatmulOptions, matmul  # noqa: E402
from .generic import (  # noqa: E402
    compile_add as compile_matmul_add,
)
from .generic import (  # noqa: E402
    compile_atomic_add as compile_matmul_atomic_add,
)
from .generic import (  # noqa: E402
    compile_epilog as compile_matmul_epilog,
)
from .generic import (  # noqa: E402
    compile_mul as compile_matmul_mul,
)
from .generic import (  # noqa: E402
    compile_prolog as compile_matmul_prolog,
)
from .generic import matrix_qualifiers_dtype as matmul_matrix_qualifiers_dtype  # noqa: E402

__all__ = [
    "advanced",
    "ust",
    "ExecutionCUDA",
    "MatmulOptions",
    "matmul_matrix_qualifiers_dtype",
    "compile_matmul_prolog",
    "compile_matmul_epilog",
    "compile_matmul_add",
    "compile_matmul_atomic_add",
    "compile_matmul_mul",
    "Matmul",
    "matmul",
    "ComputeType",
]
