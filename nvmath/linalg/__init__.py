# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath._utils import module_init_force_cupy_lib_load
from nvmath.bindings.cublas import ComputeType  # type: ignore

module_init_force_cupy_lib_load()

from . import advanced  # noqa: E402
from .generic import (  # noqa: E402
    DiagonalMatrixQualifier,
    DiagType,
    ExecutionCPU,
    ExecutionCUDA,
    FillMode,
    GeneralMatrixQualifier,
    HermitianMatrixQualifier,
    InvalidMatmulState,
    Matmul,
    MatmulOptions,
    MatrixQualifier,
    SideMode,
    SymmetricMatrixQualifier,
    TriangularMatrixQualifier,
    matmul,
    matrix_qualifiers_dtype,
)
from .generic.solvermod import (  # noqa: E402
    DirectSolver,
    DirectSolverOptions,
    InvalidDirectSolverState,
    direct_solver,
)

__all__ = [
    "advanced",
    "ComputeType",
    "DiagonalMatrixQualifier",
    "ExecutionCPU",
    "ExecutionCUDA",
    "GeneralMatrixQualifier",
    "HermitianMatrixQualifier",
    "InvalidMatmulState",
    "InvalidDirectSolverState",
    "DirectSolver",
    "DirectSolverOptions",
    "matmul",
    "Matmul",
    "MatmulOptions",
    "MatrixQualifier",
    "matrix_qualifiers_dtype",
    "direct_solver",
    "SymmetricMatrixQualifier",
    "TriangularMatrixQualifier",
    "SideMode",
    "FillMode",
    "DiagType",
]
