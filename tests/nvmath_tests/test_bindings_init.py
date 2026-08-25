# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy loading of nvmath and nvmath.bindings submodules."""

import os
import subprocess
import sys
import textwrap

import nvmath
import nvmath.bindings as bindings

# Required bindings wrappers that EAGER_IMPORT must force-load at import time.
REQUIRED_WRAPPERS = [
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
]

EXPECTED_BINDINGS_ALL = [
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

EXPECTED_NVMATH_ALL = [
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


def _run(code: str, env: dict | None = None) -> str:
    # A fresh subprocess is required: the eager/lazy behavior is decided at
    # import time of nvmath[.bindings], and those modules are already cached in
    # the pytest session's sys.modules. It also keeps EAGER_IMPORT's whole-tree
    # load from polluting sys.modules for the rest of the session.
    run_env = None
    if env is not None:
        run_env = os.environ.copy()
        run_env.update(env)
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
        env=run_env,
    )
    return result.stdout.strip()


def test_bindings_all_unchanged():
    assert bindings.__all__ == EXPECTED_BINDINGS_ALL


def test_nvmath_all_unchanged():
    assert nvmath.__all__ == EXPECTED_NVMATH_ALL


def test_dir_lists_public_wrappers():
    names = _run(
        """
        import nvmath.bindings as bindings
        print(','.join(n for n in bindings.__all__ if n in dir(bindings)))
        """
    )
    listed = names.split(",") if names else []
    assert listed == EXPECTED_BINDINGS_ALL


def test_dir_lists_nvmath_public_names():
    names = _run(
        """
        import nvmath
        print(','.join(n for n in nvmath.__all__ if n in dir(nvmath)))
        """
    )
    listed = names.split(",") if names else []
    assert listed == EXPECTED_NVMATH_ALL


def test_cycurand_does_not_import_host_or_optional_wrappers():
    loaded = _run(
        """
        import sys
        import nvmath.bindings.cycurand  # noqa: F401
        forbidden = [
            "nvmath.fft",
            "nvmath.linalg",
            "nvmath.sparse",
            "nvmath.tensor",
            "nvmath.bindings.cublas",
            "nvmath.bindings.cufft",
            "nvmath.bindings.cublasMp",
            "nvmath.bindings.cufftMp",
            "nvmath.bindings.cusolverMp",
            "nvmath.bindings.nvshmem",
        ]
        loaded = [n for n in forbidden if n in sys.modules]
        print(','.join(loaded))
        """
    )
    assert loaded == ""


def test_attribute_access_returns_wrapper_or_none():
    status = _run(
        """
        import nvmath.bindings as bindings
        import types

        curand = bindings.curand
        cublas = bindings.cublas
        assert isinstance(curand, types.ModuleType)
        assert isinstance(cublas, types.ModuleType)
        assert curand is bindings.curand
        assert cublas is bindings.cublas

        for name in ("cublasMp", "cufftMp", "cusolverMp", "nvshmem"):
            value = getattr(bindings, name)
            assert value is None or isinstance(value, types.ModuleType), name
        print("ok")
        """
    )
    assert status == "ok"


def test_nvmath_lazy_public_attributes():
    status = _run(
        """
        import sys
        import types
        import nvmath

        assert "nvmath.fft" not in sys.modules
        assert "nvmath.linalg" not in sys.modules
        assert isinstance(nvmath.ComputeType, type)
        assert isinstance(nvmath.fft, types.ModuleType)
        assert "nvmath.fft" in sys.modules
        assert nvmath.fft is nvmath.fft
        print("ok")
        """
    )
    assert status == "ok"


def test_star_import_and_unknown_attribute():
    status = _run(
        """
        from nvmath.bindings import *  # noqa: F403
        import nvmath.bindings as bindings

        assert bindings.cublas is not None
        try:
            bindings.not_a_real_binding
        except AttributeError:
            print("ok")
        else:
            raise SystemExit("expected AttributeError")
        """
    )
    assert status == "ok"


def test_eager_import_loads_required_wrappers_at_import():
    # With EAGER_IMPORT set, merely importing nvmath.bindings must pull in every
    # required wrapper (and the host-API subpackages) before any attribute
    # access, so a broken/missing import fails loudly at import time.
    required = ",".join(REQUIRED_WRAPPERS)
    missing = _run(
        f"""
        import sys
        import nvmath.bindings  # noqa: F401

        required = "{required}".split(",")
        missing = [n for n in required if f"nvmath.bindings.{{n}}" not in sys.modules]
        # Host-API subpackages should also be eagerly loaded via nvmath.__init__.
        for sub in ("nvmath.fft", "nvmath.linalg", "nvmath.sparse", "nvmath.tensor"):
            if sub not in sys.modules:
                missing.append(sub)
        print(",".join(missing))
        """,
        env={"EAGER_IMPORT": "1"},
    )
    assert missing == ""


def test_default_import_stays_lazy():
    # The complement of the eager test: with EAGER_IMPORT unset, importing
    # nvmath.bindings must not eagerly load the wrappers.
    loaded = _run(
        """
        import sys
        import nvmath.bindings  # noqa: F401

        watch = ("nvmath.bindings.cublas", "nvmath.bindings.cufft", "nvmath.fft")
        print(",".join(n for n in watch if n in sys.modules))
        """,
        env={"EAGER_IMPORT": ""},
    )
    assert loaded == ""
