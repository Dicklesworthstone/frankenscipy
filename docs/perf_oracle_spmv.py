#!/usr/bin/env python3
"""SciPy sparse SpMV oracle and persistent live-CG incumbent arm.

The default one-shot mode retains the historical random-CSR SpMV sweep. Passing
``--cg-live`` starts a line-oriented co-process used by
``perf_csr_matvec cg-vs-scipy``. The child constructs the exact same canonical
five-point-Laplacian or wide-band SPD CSR as Rust, then times only
``scipy.sparse.linalg.cg``; matrix construction, pipe I/O, and result
serialization remain outside timing.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import cg


def med(fn, r=15):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def laplacian_2d(side: int, diagonal: float) -> sp.csr_matrix:
    """Canonical row-sorted CSR matching ``laplacian_2d`` in the Rust harness."""
    n = side * side
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in range(side):
        for col in range(side):
            index = row * side + col
            if row > 0:
                indices.append(index - side)
                data.append(-1.0)
            if col > 0:
                indices.append(index - 1)
                data.append(-1.0)
            indices.append(index)
            data.append(diagonal)
            if col + 1 < side:
                indices.append(index + 1)
                data.append(-1.0)
            if row + 1 < side:
                indices.append(index + side)
                data.append(-1.0)
            indptr.append(len(data))
    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n, n),
    )


def wide_band_spd(
    n: int, half_bandwidth: int, diagonal: float, off_diagonal: float
) -> sp.csr_matrix:
    """Canonical row-sorted wide-band CSR matching the Rust completion fixture."""
    offsets = np.arange(-half_bandwidth, half_bandwidth + 1, dtype=np.int64)
    diagonals = [
        np.full(n - abs(int(offset)), diagonal if offset == 0 else off_diagonal)
        for offset in offsets
    ]
    return sp.diags(
        diagonals,
        offsets,
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


def cg_rhs(n: int) -> np.ndarray:
    return 1.0 + 0.01 * (np.arange(n, dtype=np.float64) % 17.0)


def run_cg_live() -> int:
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    cg_engine_path = Path(sys.modules[cg.__module__].__file__).resolve()
    cg_engine_sha256 = hashlib.sha256(cg_engine_path.read_bytes()).hexdigest()
    installed_path = any(
        component in {"site-packages", "dist-packages"}
        for component in scipy_path.parts
    )
    genuine = (
        cg.__module__ == "scipy.sparse.linalg._isolve.iterative"
        and installed_path
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"file={scipy_path} cg_mod={cg.__module__} "
        f"cg_engine_file={cg_engine_path} cg_engine_sha256={cg_engine_sha256} "
        f"python={Path(sys.executable).resolve()} fsci_loaded={fsci_loaded} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy-cg", flush=True)
        return 2

    matrix: sp.csr_matrix | None = None
    rhs: np.ndarray | None = None
    rtol = 1e-5
    maxiter = 1

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == "QUIT":
            break
        if parts[0] == "PREP":
            side = int(parts[1])
            diagonal = float(parts[2])
            rtol = float(parts[3])
            maxiter = int(parts[4])
            matrix = laplacian_2d(side, diagonal)
            rhs = cg_rhs(matrix.shape[0])
            print(
                f"CASE n={matrix.shape[0]} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} dtype={matrix.dtype}",
                flush=True,
            )
        elif parts[0] == "PREP_WIDE":
            n = int(parts[1])
            half_bandwidth = int(parts[2])
            diagonal = float(parts[3])
            off_diagonal = float(parts[4])
            rtol = float(parts[5])
            maxiter = int(parts[6])
            matrix = wide_band_spd(n, half_bandwidth, diagonal, off_diagonal)
            rhs = cg_rhs(matrix.shape[0])
            print(
                f"CASE n={matrix.shape[0]} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} dtype={matrix.dtype}",
                flush=True,
            )
        elif parts[0] == "PARITY":
            if matrix is None or rhs is None:
                print("FATAL case-not-prepared", flush=True)
                return 2
            iterations = 0

            def count_iteration(_x):
                nonlocal iterations
                iterations += 1

            solution, info = cg(
                matrix,
                rhs,
                rtol=rtol,
                atol=0.0,
                maxiter=maxiter,
                callback=count_iteration,
            )
            residual = float(
                np.linalg.norm(rhs - matrix.dot(solution)) / np.linalg.norm(rhs)
            )
            print(
                f"RESULT info={int(info)} iterations={iterations} "
                f"residual={residual!r} components={solution.size}",
                flush=True,
            )
            print(
                "X " + ",".join(repr(float(value)) for value in solution),
                flush=True,
            )
        elif parts[0] == "SOLVE":
            if matrix is None or rhs is None:
                print("FATAL case-not-prepared", flush=True)
                return 2
            reps = int(parts[1])
            start = time.perf_counter()
            for _ in range(reps):
                solution, info = cg(
                    matrix,
                    rhs,
                    rtol=rtol,
                    atol=0.0,
                    maxiter=maxiter,
                )
            elapsed = time.perf_counter() - start
            print(
                f"TIME {elapsed!r} {int(info)} {solution.size}",
                flush=True,
            )
        else:
            print(f"FATAL unknown-command {parts[0]}", flush=True)
            return 2
    return 0


def run_spmv_oracle() -> None:
    rng = np.random.default_rng(0)
    for (n, density) in [(100, 0.05), (1000, 0.01), (10000, 0.001)]:
        A = sp.random(n, n, density=density, format="csr", random_state=0)
        x = rng.standard_normal(n)
        print(f"scipy csr spmv {n}x{n} nnz={A.nnz}: {med(lambda: A.dot(x))*1e6:.2f} us")


if __name__ == "__main__":
    if "--cg-live" in sys.argv[1:]:
        raise SystemExit(run_cg_live())
    run_spmv_oracle()
