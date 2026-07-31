"""Live-SciPy MINRES co-process for ``perf_minres_vs_scipy``.

``--minres-live`` starts a line-oriented child that builds the exact same
canonical five-point-Laplacian CSR as the Rust harness (with the diagonal
carrying the indefinite shift), then times only
``scipy.sparse.linalg.minres``. Matrix construction, pipe I/O, and result
serialization stay outside the timed region.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import minres


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


def minres_rhs(n: int) -> np.ndarray:
    return 1.0 + 0.01 * (np.arange(n, dtype=np.float64) % 17.0)


def run_minres_live() -> int:
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    installed_path = any(
        component in {"site-packages", "dist-packages"}
        for component in scipy_path.parts
    )
    genuine = (
        minres.__module__ == "scipy.sparse.linalg._isolve.minres"
        and installed_path
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"file={scipy_path} minres_mod={minres.__module__} "
        f"python={Path(sys.executable).resolve()} fsci_loaded={fsci_loaded} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy-minres", flush=True)
        return 2

    matrix: sp.csr_matrix | None = None
    rhs: np.ndarray | None = None
    rtol = 1e-8
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
            rhs = minres_rhs(matrix.shape[0])
            print(
                f"CASE n={matrix.shape[0]} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices}",
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

            solution, info = minres(
                matrix,
                rhs,
                rtol=rtol,
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
                solution, info = minres(
                    matrix,
                    rhs,
                    rtol=rtol,
                    maxiter=maxiter,
                )
            elapsed = time.perf_counter() - start
            print(f"TIME {elapsed!r} {int(info)} {solution.size}", flush=True)
        else:
            print(f"FATAL unknown-command {parts[0]}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    if "--minres-live" in sys.argv[1:]:
        raise SystemExit(run_minres_live())
    raise SystemExit("perf_oracle_minres.py requires --minres-live")
