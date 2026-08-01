#!/usr/bin/env python3
"""Persistent genuine-SciPy arm for sparse iterative-solver comparisons.

The Rust harness sends the exact CSR arrays and right-hand side once. Matrix
construction, serialization, callback counting, and parity reporting are outside
timing; each ``SOLVE`` command times only repeated public SciPy solver calls.

Protocol::

    <- READY scipy=<ver> method=<gmres|bicgstab> ... genuine=<bool>
    -> INIT <n> <nnz> <rtol> <maxiter>
    -> INDPTR <comma-separated usize values>
    -> INDICES <comma-separated usize values>
    -> DATA <comma-separated f64 values>
    -> B <comma-separated f64 values>
    <- CASE method=<...> n=<...> nnz=<...> sorted=True finite=True ...
    -> PARITY
    <- RESULT info=<...> iterations=<...> residual=<...> components=<...>
    <- X <comma-separated f64 values>
    -> SOLVE <repetitions>
    <- TIME <seconds> <info> <components> <actual-observed-threads> <checksum>
    -> QUIT
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla


METHODS = {
    "gmres": spla.gmres,
    "bicgstab": spla.bicgstab,
    "lsqr": spla.lsqr,
    # qmr is an _isolve solver: same rtol/atol/maxiter keywords, info==0 on
    # success, and callback(xk) once per completed loop body. Left un-
    # preconditioned (M1=M2=None) so SciPy synthesises the two identity
    # LinearOperators whose dispatch cost is the mechanism under test.
    "qmr": spla.qmr,
}

# lsqr is not an _isolve solver: it takes no callback and no x0, its keyword
# names differ, and it returns a 10-tuple whose element 2 is the exact
# iteration count. Everything downstream branches on this set.
NO_CALLBACK_METHODS = frozenset({"lsqr"})

# SciPy's success code is method-dependent. The _isolve solvers return info==0,
# but lsqr returns istop, where 1 means "Ax - b is small enough" and 2 means the
# least-squares solution is good enough. Both count as converged.
LSQR_CONVERGED_ISTOP = frozenset({1, 2})


def observed_threads() -> int:
    return sum(1 for _ in Path("/proc/self/task").iterdir())


def parse_vector(
    line: str,
    label: str,
    expected: int,
    dtype: type[np.float64] | type[np.int64],
) -> np.ndarray:
    prefix = f"{label} "
    if not line.startswith(prefix):
        raise ValueError(f"expected {label}, received {line[:80]!r}")
    payload = line[len(prefix) :].strip()
    values = np.fromstring(payload, sep=",", dtype=dtype)
    if values.size != expected:
        raise ValueError(f"{label} length {values.size} != {expected}")
    return values


def cubic_spsolve_fixture(side: int) -> tuple[sp.csr_matrix, np.ndarray, str]:
    n = side * side * side
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def index(z: int, y: int, x: int) -> int:
        return (z * side + y) * side + x

    for z in range(side):
        for y in range(side):
            for x in range(side):
                row = index(z, y, x)
                rows.append(row)
                cols.append(row)
                data.append(6.001)
                for dz, dy, dx in (
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ):
                    neighbor_z = z + dz
                    neighbor_y = y + dy
                    neighbor_x = x + dx
                    if (
                        0 <= neighbor_z < side
                        and 0 <= neighbor_y < side
                        and 0 <= neighbor_x < side
                    ):
                        rows.append(row)
                        cols.append(index(neighbor_z, neighbor_y, neighbor_x))
                        data.append(-1.0)

    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    ).tocsr()
    matrix.sort_indices()
    rhs = np.asarray(
        [1.0 + 0.5 * (i % 13) for i in range(n)], dtype=np.float64
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(rhs, dtype="<f8").tobytes(order="C"))
    return matrix, rhs, digest.hexdigest()


def profile_cubic_spsolve(repetitions: int, side: int) -> int:
    solver = spla.spsolve
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("CUBIC_SCIPY_FATAL solver-source-unavailable", flush=True)
        return 2
    solver_path = Path(solver_path_text).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    genuine = (
        solver.__module__.startswith("scipy.sparse.linalg._dsolve")
        and installed
        and scipy_path.parent in solver_path.parents
        and not fsci_loaded
    )
    print(
        f"CUBIC_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={solver.__module__} scipy_file={scipy_path} "
        f"scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if not genuine or repetitions < 1 or side < 2:
        print("CUBIC_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    matrix, rhs, input_sha256 = cubic_spsolve_fixture(side)
    solution = solver(matrix, rhs)
    residual = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
    maximum_threads = observed_threads()
    started = time.perf_counter()
    checksum = float(solution.sum())
    for _ in range(repetitions):
        solution = solver(matrix, rhs)
        checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"CUBIC_SCIPY_PROFILE side={side} n={matrix.shape[0]} nnz={matrix.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} residual={residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def main() -> int:
    if len(sys.argv) in {3, 4} and sys.argv[1] == "--profile-cubic-spsolve":
        repetitions = int(sys.argv[2])
        side = int(sys.argv[3]) if len(sys.argv) == 4 else 16
        return profile_cubic_spsolve(repetitions, side)
    if len(sys.argv) != 3 or sys.argv[1] != "--live" or sys.argv[2] not in METHODS:
        print(
            "usage: scipy_sparse_arm.py --live <gmres|bicgstab|lsqr>",
            file=sys.stderr,
        )
        return 64

    method = sys.argv[2]
    solver = METHODS[method]
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("FATAL solver-source-unavailable", flush=True)
        return 2
    solver_path = Path(solver_path_text).resolve()
    solver_sha256 = hashlib.sha256(solver_path.read_bytes()).hexdigest()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    genuine = (
        solver.__module__.startswith("scipy.sparse.linalg._isolve")
        and installed
        and scipy_path.parent in solver_path.parents
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method={method} "
        f"solver_mod={solver.__module__} scipy_file={scipy_path} "
        f"scipy_engine_file={solver_path} scipy_engine_sha256={solver_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    matrix: sp.csr_matrix | None = None
    rhs: np.ndarray | None = None
    rtol = 0.0
    maxiter = 0

    def scipy_converged(status: int) -> bool:
        if method in NO_CALLBACK_METHODS:
            return status in LSQR_CONVERGED_ISTOP
        return status == 0

    def solve(
        callback: Callable[[object], None] | None = None,
    ) -> tuple[np.ndarray, int, int | None]:
        """Returns (solution, scipy_status, exact_iterations_or_None).

        The third element is None when the iteration count must come from a
        callback (the _isolve solvers) and an exact count when SciPy returns it
        directly (lsqr's itn).
        """
        if matrix is None or rhs is None:
            raise RuntimeError("solver fixture is not initialized")
        if method in NO_CALLBACK_METHODS:
            # Mirror FrankenSciPy's stopping rule |phi_bar| / ||b|| < tol
            # exactly: btol carries the relative-residual tolerance, atol=0
            # disables the least-squares test, conlim=0 disables the
            # condition-number test. SciPy's lsqr also uses phibar for rnorm,
            # so test1 is the same quantity we compare.
            result = solver(
                matrix,
                rhs,
                damp=0.0,
                atol=0.0,
                btol=rtol,
                conlim=0.0,
                iter_lim=maxiter,
            )
            # (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var)
            return result[0], int(result[1]), int(result[2])
        kwargs: dict[str, object] = {
            "rtol": rtol,
            "atol": 0.0,
            "maxiter": maxiter,
        }
        if callback is not None:
            kwargs["callback"] = callback
            if method == "gmres":
                # Count every inner Arnoldi iteration without changing the
                # incumbent's default restart length.
                kwargs["callback_type"] = "pr_norm"
        solution, info = solver(matrix, rhs, **kwargs)
        return solution, info, None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT":
            if len(parts) != 5:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            n, nnz, rtol, maxiter = (
                int(parts[1]),
                int(parts[2]),
                float(parts[3]),
                int(parts[4]),
            )
            try:
                indptr = parse_vector(
                    sys.stdin.readline(), "INDPTR", n + 1, np.int64
                )
                indices = parse_vector(
                    sys.stdin.readline(), "INDICES", nnz, np.int64
                )
                data = parse_vector(sys.stdin.readline(), "DATA", nnz, np.float64)
                rhs = parse_vector(sys.stdin.readline(), "B", n, np.float64)
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            matrix = sp.csr_matrix(
                (data, indices, indptr),
                shape=(n, n),
                copy=False,
            )
            finite = bool(np.isfinite(data).all() and np.isfinite(rhs).all())
            nonsymmetric = bool((matrix - matrix.T).nnz)
            # First-call setup is not part of the measurement.
            warm_x, warm_info, _ = solve()
            if not scipy_converged(warm_info) or warm_x.size != n:
                print(f"FATAL warmup info={warm_info}", flush=True)
                return 2
            print(
                f"CASE method={method} n={n} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} finite={finite} "
                f"nonsymmetric={nonsymmetric}",
                flush=True,
            )
            continue
        if parts[0] == "PARITY":
            iterations = 0

            def count(_state: object) -> None:
                nonlocal iterations
                iterations += 1

            solution, info, exact_iterations = solve(count)
            if exact_iterations is not None:
                # SciPy reported the count itself; the callback never fired.
                iterations = exact_iterations
            assert matrix is not None and rhs is not None
            residual = float(
                np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs)
            )
            print(
                f"RESULT info={info} iterations={iterations} residual={residual!r} "
                f"components={solution.size}",
                flush=True,
            )
            print(
                "X " + ",".join(format(float(value), ".17e") for value in solution),
                flush=True,
            )
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2:
                print(f"FATAL bad-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            maximum_threads = observed_threads()
            solution: np.ndarray | None = None
            info = -999
            started = time.perf_counter()
            for _ in range(repetitions):
                solution, info, _ = solve()
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert solution is not None
            checksum = float(solution.sum())
            print(
                f"TIME {elapsed!r} {info} {solution.size} "
                f"{maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
