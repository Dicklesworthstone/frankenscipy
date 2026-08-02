#!/usr/bin/env python3
"""Persistent genuine-SciPy arm for sparse iterative-solver comparisons.

The Rust harness sends the exact CSR arrays and right-hand side once. Matrix
construction, serialization, callback counting, and parity reporting are outside
timing; each ``SOLVE`` command times only repeated public SciPy solver calls.

Protocol::

    <- READY scipy=<ver> method=<cg|gmres|lgmres|bicg|cgs|bicgstab|lsqr|lsmr|qmr|spsolve> ... genuine=<bool>
    -> INIT <n> <nnz> <rtol> <maxiter>
    -> INDPTR <comma-separated usize values>
    -> INDICES <comma-separated usize values>
    -> DATA <comma-separated f64 values>
    -> B <comma-separated f64 values>
    <- CASE method=<...> n=<...> nnz=<...> sorted=True finite=True ...
    -> INPUT_SHA256
    <- INPUT_SHA256 <canonical CSR/RHS digest>
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
import struct
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla


METHODS = {
    "cg": spla.cg,
    "gmres": spla.gmres,
    "lgmres": spla.lgmres,
    "bicg": spla.bicg,
    "cgs": spla.cgs,
    "bicgstab": spla.bicgstab,
    "lsqr": spla.lsqr,
    "lsmr": spla.lsmr,
    # qmr is an _isolve solver: same rtol/atol/maxiter keywords, info==0 on
    # success, and callback(xk) once per completed loop body. Left un-
    # preconditioned (M1=M2=None) so SciPy synthesises the two identity
    # LinearOperators whose dispatch cost is the mechanism under test.
    "qmr": spla.qmr,
    "spsolve": spla.spsolve,
}

# The least-squares solvers take no callback or x0 and return the exact
# iteration count in tuple element 2. Their iteration-limit keyword differs.
NO_CALLBACK_METHODS = frozenset({"lsqr", "lsmr"})

# Direct sparse solve materializes only the solution. It has no convergence
# callback, tolerance, or iteration-limit parameters.
DIRECT_METHODS = frozenset({"spsolve"})

# SciPy's success code is method-dependent. The _isolve solvers return info==0,
# but lsqr returns istop, where 1 means "Ax - b is small enough" and 2 means the
# least-squares solution is good enough. Both count as converged.
LEAST_SQUARES_CONVERGED_ISTOP = frozenset({1, 2})


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


def cuboid_spsolve_fixture(
    x_extent: int, y_extent: int, z_extent: int
) -> tuple[sp.csr_matrix, np.ndarray, str]:
    n = x_extent * y_extent * z_extent
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def index(z: int, y: int, x: int) -> int:
        return (z * y_extent + y) * x_extent + x

    for z in range(z_extent):
        for y in range(y_extent):
            for x in range(x_extent):
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
                        0 <= neighbor_z < z_extent
                        and 0 <= neighbor_y < y_extent
                        and 0 <= neighbor_x < x_extent
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


def profile_cuboid_spsolve(
    repetitions: int, x_extent: int, y_extent: int, z_extent: int
) -> int:
    solver = spla.spsolve
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("CUBOID_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"CUBOID_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={solver.__module__} scipy_file={scipy_path} "
        f"scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if (
        not genuine
        or repetitions < 1
        or min(x_extent, y_extent, z_extent) < 2
    ):
        print("CUBOID_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    matrix, rhs, input_sha256 = cuboid_spsolve_fixture(
        x_extent, y_extent, z_extent
    )
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
        f"CUBOID_SCIPY_PROFILE x={x_extent} y={y_extent} z={z_extent} "
        f"n={matrix.shape[0]} nnz={matrix.nnz} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum:.17e} "
        f"residual={residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def cubic_splu_fixture(
    side: int, rhs_count: int
) -> tuple[sp.csc_matrix, np.ndarray, str]:
    matrix_csr, _, _ = cubic_spsolve_fixture(side)
    matrix = matrix_csr.tocsc()
    matrix.sort_indices()
    n = side * side * side
    right_hand_sides = np.asarray(
        [
            [1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) for index in range(n)]
            for rhs_index in range(rhs_count)
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C"))
    return matrix, right_hand_sides, digest.hexdigest()


def convection_splu_fixture(
    side: int, rhs_count: int
) -> tuple[sp.csc_matrix, np.ndarray, str]:
    diagonal = 4.001
    west = -1.2
    east = -0.8
    vertical = -1.0
    n = side * side
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for row in range(side):
        for column in range(side):
            index = row * side + column
            if row > 0:
                rows.append(index)
                cols.append(index - side)
                data.append(vertical)
            if column > 0:
                rows.append(index)
                cols.append(index - 1)
                data.append(west)
            rows.append(index)
            cols.append(index)
            data.append(diagonal)
            if column + 1 < side:
                rows.append(index)
                cols.append(index + 1)
                data.append(east)
            if row + 1 < side:
                rows.append(index)
                cols.append(index + side)
                data.append(vertical)

    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    ).tocsc()
    matrix.sort_indices()
    right_hand_sides = np.asarray(
        [
            [1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) for index in range(n)]
            for rhs_index in range(rhs_count)
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C"))
    return matrix, right_hand_sides, digest.hexdigest()


def neumann_cubic_splu_fixture(
    side: int, rhs_count: int, shift: float = 1.0e-3
) -> tuple[sp.csc_matrix, np.ndarray, str]:
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
                degree = sum(
                    (
                        z > 0,
                        z + 1 < side,
                        y > 0,
                        y + 1 < side,
                        x > 0,
                        x + 1 < side,
                    )
                )
                rows.append(row)
                cols.append(row)
                data.append(shift + float(degree))
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
    ).tocsc()
    matrix.sort_indices()
    right_hand_sides = np.asarray(
        [
            [1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) for index in range(n)]
            for rhs_index in range(rhs_count)
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C"))
    return matrix, right_hand_sides, digest.hexdigest()


def neumann_cuboid_splu_fixture(
    x_extent: int,
    y_extent: int,
    z_extent: int,
    rhs_count: int,
    shift: float = 1.0e-3,
    x_weight: float = -0.75,
    y_weight: float = -1.0,
    z_weight: float = -1.25,
) -> tuple[sp.csc_matrix, np.ndarray, str]:
    n = x_extent * y_extent * z_extent
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def index(z: int, y: int, x: int) -> int:
        return (z * y_extent + y) * x_extent + x

    for z in range(z_extent):
        for y in range(y_extent):
            for x in range(x_extent):
                row = index(z, y, x)
                diagonal = (
                    shift
                    - x_weight * int(x > 0)
                    - x_weight * int(x + 1 < x_extent)
                    - y_weight * int(y > 0)
                    - y_weight * int(y + 1 < y_extent)
                    - z_weight * int(z > 0)
                    - z_weight * int(z + 1 < z_extent)
                )
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
                for neighbor_z, neighbor_y, neighbor_x, weight in (
                    (z - 1, y, x, z_weight),
                    (z + 1, y, x, z_weight),
                    (z, y - 1, x, y_weight),
                    (z, y + 1, x, y_weight),
                    (z, y, x - 1, x_weight),
                    (z, y, x + 1, x_weight),
                ):
                    if (
                        0 <= neighbor_z < z_extent
                        and 0 <= neighbor_y < y_extent
                        and 0 <= neighbor_x < x_extent
                    ):
                        rows.append(row)
                        cols.append(index(neighbor_z, neighbor_y, neighbor_x))
                        data.append(weight)

    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    ).tocsc()
    matrix.sort_indices()
    right_hand_sides = np.asarray(
        [
            [1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) for index in range(n)]
            for rhs_index in range(rhs_count)
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C"))
    return matrix, right_hand_sides, digest.hexdigest()


def periodic_cuboid_splu_fixture(
    x_extent: int,
    y_extent: int,
    z_extent: int,
    rhs_count: int,
    shift: float = 1.0e-3,
    x_weight: float = -0.75,
    y_weight: float = -1.0,
    z_weight: float = -1.25,
) -> tuple[sp.csc_matrix, np.ndarray, str]:
    n = x_extent * y_extent * z_extent
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def index(z: int, y: int, x: int) -> int:
        return (z * y_extent + y) * x_extent + x

    diagonal = shift - 2.0 * (x_weight + y_weight + z_weight)
    for z in range(z_extent):
        for y in range(y_extent):
            for x in range(x_extent):
                row = index(z, y, x)
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
                for neighbor_z, neighbor_y, neighbor_x, weight in (
                    ((z - 1) % z_extent, y, x, z_weight),
                    ((z + 1) % z_extent, y, x, z_weight),
                    (z, (y - 1) % y_extent, x, y_weight),
                    (z, (y + 1) % y_extent, x, y_weight),
                    (z, y, (x - 1) % x_extent, x_weight),
                    (z, y, (x + 1) % x_extent, x_weight),
                ):
                    rows.append(row)
                    cols.append(index(neighbor_z, neighbor_y, neighbor_x))
                    data.append(weight)

    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    ).tocsc()
    matrix.sort_indices()
    right_hand_sides = np.asarray(
        [
            [1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) for index in range(n)]
            for rhs_index in range(rhs_count)
        ],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C"))
    return matrix, right_hand_sides, digest.hexdigest()


def periodic_cuboid_spsolve_fixture(
    x_extent: int,
    y_extent: int,
    z_extent: int,
) -> tuple[sp.csc_matrix, np.ndarray, str]:
    matrix, right_hand_sides, _ = periodic_cuboid_splu_fixture(
        x_extent,
        y_extent,
        z_extent,
        2,
    )
    rhs = np.asarray(right_hand_sides[1], dtype=np.float64)
    n = matrix.shape[0]
    digest = hashlib.sha256()
    digest.update(n.to_bytes(8, "little"))
    digest.update(int(matrix.nnz).to_bytes(8, "little"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(rhs, dtype="<f8").tobytes(order="C"))
    return matrix, rhs, digest.hexdigest()


def profile_cubic_splu(repetitions: int, side: int, rhs_count: int) -> int:
    solver = spla.splu
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("CUBIC_SPLU_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"CUBIC_SPLU_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={solver.__module__} scipy_file={scipy_path} "
        f"scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if not genuine or repetitions < 1 or side < 2 or rhs_count < 1:
        print("CUBIC_SPLU_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    matrix, right_hand_sides, input_sha256 = cubic_splu_fixture(side, rhs_count)
    factor = solver(matrix)
    warm_solutions = [factor.solve(rhs) for rhs in right_hand_sides]
    maximum_residual = max(
        float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        for rhs, solution in zip(right_hand_sides, warm_solutions, strict=True)
    )
    maximum_threads = observed_threads()
    checksum = sum(float(solution[solution.size // 2]) for solution in warm_solutions)
    started = time.perf_counter()
    for _ in range(repetitions):
        factor = solver(matrix)
        for rhs in right_hand_sides:
            solution = factor.solve(rhs)
            checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"CUBIC_SPLU_SCIPY_PROFILE side={side} n={matrix.shape[0]} nnz={matrix.nnz} "
        f"rhs_count={rhs_count} repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} max_residual={maximum_residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def profile_convection_splu(
    repetitions: int,
    side: int,
    rhs_count: int,
    output_path: Path | None,
) -> int:
    solver = spla.splu
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("CONVECTION_SPLU_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"CONVECTION_SPLU_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={solver.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if not genuine or repetitions < 1 or side < 2 or rhs_count < 1:
        print("CONVECTION_SPLU_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    matrix, right_hand_sides, input_sha256 = convection_splu_fixture(
        side, rhs_count
    )
    factor = solver(matrix)
    warm_solutions = [factor.solve(rhs) for rhs in right_hand_sides]
    maximum_residual = max(
        float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        for rhs, solution in zip(right_hand_sides, warm_solutions, strict=True)
    )
    maximum_threads = observed_threads()
    checksum = sum(float(solution[solution.size // 2]) for solution in warm_solutions)
    started = time.perf_counter()
    for _ in range(repetitions):
        factor = solver(matrix)
        for rhs in right_hand_sides:
            solution = factor.solve(rhs)
            checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())

    output_bytes = np.asarray(warm_solutions, dtype="<f8").tobytes(order="C")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if output_path is not None:
        with output_path.open("xb") as output:
            output.write(output_bytes)

    print(
        f"CONVECTION_SPLU_SCIPY_PROFILE side={side} diagonal=4.001 "
        f"west=-1.2 east=-0.8 vertical=-1.0 n={matrix.shape[0]} "
        f"nnz={matrix.nnz} rhs_count={rhs_count} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum:.17e} "
        f"max_residual={maximum_residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256} output_sha256={output_sha256}",
        flush=True,
    )
    return 0


def profile_neumann_cubic_splu(
    repetitions: int, side: int, rhs_count: int
) -> int:
    solver = spla.splu
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("NEUMANN_CUBIC_SPLU_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"NEUMANN_CUBIC_SPLU_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={solver.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if not genuine or repetitions < 1 or side < 2 or rhs_count < 1:
        print("NEUMANN_CUBIC_SPLU_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    shift = 1.0e-3
    matrix, right_hand_sides, input_sha256 = neumann_cubic_splu_fixture(
        side, rhs_count, shift
    )
    factor = solver(matrix)
    warm_solutions = [factor.solve(rhs) for rhs in right_hand_sides]
    maximum_residual = max(
        float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        for rhs, solution in zip(right_hand_sides, warm_solutions, strict=True)
    )
    maximum_threads = observed_threads()
    checksum = sum(float(solution[solution.size // 2]) for solution in warm_solutions)
    started = time.perf_counter()
    for _ in range(repetitions):
        factor = solver(matrix)
        for rhs in right_hand_sides:
            solution = factor.solve(rhs)
            checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"NEUMANN_CUBIC_SPLU_SCIPY_PROFILE side={side} shift={shift:.17e} "
        f"n={matrix.shape[0]} nnz={matrix.nnz} rhs_count={rhs_count} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} max_residual={maximum_residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def profile_neumann_cuboid_splu(
    repetitions: int,
    x_extent: int,
    y_extent: int,
    z_extent: int,
    rhs_count: int,
    output_path: Path | None,
) -> int:
    solver = spla.splu
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("NEUMANN_CUBOID_SPLU_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"NEUMANN_CUBOID_SPLU_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={solver.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if (
        not genuine
        or repetitions < 1
        or min(x_extent, y_extent, z_extent) < 2
        or rhs_count < 1
    ):
        print(
            "NEUMANN_CUBOID_SPLU_SCIPY_FATAL invalid-identity-or-controls",
            flush=True,
        )
        return 2

    shift = 1.0e-3
    x_weight = -0.75
    y_weight = -1.0
    z_weight = -1.25
    matrix, right_hand_sides, input_sha256 = neumann_cuboid_splu_fixture(
        x_extent,
        y_extent,
        z_extent,
        rhs_count,
        shift,
        x_weight,
        y_weight,
        z_weight,
    )
    factor = solver(matrix)
    warm_solutions = [factor.solve(rhs) for rhs in right_hand_sides]
    maximum_residual = max(
        float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        for rhs, solution in zip(right_hand_sides, warm_solutions, strict=True)
    )
    maximum_threads = observed_threads()
    checksum = sum(float(solution[solution.size // 2]) for solution in warm_solutions)
    started = time.perf_counter()
    for _ in range(repetitions):
        factor = solver(matrix)
        for rhs in right_hand_sides:
            solution = factor.solve(rhs)
            checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())

    output_bytes = np.asarray(warm_solutions, dtype="<f8").tobytes(order="C")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if output_path is not None:
        with output_path.open("xb") as output:
            output.write(output_bytes)

    print(
        f"NEUMANN_CUBOID_SPLU_SCIPY_PROFILE x={x_extent} y={y_extent} "
        f"z={z_extent} x_weight={x_weight:.17e} y_weight={y_weight:.17e} "
        f"z_weight={z_weight:.17e} shift={shift:.17e} n={matrix.shape[0]} "
        f"nnz={matrix.nnz} rhs_count={rhs_count} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum:.17e} "
        f"max_residual={maximum_residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256} output_sha256={output_sha256}",
        flush=True,
    )
    return 0


def profile_periodic_cuboid_splu(
    repetitions: int,
    x_extent: int,
    y_extent: int,
    z_extent: int,
    rhs_count: int,
    output_path: Path | None,
) -> int:
    solver = spla.splu
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("PERIODIC_CUBOID_SPLU_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"PERIODIC_CUBOID_SPLU_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={solver.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if (
        not genuine
        or repetitions < 1
        or min(x_extent, y_extent, z_extent) < 3
        or rhs_count < 1
    ):
        print(
            "PERIODIC_CUBOID_SPLU_SCIPY_FATAL invalid-identity-or-controls",
            flush=True,
        )
        return 2

    shift = 1.0e-3
    x_weight = -0.75
    y_weight = -1.0
    z_weight = -1.25
    matrix, right_hand_sides, input_sha256 = periodic_cuboid_splu_fixture(
        x_extent,
        y_extent,
        z_extent,
        rhs_count,
        shift,
        x_weight,
        y_weight,
        z_weight,
    )
    factor = solver(matrix)
    warm_solutions = [factor.solve(rhs) for rhs in right_hand_sides]
    maximum_residual = max(
        float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        for rhs, solution in zip(right_hand_sides, warm_solutions, strict=True)
    )
    maximum_threads = observed_threads()
    checksum = sum(float(solution[solution.size // 2]) for solution in warm_solutions)
    started = time.perf_counter()
    for _ in range(repetitions):
        factor = solver(matrix)
        for rhs in right_hand_sides:
            solution = factor.solve(rhs)
            checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())

    output_bytes = np.asarray(warm_solutions, dtype="<f8").tobytes(order="C")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if output_path is not None:
        with output_path.open("xb") as output:
            output.write(output_bytes)

    print(
        f"PERIODIC_CUBOID_SPLU_SCIPY_PROFILE x={x_extent} y={y_extent} "
        f"z={z_extent} x_weight={x_weight:.17e} y_weight={y_weight:.17e} "
        f"z_weight={z_weight:.17e} shift={shift:.17e} n={matrix.shape[0]} "
        f"nnz={matrix.nnz} rhs_count={rhs_count} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum:.17e} "
        f"max_residual={maximum_residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256} output_sha256={output_sha256}",
        flush=True,
    )
    return 0


def profile_periodic_cuboid_spsolve(
    repetitions: int,
    x_extent: int,
    y_extent: int,
    z_extent: int,
    output_path: Path | None,
) -> int:
    solver = spla.spsolve
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("PERIODIC_CUBOID_SPSOLVE_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"PERIODIC_CUBOID_SPSOLVE_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={solver.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if (
        not genuine
        or repetitions < 1
        or min(x_extent, y_extent, z_extent) < 3
    ):
        print(
            "PERIODIC_CUBOID_SPSOLVE_SCIPY_FATAL invalid-identity-or-controls",
            flush=True,
        )
        return 2

    matrix, rhs, input_sha256 = periodic_cuboid_spsolve_fixture(
        x_extent,
        y_extent,
        z_extent,
    )
    warm_solution = np.asarray(solver(matrix, rhs), dtype=np.float64)
    residual = float(
        np.linalg.norm(rhs - matrix @ warm_solution) / np.linalg.norm(rhs)
    )
    maximum_threads = observed_threads()
    checksum = float(warm_solution[warm_solution.size // 2])
    started = time.perf_counter()
    for _ in range(repetitions):
        solution = solver(matrix, rhs)
        checksum += float(solution[solution.size // 2])
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())

    output_bytes = np.asarray(warm_solution, dtype="<f8").tobytes(order="C")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if output_path is not None:
        with output_path.open("xb") as output:
            output.write(output_bytes)

    print(
        f"PERIODIC_CUBOID_SPSOLVE_SCIPY_PROFILE x={x_extent} y={y_extent} "
        f"z={z_extent} x_weight={-0.75:.17e} y_weight={-1.0:.17e} "
        f"z_weight={-1.25:.17e} shift={1.0e-3:.17e} n={matrix.shape[0]} "
        f"nnz={matrix.nnz} repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} max_residual={residual:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256} output_sha256={output_sha256}",
        flush=True,
    )
    return 0


def triangular_wavefront_fixture(
    levels: int, width: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, str]:
    if levels <= 1 or width <= 1:
        raise ValueError("wavefront dimensions must exceed one")
    coupling = sp.diags(
        (
            np.full(width - 1, -0.125, dtype=np.float64),
            np.full(width, -0.5, dtype=np.float64),
            np.full(width - 1, -0.125, dtype=np.float64),
        ),
        offsets=(-1, 0, 1),
        shape=(width, width),
        format="csr",
    )
    level_shift = sp.diags(
        np.ones(levels - 1, dtype=np.float64),
        offsets=-1,
        shape=(levels, levels),
        format="csr",
    )
    n = levels * width
    matrix = (
        sp.kron(level_shift, coupling, format="csr")
        + sp.eye(n, dtype=np.float64, format="csr") * 2.0
    ).tocsr()
    matrix.sort_indices()
    expected = 1.0 + 0.03125 * (
        (17 * np.arange(n, dtype=np.int64)) % 29
    ).astype(np.float64)
    rhs = np.asarray(matrix @ expected, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", n))
    digest.update(struct.pack("<Q", int(matrix.nnz)))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(rhs, dtype="<f8").tobytes(order="C"))
    return matrix, expected, rhs, digest.hexdigest()


def profile_triangular_wavefront(
    repetitions: int, levels: int, width: int
) -> int:
    solver = spla.spsolve_triangular
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        print("TRIANGULAR_SCIPY_FATAL solver-source-unavailable", flush=True)
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
        f"TRIANGULAR_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={solver.__module__} scipy_file={scipy_path} "
        f"scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={hashlib.sha256(solver_path.read_bytes()).hexdigest()} "
        f"actual_observed_worker_threads={observed_threads()} genuine={genuine}",
        flush=True,
    )
    if not genuine or repetitions < 1 or levels <= 1 or width <= 1:
        print("TRIANGULAR_SCIPY_FATAL invalid-identity-or-controls", flush=True)
        return 2

    matrix, expected, rhs, input_sha256 = triangular_wavefront_fixture(levels, width)
    solution = solver(matrix, rhs, lower=True, unit_diagonal=False)
    residual = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
    max_abs_error = float(np.max(np.abs(solution - expected)))
    relative_l2 = float(
        np.linalg.norm(solution - expected) / np.linalg.norm(expected)
    )
    maximum_threads = observed_threads()
    checksum = int(np.bitwise_xor.reduce(solution.view(np.uint64)))
    started = time.perf_counter()
    for _ in range(repetitions):
        solution = solver(matrix, rhs, lower=True, unit_diagonal=False)
        checksum ^= int(np.bitwise_xor.reduce(solution.view(np.uint64)))
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"TRIANGULAR_SCIPY_PROFILE levels={levels} width={width} "
        f"n={matrix.shape[0]} nnz={matrix.nnz} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum} "
        f"max_abs_error={max_abs_error:.17e} residual={residual:.17e} "
        f"relative_l2={relative_l2:.17e} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def live_cubic_splu(method: str = "splu") -> int:
    """Serve CSC factor/solve jobs for the ``splu`` and one-shot gates."""
    if method not in {"splu", "spsolve_many"}:
        print(f"FATAL unsupported-csc-method {method}", flush=True)
        return 2
    solver = spla.splu if method == "splu" else spla.spsolve
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
        solver.__module__.startswith("scipy.sparse.linalg._dsolve")
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

    matrix: sp.csc_matrix | None = None
    right_hand_sides: np.ndarray | None = None
    input_sha256: str | None = None

    def solve_all(factor: object | None) -> np.ndarray:
        if right_hand_sides is None:
            raise RuntimeError("CSC fixture is not initialized")
        if method == "spsolve_many":
            return np.asarray(
                [solver(matrix, rhs) for rhs in right_hand_sides],
                dtype=np.float64,
            )
        if factor is None:
            raise RuntimeError("splu factor is unavailable")
        return np.asarray(
            [factor.solve(rhs) for rhs in right_hand_sides],
            dtype=np.float64,
        )

    def factor_payload_bytes(factor: object | None) -> int:
        if factor is None:
            return 0
        return sum(
            int(array.nbytes)
            for triangular in (factor.L, factor.U)
            for array in (triangular.data, triangular.indices, triangular.indptr)
        )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_SPLU":
            if len(parts) != 4:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            n, nnz, rhs_count = int(parts[1]), int(parts[2]), int(parts[3])
            if n < 1 or nnz < 1 or rhs_count < 1:
                print("FATAL invalid-splu-dimensions", flush=True)
                return 2
            try:
                indptr = parse_vector(
                    sys.stdin.readline(), "INDPTR", n + 1, np.int64
                )
                indices = parse_vector(
                    sys.stdin.readline(), "INDICES", nnz, np.int64
                )
                data = parse_vector(sys.stdin.readline(), "DATA", nnz, np.float64)
                flattened_rhs = parse_vector(
                    sys.stdin.readline(), "RHS", n * rhs_count, np.float64
                )
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            right_hand_sides = flattened_rhs.reshape((rhs_count, n))
            matrix = sp.csc_matrix(
                (data, indices, indptr),
                shape=(n, n),
                copy=False,
            )
            input_hashers = [hashlib.sha256()]
            if method == "spsolve_many":
                input_hashers = [hashlib.sha256() for _ in right_hand_sides]
            for input_hasher in input_hashers:
                input_hasher.update(struct.pack("<Q", n))
                input_hasher.update(struct.pack("<Q", nnz))
                input_hasher.update(np.asarray(data, dtype="<f8").tobytes(order="C"))
                input_hasher.update(
                    np.asarray(indices, dtype="<u8").tobytes(order="C")
                )
                input_hasher.update(
                    np.asarray(indptr, dtype="<u8").tobytes(order="C")
                )
            if method == "spsolve_many":
                for input_hasher, rhs in zip(
                    input_hashers, right_hand_sides, strict=True
                ):
                    input_hasher.update(
                        np.asarray(rhs, dtype="<f8").tobytes(order="C")
                    )
            else:
                input_hashers[0].update(
                    np.asarray(right_hand_sides, dtype="<f8").tobytes(order="C")
                )
            input_sha256 = ",".join(
                input_hasher.hexdigest() for input_hasher in input_hashers
            )
            finite = bool(
                np.isfinite(data).all() and np.isfinite(right_hand_sides).all()
            )
            nonsymmetric = bool((matrix - matrix.T).nnz)
            warm_factor = solver(matrix) if method == "splu" else None
            warm_solutions = solve_all(warm_factor)
            if warm_solutions.shape != (rhs_count, n):
                print("FATAL warmup-shape", flush=True)
                return 2
            print(
                f"CASE method={method} n={n} nnz={matrix.nnz} rhs_count={rhs_count} "
                f"sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} finite={finite} "
                f"nonsymmetric={nonsymmetric}",
                flush=True,
            )
            continue
        if parts[0] == "INPUT_SHA256":
            if input_sha256 is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            print(f"INPUT_SHA256 {input_sha256}", flush=True)
            continue
        if parts[0] == "PARITY":
            if matrix is None or right_hand_sides is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            factor = solver(matrix) if method == "splu" else None
            solutions = solve_all(factor)
            maximum_residual = max(
                float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
                for rhs, solution in zip(
                    right_hand_sides, solutions, strict=True
                )
            )
            flattened = solutions.ravel(order="C")
            print(
                f"RESULT info=0 iterations=0 residual={maximum_residual!r} "
                f"components={flattened.size} "
                f"payload_bytes={factor_payload_bytes(factor)}",
                flush=True,
            )
            print(
                "X " + ",".join(format(float(value), ".17e") for value in flattened),
                flush=True,
            )
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2:
                print(f"FATAL bad-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1 or matrix is None or right_hand_sides is None:
                print("FATAL invalid-solve", flush=True)
                return 2
            maximum_threads = observed_threads()
            solutions: np.ndarray | None = None
            started = time.perf_counter()
            for _ in range(repetitions):
                factor = solver(matrix) if method == "splu" else None
                solutions = solve_all(factor)
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert solutions is not None
            flattened = solutions.ravel(order="C")
            checksum = 0
            for bits in np.asarray(flattened, dtype="<f8").view("<u8"):
                checksum = ((checksum << 1) | (checksum >> 63)) & ((1 << 64) - 1)
                checksum ^= int(bits)
            print(
                f"TIME {elapsed!r} 0 {flattened.size} "
                f"{maximum_threads} {checksum}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def expm_diagonal_fixture(n: int) -> tuple[sp.csr_matrix, str]:
    data = np.asarray(
        [((index % 23) - 11) / 64.0 + 1.0 / 256.0 for index in range(n)],
        dtype=np.float64,
    )
    indices = np.arange(n, dtype=np.int64)
    indptr = np.arange(n + 1, dtype=np.int64)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
    input_hasher = hashlib.sha256()
    input_hasher.update(struct.pack("<Q", n))
    input_hasher.update(struct.pack("<Q", matrix.nnz))
    input_hasher.update(np.asarray(data, dtype="<f8").tobytes(order="C"))
    input_hasher.update(np.asarray(indices, dtype="<u8").tobytes(order="C"))
    input_hasher.update(np.asarray(indptr, dtype="<u8").tobytes(order="C"))
    return matrix, input_hasher.hexdigest()


def sparse_expm_identity() -> tuple[Path, str, bool]:
    solver_path_text = inspect.getsourcefile(spla.expm)
    if solver_path_text is None:
        raise RuntimeError("sparse expm source is unavailable")
    solver_path = Path(solver_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        spla.expm.__module__.startswith("scipy.sparse.linalg._matfuncs")
        and installed
        and scipy_path.parent in solver_path.parents
        and not fsci_loaded
    )
    return solver_path, hashlib.sha256(solver_path.read_bytes()).hexdigest(), genuine


def profile_sparse_expm(repetitions: int, n: int) -> int:
    if repetitions < 1 or n < 1:
        print("EXPM_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    solver_path, solver_sha256, genuine = sparse_expm_identity()
    print(
        f"EXPM_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={spla.expm.__module__} scipy_engine_file={solver_path} "
        f"scipy_engine_sha256={solver_sha256} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("EXPM_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    matrix, input_sha256 = expm_diagonal_fixture(n)
    warm = spla.expm(matrix)
    if not sp.issparse(warm):
        print("EXPM_SCIPY_FATAL sparse-result-required", flush=True)
        return 2
    result: sp.spmatrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = spla.expm(matrix)
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    result_csr = result.tocsr(copy=False)
    checksum = float(result_csr.data.sum()) + float(result_csr.nnz)
    print(
        f"EXPM_SCIPY_PROFILE n={n} nnz={matrix.nnz} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} checksum={checksum:.17e} "
        f"result_format={result_csr.format} result_nnz={result_csr.nnz} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def live_sparse_expm() -> int:
    try:
        solver_path, solver_sha256, genuine = sparse_expm_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method=expm "
        f"solver_mod={spla.expm.__module__} scipy_file={Path(scipy.__file__).resolve()} "
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
    input_sha256: str | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT":
            if len(parts) != 3:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            n, nnz = int(parts[1]), int(parts[2])
            try:
                indptr = parse_vector(
                    sys.stdin.readline(), "INDPTR", n + 1, np.int64
                )
                indices = parse_vector(
                    sys.stdin.readline(), "INDICES", nnz, np.int64
                )
                data = parse_vector(sys.stdin.readline(), "DATA", nnz, np.float64)
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            matrix = sp.csr_matrix(
                (data, indices, indptr), shape=(n, n), copy=False
            )
            input_hasher = hashlib.sha256()
            input_hasher.update(struct.pack("<Q", n))
            input_hasher.update(struct.pack("<Q", nnz))
            input_hasher.update(np.asarray(data, dtype="<f8").tobytes(order="C"))
            input_hasher.update(np.asarray(indices, dtype="<u8").tobytes(order="C"))
            input_hasher.update(np.asarray(indptr, dtype="<u8").tobytes(order="C"))
            input_sha256 = input_hasher.hexdigest()
            warm = spla.expm(matrix)
            if not sp.issparse(warm):
                print("FATAL sparse-result-required", flush=True)
                return 2
            print(
                f"CASE method=expm n={n} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} "
                f"finite={bool(np.isfinite(data).all())}",
                flush=True,
            )
            continue
        if parts[0] == "INPUT_SHA256":
            if input_sha256 is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            print(f"INPUT_SHA256 {input_sha256}", flush=True)
            continue
        if parts[0] == "PARITY":
            if matrix is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            result = spla.expm(matrix)
            if not sp.issparse(result):
                print("FATAL sparse-result-required", flush=True)
                return 2
            result_csr = result.tocsr(copy=False)
            result_csr.sort_indices()
            diagonal = np.asarray(result_csr.diagonal(), dtype=np.float64)
            off_diagonal = result_csr - sp.diags(diagonal, format="csr")
            offdiag_max = (
                float(np.max(np.abs(off_diagonal.data)))
                if off_diagonal.nnz
                else 0.0
            )
            print(
                f"RESULT rows={result_csr.shape[0]} cols={result_csr.shape[1]} "
                f"nnz={result_csr.nnz} sorted={result_csr.has_sorted_indices} "
                f"canonical={result_csr.has_canonical_format} "
                f"offdiag_max={offdiag_max!r}",
                flush=True,
            )
            print(
                "DIAG "
                + ",".join(format(float(value), ".17e") for value in diagonal),
                flush=True,
            )
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or matrix is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: sp.spmatrix | None = None
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = spla.expm(matrix)
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert result is not None
            result_csr = result.tocsr(copy=False)
            checksum = float(result_csr.data.sum()) + float(result_csr.nnz)
            print(
                f"TIME {elapsed!r} {result_csr.nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-sparse-expm":
        return profile_sparse_expm(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-expm":
        return live_sparse_expm()
    if len(sys.argv) == 5 and sys.argv[1] == "--profile-triangular-wavefront":
        return profile_triangular_wavefront(
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        )
    if len(sys.argv) == 6 and sys.argv[1] == "--profile-cuboid-spsolve":
        repetitions = int(sys.argv[2])
        x_extent = int(sys.argv[3])
        y_extent = int(sys.argv[4])
        z_extent = int(sys.argv[5])
        return profile_cuboid_spsolve(
            repetitions, x_extent, y_extent, z_extent
        )
    if len(sys.argv) in {3, 4, 5} and sys.argv[1] == "--profile-cubic-splu":
        repetitions = int(sys.argv[2])
        side = int(sys.argv[3]) if len(sys.argv) >= 4 else 16
        rhs_count = int(sys.argv[4]) if len(sys.argv) == 5 else 32
        return profile_cubic_splu(repetitions, side, rhs_count)
    if (
        len(sys.argv) in {3, 4, 5, 6}
        and sys.argv[1] == "--profile-convection-splu"
    ):
        repetitions = int(sys.argv[2])
        side = int(sys.argv[3]) if len(sys.argv) >= 4 else 64
        rhs_count = int(sys.argv[4]) if len(sys.argv) >= 5 else 16
        output_path = Path(sys.argv[5]) if len(sys.argv) == 6 else None
        return profile_convection_splu(
            repetitions, side, rhs_count, output_path
        )
    if (
        len(sys.argv) in {3, 4, 5}
        and sys.argv[1] == "--profile-neumann-cubic-splu"
    ):
        repetitions = int(sys.argv[2])
        side = int(sys.argv[3]) if len(sys.argv) >= 4 else 16
        rhs_count = int(sys.argv[4]) if len(sys.argv) == 5 else 32
        return profile_neumann_cubic_splu(repetitions, side, rhs_count)
    if (
        len(sys.argv) in {7, 8}
        and sys.argv[1] == "--profile-neumann-cuboid-splu"
    ):
        return profile_neumann_cuboid_splu(
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
            Path(sys.argv[7]) if len(sys.argv) == 8 else None,
        )
    if (
        len(sys.argv) in {7, 8}
        and sys.argv[1] == "--profile-periodic-cuboid-splu"
    ):
        return profile_periodic_cuboid_splu(
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
            Path(sys.argv[7]) if len(sys.argv) == 8 else None,
        )
    if (
        len(sys.argv) in {6, 7}
        and sys.argv[1] == "--profile-periodic-cuboid-spsolve"
    ):
        return profile_periodic_cuboid_spsolve(
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            Path(sys.argv[6]) if len(sys.argv) == 7 else None,
        )
    if len(sys.argv) in {3, 4} and sys.argv[1] == "--profile-cubic-spsolve":
        repetitions = int(sys.argv[2])
        side = int(sys.argv[3]) if len(sys.argv) == 4 else 16
        return profile_cubic_spsolve(repetitions, side)
    if len(sys.argv) == 3 and sys.argv[1:] == ["--live", "splu"]:
        return live_cubic_splu()
    if len(sys.argv) == 3 and sys.argv[1:] == ["--live", "spsolve_many"]:
        return live_cubic_splu("spsolve_many")
    if len(sys.argv) != 3 or sys.argv[1] != "--live" or sys.argv[2] not in METHODS:
        print(
            "usage: scipy_sparse_arm.py --live "
            "<cg|gmres|lgmres|bicg|cgs|bicgstab|lsqr|lsmr|qmr|spsolve|splu|spsolve_many>",
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
    expected_module = (
        "scipy.sparse.linalg._dsolve"
        if method in DIRECT_METHODS
        else "scipy.sparse.linalg._isolve"
    )
    genuine = (
        solver.__module__.startswith(expected_module)
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
    input_sha256: str | None = None
    rtol = 0.0
    maxiter = 0

    def scipy_converged(status: int) -> bool:
        if method in NO_CALLBACK_METHODS:
            return status in LEAST_SQUARES_CONVERGED_ISTOP
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
        if method in DIRECT_METHODS:
            return solver(matrix, rhs), 0, 0
        if method in NO_CALLBACK_METHODS:
            kwargs = {
                "damp": 0.0,
                "atol": 0.0,
                "btol": rtol,
                "conlim": 0.0,
            }
            if method == "lsqr":
                kwargs["iter_lim"] = maxiter
            else:
                kwargs["maxiter"] = maxiter
            result = solver(matrix, rhs, **kwargs)
            return result[0], int(result[1]), int(result[2])
        kwargs: dict[str, object] = {
            "rtol": rtol,
            "atol": 0.0,
            "maxiter": maxiter,
        }
        if method == "lgmres":
            kwargs.update(
                inner_m=30,
                outer_k=3,
                store_outer_Av=True,
                prepend_outer_v=False,
            )
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
            input_hasher = hashlib.sha256()
            input_hasher.update(struct.pack("<Q", n))
            input_hasher.update(struct.pack("<Q", nnz))
            input_hasher.update(np.asarray(data, dtype="<f8").tobytes(order="C"))
            input_hasher.update(np.asarray(indices, dtype="<u8").tobytes(order="C"))
            input_hasher.update(np.asarray(indptr, dtype="<u8").tobytes(order="C"))
            input_hasher.update(np.asarray(rhs, dtype="<f8").tobytes(order="C"))
            input_sha256 = input_hasher.hexdigest()
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
        if parts[0] == "INPUT_SHA256":
            if input_sha256 is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            print(f"INPUT_SHA256 {input_sha256}", flush=True)
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
            if method in DIRECT_METHODS:
                checksum = 0
                for bits in np.asarray(solution, dtype="<f8").view("<u8"):
                    checksum = ((checksum << 1) | (checksum >> 63)) & ((1 << 64) - 1)
                    checksum ^= int(bits)
            else:
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
