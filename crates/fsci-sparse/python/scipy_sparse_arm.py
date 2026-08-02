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


def laplacian_path_fixture(n: int) -> tuple[sp.csr_matrix, str]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for index in range(n - 1):
        weight = 1.0 + (index % 17) / 32.0
        rows.extend((index, index + 1))
        cols.extend((index + 1, index))
        data.extend((weight, weight))
    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n)
    ).tocsr()
    matrix.sort_indices()
    input_hasher = hashlib.sha256()
    input_hasher.update(struct.pack("<Q", n))
    input_hasher.update(struct.pack("<Q", matrix.nnz))
    input_hasher.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    input_hasher.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    input_hasher.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return matrix, input_hasher.hexdigest()


def laplacian_cycle_fixture(n: int) -> tuple[sp.csr_matrix, str]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for index in range(n):
        neighbor = (index + 1) % n
        weight = 1.0 + (index % 29) / 64.0
        rows.extend((index, neighbor))
        cols.extend((neighbor, index))
        data.extend((weight, weight))
    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n)
    ).tocsr()
    matrix.sort_indices()
    input_hasher = hashlib.sha256()
    input_hasher.update(struct.pack("<Q", n))
    input_hasher.update(struct.pack("<Q", matrix.nnz))
    input_hasher.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    input_hasher.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    input_hasher.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return matrix, input_hasher.hexdigest()


def sparse_laplacian_identity() -> tuple[Path, str, bool]:
    solver = sp.csgraph.laplacian
    solver_path_text = inspect.getsourcefile(solver)
    if solver_path_text is None:
        raise RuntimeError("sparse laplacian source is unavailable")
    solver_path = Path(solver_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        solver.__module__.startswith("scipy.sparse.csgraph._laplacian")
        and installed
        and scipy_path.parent in solver_path.parents
        and not fsci_loaded
    )
    return solver_path, hashlib.sha256(solver_path.read_bytes()).hexdigest(), genuine


def profile_sparse_laplacian(repetitions: int, n: int) -> int:
    if repetitions < 1 or n < 2:
        print("LAPLACIAN_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    solver_path, solver_sha256, genuine = sparse_laplacian_identity()
    print(
        f"LAPLACIAN_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={sp.csgraph.laplacian.__module__} "
        f"scipy_engine_file={solver_path} scipy_engine_sha256={solver_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("LAPLACIAN_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    matrix, input_sha256 = laplacian_path_fixture(n)
    warm = sp.csgraph.laplacian(matrix, normed=True, form="array")
    if not sp.issparse(warm):
        print("LAPLACIAN_SCIPY_FATAL sparse-result-required", flush=True)
        return 2
    result: sp.spmatrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = sp.csgraph.laplacian(matrix, normed=True, form="array")
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    result_csr = result.tocsr(copy=False)
    checksum = float(result_csr.data.sum()) + float(result_csr.nnz)
    print(
        f"LAPLACIAN_SCIPY_PROFILE n={n} input_nnz={matrix.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} result_format={result_csr.format} "
        f"result_nnz={result_csr.nnz} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def profile_sparse_laplacian_cycle(repetitions: int, n: int) -> int:
    if repetitions < 1 or n < 3:
        print("LAPLACIAN_CYCLE_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    solver_path, solver_sha256, genuine = sparse_laplacian_identity()
    print(
        f"LAPLACIAN_CYCLE_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={sp.csgraph.laplacian.__module__} "
        f"scipy_engine_file={solver_path} scipy_engine_sha256={solver_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("LAPLACIAN_CYCLE_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    matrix, input_sha256 = laplacian_cycle_fixture(n)
    warm = sp.csgraph.laplacian(matrix, normed=False, form="array")
    if not sp.issparse(warm):
        print("LAPLACIAN_CYCLE_SCIPY_FATAL sparse-result-required", flush=True)
        return 2
    result: sp.spmatrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = sp.csgraph.laplacian(matrix, normed=False, form="array")
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    result_csr = result.tocsr(copy=False)
    checksum = float(result_csr.data.sum()) + float(result_csr.nnz)
    print(
        f"LAPLACIAN_CYCLE_SCIPY_PROFILE n={n} input_nnz={matrix.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} result_format={result_csr.format} "
        f"result_nnz={result_csr.nnz} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def live_sparse_laplacian(normed: bool = True) -> int:
    try:
        solver_path, solver_sha256, genuine = sparse_laplacian_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method=laplacian "
        f"solver_mod={sp.csgraph.laplacian.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
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
            warm = sp.csgraph.laplacian(matrix, normed=normed, form="array")
            if not sp.issparse(warm):
                print("FATAL sparse-result-required", flush=True)
                return 2
            print(
                f"CASE method=laplacian n={n} nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} "
                f"finite={bool(np.isfinite(data).all())} normed={normed} form=array",
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
            result = sp.csgraph.laplacian(matrix, normed=normed, form="array")
            if not sp.issparse(result):
                print("FATAL sparse-result-required", flush=True)
                return 2
            result_csr = result.tocsr(copy=False)
            result_csr.sort_indices()
            print(
                f"RESULT rows={result_csr.shape[0]} cols={result_csr.shape[1]} "
                f"nnz={result_csr.nnz} sorted={result_csr.has_sorted_indices} "
                f"canonical={result_csr.has_canonical_format}",
                flush=True,
            )
            print(
                "OUT_INDPTR "
                + ",".join(str(int(value)) for value in result_csr.indptr),
                flush=True,
            )
            print(
                "OUT_INDICES "
                + ",".join(str(int(value)) for value in result_csr.indices),
                flush=True,
            )
            print(
                "OUT_DATA "
                + ",".join(
                    format(float(value), ".17e") for value in result_csr.data
                ),
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
                result = sp.csgraph.laplacian(matrix, normed=normed, form="array")
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


def csc_add_operand(n: int, side: int) -> sp.csc_matrix:
    entries_per_column = 24
    nnz = n * entries_per_column
    data = np.empty(nnz, dtype=np.float64)
    indices = np.empty(nnz, dtype=np.int64)
    indptr = np.arange(0, nnz + 1, entries_per_column, dtype=np.int64)
    offset = 0
    for column in range(n):
        entries = [
            (
                (173 * slot + 17 * column + 89 * side) % n,
                ((column + 3 * slot + 11 * side) % 37 - 18) / 32.0,
            )
            for slot in range(entries_per_column)
        ]
        entries.sort(key=lambda entry: entry[0])
        for row, value in entries:
            indices[offset] = row
            data[offset] = value
            offset += 1
    matrix = sp.csc_matrix((data, indices, indptr), shape=(n, n), copy=False)
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise RuntimeError("generated CSC-add operand is not canonical")
    return matrix


def csc_pair_sha256(lhs: sp.csc_matrix, rhs: sp.csc_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", lhs.shape[0]))
    for matrix in (lhs, rhs):
        digest.update(struct.pack("<Q", int(matrix.nnz)))
        digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
        digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def sparse_csc_add_identity() -> tuple[Path, str, bool]:
    engine = sp.csc_matrix._binopt
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        raise RuntimeError("CSC-add engine source is unavailable")
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__.startswith("scipy.sparse._compressed")
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    return engine_path, hashlib.sha256(engine_path.read_bytes()).hexdigest(), genuine


def profile_sparse_csc_add(repetitions: int, n: int) -> int:
    if repetitions < 1 or n != 4096:
        print("CSC_ADD_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    try:
        engine_path, engine_sha256, genuine = sparse_csc_add_identity()
        lhs = csc_add_operand(n, 0)
        rhs = csc_add_operand(n, 1)
    except RuntimeError as error:
        print(f"CSC_ADD_SCIPY_FATAL {error}", flush=True)
        return 2
    print(
        f"CSC_ADD_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={sp.csc_matrix._binopt.__module__} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("CSC_ADD_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    input_sha256 = csc_pair_sha256(lhs, rhs)
    warm = lhs + rhs
    if not isinstance(warm, sp.csc_matrix):
        print("CSC_ADD_SCIPY_FATAL csc-result-required", flush=True)
        return 2
    result: sp.csc_matrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = lhs + rhs
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    checksum = float(result.data.sum()) + float(result.nnz)
    print(
        f"CSC_ADD_SCIPY_PROFILE n={n} lhs_nnz={lhs.nnz} rhs_nnz={rhs.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} result_format={result.format} "
        f"result_nnz={result.nnz} sorted={result.has_sorted_indices} "
        f"canonical={result.has_canonical_format} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={input_sha256}",
        flush=True,
    )
    return 0


def live_sparse_csc_add() -> int:
    try:
        engine_path, engine_sha256, genuine = sparse_csc_add_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method=csc_add "
        f"solver_mod={sp.csc_matrix._binopt.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    lhs: sp.csc_matrix | None = None
    rhs: sp.csc_matrix | None = None
    input_sha256: str | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_CSC_ADD":
            if len(parts) != 4:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            n, lhs_nnz, rhs_nnz = int(parts[1]), int(parts[2]), int(parts[3])
            try:
                lhs_indptr = parse_vector(
                    sys.stdin.readline(), "LHS_INDPTR", n + 1, np.int64
                )
                lhs_indices = parse_vector(
                    sys.stdin.readline(), "LHS_INDICES", lhs_nnz, np.int64
                )
                lhs_data = parse_vector(
                    sys.stdin.readline(), "LHS_DATA", lhs_nnz, np.float64
                )
                rhs_indptr = parse_vector(
                    sys.stdin.readline(), "RHS_INDPTR", n + 1, np.int64
                )
                rhs_indices = parse_vector(
                    sys.stdin.readline(), "RHS_INDICES", rhs_nnz, np.int64
                )
                rhs_data = parse_vector(
                    sys.stdin.readline(), "RHS_DATA", rhs_nnz, np.float64
                )
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            lhs = sp.csc_matrix(
                (lhs_data, lhs_indices, lhs_indptr), shape=(n, n), copy=False
            )
            rhs = sp.csc_matrix(
                (rhs_data, rhs_indices, rhs_indptr), shape=(n, n), copy=False
            )
            input_sha256 = csc_pair_sha256(lhs, rhs)
            finite = bool(np.isfinite(lhs_data).all() and np.isfinite(rhs_data).all())
            warm = lhs + rhs
            if not isinstance(warm, sp.csc_matrix):
                print("FATAL csc-result-required", flush=True)
                return 2
            print(
                f"CASE method=csc_add n={n} lhs_nnz={lhs.nnz} rhs_nnz={rhs.nnz} "
                f"lhs_sorted={lhs.has_sorted_indices} "
                f"rhs_sorted={rhs.has_sorted_indices} "
                f"lhs_canonical={lhs.has_canonical_format} "
                f"rhs_canonical={rhs.has_canonical_format} finite={finite}",
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
            if lhs is None or rhs is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            result = lhs + rhs
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format}",
                flush=True,
            )
            print(
                "OUT_INDPTR "
                + ",".join(str(int(value)) for value in result.indptr),
                flush=True,
            )
            print(
                "OUT_INDICES "
                + ",".join(str(int(value)) for value in result.indices),
                flush=True,
            )
            print(
                "OUT_DATA "
                + ",".join(format(float(value), ".17e") for value in result.data),
                flush=True,
            )
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or lhs is None or rhs is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: sp.csc_matrix | None = None
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = lhs + rhs
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert result is not None
            checksum = float(result.data.sum()) + float(result.nnz)
            print(
                f"TIME {elapsed!r} {result.nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def coo_is_strictly_lexicographic(rows: np.ndarray, cols: np.ndarray) -> bool:
    if rows.size <= 1:
        return True
    return bool(
        np.all(
            (rows[1:] > rows[:-1])
            | ((rows[1:] == rows[:-1]) & (cols[1:] > cols[:-1]))
        )
    )


def coo_sub_operand(n: int, side: int) -> sp.coo_matrix:
    entries_per_row = 40
    nnz = n * entries_per_row
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    offset = 0
    for row in range(n):
        entries = [
            (
                (313 * slot + 37 * row + 6260 * side) % n,
                (1 + ((19 * row + 29 * slot + 31 * side) % 509)) / 512.0,
            )
            for slot in range(entries_per_row)
        ]
        entries.sort(key=lambda entry: entry[0])
        for column, value in entries:
            rows[offset] = row
            cols[offset] = column
            data[offset] = value
            offset += 1
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n), copy=False)
    if not coo_is_strictly_lexicographic(matrix.row, matrix.col):
        raise RuntimeError("generated COO-sub operand is not sorted and unique")
    return matrix


def coo_pair_sha256(lhs: sp.coo_matrix, rhs: sp.coo_matrix) -> str:
    digest = hashlib.sha256()
    for matrix in (lhs, rhs):
        digest.update(struct.pack("<Q", matrix.shape[0]))
        digest.update(struct.pack("<Q", matrix.shape[1]))
        digest.update(struct.pack("<Q", int(matrix.nnz)))
        digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(matrix.row, dtype="<u8").tobytes(order="C"))
        digest.update(np.asarray(matrix.col, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def sparse_coo_sub_identity() -> tuple[Path, str, bool]:
    engine = sp.coo_matrix._sub_sparse
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        raise RuntimeError("COO-sub engine source is unavailable")
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__.startswith("scipy.sparse._coo")
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    return engine_path, hashlib.sha256(engine_path.read_bytes()).hexdigest(), genuine


def profile_sparse_coo_sub(repetitions: int, n: int) -> int:
    if repetitions < 1 or n != 24576:
        print("COO_SUB_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    try:
        engine_path, engine_sha256, genuine = sparse_coo_sub_identity()
        lhs = coo_sub_operand(n, 0)
        rhs = coo_sub_operand(n, 1)
    except RuntimeError as error:
        print(f"COO_SUB_SCIPY_FATAL {error}", flush=True)
        return 2
    print(
        f"COO_SUB_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={sp.coo_matrix._sub_sparse.__module__} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("COO_SUB_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    input_digest = coo_pair_sha256(lhs, rhs)
    warm = lhs - rhs
    if not isinstance(warm, sp.csr_matrix) or warm.nnz != 1474560:
        print("COO_SUB_SCIPY_FATAL canonical-csr-result-required", flush=True)
        return 2
    checksum = 0.0
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = lhs - rhs
        checksum += float(result.nnz) + float(result.data[result.nnz // 2])
        del result
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"COO_SUB_SCIPY_PROFILE n={n} lhs_nnz={lhs.nnz} rhs_nnz={rhs.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"checksum={checksum:.17e} result_format={warm.format} "
        f"result_nnz={warm.nnz} sorted={warm.has_sorted_indices} "
        f"canonical={warm.has_canonical_format} "
        f"actual_observed_worker_threads={maximum_threads} input_sha256={input_digest}",
        flush=True,
    )
    return 0


def live_sparse_coo_sub() -> int:
    try:
        engine_path, engine_sha256, genuine = sparse_coo_sub_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method=coo_sub "
        f"solver_mod={sp.coo_matrix._sub_sparse.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    lhs: sp.coo_matrix | None = None
    rhs: sp.coo_matrix | None = None
    input_digest: str | None = None
    expected_result_nnz: int | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_COO_SUB":
            if len(parts) != 5:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            rows_count = int(parts[1])
            cols_count = int(parts[2])
            lhs_nnz = int(parts[3])
            rhs_nnz = int(parts[4])
            try:
                lhs_rows = parse_vector(
                    sys.stdin.readline(), "LHS_ROWS", lhs_nnz, np.int64
                )
                lhs_cols = parse_vector(
                    sys.stdin.readline(), "LHS_COLS", lhs_nnz, np.int64
                )
                lhs_data = parse_vector(
                    sys.stdin.readline(), "LHS_DATA", lhs_nnz, np.float64
                )
                rhs_rows = parse_vector(
                    sys.stdin.readline(), "RHS_ROWS", rhs_nnz, np.int64
                )
                rhs_cols = parse_vector(
                    sys.stdin.readline(), "RHS_COLS", rhs_nnz, np.int64
                )
                rhs_data = parse_vector(
                    sys.stdin.readline(), "RHS_DATA", rhs_nnz, np.float64
                )
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            lhs = sp.coo_matrix(
                (lhs_data, (lhs_rows, lhs_cols)),
                shape=(rows_count, cols_count),
                copy=False,
            )
            rhs = sp.coo_matrix(
                (rhs_data, (rhs_rows, rhs_cols)),
                shape=(rows_count, cols_count),
                copy=False,
            )
            lhs_sorted = coo_is_strictly_lexicographic(lhs.row, lhs.col)
            rhs_sorted = coo_is_strictly_lexicographic(rhs.row, rhs.col)
            finite = bool(np.isfinite(lhs.data).all() and np.isfinite(rhs.data).all())
            input_digest = coo_pair_sha256(lhs, rhs)
            warm = lhs - rhs
            if not isinstance(warm, sp.csr_matrix):
                print("FATAL canonical-csr-result-required", flush=True)
                return 2
            expected_result_nnz = int(warm.nnz)
            print(
                f"CASE method=coo_sub rows={rows_count} cols={cols_count} "
                f"lhs_nnz={lhs.nnz} rhs_nnz={rhs.nnz} "
                f"lhs_sorted={lhs_sorted} rhs_sorted={rhs_sorted} "
                f"lhs_unique={lhs_sorted} rhs_unique={rhs_sorted} finite={finite} "
                f"result_format={warm.format} result_nnz={warm.nnz} "
                f"sorted={warm.has_sorted_indices} canonical={warm.has_canonical_format}",
                flush=True,
            )
            continue
        if parts[0] == "INPUT_SHA256":
            if input_digest is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            print(f"INPUT_SHA256 {input_digest}", flush=True)
            continue
        if parts[0] == "PARITY":
            if lhs is None or rhs is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            result = (lhs - rhs).tocsr(copy=False)
            result.sum_duplicates()
            result.sort_indices()
            finite = bool(np.isfinite(result.data).all())
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} format={result.format} "
                f"sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format} finite={finite} "
                f"first_pointer={int(result.indptr[0])} "
                f"middle_pointer={int(result.indptr[result.shape[0] // 2])} "
                f"last_pointer={int(result.indptr[-1])}",
                flush=True,
            )
            print(f"OUTPUT_SHA256 {compressed_matrix_sha256(result)}", flush=True)
            continue
        if parts[0] == "SOLVE":
            if (
                len(parts) != 2
                or lhs is None
                or rhs is None
                or expected_result_nnz is None
            ):
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            checksum = 0.0
            maximum_threads = observed_threads()
            started = time.perf_counter()
            result_nnz = 0
            for _ in range(repetitions):
                result = lhs - rhs
                result_nnz = int(result.nnz)
                if result_nnz != expected_result_nnz:
                    print("FATAL wrong-result-nnz", flush=True)
                    return 2
                checksum += float(result_nnz) + float(result.data[result_nnz // 2])
                del result
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            print(
                f"TIME {elapsed!r} {result_nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def compressed_matrix_sha256(matrix: sp.spmatrix) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", matrix.shape[0]))
    digest.update(struct.pack("<Q", matrix.shape[1]))
    digest.update(struct.pack("<Q", int(matrix.nnz)))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def hstack_input_sha256_parts(blocks: list[sp.csr_matrix]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(blocks)))
    for matrix in blocks:
        digest.update(struct.pack("<Q", matrix.shape[0]))
        digest.update(struct.pack("<Q", matrix.shape[1]))
        digest.update(struct.pack("<Q", int(matrix.nnz)))
        digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
        digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def hstack_csr_fixture(rows: int) -> list[sp.csr_matrix]:
    if rows != 65536:
        raise RuntimeError("canonical CSR hstack fixture requires 65536 rows")
    block_count = 8
    block_cols = 32768
    entries_per_row = 4
    block_nnz = rows * entries_per_row
    blocks: list[sp.csr_matrix] = []
    for block in range(block_count):
        data = np.empty(block_nnz, dtype=np.float64)
        indices = np.empty(block_nnz, dtype=np.int64)
        indptr = np.arange(
            0, block_nnz + 1, entries_per_row, dtype=np.int64
        )
        offset = 0
        for row in range(rows):
            entries = [
                (
                    (4099 * slot + 73 * row + 211 * block) % block_cols,
                    slot,
                )
                for slot in range(entries_per_row)
            ]
            entries.sort(key=lambda entry: entry[0])
            for column, slot in entries:
                indices[offset] = column
                data[offset] = (
                    1 + ((17 * row + 29 * slot + 31 * block) % 997)
                ) / 1024.0
                offset += 1
        matrix = sp.csr_matrix(
            (data, indices, indptr), shape=(rows, block_cols), copy=False
        )
        if not matrix.has_sorted_indices or not matrix.has_canonical_format:
            raise RuntimeError("canonical CSR hstack fixture is not canonical")
        blocks.append(matrix)
    return blocks


def sparse_hstack_identity() -> tuple[Path, str, bool]:
    engine = sp.hstack
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        raise RuntimeError("hstack engine source is unavailable")
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__ == "scipy.sparse._construct"
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    return engine_path, hashlib.sha256(engine_path.read_bytes()).hexdigest(), genuine


def profile_sparse_hstack_csr(repetitions: int, rows: int) -> int:
    if repetitions < 1 or rows != 65536:
        print("HSTACK_CSR_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    try:
        engine_path, engine_sha256, genuine = sparse_hstack_identity()
        blocks = hstack_csr_fixture(rows)
    except RuntimeError as error:
        print(f"HSTACK_CSR_SCIPY_FATAL {error}", flush=True)
        return 2
    print(
        f"HSTACK_CSR_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={sp.hstack.__module__} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("HSTACK_CSR_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    warm = sp.hstack(blocks, format="csr")
    assert warm.shape == (65536, 262144)
    assert warm.nnz == 2097152
    checksum = 0.0
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = sp.hstack(blocks, format="csr")
        checksum += float(result.nnz) + float(result.data[result.nnz // 2])
        del result
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    print(
        f"HSTACK_CSR_SCIPY_PROFILE blocks=8 rows={rows} block_cols=32768 "
        f"entries_per_row=4 block_nnz=262144 output_cols=262144 "
        f"output_nnz={warm.nnz} repetitions={repetitions} "
        f"elapsed_seconds={elapsed:.9f} result_format={warm.format} "
        f"sorted={warm.has_sorted_indices} canonical={warm.has_canonical_format} "
        f"actual_observed_worker_threads={maximum_threads} checksum={checksum!r} "
        f"input_sha256={hstack_input_sha256_parts(blocks)}",
        flush=True,
    )
    return 0


def live_sparse_hstack_csr() -> int:
    try:
        engine_path, engine_sha256, genuine = sparse_hstack_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"method=hstack_csr solver_mod={sp.hstack.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    blocks: list[sp.csr_matrix] | None = None
    input_sha256: str | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_HSTACK":
            if len(parts) != 5:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            block_count, expected_rows, expected_cols, expected_nnz = map(
                int, parts[1:]
            )
            loaded: list[sp.csr_matrix] = []
            try:
                for block_index in range(block_count):
                    block_line = sys.stdin.readline().strip().split()
                    if (
                        len(block_line) != 5
                        or block_line[0] != "BLOCK"
                        or int(block_line[1]) != block_index
                    ):
                        raise ValueError(f"bad hstack block header {block_line}")
                    rows, cols, nnz = map(int, block_line[2:])
                    if (
                        rows != expected_rows
                        or cols != expected_cols
                        or nnz != expected_nnz
                    ):
                        raise ValueError("hstack block shape or nnz mismatch")
                    indptr = parse_vector(
                        sys.stdin.readline(), "HSTACK_INDPTR", rows + 1, np.int64
                    )
                    indices = parse_vector(
                        sys.stdin.readline(), "HSTACK_INDICES", nnz, np.int64
                    )
                    data = parse_vector(
                        sys.stdin.readline(), "HSTACK_DATA", nnz, np.float64
                    )
                    if int(indptr[0]) != 0 or int(indptr[-1]) != nnz:
                        raise ValueError("invalid hstack pointers")
                    matrix = sp.csr_matrix(
                        (data, indices, indptr), shape=(rows, cols), copy=False
                    )
                    if (
                        not matrix.has_sorted_indices
                        or not matrix.has_canonical_format
                        or not bool(np.isfinite(matrix.data).all())
                    ):
                        raise ValueError("invalid canonical hstack block")
                    loaded.append(matrix)
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            blocks = loaded
            input_sha256 = hstack_input_sha256_parts(blocks)
            warm = sp.hstack(blocks, format="csr")
            print(
                f"CASE method=hstack_csr blocks={len(blocks)} "
                f"block_rows={expected_rows} block_cols={expected_cols} "
                f"block_nnz={expected_nnz} output_rows={warm.shape[0]} "
                f"output_cols={warm.shape[1]} output_nnz={warm.nnz} "
                f"sorted={warm.has_sorted_indices} "
                f"canonical={warm.has_canonical_format} "
                f"finite={bool(np.isfinite(warm.data).all())} "
                f"result_format={warm.format}",
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
            if blocks is None:
                print("FATAL fixture-not-initialized", flush=True)
                return 2
            result = sp.hstack(blocks, format="csr")
            middle = result.shape[0] // 2
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} format={result.format} "
                f"sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format} "
                f"finite={bool(np.isfinite(result.data).all())} "
                f"first_pointer={int(result.indptr[0])} "
                f"middle_pointer={int(result.indptr[middle])} "
                f"last_pointer={int(result.indptr[-1])}",
                flush=True,
            )
            print(f"OUTPUT_SHA256 {compressed_matrix_sha256(result)}", flush=True)
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or blocks is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result_nnz = 0
            checksum = 0.0
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = sp.hstack(blocks, format="csr")
                result_nnz = int(result.nnz)
                checksum += float(result_nnz) + float(result.data[result_nnz // 2])
                del result
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            print(
                f"TIME {elapsed!r} {result_nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def lil_input_sha256_parts(
    rows: int,
    cols: int,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", rows))
    digest.update(struct.pack("<Q", cols))
    digest.update(struct.pack("<Q", int(data.size)))
    digest.update(np.asarray(data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(indptr, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def lil_from_flat_parts(
    rows: int,
    cols: int,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> sp.lil_matrix:
    matrix = sp.lil_matrix((rows, cols), dtype=np.float64)
    object_rows = np.empty(rows, dtype=object)
    object_data = np.empty(rows, dtype=object)
    for row in range(rows):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        object_rows[row] = indices[start:end].tolist()
        object_data[row] = data[start:end].tolist()
    matrix.rows = object_rows
    matrix.data = object_data
    return matrix


def lil_skew_to_csr_fixture(
    rows: int,
) -> tuple[sp.lil_matrix, np.ndarray, np.ndarray, np.ndarray]:
    if rows != 65536:
        raise RuntimeError("skewed LIL fixture requires 65536 rows")
    cols = 262144
    widths = (0, 1, 3, 7, 15, 31, 63, 127)
    indptr = np.empty(rows + 1, dtype=np.int64)
    indptr[0] = 0
    for row in range(rows):
        indptr[row + 1] = indptr[row] + widths[row % len(widths)]
    nnz = int(indptr[-1])
    data = np.empty(nnz, dtype=np.float64)
    indices = np.empty(nnz, dtype=np.int64)
    offset = 0
    for row in range(rows):
        width = widths[row % len(widths)]
        entries = [
            ((4093 * slot + 97 * row) % cols, slot) for slot in range(width)
        ]
        entries.sort(key=lambda entry: entry[0])
        for column, slot in entries:
            indices[offset] = column
            data[offset] = (1 + ((29 * row + 37 * slot) % 997)) / 1024.0
            offset += 1
    matrix = lil_from_flat_parts(rows, cols, data, indices, indptr)
    return matrix, data, indices, indptr


def sparse_lil_to_csr_identity() -> tuple[Path, str, bool]:
    engine = sp.lil_matrix.tocsr
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        raise RuntimeError("LIL-to-CSR engine source is unavailable")
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__ == "scipy.sparse._lil"
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    return engine_path, hashlib.sha256(engine_path.read_bytes()).hexdigest(), genuine


def profile_sparse_lil_skew_to_csr(repetitions: int, rows: int) -> int:
    if repetitions < 1 or rows != 65536:
        print("LIL_SKEW_TO_CSR_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    try:
        engine_path, engine_sha256, genuine = sparse_lil_to_csr_identity()
        matrix, data, indices, indptr = lil_skew_to_csr_fixture(rows)
    except RuntimeError as error:
        print(f"LIL_SKEW_TO_CSR_SCIPY_FATAL {error}", flush=True)
        return 2
    print(
        f"LIL_SKEW_TO_CSR_SCIPY_READY scipy={scipy.__version__} "
        f"numpy={np.__version__} solver_mod={sp.lil_matrix.tocsr.__module__} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("LIL_SKEW_TO_CSR_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    warm = matrix.tocsr(copy=True)
    assert warm.nnz == 2023424
    result: sp.csr_matrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = matrix.tocsr(copy=True)
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    print(
        f"LIL_SKEW_TO_CSR_SCIPY_PROFILE rows={rows} cols=262144 "
        f"widths=0,1,3,7,15,31,63,127 nnz={matrix.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"result_format={result.format} result_nnz={result.nnz} "
        f"sorted={result.has_sorted_indices} canonical={result.has_canonical_format} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={lil_input_sha256_parts(rows, 262144, data, indices, indptr)}",
        flush=True,
    )
    return 0


def live_sparse_lil_skew_to_csr() -> int:
    try:
        engine_path, engine_sha256, genuine = sparse_lil_to_csr_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"method=lil_skew_to_csr solver_mod={sp.lil_matrix.tocsr.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    matrix: sp.lil_matrix | None = None
    input_sha256: str | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_LIL":
            if len(parts) != 4:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            rows, cols, nnz = int(parts[1]), int(parts[2]), int(parts[3])
            try:
                indptr = parse_vector(
                    sys.stdin.readline(), "LIL_INDPTR", rows + 1, np.int64
                )
                indices = parse_vector(
                    sys.stdin.readline(), "LIL_INDICES", nnz, np.int64
                )
                data = parse_vector(
                    sys.stdin.readline(), "LIL_DATA", nnz, np.float64
                )
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            if int(indptr[0]) != 0 or int(indptr[-1]) != nnz:
                print("FATAL invalid-lil-pointers", flush=True)
                return 2
            matrix = lil_from_flat_parts(rows, cols, data, indices, indptr)
            input_sha256 = lil_input_sha256_parts(
                rows, cols, data, indices, indptr
            )
            warm = matrix.tocsr(copy=True)
            row_lengths = np.diff(indptr)
            sorted_rows = all(
                all(left < right for left, right in zip(row, row[1:]))
                for row in matrix.rows
            )
            print(
                f"CASE method=lil_skew_to_csr rows={rows} cols={cols} "
                f"nnz={matrix.nnz} min_row_nnz={int(row_lengths.min())} "
                f"max_row_nnz={int(row_lengths.max())} "
                f"empty_rows={int(np.count_nonzero(row_lengths == 0))} "
                f"sorted={sorted_rows} canonical={sorted_rows} "
                f"finite={bool(np.isfinite(data).all())} "
                f"result_format={warm.format} result_nnz={warm.nnz}",
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
            result = matrix.tocsr(copy=True)
            middle = result.shape[0] // 2
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} format={result.format} "
                f"sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format} "
                f"finite={bool(np.isfinite(result.data).all())} "
                f"first_pointer={int(result.indptr[0])} "
                f"middle_pointer={int(result.indptr[middle])} "
                f"last_pointer={int(result.indptr[-1])}",
                flush=True,
            )
            print(f"OUTPUT_SHA256 {compressed_matrix_sha256(result)}", flush=True)
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or matrix is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: sp.csr_matrix | None = None
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = matrix.tocsr(copy=True)
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert result is not None
            checksum = float(result.data.sum()) + float(result.nnz)
            print(
                f"TIME {elapsed!r} {result.nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def bsr_input_sha256(matrix: sp.bsr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", matrix.shape[0]))
    digest.update(struct.pack("<Q", matrix.shape[1]))
    digest.update(struct.pack("<Q", matrix.blocksize[0]))
    digest.update(struct.pack("<Q", matrix.blocksize[1]))
    digest.update(struct.pack("<Q", int(matrix.indices.size)))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<u8").tobytes(order="C"))
    return digest.hexdigest()


def bsr_to_csr_fixture(n: int) -> sp.bsr_matrix:
    block_side = 4
    block_rows = n // block_side
    blocks_per_row = 24
    stored_blocks = block_rows * blocks_per_row
    data = np.empty((stored_blocks, block_side, block_side), dtype=np.float64)
    flat_data = data.reshape(stored_blocks, -1)
    indices = np.empty(stored_blocks, dtype=np.int64)
    indptr = np.arange(0, stored_blocks + 1, blocks_per_row, dtype=np.int64)
    offset = 0
    for block_row in range(block_rows):
        entries = [
            ((257 * slot + 31 * block_row) % block_rows, slot)
            for slot in range(blocks_per_row)
        ]
        entries.sort(key=lambda entry: entry[0])
        for block_column, slot in entries:
            indices[offset] = block_column
            for local_offset in range(block_side * block_side):
                flat_data[offset, local_offset] = (
                    1
                    + (
                        13 * block_row
                        + 17 * slot
                        + 19 * local_offset
                    )
                    % 251
                ) / 256.0
            offset += 1
    matrix = sp.bsr_matrix(
        (data, indices, indptr),
        shape=(n, n),
        copy=False,
    )
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise RuntimeError("generated BSR-to-CSR fixture is not canonical")
    return matrix


def sparse_bsr_to_csr_identity() -> tuple[Path, str, bool]:
    engine = sp.bsr_matrix.tocsr
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        raise RuntimeError("BSR-to-CSR engine source is unavailable")
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__ == "scipy.sparse._bsr"
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    return engine_path, hashlib.sha256(engine_path.read_bytes()).hexdigest(), genuine


def profile_sparse_bsr_to_csr(repetitions: int, n: int) -> int:
    if repetitions < 1 or n != 32768:
        print("BSR_TO_CSR_SCIPY_FATAL invalid-controls", flush=True)
        return 2
    try:
        engine_path, engine_sha256, genuine = sparse_bsr_to_csr_identity()
        matrix = bsr_to_csr_fixture(n)
    except RuntimeError as error:
        print(f"BSR_TO_CSR_SCIPY_FATAL {error}", flush=True)
        return 2
    print(
        f"BSR_TO_CSR_SCIPY_READY scipy={scipy.__version__} numpy={np.__version__} "
        f"solver_mod={sp.bsr_matrix.tocsr.__module__} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("BSR_TO_CSR_SCIPY_FATAL not-genuine-scipy", flush=True)
        return 2
    warm = matrix.tocsr(copy=True)
    result: sp.csr_matrix | None = None
    maximum_threads = observed_threads()
    started = time.perf_counter()
    for _ in range(repetitions):
        result = matrix.tocsr(copy=True)
    elapsed = time.perf_counter() - started
    maximum_threads = max(maximum_threads, observed_threads())
    assert result is not None
    print(
        f"BSR_TO_CSR_SCIPY_PROFILE n={n} block_side=4 "
        f"stored_blocks={matrix.indices.size} scalar_nnz={matrix.nnz} "
        f"repetitions={repetitions} elapsed_seconds={elapsed:.9f} "
        f"result_format={result.format} result_nnz={result.nnz} "
        f"sorted={result.has_sorted_indices} canonical={result.has_canonical_format} "
        f"actual_observed_worker_threads={maximum_threads} "
        f"input_sha256={bsr_input_sha256(matrix)}",
        flush=True,
    )
    return 0


def live_sparse_bsr_to_csr() -> int:
    try:
        engine_path, engine_sha256, genuine = sparse_bsr_to_csr_identity()
    except RuntimeError as error:
        print(f"FATAL {error}", flush=True)
        return 2
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} method=bsr_to_csr "
        f"solver_mod={sp.bsr_matrix.tocsr.__module__} "
        f"scipy_file={Path(scipy.__file__).resolve()} "
        f"scipy_engine_file={engine_path} scipy_engine_sha256={engine_sha256} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    matrix: sp.bsr_matrix | None = None
    input_sha256: str | None = None
    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] == "INIT_BSR":
            if len(parts) != 6:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            rows, cols = int(parts[1]), int(parts[2])
            block_rows, block_cols = int(parts[3]), int(parts[4])
            stored_blocks = int(parts[5])
            if rows % block_rows != 0 or cols % block_cols != 0:
                print("FATAL non-divisible-block-shape", flush=True)
                return 2
            try:
                indptr = parse_vector(
                    sys.stdin.readline(),
                    "BSR_INDPTR",
                    rows // block_rows + 1,
                    np.int64,
                )
                indices = parse_vector(
                    sys.stdin.readline(),
                    "BSR_INDICES",
                    stored_blocks,
                    np.int64,
                )
                data = parse_vector(
                    sys.stdin.readline(),
                    "BSR_DATA",
                    stored_blocks * block_rows * block_cols,
                    np.float64,
                )
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            matrix = sp.bsr_matrix(
                (
                    data.reshape(stored_blocks, block_rows, block_cols),
                    indices,
                    indptr,
                ),
                shape=(rows, cols),
                copy=False,
            )
            input_sha256 = bsr_input_sha256(matrix)
            warm = matrix.tocsr(copy=True)
            print(
                f"CASE method=bsr_to_csr rows={rows} cols={cols} "
                f"block_rows={block_rows} block_cols={block_cols} "
                f"stored_blocks={stored_blocks} scalar_nnz={matrix.nnz} "
                f"sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} "
                f"finite={bool(np.isfinite(data).all())} "
                f"result_format={warm.format} result_nnz={warm.nnz}",
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
            result = matrix.tocsr(copy=True)
            middle = result.shape[0] // 2
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} format={result.format} "
                f"sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format} "
                f"finite={bool(np.isfinite(result.data).all())} "
                f"first_pointer={int(result.indptr[0])} "
                f"middle_pointer={int(result.indptr[middle])} "
                f"last_pointer={int(result.indptr[-1])}",
                flush=True,
            )
            print(f"OUTPUT_SHA256 {compressed_matrix_sha256(result)}", flush=True)
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or matrix is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: sp.csr_matrix | None = None
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = matrix.tocsr(copy=True)
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert result is not None
            checksum = float(result.data.sum()) + float(result.nnz)
            print(
                f"TIME {elapsed!r} {result.nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def live_sparse_transpose() -> int:
    engine = sp.csr_matrix.transpose
    engine_path_text = inspect.getsourcefile(engine)
    if engine_path_text is None:
        print("FATAL transpose-engine-source-unavailable", flush=True)
        return 2
    engine_path = Path(engine_path_text).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    genuine = (
        engine.__module__ == "scipy.sparse._csr"
        and installed
        and scipy_path.parent in engine_path.parents
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"method=csr_transpose_view solver_mod={engine.__module__} "
        f"scipy_file={scipy_path} scipy_engine_file={engine_path} "
        f"scipy_engine_sha256={hashlib.sha256(engine_path.read_bytes()).hexdigest()} "
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
        if parts[0] == "INIT_TRANSPOSE":
            if len(parts) != 4:
                print(f"FATAL bad-init {line}", flush=True)
                return 2
            rows, cols, nnz = int(parts[1]), int(parts[2]), int(parts[3])
            try:
                indptr = parse_vector(
                    sys.stdin.readline(), "INDPTR", rows + 1, np.int64
                )
                indices = parse_vector(
                    sys.stdin.readline(), "INDICES", nnz, np.int64
                )
                data = parse_vector(sys.stdin.readline(), "DATA", nnz, np.float64)
            except ValueError as error:
                print(f"FATAL {error}", flush=True)
                return 2
            matrix = sp.csr_matrix(
                (data, indices, indptr), shape=(rows, cols), copy=False
            )
            input_sha256 = compressed_matrix_sha256(matrix)
            warm = matrix.T
            data_shared = np.shares_memory(matrix.data, warm.data)
            indices_shared = np.shares_memory(matrix.indices, warm.indices)
            indptr_shared = np.shares_memory(matrix.indptr, warm.indptr)
            print(
                f"CASE method=csr_transpose_view rows={rows} cols={cols} "
                f"nnz={matrix.nnz} sorted={matrix.has_sorted_indices} "
                f"canonical={matrix.has_canonical_format} "
                f"finite={bool(np.isfinite(data).all())} "
                f"result_format={warm.format} result_rows={warm.shape[0]} "
                f"result_cols={warm.shape[1]} data_shared={data_shared} "
                f"indices_shared={indices_shared} indptr_shared={indptr_shared}",
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
            result = matrix.T
            print(
                f"RESULT rows={result.shape[0]} cols={result.shape[1]} "
                f"nnz={result.nnz} format={result.format} "
                f"sorted={result.has_sorted_indices} "
                f"canonical={result.has_canonical_format} "
                f"data_shared={np.shares_memory(matrix.data, result.data)} "
                f"indices_shared={np.shares_memory(matrix.indices, result.indices)} "
                f"indptr_shared={np.shares_memory(matrix.indptr, result.indptr)}",
                flush=True,
            )
            print(f"OUTPUT_SHA256 {compressed_matrix_sha256(result)}", flush=True)
            continue
        if parts[0] == "SOLVE":
            if len(parts) != 2 or matrix is None:
                print(f"FATAL invalid-solve {line}", flush=True)
                return 2
            repetitions = int(parts[1])
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: sp.csc_matrix | None = None
            maximum_threads = observed_threads()
            started = time.perf_counter()
            for _ in range(repetitions):
                result = matrix.T
            elapsed = time.perf_counter() - started
            maximum_threads = max(maximum_threads, observed_threads())
            assert result is not None
            checksum = float(result.shape[0] + result.shape[1] + result.nnz)
            print(
                f"TIME {elapsed!r} {result.nnz} {maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {parts[0]}", flush=True)
        return 2
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-hstack-csr":
        return profile_sparse_hstack_csr(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-hstack-csr":
        return live_sparse_hstack_csr()
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-lil-skew-to-csr":
        return profile_sparse_lil_skew_to_csr(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-lil-skew-to-csr":
        return live_sparse_lil_skew_to_csr()
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-coo-sub":
        return profile_sparse_coo_sub(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-coo-sub":
        return live_sparse_coo_sub()
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-bsr-to-csr":
        return profile_sparse_bsr_to_csr(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-bsr-to-csr":
        return live_sparse_bsr_to_csr()
    if len(sys.argv) == 2 and sys.argv[1] == "--live-transpose":
        return live_sparse_transpose()
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-csc-add":
        return profile_sparse_csc_add(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-csc-add":
        return live_sparse_csc_add()
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-sparse-laplacian-cycle":
        return profile_sparse_laplacian_cycle(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-laplacian-cycle":
        return live_sparse_laplacian(False)
    if len(sys.argv) == 4 and sys.argv[1] == "--profile-sparse-laplacian":
        return profile_sparse_laplacian(int(sys.argv[2]), int(sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--live-laplacian":
        return live_sparse_laplacian()
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
