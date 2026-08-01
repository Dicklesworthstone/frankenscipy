#!/usr/bin/env python3
"""Persistent live-SciPy arm for the whole-job GMRES source screen.

The Rust parent sends compact commands only. Each timed ``JOB_TIME`` repetition
reconstructs the operator, twelve source fields, the selected preconditioner,
all twelve solutions, and the three scientific summaries per solution.
Interpreter startup, SciPy import, pipe transport, parity serialization, and
backend screening remain outside the timed regions.

Protocol::

    <- READY ...
    -> JOB_CHECK <configuration> <side>
    <- JOB_CHECK <configuration> <successes> <components> <summaries>
                 <input_sha256> <threads> <infos_csv> <iterations_csv>
                 <residuals_csv>
    <- JOB_X <comma-separated f64 values>
    <- JOB_SUMMARIES <comma-separated f64 values>
    -> JOB_TIME <configuration> <side> <repetitions>
    <- JOB_TIME <seconds> <successes> <components> <summaries> <threads>
                <checksum>
    -> JOB_SOLVE_ONLY_TIME <configuration> <side> <repetitions>
    <- JOB_SOLVE_ONLY_TIME <seconds> <successes> <components> <threads>
                           <checksum>
    -> QUIT
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla


DIAGONAL = 4.001
WEST = -1.2
EAST = -0.8
VERTICAL = -1.0
RTOL = 1.0e-5
SOURCE_ROWS = (6, 16, 25)
SOURCE_COLUMNS = (5, 12, 20, 27)
SCENARIOS = len(SOURCE_ROWS) * len(SOURCE_COLUMNS)
CONFIGURATIONS = frozenset(
    {
        "csr-matrix-none",
        "csr-array-none",
        "csc-matrix-none",
        "csc-array-none",
        "csr-matrix-jacobi",
        "csc-matrix-spilu",
    }
)


@dataclass
class JobInputs:
    matrix: sp.spmatrix | sp.sparray
    rhses: np.ndarray
    preconditioner: spla.LinearOperator | None
    canonical: sp.csr_matrix


@dataclass
class JobResult:
    fields: np.ndarray
    summaries: np.ndarray
    infos: list[int]
    iterations: list[int]
    residuals: list[float]
    maximum_threads: int


def observed_threads() -> int:
    return sum(1 for _ in Path("/proc/self/task").iterdir())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_operator(side: int) -> sp.csr_matrix:
    n = side * side
    expected_nnz = 5 * n - 4 * side
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in range(side):
        for column in range(side):
            index = row * side + column
            if row > 0:
                indices.append(index - side)
                data.append(VERTICAL)
            if column > 0:
                indices.append(index - 1)
                data.append(WEST)
            indices.append(index)
            data.append(DIAGONAL)
            if column + 1 < side:
                indices.append(index + 1)
                data.append(EAST)
            if row + 1 < side:
                indices.append(index + side)
                data.append(VERTICAL)
            indptr.append(len(data))
    if len(data) != expected_nnz:
        raise RuntimeError(f"operator nnz {len(data)} != {expected_nnz}")
    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n, n),
        copy=False,
    )


def source_fields(side: int) -> np.ndarray:
    n = side * side
    rows = np.empty((SCENARIOS, n), dtype=np.float64)
    scenario = 0
    for source_row in SOURCE_ROWS:
        for source_column in SOURCE_COLUMNS:
            for row in range(side):
                row_weight = max(0, 4 - abs(row - source_row))
                for column in range(side):
                    column_weight = max(0, 4 - abs(column - source_column))
                    rows[scenario, row * side + column] = (
                        1 + row_weight * column_weight
                    ) / 16.0
            scenario += 1
    return rows


def input_sha256(canonical: sp.csr_matrix, rhses: np.ndarray, side: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray([side, rhses.shape[0]], dtype="<u8").tobytes())
    hasher.update(canonical.data.astype("<f8", copy=False).tobytes(order="C"))
    hasher.update(canonical.indices.astype("<u8", copy=False).tobytes(order="C"))
    hasher.update(canonical.indptr.astype("<u8", copy=False).tobytes(order="C"))
    hasher.update(rhses.astype("<f8", copy=False).tobytes(order="C"))
    return hasher.hexdigest()


def build_job_inputs(configuration: str, side: int) -> JobInputs:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {configuration}")
    canonical = canonical_operator(side)
    rhses = source_fields(side)
    preconditioner: spla.LinearOperator | None = None
    if configuration == "csr-matrix-none":
        matrix = canonical
    elif configuration == "csr-array-none":
        matrix = sp.csr_array(canonical, copy=True)
    elif configuration == "csc-matrix-none":
        matrix = canonical.tocsc(copy=True)
    elif configuration == "csc-array-none":
        matrix = sp.csc_array(canonical, copy=True)
    elif configuration == "csr-matrix-jacobi":
        matrix = canonical
        inverse_diagonal = 1.0 / canonical.diagonal()

        def jacobi(vector: np.ndarray) -> np.ndarray:
            return inverse_diagonal * vector

        preconditioner = spla.LinearOperator(
            canonical.shape,
            matvec=jacobi,
            rmatvec=jacobi,
            dtype=np.float64,
        )
    else:
        matrix = canonical.tocsc(copy=True)
        ilu = spla.spilu(matrix)
        preconditioner = spla.LinearOperator(
            matrix.shape,
            matvec=ilu.solve,
            dtype=np.float64,
        )
    return JobInputs(matrix, rhses, preconditioner, canonical)


def scientific_summaries(
    fields: np.ndarray,
    rhses: np.ndarray,
    side: int,
) -> np.ndarray:
    spacing = 1.0 / float(side + 1)
    cell_area = spacing * spacing
    inventories = fields.sum(axis=1) * cell_area
    outlets = fields[:, side - 1 :: side].sum(axis=1) * spacing
    exposures = np.einsum("ij,ij->i", fields, rhses) * cell_area
    return np.column_stack((inventories, outlets, exposures)).ravel(order="C")


def solve_job_inputs(
    inputs: JobInputs,
    side: int,
    *,
    count_iterations: bool,
    postprocess: bool,
) -> JobResult:
    n = side * side
    maximum_threads = observed_threads()
    solutions: list[np.ndarray] = []
    infos: list[int] = []
    iteration_counts: list[int] = []
    residuals: list[float] = []
    for rhs in inputs.rhses:
        iterations = 0

        def count(_residual: object) -> None:
            nonlocal iterations
            iterations += 1

        kwargs: dict[str, object] = {
            "rtol": RTOL,
            "atol": 0.0,
            "maxiter": 10 * n,
            "M": inputs.preconditioner,
        }
        if count_iterations:
            kwargs["callback"] = count
            kwargs["callback_type"] = "pr_norm"
        solution, info = spla.gmres(inputs.matrix, rhs, **kwargs)
        solutions.append(np.asarray(solution, dtype=np.float64))
        infos.append(int(info))
        iteration_counts.append(iterations)
        residuals.append(
            float(
                np.linalg.norm(rhs - inputs.canonical @ solution)
                / np.linalg.norm(rhs)
            )
        )
        maximum_threads = max(maximum_threads, observed_threads())
    fields = np.stack(solutions, axis=0)
    summaries = (
        scientific_summaries(fields, inputs.rhses, side)
        if postprocess
        else np.empty(0, dtype=np.float64)
    )
    return JobResult(
        fields,
        summaries,
        infos,
        iteration_counts,
        residuals,
        maximum_threads,
    )


def run_whole_job(
    configuration: str,
    side: int,
    *,
    count_iterations: bool,
) -> tuple[JobInputs, JobResult]:
    inputs = build_job_inputs(configuration, side)
    return inputs, solve_job_inputs(
        inputs,
        side,
        count_iterations=count_iterations,
        postprocess=True,
    )


def successful(result: JobResult) -> int:
    return sum(
        info == 0
        and np.isfinite(residual)
        and residual <= 1.25 * RTOL
        and np.all(np.isfinite(field))
        for info, residual, field in zip(
            result.infos,
            result.residuals,
            result.fields,
            strict=True,
        )
    )


def write_vector(label: str, values: np.ndarray) -> None:
    print(
        label + " " + ",".join(format(float(value), ".17e") for value in values),
        flush=True,
    )


def main() -> int:
    if sys.argv != [sys.argv[0], "--live"]:
        print("usage: scipy_gmres_job_arm.py --live", file=sys.stderr)
        return 64

    fsci_loaded = any(name.startswith(("fsci", "franken")) for name in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    gmres_path_text = inspect.getsourcefile(spla.gmres)
    spilu_path_text = inspect.getsourcefile(spla.spilu)
    superlu_module = importlib.import_module("scipy.sparse.linalg._dsolve._superlu")
    superlu_path = Path(superlu_module.__file__).resolve()
    if gmres_path_text is None or spilu_path_text is None:
        print("FATAL scipy-source-unavailable", flush=True)
        return 2
    gmres_path = Path(gmres_path_text).resolve()
    spilu_path = Path(spilu_path_text).resolve()
    installed = any(
        part in {"site-packages", "dist-packages"} for part in scipy_path.parts
    )
    genuine = (
        spla.gmres.__module__.startswith("scipy.sparse.linalg._isolve")
        and spla.spilu.__module__.startswith("scipy.sparse.linalg._dsolve")
        and installed
        and scipy_path.parent in gmres_path.parents
        and scipy_path.parent in spilu_path.parents
        and scipy_path.parent in superlu_path.parents
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"gmres_mod={spla.gmres.__module__} scipy_file={scipy_path} "
        f"gmres_engine_file={gmres_path} "
        f"scipy_engine_sha256={file_sha256(gmres_path)} "
        f"spilu_source_file={spilu_path} "
        f"spilu_source_sha256={file_sha256(spilu_path)} "
        f"superlu_engine_file={superlu_path} "
        f"superlu_engine_sha256={file_sha256(superlu_path)} "
        f"python={Path(sys.executable).resolve()} "
        f"actual_observed_worker_threads={observed_threads()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    for raw_line in sys.stdin:
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        command = parts[0]
        if command == "QUIT":
            break
        if command == "JOB_CHECK":
            if len(parts) != 3:
                print(f"FATAL bad-job-check {line}", flush=True)
                return 2
            configuration, side = parts[1], int(parts[2])
            inputs, result = run_whole_job(
                configuration,
                side,
                count_iterations=True,
            )
            print(
                "JOB_CHECK "
                f"{configuration} {successful(result)} {result.fields.size} "
                f"{result.summaries.size} "
                f"{input_sha256(inputs.canonical, inputs.rhses, side)} "
                f"{result.maximum_threads} "
                f"{','.join(str(value) for value in result.infos)} "
                f"{','.join(str(value) for value in result.iterations)} "
                f"{','.join(format(value, '.17e') for value in result.residuals)}",
                flush=True,
            )
            write_vector("JOB_X", result.fields.ravel(order="C"))
            write_vector("JOB_SUMMARIES", result.summaries)
            continue
        if command == "JOB_TIME":
            if len(parts) != 4:
                print(f"FATAL bad-job-time {line}", flush=True)
                return 2
            configuration, side, repetitions = (
                parts[1],
                int(parts[2]),
                int(parts[3]),
            )
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            result: JobResult | None = None
            started = time.perf_counter()
            for _ in range(repetitions):
                _inputs, result = run_whole_job(
                    configuration,
                    side,
                    count_iterations=False,
                )
            elapsed = time.perf_counter() - started
            assert result is not None
            checksum = float(result.fields.sum() + result.summaries.sum())
            print(
                f"JOB_TIME {elapsed!r} {successful(result)} "
                f"{result.fields.size} {result.summaries.size} "
                f"{result.maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        if command == "JOB_SOLVE_ONLY_TIME":
            if len(parts) != 4:
                print(f"FATAL bad-solve-only-time {line}", flush=True)
                return 2
            configuration, side, repetitions = (
                parts[1],
                int(parts[2]),
                int(parts[3]),
            )
            if repetitions < 1:
                print("FATAL repetitions-must-be-positive", flush=True)
                return 2
            inputs = build_job_inputs(configuration, side)
            result = None
            started = time.perf_counter()
            for _ in range(repetitions):
                result = solve_job_inputs(
                    inputs,
                    side,
                    count_iterations=False,
                    postprocess=False,
                )
            elapsed = time.perf_counter() - started
            assert result is not None
            checksum = float(result.fields.sum())
            print(
                f"JOB_SOLVE_ONLY_TIME {elapsed!r} {successful(result)} "
                f"{result.fields.size} {result.maximum_threads} {checksum!r}",
                flush=True,
            )
            continue
        print(f"FATAL unknown-command {command}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
