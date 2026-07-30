#!/usr/bin/env python3
"""Live SciPy arm for the ODE head-to-head.

Runs as a PERSISTENT co-process driven by `perf_bdf_vs_scipy`: the Rust side
interleaves its own arm with `SOLVE` commands sent here, so both arms are measured
inside ONE invocation, alternating order, against the same fixture.

Protocol (line oriented, stdout is `-u` unbuffered):

    <- READY scipy=<ver> file=<path> solve_ivp_mod=<mod> fsci_loaded=<bool> ...
    -> SOLVE <n> <t_end> <rtol> <atol> <reps> <fixture> <method>
    <- TIME <secs> <nfev> <njev> <nlu> <steps> <rhs_calls> <status>
            <success> <comma-separated-final-state>
    -> MANY_CHECK <batch> <sampled|final>
    <- CHECK <successes> <nfev> <njev> <nlu> <stored_points> <rhs_calls>
             <compared_samples> <input_sha256> <min_component>
             <max_invariant_drift> <comma-separated-values>
    -> MANY_TIME <batch> <reps> <sampled|final>
    <- TIME <secs> <successes>
    -> DECAY_CHECK <n> <scenarios> <workers> <jacobian-mode>
    <- JOB_CHECK <successes> <nfev> <njev> <nlu> <stored_points> <rhs_calls>
            <scenarios> <samples> <input_sha256> <worker_processes>
            <worker_threads> <peak_rss_kib> <rhs-calls-list> <values>
            <exposures> <terminal-masses>
    -> DECAY_TIME <n> <scenarios> <workers> <reps> <jacobian-mode>
    <- JOB_TIME <secs> <successes> <worker_processes> <worker_threads>
            <peak_rss_kib>
    -> DECAY_RHSCOST <n> <workers> <comma-separated-calls>
    <- JOB_RHS_TIME <secs>
    -> RHSCOST <n> <calls>
    <- TIME <secs>
    -> QUIT

For ordinary fixtures, timing is around the `solve_ivp` loop only. For
`DECAY_TIME`, timing covers the complete screening job: deterministic input/model
construction, process-pool lifecycle, all solves, output materialization, and
scientific postprocessing. Pipe transport remains outside every measured region.

FIXTURES include the structured stiff systems and the exact historical explicit-RK
exponential/Lorenz workloads. The Rust arm computes the identical RHS.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.integrate import solve_ivp

LOTKA_T_END = 10.0
LOTKA_RTOL = 1e-8
LOTKA_ATOL = 1e-10
LOTKA_SAMPLES = 150
DECAY_SAMPLES = 65
DECAY_QUADRATIC = 0.125

_DECAY_RATES: np.ndarray | None = None
_DECAY_T_EVAL: np.ndarray | None = None
_DECAY_JACOBIAN_MODE = "none"
_DECAY_SPARSITY = None


def rates(n: int, fixture: str) -> np.ndarray:
    if fixture == "exponential":
        return np.ones(n, dtype=float)
    if fixture == "lorenz":
        return np.zeros(n, dtype=float)
    if fixture == "radau-stiff":
        denom = float(max(n - 1, 1))
        return 1.0 + 999.0 * (np.arange(n, dtype=float) / denom)
    return 1.0 + 10.0 * np.arange(n, dtype=float)


def initial_state(n: int, fixture: str) -> np.ndarray:
    if fixture in {"exponential", "lorenz", "radau-stiff"}:
        return np.ones(n, dtype=float)
    return 1.0 + 0.25 * (np.arange(n, dtype=float) % 7.0)


def make_rhs(fixture: str, r: np.ndarray):
    """RHS for the requested fixture. MUST match `rhs_into` in the Rust arm exactly,
    or the two arms solve different problems and the trap-2 agreement check aborts.

    `exponential`: scalar y'=-y, the historical explicit-RK micro-ODE.
    `lorenz`     : the historical three-component Lorenz explicit-RK workload.
    `diagonal`   : y'_i = -(1 + 10i) y_i — decoupled, Jacobian exactly diagonal.
    `coupled`    : adds nearest-neighbour coupling, so the Jacobian is TRIDIAGONAL
                   and our structural diagonal fast path cannot fire.
    """
    if fixture == "exponential":
        def rhs(_t, y):
            return np.array([-y[0]], dtype=float)
        return rhs
    if fixture == "lorenz":
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

        def rhs(_t, y):
            return np.array(
                [
                    sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2],
                ],
                dtype=float,
            )
        return rhs
    if fixture in {"diagonal", "radau-stiff"}:
        def rhs(_t, y):
            return -r * y
        return rhs
    if fixture == "dense":
        # J_ij = 1e-3/n for all i,j — structurally dense, but the RHS stays O(n) so
        # the callback cost does not change character between fixtures.
        inv_n = 1.0 / float(r.size)

        def rhs(_t, y):
            return -r * y + (1e-3 * inv_n) * float(y.sum())

        return rhs
    if fixture == "coupled":
        def rhs(_t, y):
            out = -r * y
            out[:-1] += 0.5 * y[1:]
            out[1:] += 0.5 * y[:-1]
            out -= y
            return out
        return rhs
    raise SystemExit(f"unknown fixture: {fixture}")


def lotka_initial_states(batch: int) -> np.ndarray:
    """Exact u64 LCG used by the Rust arm and the shipped batch-conformance test."""
    state = 99
    rows = np.empty((batch, 2), dtype=np.float64)
    mask = (1 << 64) - 1
    scale = float(1 << 53)
    for row in range(batch):
        for component in range(2):
            state = (
                state * 6364136223846793005 + 1
            ) & mask
            rows[row, component] = 1.0 + 4.0 * (float(state >> 11) / scale)
    return rows


def lotka_t_eval() -> np.ndarray:
    return np.arange(LOTKA_SAMPLES, dtype=np.float64) * LOTKA_T_END / float(
        LOTKA_SAMPLES - 1
    )


def lotka_rhs(_t, y):
    a, b, c, d = 1.5, 1.0, 3.0, 1.0
    return np.array(
        [
            a * y[0] - b * y[0] * y[1],
            -c * y[1] + d * y[0] * y[1],
        ],
        dtype=float,
    )


def solve_lotka(y0: np.ndarray, rhs=lotka_rhs, *, sampled: bool = True):
    return solve_ivp(
        rhs,
        (0.0, LOTKA_T_END),
        y0,
        method="RK45",
        rtol=LOTKA_RTOL,
        atol=LOTKA_ATOL,
        t_eval=lotka_t_eval() if sampled else None,
    )


def lotka_invariant(y: np.ndarray) -> np.ndarray:
    return y[0] - 3.0 * np.log(y[0]) + y[1] - 1.5 * np.log(y[1])


def decay_rates(n: int) -> np.ndarray:
    return 1.0 + 10.0 * np.arange(n, dtype=np.float64)


def decay_t_eval() -> np.ndarray:
    return np.arange(DECAY_SAMPLES, dtype=np.float64) / float(DECAY_SAMPLES - 1)


def decay_initial_states(n: int, scenarios: int) -> np.ndarray:
    base = 1.0 + 0.25 * (np.arange(n, dtype=np.float64) % 7.0)
    rows = np.empty((scenarios, n), dtype=np.float64)
    for scenario in range(scenarios):
        rows[scenario] = base * (1.0 + float(scenario) / 32.0)
    return rows


def decay_rhs(_t, y):
    if _DECAY_RATES is None:
        raise RuntimeError("decay worker was not initialized")
    return -_DECAY_RATES * y * (1.0 + DECAY_QUADRATIC * y)


def decay_jacobian(_t, y):
    if _DECAY_RATES is None:
        raise RuntimeError("decay worker was not initialized")
    diagonal = -_DECAY_RATES * (1.0 + 2.0 * DECAY_QUADRATIC * y)
    return sparse.diags(diagonal, offsets=0, format="csc")


def init_decay_worker(
    worker_rates: np.ndarray,
    worker_t_eval: np.ndarray,
    jacobian_mode: str,
) -> None:
    global _DECAY_RATES, _DECAY_T_EVAL, _DECAY_JACOBIAN_MODE, _DECAY_SPARSITY
    _DECAY_RATES = worker_rates
    _DECAY_T_EVAL = worker_t_eval
    _DECAY_JACOBIAN_MODE = jacobian_mode
    _DECAY_SPARSITY = sparse.eye(worker_rates.size, format="csc")


def solve_decay_worker(task):
    scenario, y0, count_rhs = task
    rhs_calls = 0

    if count_rhs:
        def rhs(t, y):
            nonlocal rhs_calls
            rhs_calls += 1
            return decay_rhs(t, y)
    else:
        rhs = decay_rhs

    kwargs = {}
    if _DECAY_JACOBIAN_MODE == "analytic-sparse":
        kwargs["jac"] = decay_jacobian
    elif _DECAY_JACOBIAN_MODE == "sparsity-only":
        kwargs["jac_sparsity"] = _DECAY_SPARSITY
    elif _DECAY_JACOBIAN_MODE != "none":
        raise RuntimeError(f"unknown decay Jacobian mode: {_DECAY_JACOBIAN_MODE}")

    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        y0,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
        t_eval=_DECAY_T_EVAL,
        **kwargs,
    )
    exposure = np.trapezoid(sol.y, x=sol.t, axis=1)
    terminal_mass = float(sol.y[:, -1].sum())
    return (
        scenario,
        bool(sol.success),
        int(sol.status),
        int(sol.nfev),
        int(sol.njev),
        int(sol.nlu),
        int(sol.t.size),
        rhs_calls,
        os.getpid(),
        observed_os_threads(),
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        sol.y.T.ravel(order="C"),
        exposure,
        terminal_mass,
    )


def run_decay_job(
    n: int,
    scenarios: int,
    workers: int,
    jacobian_mode: str,
    *,
    count_rhs: bool,
):
    worker_rates = decay_rates(n)
    worker_t_eval = decay_t_eval()
    rows = decay_initial_states(n, scenarios)
    context = mp.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=init_decay_worker,
        initargs=(worker_rates, worker_t_eval, jacobian_mode),
    ) as pool:
        results = pool.map(
            solve_decay_worker,
            [
                (scenario, rows[scenario], count_rhs)
                for scenario in range(scenarios)
            ],
            chunksize=1,
        )
    return worker_rates, worker_t_eval, rows, results


def decay_process_metrics(results) -> tuple[int, int, int]:
    process_metrics = {}
    for result in results:
        pid, threads, peak_rss_kib = result[8:11]
        old_threads, old_peak = process_metrics.get(pid, (0, 0))
        process_metrics[pid] = (
            max(old_threads, threads),
            max(old_peak, peak_rss_kib),
        )
    worker_threads = sum(threads for threads, _peak in process_metrics.values())
    process_tree_peak_upper_bound = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ) + sum(peak for _threads, peak in process_metrics.values())
    return len(process_metrics), worker_threads, process_tree_peak_upper_bound


def decay_input_sha256(
    worker_rates: np.ndarray,
    worker_t_eval: np.ndarray,
    rows: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray([DECAY_QUADRATIC], dtype="<f8").tobytes())
    hasher.update(worker_rates.astype("<f8", copy=False).tobytes(order="C"))
    hasher.update(worker_t_eval.astype("<f8", copy=False).tobytes(order="C"))
    hasher.update(rows.astype("<f8", copy=False).tobytes(order="C"))
    return hasher.hexdigest()


def replay_decay_rhs_worker(task):
    scenario, calls = task
    if _DECAY_RATES is None:
        raise RuntimeError("decay worker was not initialized")
    y = decay_initial_states(_DECAY_RATES.size, scenario + 1)[-1]
    checksum = 0.0
    for _ in range(calls):
        values = decay_rhs(0.0, y)
        checksum += float(values[scenario % values.size])
    return checksum


def run_decay_rhs_replay(n: int, workers: int, calls: list[int]) -> float:
    worker_rates = decay_rates(n)
    worker_t_eval = decay_t_eval()
    context = mp.get_context("fork")
    start = time.perf_counter()
    with context.Pool(
        processes=workers,
        initializer=init_decay_worker,
        initargs=(worker_rates, worker_t_eval, "none"),
    ) as pool:
        checksums = pool.map(
            replay_decay_rhs_worker,
            list(enumerate(calls)),
            chunksize=1,
        )
    elapsed = time.perf_counter() - start
    if not all(np.isfinite(checksum) for checksum in checksums):
        raise RuntimeError("non-finite decay RHS replay checksum")
    return elapsed


def observed_os_threads() -> int:
    """Return native process threads, including pools invisible to `threading`."""
    task_dir = Path("/proc/self/task")
    if task_dir.is_dir():
        return sum(1 for _entry in task_dir.iterdir())
    return threading.active_count()


def main() -> int:
    # ── TRAP 1: DISPATCH. Prove the incumbent is genuine SciPy and that nothing
    # of ours is loaded in this interpreter. franken_networkx once published 2.6x
    # while genuine NetworkX was 1.88x SLOWER, because its "incumbent" baseline
    # had already been dispatched to fnx.
    fsci_loaded = any(m.startswith(("fsci", "franken")) for m in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    scipy_engine_path = Path(sys.modules[solve_ivp.__module__].__file__).resolve()
    scipy_engine_sha256 = hashlib.sha256(scipy_engine_path.read_bytes()).hexdigest()
    actual_observed_worker_threads = observed_os_threads()
    installed_path = any(
        component in {"site-packages", "dist-packages"}
        for component in scipy_path.parts
    )
    genuine = (
        solve_ivp.__module__ == "scipy.integrate._ivp.ivp"
        and installed_path
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"file={scipy_path} solve_ivp_mod={solve_ivp.__module__} "
        f"python={Path(sys.executable).resolve()} fsci_loaded={fsci_loaded} "
        f"genuine={genuine} python_threads={threading.active_count()} "
        f"actual_observed_worker_threads={actual_observed_worker_threads} "
        f"scipy_engine_path={scipy_engine_path} "
        f"scipy_engine_sha256={scipy_engine_sha256} "
        "blas_thread_cap=1",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == "QUIT":
            break
        if parts[0] == "SOLVE":
            n, t_end, rtol, atol, reps = (
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                int(parts[5]),
            )
            fixture = parts[6] if len(parts) > 6 else "diagonal"
            method = parts[7] if len(parts) > 7 else "BDF"
            r = rates(n, fixture)
            y0 = initial_state(n, fixture)
            base_rhs = make_rhs(fixture, r)

            start = time.perf_counter()
            for _ in range(reps):
                sol = solve_ivp(
                    base_rhs,
                    (0.0, t_end),
                    y0,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    t_eval=None,
                )
            elapsed = time.perf_counter() - start

            # Count finite-difference/Jacobian callback traffic in a separate solve.
            # The counter and its extra Python dispatch must not inflate the timed
            # incumbent, especially for cheap explicit-RK right-hand sides.
            rhs_calls = 0

            def counted_rhs(_t, y):
                nonlocal rhs_calls
                rhs_calls += 1
                return base_rhs(_t, y)

            counted_sol = solve_ivp(
                counted_rhs,
                (0.0, t_end),
                y0,
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=None,
            )
            if (
                int(counted_sol.status) != int(sol.status)
                or bool(counted_sol.success) != bool(sol.success)
                or not np.array_equal(counted_sol.y[:, -1], sol.y[:, -1])
            ):
                print("FATAL counted-solve-diverged", flush=True)
                return 2
            nfev, njev, nlu = int(sol.nfev), int(sol.njev), int(sol.nlu)
            steps = int(sol.t.size)
            final_values = ",".join(repr(float(value)) for value in sol.y[:, -1])
            print(
                f"TIME {elapsed!r} {nfev} {njev} {nlu} {steps} {rhs_calls} "
                f"{int(sol.status)} {sol.success} {final_values}",
                flush=True,
            )
        elif parts[0] == "MANY_CHECK":
            batch = int(parts[1])
            sampled = len(parts) < 3 or parts[2] == "sampled"
            rows = lotka_initial_states(batch)
            input_sha256 = hashlib.sha256(
                rows.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest()
            solutions = []
            total_rhs_calls = 0
            for y0 in rows:
                rhs_calls = 0

                def counted_rhs(t, y):
                    nonlocal rhs_calls
                    rhs_calls += 1
                    return lotka_rhs(t, y)

                sol = solve_lotka(y0, counted_rhs, sampled=sampled)
                total_rhs_calls += rhs_calls
                solutions.append(sol)
            if any(
                not sol.success
                or int(sol.status) != 0
                or sol.t.size == 0
                or sol.y.shape != (2, sol.t.size)
                or sol.t[-1] != LOTKA_T_END
                or (sampled and sol.t.size != LOTKA_SAMPLES)
                for sol in solutions
            ):
                print("FATAL many-check-incomplete", flush=True)
                return 2
            min_component = min(float(sol.y.min()) for sol in solutions)
            max_invariant_drift = max(
                float(
                    np.max(
                        np.abs(
                            lotka_invariant(sol.y)
                            - float(lotka_invariant(y0.reshape(2, 1))[0])
                        )
                    )
                )
                for y0, sol in zip(rows, solutions, strict=True)
            )
            compared_samples = LOTKA_SAMPLES if sampled else 1
            flattened = ",".join(
                repr(float(value))
                for sol in solutions
                for value in (
                    sol.y.T.ravel(order="C") if sampled else sol.y[:, -1]
                )
            )
            print(
                "CHECK "
                f"{len(solutions)} "
                f"{sum(int(sol.nfev) for sol in solutions)} "
                f"{sum(int(sol.njev) for sol in solutions)} "
                f"{sum(int(sol.nlu) for sol in solutions)} "
                f"{sum(int(sol.t.size) for sol in solutions)} "
                f"{total_rhs_calls} {compared_samples} {input_sha256} "
                f"{min_component!r} {max_invariant_drift!r} {flattened}",
                flush=True,
            )
        elif parts[0] == "MANY_TIME":
            batch, reps = int(parts[1]), int(parts[2])
            sampled = len(parts) < 4 or parts[3] == "sampled"
            rows = lotka_initial_states(batch)
            solutions = []
            start = time.perf_counter()
            for _ in range(reps):
                solutions = [solve_lotka(y0, sampled=sampled) for y0 in rows]
            elapsed = time.perf_counter() - start
            successes = sum(
                sol.success
                and int(sol.status) == 0
                and sol.t.size > 0
                and sol.y.shape == (2, sol.t.size)
                and sol.t[-1] == LOTKA_T_END
                and (not sampled or sol.t.size == LOTKA_SAMPLES)
                for sol in solutions
            )
            print(f"TIME {elapsed!r} {successes}", flush=True)
        elif parts[0] == "DECAY_CHECK":
            n, scenarios, workers = int(parts[1]), int(parts[2]), int(parts[3])
            jacobian_mode = parts[4]
            worker_rates, worker_t_eval, rows, results = run_decay_job(
                n,
                scenarios,
                workers,
                jacobian_mode,
                count_rhs=True,
            )
            if [result[0] for result in results] != list(range(scenarios)):
                print("FATAL decay-check-order", flush=True)
                return 2
            successes = sum(
                result[1]
                and result[2] == 0
                and result[6] == DECAY_SAMPLES
                and result[11].size == DECAY_SAMPLES * n
                and result[12].size == n
                and np.all(np.isfinite(result[11]))
                and np.all(np.isfinite(result[12]))
                and np.isfinite(result[13])
                for result in results
            )
            worker_processes, worker_threads, peak_rss_kib = (
                decay_process_metrics(results)
            )
            input_sha256 = decay_input_sha256(
                worker_rates,
                worker_t_eval,
                rows,
            )
            rhs_calls = [result[7] for result in results]
            flattened_values = ",".join(
                repr(float(value))
                for result in results
                for value in result[11]
            )
            flattened_exposures = ",".join(
                repr(float(value))
                for result in results
                for value in result[12]
            )
            terminal_masses = ",".join(
                repr(float(result[13])) for result in results
            )
            print(
                "JOB_CHECK "
                f"{successes} "
                f"{sum(result[3] for result in results)} "
                f"{sum(result[4] for result in results)} "
                f"{sum(result[5] for result in results)} "
                f"{sum(result[6] for result in results)} "
                f"{sum(rhs_calls)} {scenarios} {DECAY_SAMPLES} "
                f"{input_sha256} {worker_processes} {worker_threads} "
                f"{peak_rss_kib} "
                f"{','.join(str(value) for value in rhs_calls)} "
                f"{flattened_values} {flattened_exposures} {terminal_masses}",
                flush=True,
            )
        elif parts[0] == "DECAY_TIME":
            n, scenarios, workers, reps = (
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
            )
            jacobian_mode = parts[5]
            results = []
            start = time.perf_counter()
            for _ in range(reps):
                _rates, _t_eval, _rows, results = run_decay_job(
                    n,
                    scenarios,
                    workers,
                    jacobian_mode,
                    count_rhs=False,
                )
            elapsed = time.perf_counter() - start
            successes = sum(
                result[1]
                and result[2] == 0
                and result[6] == DECAY_SAMPLES
                and result[11].size == DECAY_SAMPLES * n
                and result[12].size == n
                and np.all(np.isfinite(result[11]))
                and np.all(np.isfinite(result[12]))
                and np.isfinite(result[13])
                for result in results
            )
            worker_processes, worker_threads, peak_rss_kib = (
                decay_process_metrics(results)
            )
            print(
                f"JOB_TIME {elapsed!r} {successes} {worker_processes} "
                f"{worker_threads} {peak_rss_kib}",
                flush=True,
            )
        elif parts[0] == "DECAY_RHSCOST":
            n, workers = int(parts[1]), int(parts[2])
            calls = [int(value) for value in parts[3].split(",")]
            elapsed = run_decay_rhs_replay(n, workers, calls)
            print(f"JOB_RHS_TIME {elapsed!r}", flush=True)
        elif parts[0] == "MANY_RHSCOST":
            calls = int(parts[1])
            y = lotka_initial_states(1)[0]
            lotka_rhs(0.0, y)
            start = time.perf_counter()
            for _ in range(calls):
                lotka_rhs(0.0, y)
            print(f"TIME {time.perf_counter() - start!r}", flush=True)
        elif parts[0] == "RHSCOST":
            # ── TRAP 6: SHARED/ASYMMETRIC COMPONENT. SciPy's RHS is a Python
            # callback; ours is an inlined Rust closure. A stiff solve makes
            # thousands of RHS calls, so an undecomposed end-to-end ratio would
            # be substantially callback overhead attributed to "solver quality".
            # This measures the callback alone so the ratio can be split.
            n, calls = int(parts[1]), int(parts[2])
            fixture = parts[3] if len(parts) > 3 else "diagonal"
            r = rates(n, fixture)
            y = initial_state(n, fixture)
            rhs = make_rhs(fixture, r)

            rhs(0.0, y)
            start = time.perf_counter()
            for _ in range(calls):
                rhs(0.0, y)
            print(f"TIME {time.perf_counter() - start!r}", flush=True)
        else:
            print(f"FATAL unknown-command {parts[0]}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
