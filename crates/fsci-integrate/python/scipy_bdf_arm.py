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
    -> RHSCOST <n> <calls>
    <- TIME <secs>
    -> QUIT

TIMING IS TAKEN HERE, around the `solve_ivp` loop only, so the pipe round-trip is
outside the measured region (trap 5: never measure the client).

FIXTURES include the structured stiff systems and the exact historical explicit-RK
exponential/Lorenz workloads. The Rust arm computes the identical RHS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp


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


def main() -> int:
    # ── TRAP 1: DISPATCH. Prove the incumbent is genuine SciPy and that nothing
    # of ours is loaded in this interpreter. franken_networkx once published 2.6x
    # while genuine NetworkX was 1.88x SLOWER, because its "incumbent" baseline
    # had already been dispatched to fnx.
    fsci_loaded = any(m.startswith(("fsci", "franken")) for m in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
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
        f"genuine={genuine}",
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
