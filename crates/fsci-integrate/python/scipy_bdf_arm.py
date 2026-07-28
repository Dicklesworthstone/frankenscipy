#!/usr/bin/env python3
"""Live SciPy arm for the BDF stiff-ODE head-to-head.

Runs as a PERSISTENT co-process driven by `perf_bdf_vs_scipy`: the Rust side
interleaves its own arm with `SOLVE` commands sent here, so both arms are measured
inside ONE invocation, alternating order, against the same fixture.

Protocol (line oriented, stdout is `-u` unbuffered):

    <- READY scipy=<ver> file=<path> solve_ivp_mod=<mod> fsci_loaded=<bool> ...
    -> SOLVE <n> <t_end> <rtol> <atol> <reps>
    <- TIME <secs> <nfev> <njev> <nlu> <steps> <rhs_calls> <status>
            <success> <comma-separated-final-state>
    -> RHSCOST <n> <calls>
    <- TIME <secs>
    -> QUIT

TIMING IS TAKEN HERE, around the `solve_ivp` loop only, so the pipe round-trip is
outside the measured region (trap 5: never measure the client).

FIXTURE. `y' = -(1 + 10*i) * y_i` — decoupled stiff decay, so the
finite-difference Jacobian is EXACTLY diagonal. This is the shape the structural
claim is about: SciPy has no diagonal path and always `lu_factor`s the dense
system. The Rust arm computes the identical RHS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp


def rates(n: int) -> np.ndarray:
    return 1.0 + 10.0 * np.arange(n, dtype=float)


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
            r = rates(n)
            y0 = 1.0 + 0.25 * (np.arange(n, dtype=float) % 7.0)
            rhs_calls = 0

            def rhs(_t, y):
                nonlocal rhs_calls
                rhs_calls += 1
                return -r * y

            start = time.perf_counter()
            for _ in range(reps):
                sol = solve_ivp(
                    rhs,
                    (0.0, t_end),
                    y0,
                    method="BDF",
                    rtol=rtol,
                    atol=atol,
                    t_eval=None,
                    jac=None,
                )
            elapsed = time.perf_counter() - start
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
            r = rates(n)
            y = 1.0 + 0.25 * (np.arange(n, dtype=float) % 7.0)

            def rhs(_t, yy):
                return -r * yy

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
