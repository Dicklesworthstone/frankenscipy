#!/usr/bin/env python3
"""Live SciPy arm for the sparse iterative-solver head-to-head.

Same co-process contract as the ODE arm: the Rust driver interleaves its own arm
with `SOLVE` commands sent here, so both arms are measured inside ONE invocation
against the same matrix, and timing is taken HERE around the solver call only.

    <- READY scipy=<ver> ... genuine=<bool>
    -> SOLVE <side> <tol> <maxiter> <reps> <method>
    <- TIME <secs> <iters> <converged> <resid> <xsum> <xfirst>
    -> QUIT

FIXTURE. 2-D 5-point Laplacian on a `side x side` grid — the standard SPD test
problem for CG, and the same shape the 2026-07-23 measurement row used. Built here
with `scipy.sparse` so the incumbent gets its native CSR and its C-level SpMV; there
is deliberately NO Python matvec callback, so unlike the ODE arm there is no
callback asymmetry to decompose (trap 6 does not arise — the work is all compiled on
both sides).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian(side: int):
    """2-D 5-point Laplacian, CSR, SPD. Identical assembly to the Rust arm."""
    n = side * side
    main = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    off[np.arange(1, n) % side == 0] = 0.0  # no wrap across row boundaries
    far = -1.0 * np.ones(n - side)
    return sp.diags(
        [far, off, main, off, far],
        [-side, -1, 0, 1, side],
        format="csr",
        dtype=float,
    )


def rhs(n: int) -> np.ndarray:
    return 1.0 + 0.25 * (np.arange(n, dtype=float) % 7.0)


def main() -> int:
    fsci_loaded = any(m.startswith(("fsci", "franken")) for m in sys.modules)
    scipy_path = Path(scipy.__file__).resolve()
    installed = any(p in {"site-packages", "dist-packages"} for p in scipy_path.parts)
    genuine = (
        spla.cg.__module__.startswith("scipy.sparse.linalg")
        and installed
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} file={scipy_path} "
        f"cg_mod={spla.cg.__module__} python={Path(sys.executable).resolve()} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy", flush=True)
        return 2

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == "QUIT":
            break
        if parts[0] != "SOLVE":
            print(f"FATAL unknown-command {parts[0]}", flush=True)
            return 2
        side, tol, maxiter, reps = (
            int(parts[1]),
            float(parts[2]),
            int(parts[3]),
            int(parts[4]),
        )
        method = parts[5] if len(parts) > 5 else "cg"
        a = laplacian(side)
        b = rhs(a.shape[0])

        iters = 0

        def count(_xk):
            nonlocal iters
            iters += 1

        solver = {"cg": spla.cg, "bicgstab": spla.bicgstab, "gmres": spla.gmres}[method]
        # Warm-up outside the timed region: first-call setup is not the claim.
        solver(a, b, rtol=tol, atol=0.0, maxiter=maxiter)
        start = time.perf_counter()
        for _ in range(reps):
            x, info = solver(a, b, rtol=tol, atol=0.0, maxiter=maxiter)
        elapsed = time.perf_counter() - start
        iters = 0
        solver(a, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=count)
        resid = float(np.linalg.norm(b - a @ x) / np.linalg.norm(b))
        print(
            f"TIME {elapsed!r} {iters} {int(info) == 0} {resid!r} "
            f"{float(x.sum())!r} {float(x[0])!r}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
