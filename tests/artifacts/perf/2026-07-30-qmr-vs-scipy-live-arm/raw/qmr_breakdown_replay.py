#!/usr/bin/env python3
"""Localize the fsci qmr premature-breakdown gate (frankenscipy-9pfja).

Replays SciPy 1.17.1's own un-preconditioned qmr recurrences on the same
convection-diffusion fixture the perf harness builds, logging the four
quantities fsci gates on (rho, xi, delta, epsilon) and reporting the first
iteration at which each would fall below fsci's threshold.

fsci used  f64::EPSILON * 1e6 = 2.220e-10
SciPy uses np.finfo(float).eps = 2.220e-16

Expected output: no trip at side 48 (both arms converge at 108), a trip on
epsilon at iteration 121 for side 64, and on delta at 151 for side 96 --
matching the iteration at which fsci actually bailed (121 and 152).
"""

import numpy as np
import scipy.sparse as sp

FSCI_OLD_TOL = np.finfo(float).eps * 1e6
SCIPY_TOL = np.finfo(float).eps


def convection_diffusion_2d(side):
    """Same five-point stencil as perf_sparse_vs_scipy.rs."""
    n = side * side
    rows, cols, vals = [], [], []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            rows.append(i); cols.append(i); vals.append(4.001)
            if c > 0:
                rows.append(i); cols.append(i - 1); vals.append(-1.2)
            if c < side - 1:
                rows.append(i); cols.append(i + 1); vals.append(-0.8)
            if r > 0:
                rows.append(i); cols.append(i - side); vals.append(-1.0)
            if r < side - 1:
                rows.append(i); cols.append(i + side); vals.append(-1.0)
    return sp.csr_array((vals, (rows, cols)), shape=(n, n))


def replay(side, rtol=1e-5, max_iter=4000):
    A = convection_diffusion_2d(side)
    n = side * side
    b = np.array([1.0 + 0.01 * (i % 17) for i in range(n)])
    x = np.zeros(n)
    r = b.copy()
    vt = r.copy(); y = vt; rho = np.linalg.norm(y)
    wt = r.copy(); z = wt; xi = np.linalg.norm(z)
    gamma, eta, theta = 1.0, -1.0, 0.0
    v = np.empty_like(vt); w = np.empty_like(wt)
    eps_ = q = d = p = s = None
    atol = rtol * np.linalg.norm(b)
    mins = {"rho": np.inf, "xi": np.inf, "delta": np.inf, "epsilon": np.inf}
    first_trip = None
    it = 0
    for it in range(max_iter):
        if np.linalg.norm(r) < atol:
            break
        v[:] = vt[:]; v *= (1 / rho); y = y * (1 / rho)
        w[:] = wt[:]; w *= (1 / xi); z = z * (1 / xi)
        delta = np.dot(z, y)
        yt, zt = y.copy(), z.copy()
        if it > 0:
            yt -= (xi * delta / eps_) * p; p[:] = yt[:]
            zt -= (rho * (delta / eps_)) * q; q[:] = zt[:]
        else:
            p, q = yt.copy(), zt.copy()
        pt = A @ p
        eps_ = np.dot(q, pt)
        beta = eps_ / delta
        for name, val in (("rho", rho), ("xi", xi),
                          ("delta", delta), ("epsilon", eps_)):
            mins[name] = min(mins[name], abs(val))
            if first_trip is None and abs(val) < FSCI_OLD_TOL:
                first_trip = (it, name, abs(val))
        vt[:] = pt[:]; vt -= beta * v; y = vt.copy()
        rho_prev, rho = rho, np.linalg.norm(y)
        wt[:] = w[:]; wt *= -beta; wt += A.T @ q; z = wt.copy()
        xi = np.linalg.norm(z)
        gp, tp = gamma, theta
        theta = rho / (gp * abs(beta))
        gamma = 1 / np.sqrt(1 + theta ** 2)
        eta *= -(rho_prev / beta) * (gamma / gp) ** 2
        if it > 0:
            d *= (tp * gamma) ** 2; d += eta * p
            s *= (tp * gamma) ** 2; s += eta * pt
        else:
            d = p.copy() * eta
            s = pt.copy() * eta
        x += d
        r -= s
    return it, mins, first_trip


def main():
    print(f"fsci OLD breakdown tol = eps*1e6 = {FSCI_OLD_TOL:.3e}")
    print(f"SciPy breakdown tol    = eps     = {SCIPY_TOL:.3e}\n")
    for side in (48, 64, 96):
        it, mins, trip = replay(side)
        print(f"side={side} n={side * side} scipy_converges_at={it}")
        print(f"   min|rho|={mins['rho']:.3e} min|xi|={mins['xi']:.3e} "
              f"min|delta|={mins['delta']:.3e} min|epsilon|={mins['epsilon']:.3e}")
        print(f"   first trip under the OLD fsci tol: {trip}\n")


if __name__ == "__main__":
    main()
