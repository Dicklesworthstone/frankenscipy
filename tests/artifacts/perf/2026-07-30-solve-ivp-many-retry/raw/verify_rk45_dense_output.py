#!/usr/bin/env python3
"""Validate the RK45 dense-output arithmetic implemented in
`RkSolver::dense_output_at` against SciPy's own `RkDenseOutput`, and quantify
how far the previous generic cubic Hermite was from it (frankenscipy-3m5ip).

Deliberately independent of the Rust build: it re-implements the *exact*
arithmetic written in Rust (same Horner order, same P constants) in Python and
feeds it SciPy's own K / h / y_old from real RK45 steps. If this agrees to
machine precision, the constants and the formula are right, and any remaining
disagreement in the live arm is step-sequence divergence rather than the
interpolant.

Run: python3 verify_rk45_dense_output.py
"""

import numpy as np
from scipy.integrate._ivp.rk import RK45

P = np.array(RK45.P)

# The exact deterministic trajectory the live arm reported as failing:
# max 711.439 tolerance units at t=9.2617449664.
FAILING_Y0 = [1.6202043337900727, 3.8326403684009982]


def lotka_volterra(_t, y):
    return [1.5 * y[0] - 1.0 * y[0] * y[1], -3.0 * y[1] + 1.0 * y[0] * y[1]]


def rust_horner(K, y_old, h, x):
    """Mirror of RkSolver::dense_output_at, including summation order."""
    out = list(y_old)
    for i in range(len(y_old)):
        acc = 0.0
        for j in (3, 2, 1, 0):
            q_ij = sum(K[s][i] * P[s][j] for s in range(7))
            acc = acc * x + q_ij
        out[i] += h * acc * x
    return out


def cubic_hermite(y0, y1, f0, f1, h, x):
    """The interpolant fsci used before this fix."""
    x2, x3 = x * x, x * x * x
    h00, h10 = 2 * x3 - 3 * x2 + 1, x3 - 2 * x2 + x
    h01, h11 = -2 * x3 + 3 * x2, x3 - x2
    return [h00 * a + h10 * h * c + h01 * b + h11 * h * d
            for a, b, c, d in zip(y0, y1, f0, f1)]


def main():
    solver = RK45(lotka_volterra, 0.0, FAILING_Y0, 20.0, rtol=1e-8, atol=1e-10)
    worst_rust = worst_herm = 0.0
    checked = 0
    while solver.status == "running":
        solver.step()
        dense = solver.dense_output()
        K, h, t_old, y_old = solver.K, solver.h_previous, solver.t_old, solver.y_old
        for x in (0.1, 0.25, 0.5, 0.6180339887, 0.75, 0.9):
            ref = dense(t_old + x * h)
            scale = np.maximum(np.abs(ref), 1e-10)
            rust = np.array(rust_horner(K, y_old, h, x))
            herm = np.array(cubic_hermite(y_old, solver.y, K[0], solver.f, h, x))
            worst_rust = max(worst_rust, float(np.max(np.abs(rust - ref) / scale)))
            worst_herm = max(worst_herm, float(np.max(np.abs(herm - ref) / scale)))
            checked += 1

    print(f"samples checked: {checked}")
    print(f"  rust Horner form vs scipy dense_output : max rel err {worst_rust:.3e}")
    print(f"  cubic Hermite    vs scipy dense_output : max rel err {worst_herm:.3e}")
    print(f"  cubic Hermite is {worst_herm / max(worst_rust, 1e-18):.3g}x worse")
    assert worst_rust < 1e-12, "dense-output arithmetic does not match SciPy"
    print("\nPASS: the implemented arithmetic reproduces SciPy's dense output.")


if __name__ == "__main__":
    main()
