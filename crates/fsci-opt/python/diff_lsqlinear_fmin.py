"""SciPy comparator for diff_lsqlinear_fmin.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.optimize.lsq_linear, fmin_bfgs, fmin_powell and fmin_cg -- entry points with no
differential coverage (frankenscipy-ivxx6).

`quadratic_assignment` is deliberately excluded: SciPy's default `faq` method is a randomised
heuristic, so comparing permutations would compare two draws rather than two answers.
"""
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_powell, lsq_linear

A = np.array([
    [1.0, 0.5, -0.25],
    [0.0, 2.0, 1.0],
    [3.0, -1.0, 0.5],
    [-1.0, 1.5, 2.0],
    [0.5, 0.5, -3.0],
])
B = np.array([1.0, -2.0, 3.0, 0.5, -1.5])


def quad(x):
    return (x[0] - 1.0) ** 2 + 2.0 * (x[1] + 2.0) ** 2 + 3.0 * (x[2] - 0.5) ** 2


def rosen(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    print(f"{name}|" + ";".join(f"{v:.17e}" for v in flat))


def main():
    dump("lsq_unbounded",
         lsq_linear(A, B, bounds=(np.full(3, -1.0e6), np.full(3, 1.0e6))).x)
    dump("lsq_bounded",
         lsq_linear(A, B, bounds=(np.array([-0.2, -0.5, 0.0]),
                                  np.array([0.5, 0.4, 1.0]))).x)

    q0 = np.zeros(3)
    r0 = np.array([-1.2, 1.0])
    dump("bfgs_quad", fmin_bfgs(quad, q0, disp=False))
    dump("powell_quad", fmin_powell(quad, q0, disp=False))
    dump("cg_quad", fmin_cg(quad, q0, disp=False))
    dump("bfgs_rosen", fmin_bfgs(rosen, r0, disp=False))
    dump("powell_rosen", fmin_powell(rosen, r0, disp=False))


if __name__ == "__main__":
    main()
