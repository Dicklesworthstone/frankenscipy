"""SciPy comparator for diff_ndinterp_scipy.

Emits the same `name,i,0,value` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.interpolate.LinearNDInterpolator and CloughTocher2DInterpolator, the only
conformance-uncovered entry points that a harness with a LIVE SciPy arm already exercises
(frankenscipy-ivxx6).
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

SITES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.31830988618379069, 0.15915494309189535],
    [0.69314718055994531, 0.43429448190325176],
    [0.57721566490153286, 0.86602540378443865],
    [0.13533528323661270, 0.60653065971263342],
    [0.86602540378443865, 0.20787957635076193],
    [0.41421356237309515, 0.73205080756887729],
    [0.22360679774997896, 0.33166247903553998],
    [0.78539816339744828, 0.61803398874989479],
])

QUERIES = np.array([
    [0.25, 0.25],
    [0.5, 0.5],
    [0.75, 0.25],
    [0.4, 0.8],
    [0.6, 0.15],
    [0.15, 0.85],
    [-0.25, 0.5],
    [1.4, 0.5],
    [0.5, -0.3],
])


def linear(p):
    return 2.0 * p[:, 0] - 3.0 * p[:, 1] + 1.0


def nonlinear(p):
    return np.sin(3.0 * p[:, 0]) * np.cos(2.0 * p[:, 1]) + 0.5 * p[:, 0]


def emit(name, vals):
    for i, v in enumerate(np.asarray(vals, dtype=float).ravel()):
        if np.isnan(v):
            print(f"{name},{i},0,nan")
        else:
            print(f"{name},{i},0,{v:.17e}")


def main():
    for label, f in (("linear", linear), ("nonlinear", nonlinear)):
        vals = f(SITES)
        emit(f"linearnd_{label}", LinearNDInterpolator(SITES, vals)(QUERIES))
        emit(f"clough_{label}", CloughTocher2DInterpolator(SITES, vals)(QUERIES))


if __name__ == "__main__":
    main()
