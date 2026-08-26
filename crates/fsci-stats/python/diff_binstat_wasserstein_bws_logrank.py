"""SciPy comparator for diff_binstat_wasserstein_bws_logrank.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.stats binned_statistic_2d, wasserstein_distance_nd, bws_test and logrank -- entry
points with no differential coverage (frankenscipy-ivxx6).

Randomised entry points from the same backlog (monte_carlo_test, sobol_indices, dunnett) are
excluded on purpose: comparing two draws is not comparing two answers.
"""
import numpy as np
from scipy.stats import (binned_statistic_2d, bws_test, logrank,
                         wasserstein_distance_nd)

X = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.75, 0.35, 0.95, 0.05, 0.55, 0.8, 0.45])
Y = np.array([0.2, 0.1, 0.7, 0.3, 0.85, 0.55, 0.4, 0.95, 0.6, 0.15, 0.9, 0.5])
V = np.array([1.0, 2.5, -1.0, 4.0, 0.5, 3.5, -2.0, 6.0, 1.5, 2.0, -0.5, 3.0])

U_PTS = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.5], [2.0, 1.0]])
V_PTS = np.array([[0.5, 0.25], [1.5, 1.0], [2.5, 0.5]])
UW = np.array([1.0, 2.0, 0.5, 1.5])
VW = np.array([2.0, 1.0, 3.0])

A = np.array([1.2, 2.4, 0.7, 3.1, 1.9, 2.8, 0.4])
B = np.array([2.9, 3.6, 1.8, 4.2, 3.3, 2.2])

S1 = np.array([6.0, 7.0, 10.0, 15.0, 19.0, 25.0, 30.0])
S2 = np.array([4.0, 8.0, 11.0, 13.0, 16.0, 21.0])


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    parts = ["nan" if np.isnan(v) else f"{v:.17e}" for v in flat]
    print(f"{name}|" + ";".join(parts))


def main():
    for stat in ("mean", "sum", "count", "median", "min", "max", "std"):
        res = binned_statistic_2d(X, Y, V, statistic=stat, bins=4)
        dump(f"binstat2d_{stat}", res.statistic)
        if stat == "mean":
            dump("binstat2d_xedges", res.x_edge)
            dump("binstat2d_yedges", res.y_edge)

    dump("wass_nd_plain", [wasserstein_distance_nd(U_PTS, V_PTS)])
    dump("wass_nd_weighted",
         [wasserstein_distance_nd(U_PTS, V_PTS, UW, VW)])

    bws = []
    for alt in ("two-sided", "less", "greater"):
        try:
            r = bws_test(A, B, alternative=alt)
            bws.extend([float(r.statistic), float(r.pvalue)])
        except Exception:
            bws.extend([np.nan, np.nan])
    dump("bws", bws)

    lr = []
    for alt in ("two-sided", "less", "greater"):
        r = logrank(S1, S2, alternative=alt)
        lr.extend([float(r.statistic), float(r.pvalue)])
    dump("logrank", lr)


if __name__ == "__main__":
    main()
