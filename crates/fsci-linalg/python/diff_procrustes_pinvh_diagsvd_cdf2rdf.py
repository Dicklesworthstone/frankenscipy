"""SciPy comparator for diff_procrustes_pinvh_diagsvd_cdf2rdf.

Emits the same `name,r,c,value` lines as the Rust probe so the two can be diffed directly.
Inputs MUST match the Rust side exactly.

Covers four scipy.linalg entry points that had no differential coverage (frankenscipy-ivxx6):
orthogonal_procrustes, pinvh, diagsvd, cdf2rdf.
"""
import numpy as np
from scipy.linalg import cdf2rdf, diagsvd, orthogonal_procrustes, pinvh


def dump(name, m):
    m = np.atleast_2d(np.asarray(m, dtype=float))
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            print(f"{name},{r},{c},{m[r, c]:.17e}")


def main():
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 10.0],
                  [-1.0, 0.5, 2.0]])
    b = np.array([[2.0, 1.0, 3.5],
                  [5.0, 4.5, 6.5],
                  [8.0, 7.5, 9.0],
                  [0.0, -0.5, 1.5]])
    r, scale = orthogonal_procrustes(a, b)
    dump("procrustes_r", r)
    print(f"procrustes_scale,0,0,{scale:.17e}")

    sym = np.array([[4.0, 1.0, -2.0],
                    [1.0, 3.0, 0.5],
                    [-2.0, 0.5, 5.0]])
    dump("pinvh_sym", pinvh(sym))

    rank_deficient = np.array([[2.0, 1.0, 3.0],
                               [1.0, 2.0, 3.0],
                               [3.0, 3.0, 6.0]])
    dump("pinvh_rank2", pinvh(rank_deficient))

    dump("diagsvd_5x3", diagsvd(np.array([3.0, 2.0, 1.0]), 5, 3))
    dump("diagsvd_3x5", diagsvd(np.array([3.0, 2.0, 1.0]), 3, 5))

    w = np.array([0.0 + 1.0j, 0.0 - 1.0j, 2.0 + 0.0j])
    v = np.array([
        [0.0 - 0.70710678118654746j, 0.0 + 0.70710678118654746j, 0.0 + 0.0j],
        [0.70710678118654746 + 0.0j, 0.70710678118654746 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
    ])
    wr, vr = cdf2rdf(w, v)
    dump("cdf2rdf_w", wr)
    dump("cdf2rdf_v", vr)


if __name__ == "__main__":
    main()
