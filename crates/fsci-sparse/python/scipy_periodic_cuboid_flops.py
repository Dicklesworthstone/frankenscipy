"""SciPy's spsolve on the periodic-cuboid cell fsci's g68jq widening changed.

Counts, not times, so this is valid on a loaded host -- which is what this cell has needed all
along: the live harness's quiescence gate (host_mean_busy <= 0.200) has refused every attempt.

The matrix is the shifted anisotropic 7-point PERIODIC operator, identical in structure to
`laplacian_3d_periodic_cuboid(x, y, z, 1e-3, -0.75, -1.0, -1.25)` in perf_spsolve.rs.

Two modes so the interpreter, imports and matrix construction are SUBTRACTED rather than assumed
negligible -- on the cubic Dirichlet cell they were 4.5x the factorization itself:
  reps=0  build the matrix, solve ZERO times
  reps=N  build the matrix, solve N times
Run both under `perf stat` and difference them.
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def periodic_cuboid(x_extent, y_extent, z_extent,
                    shift=1.0e-3, wx=-0.75, wy=-1.0, wz=-1.25):
    n = x_extent * y_extent * z_extent
    diagonal = shift - 2.0 * (wx + wy + wz)

    def idx(z, y, x):
        return (z * y_extent + y) * x_extent + x

    rows, cols, data = [], [], []
    for z in range(z_extent):
        for y in range(y_extent):
            for x in range(x_extent):
                r = idx(z, y, x)
                rows.append(r)
                cols.append(r)
                data.append(diagonal)
                for nz, ny, nx, w in (
                    ((z - 1) % z_extent, y, x, wz),
                    ((z + 1) % z_extent, y, x, wz),
                    (z, (y - 1) % y_extent, x, wy),
                    (z, (y + 1) % y_extent, x, wy),
                    (z, y, (x - 1) % x_extent, wx),
                    (z, y, (x + 1) % x_extent, wx),
                ):
                    rows.append(r)
                    cols.append(idx(nz, ny, nx))
                    data.append(w)
    return sp.csc_matrix((data, (rows, cols)), shape=(n, n))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    x, y, z = (int(v) for v in (sys.argv[2:5] or (11, 11, 11)))
    a = periodic_cuboid(x, y, z)
    n = a.shape[0]
    rhs = np.array([1.0 + 0.125 * ((17 * i + 23) % 29) for i in range(n)])
    checksum = 0.0
    resid = 0.0
    for _ in range(reps):
        sol = spla.spsolve(a, rhs)
        checksum += float(sol[n // 2])
        resid = float(np.max(np.abs(a @ sol - rhs)) / np.max(np.abs(rhs)))
    print(f"SCIPY_PERIODIC reps={reps} extents={x}x{y}x{z} n={n} nnz={a.nnz} "
          f"checksum={checksum:.17e} max_relative_residual={resid:.6e} "
          f"scipy={__import__('scipy').__version__} numpy={np.__version__}")


if __name__ == "__main__":
    main()
