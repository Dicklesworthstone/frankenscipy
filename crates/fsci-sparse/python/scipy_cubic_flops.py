"""SciPy's splu on the SAME cubic cell fsci is profiled on, for a FLOPs-per-instruction read.

Counts, not times, so this is valid on a loaded host. The matrix is byte-identical in structure
to `laplacian_3d_cubic(16)`: 7-point Dirichlet Laplacian, diagonal 6.0, neighbours -1.0.

Two modes so the interpreter and import cost can be SUBTRACTED rather than assumed negligible:
  baseline  -- build the matrix, factor ZERO times
  measure   -- build the matrix, factor N times
Run both under `perf stat` and difference them.
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian_3d_cubic(side):
    n = side * side * side

    def idx(z, y, x):
        return (z * side + y) * side + x

    rows, cols, data = [], [], []
    for z in range(side):
        for y in range(side):
            for x in range(side):
                r = idx(z, y, x)
                rows.append(r)
                cols.append(r)
                data.append(6.0)
                for dz, dy, dx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                                   (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < side and 0 <= ny < side and 0 <= nx < side:
                        rows.append(r)
                        cols.append(idx(nz, ny, nx))
                        data.append(-1.0)
    return sp.csc_matrix((data, (rows, cols)), shape=(n, n))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    side = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    a = laplacian_3d_cubic(side)
    checksum = 0.0
    lu_nnz = 0
    for _ in range(reps):
        lu = spla.splu(a, permc_spec="COLAMD")
        lu_nnz = lu.L.nnz + lu.U.nnz
        checksum += float(lu.U.diagonal()[side])
    print(f"SCIPY_CUBIC reps={reps} side={side} n={a.shape[0]} nnz={a.nnz} "
          f"lu_nnz={lu_nnz} checksum={checksum:.17e} "
          f"scipy={__import__('scipy').__version__} numpy={np.__version__}")


if __name__ == "__main__":
    main()
