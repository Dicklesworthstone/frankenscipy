"""SciPy's splu on the SAME cubic cell fsci is profiled on, for a FLOPs-per-instruction read.

Counts, not times, so this is valid on a loaded host. The matrix is byte-identical in structure
to `laplacian_3d_cubic(16)`: 7-point Dirichlet Laplacian, diagonal 6.0, neighbours -1.0.

Two modes so the interpreter and import cost can be SUBTRACTED rather than assumed negligible:
  baseline  -- build the matrix, factor ZERO times
  measure   -- build the matrix, factor N times
Run both under `perf stat` and difference them.

THE SUBTRACTION IS SAFE FOR FLOPs AND FRAGILE FOR INSTRUCTIONS, and a row that quotes one
without saying which counter it used cannot be audited. Measured on this host at side 16,
reps 5 (frankenscipy-6940p follow-up):

    counter                  baseline (reps=0)   total (reps=5)   baseline share
    fp_ret_sse_avx_ops.all              28,036    2,176,174,766           0.001%
    instructions, 1 cpu          1,182,118,432    3,574,596,842            33.1%
    instructions, 8 cpus         2,730,395,256    5,580,099,864            48.9%

For FLOPs the baseline is noise and the difference is the measurement. For INSTRUCTIONS the
baseline is a third to a half of the total, so the difference is a large cancellation and
inherits the baseline's variance — and that baseline moves with CPU affinity, because
OpenBLAS thread startup and spin-wait retire instructions without retiring FLOPs. The same
5 factorizations read 2.392e9 instructions pinned to one CPU and 2.850e9 across eight, a
19% swing, while their FLOP count is identical to the last few thousand ops.

So: quote FLOP-derived quantities from this script freely; quote instruction-derived ones
only with the affinity stated and both raw numbers reported, which is why `main` now prints
them instead of leaving the caller to difference two runs and record only the result.
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
    # Affinity is reported because this script's INSTRUCTION count depends on it (BLAS
    # thread spin-wait) while its FLOP count does not. A row quoting instructions without
    # it is not reproducible; see the module docstring.
    try:
        import os
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = -1
    print(f"SCIPY_CUBIC reps={reps} side={side} n={a.shape[0]} nnz={a.nnz} "
          f"lu_nnz={lu_nnz} checksum={checksum:.17e} affinity={affinity} "
          f"scipy={__import__('scipy').__version__} numpy={np.__version__}")


if __name__ == "__main__":
    main()
