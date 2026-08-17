"""Factor the same cubic Dirichlet Laplacian with SciPy's SuperLU, once.

Used under callgrind to count the INCUMBENT's instructions for the identical
factorization, so "are we doing more work or the same work less efficiently" can be
answered with a number instead of an argument.

The fixture is reproduced to match the Rust harness by shape: 3-D 7-point Dirichlet
stencil on side^3, diagonal 6, off-diagonal -1. For side=16 that is n=4096, nnz=27136,
which is what the harness reports.
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

side = int(sys.argv[1]) if len(sys.argv) > 1 else 16
n = side ** 3
idx = lambda z, y, x: (z * side + y) * side + x

rows, cols, vals = [], [], []
for z in range(side):
    for y in range(side):
        for x in range(side):
            r = idx(z, y, x)
            rows.append(r); cols.append(r); vals.append(6.0)
            for dz, dy, dx in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < side and 0 <= ny < side and 0 <= nx < side:
                    rows.append(r); cols.append(idx(nz, ny, nx)); vals.append(-1.0)

A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
print(f"fixture n={n} nnz={A.nnz}", flush=True)

# `setup` runs everything except the factorization, so subtracting the two callgrind
# totals isolates SuperLU's own work from Python and NumPy startup.
if len(sys.argv) > 2 and sys.argv[2] == "setup":
    print("setup-only", flush=True)
else:
    lu = spl.splu(A)
    # DO NOT touch lu.L / lu.U here. Materialising them builds two new sparse matrices,
    # which the harness's timed arm never does -- including that work inflated SuperLU's
    # instruction count to a level implying IPC 7.6, which is impossible and is how the
    # confound was caught.
    print(f"perm_r0={lu.perm_r[0]}", flush=True)
