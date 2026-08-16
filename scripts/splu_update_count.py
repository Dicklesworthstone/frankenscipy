#!/usr/bin/env python3
"""Instructions per elimination update for the general sparse LU, ours and the incumbent's.

WHY THIS EXISTS. The splu cubic cell cannot be adjudicated by wall time on this
box: the same source rebuilt on two rch workers reads it 3.4% apart, and two
sanctioned ELFs of the balanced-square harness have read it 14% apart, both with
passing A/A nulls (docs/NEGATIVE_EVIDENCE.md, frankenscipy-kapqa). A lever worth
less than ~5% is therefore undecidable there no matter how many rounds are run.

Instruction counts do not have that problem. They are independent of host load,
of which worker built the binary, and of the balanced square's drift. So for this
kernel the rule is: COUNT FIRST, and time only what the count says is large.

WHAT IT COMPUTES. The denominator is the work itself. For a right-looking
elimination the number of per-entry updates is

    sum over pivots k of  |L below the diagonal in column k| * |U right of the diagonal in row k|

— one multiply-subtract into the trailing submatrix per pair — taken from the
actual symbolic factor of the RCM-ordered matrix, not from a proxy like nnz.
Live SuperLU is used only as the symbolic oracle here, under the SAME reverse
Cuthill-McKee ordering our kernel applies and with `permc_spec="NATURAL"` so it
cannot reorder underneath the comparison. Our fill is at parity with it, which is
what makes a cost-per-update comparison meaningful in the first place.

HOW TO GET THE NUMERATORS.

  ours:
    valgrind --tool=callgrind --cache-sim=yes \
      --callgrind-out-file=/path/cg.out \
      ./target/release/perf_splu <side> 9 0 off cubic
    callgrind_annotate /path/cg.out | grep factorize_csr

    Divide by the number of factorizations, which the harness reports as
    `cubic_spectral_toggle_reads` (4 per round + 1 per warmup + 1 parity). Read
    it; do not assume it.

  the incumbent:
    valgrind --tool=callgrind --cache-sim=yes python3 scripts/splu_update_count.py --factor-only
    callgrind_annotate <out> | grep _superlu     # sum every symbol in the library

    Python, imports, matrix construction and any OpenBLAS spin-wait thread are
    EXCLUDED from that sum, which is the conservative direction for us.

MEASURED 2026-08-16 (PeachSummit), thinkstation1:

    side  updates      ours instr/update   SuperLU instr/update   ratio
    10    3,738,282    50.30               15.11                  3.33x
    12    13,186,899   48.96               13.45                  3.64x

Flat to 2.7% across a 3.5x change in update count, so it is a per-entry constant.
Our LL miss rate is 0.0% at both sizes, so the kernel is cache-resident and the
gap is instructions and their efficiency, NOT locality. Target for any lever on
frankenscipy-llywn: instructions/update at or below ~15. A change that does not
move this number should not be timed at all.
"""

import argparse

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee

NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def cubic_dirichlet_laplacian(side):
    """The `laplacian_3d_cubic` fixture: 7-point Dirichlet stencil on a side^3 grid."""
    n = side**3

    def index(i, j, k):
        return (i * side + j) * side + k

    rows, cols, data = [], [], []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                r = index(i, j, k)
                rows.append(r)
                cols.append(r)
                data.append(6.0)
                for di, dj, dk in NEIGHBOURS:
                    a, b, c = i + di, j + dj, k + dk
                    if 0 <= a < side and 0 <= b < side and 0 <= c < side:
                        rows.append(r)
                        cols.append(index(a, b, c))
                        data.append(-1.0)
    return sp.csc_matrix((data, (rows, cols)), shape=(n, n))


def rcm_permuted(matrix):
    """Our kernel's ordering: symmetric reverse Cuthill-McKee on A."""
    perm = reverse_cuthill_mckee(matrix.tocsr(), symmetric_mode=True)
    return matrix[perm, :][:, perm].tocsc()


def factor(matrix):
    return spla.splu(matrix, permc_spec="NATURAL", diag_pivot_thresh=1.0)


def right_looking_updates(lu):
    lower = lu.L.tocsc()
    upper = lu.U.tocsr()
    updates = 0
    for k in range(lower.shape[0]):
        below = int((lower.indices[lower.indptr[k] : lower.indptr[k + 1]] > k).sum())
        right = int((upper.indices[upper.indptr[k] : upper.indptr[k + 1]] > k).sum())
        updates += below * right
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", type=int, nargs="+", default=[10, 12])
    parser.add_argument(
        "--factor-only",
        action="store_true",
        help="factor once and stop — the shape to run under callgrind when counting "
        "what the incumbent spends, since the update count must not be in the profile",
    )
    args = parser.parse_args()

    for side in args.sides:
        permuted = rcm_permuted(cubic_dirichlet_laplacian(side))
        lu = factor(permuted)
        if args.factor_only:
            print(f"side={side} lu_nnz={lu.L.nnz + lu.U.nnz}", flush=True)
            continue
        updates = right_looking_updates(lu)
        print(
            f"side={side} n={permuted.shape[0]} nnz={permuted.nnz} "
            f"lu_nnz={lu.L.nnz + lu.U.nnz} right_looking_updates={updates:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
