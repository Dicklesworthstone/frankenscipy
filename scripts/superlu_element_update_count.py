#!/usr/bin/env python3
"""Count element-updates for the SAME fill, ours against SuperLU's.

WHY THIS EXISTS. `docs/NEGATIVE_EVIDENCE.md` closed the `apply_sorted_pivot_tail` line with
an explicit entry condition for any further micro-lever:

    "the next real question is no longer 'which line item' but whether this elimination shape
     can reach 4.05 at all.  Before more micro-levers, count what SuperLU actually does per
     element -- its supernodal panels perform dense BLAS-3 style updates, so a like-for-like
     comparison needs ITS update count, not just its instruction total.  If SuperLU performs
     materially fewer element-updates for the same fill, the remaining gap is algorithmic and
     no amount of kernel work closes it."

This is that count.  It decides a direction, not a lever:

  * SuperLU does MATERIALLY FEWER updates  -> the gap is algorithmic; kernel work cannot close
    it, and the ledger's closure stands permanently.
  * SuperLU does the SAME or MORE updates  -> the gap is per-element COST, i.e. kernel quality,
    and kernel work is admissible again.

## The measure, and why it is algorithm-independent

For a fixed fill pattern, the multiply-subtract count of Gaussian elimination is a property of
the FACTOR, not of the traversal order:

    updates = sum over pivots k of  |L[k+1:, k]| * |U[k, k+1:]|

A right-looking elimination (ours) performs exactly that many.  A left-looking supernodal code
(SuperLU) performs the same updates grouped differently -- and, because its panels are DENSE,
it additionally computes entries that are structurally zero inside a panel.  So SuperLU's
element-update count is this number PLUS the panel padding, and can never be below it.

That asymmetry is the whole answer, and it is why this can be settled from the factor alone
rather than from SuperLU's internals: the sparse count is a LOWER BOUND on what SuperLU does,
so if we already perform only the sparse count, SuperLU cannot be doing materially fewer.

## What is reported

  sparse_updates     the fill-determined lower bound, from SciPy's own factor
  panel_updates      SuperLU-style dense-panel count over the supernodes of that same factor
  padding_ratio      panel_updates / sparse_updates -- how much extra a dense panel pays

Run:  python3 scripts/superlu_element_update_count.py [side]
"""

import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


DIAGONAL = 4.001
WEST = -1.2
EAST = -0.8
VERTICAL = -1.0


def convection_diffusion_2d(side):
    """The fixture `perf_spsolve.rs` builds, entry for entry."""
    n = side * side
    data, indices, indptr = [], [], [0]
    for row in range(side):
        for column in range(side):
            index = row * side + column
            if row > 0:
                indices.append(index - side)
                data.append(VERTICAL)
            if column > 0:
                indices.append(index - 1)
                data.append(WEST)
            indices.append(index)
            data.append(DIAGONAL)
            if column + 1 < side:
                indices.append(index + 1)
                data.append(EAST)
            if row + 1 < side:
                indices.append(index + side)
                data.append(VERTICAL)
            indptr.append(len(data))
    assert len(data) == 5 * n - 4 * side, (len(data), 5 * n - 4 * side)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def sparse_update_count(lower, upper):
    """sum_k |L[k+1:,k]| * |U[k,k+1:]| -- the fill-determined element-update count."""
    lower_csc = lower.tocsc()
    upper_csr = upper.tocsr()
    n = lower.shape[0]
    total = 0
    below = np.zeros(n, dtype=np.int64)
    right = np.zeros(n, dtype=np.int64)
    for k in range(n):
        cols = lower_csc.indices[lower_csc.indptr[k] : lower_csc.indptr[k + 1]]
        below[k] = int((cols > k).sum())
        rows = upper_csr.indices[upper_csr.indptr[k] : upper_csr.indptr[k + 1]]
        right[k] = int((rows > k).sum())
        total += below[k] * right[k]
    return total, below, right


def supernode_widths(lower):
    """Maximal runs of consecutive columns of L sharing the same below-diagonal pattern."""
    lower_csc = lower.tocsc()
    n = lower.shape[0]

    def pattern(k):
        cols = lower_csc.indices[lower_csc.indptr[k] : lower_csc.indptr[k + 1]]
        return tuple(int(c) for c in cols if c > k)

    widths = []
    start = 0
    while start < n:
        base = pattern(start)
        end = start + 1
        # A column with NO below-diagonal structure cannot anchor a useful supernode: there is
        # no shared dense block to exploit, and grouping such columns would inflate the mean
        # width without changing any update count (an r=0 panel does no work).  This is the
        # same convention the in-tree `supernode_block_density` uses when it excludes width-1
        # blocks.  Found by the MUST-MISS arm of `_assert_detector_can_see_a_supernode`, which
        # a diagonal factor failed by reporting one supernode spanning every column.
        while base and end < n:
            nxt = pattern(end)
            # A supernode continues while the next column's pattern is the previous one
            # minus the retired pivot row -- the standard exact-supernode test.
            if nxt != tuple(c for c in base if c > end):
                break
            end += 1
        widths.append(end - start)
        start = end
    return widths


def panel_update_count(lower, upper, widths):
    """SuperLU-style DENSE panel updates over the same factor.

    A supernode of width w whose columns collectively touch r rows below the diagonal is a
    dense w x r block; applying it to a target column costs r multiply-subtracts per column of
    the supernode regardless of whether the entry is structurally present.  Summed over the
    targets that supernode updates.
    """
    lower_csc = lower.tocsc()
    upper_csr = upper.tocsr()
    n = lower.shape[0]
    total = 0
    start = 0
    for width in widths:
        end = start + width
        rows = set()
        for k in range(start, end):
            cols = lower_csc.indices[lower_csc.indptr[k] : lower_csc.indptr[k + 1]]
            rows.update(int(c) for c in cols if c >= end)
        targets = set()
        for k in range(start, end):
            cols = upper_csr.indices[upper_csr.indptr[k] : upper_csr.indptr[k + 1]]
            targets.update(int(c) for c in cols if c >= end)
        # WIDTH MATTERS. Applying a width-`w` panel to one target column is a rank-`w`
        # update: each of the `w` supernode columns contributes a multiply-subtract to each of
        # the `r` rows.  Omitting `w` here made the panel count come out 14x BELOW the sparse
        # lower bound (padding_ratio 0.0724), which is impossible -- a dense panel can only do
        # MORE work than the sparse count for the same fill.  The implausible ratio is what
        # exposed the missing factor.
        total += width * len(rows) * len(targets)
        start = end
    return total


def _assert_detector_can_see_a_supernode():
    """MUST-HIT / MUST-MISS on the detector, before any count is believed.

    A detector that always returned width 1 would print exactly the result this script
    produced on its first run, and would look like a finding.  So: a DENSE lower factor has one
    supernode spanning every column and must come back wide; a diagonal factor has none and
    must come back all-ones.  Both are checked, and a failure here voids the run.
    """
    n = 6
    dense = sp.csc_matrix(np.tril(np.ones((n, n))))
    widths = supernode_widths(dense)
    assert widths == [n], f"dense factor must be ONE supernode of width {n}, got {widths}"

    diagonal = sp.csc_matrix(np.eye(n))
    widths = supernode_widths(diagonal)
    assert widths == [1] * n, f"diagonal factor must have NO supernodes, got {widths}"


def main():
    _assert_detector_can_see_a_supernode()
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    matrix = convection_diffusion_2d(side)
    n = matrix.shape[0]
    print(f"fixture: convection_diffusion_2d(side={side})  n={n}  nnz={matrix.nnz}")
    print(f"scipy {sp.__name__} {__import__('scipy').__version__}  numpy {np.__version__}")

    factor = spla.splu(matrix.tocsc())
    lower, upper = factor.L, factor.U
    lu_nnz = lower.nnz + upper.nnz
    print(f"SuperLU factor: L nnz={lower.nnz:,}  U nnz={upper.nnz:,}  L+U nnz={lu_nnz:,}")

    sparse_updates, below, right = sparse_update_count(lower, upper)
    print(f"sparse_updates (fill-determined lower bound) = {sparse_updates:,}")

    widths = supernode_widths(lower)
    wide = [w for w in widths if w >= 2]
    mean_width = n / len(widths)
    print(
        f"supernodes: {len(widths):,} groups, mean width {mean_width:.2f}, "
        f"{len(wide):,} of width >= 2"
    )

    panel_updates = panel_update_count(lower, upper, widths)
    print(f"panel_updates (dense SuperLU-style over the same factor) = {panel_updates:,}")
    if sparse_updates:
        print(f"padding_ratio = panel/sparse = {panel_updates / sparse_updates:.4f}")

    print()
    # SELF-CONSISTENCY GATE ON MY OWN MODEL, and it currently FAILS.
    #
    # A faithful dense-panel count can never fall below the sparse count for the same fill:
    # densification only ADDS structurally-zero entries.  So `panel_updates < sparse_updates`
    # is proof that the panel model is wrong, not a finding about SuperLU.  This gate exists
    # because the first two versions of `panel_update_count` printed confident verdicts off
    # exactly such numbers -- 0.0724 (missing the width factor) and then 0.5395.
    #
    # The residual defect is known: `rows`/`targets` are taken at `>= end`, so the count covers
    # only the EXTERNAL panel-to-target updates, while `sparse_updates` also includes the
    # intra-panel elimination (the w x w triangle and its w x r TRSM).  The two therefore
    # measure different regions and are not comparable until the intra-panel work is added.
    if panel_updates < sparse_updates:
        print("VERDICT: **VOID -- the panel model is not faithful and no verdict is printed.**")
        print(f"  panel_updates ({panel_updates:,}) < sparse_updates ({sparse_updates:,}),")
        print("  which is impossible for a correct dense-panel count: densification only adds.")
        print("  Cause: this model counts only external panel-to-target updates and omits the")
        print("  intra-panel elimination that sparse_updates includes.")
        print()
        print("  WHAT IS STILL SOLID, because it does not depend on the panel model:")
        print(f"    * SuperLU fill on this fixture: L+U nnz = {lu_nnz:,}")
        print(f"    * fill-determined element-updates = {sparse_updates:,} (algorithm-independent)")
        print(f"    * EXACT supernode structure: mean width {mean_width:.2f}, "
              f"{len(wide):,} of {len(widths):,} groups have width >= 2")
        print()
        print("  WHAT THE STRUCTURE SUGGESTS (an inference, NOT the count the predicate asked")
        print("  for): at mean exact width 1.78 SuperLU has little dense-panel leverage on this")
        print("  fixture, so a large BLAS-3 advantage is unlikely to be the explanation -- which")
        print("  independently corroborates frankenscipy-9nw95's finding that exact supernodes")
        print("  on its loss fixture averaged 1.10 and were useless.  Treat as a lead, not a")
        print("  verdict: the predicate asks for a COUNT and this model cannot yet supply one.")
        return 1

    print("VERDICT (the entry condition this script exists to settle):")
    print("  SuperLU does NOT do fewer element-updates for this fill -- its dense panels")
    print("  compute at least the sparse count and generally more.  The remaining gap is")
    print("  therefore per-element COST, not update count, so kernel quality is the live")
    print("  variable and further kernel work is admissible.")
    print(f"  padding_ratio {panel_updates / sparse_updates:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
