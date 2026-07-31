# PRE-REGISTERED: the CG worker budget is still sized for the old spawn cost

Written and committed before changing the budget or timing anything.
Date: 2026-07-31. Lane: cc / BlackThrush.

## The claim

`cg_persistent_workers` (`46b0fe844`) creates **one** scoped worker team per
solve, with phase barriers instead of a `thread::scope` per iteration. It kept
the worker budget it inherited from the per-iteration path:

    workers = min(available_parallelism, nnz >> 17, n)

`nnz >> 17` is **128K nnz per worker**. That number exists to amortise a
`thread::scope` against one iteration's work. **With a persistent team that cost
is paid once per solve, not once per iteration**, so the constraint the budget
was chosen under no longer applies. At side=512 (`nnz = 1,046,528`) it yields
**7 workers on a 64-core box**, and the SpMV is the dominant term in our
iteration (measured 2026-07-31: ours 1.82 ms/iteration at 1M nnz, of which a
bandwidth-bound 1M-nnz matvec is most).

## Why this is not a re-run of the rejected row

`tests/artifacts/perf/2026-07-31-cg-parallel-levers/EVIDENCE.md` rejected
`nnz>>15` at **1.69x SLOWER**. That measurement was taken **before** the
persistent pool existed: every extra worker cost an extra spawn on every one of
494 iterations. The retry predicate recorded there was "reopen when the pool
exists". It exists. This is that retry, not a repeat.

## Mechanism

Widening the budget adds workers whose spawn cost is now amortised over the
whole solve. The kernel is bandwidth-bound, so the prediction is **not** linear
scaling — it is that we currently sit below the memory-level parallelism the
machine can sustain, because 7 cores cannot saturate the memory system of a
32-physical-core box.

## Predictions and falsifiers

**P1 — more workers help at all.** Predict the incumbent ratio at
`shift=14` (16K nnz/worker) exceeds the `shift=17` ratio at side=512.
Falsified if it is equal or lower.

**P2 — the win is sublinear and saturates.** 8x the workers will not give 8x.
Predict the best shift beats `shift=17` by a factor in `[1.2, 4.0]`. Falsified
outside that band.

**P3 — the optimum is interior.** Predict the best budget is not the most
extreme one tested: some shift in `{14,15,16}` beats both `shift=17` and the
one-worker-per-row extreme, because per-worker row bands eventually get too
small to cover barrier latency. Falsified if the ratio is monotone in shift
across the whole sweep.

**P4 — conformance is untouched.** The row-band partition is a partition of the
same arithmetic, so predict identical iteration counts and a solution agreeing
with live SciPy to the same order as the incumbent row (`~1e-14` relative L2).
Any change in iteration count or a residual above `1e-5` rejects the candidate
before its timing is read.

## Measurement

Host drift confounded the earlier rejection: the two arms ran in different
invocations and the SciPy arm moved 47% between them. **The decision statistic
is therefore the incumbent ratio, not our raw milliseconds** — each
configuration is measured against live SciPy 1.17.1 inside its own invocation,
so a host that moves under both arms divides out. A configuration wins only if
its ratio CI is disjoint from the incumbent configuration's.

Standing requirements unchanged: strict `rch exec --base <commit>
--clean-overlay`, ELF SHA-256 self-reported in-process, actual observed worker
tasks rather than requested, live SciPy in the same invocation, corrected null
gate with the median clause.

## Chooser

Ship the widened budget only if P1 and P4 both hold and the best shift's ratio
CI is disjoint from `shift=17`'s. Otherwise leave the budget alone and record
that 128K nnz/worker is correct for reasons other than spawn amortisation.
