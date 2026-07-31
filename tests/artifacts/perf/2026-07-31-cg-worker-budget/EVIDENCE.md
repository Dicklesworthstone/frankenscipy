# CG worker budget: widening it LOSES, and not for the reason it was chosen

Date: 2026-07-31. Lane: cc / BlackThrush. Host `thinkstation1`, 32 physical /
64 logical, affinity `0-63`, governor `powersave`.
Pre-registration: `PREREGISTERED_MECHANISM.md`, committed `551625d93`, before
the budget was made settable or anything was timed.
Harness `perf_csr_matvec cg-vs-scipy`, Dirichlet five-point Laplacian side=512
(`n=262,144`, `nnz=1,046,528`), `rtol=1e-5`, live SciPy 1.17.1 in the same
invocation with single-thread caps, 5 rounds × 1 rep.
ELF `d4ca76ff1ea8ad852891dde0a8dfe272da9702728d1ce821e586a25e3919d8b5`,
built `rch exec --base 551625d93 --clean-overlay`, hash self-reported in-process.

## Result

| nnz/worker | shift | observed worker tasks | OURS p50 | SciPy p50 | Incumbent ratio: SciPy / FrankenSciPy | CI95 |
|---|---|---:|---:|---:|---|---|
| 128K (incumbent) | 17 | 9 | 659.117 ms | 10090.149 ms | **15.0648x** | [10.2799, 15.6667] |
| 64K | 16 | 19 | 966.854 ms | 11515.830 ms | 11.9106x | [10.6646, 20.2820] |
| 32K | 15 | 39 | 889.136 ms | 9171.760 ms | 8.7745x | [8.5534, 13.5011] |
| 16K | 14 | 64 | — | — | run cut off at the 10-minute cap; not reported | — |

Every cell converged in **494/494 iterations** with residual `9.870e-6`,
identical to the incumbent, so **P4 holds**: the row-band partition changes
worker count and nothing else. All three cells are DECIDED FrankenSciPy wins
against live SciPy; the question here is only which of *ours* is fastest.

## Verdict: P1 FALSIFIED, hypothesis retired

**P1 predicted more workers would help. The ratio falls monotonically as
workers rise: 15.06 → 11.91 → 8.77.** P2 and P3 are moot — there is no win to
size or to locate an optimum inside.

State the uncertainty honestly: the ratio CIs overlap (CV 15–27%; the host was
carrying co-tenant load, and the SciPy arm itself moved 9172–11516 ms across the
three cells). Under the pre-registered rule — *a configuration wins only if its
ratio CI is disjoint from the incumbent's* — **no configuration wins, and the
incumbent 128K budget stands.** The monotone direction across three independent
cells is what makes this a retirement rather than a null result; a single noisy
pair would not be.

**Mechanism.** The budget was inherited from the per-iteration path, where it
amortised a `thread::scope` against one iteration. The pre-registration argued
that a persistent team (`46b0fe844`) removes that constraint. It does — and the
budget is still right, for a different reason: the 1M-nnz matvec is
**memory-bandwidth-bound**, so workers past the point that saturates the memory
system add barrier latency and cache pressure without adding bandwidth. Nine
tasks already saturate it on this box. This also confirms, now under the pool,
the provisional reject recorded in
`tests/artifacts/perf/2026-07-31-cg-parallel-levers/EVIDENCE.md` — that row
suspected the loss was spawn cost; it was not, it is bandwidth.

## What this retires, and what it points at

Lever "parallelise the sparse kernels harder across 64 cores" is **closed on the
CG surface**. It is not a thread-count problem: we are already at the bandwidth
roof at 9 workers, and both the persistent pool and the wider budget confirm it
from opposite directions.

The kernel being bandwidth-bound is itself the pointer. The matvec streams, per
iteration, `data` (1,046,528 × 8 B) plus `indices` (1,046,528 × 8 B as `usize`)
= **~16 MB of matrix traffic**, against which the arithmetic is trivial. The
lever that moves a bandwidth roof is not more cores, it is **fewer bytes**:
`f32` values and `u32` indices halve that to ~8 MB. That is the registered next
step.

## Kept

`CG_WORKER_NNZ_SHIFT` stays as a `#[doc(hidden)]` runtime knob (one relaxed load
per solve, not per iteration) so this budget can be A/B'd inside one invocation
in future rather than across invocations, which is what made the earlier
rejection only provisional. `CG_WORKER_NNZ_SHIFT_DEFAULT` is **17**, unchanged
in behaviour from before this work.
