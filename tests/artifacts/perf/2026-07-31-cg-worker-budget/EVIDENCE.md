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

---

## Quiet-host recheck (same day, later): the retirement is CONFIRMED and stronger

The sweep above ran while co-tenant repos were loading the box (CV 15–27%, SciPy
arm drifting 9172–11516 ms). The host later quieted. Re-ran the same cells,
narrow-index path disabled so only the budget varies, ELF
`fad1abaa05f4022a573f99aca0fc43bd8b1150d5526580d9b3522fad6f7478d3`:

| nnz/worker | observed tasks | OURS p50 | SciPy p50 | Incumbent ratio | CI95 | CV |
|---|---:|---:|---:|---|---|---|
| 128K (incumbent) | 9 | **240.724 ms** | 6536.926 ms | **26.1388x** | [13.6171, 46.4746] | 38.7% |
| 32K | 39 | 383.493 ms | 5379.886 ms | 15.0584x | [13.0184, 17.2739] | 9.1% |
| 16K | 14 → 64 | 885.240 ms | 6905.944 ms | 7.4571x | [4.4237, 8.0281] | 19.2% |

**Raw milliseconds now separate cleanly: 240.7 → 383.5 → 885.2, a 3.68x
degradation from the incumbent budget to 64 workers, monotone.** On the loaded
host the direction was suggestive and the CIs overlapped; on the quiet host the
effect is unambiguous. P1 is falsified conclusively and the 128K budget stands.

**Correction to a number this repo now carries.** The CG incumbent ratio at
side=512 recorded in
`tests/artifacts/perf/2026-07-31-cg-parallel-levers/EVIDENCE.md` is **7.4638x**.
That cell was measured under co-tenant load. The same code and fixture on the
quiet host measures **26.14x** here and **30.38x / 32.01x** in the narrow-index
A/B below. The 7.46x figure is a floor observed under contention, not the
surface's ratio; anything quoting it should say so. The quiet-host cells still
carry CV 15–39%, so the honest statement is **"26–32x on an unloaded host,
7.5x under co-tenant load"**, not a single number.

## Narrow `u32` indices: NOT DECIDED

Same ELF, side=512, budget fixed at the incumbent 128K, toggling only index
width. Bit-identical by construction — same indices, same order, only storage
width changes — and confirmed identical at 494/494 iterations, residual
`9.870e-6`.

| index width | OURS p50 | SciPy p50 | Incumbent ratio | CI95 |
|---|---:|---:|---|---|
| `u32` (candidate) | 230.098 ms | 7333.306 ms | 30.3781x | [28.8357, 42.0565] |
| `usize` (incumbent) | 238.477 ms | 7335.793 ms | 32.0082x | [26.5017, 41.4995] |

3.5% apart in raw milliseconds with CV ~15% and CIs that almost entirely
overlap; the ratio point estimates even order the other way. **No decidable
effect.** The bandwidth argument that motivated it is sound in principle but
does not show up here, most likely because a 16 MB matrix on this box is not
DRAM-bound — the 5-point Laplacian averages 4 nonzeros per row, so the cost is
the latency of gathering `p`, which narrowing the index stream does not touch.

**Status note:** this lever is nonetheless present in `main` — a concurrent
commit (`4e88f73fd`) swept it in from the shared working tree, default-on, while
this measurement was in flight. It is byte-identical and harmless, but on this
evidence it is unjustified rather than beneficial. It should either get a
decisive measurement on a workload that is genuinely DRAM-bound (many more
nonzeros per row than a 5-point stencil) or be removed.
