# CG on 64 cores vs live SciPy 1.17.1 — two levers attempted, both REJECTED,
# and a 7.46–17.25x incumbent win that was already there

Date: 2026-07-31. Lane: cc / BlackThrush. Host: `thinkstation1`, 32 physical
cores / 64 logical threads, AMD, governor `powersave`, affinity `0-63`.
Harness: `crates/fsci-sparse/src/bin/perf_csr_matvec.rs` `cg-vs-scipy`.
Fixture: Dirichlet five-point Laplacian, `diagonal=4.001`,
`rhs=1+0.01*(i%17)`, `rtol=1e-5`, `atol=0`, `x0=zeros`.
Build route for every binary below:
`RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec --base
eb2d4945765d5d9c0afaf882847194ff7b7d8af8 --clean-overlay --overlay-path … --
cargo build --profile release-perf -p fsci-sparse --bin perf_csr_matvec
--features live-scipy-bench`, ELF scp'd back and SHA-256 verified, self-reported
from inside the process.

## The headline is the baseline, not either lever

Unchanged `main` (`eb2d49457`), same invocation as live SciPy 1.17.1, SciPy
capped to one thread, identical iteration counts, agreement `6.5e-15`:

| side | n | nnz | iters (ours/SciPy) | OURS p50 | SciPy p50 | Incumbent ratio: SciPy / FrankenSciPy | gate |
|---|---:|---:|---|---:|---:|---|---|
| 256 | 65,536 | 326,656 | 330 / 330 | 249.372 ms | 4175.790 ms | **17.2525x** ci95 [16.1177, 20.2246] | worst_null_edge 1.2070 < required 1.4140 → **DECIDED WIN** |
| 512 | 262,144 | 1,046,528 | 494 / 494 | 901.564 ms | 6729.054 ms | **7.4638x** ci95 [6.8837, 7.7016] | worst_null_edge 1.2448 < required 1.4896 → **DECIDED WIN** |

Baseline ELF SHA-256 `27a5ccee5b603a790085fe9305a489763adecf259366adad23a390aae5788ae6`.
Observed FrankenSciPy worker tasks: 3 (side 256), 9 (side 512) — *actual*, sampled
off the timing path, not requested.

The previously recorded CG row is **1.2116x at side=80, `n=6400`, pinned to one
CPU** (`tests/artifacts/perf/2026-07-28-sparse-cg-vs-scipy-live-arm/`). That
number was never wrong; it was measured on one core. Letting the same code use
the box turns it into 7.46–17.25x. The ratio falls from 17.25x to 7.46x as n
grows, so this is a fixed-overhead advantage over SciPy's per-iteration Python
tax diluting into bandwidth-bound work, not a scaling win.

## Lever 1 — fuse and parallelise the per-iteration vector sweeps: REJECTED

Mechanism registered before measuring: a CG iteration is one parallel SpMV plus
**four serial O(n) passes** (`p·Ap`, the x/r update, `r·r`, the p update). Fold
`p·Ap` into the SpMV so the matrix is touched once, carry `r·r` inside the x/r
update, and run all three remaining sweeps across the cores, chunked on whole
8192-element reduction blocks with block partials combined in ascending order so
the result is a pure function of the data — identical for 1 thread and for 64.

Candidate ELF `183ef34147a6b7e0a1c7a7fdbb8339bcae87df3924224bc459ff57a428f4fb58`.

| side | baseline OURS p50 | candidate OURS p50 | candidate vs baseline |
|---|---:|---:|---|
| 256 | 249.372 ms | 426.469 ms | **1.71x SLOWER** |
| 512 | 901.564 ms | 1363.820 ms | **1.51x SLOWER** |

Iteration counts and residuals identical to baseline at both sizes, so the
blocked reduction changed nothing numerically that mattered; the loss is pure
overhead. **Mechanism of the loss:** at n=65,536 a vector is 512 KB and the four
"serial" passes are L2-resident and fast; a `thread::scope` per sweep costs tens
of microseconds and buys nothing. Two sweeps per iteration × 330 iterations is
660 spawns against work that never left cache. Growing n to 262,144 moved the
ratio only 1.71x → 1.51x, so this is not a gate-placement problem that a higher
threshold would fix. This is the same per-step-spawn wall already recorded for
the eigh tridiagonal reduction and the Cholesky panel TRSM.

## Lever 2 — raise the SpMV thread cap: REJECTED, with a caveat

`csr_matvec_into` budgets 128K nnz per thread (`nnz >> 17`), so a 1M-nnz matvec
gets 7 threads on a 64-core box. Tried 32K nnz per thread (`nnz >> 15`).

| side | baseline OURS p50 | candidate OURS p50 | candidate vs baseline |
|---|---:|---:|---|
| 512 | 901.564 ms | 1520.261 ms | **1.69x SLOWER** |

Candidate ELF `9fc5e40734f840c0b17dc85a9829b0866cd780ad7e84b90b3663d2b042bc2b89`.

**Stated plainly: this row is confounded and is a provisional reject, not a firm
one.** The two arms ran in different invocations and the host drifted between
them — the live SciPy arm moved 6729 ms → 9923 ms (+47%) across the same gap,
so some of our +69% is host, not lever. The direction agrees with lever 1's
mechanism (more spawns against bandwidth-bound work), which is why it is
recorded as a reject rather than re-run immediately, but a thread cap is a
runtime constant and therefore **can** be A/B'd inside one invocation. Anyone
promoting or overturning this row should do that first.

## Retry predicate for both levers

Do not re-attempt either as a per-iteration `thread::scope`. Both are blocked on
the same missing substrate: a **persistent worker pool** that a Krylov iteration
can dispatch to without paying spawn cost per sweep. That is the same blocker
already named for the eigh tridiagonalisation, the Cholesky panel TRSM, and the
blocked-reduction retry. Reopen when the pool exists, or under an s-step /
communication-avoiding restructure that does s matvecs per synchronisation
instead of one, which changes the spawn count per unit work rather than the
spawn cost.

## Harness change kept

`perf_csr_matvec.rs` previously aborted unless pinned to exactly one CPU, which
makes it structurally incapable of measuring a parallelism lever. It now admits
a multi-CPU cell, reports `affinity_cpu_count` and `available_parallelism`, and
reports **actual observed** FrankenSciPy worker tasks sampled from
`/proc/self/task` around an untimed warm-up solve while the SciPy arm keeps its
single-thread caps. Requested threads are never substituted for observed ones.
