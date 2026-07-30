# A/A null gate audit: does frankenscipy veto on CI-straddle?

Agent `cc/NobleCedar`. Prompted by frankenlibc's finding that a `nulls_hold`
clause requiring each A/A null CI to **include 1.0** couples the verdict to the
null's *precision* with the coupling running backwards.

## Answer: NO. frankenscipy has no straddle veto anywhere. Gate left unchanged.

### 1. Structural proof

The gate is:

```rust
let null_edge = ours_null_high
    .max(scipy_null_high)
    .max(1.0 / ours_null_low.max(1e-9))
    .max(1.0 / scipy_null_low.max(1e-9))
    .max(1.0);
let required = 1.0 + 2.0 * (null_edge - 1.0);
// WIN iff ratio_low > required ; LOSS iff ratio_high < 1.0 / required
```

Null CIs enter **only** through `null_edge`, which is the largest multiplicative
deviation of any null CI endpoint from 1.0, and it sets the *margin the effect
must beat*. There is no clause anywhere requiring a null CI to contain 1.0.

**The coupling runs the correct direction.** A tighter null gives `null_edge`
closer to 1.0, which gives `required` closer to 1.0, which makes the row
*easier* to decide. Better measurement lowers the bar. That is the opposite of
the reported defect, where better measurement raised a veto.

Repo-wide search for the defect: `grep -rniE
'null_low[[:space:]]*[<>]=?[[:space:]]*1\.0|null_high...|straddle|nulls_hold'`
over `crates/**/*.rs` returns **no gate logic** — the only `straddle` hits are
root-finding brackets in `airy.rs`, `metamorphic.rs`, `stats`, `signal`,
`ndimage`, `linalg`. All six null-gated harnesses in this repo
(`perf_sparse_vs_scipy`, `perf_csr_matvec`, `perf_bdf_vs_scipy`,
`perf_kv_frac`, `special_bench`, `interpolate_bench`) use the identical
`required = 1.0 + 2.0 * (null_edge - 1.0)` form. **The whole repo is free of the
defect, including cod/BlackThrush's BDF harness.**

### 2. Empirical: the defect would have suppressed 2 of my 13 findings

Had frankenscipy carried the straddle clause, these rows would have been vetoed:

| artifact | method | side | ratio | null CI that excludes 1.0 | misses 1.0 by | margin over requirement |
|---|---|---:|---:|---|---:|---:|
| 2026-07-29-sparse-nonsymmetric | gmres | 32 | **3.9679x** | ours `[0.997426, 0.999946]` | **0.0054%** | **570.1x** |
| 2026-07-29-sparse-nonsymmetric | gmres | 96 | **0.8899x** | scipy `[1.000261, 1.002005]` | **0.0261%** | **27.4x** |

A 297% effect vetoed by a 0.005% arm-order asymmetry, at 570x the required
margin. This independently corroborates frankenlibc's diagnosis from a different
codebase and a different workload, and it is the same order of pathology
(their vetoes missed 1.0 by 0.04%–0.5% against effects 130%–265% away).

Across all 13 cells, **0 fail the corrected rule's clause 3** as originally
measured; max `|null median - 1|` was 0.288% against the 2.000% allowance.

### 3. Replication to frankenlibc's standard — and the verdict IS unstable

Same ELF `f98a82f228046e45dd53754e6a5a9d21668e435835753001a819fb1831b565ed`,
three pinnings, `lsqr`, 21 rounds:

| side | cpu63 (core31) | cpu31 (**SMT sibling of core31**) | cpu15 (core15, independent) | spread | verdicts |
|---:|---:|---:|---:|---:|---|
| 16 | 12.0172 | 12.1793 | 12.2478 | 1.92% | WIN/WIN/WIN |
| 32 | 3.9272 | 3.8470 | 3.9305 | 2.17% | WIN/WIN/WIN |
| 48 | 2.2963 | 2.4399 | 2.3059 | 6.25% | WIN/WIN/WIN |
| 64 | 1.7910 | 1.7215 | 1.6081 | 11.37% | WIN/WIN/WIN |
| 96 | 1.2936 | 1.0099 | 1.0091 | **28.19%** | **WIN/INDET/INDET** |

**Methodology error I made and then corrected:** my first replication core, cpu31,
is the **SMT sibling of the same physical core 31** as cpu63
(`/sys/.../cpu31/topology/thread_siblings_list` = `31,63`). That is not an
independent core, so I re-ran on cpu15 (core 15). Both non-original pinnings agree
with each other at side 96, so the conclusion does not rest on the flawed pair.

**This instability is NOT the gate defect.** frankenlibc's signature is a
*reproducible effect* with a *randomly moving verdict*. Mine is the inverse: the
**effect itself moves**. At side 96 our arm's own time went 217.31 → 285.46 ms
(**+31.4%**) while SciPy's barely moved (282.10 → 289.76 ms, +2.7%), and the A/A
null median degraded from 0.998953 to 0.974312 (cpu15) and 0.941922 (cpu31).
The gate then correctly raised `required` from 1.0100 to 1.1037 and declared
INDETERMINATE. **A moving verdict that tracks a moving effect is a gate working,
not a gate failing.** Per the instruction, the gate is left untouched.

## 4. What this audit falsifies in MY OWN previous result

The replication does not just clear the gate — it **retracts the headline of
`2c651d13e`**, which is the honest cost of running this audit.

| claim in `2c651d13e` | status after replication on an independent core |
|---|---|
| marginal per-unknown `b_ours/b_scipy` = **1.006x (PARITY)** | **RETRACTED.** Measures **1.306x** on cpu15. Parity was a single-core artifact. |
| extrapolated crossover `n ≈ 508,205` (side ≈ 713) | **RETRACTED.** `n ≈ 10,671` (side ≈ 103) on cpu15. |
| side 96 = `1.2936x` win | **RETRACTED.** `1.0091x`, INDETERMINATE, on two independent pinnings. |
| "we lose on dense streaming, **tie** on sparse matvec" | **Strong form refuted.** lsqr's deficit is 1.306x — real, but smaller than GMRES's 1.823–2.668x. Only the weak form survives: SpMV *narrows* the deficit, it does not erase it. |
| P4 scored "DECISIVELY FALSIFIED (parity)" | **Mis-scored.** P4 predicted a deficit *worse* than GMRES's ~2x; the truth is 1.306x — still falsified, but because the deficit is *smaller*, not because it vanished. |

### What survives replication, and is therefore the real result

- `a_scipy` = **29.654 / 29.742 / 27.817 µs** across three independent
  measurement paths (harness on two cores, plus the standalone in-process probe).
  **The interpreter-tax mechanism is robust.** This was always the load-bearing
  claim and it holds.
- Small-`n` wins are rock solid: side 16 spread **1.92%**, side 32 spread
  **2.17%** across three pinnings.
- Monotone decay survives on every pinning (cpu15: 12.2478 → 3.9305 → 2.3059 →
  1.6081 → 1.0091, strictly decreasing).
- `a_ours ≈ 0` on every pinning.

The large-`n` tail (sides 64 and 96) is **not measurable to better than ~10–28%
on this contended host** and must not be quoted at all until re-run on a booked
exclusive host. This is exactly the failure the `PROVISIONAL_NON_EXCLUSIVE` label
was created to anticipate, and the label did its job.

## 5. Feedback on the corrected rule's clause 3

Applying "each null **median** within 2% of 1.0" to all 15 replication cells:

| cell | verdict | `|o_med-1|` | `|s_med-1|` | clause 3 |
|---|---|---:|---:|---|
| cpu15 side 96 | INDET | 2.57% | 0.23% | vetoes — **correct**, unstable cell |
| cpu31 side 96 | INDET | 5.81% | 1.33% | vetoes — **correct**, unstable cell |
| cpu15 side 64 | WIN | 3.17% | 0.18% | vetoes — **correct**, that ratio drifted 10.2% |
| cpu15 side 16 | WIN **12.2478x** | 0.51% | **2.02%** | **vetoes — over-tight** |

Clause 3 earns its place on sides 64 and 96: it independently flags precisely the
cells replication proved unstable, which my margin form only caught at side 96.
**But it is not effect-size aware, and on cpu15 side 16 it vetoes a 12.25x effect
for a 2.02% arm-order median bias** — a milder instance of the same coupling
frankenlibc identified, differing in degree rather than kind. My existing margin
form handles that cell correctly: the degraded nulls raised `required` to 1.0508
and the 12.25x effect clears it easily.

Suggestion, not a unilateral change: make clause 3 **relative** — veto when the
null median deviation exceeds some fraction of the effect deviation (e.g. bias
must be under ~10% of the effect) with an absolute ceiling for genuinely broken
arms. That keeps the bias bound clause 3 exists to provide without re-importing
precision coupling at a different threshold. **I have not adopted clause 3**,
because my gate exhibits no defect and the instruction was to leave a
non-defective gate alone.

## 6. Integrity check

frankenlibc's check: a gate fix that turns losses into wins was really a
loosening. **I changed no gate**, so there is nothing to loosen. And the audit
moved my results strictly *against* my own interest: one win retracted to
indeterminate, a claimed parity revealed as a 1.306x deficit, and a crossover
estimate cut by 48x. No finding improved.

## Reproduction

```
# three pinnings, same ELF
for CPU in 63 31 15; do
  FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1 taskset -c $CPU \
    ./target/release-perf/perf_sparse_vs_scipy <side> 21 lsqr \
    crates/fsci-sparse/python/scipy_sparse_arm.py
done
```

Raw logs in `raw_cpu31/` and `raw_cpu15/`; the cpu63 originals are in
`../2026-07-29-lsqr-vs-scipy-live-arm/raw/`. `audit_gate.py` regenerates the
straddle and clause-3 tables from any set of these logs.
