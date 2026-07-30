# Result 2: lsqr two-arm incumbent ratio — P3 and P4 falsified, marginal cost at PARITY

Date: 2026-07-29. Agent `cc/NobleCedar`. Pre-registration: `8ef32a3c9`, committed
before any measurement. Result 1 (SciPy arm alone): `2df3d80a3`.

## Evidence class: PROVISIONAL_NON_EXCLUSIVE — read this first

**Every cell below is `host_wide_quiescence=NOT_CERTIFIED`.** The host-wide
exclusivity gate did not pass and was explicitly waived via
`FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1`. These numbers are **not DECIDED-class
evidence**. The waiver permanently suppresses the verdict token: the raw logs
contain zero occurrences of `DECIDED FRANKENSCIPY WIN` or
`DECIDED FRANKENSCIPY LOSS`, the only strings a reader or script would grep to
find a gated result. The sole appearance of the word is inside the literal
`NOT DECIDED-class evidence` warning. A provisional row therefore cannot be
laundered into a gated one by quoting its verdict line.

Contamination is quantified rather than hidden. At measurement time each cell
saw `maximum_busy_fraction=1.000` with this many CPUs above the 20% limit:

| side | CPUs above limit |
|---:|---:|
| 16 | 9 |
| 32 | 13 |
| 48 | 21 |
| 64 | 10 |
| 96 | 8 |

The gate cannot pass on this host at all while the swarm is active: it requires
**all 64 CPUs** below 20% busy for 300 ms, and a load average of ~12 means ~12
CPUs are runnable at any instant. An un-waived attempt was run first and
recorded to confirm this — it aborted with 29 CPUs above the limit, several at
100%. Default behaviour remains fail-closed; the waiver is opt-in and, once
tripped, permanently downgrades the run's verdict string.

**Why the numbers are still worth reporting.** The statistical gate is
self-calibrating and independent of the environmental one: contention widens the
per-arm A/A nulls, which raises the required margin. The nulls stayed tight and
every cell cleared the 2x-null margin by a wide factor:

| side | worst null edge | required | ratio CI95 | margin over requirement |
|---:|---:|---:|---|---:|
| 16 | 1.0105 (scipy) | ~1.021 | [11.8858, 12.7373] | ~582x |
| 32 | 1.0061 | ~1.012 | [3.9038, 3.9422] | ~386x |
| 48 | 1.0091 | ~1.018 | [2.2899, 2.3191] | ~125x |
| 64 | 1.0046 | ~1.009 | [1.7842, 1.7951] | ~86x |
| 96 | 1.0028 | ~1.006 | [1.2594, 1.3036] | ~44x |

The weakest cell still clears by ~44x. Contamination of this magnitude cannot
manufacture these ratios; it can only add noise, and the nulls show how much it
added. A booked exclusive host is still required before any of this is quoted as
DECIDED.

## Provenance

- Host `thinkstation1`, AMD Ryzen Threadripper PRO 5975WX, `physical_cores=32`,
  `logical_threads=64`, 215 GiB RAM, `runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`,
  **`avx512f=false`**. Pinned `affinity=63`, `cpuset_logical_cap=1`, one requested
  and observed worker thread per arm.
- `scaling_governor=powersave`, `energy_performance_preference=balance_performance`.
  **This differs from the GMRES campaign on `threadripperje`, which ran
  `performance`/`performance`.** Absolute levels are therefore not comparable
  across the two hosts; per-cell ratios and per-iteration slopes are the
  comparable quantities.
- Executed-binary ELF SHA-256
  `f98a82f228046e45dd53754e6a5a9d21668e435835753001a819fb1831b565ed`,
  **built on rch worker `vmi1293453`** using the deterministic-overlay form
  (`--base 2df3d80a3 --clean-overlay --overlay-path <one file>`) and retrieved by
  scp from the worker pool directory (~2.1 MB). No local Cargo build.
- SciPy 1.17.1, NumPy 2.4.3, CPython 3.13.12. SciPy engine SHA-256
  `f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f` — byte
  identical to the GMRES campaign.
- 21 rounds per cell, construction/serialization/counting outside timing.

## Measurement

| side | n | iterations ours / SciPy | matched | ours p50 µs/iter | SciPy p50 µs/iter | **incumbent ratio** |
|---:|---:|---:|---|---:|---:|---:|
| 16 | 256 | 177 / 177 | YES | 2.564 | 30.775 | **12.0172x** |
| 32 | 1,024 | 506 / 506 | YES | 10.034 | 39.424 | **3.9272x** |
| 48 | 2,304 | 875 / 874 | **NO** | 22.559 | 51.881 | **2.2963x** |
| 64 | 4,096 | 1,355 / 1,355 | YES | 41.616 | 74.530 | **1.7910x** |
| 96 | 9,216 | 2,325 / 2,325 | YES | 93.467 | 121.332 | **1.2936x** |

Iteration counts match **exactly** in four of five cells, by construction from
the stopping-rule mapping (`atol=0, btol=rtol, conlim=0` reduces SciPy's test1 to
our `|phi_bar|/||b|| < tol`). Side 48 differs by one iteration — **ours did 875
against SciPy's 874**, i.e. we did *more* work, so its `2.2963x` is conservative,
not inflated. It is excluded from the fit below; the analysis script prints
`EXCLUDED from fit` for it automatically.

Conformance passed in every cell: zero tolerance mismatches, `max_abs_diff`
between `1.9e-6` and `2.5e-6`, `relative_l2_diff` between `5.1e-9` and `1.1e-8`,
and true relative residuals agreeing to three digits (e.g. side 96:
`9.983e-6` ours vs `9.966e-6` SciPy).

## Per-iteration decomposition (4 matched cells)

| arm | `a` (fixed µs/iteration) | `b` (µs/unknown) | R² |
|---|---:|---:|---:|
| FrankenSciPy | **-0.156** (i.e. ~0) | **0.010163** | 1.0000 |
| SciPy 1.17.1 | **29.654** | **0.010105** | 0.9967 |

- Fixed per-iteration overhead, SciPy minus ours: **29.809 µs/iteration**
- **Marginal per-unknown cost, ours / SciPy: `1.006x` — parity.**
- Extrapolated equal-cost crossover: **n ≈ 508,205** (side ≈ 713)

Cross-validation: the independent single-arm probe in Result 1 measured
`a_scipy = 27.817 µs` and `b_scipy = 0.010035`. The harness, on a different
measurement path (subprocess + IPC, different contention), gives `29.654` and
`0.010105` — within 6.6% and 0.7%. Two independent paths agree.

## Prediction scorecard

| # | Prediction | Outcome |
|---|---|---|
| P1 | Win at smallest size; ratio decays monotonically; a crossover exists | **Confirmed on the first two clauses.** 12.0172 → 3.9272 → 2.2963 → 1.7910 → 1.2936 is strictly monotone decreasing. The crossover exists only by extrapolation at n≈508k, ~60x beyond the tested range — so the third clause is not confirmed *within* the range and should not be scored as a hit. |
| P2 | `a_scipy(lsqr) >= 87.2 µs` | **FALSIFIED** (Result 1, confirmed here: 29.654 µs, low by 2.9x). |
| P3 | Crossover in the same order as GMRES's 8,242, i.e. order `10^4` | **FALSIFIED.** Measured n* ≈ 5.1×10^5, order `10^5`–`10^6`. My mid-course revision to order `10^3` (n*≈2,900, after P2 fell) was **also wrong, in the opposite direction**. Both the original prediction and its correction missed. |
| P4 | Marginal disadvantage worse than GMRES's ~2x, hence an *earlier* crossover, because `A^T v` on CSR is a scatter for us | **DECISIVELY FALSIFIED, and it was my highest-risk prediction.** Measured marginal ratio is **1.006x — parity, not a 1.8–2.7x deficit** — and the crossover is ~62x *later*, not earlier. The stated `A^T`-scatter mechanism is wrong: our transpose path is competitive. |

Three of four predictions falsified, including the one I flagged as riskiest and
the one I revised mid-course.

## What the falsifications teach

**The interesting finding is the marginal-cost parity, and it is the opposite of
the GMRES result.** Same fixture family, same protocol, same two engine hashes —
but note the GMRES figures were taken on `threadripperje` under the
`performance` governor and these on `thinkstation1` under `powersave`, so this is
a **cross-host** comparison. It is defensible only because each side is a
*within-host ratio* of two arms measured in the same invocation, which cancels
host and governor effects to first order; the raw `b` values themselves are not
comparable across the two hosts.

| method | `b_ours / b_scipy` |
|---|---:|
| GMRES restart-20 | 1.823x – 2.668x (we are worse) |
| lsqr | **1.006x (parity)** |

Proposed mechanism, stated as a hypothesis and not as a measured fact: the two
methods spend their `O(n)` per-iteration budget on different primitives.

- GMRES restart-20's per-iteration `O(n)` work is dominated by modified
  Gram–Schmidt: ~9.3 separate `dot`/`axpy` pairs streaming dense length-`n`
  vectors. That is a **dense streaming** workload where NumPy's per-call
  BLAS-backed loops are strong and our kernels are ~2x weaker.
- lsqr's per-iteration `O(n)` work is dominated by **two sparse matvecs**
  (`A @ v` and `A.T @ u`). SpMV is memory-bandwidth-bound with irregular access,
  so both implementations sit against the same bandwidth wall and neither can
  pull ahead.

If that hypothesis holds, the campaign-level rule is: **we lose on dense vector
streaming and tie on sparse matvec.** That predicts where to expect parity versus
deficit in other methods, and it is testable — `bicg`, `cgs` and `qmr` are
implemented on our side and run on this same fixture with different
streaming-to-SpMV ratios. It also means the pre-registered framing was too
coarse: `b_ours/b_scipy` is not a property of our sparse code in general, it is a
property of which primitive the method's inner loop is made of.

**The structural argument survives and is stronger here than on GMRES.** With
`a_ours ≈ 0` and marginal parity, the entire advantage is SciPy's ~29.7
µs/iteration interpreter tax, uncontaminated by any marginal-cost penalty on our
side — which is exactly why the ratio decays so slowly and the crossover sits
two orders of magnitude out rather than at n≈8k.

## Reproduction

```
FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1 taskset -c 63 \
  ./target/release-perf/perf_sparse_vs_scipy <side> 21 lsqr \
  crates/fsci-sparse/python/scipy_sparse_arm.py
```

Drop the environment variable to get the fail-closed default. Analyze with
`decompose.py` from
`tests/artifacts/perf/2026-07-29-gmres-per-iteration-decomposition/`, which
refuses any cell whose arms disagree on iteration count.

**Retry predicate:** do not re-run these five cells to "confirm" them. Reopen
only to (a) re-measure on a booked exclusive host to convert PROVISIONAL to
DECIDED, or (b) test the dense-streaming-versus-SpMV hypothesis by predicting
`b_ours/b_scipy` for `bicg`/`cgs`/`qmr` from their inner-loop primitive mix
*before* measuring.

## Raw log SHA-256

Raw stdout/stderr for all five cells is in `raw/` alongside this file:

- `raw/lsqr_side_16.txt` — `b3d1b877d166c6e4b3e24c31fe240472b1c680b114d2620ff4930fc6e60647d6`
- `raw/lsqr_side_32.txt` — `0ae90470f6a3a92eb0769e9964ee89ee467e8db46a30aa2ea611c519b38e8843`
- `raw/lsqr_side_48.txt` — `21a5e1ecee875f1fe97efa4c823551883bda1d54af76e60e87337281d64e2d7b`
- `raw/lsqr_side_64.txt` — `4fddfb2cce4477dd8f5cde5f15185a8ea6b9dec0b9752a282159e0a3018bbb03`
- `raw/lsqr_side_96.txt` — `4ac4d445587dba4b6049847a923e8a9f142c82b6d8a3762b650be51f2829651b`
