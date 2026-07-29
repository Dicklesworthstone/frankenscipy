# Ledger Resurrection Audit — frankenscipy

**Campaign:** `perf-campaign-20260725`, Fleet-Wide Meta-Lever #1.
**Lane:** cc / STRUCTURAL (`CopperFalcon`). **Date:** 2026-07-25.
**Sources audited:** `docs/NEGATIVE_EVIDENCE.md` (948 entries) + `docs/progress/perf-negative-results.md`
(249 entries) = **1,197 headings**, of which **194 are REJECT-class** (REJECT / INVALID / NO-SHIP /
BLOCKER / dead-end).

The premise, from three independent fleet discoveries (frankenlibc `bd-3ollh0`, frankenmermaid,
franken_networkx): a large fraction of standing REJECTs are **VOID** — the measurement could not have
detected the lever, so the harness was rejected, not the lever.

> **CANONICAL CORRECTION — 2026-07-25 allocation addendum.** The owner required
> frankenfs's six classes verbatim and required every screened row to be hand
> adjudicated. The earlier sections below are retained as historical provenance,
> but their added classes, their broad ISA candidate list, and their treatment of
> `.168` as `VOID-CV` are superseded by this correction and Appendix A.

## Canonical v3 method and taxonomy

The current ledgers contain 1,203 `##` headings. A title screen selected 199
headings containing REJECT, INVALID, NO-SHIP, BLOCKER, or dead-end language. The
screen was triage only: every selected section was read and adjudicated by hand.
Thirty headings are routing-only blockers, surveys, superseded corrections,
feasibility decisions, or correctness-only rejects. Appendix A shows them as
`EXCLUDED`, but they are not counted as performance-lever rejections. The remaining
169 rows use the six fleet classes exactly:

| Class | Binding meaning |
|---|---|
| `VALID-PROFILE` | Rejected before any source edit on a named, non-zero profile frame with a computed Amdahl ceiling. |
| `VALID-MECHANISM` | No A/A null, but a counted mechanism (instructions, cycles, syscalls, allocations, faults, or an equivalent exact work count) refutes the lever. A null cannot make removed work appear when the count did not move. |
| `VALID-AB` | A/B carried a real A/A null and the claimed effect sits inside it. |
| `VOID-CV` | A/B ran and was killed **only** by a `cv < 5%` rule. |
| `VOID-ZEROSELF` | The verdict's profile put the candidate target at approximately 0% self-time, or was proven to predate the candidate and therefore could not contain its frame. |
| `VOID-NONULL` | A/B was rejected on a near-unity or otherwise non-separating wall ratio with no A/A null and no counted-mechanism refutation. |

`VOID-ISA` is the narrow supplemental class used by frankenfs §6, not a license
to reopen every SIMD-adjacent row. It applies only when the rejected mechanism
was vector-width-shaped and the measured binary was pinned to the former SSE2
fleet floor. `ovh-b` is now out of the Rust pool; the remaining 73 Rust slots on
11 workers all expose AVX2+FMA.

### Corrected counts

| Metric | Count |
|---|---:|
| Title-screen hits, including routing/survey/correctness records | 199 |
| Excluded after hand adjudication | 30 |
| **Performance REJECT rows audited** | **169** |
| `VALID-PROFILE` | 7 |
| `VALID-MECHANISM` | 47 |
| `VALID-AB` | 20 |
| `VOID-NONULL` | **77** |
| `VOID-CV` | 11 |
| `VOID-ZEROSELF` | 4 |
| `VOID-ISA` | 3 |
| **VOID total** | **95 / 169 = 56.2%** |
| Rows with an explicit 64-hex executed-ELF SHA in the row | 8 / 169 = 4.7% |

The corrected composition matches the fleet finding: **`VOID-NONULL` is 77 of
95 void rows (81.1%)**. CV is not the epidemic. The old `.168` KDE entry is the
important correction: it failed the recorded A/A floor as well as CV, so it is
`VALID-AB`, not `VOID-CV`. Its later median-bootstrap rerun correctly re-stood
the reject at 1.118× with candidate CI-low inside the doubled-null floor.

Raw counts deliberately retain duplicate entries where both ledgers record the
same result. Ranking and retry work deduplicate by mechanism family.

## Corrected top-five queue, ranked by target-frame self-time

| Rank | Deduplicated VOID family | Target self-time | Corrected class | Outcome |
|---:|---|---:|---|---|
| 1 | `.166` segmented cubic cursor | `CubicSplineStandalone::eval` **92.69%** | `VOID-CV` | Literal restoration re-run: **4.268× KEEP**, `f31cdeb90` |
| 2 | `.167` trust-exact SPD Cholesky solve | `solve_augmented_flat` **89.09%** | `VOID-CV` | Literal restoration re-run: **1.49× KEEP**, `5bc336436` |
| 3 | `.165` BDF exact-diagonal Newton | dense LU **80.09%** + solve **5.16%** | `VOID-CV` | Literal restoration re-run: **97.68–109.37× KEEP** at n=512, `2e7110315` |
| 4 | MR4×NR4 packed-SYRK high-CV family | candidate SYRK **53.88–65.0%** | `VOID-CV` | Superseded by the same structural family as MR4×NR8: **1.143× KEEP**, `23355d1c5` |
| 5 | fused panel-TRSM pack family | fused producer **23.05–29.42%** | `VOID-CV` / `VOID-NONULL` | Superseded by blocked-FMA panel TRSM: **1.115× KEEP**, `c7e9062bf` |

The queue has outcome coverage for all five families. The first three were
literal source restorations; ranks 4–5 were independently re-derived and
shipped before this taxonomy correction. They are reported as superseding
family outcomes, not falsely described as literal reruns. The current
allocation is Lane B, so this correction ran **no benchmark and took no
worker**.

Concrete retry predicate for any still-open `VOID-NONULL` row: first obtain a
fresh profile showing non-zero target self-time; then use one executed ELF that
self-reports its SHA-256 and runs interleaved A/A plus A/B in the same
invocation; decide only on the deterministic bootstrap-median CI and record the
counted mechanism. A row with no recorded self-time ranks below every row with
a measured non-zero target frame.

## Narrow ISA-floor adjudication

Only three current rows survive the hand screen as `VOID-ISA`:

| Rank | Entry | Why the old verdict is ISA-shaped | Retry predicate |
|---:|---|---|---|
| 1 | `NEGATIVE_EVIDENCE.md:5193` — Cholesky SYRK 8-dot register tile | Register-width tile measured against the SSE2 floor; the 8-dot lowering and spill budget change under AVX2. | AVX2+FMA worker, shipped ELF self-hash, same-invocation A/A+A/B, bootstrap-median CI, exact factor bits. |
| 2 | `NEGATIVE_EVIDENCE.md:5735` — Cholesky panel SIMD dots | Portable-SIMD panel dots were judged neutral on the former floor; AVX2 changes lane width in the exact target. | Same contract, plus candidate-frame attribution and serial-baseline comparison. |
| 3 | `NEGATIVE_EVIDENCE.md:18634` — `ndtri` central-region SIMD | The vector arm itself was the rejected mechanism; AVX2 changes central-lane throughput while scalar log/sqrt tails remain the explicit ceiling. | Same contract, plus per-region cycle counts and exact-bit proof. |

SIMD-adjacent does not automatically mean void. The tensor-leaf SIMD row
already had a real A/A control and ran after the ISA correction (`VALID-AB`);
pdist SoA lost because strided reads defeated a contiguous auto-vectorized
baseline (`VALID-MECHANISM`, a direction that strengthens under AVX2); batched
FFT SIMD lost on added cache-cold traffic and bandwidth
(`VALID-MECHANISM`). These exclusions keep the class small and specific.

## Appendix A — complete hand-adjudication index

This index makes the hand pass reproducible without copying 199 long ledger
bodies into this document. Audit index `N` means line `N` from:

```bash
rg -in '^## .*(REJECT|INVALID|NO.SHIP|BLOCKER|dead.end)' \
  docs/NEGATIVE_EVIDENCE.md docs/progress/perf-negative-results.md | nl -ba
```

The inputs adjudicated here have SHA-256
`86bd32d362edec962afebdc19e9be717d052a908cf31c7b4c5b1704c2c6c17af`
(`NEGATIVE_EVIDENCE.md`) and
`0ca0a77545fe3ec0f80bdc028aa9f5b9c14e9c7eaa4b0a7e0e726ab6a3408f27`
(`progress/perf-negative-results.md`). Each number below was assigned only
after reading the complete section body through the next `##` heading. The
source row remains the evidence record for its ratio, null interval, target
self-time, counted mechanism, and ELF SHA; the index records the resulting
verdict. Counts sum to all 199 screen hits.

| Verdict | Audit indices |
|---|---|
| `VALID-AB` (20) | 1, 4, 8, 10, 106, 108, 109, 111, 112, 114, 143, 147, 148, 149, 154, 157, 161, 163, 195, 197 |
| `VALID-PROFILE` (7) | 36, 48, 97, 105, 176, 186, 194 |
| `VALID-MECHANISM` (47) | 6, 12, 15, 16, 18, 22, 23, 29, 39, 41, 49, 50, 51, 52, 53, 57, 59, 61, 67, 73, 78, 79, 81, 82, 86, 89, 93, 110, 113, 122, 130, 141, 146, 150, 151, 153, 162, 165, 169, 170, 171, 174, 175, 177, 178, 196, 199 |
| `VOID-NONULL` (77) | 14, 17, 19, 20, 21, 24, 25, 26, 27, 31, 32, 33, 34, 37, 40, 43, 44, 45, 46, 47, 55, 56, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 74, 75, 80, 84, 91, 94, 98, 104, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 144, 145, 167, 168, 172, 173, 179, 180, 181, 182, 183, 187, 193 |
| `VOID-CV` (11) | 5, 100, 101, 103, 158, 159, 160, 188, 189, 190, 192 |
| `VOID-ZEROSELF` (4) | 95, 96, 184, 185 |
| `VOID-ISA` (3) | 35, 42, 92 |
| `EXCLUDED` (30) | 2, 3, 7, 9, 11, 13, 28, 30, 38, 54, 72, 76, 77, 83, 85, 87, 88, 90, 99, 102, 107, 128, 129, 152, 155, 156, 164, 166, 191, 198 |

---

## Historical v2 headline — superseded by the canonical correction above

**Superseded 2026-07-25 (second pass).** The first pass used a taxonomy I invented. The fleet has
since standardised on **frankenfs's six classes**, which are better, and this section is the re-audit
under them. The scoreboard line is:

```
frankenscipy | 1203 | 196 | 48 | 24.5% | 4 | 3 | 109.37x
```

| Class | Meaning | Sound? | Rows |
|---|---|---|---:|
| `VALID-DECISIVE` † | No recorded null, but the ratio is far outside the fleet band [0.905, 1.105] in the losing direction | ✅ | 103 |
| `VALID-AB` | A/B with a recorded A/A null; the effect sits inside it | ✅ | 13 |
| `VALID-INFEASIBLE` † | Refuted on feasibility/correctness, not on timing (inexpressible in safe Rust, not bit-identical, premise measured false, invalid comparator) | ✅ | 13 |
| `VALID-MECHANISM` | No null, but refuted on a **counted** mechanism — instructions/cycles/syscalls/allocations/faults unchanged | ✅ | 10 |
| `VOID-NONULL` | Near-1.0 ratio, no null, no counted mechanism — cannot distinguish lever from harness | ❌ | 30 |
| `VOID-CV` | Killed **only** by a `cv < 5%` gate | ❌ | 12 |
| `VOID-ZEROSELF` | Target frame ~0% self-time in the profile the bench actually ran | ❌ | 2 |
| `UNMEASURED` | No number was ever produced (rch admission / build breakage) | ❌ | 4 |
| `UNCLASSIFIED` | No ratio recorded and no class determinable from the text | — | 9 |

† Two additions to the fleet taxonomy, labelled as additions so they cannot silently inflate "valid".
`VALID-DECISIVE` exists because `VOID-NONULL` is explicitly about **near-1.0** ratios: a measured
0.5× with no null is not ambiguous. `VALID-INFEASIBLE` exists because this repo rejects a lot of
levers on feasibility, and a timing gate cannot void a feasibility finding.

**VOID: 48 / 196 = 24.5%.** Rows carrying a binary sha256: **13 / 196 = 6.6%** — worse than
frankenfs's 10.9%, and squarely on us. Rows with no A/A null of any kind: 164 / 196.

### frankenscipy is the fleet's counter-example on `VOID-CV`

The broadcast's correction — *"I predicted the CV gate would be the dominant void class. It is NOT"* —
holds for frankenfs and **inverts here**:

| | frankenfs | frankenscipy |
|---|---:|---:|
| `VOID-CV` | 4 / 219 = **1.8%** | 12 / 48 = **25.0%** |
| `VOID-NONULL` | 214 / 219 = 97.7% | 30 / 48 = 62.5% |

These are two different diseases. frankenfs's void pile is *old* prose written before that repo had
null controls at all — an **absence** of a control. Ours is *recent*: `cv < 5%` was written into this
repo's KEEP gate and applied with discipline through 2026-07-23, so our rejects have nulls **and were
killed anyway** — an actively **wrong** control. A fleet scoreboard on `void_pct` alone would have
shown us as healthier than frankenfs at the exact moment we were discarding 17–19× wins.

## 2. Method, and what hand-adjudication changed

Mechanical screen (`scratchpad/audit_ledger_v2.py`, read-only), then **every VOID row read in full and
adjudicated by hand**. The screen is triage, not a verdict — it was wrong on three rows, all in the
*valid* direction, and all three corrections are recorded here rather than quietly applied:

| Row | Screen | Hand verdict | Why |
|---|---|---|---|
| `NEGATIVE_EVIDENCE.md:23810` rayon persistent pool | VOID-NONULL | **VALID-MECHANISM** | Records a counted mechanism: cycles 1.10e9 (scope) vs 1.74e9 (rayon) — the candidate does 58% *more* work. A null cannot change that. My own row; my own screen nearly voided it. |
| `:6289` SphericalVoronoi `u32` stamps | VOID-NONULL | **VALID-DECISIVE** | The decisive large row is **5.32× slower**; the screen latched onto a 1.04 small-n ratio. |
| `:7877` SphericalVoronoi adjacency patch | VOID-NONULL | **VALID-DECISIVE** | 0.689× vs parent, 3.16× slower than SciPy. Not near-1.0. |

**`VALID-MECHANISM` cuts both ways, and applying it honestly cost me a row I would otherwise have
claimed.** Anyone publishing a yield should re-grep their `VOID-NONULL` pile for
`cycles`/`instructions`/`faults` counters first.

## 3. Institutionalized — the audit now runs every time

Per the decay lesson (frankensqlite audited once four months ago, institutionalized the check, and
sits at 1.7%): **`scripts/ledger_preflight.py`**, exit 2 = BLOCKED.

- `--propose "<lever>"` before touching source — blocks if a prior REJECT with a *sound* class
  already covers it.
- `--check-row <ledger>` before committing a row — blocks a REJECT that records neither an A/A null
  nor a counted mechanism, and blocks one that rests on a `cv` ceiling.

Writing a null-less REJECT is now refused rather than merely discouraged. Testing it against the rows
whose right answer this repo learned the hard way found three real bugs in it — a profile attribution
being mistaken for a counted mechanism, a negation trap (`.165` says "*not* an IN-FLOOR result" and
the matcher fired on the phrase inside its own denial), and ratio-picking that mislabelled a 2.5×
regression as void. All three are fixed and pinned by the five canonical rows in its commit message.

### 2026-07-25 allocation addendum — ISA-floor VOID candidates

The original audit predated the fleet-wide resolution of `frankenscipy-hhr7j`. The orchestrator
subsequently surveyed `/proc/cpuinfo` on all 12 RCH workers: `ovh-b` (Ivy Bridge E3-1245 V2) was the
only host without AVX2+FMA. Its `rust` tag is now removed, leaving 73 Rust slots on 11 workers that
all expose AVX2+FMA. The workspace-wide AVX2+FMA pin landed in `d89ca19f6`; before that commit,
FrankenSciPy artifacts used the generic x86-64 SSE2 floor.

That discovery adds a new candidate class without retroactively inflating the counts above:
**`VOID-ISAFLOOR`**. Every pre-`d89ca19f6` REJECT/NO-SHIP whose proposed mechanism was SIMD,
vector-width, vectorization, or ISA-shaped is now a VOID **candidate** because its timing answered
the wrong deployment question. Initial ledger grep includes the Cholesky 8-dot SYRK tile, cdist
metric SIMD, ndimage output-pixel SIMD, batched FFT SIMD-across-rows, the DCT/FFT SIMD walls, and
ndtri central-region SIMD. This label does not erase a correctness refutation (bit mismatch),
feasibility proof, or a real-workload memory-latency diagnosis; it voids only the old timing
verdict until those rows are separated from their mechanism findings.

Concrete retry predicate for every `VOID-ISAFLOOR` candidate: Lane M may re-decide it only from an
AVX2+FMA worker admitted by the new 11-worker set, with the executed ELF SHA-256 self-reported,
same-invocation A/A and A/B arms, and the deterministic bootstrap-median CI gate. Until then the
old SSE2-floor number is historical provenance, never a current KEEP/REJECT verdict.

**The frankenscipy-specific finding is not the 21.6%.** It is the composition: this repo's void rows are
**not** dominated by the frankenlibc "in-band with no control" class. They are dominated by
**decisively-outside-the-null results that were thrown away by a `cv` gate**. Four of them — `.165`
through `.168`, exactly the entries the campaign flagged — carry claimed ratios of **1.20×, 1.47×,
3.82× and 17.4–19.3×** against recorded A/A nulls of **1.00–1.02**. Those are not undecidable
measurements. They are wins that the gate refused to look at.

---

## 2. Method

Read-only parse of both ledgers, one record per `##` heading. Per REJECT-class row: extract the
decisive ratio (preferring `paired median` / `median` / `centered` / `speedup p50` over any loose `Nx`
in the prose), then test for an A/A null control, a binary sha256, the target frame's self-time, an
explicit `cv`-ceiling rejection, and mechanism-refutation language.

VOID criteria are the campaign's, verbatim, with one addition and one subtraction:

- **Added:** an explicit `cv`-ceiling rejection voids the row *even when a null control exists*, because
  the gate — not the null — is what killed it. This is the dominant class here.
- **Subtracted:** a row whose rejection rests on a **mechanism refutation** (inexpressible in safe Rust,
  premise measured false, not bit-identical, invalid comparator, measurement artifact) **stands**
  regardless of where its ratio sat. A feasibility finding is not a timing claim, so a bad timing gate
  cannot void it. 46 rows land here, including `cholesky pack-fusion is inexpressible in safe Rust` and
  `compute_axis_support fold interior — premise was FALSE (zero idiv emitted)`.

Auditor script: `scratchpad/audit_ledger.py` (read-only; it never touches repo source). Its self-time
extraction is a screen, not an oracle — the self-times in §3 were **hand-read from the entry bodies**,
and three of the top rows were corrected upward by 15–95 points against the script's guess.

---

## 3. Rehabilitation queue — VOID rows ranked by target-frame self-time

Self-time is the profile figure the entry itself records for the code under test. "Recoverable" is the
claimed ratio, i.e. what the re-run would have to reproduce.

| # | Entry | Target frame (self-time) | Claimed ratio | Recorded A/A null | Void reason | Verdict |
|---|---|---|---|---|---|---|
| 1 | `.168` N-D KDE four-query tile (2026-07-23) | `GaussianKdeNd::evaluate` **99.27%** | 1.196× p50 (p05 0.967) | null p95 1.068 | CV-GATE | VOID, but **reject RE-STOOD** on re-run (1.118× IN-FLOOR, `344c51020`) — see §5 |
| 2 | `.166` segmented cubic cursor (2026-07-23) | `CubicSplineStandalone::eval` **92.69%** | **3.819× p50 / 3.368× p05** | null p95 1.994 | CV-GATE | **VOID — RESURRECTED 2026-07-25 at 4.268×, `f31cdeb90`** |
| 3 | `.167` trust-exact SPD Cholesky solve (2026-07-23) | `solve_augmented_flat` **89.09%** | **1.467× p50 / 1.324× p05** | null p95 1.160 | CV-GATE | **VOID — RESURRECTED 2026-07-25 at 1.49×, `5bc336436`** |
| 4 | `.165` BDF exact-diagonal structured Newton (2026-07-23) | dense `LU` factorization **80.09%** + dense solve 5.16% | **17.384× / 17.069× / 19.283× / 18.964× p50 across four runs** | null p50 1.002–1.020 | CV-GATE | **VOID — RESURRECTED 2026-07-25 at 97.68–109.37×, `2e7110315`** |
| 5 | MR4×NR4 packed-SYRK pilot + stabilization + per-factor runs (2026-07-10, 3 rows × 2 files) | SYRK **60.1–65.0%** | 1.002–1.058× | none recorded | CV-GATE / INFLOOR | VOID — **already resurrected**, see §4 |
| 6 | fused panel-TRSM pack A/B + dual-pack write-through (2026-07-10, 2 rows × 2 files) | panel TRSM **23.1–59.2%** | 0.994–1.012× | none recorded | INFLOOR / CV-GATE | VOID — superseded by the landed blocked-FMA TRSM (`c7e9062bf`, 1.115×) |
| 7 | MR2 panel-TRSM A/B comparator (2026-07-10) | **0.1%** (dead arm) | 1.124× | none | ZEROSELF | VOID — the comparator never ran the code under test; already self-corrected in-ledger |
| 8–36 | 29 further in-band rows with no control recorded (sparse `eigsh` Lanczos, FFT strided small-tail gather, ODR beta-gradient reuse, `squareform_to_condensed`, binary-erosion bit-pack, `loadtxt` prepass bypass, WAV PCM encode, Gauss–Kronrod stack samples, affinity-propagation scratch, watershed bucket queue, SphericalVoronoi ×2, Mathieu matrix-free, linkage lazy arena, MatrixMarket scatter, `rosen` four-chain, …) | not recorded | 0.94–1.09× | none | INFLOOR | VOID-by-criterion, **low EV** — all sit inside ±10% of unity, so even a perfect harness buys ≤1.09× |

### Ranking by expected value, not by self-time

Self-time ranks *where the time is*; it does not rank *how much is recoverable*. The four CV-gate rows
carry claimed effects of 1.20× / 3.82× / 1.47× / **17–19×** against nulls of ~1.00. Re-run order is
therefore **`.165` → `.166` → `.167` → `.168`**, and the 29 tail rows are explicitly **not** queued:
their own numbers cap them at ≤1.09×, which the campaign's own decidability rule (2× margin over the
null) will not admit on this hardware. Ledger-completeness is served by recording them as VOID; agent
time is not served by re-running them.

---

## 4. Yield already banked — the audit's own proof case

Rows 5 and 6 of the queue were **independently re-derived and shipped by a later agent without knowing
they were voided**:

- The 2026-07-10 `MR4×NR4 packed-SYRK` pilots were invalidated on `cv 6.976%` / `raw CV > 5%` with no
  null control. On 2026-07-22 the same structural idea landed as the AVX2+FMA **MR4×NR8 trailing-SYRK
  micro-kernel — 1.143× DECIDED** (`23355d1c5`), against a properly-recorded A/A null.
- The 2026-07-10 `fused panel-TRSM pack` rows (`paired CV 11.564%`) were superseded by the **blocked
  GEMM-shaped panel TRSM — 1.115× DECIDED** (`c7e9062bf`), same lane, same frame.

Two of the top-6 void rows contained real, shippable wins. They cost the project twelve days and a
full re-derivation because the rows said REJECT instead of VOID. **That is the entire argument for this
audit**, and it is now measured rather than asserted.

---

## 5. Honest exclusions

Three findings that a less careful audit would have claimed and that this one does not:

1. **`.168` N-D KDE four-query tile is not resurrected.** Its batch-64 run has p05 **0.967** against a
   null p95 of **1.068** — the candidate's lower bound sits inside the null. It is void *as a cv-gate
   kill* and the row must be relabelled, but under the median-CI gate with a 2× margin the reject
   **still stands**. Void ≠ win.
2. **The 29-row in-band tail is void-by-criterion but not worth re-running.** Claiming "36 buried wins"
   from this audit would be exactly the overclaim the campaign is trying to stamp out.
3. **`85% of audited rows carry no A/A null` is a provenance fact, not 165 void rows.** A row that
   measured 0.41× does not need a null control to be safely rejected; the band test is what separates
   the two, and it is applied per row rather than as a blanket.

---

## 6. Resurrection yield

| Stage | Count |
|---|---:|
| Entries audited | 194 |
| VOID | 42 rows / 36 distinct results |
| Queued for re-run under the corrected harness | 4 (`.165`, `.166`, `.167`, `.168`) |
| Already resurrected by independent re-derivation before this audit | 2 (SYRK MR4×NR8 1.143×, blocked TRSM 1.115×) |
| Re-run this session | **4 of 4 — the entire queue** |
| **Re-won** | **3 — `.165`, `.166`, `.167`** |
| Reject correctly re-stood | **1 — `.168`, exactly as this audit predicted** |

The whole queue was worked in one session, across both lanes:

| # | Entry | Outcome | Commit |
|---|---|---|---|
| 1 | `.165` BDF exact-diagonal Newton | **RESURRECTED — 97.68–109.37× @n=512** (bit-identical) | `2e7110315` (cc) |
| 2 | `.166` segmented cubic cursor | **RESURRECTED — 4.268×**, CI [4.186, 4.367] vs null [0.980, 1.021], 0 bit mismatches over 100,000 outputs | `f31cdeb90` (cod) |
| 3 | `.167` trust-exact SPD Cholesky | **RESURRECTED — 1.49×**, max final-`x` Δ 2.45e-9 and objective Δ 7.15e-17 against `1e-5`/`1e-10` contracts | `5bc336436` (cod) |
| 4 | `.168` N-D KDE four-query tile | **REJECT RE-STANDS — 1.118× point estimate, IN-FLOOR** | `344c51020` (cod) |

**Three for four, and the fourth failed exactly where this audit said it would.** §5 called `.168` void as a
cv-kill but explicitly *not* a win, because its own p05 (0.967) sat inside its own null p95 (1.068), and told the
cod lane to relabel rather than re-run. Re-run under the corrected harness it measures 1.118× IN-FLOOR. An audit
that only ever says "this is secretly a win" is not an audit; the prediction that held is the one that says no.

### `.165` — BDF exact-diagonal structured Newton solve: RESURRECTED

Re-implemented (`crates/fsci-integrate/src/bdf.rs`, `enum NewtonFactor`) and re-decided under the
§2 contract with `crates/fsci-integrate/src/bin/perf_bdf_diag_newton.rs`
(`elf_sha256=17f7355509ea7fa9a6117f2474ed2110a230236be5881145e2b19db7146cf3b9`, self-reported by the
binary and equal to the shell-side sha of the shipped file; A/A null and A/B in one invocation, arms
interleaved with per-round alternation, median of per-round ratios, `min_of=3`):

| n | base p50 | cand p50 | ratio_p50 | cand ci95 | A/A null ci95 | gate | bitmism |
|---:|---:|---:|---:|---|---|---|---:|
| 32 | 30.88 ms | 16.16 ms | **1.912×** | [1.802, 2.033] | [0.971, 1.085] | DECIDED | 0 |
| 64 | 44.33 ms | 15.23 ms | **2.908×** | [2.792, 3.028] | [0.968, 1.024] | DECIDED | 0 |
| 128 | 71.51 ms | 10.05 ms | **7.113×** | [7.027, 8.313] | [0.980, 1.020] | DECIDED | 0 |
| 256 | 118.22 ms | 5.20 ms | **22.721×** | [21.964, 22.797] | [0.985, 1.019] | DECIDED | 0 |
| 512 | 1275.41 ms | 11.65 ms | **109.367×** | [107.490, 112.762] | [0.990, 1.007] | DECIDED | 0 |

The ratio grows as `O(n³)/O(n)` exactly as the mechanism predicts, and `.165`'s original 17–19×
claim sits inside this curve between n=128 and n=256 — the rejected entry was not merely decidable,
it was **conservative**.

**The ELF-sha rule paid for itself inside this one session.** Two earlier candidate binaries ran the
`O(n²)` structural scan once per FACTORIZATION (`nlu` = 127 at n=512) instead of once per JACOBIAN
(`njev` = 1). They measured 45.82× and 14.80× at n=512 — both large, both DECIDED, and both quietly
leaving 2.4× and 7.4× on the table. It surfaced only because §2.1 forces the artifact's self-reported
ELF sha to match the binary built from the shipped source, which forced a re-measurement after a
refactor that "obviously could not change performance". Caching the scan on the solver — exactly as
`RadauSolver` already did — is what turns 45.82× into **109.37×**.

Full row, mechanism, and bit-identity argument:
`docs/perf_ledger_cc.md` (2026-07-25). Artifact:
`tests/artifacts/perf/2026-07-25-bdf-diag-newton/bench_stdout_stderr.txt`. Bead `frankenscipy-43vfn`.

**Cost of the void row:** two days, plus two further BLOCKER entries (`.169`, `.170`) spent trying
to satisfy a gate that could not be satisfied, for a lever that measures 45× and was already proven
bit-identical on 2026-07-23.

**Side finding worth its own vein:** `radau.rs` has exploited exactly-diagonal Jacobians since it was
written (it splits `M_3n` into `n` independent 3×3 systems). BDF was the sibling straggler for the
same structural fact. The BDF predicate now calls Radau's `diagonal_jacobian_entries` rather than
duplicating it. **Sibling-straggler audit — where one solver in a family exploits a structural
property and its siblings do not — is a live vein, not an exhausted one.**

---

## 7. Stale-target correction (a second class of ledger rot)

The campaign brief names three per-routine losses for this repo — *"pdist 0.11–0.51×, gaussian_filter 0.35×,
kmeans2 0.41×"* — and asks the cc lane to attack them algorithmically. **All three were already flipped**, and
`docs/GAUNTLET_RELEASE_SCORECARD.md` records the flips:

| Named "loss" | What the scorecard actually records |
|---|---|
| `pdist` 0.11–0.51× | dim-4 SoA SIMD-across-pairs: **1.63–3.87× FASTER** (euclidean/cosine/sqeuclidean/cityblock); wide Chebyshev d=64 **2.33–3.87× faster**; the d=16/d=64 "4.8×/4.4×/3.3× slower" rows are explicitly marked *superseded/closed*. `pdist` small-d is annotated **"already parity (don't chase)"** |
| `gaussian_filter` 0.35× | `8l8r1.132` tile-local cache-blocked separable pass: **1.20× faster** than SciPy, "closes previous Gaussian residual loss". The 3.03×/2.91×/1.34×-slower rows are earlier, superseded candidates |
| `kmeans2` 0.41× | small-k full-scan `nearest_centroid`: **flips 2.78× slower → 1.25× faster**; fixed-shape fused SIMD Lloyd assignment **4.29× faster** at n=2000/k=4/d=4 |

This is the same disease as a VOID reject row, mirrored: **a ledger that records a fix but a scorecard summary
that still quotes the pre-fix number.** Acting on the brief as written would have burned the turn re-deriving
closed work — precisely what the campaign's own HARD GATE ("grep the ledger before you propose a lever") exists
to prevent. Recorded here so the next reader of the brief does not re-spend it.

### …and the correction is itself measured, not just read off the scorecard

`pdist` re-measured on this host, 2026-07-25 (fsci: prebuilt `target/release/perf_pdist_sweep` from 2026-07-16;
scipy 1.17.1 on shape-matched inputs). **CAVEAT: shape-matched, not identical-data, and not paired** — this is a
staleness check, not a claimable ratio:

| shape | fsci | scipy | |
|---|---:|---:|---|
| euclidean / cityblock / sqeuclidean / chebyshev, n=512 d=4 | 0.060–0.114 ms | 0.180–0.423 ms | **3.0–3.7× faster** |
| euclidean / cityblock / sqeuclidean / chebyshev, n=512 **d=16** | 0.731–0.791 ms | 0.543–0.783 ms | **0.74–0.99× — the one real residual** |
| euclidean / cityblock / sqeuclidean / chebyshev, n=512 d=64 | 0.827–0.884 ms | 2.158–2.982 ms | **2.5–3.5× faster** |
| euclidean / cosine, n=4096 d=4 | 11.1–11.5 ms | 54.1–54.9 ms | **4.7–4.9× faster** |
| chebyshev / cityblock, n=2048 d=64 | 4.4–4.6 ms | 39.8–45.6 ms | **8.6–10.2× faster** |

12 of 16 shapes are 2.5–10.2× **faster**, not 0.11–0.51× slower. **The one genuine residual is the d=16 band**
(0.74–0.78×), which falls between the small-d SoA-across-pairs fast path and the wide-d coordinate-lane SIMD
helper — a routing gap, not an algorithmic wall. That is the real next lever in this family, and it is a
different, much smaller target than the brief describes. Queued, not claimed: it needs a same-data paired A/B
under the §2 contract before anyone quotes it.

**Rule 5 (added to §8):** a scorecard row that a later ledger entry supersedes must be edited in place with a
pointer, not left standing next to its own correction.

---

## 8. Standing rules this audit adds

1. **A `cv` ceiling may never be the sole cause of a REJECT.** Record `cv` as provenance; decide on
   whether the paired-median ratio clears the A/A null 95% CI with a 2× margin. Every row killed on
   `cv` alone in this repo's history had a signal that cleared the null.
2. **Every REJECT records the self-time of the frame under test.** Two rows here rejected a lever the
   bench never executed.
3. **A REJECT that supersedes an earlier VOID row must cite it.** Had `23355d1c5` cited the
   2026-07-10 MR4×NR4 pilot, the twelve-day gap would have been visible on the day it landed.
4. **Void-by-criterion is not a win.** Re-run only where the entry's own recorded ratio, taken at face
   value, would clear the decidability rule.

---

## 9. 2026-07-29 incumbent-arm addendum — `solve_ivp_many`

The historical `docs/perf_ledger_cc.md` row claiming **1481–1599x** for a
150-sample Lotka–Volterra ensemble is `VOID-NONULL`: it used separate
invocations, recorded no A/A null, no executable SHA-256, and no counted
mechanism capable of making the missing null irrelevant.

The resurrection produced two different outcomes that must not be conflated:

1. **The stated 150-sample surface remains invalid.** The live SciPy 1.17.1
   arm aborted before timing at `711.439` tolerance units. Tight RK45/DOP853
   reference values attribute the mismatch to FrankenSciPy's sampled RK45
   dense-output path. Retry predicate: fix `frankenscipy-3m5ip`, pass all 150
   samples at `<=100` scaled units, then rerun with dual nulls.
2. **A distinct completion-only phase-2 job passed conformance but its scaling
   sweep is `INVALID-COTENANCY`.** With `t_eval=None`, 128/128 trajectories
   completed and final states agreed within `6.573e-14`. The raw `trj` capture
   ranged from `117.0976x` to `467.7619x`, but seven repositories had been
   directed to the same machine without an exclusive booking protocol. The
   local dual-null calculations cannot identify which process created the
   cross-thread shape. None of those ratios is competitive evidence.

The raw host/cpuset fields are retained in
`tests/artifacts/perf/2026-07-29-solve-ivp-many-vs-scipy-live-arm/bench_stdout_stderr.txt`:
host `threadripperje`, 64 physical cores / 128 logical threads, requested cpuset
per cell, runtime ISA flags, and FrankenSciPy ELF SHA-256
`01b60dc3ad1f0d29c561b2992c00d73caa73cc3a1aac186ca15637d2e898b118`.
They are insufficient under the corrected fleet standard: the run recorded
neither an exclusive `trj-booking` CLAIM/RELEASE window, RAM, actual observed
worker threads distinct from requested CPUs, nor the SciPy engine-artifact
SHA-256.

**Concrete retry predicate:** honor queue position five. After frankensearch,
frankenpandas, frankenfs, and frankenredis release `trj`, post
`[trj] CLAIM frankenscipy`, confirm the machine is quiet, and rerun the exact
`1/2/4/8/16/32/64/128` completion sweep. Every row must include requested CPUs,
affinity, actual observed workers for both engines, host/core/thread/RAM/NUMA
identity, both engine SHA-256s, independent A/A controls, and a
bootstrap-median CI decision with a 2x null margin. Post
`[trj] RELEASE frankenscipy` even if the rerun fails.
