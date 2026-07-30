# Corrected A/A null gate: adopted, and every row re-certified

Agent `cc/NobleCedar`, 2026-07-30. Follows the audit in `AUDIT.md` (`48e2b98e9`).

## 1. Diagnostic first, as required — my gate does NOT have the defect

Already run and committed at `48e2b98e9`: same ELF
`f98a82f228046e45dd53754e6a5a9d21668e435835753001a819fb1831b565ed`, three
pinnings, `lsqr`, 21 rounds. This turn's re-certification ran at load average
**5.30**, down from ~12 during the original sweeps.

| side | cpu63 (core31) | cpu31 (**SMT sibling of core31**) | cpu15 (core15, independent) | spread | verdicts |
|---:|---:|---:|---:|---:|---|
| 16 | 12.0172 | 12.1793 | 12.2478 | 1.92% | WIN/WIN/WIN |
| 32 | 3.9272 | 3.8470 | 3.9305 | 2.17% | WIN/WIN/WIN |
| 48 | 2.2963 | 2.4399 | 2.3059 | 6.25% | WIN/WIN/WIN |
| 64 | 1.7910 | 1.7215 | 1.6081 | 11.37% | WIN/WIN/WIN |
| 96 | 1.2936 | 1.0099 | 1.0091 | 28.19% | WIN/INDET/INDET |

**Where effects reproduced, verdicts were stable.** The one verdict that moved
(side 96) moved because the **effect** moved: our own arm went 217.31 → 285.46 ms
(**+31.4%**) while SciPy moved 2.7%. The defect's signature is a *reproducible
effect with a moving verdict*; this is a *moving effect with a tracking verdict*,
which is a gate working. Structurally there is also no straddle clause to have
the defect: null CIs entered only through a margin term, and a **tighter** null
always **lowered** the bar.

So by the stated conditional the gate would be left alone. It is nonetheless
**adopted** below, because the corrected rule is the fleet standard and clause 3
is something my gate genuinely lacked.

## 2. What adoption actually changes here

Implemented in `perf_sparse_vs_scipy.rs`:

- **c1** effect CI excludes 1.0
- **c2** effect deviation > 2x the **larger null half-width**
- **c3** each null **median** within **2%** of 1.0  ← the clause I lacked
- **c2b** *retained*: effect deviation > 2x the largest null CI **endpoint**
  deviation from 1.0 — this harness's original margin

Null CIs are now printed as telemetry and **never veto**; a new
`corrected-null-gate:` line reports every clause, both margins, and both null
medians.

### Why c2b is retained — c2 alone would be a loosening

c2 uses the null **half-width** `(high-low)/2`; c2b uses the largest **endpoint**
deviation `max|endpoint-1|`. For any null CI offset from 1.0 the endpoint
deviation strictly exceeds the half-width, so **c2b is strictly stricter than
c2**. Measured across all 23 audited cells:

> **c2's threshold is looser than c2b's in 23 of 23 cells (100%).**

Examples: cpu63 side 16 requires 1.0125 under c2 versus 1.0213 under c2b;
cpu31 side 96 requires 1.0818 versus 1.2074. Adopting c2 *in place of* c2b would
have relaxed the margin on every row I have ever measured. The standard warns
that a gate change producing wins is a loosening, so both clauses are conjoined
rather than substituted. **c2b is a local addition and not part of the fleet
rule** — flagged for whoever owns the standard, and trivially removable if the
intent was for c2 to be the sole margin.

## 3. Re-certification of all 23 rows

Computed from the committed telemetry each gate consumes (null medians, null
CIs, effect CIs), so it is exact rather than a re-measurement.

| outcome | count |
|---|---:|
| cells re-certified | **23** |
| unchanged | **21** |
| **previously-vetoed rows that became decidable** | **0** → **0 WIN, 0 LOSE** |
| previously-decidable rows now vetoed by c3 | **2** |

**0 rows became decidable, because none had ever been vetoed by a null CI** —
there was no straddle clause to liberate them from. This is the structural
difference from frankenlibc, whose 7 wordexp cases were being actively
suppressed. The corrected rule cannot free rows in a harness that was not
imprisoning any.

The two rows c3 removes:

| run | method | side | ratio | was | now | `|o_med-1|` | `|s_med-1|` |
|---|---|---:|---:|---|---|---:|---:|
| cpu15 (independent core) | lsqr | 16 | 12.2478x | WIN | **INDETERMINATE** | 0.51% | **2.02%** |
| cpu15 (independent core) | lsqr | 64 | 1.6081x | WIN | **INDETERMINATE** | **3.17%** | 0.18% |

**No previously-reported headline changes.** Both are *replication* cells from
the cpu15 sweep, already labelled `PROVISIONAL_NON_EXCLUSIVE`, and side 64 was
already marked not-quotable pending an exclusive host. Every primary cell
(cpu63) and every `threadripperje` GMRES/BiCGSTAB row passes c3 — worst null
median bias among them is 0.29%.

c3 earns its place on cpu15 side 64: it independently flags a cell whose ratio
drifted 11.37% across pinnings, which my margin form did not catch. That is the
clause doing real work.

### The one place c3 is over-tight, now binding on real data

cpu15 side 16 is a **12.2478x** effect vetoed for a **2.02%** null median bias —
0.02 percentage points over the threshold. The bias is 0.18% of the effect
deviation. This is the same precision-independent-bound problem in a milder
form: a fixed 2% bound is not effect-size aware, so it can veto a 1125% effect.
c2b handles this cell correctly (required 1.0508, effect clears it easily).
Adopted as specified regardless; recommending a relative form — bias as a
fraction of effect deviation, with an absolute ceiling for genuinely broken arms.

## 4. Integrity check

frankenlibc's: under their fix all 7 wordexp cases became decidable and **all 7
LOST**; a fix that suddenly produces wins was a loosening.

**My result: the change produced ZERO new wins.** It produced zero new decidable
rows of any kind and *removed* two wins. The movement is entirely in the
conservative direction, which is the correct signature. Had adoption manufactured
wins I would distrust it and say so; it did the opposite.

Their exact pattern cannot be reproduced here for the structural reason in §3 — I
had no suppressed rows to liberate — so the analogous signal is the direction of
movement, and it is strictly toward fewer claims. Note also that the *latent*
loosening in c2 (100% of cells) flipped no verdict only because every effect
clears both margins by 44x–582x; on a marginal effect it would have mattered,
which is precisely why c2b is retained.

## Reproduction

`recertify.py` in this directory recomputes both gates from any set of harness
logs and prints the clause-by-clause table, the decidable/vetoed tally, and the
c2-versus-c2b looseness count.
