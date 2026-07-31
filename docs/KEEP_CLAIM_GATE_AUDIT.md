# KEEP-claim gate audit — how much of our claimed ground rests on a live incumbent?

Run 2026-07-30 by the cc pane (BlackThrush), unprompted by any specific defect,
following the fleet policy that **a perf KEEP requires a vs-incumbent ratio**.
Reproduce with `python3 scripts/keep_claim_gate_audit.py` (read-only).

**Inventory only. No claim was deleted, weakened, or reworded in this pass.**

## Re-run 2026-07-31 05:47 — the number did not move, and one bucket was wrong

Re-run at the top of this session on `eb2d49457`. **Identical partition: 481 /
14 / 467.** Two things are worth stating plainly rather than burying:

1. **Nothing converted in the intervening 30 hours.** Seven more conversions
   were *pre-registered* in that window — `minimize_many`, `tplquad_many`,
   `curve_fit_many`, `root_many`, `quad_many`, the normality screen, and
   `newton_many` — each with a committed `PREREGISTERED_MECHANISM.md` and in
   four cases a full harness binary. **Zero were measured.** Pre-registration is
   the cheap half; it moves no number. The one high-magnitude conversion that
   did complete in the window (`solve_ivp_many` exclusive rerun, `fdc0d818b`)
   landed as **MIXED**, not KEEP — a 464.4884× whole-job win with its
   pre-registered post-16 thread-cap mechanism falsified — so it never entered
   the KEEP claim base and the gated count could not move.

2. **The "cannot be converted" bucket is not 2. It is 0.** Hand-checking both
   rows the classifier put in `NO_INCUMBENT_EXISTS` shows both are
   misclassified, in opposite directions:

   - `docs/NEGATIVE_EVIDENCE.md:3014` — `signal.autocorrelation` 14–509×. The
     row says "no scipy peer", which is what the regex caught, but its own
     MEASURED line reads *same-binary A/B*. The 14–509× is a **self-speedup**,
     so it is not an unconverted incumbent claim at all; it belongs in the 208.
     And the no-peer statement is about the *function name*: SciPy computes the
     same quantity via `scipy.signal.correlate(x, x)` / `numpy.correlate`, so an
     incumbent arm exists in substance.
   - `docs/NEGATIVE_EVIDENCE.md:17330` — `expm_many`/`logm_many`/`sqrtm_many`/
     `cosm_many`/`sinm_many`. The body says "scipy has no batched matrix-function
     API", but the row's own **headline** is "3.6-48x faster than a
     single-threaded scipy loop". The incumbent was built, pinned
     (`OPENBLAS_NUM_THREADS=1`) and run. This is `RATIO_NOT_SAME_INVOC`.

   So the honest split of the 467 is **0 unconvertible / 467 unmeasured**, and
   the "we're benchmarking something SciPy doesn't have" defence survives in
   exactly zero rows out of 481. The classifier's `NO_INCUMBENT_EXISTS` rule
   (an explicit no-peer phrase plus no vs-incumbent ratio) is kept as-is rather
   than tuned to this finding, because tuning a classifier to its two known
   failures is how you stop finding the third.

## The number

> **We hold 481 KEEP claims. 14 (2.9%) carry a vs-incumbent ratio measured with
> the incumbent live in the same invocation. 467 (97.1%) do not.**
>
> **Of the 467, zero cannot be converted for want of an incumbent. All 467 are
> simply unmeasured.** After the two reclassifications above, **258** have a real
> vs-incumbent ratio somebody ran, just not in the same invocation, and **209**
> have no incumbent comparison of any kind.
>
> For calibration against the peer repo that set this standard: frankenfs found
> 67 of 186 (36.0%) with no ratio. Our comparable figure — claims with no
> incumbent comparison at all — is 209 of 481 (43.5%). **We are worse than
> frankenfs on the metric it published, and far worse on the stricter
> same-invocation metric.**

De-duplicated, it is slightly worse than it looks and slightly better than it
reads: 4 of the 14 are the same claim recorded in both ledgers, so there are
**10 distinct gated claims**. All 10 date from 2026-07-28 or later, which is
exactly when the same-invocation live-arm harness came online. Nothing before
that date was ever gated, because there was nothing to gate with.

Full partition:

| bucket | count | share | meaning |
|---|---:|---:|---|
| `GATED_SAME_INVOCATION` | 14 | 2.9% | incumbent ratio, incumbent live in the same invocation |
| `RATIO_NOT_SAME_INVOC` | 257 | 53.4% | a real vs-incumbent ratio, but the incumbent was run separately or quoted historically |
| `NO_INCUMBENT_RATIO` | 208 | 43.2% | self-speedup / vs-serial / byte-identical N× only |
| `NO_INCUMBENT_EXISTS` | 2 | 0.4% | genuinely nothing to compare against |

By unit: 450 ledger entries (`docs/NEGATIVE_EVIDENCE.md` `##` entries +
`docs/perf_ledger_cc.md` `###` sections) and 31 scorecard table rows. The 31
scorecard rows are per-workload cells of roughly a dozen underlying levers, so
the "claim" count is inflated relative to "distinct levers" — the *ratio* is the
meaningful figure, not the absolute.

### The two numbers are different problems

- **208 claims have no incumbent ratio at all.** These are self-speedups. Under
  the policy they are maintenance, not campaign wins. This is the same class the
  ledger already corrected once for three BDF rows
  (`docs/perf_ledger_cc.md:3560`); that correction was never generalised.
- **257 claims have a real incumbent ratio but not a same-invocation one.**
  These are *not* unsupported in the way a self-speedup is. Somebody did run
  SciPy. They were just run in a separate process, without an A/A null and
  usually without an executed-binary SHA-256, so they cannot exclude
  cross-invocation drift. Converting these is re-measurement, not new science.
- **Only 2 claims cannot be converted because no incumbent exists.** This is the
  most useful surprise in the audit: the "we're benchmarking something SciPy
  doesn't have" excuse is almost never actually true. Even the batched `*_many`
  gap-fills have a legitimate incumbent arm — a Python loop over the scalar
  SciPy function — and most of them already quote one. Only 2 rows genuinely
  have no peer.

## Public exposure: much smaller than the ledger count suggests

The ranking key the audit was asked for is *how load-bearing a claim is where a
user might act on it*. Scanning every document the README's Documentation Map
links as public:

| public doc | numeric speed claims | asserts a FrankenSciPy win? |
|---|---:|---|
| `README.md` | 4 | **No** — see below |
| `CHANGELOG.md` | 2 | **Yes** — one claim, quoted twice |
| `AGENTS.md` | 1 | No (describes a third-party tool's flag) |
| `FEATURE_PARITY.md`, `PROPOSED_ARCHITECTURE.md`, `COMPREHENSIVE_SPEC…`, `PLAN_TO_PORT…`, `EXISTING_SCIPY_STRUCTURE.md`, `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 0 | — |

**Total: 7 numeric speed claims across the entire public surface, of which
exactly one asserts a win of ours.**

The README's four are not our claims at all. `~3-10× faster on large dense
linalg` is a statement about *BLAS FFI*, in a table arguing why we deliberately
do not link it. `Within ~2-3× on uncontested machines` and its follow-up
concede that SciPy wins large dense linalg. The FAQ answer to "Is this faster
than SciPy?" already reads *"mileage varies by problem size, conditioning, and
structure."* The public entry point is, on this evidence, more conservative than
the ledger behind it.

`docs/GAUNTLET_RELEASE_SCORECARD.md` carries **220** numeric claims and its name
implies publication, but it is **not linked from the Documentation Map**, so it
is an internal artifact today. It is the single largest concentration of
un-gated numbers in the repo and would become the top exposure the moment
anything links it.

## Ranked conversion queue

Ordered by where a user could act on the claim, then by magnitude.

**Tier 1 — public, load-bearing, convert first (1 claim).**

1. `CHANGELOG.md:75` and `:215` — incomplete-gamma closed-form identity,
   **~370× faster**. This is the only numeric win claim on our public surface.
   A user reads the changelog to decide whether to adopt. It needs a
   same-invocation live-SciPy arm on the exact identity and input regime named.

**Tier 2 — the marquee cluster: biggest numbers, mechanically convertible (7).**
Not public today, but these are the figures most likely to be quoted, and they
share one harness pattern so the marginal cost per conversion is small:

2. `solve_ivp_many` — 1481× / 1599× — *already in flight* under a committed
   pre-registration (`31b13ba7c`), unblocked by fixing frankenscipy-3m5ip.
3. `minimize_many` — 271–275×
4. `dblquad_many` — 62.7–211×
5. `tplquad_many` — 83–159×
6. `curve_fit_many` — 113×
7. `quad_many` — 14.5–61×
8. `root_many` — 11–25×

All seven are `RATIO_NOT_SAME_INVOC`: each was measured "SAME-BOX head-to-head"
against a Python loop over its SciPy peer, so an incumbent arm demonstrably
exists and the conversion is re-measurement under the gate. `solve_ivp_many` is
already marked `VOID-NONULL`; the other six sit in the same evidential position
and are **not** marked, which is the live inconsistency.

**Tier 3 — the 208 self-speedups.** These need a decision, not a measurement:
either attach an incumbent arm or relabel them as maintenance. Recommended
default is relabel, because most are byte-identical internal optimisations where
the incumbent comparison was never the point. Cheap and honest; the precedent
already exists at `docs/perf_ledger_cc.md:3560`.

**Tier 4 — the remaining ~250 `RATIO_NOT_SAME_INVOC` rows.** Low individual
value, high volume, mostly old and small. Bulk-editing them would destroy
provenance for little gain. The obligation here is forward-looking: stop new
claims entering un-gated, which the ledger-preflight hook already enforces for
new rows.

**Cannot be converted (2).** Two rows genuinely have no incumbent. They should
be labelled `NO_INCUMBENT_EXISTS` rather than left reading as unmeasured, since
that is a different problem from nobody having gotten around to it.

## Methodology, including what it gets wrong

The classifier is regex-based over ledger prose and is therefore approximate. It
was validated against hand-checked cases at both ends before publication, and
**two real faults were found and fixed during that validation**, both of which
had moved the headline number materially:

1. `[x×]\b` never matches `4.05× faster` — `×` is a non-word character, so a
   trailing `\b` fails before a space, while `4.05x faster` matches. This
   silently misclassified every claim written with the `×` glyph, which is most
   of the older ledger, and had put 70 claims in the wrong bucket.
2. Rows the ledger *itself* labels `**SELF-SPEEDUP**` were being counted as
   gated wins because their prose mentioned live-arm context. Three BDF rows had
   inflated the gated count. The verdict region, not the whole body, decides.

Residual known limitations, stated so the number is not over-trusted: prose
markers are a proxy for evidence rather than the evidence itself, so a row could
say "same-invocation" without an A/A null and still be counted gated; the 481
figure double-counts claims recorded in both ledgers and counts scorecard cells
individually; and `NEGATIVE_EVIDENCE.md` spans every lane and agent, not just
this one, so this is the repo's whole claim base rather than one pane's.

The direction of the headline is not sensitive to any of these. Even the most
generous reading — count every `RATIO_NOT_SAME_INVOC` row as adequately
supported — leaves 208 self-speedups, 43% of the claim base, with no incumbent
comparison at all.
