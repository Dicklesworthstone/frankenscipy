# PRE-REGISTERED: converting the only numeric win claim on our public surface

Written and committed **before any `TruncWeibullMin` timing exists**. Author: cc
pane (BlackThrush). Date: 2026-07-31.

## Why this claim, ahead of the 1481× ones

`docs/KEEP_CLAIM_GATE_AUDIT.md` (commit `2d8bc677f`) ranked the conversion queue
by *where a user could act on the claim*, not by magnitude. That put this one
first, and it is not close: across every document the README's Documentation Map
links as public, there are **7 numeric speed claims, and exactly one asserts a
win of ours**:

> `CHANGELOG.md:213` — `TruncWeibullMin.mean/var` quadrature replaced with
> closed-form incomplete-gamma:
> `E[X^k] = e^{a^c}/s · Γ(k/c+1) · (P(k/c+1, b^c) − P(k/c+1, a^c))`,
> **~370× faster** and structurally exact

Quoted twice (`:75` and `:215`). A user reads the changelog to decide whether to
adopt.

## The problem with the claim as written

**The ~370× is a self-speedup.** Read in context, it is "replaced [our Simpson
quadrature] with [a closed form], ~370× faster" — faster than *our own previous
implementation*, not than SciPy. The line does not say so. A reader encountering
"~370× faster" in a SciPy-port changelog will reasonably infer "~370× faster
than SciPy", which is a claim nobody has ever measured.

This is the exact failure mode the fleet policy targets, sitting on the single
most load-bearing line we publish.

## An incumbent exists — this is not a gap-fill

Established before pre-registering, so it is a fact and not a prediction:
`scipy.stats.truncweibull_min` is present in SciPy 1.17.1 and both moments are
callable. Timed casually at `c=2.0, a=0.5, b=3.0`: `mean()` ≈ 42.3 µs/call,
`var()` ≈ 64.2 µs/call. So this claim falls in the audit's
*nobody-has-measured-it* bucket, **not** the *no-incumbent-exists* bucket. It is
convertible.

## Predictions

**P1 — we win, but by far less than 370×.** The 370× was measured against our
own Simpson quadrature, which was doing hundreds of PDF evaluations. SciPy's
`truncweibull_min.mean()` is not doing that: it has its own `_munp` path. Predict
the gated same-invocation ratio lands in **[3×, 60×]**, point estimate **15×**.
Falsified if it exceeds 60× — which would mean the public number is accidentally
defensible — or if it falls below 3×.

**P2 — we do not lose.** Predict the ratio exceeds 1.0 at every parameter point
screened. This is the prediction that actually matters for the changelog: if it
is falsified, the public line is not merely unqualified, it is wrong-signed and
must be corrected immediately.

**P3 — the mechanism is SciPy's generic-moment dispatch, not arithmetic.**
Predict that ≥ 60% of SciPy's per-call time is `rv_continuous` machinery —
argument broadcasting, `_argcheck`, the generic `_munp`/`integrate` path — rather
than the incomplete-gamma evaluation itself. Same shape as the finding in
`726cf5a20`, where SciPy's ODE cost was 89% driver rather than kernel. Tested by
comparing against a direct call to the private `_munp`/`_stats` path where one
exists.

**P4 — the advantage is roughly flat in `(c, a, b)`.** Both sides evaluate a
closed form with no size parameter, so predict the ratio varies by less than 3×
across a screen of at least six parameter points spanning narrow and wide
truncation windows. A large spread would mean one side has a regime-dependent
fallback, which is a separate finding.

**P5 — conformance holds where SciPy is itself correct.** Predict agreement to
`≤ 100` scaled tolerance units at every screened point. Recorded because the
changelog also calls the identity "structurally exact", which is a *correctness*
claim riding alongside the speed claim, and the audit found the ledger already
carries known cdflib-adjacent SciPy defects; if SciPy is wrong at some point,
that is reported as a parity note rather than as our failure.

## Method

A new same-invocation live-arm cell for `fsci-stats` — none exists yet, so the
harness is part of this work. Standing requirements: corrected null gate
including its median clause, **actual observed** worker threads for both arms,
host identity, ELF SHA-256 self-reported in-process, and `rch exec --base
<sha> --clean-overlay` so no co-tenant edits enter the binary. Construction and
conformance checks outside timing.

## What happens to the changelog line either way

Not a prediction, a commitment. Whatever the ratio, `CHANGELOG.md:75` and `:215`
get corrected to say what the number is measured against. If the gated
vs-SciPy ratio is respectable, the line should quote *that* instead. If the only
honest number is the self-speedup, the line must say "faster than our previous
quadrature". **No public number stays ambiguous once measured.** The claim is not
weakened in this pass — it is qualified, which is a different act.
