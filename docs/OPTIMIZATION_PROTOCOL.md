# Optimization Protocol

The decision rule for every performance change in FrankenSciPy.

**Rewritten 2026-07-25.** The previous version told you to save a Criterion baseline and flag a
regression when p50 moved more than 5%. That rule has no null control, so it cannot tell a real 4%
effect from the machine being 4% busier — and its companion `cv < 5%` KEEP gate is *unreachable* on
this hardware (measured floor ~12%; a 20× longer sample moves `cv` about 4 points). Between
2026-07-10 and 2026-07-24 that gate rejected four frontier candidates whose measured effects were
**1.20×, 1.47×, 3.82× and 17–19×** against A/A nulls of 1.00–1.02. Three of the four were later
re-measured under the rule below and **kept**. See `docs/LEDGER_RESURRECTION.md`.

**`cv` is provenance. It is never a verdict.**

---

## The gate

A speedup claim is **DECIDED** if and only if:

> the candidate's **95% percentile-bootstrap CI lower bound on the median of per-round ratios**
> (10,000 deterministic resamples) exceeds
> `1 + 2 × (null_edge − 1)`, where `null_edge` is the worse side of the A/A null's own 95% CI.

Everything else is `IN-FLOOR` / `NOT DECIDED`. Nothing below ~1.01× is decidable on this hardware;
do not claim it. A `NOT DECIDED` result is a real result — ledger it with a retry predicate.

---

## The seven requirements

1. **Profile first, and name the frame.** A lever must be attributed to a frame with **≥0.1%
   self-time in a profile of the workload the bench actually runs**. `perf record --call-graph=dwarf`,
   then quote the percentage in the ledger row. *Two entries in this repo's history rejected a lever
   the benchmark never executed.*

2. **Prove behaviour first, time second.** Compare full results as **raw bits** (`to_bits()`), not
   with a tolerance, unless the change is explicitly under a documented tolerance contract. Abort the
   run on any mismatch — a timing number from a diverging candidate is not evidence of anything.

3. **Execution proof.** The candidate arm must *demonstrate* it ran: a hit counter, a checksum that
   differs between arms, anything falsifiable. `hits == 0` in the candidate arm means the A/B measured
   nothing. Assert it in the harness, not in your head.

4. **Same binary, both arms, one invocation.** Switch arms with a `#[doc(hidden)] pub static
   …: AtomicBool` (see `BDF_FORCE_DENSE_NEWTON`, `RADAU_FORCE_PER_ITER_ALLOC`). Whole-binary A/Bs
   (ISA flags, LTO, allocator) are the exception and need §6 below.

5. **A/A null in the same invocation.** Run `paired(base, base)` and *then* `paired(base, cand)`,
   both with arms **interleaved inside each round** and the order **alternating per round**. Report
   both. The cost is exactly 2× wall time and it is the entire price of being allowed to believe your
   own numbers.

6. **Self-reporting ELF SHA-256, line 1.** The binary hashes its own `env::current_exe()` and prints
   it. A shell-side hash *next to* the run proves nothing about which ELF executed, and rch compiles
   into an opaque per-worker target dir you cannot predict. **This is not ceremony:** on 2026-07-25 it
   caught a refactor that "obviously could not change performance" and was in fact costing 7.4×.
   Whole-binary A/Bs additionally require: same worker for both arms, ELF shas that *differ* (proving
   the flag reached the compiler), and a verdict on wall/cycles — never on instruction count.

7. **`min_of` beats long samples.** Take the **minimum of `min_of` inner replicates** per sample.
   Lane default: **`min_sample = 2 ms`, `min_of = 3`**. Beyond ~10 ms a longer sample buys nothing and
   40 ms can be *worse* — a longer sample is a bigger target for preemption.

---

## Reference implementations

- `crates/fsci-conformance/src/perf_gate.rs` — `paired()`, `decide()`, `Paired`, `elf_sha256()`.
  Use these rather than hand-rolling; a hand-rolled gate is how the `cv` rule survived for weeks.
- `crates/fsci-integrate/src/bin/perf_bdf_diag_newton.rs` — a complete worked example: ELF sha,
  bit-identity gate, execution-proof counters, A/A-then-A/B, median-CI verdict, a negative-control
  arm proving the fast path's predicate is free, and an arm-isolated mode for profiling (the paired
  A/B *cannot* be profiled — both arms share a process and the slower one swamps the samples).

---

## Machine hygiene

Pin the process (`taskset -c <cpu>`). Record host, load average at start **and end**, worker id, and
the toolchain. **Cross-worker comparison is invalid, always** — ~2× worker variance has made a 10×
win look like a 1.3× loss in this fleet. Under the campaign allocation, only Lane M holds measurement
rights; if you are in Lane B or L and need a number, ask for a window on the campaign thread rather
than taking a slot.

---

## Ledgering

Every result — win or loss — gets a row. **KEEP → `docs/perf_ledger_cc.md`** (cc lane) or
`docs/NEGATIVE_EVIDENCE.md`; **REJECT → `docs/NEGATIVE_EVIDENCE.md`** and
`docs/progress/perf-negative-results.md`. Never delete a row.

A row records: hypothesis · profile attribution (sample count, % self-time) · the ONE lever ·
behaviour proof · A/B **and A/A null** with worker id and binary sha · verdict · **a concrete retry
predicate**.

A retry predicate is a *testable condition*, never "later" or "if it seems important":

> retry only if (1) `decode_bitmap_payload` exceeds 5% exact self-time in a symbolized profile, AND
> (2) the A/A null floor on the target worker is below 1.02×.

**Grep both ledgers before proposing a lever.** Agents in this repo have re-derived already-rejected
levers within a single turn, and two rows rejected in July had already been shipped by a different
agent twelve days earlier.

### Enforced ledger preflight

`scripts/ledger_preflight.py` is installed in this checkout's chained pre-commit
hook as `hooks.d/pre-commit/40-ledger-preflight`; invoking it with no arguments is
the hook entry point. It reads the Git index rather than the worktree and checks
every newly staged row:

- a REJECT without either measured same-invocation A/A values or a counted
  mechanism exits **2 / BLOCKED**;
- a cv-only REJECT exits **2 / BLOCKED**;
- a KEEP without a 64-hex SHA-256 identified as the executed ELF/binary exits
  **2 / BLOCKED**.

Historical rows are not re-litigated merely because their file is staged. For a
fresh checkout using the Agent Mail hook chain, install the checked-in gate ahead
of the reservation guard:

```bash
ln -s ../../../../scripts/ledger_preflight.py \
  .git/hooks/hooks.d/pre-commit/40-ledger-preflight
```

Run proposal preflight with both the lever and its target surface:

```bash
scripts/ledger_preflight.py \
  --propose "persistent worker pool for trailing update" \
  --surface "fsci-linalg cholesky"
```

Every matching row prints its retry predicate (or `NOT_RECORDED`). Exit 2 means
a sound prior rejection already covers the proposal; state the satisfied retry
predicate and the materially different mechanism before proceeding.

### Two failure modes this protocol exists to prevent

- **A `cv` ceiling may never be the sole cause of a REJECT.** Every row killed on `cv` alone in this
  repo's history had a signal that cleared its own null.
- **A REJECT that supersedes an earlier row must cite it**, and a scorecard row that a later ledger
  entry supersedes must be edited in place with a pointer. The campaign brief for this repo listed
  three "losses" (`pdist`, `gaussian_filter`, `kmeans2`) that had all already been flipped to wins.

---

## Conformance is not optional

Numerical stability contracts outrank speed. Before any KEEP:

```bash
cargo test -p <crate>                       # the crate's own suite
cargo test -p fsci-runtime --test differential_metamorphic
cargo test -p fsci-conformance              # scipy-facing parity
cargo fmt --all -- --check && cargo clippy --workspace --all-targets
```
