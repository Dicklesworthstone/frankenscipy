# RUN STATUS 2026-07-31 — BLOCKED on host quiescence, not on analysis

Companion to `PREREGISTERED_mechanism.md` (committed before any timing) and
`docs/LEDGER_RESURRECTION.md` §10. **No timed sample of the primary effect has
been taken.** This file records why, with the evidence, so the next sweep
inherits a diagnosis rather than a silence.

## What is ready

Everything except the measurement.

| Item | State |
|---|---|
| Pre-registration | Committed before the harness, `2d8bc677f` / §10 |
| Harness | `crates/fsci-stats/src/bin/perf_truncweibull_scipy.rs`, committed `85dbdaf6e`, source SHA-256 `353575cefdf7c20d090e825fde630c4e40f40b84611c47726c92ee2f9e823bdf` |
| Build | `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec --base eb2d4945765d5d9c0afaf882847194ff7b7d8af8 --clean-overlay --no-overlay -- cargo build --profile release-perf -p fsci-stats --bin perf_truncweibull_scipy --features stats-incumbent-bench`; remote worker `hz1`, exit 0 in 116.3 s |
| Executed ELF | SHA-256 `70e54101272f019ff7e0a1a988588ef2454c2ee8c5b85962c987de9af3a7b1b8`, 6,384,464 bytes; hash matched on `hz1` before scp and at `/data/tmp/frankenscipy-truncweibull-70e54101/` after, and self-reported from inside the process on every attempt |
| Live SciPy arm | 1.17.1 from the pinned incumbent tree `/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages` (numpy 2.4.6) |
| Exclusive booking | `[trj] CLAIM frankenscipy` agent-mail message **7749** |
| Plumbing | Verified end-to-end by a non-evidence smoke (`7 50`): provenance, ELF self-hash, affinity, hardware and frequency-policy reporting all emit correctly before the gate refuses |

## Why it has not run

`perf_truncweibull_scipy.rs:734` `require_host_wide_quiescence` fails closed
unless **every** CPU is under `HOST_QUIESCENCE_MAX_BUSY = 0.20` across a 400 ms
sample, taken both before and after timing. §10 registered the same requirement
in prose ("execute on an exclusive, quiescent host"), so it is a pre-registered
condition, not an implementation detail — it cannot be relaxed to obtain a
number.

**126 attempts across ~50 minutes on `thinkstation1`, 0 passes.**

| CPUs over 20% at refusal | attempts |
|---:|---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 2 |
| 4 | 8 |
| 5–13 | most of the remainder |
| 26, 52 | 1 each |

The floor is 1, never 0. Host load average over the window ranged 8.9 → 23.4.
The traffic is **not** from `frankenscipy`: it is co-tenant agent swarms in
`franken-networkx` (a `contended_claim_run.py` holding ~101% of a core) and
`frankensearch` (local `cargo test` binaries plus `rch exec` jobs). A cross-repo
courtesy note went to the `franken-networkx` pane (message 7757) asking whether
its contention is deliberate — its script name suggests it may be measuring
*under* contention, in which case pausing it would corrupt its experiment and
should not be requested. No reply at time of writing.

One co-tenant thread pinned to one core is sufficient to hold this gate shut on
a 64-CPU box, which is the whole mechanism: the gate is calibrated for the
exclusive-booking regime of `threadripperje`, where prior `trj` rows were taken.

## Alternatives considered and rejected

- **Relax or bypass the gate.** Refused. It is pre-registered and it is the
  reason a number from this harness would be worth anything.
- **Measure on the quiet rch worker `hz1`** (8 vCPU, load average 0.98).
  Rejected on two independent grounds: its root filesystem is at **97% (7.7 GB
  free)**, and installing SciPy + NumPy there risks the build capacity 11
  repositories depend on; and it runs Python 3.14.4, so the pinned 3.13
  incumbent tree is not importable and a fresh SciPy would not be the pinned
  1.17.1 artifact the other rows were measured against.
- **Wait longer.** Adopted. The retry loop remains running at ~1 attempt / 8 s
  and will write `run_SUCCESS.txt` if a window opens.

## What was discharged anyway

The pre-registration's commitment — *"Whatever the ratio, `CHANGELOG.md:75` and
`:215` get corrected to say what the number is measured against"* — did **not**
depend on the measurement, because the ambiguity is a property of the sentence.
Both lines now name the comparator and state explicitly that the ~370× is a
self-speedup against our own deleted Simpson quadrature and **not** a claim
against `scipy.stats.truncweibull_min` (`0836e3665`). The public exposure is
closed; only the vs-SciPy ratio is outstanding.

## Standing conclusion for the campaign

This is the concrete reason seven pre-registered conversions
(`minimize_many`, `tplquad_many`, `curve_fit_many`, `root_many`, `quad_many`,
the normality screen, `newton_many`) have harnesses and zero measurements. The
bottleneck is **not** analysis, harness construction, or gate design. It is that
the fleet's evidence gate presumes an exclusive host, while the host is shared
by agent swarms in at least three repositories with no cross-repo booking
mechanism — `trj` claims are project-scoped agent-mail messages, and the
agent-mail Product Bus that would carry a cross-project claim is disabled
(`WORKTREES_ENABLED` unset). Until a cross-repo booking exists, or measurement
moves to a dedicated quiet host with the pinned incumbent installed, the
conversion queue cannot drain regardless of how many rows are pre-registered.
