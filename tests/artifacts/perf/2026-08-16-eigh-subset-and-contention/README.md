# eigh measurement logs, 2026-08-16 (RainyPrairie)

Raw output of every `perf_eigh_vs_scipy` invocation behind the eigh rows banked
in `docs/NEGATIVE_EVIDENCE.md` on this date. Each ledger row quotes numbers from
one of these; the logs are here so a reader can check the quote against the full
run rather than taking the row's word for it, and so the discarded cells are
visible alongside the reported ones.

Every log carries, in its own first lines, the `elf_sha256` the binary
self-reported from `/proc/self/exe`, the worker host, the runtime ISA, the
cpuset, and (from `eigh_gated.log` onward) `loadavg_pre` / `loadavg_post`.

| log | worker | what it was for | outcome |
|---|---|---|---|
| `eigh_impl_sweep.log` | hz2 | first implementation x size sweep | the crossover claim, later WITHDRAWN |
| `eigh_run2.log` | hz2 | replication at 256/512 | one cell self-voided (`nulls=FAIL`) |
| `eigh_scaling.log` | hz2 | 512/768/1024, one implementation | refuted "the ratio grows with n" |
| `eigh_crossgrid.log` | vmi1227854 | finer grid, 6 of 10 cells | too noisy to place a crossover; died on rch's 1800s SSH ceiling |
| `eigh_hz2_lower.log` | hz2 | replication of the lower grid | n=512 ordering REVERSED; crossover claim withdrawn |
| `eigh_paired.log` | vmi1149989 | first paired-implementation attempt | VOID: stale binary, no `IMPL` line (ozg54) |
| `eigh_gated.log` | vmi1227854 | paired design, first fresh run | undecided; worker at 181% then 227% load |
| `eigh_contention.log` | vmi1149989 | cross-arm contention probe | inconclusive; drift exceeded the effect |
| `eigh_cert.log` | vmi1293453 | certification on a quiet worker | scipy rows VOID under the margin gate; contention ~11% in the one drift-clean cell |
| `eigh_place2.log` | vmi1293453 | CPU placement / SMT / MHz capture | no SMT, 3195 MHz flat, arms unpinned |

## Why the void runs are kept

Four of these ten produced nothing reportable — a stale binary, a self-voided
null, a run that ran out of SSH time, and a pair of cells that could not resolve.
They are kept deliberately. A directory containing only the runs that worked
would misrepresent how much of this measurement effort certified, and two of the
findings that mattered most this day — that `rch` served a binary older than
mainline HEAD, and that a passing A/A null does not license a comparison the null
does not span — are visible only in the runs that failed.

## What these logs do NOT establish

No eigh speed claim survives from this set. The two rows that quote ratios from
them (`eigh n=512 >= 1.46x slower`, and the flat 512/768/1024 sweep) predate the
tightened margin gate, and the certification run that applied that gate voided
its own scipy comparisons at margins of 1.85x and 0.83x against a required 2.00x.
Read the ledger rows for the adjudication; these files are the evidence, not the
verdict.
