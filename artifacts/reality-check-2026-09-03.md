# Reality Check — 2026-09-03

**Vision (from README + COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1):** clean-room Rust reimplementation of
SciPy's core routines with the Condition-Aware Solver Portfolio (CASP) at the center, strict-vs-hardened
mode separation, differential conformance against a live pinned SciPy oracle (1.17.1 + numpy 2.4.3),
RaptorQ-backed artifact durability, and CI gates G1–G9 that make the tolerance contracts enforceable.

**Method:** every number below was measured today against HEAD `2cc2a79de` (main, 2026-09-03 01:27 -0400,
1 dirty file from concurrent agent work — untouched). Claims were checked against the tree, the beads
JSONL, GitHub Actions run history, and the probe scripts the repo itself ships. Nothing was compiled
locally (target/ is cold); compile/test health is cited from today's CI runs.

---

## Today's Working Surface (counted, not quoted)

| Metric | Live count | README claim | Verdict |
|---|---:|---:|---|
| Workspace crates | 19 | 19 | MATCH |
| Lines under `crates/*/src` | 611,277 | ~610,000 | MATCH |
| `#[test]` fns (src + tests/) | 7,727 + 2,329 = 10,056 | 10,057 | MATCH (±1 drift) |
| fsci-conformance test files | 793 | 796 | off by 3 |
| `diff_*` files | 731 | 731 | EXACT |
| Python oracles | 15 | 15 | EXACT |
| FSCI-P2C packets / legacy P2C | 18 / 13 | 18 / "legacy tree" | MATCH |
| Fuzz targets / corpus files / seeds | 96 / 923 / 569 | 98 / 923 / 569 | corpus+seeds EXACT; targets −2 |
| Beads open / in-progress / closed | 29 / 26 / 4,285 | 13 / 33 / 4,282 | same-day stale, direction consistent |
| Published baselines in `docs/` | 5 (linalg, sparse, fft, opt, integrate), each with `.raptorq.json` + `.decode_proof.json` | "published" | MATCH |
| Conformance test files, all crates `tests/` | 818 | — | — |

Per-crate line counts in the README table are accurate to within rounding (e.g. linalg 60,758 vs "~60,700",
stats 126,439 vs "~126,200", sparse 73,126 vs "~73,100"). The `scripts/conformance_coverage_audit.py`
probe exits 0 today: **zero SciPy-named public entry points without differential coverage**.

---

## The Load-Bearing Finding: CI

`ci.yml` was restructured today (`frankenscipy-liel6`). Its own header records that the previous revision
**ran 7,914 times and succeeded zero times** (G1 rustfmt drift killed every run in ~30 s; later gates never
executed; oracle-less lanes silently skipped). Today's runs, from the Actions API:

| Run | Trigger | Time (UTC) | Result |
|---|---|---|---|
| Full gate fan-out | `workflow_dispatch` | 04:23 | **FAILURE** (G1 fmt+clippy + many G2/G3 jobs; drift since cleaned — G1 went green at 05:28) |
| Push (runs **only** G1 by design) | `push` | 05:28 | SUCCESS (fmt + clippy only; all other jobs skipped) |
| Fuzz nightly (ASan+UBSan, `fuzz special`) | `schedule` | 08:57 | **FAILURE** |
| Full gate fan-out (nightly) | `schedule` | 09:55 | **FAILURE** — ≥13 jobs, including G2 unit tests in 6 crates (linalg, sparse, signal, ndimage, interpolate, conformance), G3 live-SciPy shards (linalg, optimize, …), G3b oracle capture, **G3 control**, G3c, G5, G6, **G9 tolerance ratchet** |

**Roadmap item 3 ("a CI run that passes") is still open.** The first fully green run has not happened; two
full-fan-out attempts failed today. Job-level logs for the 09:55 run return empty from the API, so the root
causes (ratchet breach vs gate defect on G9; oracle-install vs lane defect on G3b/G3-control) are not
attributable from here — that diagnosis is the single most valuable next action.

**The sharpest fact in this report:** the committed `parity_report.json` files are green (1,011 cases
passed, 0 failed across the 16 packets that carry one), but they were generated 2026-05-04 (P2C-001),
2026-06-27 (P2C-012), and 2026-08-17 (P2C-006). The *live* differential lane against pinned SciPy — the
thing G3 ran today — failed. Committed reports are evidence of past regens, not of today's HEAD.

---

## Vision Checklist (updated from 2026-05-03)

| # | Goal | Status | Evidence |
|---|---|---|---|
| 1 | 19-crate workspace, ~610K lines, 10K tests | WORKING | 19 members; 611,277 lines; 10,056 tests (counted) |
| 2 | Name-census parity (1,194/1,300) | WORKING | `PARITY-COVERAGE.md` headline + per-crate % match README table exactly; doc itself states it is name-matching, not behavioural |
| 3 | CASP on the dense solve family | WORKING | 5×4 loss matrix literal in `fsci-runtime/src/lib.rs` (DirectLU 1/5/40/120 …), `scipy_incumbent` pin 1.17.1/2.4.3, `HARDENED_MAX_DIM = 10_000` in `fsci-linalg` |
| 4 | Rule-based selectors elsewhere (not full CASP) | WORKING, as documented | `select_casp_iterative_solver` (sparse), `select_minimize_method` (opt), `select_hypergeometric_branch` (special) all exist; no loss matrix/posterior outside linalg |
| 5 | Strict/Hardened mode model | WORKING | `RuntimeMode` plumbed; hardened dimension cap enforced with fail-closed + audit at `fsci-linalg/src/lib.rs:1709,2110` |
| 6 | No tokio ecosystem | WORKING | `cargo tree -i {tokio,hyper,reqwest,axum,async-std,smol,tower}` → "did not match any packages" for all seven |
| 7 | `#![forbid(unsafe_code)]` | WORKING | `[workspace.lints.rust] unsafe_code = "forbid"` in root manifest |
| 8 | Live SciPy oracle pipeline | BUILT, RED IN CI | `live_oracle_capture` + G3b exist; today's G3b/G3 shards failed; incumbent pin verified in source |
| 9 | Benchmark baselines + RaptorQ sidecars | BUILT | 5 `docs/baseline_*.json`, each with sidecar + decode proof; G6 validates; no criterion regression compare in CI (documented) |
| 10 | Coverage audit (bead `ivxx6`) | GREEN TODAY | audit script exit 0, zero unreferenced entry points (README claims the same as of 08-30) |
| 11 | Toggle-driver census gate (`5f06d`) | GREEN TODAY | exit 0 on fsci-stats/linalg/ndimage/spatial; **0 switches exercised nowhere**; BUT drqu7 contract ratchet: **4 fsci-linalg switches lack accuracy contracts** (EIGH_BACKTRANSFORM_BLOCKED_ENABLE, EIGH_REDUCE_SUBSTAGE_TIMING, EIGH_SOLVE_SUBSTAGE_TIMING, EIGH_DSYMV_FORCE_SCALAR) |
| 12 | First fully green CI run (roadmap 3) | **NOT MET** | see CI table above |
| 13 | `fsci-arrayapi` promotion (roadmap 4) | NOT MET, honestly labeled | zero domain crates depend on it (only fsci-conformance, for tests); the crate's own `lib.rs`/`integration.rs` docs say "aspirational" |
| 14 | CASP beyond linalg (roadmap 5) | NOT MET, honestly labeled | rule-based selectors only |
| 15 | Tagged release + `[profile.release]` (roadmap 6) | NOT MET, honestly labeled | version 0.1.0, no tags; root has `release-perf`/`release-lines` but no `[profile.release]` override |
| 16 | Artifact topology convergence (roadmap 7) | NOT MET | 18 FSCI-P2C + 13 legacy P2C packet trees coexist; **FSCI-P2C-016 and -018 have no root `parity_report.json`** (partial topologies) |
| 17 | Named open defects filed and real | VERIFIED | `5lz5e` (open), `2sjwo`, `jyfke`, `drb0i` (open), `0xy3l`, `6d400` all present in JSONL with matching titles/statuses |

---

## What Changed Since the 2026-05-03 Reality Check

Closed since then:
1. **`fsci-datasets` and `fsci-odr` now exist** (670 and 2,993 source lines) — were NO_BEAD gaps.
2. **Live oracle capture pipeline exists** (`live_oracle_capture` binary, G3b lane, pinned-incumbent
   resolver module) — was "baked JSONs only".
3. **Benchmark baselines are published** with RaptorQ sidecars + decode proofs for 5 crates — was "no
   published numbers".
4. **Parity reports are green on paper**: 1,011/1,011 cases across 16 packets — in May every packet read
   `parity_gap`.
5. **fsci-special/fsci-signal defect triads closed** (periodogram normalization, iirnotch, gausspulse).
6. **CI was restructured** so gates could in principle pass — the 7,914×0 streak is documented and the
   first real fan-outs ran today.

Still open (unchanged or newly visible):
1. **CI is red on the full fan-out** (G2 in 6 crates, G3 shards, G3b, G3-control, G6, G9) and fuzz nightly
   is red. Diagnose the 09:55 run first; logs must be re-fetchable for that.
2. **Packets FSCI-P2C-016 / -018 lack parity reports** — either generate them or the "18 packets" claim
   overcounts by 2.
3. **4 fsci-linalg A/B switches without accuracy contracts** (drqu7 ratchet, named above).
4. CASP extension, arrayapi wiring, topology convergence, tagged release — all still roadmap.
5. **Beads-directory hygiene**: `.beads/` carries ~30 stale/rebuild/recovery db files alongside the truth
   (`issues.jsonl`); JSONL-first discipline held (live counts came from it), but the db debris invites
   confusion.
6. README numeric drift (same-day): beads 13/33 → live 29/26; conformance files 796 → 793; fuzz targets
   98 → 96.

---

## Honesty Notes

- **"parity_green" crate statuses measure name census + presence of `diff_*` lanes, not a passing run.**
  The README says gates aren't enforced until the first green run; the crate table's branding still
  invites over-reading. Committed reports are 2 weeks–4 months old; today's live differential lane failed.
- The README's self-critical claims all checked out: no `[profile.release]` (true), arrayapi aspirational
  with zero dependents (true, and the crate's own docs admit it), CASP linalg-only (true), tolerance
  ratchet not yet scanning `diff_*.rs` tolerances (matches ci.yml comment), fuzz nightly covers only 9 of
  96 targets (matches).
- Local compile health was **not** independently verified (cold `target/`); the evidence is CI's: clippy
  `-D warnings` passed at 05:28, runtime test failures appeared in 6 crates at 09:55.
- Counts drift by the hour in a tree with per-minute auto-commits; the live numbers here beat the README's
  snapshot wherever they differ, and the differences found are cosmetic.

---

## Follow-ups executed 2026-09-04 (post-reality-check)

1. **First full CI fan-out diagnosed and repaired.** Every failing gate of the
   09:55 UTC run (25 failed jobs) was root-caused: G3/G3b setup-python
   requirements file, G3-control stdlib-only canary, G3c missing incumbent
   install, G9 ratchet breach (4 unexplained savgol relaxations — all proven
   bit-exact vs scipy and tightened to baseline, count 365→360, threshold
   lowered), 12 broken implicit-Rust doc blocks across 4 crates, aspirational
   clough_tocher tolerances (scipy itself deviates 4.4e-8), the ndimage zoom
   mirror fail-closed + Constant-mode tap lookup + stencil degradation, a NaN
   panic through nalgebra SVD, bit-exact golden journeys (cross-host ulp
   drift), a G6 meta-test stale against the glob-loop workflow, and
   oracle-capture git-tracking drift. All verified by targeted rch runs.
2. **Fuzz nightly never fuzzed.** `-Zsanitizer=address,undefined` is rejected
   by rustc outright (`undefined` is not a sanitizer value); every nightly
   died building target 1. Fixed to ASan-only; a second silent failure (10-min
   link, no diagnostics, no libFuzzer banner) was addressed with ld
   memory-pressure flags. Dispatches: 33837902066 and later.
3. **drqu7 closed to zero uncontracted** in fsci-linalg (59/59; stats 139/139,
   ndimage 38/38, spatial 8/8). Census gate exit 0.
4. **ndimage boundary ratchet shrunk 23 → 15**: the short-axis stencil fix
   matched 8 formerly-excused Reflect cases to the incumbent, and the removed
   fail-closed emptied KNOWN_REFUSALS_TO_FIX. Verified 2/2 green on a worker
   even running non-pinned scipy 1.18.1.
5. **diff_ndimage_rotate worker failure adjudicated as oracle-side**: the
   worker reported max_abs=24 on rot180_order0_wrap for a fixture whose input
   maximum IS 24 (0..24) — the all-zeros-comparison signature of an oracle
   returning data=None. The identical call on pinned scipy 1.17.1 returns the
   correct rotation [24,23,…,0] with no exception (probed 2026-09-04). On the
   worker, scipy 1.18.1 raises inside the oracle's per-case try/except for
   every case. Bead ax6b0 tracks the genuine-incumbent runner confirmation
   (G3-ndimage shard of run 33837591299).
6. **Known-open after this session**: evidence runs 33837591299 (full fan-out,
   pinned 07-20 toolchain) and 33837900836 (G1 on the re-pinned 08-31
   toolchain) queued on the hosted-runner backlog; 33837902066 (fuzz build
   verification); sparse diff find_tril_triu / floyd_warshall unsampled; bead
   8oub0 (016/018 root parity reports).
