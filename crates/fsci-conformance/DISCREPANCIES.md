# Conformance Divergences (DISCREPANCIES)

Every intentional divergence from the reference SciPy/NumPy behavior
lives here. Every entry has a DISC-NNN ID, a status
(ACCEPTED / INVESTIGATING / WILL-FIX), the tests affected, and the last
review date. Stale divergences (>= 6 months since review) are flagged
at the bottom for re-evaluation.

> **Rule:** Tests that exercise accepted divergences must use
> `#[ignore]` or a dedicated XFAIL marker — never plain `#[test]` that
> silently passes on the divergent side.

## DISC-001 — rand 0.10 numerical drift in stats.rvs

- **Reference:** scipy.stats.<dist>.rvs(seed=42) on numpy's PCG64.
- **Our impl:** fsci-stats uses the rand 0.10 StdRng (ChaCha12).
- **Impact:** Sample streams differ bit-for-bit; mean/variance converge
  but per-sample values never match scipy.
- **Resolution:** ACCEPTED — porting StdRng to PCG64 is a separate
  project. Statistical invariants (sample mean within tolerance)
  suffice for conformance; sample-by-sample parity is not a MUST.
- **Tests affected:** `fsci-stats::tests::rvs_*` (seeded under br-otdp).
- **Review date:** 2026-04-23

## DISC-002 — Complex Matrix Market rejection

- **Reference:** scipy.io.mmread returns a complex-valued sparse matrix
  when the header declares `field complex`.
- **Our impl:** fsci-io returns an error on `field complex`.
- **Impact:** Files with complex matrices cannot round-trip.
- **Resolution:** ACCEPTED for Phase 2 — fsci-sparse does not yet
  expose a complex CSR type. Re-enable once sparse is complex-capable.
- **Tests affected:** `e2e_io::mm_read_complex_rejected`.
- **Review date:** 2026-04-23
- **Related beads:** frankenscipy-r5tv

## DISC-003 — cubature bounds dimensionality guard

- **Reference:** scipy.integrate.cubature is tolerant of mismatched
  lower/upper bound lengths when ndim > 1 (broadcast-style).
- **Our impl:** fsci_integrate::cubature hard-errors on mismatched
  bound dimensionality to prevent silent misintegration.
- **Resolution:** ACCEPTED — stricter validation is safer than scipy's
  permissive broadcast, and no real workload we've seen relies on it.
- **Tests affected:** fixture case `cubature_rejects_mismatched_bounds`
  (P2C-013, Hardened mode).
- **Review date:** 2026-04-23

## DISC-004 — Cluster labels exact-equality → permutation-invariant

- **Reference:** scipy.cluster.hierarchy.fcluster / leaves_list emit
  labels in a specific numeric order determined by their internal
  linkage walk.
- **Our impl:** fsci_cluster produces the same partition but no
  contract guarantees identical numeric IDs. The conformance
  comparator now canonicalizes by first-occurrence order
  (br-7eaq), so partitions match regardless of label assignment.
- **Resolution:** ACCEPTED — label numeric value is not semantic.
  Fixture cases that rely on exact label IDs opt in via
  `expected.deterministic_labels: true`.
- **Tests affected:** P2C-009 cluster fixture, all `kind: "labels"` cases.
- **Review date:** 2026-04-23
- **Related beads:** frankenscipy-7eaq

## DISC-005 — e2e-only io was excluded from oracle parity counts

- **Reference:** scipy.io exposes observable behavior that should
  eventually be compared by P2C fixture packets and Python oracle
  captures.
- **Our impl:** fsci-io previously had crate-level e2e tests but no
  `fixtures/FSCI-P2C-*.json` packet lane in `fsci-conformance`.
  FSCI-P2C-017 now seeds an oracle-backed lane for Matrix Market,
  MAT v4 real-double matrices, numeric text, and WAV metadata/value parity.
- **Impact:** Users reading parity numbers see fixture-backed families
  only; io now has a narrow fixture-backed score, while any broader
  scipy.io claims remain out of scope until more cases land.
- **Resolution:** RESOLVED for the seed io lane. MAT parity is currently
  narrowed to SciPy-compatible `savemat(..., format="4")` / `loadmat` for
  full real double matrices. MAT v5/v7.3, structs, cells, sparse matrices,
  chars, and complex arrays remain out of scope until new fixture packets
  expand the surface.
- **Tests affected:** `e2e_io.rs`, `FSCI-P2C-017_io_core.json`.
- **Review date:** 2026-04-25
- **Related beads:** frankenscipy-3m6f

## DISC-007 — fsci-integrate LSODA early-termination on highly stiff oscillators (RESOLVED 2026-09-16)

- **Reference:** scipy.integrate.solve_ivp(method='LSODA') handles
  stiff Van der Pol mu=10 (and similar transition-stiff problems)
  by adaptively shrinking step size and switching between Adams and
  BDF formulas internally.
- **History:** Previously fsci_integrate::LsodaSolver failed to advance
  past t=0 on Van der Pol mu=10 before adaptive step-size switching
  stabilized.
- **Resolution:** RESOLVED — `fsci_integrate::LsodaSolver` correctly
  detects stiffness and transitions to BDF stepping, successfully
  integrating the Van der Pol oscillator ($\mu=10$) over $[0, 20]$ (632 steps,
  matching SciPy reference $y(20) \approx [1.939356, -0.070082]$), and over
  $[0, 2]$ with `t_eval` matching SciPy reference to $< 10^{-4}$.
  Locked by unit test `solve_ivp_lsoda_van_der_pol_matches_scipy_reference`
  and differential oracle binary `diff_lsoda_vdp.rs`.
- **Tests affected:** `crates/fsci-integrate/src/bin/diff_lsoda_vdp.rs`,
  `crates/fsci-integrate/src/api.rs::tests::solve_ivp_lsoda_van_der_pol_matches_scipy_reference`.
- **Review date:** 2026-09-16
- **Related beads:** frankenscipy-r8ug, frankenscipy-ljmg

## DISC-008 — fsci-signal remez least-squares approximation (RESOLVED 2026-06-08)

- **Reference:** scipy.signal.remez (Parks-McClellan / Remez exchange,
  equiripple minimax).
- **History:** fsci_signal::remez was a "Simplified Remez" *weighted
  least-squares* cosine fit, NOT the minimax exchange — diverging from
  scipy by ~0.02–0.05 (remez_lp_11 1.9e-2, remez_bp_15 4.5e-2).
- **Resolution:** RESOLVED for odd `numtaps` (Type I) — replaced with a
  true Parks-McClellan exchange (`remez_type1_pm`): dense band grid,
  barycentric-Lagrange deviation + amplitude interpolation, alternating-
  extrema search, Chebyshev DCT-II coefficient recovery. The equiripple
  optimum is unique, so it matches scipy 1.17.1 to machine precision —
  verified across lowpass/bandpass/highpass/order-21/weighted designs:
  remez_lp_11 1.1e-16, remez_bp_15 4.2e-16, remez_lp_21 1.9e-16,
  remez_hp_17 9.4e-16, remez_bp_25(w=[1,2,1]) 4.6e-16. P2C-011 remez
  fixture tolerances tightened `0.06/0.05 → 1e-9/1e-9`; unit test
  `remez_matches_scipy_reference` locks it.
- **Remaining → RESOLVED 2026-06-09 (frankenscipy-h4ejo):** even `numtaps`
  (Type II) now uses true Parks-McClellan via the `A(ω)=cos(ω/2)·P(ω)`
  factorization (`remez_type2_pm`): fold `cos(ω/2)` into desired/weight
  (`D'=D/cos(πf)`, `W'=W·cos(πf)`), run the same Type-I exchange on `P` over
  `m=N/2` cosines, recover Type-II taps via `a₁=b̃₀+b̃₁/2`,
  `aⱼ=(b̃ⱼ₋₁+b̃ⱼ)/2`, `aₘ=b̃ₘ₋₁/2`, `h[m-j]=h[m-1+j]=aⱼ/2`. Nyquist grid
  points (`cos(πf)≈0`, the forced `A(0.5)=0`) are dropped. The minimax optimum
  is unique, so the equiripple solution equals scipy's to machine precision;
  verified oracle-free by `remez_type2_is_equiripple_optimal` (symmetric taps +
  `A(0.5)≈0` + passband ripple == stopband ripple to ~1%, vs the old LS
  fallback's >50% mismatch). LS fallback retained only for pathological grids
  where the exchange can't form `nz` alternations.
- **Tests affected:** P2C-011 remez_lp_11_passband_0p2_stopband_0p3,
  remez_bp_15_3band; unit `remez_type2_is_equiripple_optimal`.
- **Review date:** 2026-06-09
- **Related beads:** frankenscipy-7jrx, frankenscipy-zxxdi, frankenscipy-h4ejo

## DISC-008a — fsci-signal firwin2 (RESOLVED 2026-06-08)

- **Reference:** scipy.signal.firwin2 (frequency-sampling + inverse-FFT
  + windowing).
- **History:** firwin2 previously diverged ~0.05 (under-sampled grid,
  circular shift instead of a linear-phase ramp, spurious DC
  normalization). That was fixed (phase-ramp shift + proper
  `1 + 2^⌈log2(numtaps)⌉` grid + symmetric design window).
- **Resolution:** RESOLVED — fsci_signal::firwin2 now matches scipy
  1.17.1 to machine precision on the P2C-011 fixture cases
  (firwin2_lp_11 maxdiff 2.0e-17, firwin2_bp_15 maxdiff 9.0e-17). The
  P2C-011 firwin2 fixture tolerances were tightened from
  `atol/rtol = 0.06/0.05` back to `1e-9/1e-9` to re-arm the gate against
  regression. The 1e-9 unit test `firwin2_matches_scipy_reference`
  (51-tap LP, 33-tap BP) already locks the algorithm.
- **Tests affected:** P2C-011 firwin2_lp_11_freqsamp, firwin2_bp_15_3band.
- **Review date:** 2026-06-08
- **Related beads:** frankenscipy-7jrx

## Stale / needs re-review

_None yet — first revision._

---

## Adding a new divergence

1. Pick the next DISC-NNN number.
2. Fill all fields (reference / our impl / impact / resolution /
   tests affected / review date).
3. Link back from the affected test or fixture case by the DISC ID in
   a comment.
4. Never remove an entry; mark it `RESOLVED` with the commit SHA when
   the divergence disappears.
