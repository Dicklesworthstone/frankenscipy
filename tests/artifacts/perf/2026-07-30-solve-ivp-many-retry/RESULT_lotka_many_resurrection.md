# RESULT — `solve_ivp_many` resurrected: VOID-NONULL → DECIDED, and the ledger's stated mechanism is wrong

Completes the campaign pre-registered in `31b13ba7c`, committed before any gated
`lotka-many` timing existed. Conformance fix in `8d95e678f`.

**Scorecard: two confirmed, three falsified.** The conformance prediction and the
headline-magnitude prediction held. Every prediction about *why* the win exists
was wrong.

| | prediction | measured | verdict |
|---|---|---|---|
| **P1** gate now passes | `max_scaled_diff ≤ 100`, point ≤ 10 | **0.000** at both N | ✅ **CONFIRMED** |
| **P2** ratio at N=1000 | `[800×, 3200×]` | **2749.45×** | ✅ **CONFIRMED** (on a NOT-DECIDED cell) |
| **P3** callback lever 20–30× | product matches within 1.5× | callbacks worth **1.09–1.14×** | ❌ **FALSIFIED** |
| **P4** observed threads = requested | at both N | **29 of 32** at N=200 | ❌ **FALSIFIED** |
| **P5** sampled ≈ completion within 1.3× | | **1.563× / 2.089×** | ❌ **FALSIFIED** |

## Measurement

Host `thinkstation1` (5975WX, 32C/64T, AVX2+FMA), `powersave`,
`taskset -c 32-63`, 11 rounds, SciPy 1.17.1 / NumPy 2.4.3, SciPy engine
`aa16f42cc85fa027…`, ELF **`51e17a303c84041b9ffe51d92663ac01f845e9da85552d147410a6ac95a821d9`**
self-reported from inside the process, built via
`rch --base 8d95e678f --clean-overlay --no-overlay` on `vmi1264463` — a clean
baseline with **no overlay at all**, so no co-tenant agent's working-tree edits
could enter the binary.

| cell | ratio (SciPy / FrankenSciPy) | bootstrap CI95 | null gate | verdict |
|---|---:|---|---|---|
| `lotka-many` N=200 | **1163.7587×** | [868.83, 1247.09] | c1✓ c2✓ c2b✓ **c3✓** | **DECIDED FRANKENSCIPY WIN** |
| `lotka-many` N=1000 | 2749.4527× | [1967.94, 3203.67] | c3 ✗ (null median 1.065) | NOT DECIDED |
| `lotka-final-many` N=200 | 744.7237× | [662.71, 787.15] | c3 ✗ | NOT DECIDED |
| `lotka-final-many` N=1000 | **1315.9272×** | [1027.94, 1622.09] | c1✓ c2✓ c2b✓ **c3✓** | **DECIDED FRANKENSCIPY WIN** |

Two cells are DECIDED and two are not. The two that are not failed the corrected
gate's **median clause** — our A/A null median drifted to 1.065 and 1.204 against
a 2% allowance — on a host carrying other agents' load. The ratios in those cells
clear the 2× null margin enormously, but the clause is the clause and they are
reported as NOT DECIDED. Run-to-run variability is real here: an earlier 7-round
N=200 probe gave 961.9× where the 11-round cell gives 1163.8×, a 21% swing at
cv≈20%.

**This is the first DECIDED same-invocation gated win in the vmap-over-solver
family**, and it converts the cc ledger's only `VOID-NONULL` row.

## P1 — the conformance blocker is gone

The row was blocked by **711.439 tolerance units** at sample 138. After routing
`t_eval` through SciPy's quartic Dormand–Prince dense output:

```
N=200   max_abs_diff 8.527e-14   max_scaled_diff 0.000   60,000 components compared
N=1000  max_abs_diff 1.217e-13   max_scaled_diff 0.000  300,000 components compared
```

All 200/200 and 1000/1000 trajectories reached `t_end`; Lotka–Volterra invariant
drift agrees between arms at 2.269e-7 in every cell. Counted work is nearly
identical (`nfev` 1,246,300 ours vs 1,247,300 SciPy at N=1000 — 0.08% apart), so
this is not a case of winning by doing less.

## P3 — the ledger's mechanism is wrong, and this is the finding that matters

The ledger explains the win as *"The callback lever (inline Rust RHS, no Python
per-step) gives ~25× per-solve; the N-way parallelism multiplies it to ~1500×."*
The harness measures the callback share directly, and it is small:

```
N=200    SciPy 2179.54 ms/batch, Python RHS callbacks (248,488)   = 250.93 ms = 11.5%
N=1000   SciPy 11654.99 ms/batch, Python RHS callbacks (1,247,300) = 1279.79 ms = 11.0%
```

Removing the callbacks *entirely* moves the ratio 1163.8× → 1022.3× and
2749.5× → 2524.1×. **The callback lever is worth 1.138× and 1.089× — not 25×.**
Roughly 89% of SciPy's time is its own Python driver: step-size control, error
norms, and dense-output object construction.

The honest decomposition, from observed threads rather than requested:

| N | ours/traj | observed threads | serial-equivalent/traj | SciPy/traj | 1-thread ratio | × threads |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 9.432 µs | 29 | 273.5 µs | 10.898 ms | **39.8×** | 1155× ≈ 1164× observed |
| 1000 | 4.110 µs | 32 | 131.5 µs | 11.655 ms | **88.6×** | 2836× ≈ 2749× observed |

So the structure is right — a per-trajectory factor times a thread factor, and
the product reconstructs the measured ratio — but the per-trajectory factor is
**not** the RHS callback. My P3 predicted 20–30× from callback elimination; the
magnitude bracket happens to contain the N=200 figure (39.8× is close), but the
attribution is wrong, and at N=1000 the per-trajectory factor is 88.6×, outside
it. This is the same mechanism the GMRES/lsqr/qmr campaign kept finding: SciPy's
cost is dominated by interpreted *driver* bookkeeping, not by the user's kernel.

Note also that our per-trajectory cost *improves* with batch size (9.43 µs →
4.11 µs) while SciPy's is flat (10.90 ms → 11.65 ms). The win grows with N
because our batch amortises, not because SciPy degrades.

## P5 — `t_eval` sampling is a large cost asymmetry, the opposite of what I predicted

I predicted the sampled and completion-only surfaces would agree within 1.3×
because sampling should cost both arms proportionally. It does not:

```
N=200   sampled 1163.76× / completion 744.72× = 1.563×
N=1000  sampled 2749.45× / completion 1315.93× = 2.089×
```

Adding 150-sample `t_eval` raises SciPy's time from 1453.5 → 2179.5 ms (N=200)
and 7686.9 → 11655.0 ms (N=1000) — about **+50%** — while ours does not rise at
all. SciPy builds an `RkDenseOutput` object per step and evaluates it from
Python per sample; we evaluate a Horner loop in Rust. The callback share falling
from 17.6%/16.5% (completion) to 11.5%/11.0% (sampled) is the same effect seen
from the other side: the added time is all Python dense-output machinery.

The irony is worth recording: **the interpolation path we had to fix for
correctness turns out to be the single largest component of the win.**

## P4 — observed threads are not requested threads

At N=200 we requested 32 worker threads and **observed 29**. At N=1000, 32 = 32.
SciPy was 1 = 1 in every cell with BLAS capped at one thread. This is exactly why
the standing requirement specifies observed rather than requested; the N=200
decomposition above uses 29, not 32, and would be 10% wrong otherwise.

## Versus the historical claim

The historical, ungated numbers were 1481× (N=200) and 1599× (N=1000). Measured
under the gate: **1163.76×** at N=200 (historical was 1.27× optimistic) and
**2749.45×** at N=1000 (historical was 1.72× *pessimistic*). My pre-registration
predicted the resurrection would land below its historical number, as every
previous one in this campaign had; at N=1000 it landed well above. The
historical figures were not systematically inflated — they were just ungated,
which is a different fault and is now repaired.

## Chooser statement

> For an ensemble of independent non-stiff ODE trajectories sharing one vector
> field — 200–1000 Lotka–Volterra systems, RK45, `rtol=1e-8 / atol=1e-10`, 150
> retained samples each — **FrankenSciPy `solve_ivp_many` finishes the whole job
> 1163.76× faster than a Python loop over `scipy.integrate.solve_ivp`** at N=200
> (DECIDED), rising to ~2749× at N=1000 (measured, not decided). About 89% of the
> gap is SciPy's interpreted solver driver and its per-sample dense-output
> machinery, not the Python RHS callback, and roughly a 30× factor of it is
> N-way parallelism that a single-core user will not see. On one core the
> advantage is ~40–89× per trajectory. This says nothing about stiff problems,
> coupled ensembles, or a single trajectory.

## Artifacts

- `PREREGISTERED_mechanism.md` — predictions, committed `31b13ba7c`
- `raw/lotka_many_n{200,1000}.txt`, `raw/lotka_final_many_n{200,1000}.txt`
- `raw/verify_rk45_dense_output.py` + `.out` — the SciPy-oracle validation of the
  dense-output arithmetic, independent of the Rust build
