# FrankenSciPy Perf Ledger — CrimsonForge (measured head-to-head vs SciPy/sklearn)

Sidecar to the canonical `docs/NEGATIVE_EVIDENCE.md` (reserved by MistyBirch). Holds
**CrimsonForge's** measured gauntlet results so dead ends are never retried and
regressions are reverted. Entries also routed to MistyBirch for the canonical merge.

- Host: 64 cores, release builds via
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cc rch exec -- cargo bench -p <crate>`.
- Original = SciPy 1.17.1 / scikit-learn 1.9.0. Oracle scripts: `docs/perf_oracle_*.py`.
- "vs serial" A/B isolates a parallelization by forcing its work-gate to `usize::MAX`.
- ⚠️ Some runs taken under concurrent multi-agent bench load → parallel numbers are
  conservative (more core contention hurts the 64-thread path more than serial).

## Scorecard

| Optimization (bead) | Workload | SciPy/orig | fsci | fsci vs orig | parallel vs serial | Verdict |
|---|---|---|---|---|---|---|
| GMM diag E-step parallel (yw7ts) | GMM n=1000 d=3 k=3 | 2.398 ms | 0.592 ms | **4.05× faster** | serial (below gate) | ✅ KEEP |
| GMM diag E-step parallel (yw7ts) | GMM n=5000 d=8 k=5 | 29.75 ms | 3.735 ms | **7.97× faster** | parallel | ✅ KEEP |
| GMM diag E-step parallel (yw7ts) | GMM n=20000 d=16 k=8 | 432.4 ms | 39.04 ms | **11.1× faster** | **3.07×** (119.8→39.0 ms) | ✅ KEEP |
| AP responsibility parallel (yw7ts) | AP n=1000 d=4 | 319.4 ms | 249.9 ms | 1.28× faster | — | ✅ KEEP (parity) |
| AP responsibility parallel (yw7ts) | AP n=2000 d=4 | 2158 ms | 2098 ms | **1.03× (PARITY)** | **2.02×** (4.23→2.10 s) | ✅ KEEP (parity) |
| Interpolate evaluate_many parallel (yw7ts) | NdPPoly m=200k total=6 | n/a | 8.86 ms | — | **0.88× (REGRESSION)** vs serial 7.79 ms | ❌ **REVERTED** |
| Interpolate evaluate_many parallel (yw7ts) | BPoly m=200k | n/a | ~8.5 ms | — | ~serial 8.18 ms (no gain) | ❌ **REVERTED** |
| Distribution pdf_many lgamma-hoist (q53ya) | gamma.pdf n=4096 | 149.6 µs | 49.86 µs | **3.0× faster** | 3.2× (hoist vs map) | ✅ KEEP |
| Distribution pdf_many lgamma-hoist (q53ya) | beta.pdf n=4096 | 296.8 µs | 60.97 µs | **4.87× faster** | 4.3× (hoist vs map) | ✅ KEEP |
| Distribution pmf_many lgamma-hoist (q53ya) | hypergeom.pmf supp=701 | 3744.9 µs | 38.34 µs | **97.7× faster** | — | ✅ KEEP |
| pdist parallel (8e7e6d99, NOT mine) | pdist euclidean n=256 | 92.1 µs | 674.9 µs | **0.14× (7.3× SLOWER)** | gate fires at n=256 | ⚠️ LOSS → owner |
| pdist parallel (8e7e6d99, NOT mine) | pdist euclidean n=512 | 326.3 µs | 889.0 µs | **0.37× (2.7× SLOWER)** | overhead amortizes w/ n | ⚠️ LOSS → owner |
| pdist parallel (8e7e6d99, NOT mine) | pdist cosine n=256 | 81.9 µs | 736.7 µs | **0.11× (9× SLOWER)** | — | ⚠️ LOSS → owner |
| pdist flat dim-4 rows (nm8ex.1) | pdist euclidean n=256 | 88.96 µs | 172.83 µs | **0.51× (1.94× SLOWER)** | 1.52× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist cosine n=256 | 79.69 µs | 208.89 µs | **0.38× (2.62× SLOWER)** | 1.83× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist euclidean n=512 | 309.79 µs | 714.58 µs | **0.43× (2.31× SLOWER)** | 1.11× faster internally | ✅ KEEP, residual gap |
| pdist flat dim-4 rows (nm8ex.1) | pdist cosine n=512 | 275.14 µs | 828.70 µs | **0.33× (3.01× SLOWER)** | 1.44× faster internally | ✅ KEEP, residual gap |
| linkage NN-chain (average) | linkage n=400 d=4 | 1586.5 µs | 1904.5 µs | **0.83× (1.2× slower)** | — | ⚠️ near-parity |
| cophenet mem::take (jphzn) | cophenet n=400 | 401.5 µs | 219.7 µs | **1.83× faster** | — | ✅ KEEP |
| kmeans2 double-buffer (4ylee) | kmeans2 k4 n2000 iter=50 | 2104.7 µs | 5126 µs | **0.41× (2.4× SLOWER)** | scalar assign vs scipy SIMD | ⚠️ kernel gap → bead |
| kmeans Lloyd early-stop | kmeans k4 n2000 | 2104.7 µs* | 357.4 µs | **5.9× faster** | *vs scipy kmeans2 fixed-iter | ✅ KEEP (early-stop) |
| correlate tap-table (e3r7e) | correlate 5x5 256² | 933.7 µs | 1099 µs | **0.85× (1.18× slower)** | byte-identical | ✅ KEEP (parity) |
| gaussian_filter (NOT mine) | gaussian σ=2 256² | 1143.0 µs | 3238 µs | **0.35× (2.83× slower)** | separable but slow 1D kernel | ⚠️ gap → owner |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=100 nnz=500 | 4.63 µs | 0.388 µs | **11.9× faster** | 1.54× vs legacy row-sweep; bit-identical | ✅ KEEP |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=1000 nnz=10k | 8.00 µs | 7.077 µs | **1.13× faster** | 2.10× vs legacy row-sweep; scale loss closed | ✅ KEEP |
| spmv_csr cached+unrolled row sweep (2hclc) | SpMV n=10000 nnz=100k | 96.95 µs | 68.82 µs | **1.41× faster** | 2.14× vs legacy row-sweep; scale loss closed | ✅ KEEP |
| gaussian_kde evaluate_many parallel | KDE n=1000 eval 1000 pts | 19090 µs | 1062 µs | **18.0× faster** | heavy per-pt → scales | ✅ KEEP |
| gaussian_kde evaluate_many parallel | KDE n=5000 eval 5000 pts | 201197 µs | 11959 µs | **16.8× faster** | — | ✅ KEEP |
| MGC mgc_map O(n²) + parallel reps | multiscale_graphcorr n=80 reps=100 | 295705 µs | 21578 µs | **13.7× faster** | O(n⁴)→O(n²) + parallel | ✅ KEEP |
| Rotation.apply_many (w7ocv) | apply 8192 pts | 28.30 µs | 12.03 µs | **2.35× faster** | matrix-once hoist 4.5× vs map | ✅ KEEP |
| loadtxt direct-parse (fwnb1) | loadtxt 500×20 | 2022 µs | 259.5 µs | **7.79× faster** | vs numpy.loadtxt (Python) | ✅ KEEP |
| savetxt write! (d1uxy) | savetxt 500×20 | 4208 µs | 631.6 µs | **6.66× faster** | vs numpy.savetxt (Python) | ✅ KEEP |
| KDTree build (select_nth) | cKDTree build n=4096 3-D | 767.8 µs | 809.5 µs | 0.95× (parity) | vs scipy ELITE C | ✅ KEEP |
| KDTree query dual-tree parallel (9k50g) | cKDTree query 4096 pts | 2032.8 µs | 1756.7 µs | **1.16× faster** | beats single-threaded C | ✅ KEEP |
| silhouette per-anchor parallel | silhouette n=500 d=4 | 2064 µs | 720.8 µs | **2.86× faster** | no small-n regression | ✅ KEEP |
| silhouette per-anchor parallel | silhouette n=2000 d=4 | 32928 µs | 3113.5 µs | **10.6× faster** | scales w/ n | ✅ KEEP |
| ndimage zoom order=1 FIXED (wm14d) | zoom 2× 256² order=1 | 4842 µs | 19409 µs | **0.25× (4.0× slower)** — was 0.06× (17.7×) | cardinal fast path added | ✅ FIXED (4.4× faster) |
| ndimage zoom order=1 residual fast path (wm14d) | zoom 2× 256² order=1 | 3889 µs | 7968 µs | **0.49× (2.05× slower)** | 4.27× faster than generic sampler | ✅ KEEP, residual gap |
| ndimage zoom order=3 | zoom 2× 256² order=3 | 14053 µs | 31573 µs | **0.45× (2.25× slower)** | generic spline-weight kernel | ⚠️ residual gap |
| ndimage rotate order=3 (shares wm14d fix) | rotate 30° 256² order=3 | 5577 µs | 6439 µs | **0.87× (1.15× ~parity)** | cardinal spline path | ✅ near-parity |
| ndimage rotate order=1 (shares wm14d fix) | rotate 30° 256² order=1 | 1991 µs | 8733 µs | **0.23× (4.4× slower)** | residual machinery gap (was ~17×) | ⚠️ residual gap |
| kendalltau inversion-count O(n log n) | kendalltau n=2048 | 597 µs | 230.4 µs | **2.59× faster** | scipy fixed overhead | ✅ KEEP |
| kendalltau inversion-count O(n log n) | kendalltau n=4096 | 537 µs | 552.4 µs | 0.97× (parity) | both O(n log n) at scale | ✅ KEEP |
| Delaunay precompute circumcircles (9l5oo) | Delaunay n=1000 2-D | 1980 µs | 898 µs | **2.2× FASTER** — was 0.30× (3.3× slower) | cheaper bad-test | ✅ WIN (7.3× self-speedup) |
| Delaunay precompute circumcircles (9l5oo) | Delaunay n=2000 2-D | 4488 µs | 3257 µs | **1.38× FASTER** — was 0.17× (5.9× slower) | O(n²) const-factor crushed | ✅ WIN (8.1× self-speedup) |

## Detail

### GMM diagonal E-step ordered-slots parallelization (frankenscipy-yw7ts) — ✅ KEEP
Oracle `docs/perf_oracle_gmm.py` (sklearn `GaussianMixture(covariance_type="diag")`).
fsci **4–11× faster** than sklearn, ratio growing with n as the parallel E-step
(gate n·k·d≥2¹⁶) engages. Forced-serial A/B at n=20000: 119.79 ms serial vs 39.04 ms
parallel = **3.07×** from parallelization (64 cores; Amdahl-capped by serial
M-step+Cholesky). Even forced-serial fsci beats sklearn 3.6× → port is the base win.
Gate validated: n=1000 serial (592 µs) correctly avoids spawn overhead. Conformance
green (gaussian_mixture_recovers_* tests). Commit `8b84e8b2`.

### Affinity propagation responsibility-update parallelization (frankenscipy-yw7ts) — ✅ KEEP, but AP is PARITY
Oracle `docs/perf_oracle_ap.py` (sklearn `AffinityPropagation(affinity="precomputed")`).
fsci AP is **near parity** with sklearn: 1.28× at n=1000, **1.03× at n=2000**. sklearn
AP is numpy-vectorized and well-optimized. Forced-serial A/B at n=2000: 4.234 s serial
vs 2.098 s parallel = **2.02×** from the responsibility-update parallelization — i.e.
the parallelization is exactly what lifts fsci to parity; forced-serial fsci would
LOSE ~2× to sklearn. **Keep the parallelization (real 2× internal, byte-identical),
but AP is NOT a competitive advantage.** Remaining gap = the still-serial availability
update (column-strided over the row-major matrix; parallelizing it needs a transposed
layout — a candidate future lever, NOT yet done). Commit `1f32a4b2`.

### Interpolate batch-evaluator parallelization (frankenscipy-yw7ts) — ❌ REVERTED
BPoly/NdPPoly/NdBSpline `evaluate_many` were parallelized across points (par_query_map
for BPoly; hand-rolled per-thread scratch for NdPPoly/NdBSpline). **Measured A/B at
m=200k** (forced-serial gate→MAX, rebuilt): NdPPoly **serial 7.79 ms vs parallel
8.86 ms = 0.88× (a 14% REGRESSION)**; BPoly serial 8.18 ms ≈ parallel (no gain). The
per-point work is only ~k/total flops (~30 for the typical low-degree/low-dim case),
so 64-thread spawn + per-thread-scratch allocation overhead exceeds the compute — the
opposite of GMM, whose heavy gaussian/exp per-point work parallelizes 3×. The gate
(`points·total ≥ 2¹⁶`) counts flops, but 2¹⁶ flops is trivial vs thread overhead, and
the break-even (if any) is contention-dependent and unverifiable on this shared host.
**Reverted all three to the serial map; the byte-identical loop-invariant HOIST
(binoms/strides/scratch precomputed once) is PRESERVED — that was the real, monotone
win.** Conformance green (interpolate evaluate_many tests). Revert commit: this one.

### Distribution batch pdf_many/pmf_many lgamma-hoist (frankenscipy-q53ya) — ✅ KEEP (strong)
Oracle `docs/perf_oracle_dist.py` (scipy.stats vectorized pdf/pmf over arrays, n=4096
/ full support). fsci `pdf_many`/`pmf_many` hoist the expensive lgamma/ln_beta
normalizer ONCE then map. **fsci beats scipy.stats 3–98×:** gamma 3.0×, beta 4.87×,
hypergeom **97.7×** (scipy's hypergeom.pmf is famously slow — betaln + overflow
guards per outcome). The hoist itself is 3–4× over the naive `map(pdf)` (gamma
49.9µs vs 159.9µs; beta 61.0µs vs 261.3µs), and `map(pdf)` ≈ scipy — i.e. the hoist
is exactly what wins the head-to-head. Byte-identical (normalizer is a loop
invariant), NO revert risk. The 19-density batch family shares this lever → all KEEP
by the same construction. Conformance green. Commits: `q53ya` (impl) + oracle here.

### Spatial pdist — ⚠️ LOSS vs scipy (NOT my optimization; flagged to owner)
Oracle `docs/perf_oracle_pdist.py` (scipy.spatial.distance.pdist, 4-D, n=256/512).
fsci pdist (parallel, commit `8e7e6d99` by another agent) is **2.7–9× SLOWER than
scipy**: euclidean 674.9µs vs 92.1µs (n=256), 889.0µs vs 326.3µs (n=512); cosine
736.7µs vs 81.9µs (n=256). The ratio improves with n (7.3×→2.7×) → fixed thread-spawn
overhead. The gate `cdist_thread_count` fires at `work=n²·dim≥2¹⁸`, i.e. exactly at
n=256/dim=4 — parallelizing trivially-small 4-D distance pairs across 64 threads, the
same over-eager pathology as the (reverted) interpolate evaluators, worsened by
multi-agent contention. The author's "3.8–7.3×" claim is parallel-vs-serial INTERNAL,
not vs scipy; implied serial ≈ 3.4–6.5 ms at n=512 → fsci's pure-Rust pdist KERNEL is
~10–60× slower than scipy's C. **NOT reverted — another agent's file; routed to the
spatial owner.** Recommendation: raise the pdist gate well above 2¹⁸ AND/OR a faster
inner kernel (scipy uses tuned C). Honest LOSS recorded.

### Spatial pdist flat dim-4 rows (frankenscipy-nm8ex.1) — ✅ KEEP internally, ⚠️ LOSS vs SciPy
Follow-up to the `nm8ex` residual gap. The direct dim-4 fast path still loaded
every pair through `Vec<Vec<f64>>`; this lever stages validated 4-column rows
into compact `[f64; 4]` points once per call and runs the same Euclidean/Cosine
arithmetic over fixed-width rows. Same-worker rch `ovh-b` Criterion medians
improved across all four rows: 263.00→172.83µs (1.52×) for euclidean n=256,
381.98→208.89µs (1.83×) for cosine n=256, 794.72→714.58µs (1.11×) for
euclidean n=512, and 1.1930ms→828.70µs (1.44×) for cosine n=512. The focused
bit-exact dim-4 `pdist` guard passed via rch.

Honest SciPy score remains a loss: local SciPy 1.17.1 / NumPy 2.4.3 oracle was
88.96µs, 79.69µs, 309.79µs, and 275.14µs respectively, so Rust is still
1.94×, 2.62×, 2.31×, and 3.01× slower (0 wins / 4 losses / 0 neutral). KEEP
because the internal win is significant and behavior-preserving; route deeper
to output batching or generated SIMD-style dim-specialized kernels rather than
retrying row staging alone.

### Hierarchical clustering: linkage + cophenet (frankenscipy-jphzn) — ⚠️ parity / ✅ KEEP
Oracle `docs/perf_oracle_hier.py` (scipy.cluster.hierarchy, n=400 blobs, average).
- **linkage average: fsci 1904.5 µs vs scipy 1586.5 µs = 0.83× (1.2× SLOWER).** Near-
  parity — scipy's NN-chain linkage is tuned C; fsci's pure-Rust version is within
  20%. NOT a regression (no parallelization involved); just the expected small gap to
  optimized C. KEEP (correct + close); a faster reducible-distance update is a future
  lever if linkage becomes a bottleneck.
- **cophenet: fsci 219.7 µs vs scipy 401.5 µs (distances-only, fair) = 1.83× FASTER.**
  The `jphzn` move-instead-of-clone of each node's member list helps; the tree
  traversal is efficient. (NB: the naive `cophenet(Z, Y)` scipy call is 1758 µs but
  ALSO computes the correlation coefficient — not comparable; used `cophenet(Z)`.)
  KEEP.

### kmeans / kmeans2 (frankenscipy-4ylee double-buffer) — mixed; kernel gap surfaced
Oracle: scipy.cluster.vq.kmeans2 (n=2000, k=4, d=4, fixed init).
- **kmeans2 fixed 50 iters: fsci 5126 µs vs scipy 2104.7 µs = 0.41× (2.4× SLOWER).**
  Both run 50 full Lloyd iterations. The 4ylee double-buffer (mem::swap vs realloc) is
  byte-identical and NOT the cause — the gap is the **scalar nearest-centroid
  assignment** (n·k·d per iter) vs scipy's vectorized C. At k=4/d=4 (~16 flops/point)
  PARALLELIZING would regress (cheap-work pathology, cf. interpolate/pdist) — the fix
  is **SIMD the distance kernel**, not threads. Bead `→` filed. Double-buffer KEEP.
- **kmeans (early-stop Lloyd): fsci 357.4 µs — 5.9× faster than scipy kmeans2's fixed
  50 iters** (scipy.cluster.vq.kmeans2 has no convergence check). fsci's early-stop is
  a real practical advantage on converged data. KEEP. (Not a per-iteration kernel
  claim — it converges in ~5 iters.)

### ndimage correlate + gaussian_filter (correlate = frankenscipy-e3r7e) — parity / gap
Oracle `docs/perf_oracle_ndimage.py` (scipy.ndimage, 256² image).
- **correlate 5×5: fsci 1099 µs vs scipy 933.7 µs = 0.85× (1.18× slower).** Near-
  parity. The `e3r7e` precomputed tap-delta table is byte-identical (not a regression);
  fsci's direct correlation is within 18% of scipy's C. KEEP.
- **gaussian_filter σ=2: fsci 3238 µs vs scipy 1143 µs = 0.35× (2.83× SLOWER).** NOT my
  optimization. fsci IS separable (per-axis `gaussian_filter1d_axis` passes), so the
  gap is a slow 1D convolution kernel vs scipy's tuned C `correlate1d` — a SIMD/inner-
  loop opportunity (same class as kmeans2/pdist), not a parallelization. Noted for the
  ndimage owner; not reverted (not mine, not a regression).

### Sparse SpMV — `spmv_csr` cached+unrolled row sweep (frankenscipy-2hclc) — ✅ KEEP, scale LOSS closed
Oracle: scipy.sparse.random CSR `.dot(x)` (same n/density; SpMV time≈O(nnz)).
`spmv_csr` is the public serial row-sweep (NOT the parallel internal `csr_matvec`).
The old public route won only tiny calls and lost at scale; the cached-slice +
4-lane unrolled row loop closes those losses without changing accumulation order.
- **n=100 nnz=500: fsci 0.388 µs vs scipy 4.63 µs = 11.9× FASTER.**
- **n=1000 nnz=10k: fsci 7.077 µs vs scipy 8.00 µs = 1.13× FASTER.**
- **n=10000 nnz=100k: fsci 68.82 µs vs scipy 96.95 µs = 1.41× FASTER.**
- Same-process A/B on rch `ovh-a` (`FSCI_PUBLIC_SPMV_AB=1 cargo run --profile
  release-perf -p fsci-sparse --bin perf_csr_matvec`) compared the legacy public
  row sweep to current in one binary: 550 ns→356 ns (1.54×), 12.074 µs→5.741 µs
  (2.10×), 135.043 µs→63.231 µs (2.14×), all `identical=true`.
- Score vs SciPy after this lever: **3 wins / 0 losses / 0 neutral**. Prior
  ledger status was 1 win / 2 losses. Remaining route is explicit SIMD or
  sparse-BLAS-style row blocking only if a fresh profile shows public SpMV still
  matters after this constant-factor win.

### Sparse eigsh / svds (frankenscipy-fo9cj Arnoldi arena) — REJECT, restored route 4W/1L/1N
Oracle: SciPy 1.17.1 `scipy.sparse.linalg.eigsh` / `svds` on the same deterministic
matrix family as the sparse perf bins. rch same-worker A/B on `ovh-a` rejected the
row-major Arnoldi basis arena plus mutable matvec scratch: `eigsh` regressed all rows
(1.667/6.594/16.147 ms vs parent 1.184/5.548/11.599 ms). `svds` movement was too small
to save it (0.99x, 1.06x, 1.01x parent/candidate-style ratios across the sweep), so
the source route was reverted.
- **eigsh n=2000 k=6: fsci 1.184 ms vs scipy 3.000 ms = 2.53x FASTER.**
- **eigsh n=8000 k=6: fsci 5.548 ms vs scipy 2.768 ms = 0.50x (2.00x SLOWER).**
  This is the next real sparse eigensolver loss: optimize the mid-size restart /
  iteration path or matvec throughput, not the discarded basis-arena copy route.
- **eigsh n=20000 k=8: fsci 11.599 ms vs scipy 43.023 ms = 3.71x FASTER.**
- **svds 2200x2000 k=6: fsci 1.191 ms vs scipy 17.567 ms = 14.75x FASTER.**
- **svds 8200x8000 k=6: fsci 4.929 ms vs scipy 4.861 ms = 0.99x neutral.**
- **svds 20200x20000 k=8: fsci 12.534 ms vs scipy 42.018 ms = 3.35x FASTER.**
KEEP restored route. DO NOT retry the row-major `Vec<Vec>` replacement without
allocator/profile proof that Arnoldi basis allocation is again a top-five cost and a
new layout avoids the per-step basis copy cost.

### Gaussian KDE evaluate_many (parallel) — ✅ KEEP (marquee win)
Oracle `docs/perf_oracle_kde.py` (scipy.stats.gaussian_kde, scott bw, n=1000/5000
1-D, evaluate at n points). **fsci 17–18× FASTER:** n=1000 1.06 ms vs scipy 19.09 ms;
n=5000 11.96 ms vs scipy 201.2 ms. KDE evaluation is O(m·n_data) with HEAVY per-point
work (a full sum over the dataset per query) — exactly the profile where the ordered-
slots parallelization pays off, the mirror image of the (reverted) interpolate case
(~30 flops/point). scipy's gaussian_kde is a non-vectorized Python/broadcast path,
which fsci's parallel Rust crushes. **This is the cleanest validation of the gauntlet's
central lesson: parallelize HEAVY per-element work (KDE ✅), not cheap (interpolate ❌).**
KEEP. Conformance green.

### Multiscale graph correlation (MGC) — ✅ KEEP (marquee win)
Oracle `docs/perf_oracle_mgc.py` (scipy.stats.multiscale_graphcorr, n=80, reps=100).
**fsci 21.58 ms vs scipy 295.7 ms = 13.7× FASTER.** MGC is one of scipy's slowest
functions — a pure-Python permutation loop (reps × the O(n²) statistic). fsci's
`mgc_map` is the O(n⁴)→O(n²) prefix-sum form AND the `reps` permutation scoring is
parallelized. Double lever (better asymptotics + parallel heavy work) → big win vs
scipy's non-vectorized path. KEEP. Conformance green.

### Spatial Rotation.apply_many (frankenscipy-w7ocv) — ✅ KEEP (win)
Oracle `docs/perf_oracle_xform.py` (scipy.spatial.transform.Rotation.apply, 8192 pts).
**fsci 12.03 µs vs scipy 28.30 µs = 2.35× FASTER.** The `apply_many` batch path builds
the 3×3 rotation matrix ONCE then applies in a tight Rust loop — 4.5× over the naive
per-point `map(apply)` (54 µs). NOTABLE: this is a CHEAP per-point op (3×3 matvec, ~9
flops) yet fsci WINS at n=8192 — because the kernel is REGULAR (dense matrix, linear
access) and scipy's Rotation.apply carries numpy dispatch + intermediate-array overhead.
Refines the boundary: fsci beats scipy on regular low-overhead batch kernels even when
cheap; it loses only on IRREGULAR kernels where scipy's C is tightly tuned (SpMV gather,
pdist). KEEP. Conformance green.

### Text I/O: loadtxt / savetxt (frankenscipy-fwnb1, d1uxy) — ✅ KEEP (win)
Oracle `docs/perf_oracle_io.py` (numpy.loadtxt/savetxt, 500×20 matrix). **fsci 6.7–7.8×
FASTER:** loadtxt 259.5 µs vs numpy 2022 µs (7.79×); savetxt 631.6 µs vs numpy 4208 µs
(6.66×). numpy's text I/O is pure-Python parsing/formatting; fsci's direct-parse (parse
straight into the output buffer) + `write!`-into-buffer crush it. Same family as
KDE/MGC — fsci wins decisively where the original leans on non-vectorized Python. KEEP.
Conformance green.

### KDTree build + query (frankenscipy-9k50g query) — ✅ KEEP (parity + win vs ELITE C)
Oracle `docs/perf_oracle_kdtree.py` (scipy.spatial.cKDTree, 4096 3-D points). cKDTree
is one of scipy's most-optimized C structures — the hardest target in this suite.
- **build: fsci 809.5 µs vs scipy 767.8 µs = 0.95× (PARITY).** fsci's O(n) select_nth
  median build matches elite C within 5%.
- **query: fsci 1756.7 µs vs scipy 2032.8 µs = 1.16× FASTER.** The dual-tree parallel
  query (`9k50g`) edges out scipy's single-threaded cKDTree — and this is UNDER
  multi-agent contention, so single-tenant the margin is larger.
**Significance:** even vs scipy's BEST C (not a Python path), fsci reaches parity and
WINS on the parallelizable half. This narrows the "irregular-kernel loss" further: the
losses are specific to SpMV-gather and pdist's tight C inner loop, NOT tree/spatial
structures generally. KEEP both. Conformance green.

### Silhouette score (per-anchor parallel) — ✅ KEEP (win, regression-hunt NEGATIVE)
Oracle `docs/perf_oracle_silhouette.py` (sklearn.metrics.silhouette_score, blobs).
**fsci 2.86×→10.6× FASTER:** n=500 720.8 µs vs sklearn 2064 µs; n=2000 3113.5 µs vs
32928 µs. Ran this specifically to HUNT for a second over-eager-parallelization
regression (like interpolate) — found NONE: the per-anchor work is O(n·d) (heavy,
unlike interpolate's ~30 flops), so the parallel gate is well-calibrated and even
n=500 wins 2.86×. The ratio grows with n (parallel scales). KEEP. This negative
regression-hunt result is itself evidence: the cluster/spatial parallelizations are
correctly gated; interpolate was the lone over-eager case (already reverted).

### ndimage zoom — ⚠️ LARGEST LOSS (kernel, not my optimization; bead filed)
Oracle `docs/perf_oracle_zoom.py` (scipy.ndimage.zoom 2×, 256² image). **fsci LOSES
2.25–17.7×:** order=1 85.95 ms vs scipy 4.84 ms (17.7×!); order=3 31.57 ms vs 14.05 ms
(2.25×). **Smoking gun: fsci order=1 (86 ms) is SLOWER than order=3 (31.6 ms)** — the
reverse of correct (bilinear should be cheaper than cubic). The output-pixel loop IS
parallelized (fill_pixels_parallel, gate `pixels·kernel_work≥2¹⁸` fires for both), so
this is NOT the parallelization — it's the per-pixel `sample_interpolated` computing
B-spline weights generically with no fast low-order (bilinear) special case, so order=1
pays nearly the full generic-spline cost AND apparently more (likely a per-pixel
prefilter/weight recompute). **NOT my optimization (the geometric-transform
parallelization is correct/byte-identical; the slow kernel is the underlying spline
interpolation). Bead filed for the ndimage owner.** This is the gauntlet's single
biggest loss and the clearest fix-target: special-case order≤1 (direct bilinear) +
hoist any per-pixel weight setup. Honest LOSS recorded.
- **REPRODUCED (2nd run, integrity check):** order=3 STABLE (31.6→33.2 ms) while
  order=1 consistently far slower (86→133 ms). So the anomaly is a REAL kernel
  pathology, NOT contention variance (a contention spike would have hit order=3 too).
  Refined hypothesis: for cheap order=1 pixels, the per-pixel `thread_local`
  INTERP_SCRATCH borrow + generic B-spline weight path dominates — overhead that is
  amortized away by order=3's heavier interpolation. Fix = add an order≤1 fast path
  (direct linear weights, no thread_local borrow per pixel). Bead `wm14d` confirmed.
- **EXACT ROOT CAUSE (round 14 code read):** in `sample_interpolated`, order 2..=5 use
  the fast `cardinal_reflect_nearest` path (`cardinal_bspline` direct weights, gated
  `matches!(order, 2..=5)`) and order==3 has a Wrap/Constant fast path — but **order=1
  is excluded from BOTH** and falls through to the generic `uniform_interpolation_knots`
  + `eval_bspline_basis_all` (full knot-vector B-spline basis eval per pixel per axis),
  which is far slower than the cardinal kernel. order=0 has its own fast branch. So
  order=1 is the LONE interpolating order with no fast path. **Fix: extend the cardinal
  path to order=1 (`matches!(order, 1..=5)`)** — `cardinal_bspline(1, cc-k)` over the
  3-tap span yields the linear weights `[(floor,1-t),(floor+1,t)]`. CAVEAT: must verify
  byte-identity vs `eval_bspline_basis_all` for order=1 (the linear B-spline weights are
  mathematically equal but the FP computation order differs — needs the conformance
  tests, which exceed this phase's build/bench allowance). Reduced to a ~1-line fix +
  a byte-identity check for the ndimage owner. Bead `wm14d` updated.

### kendalltau (inversion-count O(n log n)) — ✅ KEEP (win small-n, parity at scale)
Oracle `docs/perf_oracle_kendall.py` (scipy.stats.kendalltau, same x/y). fsci **2.59×
faster at n=2048** (230.4 vs 597 µs) but **parity at n=4096** (552.4 vs 537 µs, 0.97×).
scipy has a ~500 µs fixed overhead (array conversion + tie-handling setup) that
dominates at smaller n; at n=4096 both O(n log n) merge-sort kernels converge. Honest
read: fsci's algorithmic kernel MATCHES scipy's C (parity at scale) and WINS where
scipy's per-call overhead dominates — the same low-overhead advantage seen in
SpMV-small/Rotation/transform. KEEP. Conformance green.

### Delaunay (Bowyer-Watson) — ⚠️ COMPLEXITY gap vs Qhull (first asymptotic gap found)
Oracle `docs/perf_oracle_delaunay.py` (scipy.spatial.Delaunay = Qhull, 2-D). fsci
**3.3→5.9× SLOWER:** n=1000 6.53 ms vs 1.98 ms; n=2000 26.31 ms vs 4.49 ms. The
SCALING is the finding: **fsci 6.53→26.31 ms = 4.0× per 2× n (≈O(n²))** while Qhull is
2.27× (≈O(n log n)). Unlike every prior loss (constant-factor kernel gaps), this is an
ASYMPTOTIC gap — fsci's Bowyer-Watson does linear point-location (walk all triangles)
instead of a spatial-index-accelerated locate. The `8d2z2` buffer hoist is a real
constant-factor win but sits atop an O(n²) algorithm, so the gap WIDENS with n. NOT a
regression (the hoist helps); the complexity is the underlying triangulation. Fix:
spatial-accelerated point location (grid/quadtree/jump-and-walk) → O(n log n). Bead
filed for the spatial owner. This is the highest-leverage spatial fix (the others are
SIMD constant-factor; this is an algorithm-class change).

### ✅ ndimage zoom order=1 — FIXED (BOLD-VERIFY, frankenscipy-wm14d) — biggest loss closed
The gauntlet's single biggest loss (zoom order=1 17.7× slower than scipy) is now FIXED
and shipped (commit `3c027183`). Root cause was that order=1 was the lone interpolating
order with no fast path: order=1 Reflect/Mirror is PADDED (coord_offsets=SPLINE_NEAREST_
PAD) so the cardinal `coord_offsets==0` gate excluded it → it fell through to the slow
generic `eval_bspline_basis_all`. Fix: route padded order=1 through the cardinal fast
path with clamp(Nearest) fold (the padding already encodes the reflection, so the linear
support always lands in range) + made `cardinal_bspline` use stack arrays instead of
per-call heap Vecs (hot per-tap-per-pixel path). **MEASURED: order=1 zoom 85.95 ms →
19.41 ms = 4.4× faster; loss vs scipy 17.7× → 4.0× (≈2.4× contention-adjusted); the
order=1>order=3 inversion is gone (19.4 ms < order=3's 54 ms).** Conformance: ndimage
296 passed / 0 failed (verified twice). The residual ~2.4–4× gap is the cardinal_bspline
arithmetic + parallel overhead vs scipy's tight C — a follow-up SIMD/branchless target.
This is the BOLD-VERIFY loop end-to-end: measured loss → root-caused → bold fix →
conformance-verified → measured win → shipped.

2026-06-19 cod-b residual pass: a narrower 2D Reflect/order=1 direct bilinear zoom
path precomputes row/column supports and replaces the generic per-pixel sampler with a
fixed four-load sum. Same-worker `ovh-b` A/B improved 34.034 ms to 7.9684 ms
(4.27× faster), but local SciPy still measured 3.88937 ms, so the lane remains a
2.05× SciPy loss. A serial fill probe regressed to 9.6976 ms and was reverted; do not
retry scheduler-only variants without a fresh profile.

### ndimage rotate — zoom fix's BROAD REACH (frankenscipy-wm14d)
Oracle `docs/perf_oracle_rotate.py` (scipy.ndimage.rotate 30°, 256²). rotate shares
`sample_interpolated` with zoom, so the order=1 cardinal fast path (`3c027183`) rescued
it from the same ~17× padded-order=1 pathology too. MEASURED post-fix: **order=3 6.44 ms
vs scipy 5.58 ms = 0.87× (NEAR-PARITY)** — fsci's cubic-spline rotate is competitive with
scipy's C; **order=1 8.73 ms vs 1.99 ms = 4.4× slower** (residual general-machinery +
parallel overhead, the same gap as zoom order=1 post-fix). Takeaway: one targeted fix
removed the pathology across the WHOLE geometric-transform family (zoom/rotate/affine/
map_coordinates all share the path); the residual ~4× order=1 gap is the general
`sample_interpolated` support-computation machinery — a wholesale specialized-bilinear
rewrite, not the weight arithmetic (the direct-weights micro-opt was measured ~0-gain
and reverted). order=3 is already a release-ready near-parity result.

### gaussian_filter gap DIAGNOSED (refined negative evidence)
Dug into the 2.83× gaussian loss. fsci `gaussian_filter` is SEPARABLE (per-axis 1D),
already uses the fast `convolve1d_along_axis` for inner axes (parallel across slabs),
and only falls back to general `convolve` for the OUTERMOST axis (1 slab). BUT the key
fact: **scipy.ndimage is single-threaded, and fsci's gaussian runs on 64 threads yet is
still 2.83× slower** → fsci's per-element 1D-convolution KERNEL is ~10–60× slower than
scipy's C (parallelism masks it to 2.83×). So the gap is NOT the outer-axis handling or
parallelization — it's the inner dot-product kernel, same SIMD-class gap as pdist/SpMV.
Fix = SIMD-vectorize `convolve1d_along_axis`'s window·weights dot product (conformance
tolerance-OK since gaussian isn't chaotic), but that's shared kernel code. Also checked:
`uniform_filter` already O(1) running-sum, `correlate1d`/`convolve1d` already specialized
1D-axis — the ndimage filters are otherwise well-optimized; the residual is SIMD-kernel.

### ✅✅ Clough-Tocher LOSS → WIN: precompute Bézier patches, 26.6× self-speedup (9l5oo lever)
Oracle `/tmp/oracle_ct.py` (scipy.interpolate.CloughTocher2DInterpolator eval_many, 576
pts / 1024 q). **BEFORE: fsci 2222.8 µs vs scipy 537 µs = 4.1× SLOWER (a LOSS).** The
per-query `clough_tocher_triangle_eval` rebuilt the ENTIRE macro-patch every query — 3
edge vectors, 6 directional derivatives, 19 cubic Bézier control points, AND a neighbour
loop (3 neighbours × barycentric + centroid + division) — all query-INVARIANT. FIX:
split into `clough_tocher_patch → [f64;19]` (the invariant patch) + `clough_tocher_eval_
patch` (the Bézier sum, the only query-dependent step); precompute `patches: Vec<[f64;19]>`
once per triangle in `with_options`. **AFTER: fsci 83.5 µs = 26.6× self-speedup = 6.4×
FASTER than scipy.** A 4.1× LOSS flipped to a 6.4× WIN. BYTE-IDENTICAL (patch + eval are
the original code verbatim, just reorganised), conformance interpolate **227/0**. KEEP.
The 4th application of the precompute-per-element-predicate lever, biggest self-speedup
yet (the neighbour loop made the per-query cost enormous).

### ✅ griddata / LinearND 46.5× faster than scipy + barycentric precompute (9l5oo lever)
Oracle `/tmp/oracle_griddata.py` (scipy.interpolate.griddata linear, 576 pts / 1024
queries, same data as bench_scattered). **fsci griddata 118.3 µs vs scipy 5507 µs =
46.5× FASTER** (eval-only `linear_nd_eval_many` 59.8 µs). The big ratio is fsci's
low-overhead Rust vs scipy's Python griddata + Qhull-setup-per-call. ON TOP of that I
applied the precompute-per-element-predicate lever to `Delaunay2D::find_simplex`:
precompute each simplex's query-invariant barycentric basis (point a + Gram matrix
(d00,d01,d11) + denom) once in `new()`, so the grid-restricted point-location scan does
only the v2-dependent work per query instead of rebuilding the Gram matrix per
(query, candidate). BYTE-IDENTICAL (`SimplexBary::weights` = `barycentric` same float
ops/order), conformance interpolate **227/0**. Monotone eval speedup. KEEP.

### ✅✅ Delaunay LOSS → WIN/PARITY (frankenscipy-9l5oo) — large-n gap closed to parity
The first pass flipped n=1000/2000 by precomputing circumcircles instead of calling the
full in-circle determinant for every (point, triangle) pair. The 2026-06-20 cod-b pass
expanded the gauntlet to n=4000/8000 and found the predicted crossover was real:
pre-grid n=4000 14.935 ms vs scipy 9.50086 ms (1.57× slower), n=8000 55.761 ms vs
20.62714 ms (2.70× slower). New lever: for n>=4096, stable triangle IDs plus a fixed
grid over circumcircle bounding boxes. Each point checks only candidate circles in its
cell and then applies the exact `dist² < r²` predicate; inactive stale IDs are skipped
and an empty candidate lookup falls back to the full active scan. **MEASURED final:
n=1000 0.754 ms vs scipy 1.933 ms (2.56× faster), n=2000 2.613 ms vs 4.550 ms
(1.74× faster), n=4000 9.463 ms vs 9.501 ms (parity), n=8000 20.622 ms vs
20.627 ms (parity). Score: 2 wins / 0 losses / 2 neutral.** Conformance/gates:
spatial lib 208 passed / 0 failed / 2 ignored; e2e_spatial 16/0; check, clippy
`-D warnings`, fmt, UBS clean for touched files. Remaining caution: this is still
Bowyer-Watson with a grid candidate accelerator, not full Qhull-class history-DAG
location; re-measure beyond n=8000 before claiming asymptotic dominance.

## IO crate — head-to-head vs numpy/scipy.io (2026-06-19) — fsci DOMINATES
fsci vs numpy (loadtxt/savetxt) + scipy.io (mmread/mmwrite), in-memory:

| function | fsci | numpy/scipy | ratio |
|---|---|---|---|
| mmread 100×100 | 289 µs | 4282 µs | **14.8× faster** |
| mmwrite 100×100 | 619 µs | 3747 µs | **6.1× faster** |
| savetxt 500×20 | 584 µs | 2951 µs | **5.0× faster** |
| loadtxt 500×20 | 267 µs | 929 µs | **3.5× faster** |

Same structural reason as opt/integrate: fsci's Rust text/MatrixMarket parse+format has no
Python interpreter overhead; numpy/scipy pay it on every cell. IO HARVESTED — fsci dominates.

## Special crate — array (RealVec) sweep vs scipy (2026-06-19) — measured slower, cause CORRECTED
Bench added (`special_array_65536`). fsci vs scipy.special over a 65536 RealVec:

| function | fsci | scipy | ratio |
|---|---|---|---|
| gamma | 1.04 ms | 426 µs | 0.41× (2.4× SLOWER) |
| j0 | 1.08 ms | 664 µs | 0.61× (1.6× SLOWER) |
| erf | 4.49 ms | 757 µs | **0.17× (5.9× SLOWER)** |

⚠️ CORRECTION (my first root-cause was WRONG — I grepped only lib.rs): special DOES
parallelize. `gamma_dispatch`/etc. call `par_map_indices` (defined in airy.rs, gate `n<256
→serial` else `available_parallelism`), so RealVec maps the kernel across all cores. The
real causes of the measured slowness: (1) `par_map_indices` SPAWNS ~64 threads PER CALL (no
pool) — slow under heavy multi-agent load (the same contention that regressed my minmax-slab/
linkage-distance attempts; absolute bench numbers here are unreliable), and (2) fsci's
per-element kernels are ~2× scipy's Cephes (e.g. gamma). So the parallel path may NOT pay for
cheap kernels at moderate n: spawn overhead can exceed the per-core work. POSSIBLE LEVERS (in
the shared, actively-developed special crate — flagged, not dived into): a persistent thread
POOL (amortize the per-call spawn) and/or a COST-AWARE gate (cheap kernels need a higher n
threshold than 256). The measured ratios are contention-influenced; the bench is the harness
to re-check on an idle machine. NOT a clean serial-dispatch loss as first claimed.
KERNEL FINDING (dug to erf_scalar, error.rs:206): erf (4.49 ms) is 4× slower than the more-
expensive gamma (1.04 ms) because the kernel is ITERATIVE — `erf_series_real` (Maclaurin, up
to 80 terms for |x|<1) + `erfc_cf_real` (continued fraction for x≥1, ~10-30 iters) — whereas
scipy's Cephes uses a fixed-degree RATIONAL approximation (~10 mults, no loop). REAL LEVER:
port Cephes' rational erf/erfc (faster + matches scipy exactly, conformance-safe since scipy
IS Cephes). Out of MY reach (needs Cephes's exact coefficients — no source access to
transcribe — or a custom minimax rational fit). Flagged for the special owner: the per-element
kernel speed (not just the thread spawn) is the real gap for the iterative special functions.

## Opt crate — minimize sweep vs scipy (2026-06-19) — fsci DOMINATES (largest ratios of phase)
fsci vs scipy.optimize.minimize(method='BFGS') on Rosenbrock, x0=zeros:

| dim | fsci | scipy | ratio |
|---|---|---|---|
| 2 | 10.7 µs | 3914 µs | **367× faster** |
| 5 | 21.7 µs | 10672 µs | **491× faster** |
| 10 | 76.4 µs | 27285 µs | **357× faster** |

Same structural reason as solve_ivp, amplified: BFGS does MANY objective+gradient
evaluations (numerical gradient + line search), each a Python callback in scipy; fsci runs
the whole optimizer + Rust objective with zero callback overhead. Optimizer/root/ODE crates
(any iterative solver over a user function) are fsci's biggest categorical win vs scipy.

## Integrate crate — ODE sweep vs scipy (2026-06-19) — VOID-NONULL
This was a separate-invocation timing, not admissible incumbent evidence: it
recorded no A/A null control, no executed-binary SHA-256, and no counted
mechanism capable of distinguishing the solver from the harness. The figures
below are retained only as resurrection history and MUST NOT be quoted as
competitive claims.

| ODE | fsci | scipy | ratio |
|---|---|---|---|
| exponential decay (0,10) | 18.589129 µs | 1443.255860 µs | 77.64x (VOID-NONULL) |
| Lorenz (0,1) | 28.266539 µs | 2062.735365 µs | 72.97x (VOID-NONULL) |

**Concrete retry predicate (satisfied 2026-07-28):** run the exact two fixtures
through a genuine SciPy arm side-by-side in the same invocation, give both arms
independent A/A null controls, self-report the executed ELF SHA-256, verify the
full final vector, and decide on the bootstrap-median CI. The live-arm row at
the end of this ledger supersedes these values.

### frankenscipy-bpzha: RK scratch double-buffer measured reject
The solver-owned scratch/double-buffer idea was tested and reverted. It had one
scalar exponential win on `hz2` (`17.356838 -> 13.863079 µs/call`) but regressed
paired Lorenz/vector rows on `hz2` (`21.951172 -> 23.402816 µs/call`), `hz1`
(`28.621224 -> 31.335899 µs/call`), and `ovh-a`
(`20.597014 -> 32.037205 µs/call`). Final helper-dispatch sanity also measured
`27.755498 µs/call` on exponential on `ovh-b`, a red flag against all parent
exponential rows. Decision: reject/revert; next integrate work should come from a
fresh profile, not from RK scratch reuse.

## Ndimage crate — filter/morphology sweep vs scipy (2026-06-19)
fsci vs scipy.ndimage (256² / 160² images):

| function | fsci | scipy | ratio |
|---|---|---|---|
| median_filter 160² s7 | 1.84 ms | 6.03 ms | **3.28× faster** |
| median_filter 160² s15 | 9.32 ms | 26.46 ms | **2.84× faster** |
| minimum_filter 256² s7 | 2.24 ms | 0.99 ms | 0.44× (2.26× slower, OPEN) |
| minimum_filter 256² s15 | 1.84 ms | 1.01 ms | 0.55× (1.82× slower, OPEN) |
| binary_erosion 256² s7 (IMPROVED) | 1.81 ms | 0.60 ms | 0.33× (3.0× slower, was 3.7×) |
| binary_erosion 256² s15 (IMPROVED) | 1.62 ms | 0.81 ms | 0.50× (2.0× slower, was 2.76×) |

### ✅✅✅ binary_erosion LOSS → WIN: 2D BIT-PACKING (radical lever, flipped)
The radical lever LANDED — byte-identical on the first try. `binary_erode_bitpack_2d`:
pack each row into u64 words (64 px/word), erode HORIZONTALLY via shift-AND (out[c] = AND of
in[c-lo..c-lo+size-1], computed as the left-anchored `size`-fold shift-AND then `shift_bits_
down` by `size-1-lo` to re-center) and VERTICALLY via word-AND of the `size` rows in the
window. Constant-0 border falls out free (out-of-range bits/rows are 0 → AND is 0). Gated to
2D + size<64 (single word-boundary shifts); N-D / huge windows fall back to the running count.
**MEASURED: s7 2.20 ms→630 µs (3.5× self) = PARITY vs scipy 596 µs (was 3.7× slower); s15
2.22 ms→180 µs (12× self) = 4.5× FASTER vs scipy 805 µs (was 2.76× slower).** Conformance
ndimage **296/0**. A 2.76–3.7× LOSS flipped to parity-to-4.5×-WIN. The op-count math (~30×
fewer ops) predicted it. KEY: erosion AND is commutative across axes so horizontal-then-
vertical order is byte-identical; the Constant-0 border needs no special-casing in bit-space.
EARLIER (superseded): running-count partial got 1.2–1.4×; the bit-pack subsumes it.
SYMMETRIC: `binary_dilate_bitpack_2d` (OR instead of AND, reflected-SE origin lo=size/2+refl,
out-of-range = OR-identity 0). **s7 ~2.2 ms→642 µs (3.4× self) = 2.3× slower (was 6.6×);
s15 ~2.2 ms→159 µs (14× self) = 3.3× FASTER vs scipy 521 µs (was 3.1× slower)**, 296/0.
dilation-s7 still loses to scipy's very-fast 279 µs (mostly-set image) but improved 3.4×.
Net: binary morphology 3 of 4 cases now parity-or-WIN (was all losses).
BROAD REACH (measured, bench added): binary_opening (erosion∘dilation) and binary_closing
(dilation∘erosion) INHERIT the bit-pack → **opening s7 2.8× / s15 9.0× faster; closing s7
1.6× / s15 4.7× faster than scipy** (scipy 2.2–10.8 ms — it does NOT decompose the box
structure, scanning the full s² footprint, where fsci's bit-packed separable path is
1.1–1.7 ms). The bit-pack flip propagates to all higher-level binary morphology (opening/
closing/tophat/fill_holes). SAME lever applied to
`binary_dilation` (`binary_dilate_separable`: running count of ONES > 0, origin-aware lo =
size/2 + refl to match the reflected-SE max-filter; even sizes use refl=−1): byte-identical
**296/0**, dilation ~1.84/1.64 ms (same ~1.2–1.4× self-speedup). Dilation is still 3–6.6×
slower than scipy (279/521 µs — the mostly-set bench image favours scipy's algorithm); same
bit-pack lever needed to flip. Both binary-morph paths now use the simpler integer-count
kernel; the float deque remains for non-default origins + float minmax.

median is a big WIN. minimum_filter + binary_erosion are CONSTANT-FACTOR losses: both go
through `separable_minmax_filter` → `minmax_filter_along_axis`, already an O(1)/pixel
monotonic-deque sliding min (flat across window size, confirmed), so the gap is Rust-deque
overhead vs scipy's specialized C, NOT algorithm. RADICAL LEVER (future, substantial):
**binary_erosion/dilation on a binary image should bit-pack** (64 px/u64; horizontal =
`s` shift-ANDs per word, vertical = `s` word-ANDs per row) → ~10-30× over the float deque,
would FLIP both binary-morph losses to wins. scipy's NI_BinaryErosion is a specialized
binary path; fsci runs the general float min-filter on booleanized data. Byte-identical
(same 0/1 output). Needs exact window-origin-semantics matching with the deque path —
high-risk multi-cycle, filed as a focused future effort, not started blind. minimum_filter
(float) has no bit-pack lever; its constant factor needs SIMD on the deque (hard). ANALYSIS
(no clean lever — DON'T re-chase byte-identically): the monotonic deque is already amortized
O(1)/pixel (~1 total_cmp); van Herk/Gil-Werman does MORE (3 total_cmp/pixel: prefix+suffix+
combine) so it's not faster; shift-min (f64-min shifted s times, the bit-pack analogue) is
vectorizable and would win for small s, BUT requires `f64::min` not `total_cmp` → silently
changes NaN semantics (no NaN minmax test exists, so it'd pass conformance, but it's a latent
behaviour divergence from scipy — NOT shipped). A true flip needs an explicit SIMD min with
total_cmp NaN ordering. The deque is the right scalar algorithm; the gap is scipy's tighter
vectorized C. EMPIRICALLY CONFIRMED (attempted no-NaN-gated shift-min, REVERTED clean 296→
237/1): `f64::min`/`max` diverge from `total_cmp` not only on NaN but on SIGNED ZEROS —
`total_cmp(-0.0,+0.0)=Less` so min=-0.0/max=+0.0, but `f64::min(-0.0,+0.0)` is order-dependent
(x86 minsd). The byte-for-byte `separable_minmax_matches_rank_filter` test (which seeds ±0.0)
caught it. A correct vectorized version needs the f64→monotonic-i64 transform + SIMD i64 min,
which is AVX-512-only (AVX2 lacks `vpminsq`) → not portable. NO clean lever; the conformance
gate prevented shipping a subtly-wrong (signed-zero) result.
DEFINITIVE (2nd attempt, REVERTED clean): the SIGNED-ZERO-CORRECT version — shift-min on the
ordered-i64 (`f64::total_cmp`'s own monotonic transform `bits ^ ((bits>>63 as u64)>>1)`, an
involution) — IS byte-identical for all inputs (296/0, signed-zero test passes). But PERF
REGRESSES: s7 ~same (2.21 vs 2.24 ms), s15 SLOWER (2.40 vs 1.84), s31 much SLOWER (3.72 vs
1.89). The i64 min/max didn't autovectorize on AVX2 (no `vpminsq`), so the O(s·n) shift loses
to the deque's amortized O(1). CONCLUSION: the monotonic-deque IS the optimal portable scalar
algorithm for float window min/max; beating scipy needs explicit C-style SIMD (vpminsq/AVX-512
or hand-written AVX2 compare+blend on the i64 reps). Float minmax CLOSED — no portable lever.
DEAD-END (reverted clean, 296/0): rewrote `minmax_filter_along_axis` to the correlate1d
slab pattern + parallelize over outer slabs (byte-identical). REGRESSED ~1.5-2× even after
hoisting the per-slab VecDeque alloc to per-thread reuse. At 256² the filter is below the
parallel gate (work < 2¹⁸ → serial) so the parallel path wasn't even engaged — the slab
restructure measured slower under heavy multi-agent load, and the parallel path would add
core-contention at larger sizes. Don't re-chase slab-parallel for minmax; the win (if any)
needs the bit-pack (binary) or SIMD-deque (float) lever, not coarse line parallelism.

## Cluster crate — head-to-head sweep vs scipy (2026-06-19)
fsci vs scipy.cluster.hierarchy: **cophenet n400 206µs vs 290µs = 1.40× faster** (WIN);
**linkage_average n400 1.847ms vs 1.655ms = 1.12× slower** (near-parity, OPEN). GMM/
silhouette already wins (gauntlet ledger); kmeans2 chaotic-iteration loss (unfixable
byte-identically). DEAD-END (reverted clean): parallelizing linkage's O(n²·dim) distance
build via split_at_mut row-blocks was BYTE-IDENTICAL (193/0) but **2.5× SLOWER** (1.847→
4.61ms) — the distance build is NOT the bottleneck (the NN-chain `agglomerate_nnarray`
is), so parallelizing the small part added thread overhead + 2× redundant sqrt + cache
thrash on the 5 MB arena. The 1.12× gap lives in the NN-chain, not the distance fill —
don't re-chase the distance parallelization.

## Stats crate — head-to-head sweep vs scipy (2026-06-19) — fsci DOMINATES
fsci vs scipy.stats / scipy.stats.qmc — all WINS, no losses:

| function | fsci | scipy | ratio |
|---|---|---|---|
| QMC Sobol 4096×2 | 9.6 µs | 200.7 µs | **20.8× faster** |
| QMC Halton 4096×4 | 95.5 µs | 1326.7 µs | **13.9× faster** |
| discrepancy centered 512×2 | 221 µs | 622 µs | **2.8× faster** |
| discrepancy mixture 512×2 | 288 µs | 720 µs | **2.5× faster** |
| kendalltau 2048 | 196 µs | 380 µs | **1.94× faster** |
| discrepancy wraparound 512×2 | 225 µs | 412 µs | 1.83× faster |
| discrepancy l2_star 512×2 | 227 µs | 383 µs | 1.69× faster |
| kendalltau 4096 | 440 µs | 533 µs | 1.21× faster |

Stats is HARVESTED — fsci wins every measured function (QMC sampling especially, where
scipy's Python-loop generators are 14-21× slower than fsci's vectorized Rust).

### ✅ remez even-WLS cos-basis Chebyshev recurrence (1.17×, signal)
The even-numtaps remez fallback (WLS frequency-sampling; the benched odd-257 path uses the
already-efficient PM+barycentric route) rebuilt its cos-basis with `n_coeffs` separate
`cos(2π·j·f)` calls per grid point. Replaced with the Chebyshev recurrence `cos(jθ)=2cos(θ)
cos((j-1)θ)-cos((j-2)θ)` — ONE cos() per grid point. **A/B MEASURED: 3.58→3.06 ms = 1.17×**
(the O(ng·n_coeffs) cos was ~15% of the work; the O(ng·n_coeffs²) normal-equations build
dominates the rest). Accurate to ~1e-14 (within remez's ~1e-6 tolerance), conformance signal
**707/0**. scipy.signal.remez ERRORS on this case (PM non-convergence) so fsci's WLS fallback
has no head-to-head, but it's a real self-speedup. KEEP (not ~0-gain). The recurrence lever
applies to any cos(2π·k·f) response-basis loop.

### ✅ freqz Horner's method (5.2× self-speedup, signal) — closes a 12× loss to 2.3×
`eval_poly_on_unit_circle` (used by `freqz`/`freqz_with_whole`, the frequency-response
function) computed `cos(kω)` AND `sin(kω)` PER COEFFICIENT per frequency — despite a comment
falsely claiming "Horner's method." Implemented ACTUAL Horner: z⁻¹=e^{-jω} via ONE cos+sin
per frequency, then a complex-multiply accumulation `acc=acc·z⁻¹+c[k]`. **A/B MEASURED on a
128-tap FIR / 512 freqs: 978→187 µs = 5.2×.** Same polynomial value as the direct sum
(~1e-13), conformance signal **707/0**. Head-to-head vs scipy.signal.freqz (81 µs, FFT-based):
fsci was **12× slower → now 2.3× slower** — Horner cuts most of the gap; the residual is the
O(n_freqs·n_coeffs) Horner vs scipy's O(n log n) FFT-of-coefficients. ✅ RESIDUAL NOW DONE
(see freqz FFT-hybrid below). Added freqz/fir128_512 bench.

### ✅✅ freqz FFT-hybrid (FLIPS the residual loss to a WIN, signal)
DONE — the flagged residual. B(e^jω)/A(e^jω) on the linear ω-grid IS the DFT of the zero-
padded coefficients (whole: nfft=n; half: nfft=2n), so for large filters use `fsci_fft::fft`
(O(N log N)) instead of the O(n·n_coeffs) Horner loop; small filters (b.len+a.len < 16 or
n < 64) keep Horner (so biquads don't regress). **MEASURED freqz/fir128_512: Horner 187 µs →
FFT 49.9 µs = 3.7×; vs scipy 81 µs now 1.6× FASTER.** Full freqz journey: 12× slower → 2.3×
slower (Horner) → **1.6× faster than scipy**. Same response within ~1e-13, conformance 707/0
(tolerance tests; biquad path unchanged). Falls back to Horner if the FFT errors.
PLUS: routed `group_delay_from_ba` + `magnitude_response_db` (two MORE functions with the
same inline per-coefficient cos/sin loop, not previously using the helper) through the Horner
`eval_poly_on_unit_circle` — inherit the 5.2× large-filter speedup, conformance 707/0. Also
NOTED (bigger lever, not done): the MFCC power spectrum (lib.rs ~5949) is a naive O(N²) DFT
(`re += s·cos(2πkn/N)`) that should be an fsci_fft O(N log N) FFT.

### ✅✅ mfcc naive-DFT → fsci_fft (181× !!, signal) — MARQUEE algorithmic flip
DONE. The mfcc per-frame power spectrum computed `|Σ s[n]·e^{-j2πkn/N}|²` as a NAIVE O(N²) DFT
(n_freq×frame_len per frame, every frame). Replaced with `fsci_fft::fft` (O(N log N)) — pad
frame to complex, FFT, |·|²/n_fft. **A/B MEASURED on 16384 samples / frame_len 512:
149.45 ms → 0.825 ms = 181×.** Same DFT value (~1e-13), conformance signal **707/0**. The
single biggest self-speedup of the phase — a naive DFT in a hot per-frame loop is catastrophic
(149 ms). LEVER: grep nested-loop `cos(2πkn/N)`/`sin` (DFT-by-hand) in any transform/feature
fn → replace with fsci_fft. Added mfcc/16384_frame512 bench.

## Signal crate — head-to-head sweep vs scipy (2026-06-19)
Oracle `docs/perf_oracle_signal.py` + `/tmp/oracle_sig2.py`. fsci vs scipy.signal:

### ⚠️ OPEN LOSS — fftconvolve 2.7× slower; bottleneck is fsci_fft (8l8r1's crate), not signal
fftconvolve 4096×257 same: **fsci 323 µs vs scipy 119.6 µs = 2.7× SLOWER.** scipy pads to
next_fast_len (~4400, 5/7/11-smooth) + uses rfft. TRIED BOTH in fftconvolve, BOTH FAILED →
REVERTED (signal lib.rs back to origin, clean): (1) `fsci_fft::next_fast_len` (4400) +
rfft → **393 µs, REGRESSION** — fsci_fft's mixed-radix path is much slower than its radix-4
power-of-two path, so the smaller 5-smooth size is a net loss. (2) power-of-two + rfft →
**326 µs, ~0 gain** — fsci_fft's `rfft` is NOT faster than its complex `fft` (no real-
symmetry speedup). CONCLUSION: the gap is inside **fsci_fft** (no fast rfft, slow mixed-
radix), 8l8r1's crate — not fixable from signal. Routed to 8l8r1. welch 8× FASTER, so
fsci's own spectral path is fine; only the raw FFT primitive lags. Also welch/coherence
already win, so this is isolated to convolution.

| function | fsci | scipy | ratio |
|---|---|---|---|
| fftconvolve 4096×257 | 323 µs | 119.6 µs | **0.37× (2.7× SLOWER — fsci_fft, OPEN)** |
| **firls 257 (FIXED)** | **296.5 µs** | 366 µs | **1.24× faster** (was 0.42×) |
| filtfilt 4096 biquad | 80.3 µs | 120.2 µs | 1.50× faster |
| sosfilt 4096×2 | 34.0 µs | 46.0 µs | 1.35× faster |
| lfilter 4096 biquad | 37.4 µs | 24.5 µs | **0.65× (1.53× SLOWER — OPEN)** |

### ✅✅ firls LOSS → WIN: precompute integrate_cos table (9l5oo lever, signal crate)
firls builds the LS normal-equations matrix Q[i,j] = w/2·(∫cos(2π(i-j)f)df + ∫cos(2π(i+j)f)df)
over an O(n²) (i,j) double loop, calling `integrate_cos` (2 sin each) PER CELL — but it
depends only on the integer args (i-j) and (i+j). FIX: precompute `ic[arg]` once per band
(offset-indexed over [-(m)..2m], same arg signs ⇒ BYTE-IDENTICAL via exact negation), read
Q from the table. O(n²) sin → O(n). **MEASURED: 873.6 → 296.5 µs = 2.95× self-speedup =
1.24× FASTER than scipy** (was 2.38× slower). Conformance signal **707/0**. The precompute-
element-invariant lever's 5th call site, first OUTSIDE interpolate/spatial. OPEN: lfilter
1.53× slower (sequential IIR recurrence — scipy's tight C, no obvious lever).

## Interpolate crate — FULL head-to-head sweep vs scipy (2026-06-19)
Measured every major interpolator (oracles `docs/perf_oracle_{griddata,clough_tocher,rect}.py`
+ `/tmp/oracle_{1d,rgi}.py`). fsci DOMINATES or matches scipy across the board — no
remaining losses:

| function | fsci | scipy | ratio |
|---|---|---|---|
| griddata / LinearND (576/1024) | 118 µs | 5507 µs | **46.5× faster** |
| CloughTocher eval (576/1024) | 83.5 µs | 537 µs | **6.4× faster** |
| RegularGrid nearest (32³/4096) | 69.8 µs | 361 µs | **5.2× faster** |
| RegularGrid linear (32³/4096) | 178 µs | 608 µs | **3.4× faster** |
| CubicSpline construct (1024) | 25.1 µs | 237.7 µs | **9.5× faster** |
| CubicSpline eval (1024/4096) | 38.8 µs | 73.6 µs | **1.9× faster** |
| interp1d linear (4096/8192) | 39.2 µs | 38.4 µs | parity |
| RectBivariate eval_grid (32²→64²) | 65.8 µs | 48.3 µs | 0.73× (near-parity, was 0.20×) |

LinearND/Clough-Tocher/eval_grid wins came from the precompute-element-invariant lever
(this phase); cubic/RGI/interp1d were already competitive. The ONLY non-win is eval_grid
at 1.36× off scipy's elite Fortran (down from 5.1×). Interpolate is HARVESTED.

## BOLD-VERIFY phase outcome (implemented levers, not just measured)

This phase moved from MEASURING gaps to FIXING them, conformance-gated via `cargo test`:
- **✅ FIXED & SHIPPED — zoom order=1** (`wm14d`, `3c027183`): the gauntlet's biggest
  loss, 17.7× → 4.0× slower (85.95→19.41 ms, 4.4× faster). Root-caused to order=1 being
  the lone interpolating order with no fast path (padded coord_offsets hid it). Cardinal
  fast path for padded linear + stack-array `cardinal_bspline`. Conformance 296/0.
- **✅ BROAD REACH — rotate/affine/map_coordinates** share `sample_interpolated`, so the
  same fix rescued them. Measured: **rotate order=3 6.44 ms vs scipy 5.58 ms = NEAR-PARITY**;
  order=1 4.4× (residual). The whole geometric-transform family is now release-ready at
  order=3 and pathology-free at order=1.
- **↩️ REVERTED ~0-gain** — direct order=1 weights (skip cardinal_bspline calls): measured
  no gain (bottleneck is the general support-machinery, not the weight arithmetic).

**Remaining LOSS gaps — all assessed, all need substantial SIMD/algorithm work (prioritized):**
1. `nm8ex` pdist — needs SIMD distance kernel (the parallel path HELPS; the serial kernel
   is ~10–60× slower than scipy C). NOT a gate fix.
2. `9g6ku` kmeans2 — SIMD distance, but iterative/CHAOTIC so SIMD FP-reorder breaks
   conformance (a 1-ULP distance change cascades to a different clustering). Needs the
   gemm-trick (≠ byte-identical) + tolerance acceptance.
3. gaussian_filter 2.83× — specialized 1D-axis correlate (routes through shared `convolve`).
4. geometric order=1 4× — specialized 2D bilinear bypassing the support machinery (low-ROI;
   order=3 already near-parity).

## Release-readiness summary (CrimsonForge beads, as of this round)

5 beads measured head-to-head vs scipy/sklearn (release, 64 cores, multi-agent load):
- **3 KEEP (wins):** GMM E-step parallel (4–11×), distribution pdf_many/pmf_many
  (3–98×). These are release-ready marquee wins.
- **1 KEEP (parity):** AP responsibility parallel (1.03–1.28× vs sklearn; the
  parallelization itself is a real 2× internal). Acceptable; not a differentiator.
- **1 REVERT:** interpolate `evaluate_many` parallel (0.88× regression) — reverted,
  byte-identical hoist preserved.
- **1 LOSS (not mine):** spatial pdist 2.7–9× slower than scipy → bead `nm8ex`
  filed for the spatial owner (over-eager gate + slow serial kernel).

**Cross-cutting lesson for release:** parallelize only HEAVY per-element work. The
work-gate threshold must scale with the COST of the work unit, not just its count —
2¹⁶ gaussian/exp evals (GMM) is worth parallelizing; 2¹⁶ flops (interpolate, pdist
low-dim) is not. Gates expressed as raw `count·dim` flop-products fire too early for
cheap kernels. Recommend auditing every `< 1 << 1x` parallel gate in the codebase
against the per-element op cost.

### Updated tally (11 beads, rounds 1–3)
- **WINS (algorithmic / byte-identical lever + heavy work):** GMM 4–11×, distribution
  3–98×, kmeans early-stop 5.9×, cophenet 1.83×, AP parallel 2.02× internal.
- **PARITY (within ~20% of scipy's tuned C):** AP vs sklearn, linkage 0.83×,
  correlate 0.85×.
- **REVERTED:** interpolate evaluate_many parallel (0.88×).
- **KERNEL GAPS (fsci scalar inner loop vs scipy SIMD/C; flagged, not mine to fix):**
  pdist 2.7–9× (`nm8ex`), kmeans2 2.4× (`9g6ku`), gaussian_filter 2.83×.

**Emerging release pattern:** fsci WINS decisively where the lever is algorithmic
(better asymptotics, early-stop, normalizer-hoist) and the work is heavy; it reaches
PARITY-to-LOSS on tight inner numeric kernels (distance, 1D convolution, centroid
assignment) where scipy's C is SIMD-vectorized and fsci's is scalar. **The highest-
leverage release work is SIMD-vectorizing those 3–4 inner kernels** (`nm8ex`/`9g6ku`
+ ndimage 1D filter) — NOT more threads (cheap-per-element parallelism regresses, as
the interpolate revert proves). The byte-identical alloc/precompute/batch wins are all
safe KEEPs by construction.

## Notes / negative evidence

### ✅ RectBivariateSpline.eval_grid 3.75× self-speedup — 5.1× loss → near-parity (separable basis)
Oracle `docs/perf_oracle_rect.py` (scipy.interpolate.RectBivariateSpline `(q,q,grid=True)`,
32×32 → 64×64, kx=ky=3). **BEFORE: fsci 246.9 µs vs scipy 48.3 µs = 5.1× SLOWER.** `eval_grid`
ran the full scalar de Boor recurrence per evaluation (`eval_parts`: span search + per-step
alphas + blend) — the x-recurrence rebuilt for all `ny` rows per xv, the y-recurrence per
(xv,yv). FIX: adopt scipy's FITPACK `bispev` SEPARABLE approach — precompute each axis' k+1
non-zero B-spline basis weights ONCE per query coord (`bspline_basis_funs`, Cox-de Boor
A2.2 with the standard 0/0→0 guard so clamped end-knots are safe), then tensor-contract the
(kx+1)×(ky+1) coefficient window. Added `BSpline::find_span_n` (span by count). **AFTER:
fsci 65.8 µs = 3.75× self-speedup; now 1.36× of scipy (was 5.1×) — near-parity vs elite
Fortran.** NOT byte-identical (different summation order, ~1e-13) but conformance interpolate
**227/0** (rect tests are 1e-10 tolerance vs analytical, incl. the eval(0,0)/(1,1) clamped-
knot boundaries — my clamped-knot worry was unfounded; BasisFuns is built for clamped knots).
KEEP — a 3.75× gain that nearly closes a 5.1× loss. Residual 1.36× is scipy's tighter
vectorized contraction; a SIMD/unrolled kx=ky=3 contraction could reach parity (future).

- The ~50 byte-identical allocation/precompute/batch wins (buffer reuse, mem::take,
  loop-invariant hoist, interval binary-search, write!-amplification, retain) carry
  **no correctness-regression risk** and are monotone by construction (removing an
  alloc/recompute cannot be slower), so they are NOT individually re-benched here;
  the gauntlet revert-risk lives in the **parallelizations** (spawn overhead at small
  n) — those are gate-validated above.
- AP availability-update parallelization is the one OPEN lever surfaced by measurement.

### 🔬 pdist/cdist parallel gate parallelizes BELOW spawn break-even (bead nm8ex) — ROOT-CAUSED, handed to MistyBirch
MEASURED: pdist/euclidean/256 (N=256, d=4) = ~2.68ms vs scipy ~96µs (28×), but SERIAL would be
~131µs (~1.4× scipy = near parity). The 20× inflation is a GATE BUG: `cdist_thread_count`
(spatial lib.rs ~913) goes parallel when `work = na·nb·dim ≥ 1<<18` (262144). N=256,d4 hits
work==262144 exactly → spawns ~64 OS threads (`cores.min(na/2)`) for ~131µs of serial work.
Spawning ~64 threads costs hundreds of µs, so parallel LOSES to serial even on an idle machine
— the gate parallelizes below the spawn break-even. FIX (byte-identical, serial==parallel
offset-fill): raise threshold to `1<<21` (2M ≈ 2·spawn/per-op break-even) so only genuinely
large matrices parallelize; cap thread count for medium work. spatial lib.rs is RESERVED by
MistyBirch → sent the finding+fix via agent-mail (msg 1336) rather than collide. This is the
real mechanism behind the documented pdist loss: it's a parallel-gate-below-break-even bug, NOT
the SIMD kernel (sqeuclidean is already explicit Simd<f64,8>) and NOT a structural layout wall.

### ⚖️ ndimage_filter_thread_count work-capped threads — PRINCIPLED but UNMEASURABLE (reverted)
Hypothesis (byte-identical): cap thread count by work (`min(cores, pixels/2, work>>18)`) so each
thread does ≥256k ops — a separable σ=2 Gaussian pass (~1.1M work) otherwise spawns 64 threads
for ~17µs each. Conformance 297/0 (chunk count ⊥ per-pixel value). BUT the A/B was destroyed by
RAMPING multi-agent load: gaussian_sigma2/256 measured 3.73ms (orig, early window) → 5.20ms
(capped, mid) → 6.70ms (reverted, late) — a monotonic climb that is the LOAD, not the change.
Per demonstrate-or-revert + "same-worker A/B in ONE binary mandatory" (cross-run variance ≫
signal here), REVERTED. The lever is sound for normal/idle machines (fewer threads = less spawn
for medium filters) but needs a same-process atomic-toggle bench in an idle window to prove.
Flagged. (Sibling of the pdist nm8ex gate finding handed to MistyBirch.)

### 📋 PARALLEL-GATE AUDIT (all crates) — pdist is the lone bug; rest are correctly designed
Audited every `*_thread_count` / `work < 1<<N → serial` gate for the gate-below-spawn-break-even
bug. Verdict by per-op cost (the real determinant — cheap multiply/subtract ops need a HIGH
threshold; expensive sort/trig ops can use a low one):
- **pdist/cdist** `cdist_thread_count` 1<<18, cheap subtract-square → **BUG** (handed to MistyBirch, nm8ex).
- **interpolate** `par_query_map` 1<<18 → OK: cost-aware, caller passes `work_per_query`.
- **stats** `compute_row_ranks` 1<<18, **stats** + **cluster** `landmark_isomap` 1<<16 → OK: O(n log n) sort per row.
- **signal** `lombscargle_thread_count` 1<<16 → OK: sin/cos per op (expensive).
- **ndimage** `ndimage_filter_thread_count` 1<<18, cheap mul-add → benched cases (gaussian 1.1M,
  correlate 1.6M) sit AT/above break-even, not clearly below; thread-cap fix unmeasurable under
  ramping load (reverted). The class is otherwise clean — no further gate bugs in my crates.

### ✅ RESOLVED: ndimage filter thread-cap is NEUTRAL (load-invariant same-process A/B)
Built the tool the degraded environment demanded: a same-process atomic-toggle A/B (FILTER_WORK_
CAP_AB, interleaved OFF/ON 50× in one process → load cancels). VERDICT for gaussian_sigma2/256:
cap OFF 5.889 ms vs cap ON 5.980 ms = **NEUTRAL** (~1.5%, within noise). The work-cap does NOT
help under contention — the hypothesis (fewer threads = less oversubscription) is REFUTED by
reliable measurement. Not shipped; toggle+test removed (ndimage back to origin). Supersedes the
earlier "unmeasurable" note. LESSON: the same-process interleaved A/B is THE working method for
contention-sensitive levers when separate-run benches drift 2×; it cleanly settled this one as
neutral. (The pdist nm8ex gate remains a real bug — its fix is math-provable, no A/B needed.)

### ✅✅ erf/erfc Cephes rational kernel (5.0× self-speedup, FLIPS 5.9× loss → 1.2× parity)
The WORST special loss. erf_scalar used an iterative Maclaurin series (≤80 terms) + Lentz
continued fraction (~30 iters for x≥1); scipy's xsf uses Cephes' fixed-degree RATIONAL erf/erfc.
Ported the EXACT Cephes T/U (erf) and P/Q/R/S (erfc) coefficients (fetched from scipy/xsf via
gh) → byte-identical to scipy.special.erf/erfc. **MEASURED special_array_65536/erf: 4.49ms →
904µs = 5.0×; vs scipy 757µs: 5.9× slower → 1.2× (near parity).** Conformance: all erf/erfc/
ndtr/erfcx/erfinv/erfcinv tests PASS. (4 unrelated tests fail on origin — digamma/polygamma/
exp2/powm1 — another agent's in-progress gamma/convenience work; those fns don't call erf, so
not caused by this change; verified by static isolation.) LEVER PAID OUT: fetch scipy's xsf
Cephes coefficients via gh + port the rational → byte-matches scipy AND replaces iterative
kernels. Removed now-unused erf_series_real + erfc_cf_real (erfc_cf_h kept for erfcx_cf_real).

### 📋 Remaining special-kernel Cephes-port candidates (lever PROVEN via erf, lower ROI)
After the erf 5.9×→1.2× flip, audited the other measured special losses for the same
iterative-kernel→Cephes-rational lever:
- **gamma (2.4×)**: gamma_core uses LANCZOS — a fixed ~15-coeff approximation, NOT iterative.
  Its gap is the `powf(x, x+0.5)` cost vs Cephes' recurrence-to-[2,3] rational (avoids powf for
  moderate x). Nuanced, not a clean flip. gamma.rs also has another agent's in-progress
  breakage (digamma/polygamma failing). → leave to that owner.
- **j0/j1/y0/y1 (j0 1.6×)**: j0_core uses a genuine convergence-loop power series for x<14
  (`j0_series_small`, ~15-25 terms in the bench range). CLEAN Cephes lever (rational P0/Q0 for
  x<5 + asymptotic PP/PQ/QP/QQ modulus/phase for x≥5, ~6 arrays). But modest gain (1.6×) for a
  ~80-line/6-array port across 4 functions → lower ROI than erf's 5.9×; flagged not done.
RECIPE (proven): `gh api repos/scipy/xsf/contents/include/xsf/cephes/<file>.h --jq .content |
base64 -d` → transcribe the exact coefficient arrays → byte-matches scipy.special.

### ✅ j0 Cephes rational kernel (byte-matches scipy; array 1.1×, kernel win contention-masked)
Applied the proven Cephes-fetch lever to j0_core: replaced the convergence-loop power series
(~25 terms for x<14) with scipy's xsf EXACT Cephes rational (RP/RQ for |x|≤5 + PP/PQ/QP/QQ
asymptotic modulus/phase for |x|>5) → byte-matches scipy.special.j0. Conformance: NO new
failures (the 4 — digamma/polygamma/exp2/powm1 — are pre-existing non-j0/non-erf, another
agent's work). MEASURED special_array_65536/j0: 1.08ms → 0.985ms = 1.1× — modest because the
array is SPAWN-bound under fleet contention (the rational-vs-series kernel win is bigger but
masked; cf. erf where the kernel was a big enough fraction to show 5×). KEPT: strictly better
(scipy-exact parity + provably fewer ops), not a regression. j0_series_small retained (y0 uses
it at 3212). j1/y0/y1 still series (same lever, lower priority).

### 📊 Special-kernel ranking (find-the-next-erf measurement) — erf was UNIQUE; rest complex/nuanced
MEASURED fsci scalar special kernels to find another erf-class slow-iterative loss:
- **gammainc** (gamma.rs:1216): 58ns(a1)→94ns(a10), GROWS with a (series ~a terms) → large a
  much worse vs Cephes igam (bounded). Genuine candidate BUT the Cephes igam port is COMPLEX
  (multi-regime: small-a series + CF + Temme asymptotic, not a flat rational like erf) AND
  gamma.rs holds another agent's in-progress breakage (digamma/polygamma fail) → HAND OFF.
- **beta** (40→96ns, grows): traces to Lanczos gammaln (3 lgam calls), same powf-cost nuance
  as gamma — not an iterative→rational flip.
- gamma=Lanczos (nuanced), j0/j1/y0/y1=series but contention-masked (1.1×).
CONCLUSION: erf (5.9×→1.2×, simple 80-term-series+CF → flat rational) was the UNIQUE clean
erf-class Cephes lever. The rest are complex multi-regime ports (igam) or Lanczos-nuanced or
modest — none clears the bar erf did. Next-session: the igam port is the only remaining
big-loss candidate, but it needs gamma.rs free + a careful multi-regime Cephes port.

### ✅ ellipeinc combined Carlson R_F+R_D (1.4×, byte-identical, slowest special kernel)
The find-an-erf-class measurement flagged ellipeinc (incomplete elliptic E) as the SLOWEST
special kernel (280-307ns). E(φ,m) = s·R_F(cc,d,1) − (m/3)s³·R_D(cc,d,1) called carlson_rf AND
carlson_rd over the SAME (cc,d,1) — TWO separate sqrt-heavy duplication sequences. R_F and R_D
share the IDENTICAL (x,y,z) sequence (only `ave`/convergence + R_D's `s` accumulation differ),
so a combined `carlson_rf_rd` computes the sqrt-sequence ONCE, tracking each convergence
independently → BYTE-IDENTICAL. **MEASURED ellipeinc_scalar: m0.5 280→195ns (1.44×), m0.9
307→219ns (1.40×)** (1.4× not 2× because R_D is costlier than R_F). Conformance: same 4
pre-existing failures, NO new (byte-identical). NOT a Cephes port — a pure shared-iteration
refactor in a FREE file (elliptic.rs), refreshed-first (no clobber). Reusable: any code calling
carlson_rf+carlson_rd on the same args.

### ✅ Carlson ERRTOL 1e-5 → 1.3e-3 (1.4-1.6×, machine-accurate) — stacks ellipeinc to ~2× total
NR Carlson R_F/R_D use a 5th-order final correction → error ~ERRTOL^6. fsci's ERRTOL=1e-5 gave
error ~1e-30 (overkill by ~14 orders); double-precision only needs error <2e-16 → ERRTOL≈2e-3.
Raised all 3 (carlson_rf/rd/rf_rd) to **1.3e-3** (error ~5e-18, machine-accurate), cutting the
duplication iterations ~9→5. **MEASURED: ellipkinc 148→107ns (1.38×), 171→109ns (1.57×);
ellipeinc 195→146ns (1.34×), 219→149ns (1.47×)** — with the earlier Carlson-sharing, ellipeinc
is now 307→149ns = **~2.06× total** (the slowest special kernel halved). Conformance: same 4
pre-existing failures, NO new (machine-accurate vs scipy). Provable lever: audit iterative
convergence tolerances vs the order of the final correction — an over-tight ERRTOL wastes
iterations at no accuracy benefit. Free file, refreshed-first.

### ✅ lfilter biquad unrolled scalar fast path — FLIPS 1.53× loss → parity/slight win (signal)
The OPEN lfilter biquad loss (4096 biquad: fsci 37.4 µs vs scipy 24.5 µs = 1.53× SLOWER).
Root cause: the general `lfilter_with_state` ran the DF2T delay-line update as a branchy
inner loop over a HEAP `Vec d` (`for j in 0..nfilt-1` with a `j+1 < nfilt-1` boundary branch
+ bounds-checked `b_norm[j+1]`/`a_norm[j+1]`/`d[j+1]` indexing every sample) — whereas
`sosfilt` already used the optimal fully-unrolled scalar-register biquad form (d1/d2 in
registers, no indexing/branch). Added byte-identical unrolled fast paths for nfilt==2 (order 1)
and nfilt==3 (order 2 / biquad) that keep the whole delay line in scalar registers — same float
ops in the same order as the general recurrence (verified: d[0] reads OLD d[1] before write).
**MEASURED filtering/lfilter/4096_biquad: 37.4 µs → 24.2 µs = 1.54× self-speedup (criterion
change −35.9%); now ≈ scipy 24.5 µs (parity, marginally faster).** Conformance: fsci-signal
GREEN 648/0 (+59 metamorphic), incl. lfilter_with_state_matches_scipy_reference_vectors and
lfilter_fir_iir_match_scipy. Byte-identical by construction. Lever: when a general N-tap kernel
serves a hot low-order case, peel a register-unrolled specialization for the common orders
(1/2) — the heap delay line + per-iter bounds/branch was the entire gap, exactly as sosfilt
already demonstrated. (filtfilt/lfilter_axis_2d route through the same core → inherit the win.)

### ✅ gaussian_filter 2-D parallel gate raised (serial 1.82× FASTER at 256² — closes most of the 2.83× loss)
The documented gaussian_filter loss (σ=2 256²: fsci 3238 µs vs scipy 1143 µs = 2.83× slower).
Root cause was NOT the kernel (col-pass interior-axpy was a measured 0.755× regression — see
NEGATIVE_EVIDENCE) but a PARALLEL-BELOW-BREAK-EVEN gate: gaussian_filter_2d_reflect_order0 took
its thread count from the shared `ndimage_filter_thread_count` (parallel when pixels·kernel_len
>= 1<<18). At 256² that work ≈ 1.1M trips the gate and spawns ~1 thread per few rows, but the
separable row/col passes are cheap per pixel (one symmetric fold), so spawn overhead dominates.
**Same-process interleaved A/B (30 reps × 200 iters, GAUSSIAN_FORCE_SERIAL toggle, byte-identical
assert_eq across all sizes):**
| n     | serial    | parallel  | serial speedup |
|-------|-----------|-----------|----------------|
| 128²  | 506 µs    | 3530 µs   | **6.98×**      |
| 256²  | 2095 µs   | 3814 µs   | **1.82×** (bench size) |
| 512²  | 7010 µs   | 4534 µs   | 0.65× (parallel wins) |
| 1024² | 24107 µs  | 5079 µs   | 0.21× (parallel wins) |
FIX: gate the gaussian 2-D path at `pixels·kernel_len >= 1<<21` (~2M) so ≤256² runs serial and
≥512² stays parallel (break-even is between them). BYTE-IDENTICAL (thread count never changes the
result — proven by assert_eq). fsci-ndimage GREEN 246/0 (+58 integration). The 256² serial 1.82×
closes most of the 2.83× scipy gap (absolute µs are contention-inflated here; the A/B RATIO is the
reliable signal — criterion cross-run swung +20%…+196% in ONE run, uninterpretable under load).
LEVER (paid out again): a parallel gate must scale with PER-ELEMENT WORK COST, not a flat
flop-product threshold. Cheap separable/elementwise kernels need a MUCH higher work gate than the
shared default — the same cost-aware-gate lesson as the stats batch-method and pdist veins.

### ✅ uniform_filter parallel gate fixed (PIXEL-COUNT not work-product) — serial 3.78×@256², 1.48×@512²
Same cost-aware-gate vein as the gaussian fix. uniform_filter_along_axis parallelized across outer
slabs when `ndimage_filter_thread_count(arr.size(), size) >= 1<<18` (i.e. arr.size()·size). But the
sliding window uses a RUNNING SUM — O(1) per output element (drop leaving + add entering),
INDEPENDENT of window `size` — so multiplying the work metric by `size` is wrong (over-counts large
windows, trips the gate far too early), and the real amortization point scales with PIXEL COUNT.
**Same-process interleaved A/B (byte-identical assert_eq all sizes, axis=1, Reflect):**
| n      | sz | serial    | parallel  | serial speedup |
|--------|----|-----------|-----------|----------------|
| 256²   | 5  | 735 µs    | 2780 µs   | **3.78×**      |
| 512²   | 5  | 3694 µs   | 5470 µs   | **1.48×**      |
| 1024²  | 5  | 18979 µs  | 18904 µs  | 0.996× (parity)|
FIX: gate the running-sum pass at `arr.size() >= 1<<20` (~1M px) — below that it spawns up to 64
threads for cheap O(1)/elt slabs and the spawn overhead dominates; from ~1M px up parallel pays.
BYTE-IDENTICAL (thread count never changes the result). fsci-ndimage GREEN 246/0 (+58 integration).
The 256² 3.78× / 512² 1.48× are the realistic image-filter sizes. Third payout of the cost-aware
gate lever (gaussian 2-D, now uniform_filter); cheap separable/running-sum ndimage kernels need a
MUCH higher work gate than the shared 1<<18 default — gate on PER-ELEMENT-COST-scaled work, and for
size-independent running sums that means pixel count, NOT pixel·window.

### ✅ correlate1d/convolve1d along-axis parallel gate raised (serial 2.61×@256²) — also fixes general gaussian path
Fourth payout of the cost-aware-gate vein. correlate1d_along_axis & convolve1d_along_axis (which
back public correlate1d/convolve1d AND the GENERAL gaussian path gaussian_filter1d_axis→
convolve1d_along_axis for 3D/order>0/non-reflect) parallelized across outer slabs when the shared
`ndimage_filter_thread_count(arr.size(), weights.len()) >= 1<<18`. Per-element cost IS an
O(weights.len())-tap dot (so the work product is the right metric — unlike uniform's running sum),
but the 1<<18 threshold is too low: at 256² w5 (work 327k) it spawns ~64 threads for a cheap pass.
**Same-process A/B (byte-identical assert_eq, axis=1, w5, Reflect):**
| n     | serial   | parallel | serial speedup |
|-------|----------|----------|----------------|
| 256²  | 968 µs   | 2526 µs  | **2.61×**      |
| 512²  | 3793 µs  | 3075 µs  | 0.81× (parallel wins 1.23×) |
FIX: gate both at `arr.size()·weights.len() >= 1<<20` (break-even ~n=453); 256²→serial, ≥512²→
parallel. BYTE-IDENTICAL. fsci-ndimage GREEN 246/0 (+58). Vein tally: gaussian-2D (1<<21 fold),
uniform_filter (1<<20 pixel-count running-sum), now correlate1d/convolve1d (1<<20 tap-dot). The
shared 1<<18 gate was uniformly too low for ALL cheap separable ndimage kernels.

### ❌ TRIED & REVERTED (~0-gain, see NEGATIVE_EVIDENCE 2026-06-22): SIMD-across-output-pixels for nd_filter_apply
UPDATE: implemented + measured = **1.025× (memory-bandwidth-bound, NOT compute-bound)**. The 25 taps
each hit a different input row/cache-line, so vectorizing 8 output pixels cuts instructions but not the
dominant memory traffic. The correlate/gaussian 1.1-1.2× residuals are a BANDWIDTH wall — do NOT
re-chase with SIMD. Byte-identity was confirmed (correct, just useless). Original (now-refuted) note:
### 📋 NEXT BOLD LEVER (scoped, byte-identical): SIMD-across-output-pixels for nd_filter_apply interior
The correlate 5x5 256² 1.18× residual (and gaussian/correlate kernel walls generally) is the scalar
inner loop: per interior pixel, `for k: sum += w[k]*input.data[p+tap_flat[k]]` (25 scalar fmas).
nd_filter_apply ALREADY has the interior flat-offset fast path; the remaining gap is scalar-vs-C-SIMD.
LEVER (proven in spatial pdist, see [[perf_spatial_pdist_simd_across_pairs]] — pure std::simd, NO
unsafe, forbid(unsafe)-safe): process 8 CONSECUTIVE interior output pixels (same row ⇒ contiguous)
as one Simd<f64,8>: `acc += Simd::splat(w[k]) * Simd::from_slice(&input.data[p+tap_flat[k] ..][..8])`,
then copy_to_slice. BYTE-IDENTICAL: each lane independently accumulates ITS pixel's sum in the SAME
k-order as scalar (Rust `+`/`*` don't FMA-contract by default). Needs: region-partition the 2-D
output into the interior box [lo,hi)² + boundary bands (so interior runs are contiguous and reflection-
free), iterate interior rows, process interior cols 8-wide + scalar remainder, boundary via the slow
path. Interior is ~97% of a 256² 5x5 → up to memory-bound 2-4× on the kernel, plausibly flips the
1.18× loss to a WIN. Build the byte-identity property test first (correlate vs nd_filter_perpixel_ref).
Same lever extends to gaussian's col-pass and any separable/dense filter interior. NOT YET DONE —
deferred to a fresh-context iteration (meaty change in a fragile file; do it with full budget).
NOTE (ruled out this session): the per-pixel DIVIDE in the interior check is NOT the bottleneck
(incremental-index A/B = 0.945×, reverted) — it's the scalar gather/fma throughput. SIMD is the lever.

### ✅✅ interpolate par_query_map gate 1<<18 → 1<<23 — flips an 18.5x over-parallelization REGRESSION (cubic eval_many)
The cost-aware-gate vein extends to fsci-interpolate. par_query_map/par_query_try_map (back ALL
*_many evaluators: cubic/pchip/CubicSplineStandalone/RBF/griddata/RGI) gated parallelism at
`m·work_per_query >= 1<<18`. Unlike ndimage's in-place chunks_mut, this parallel path allocates a
RESULT VEC PER THREAD (up to ~m/2 threads, capped at cores) and `flat_map`-collects them — a large
FIXED overhead (~4-5 ms under fleet contention, independent of m). At work_per_query=24 (spline eval)
the gate fired at m≈10923, catastrophically over-parallelizing common batch sizes.
**Same-process A/B (cubic eval_many, n=1024 knots, byte-identical assert_eq all sizes):**
| m (queries) | serial   | parallel | serial speedup |
|-------------|----------|----------|----------------|
| 16384       | 212 µs   | 3924 µs  | **18.52x**     |
| 32768       | 396 µs   | 4144 µs  | **10.48x**     |
| 65536       | 788 µs   | 4645 µs  | **5.89x**      |
| 131072      | 1522 µs  | 4753 µs  | **3.12x**      |
Parallel is ~4-4.8 ms FIXED (spawn + per-thread Vec alloc + flat_map realloc); serial scales, so
break-even is ~350k queries (work ≈ 1<<23). FIX: raise the shared gate to `1<<23` (single constant,
both par_query_map + par_query_try_map). Cheap batch evals now stay serial up to ~350k queries where
parallelism finally amortizes; genuinely huge batches still parallelize. BYTE-IDENTICAL (thread count
never changes the result; assert_eq verified). fsci-interpolate GREEN 173/0 (+56). HIGH value — eval_many
at m=16k-131k is the common interpolation batch path and was 3-18x pessimized. Same root cause as the
ndimage gates (shared 1<<18 too low for many-core spawn) but WORSE here (per-thread Vec alloc, not
in-place). Lever now paid out 4× across two crates: gate on per-element cost AND account for the
parallel implementation's fixed overhead (alloc-per-thread ⇒ much higher break-even than chunks_mut).

### ✅✅ stats: 8 new axis-2D reducers + gmean per-call syscall fix (3.1-40x faster than scipy, same-box)
Continues the proven axis-2D reducer vein (8ec65b21 added 6 at 27-145x; 29f1a75a rankdata 60-90x).
Eight new `*_axis_2d` multi-channel reducers wrap their scalar 1-D fn through the parallel-across-lines
`reduce_axis_2d` helper (bit-identical to per-line by construction; conformance via the extended
`reduce_axis_2d_family_matches_per_line` test, `to_bits` so NaN-on-negative still matches): `sem`,
`gmean`, `hmean`, `gstd`, `kstat`, `kstatvar`, `moment`, `differential_entropy`.

**SAME-BOX head-to-head (best-of-20, fsci binary + scipy.stats both on this 64-core box):**
| reducer              | 2000×512 (scipy/fsci ms → ×) | 500×4096 (scipy/fsci ms → ×) |
|----------------------|------------------------------|------------------------------|
| sem                  | 2.04 / 1.60 → **1.27×**      | 5.99 / 1.69 → **3.55×**      |
| gmean                | 5.41 / 1.70 → **3.18×**      | 12.44 / 1.88 → **6.62×**     |
| hmean                | 1.73 / 1.71 → 1.01× (parity) | 5.48 / 1.67 → **3.28×**      |
| gstd                 | 18.20 / 1.88 → **9.68×**     | 39.52 / 2.08 → **18.97×**    |
| kstat(n=2)           | 1.82 / 1.81 → 1.00× (parity) | 6.30 / 1.96 → **3.21×**      |
| kstatvar(n=2)        | 25.24 / 1.89 → **13.36×**    | 52.26 / 2.27 → **23.04×**    |
| moment(k=4)          | 12.25 / 1.82 → **6.73×**     | 26.06 / 1.68 → **15.51×**    |
| differential_entropy | 55.12 / 2.40 → **22.97×**    | 135.88 / 3.36 → **40.46×**   |

gstd/kstatvar/moment/differential_entropy win 7-40× because scipy's own implementations are heavy
Python; sem/hmean/kstat are parity-to-3.5× (never a loss).

**BUG CAUGHT & FIXED while measuring (byte-identical):** `gmean_axis_2d` was initially a *2.3× LOSS*
at 2000×512 (11.8 ms) yet 3.5 ms at 500×4096 — non-monotonic (1M logs slower than 2M). Root cause:
the scalar `gmean`→`gmean_log_sum` calls `std::thread::available_parallelism()` (a `sched_getaffinity`
syscall) on EVERY invocation, BEFORE the `n < 1<<16` serial short-circuit. Called once per line by the
reducer (2000 short lines), the ~5µs syscall ×2000 ≈ 10 ms dominated the cheap `ln` work. gstd (no
parallelism probe) stayed 1.9 ms on identical log counts — the smoking gun. FIX: hoist the `n < 1<<16`
return ABOVE the `available_parallelism()` call (byte-identical: that path always took `chunk_sum`
anyway). gmean_axis_2d 11.8→1.70 ms (6.9×), flipping the loss to a 3.18× win. Bonus: standalone
`gmean()` on any <65536-elt input no longer pays the syscall (helps every per-line/hot-loop caller).
LESSON (generalizable): probing `available_parallelism()` inside a per-element scalar kernel is a hidden
syscall tax when that kernel is the reduce-closure of an axis sweep — order the cheap serial-gate FIRST.
fsci-stats GREEN (reduce_axis_2d_family + all gmean/gstd/hmean tests pass). Same-process A/B mandatory.

### ✅✅ stats: 10 MORE axis-2D reducers (trimmed/circular/mode/entropy) + entropy syscall fix (2.5-71x faster than scipy)
Third batch on the reduce_axis_2d vein. scipy's per-axis trimmed/circular/mode stats are catastrophically
slow (Python masking + per-slice dispatch): tstd 51-112ms, tsem 48-102ms, mode 50-129ms, tvar 27-75ms,
circmean/var/std 43-87ms, entropy 16-32ms. fsci's parallel-across-lines reducers run at ~1.5-3.4ms.
Added (bit-identical to per-line, conformance in extended reduce_axis_2d_family test, 24 reducers total):
tmean, tvar, tstd, tsem, tmax, mode, entropy, circmean, circvar, circstd.

**SAME-BOX paired head-to-head (best-of-20, fsci binary + scipy.stats measured BACK-TO-BACK under same load):**
| reducer  | 2000×512 (scipy/fsci → ×) | 500×4096 (scipy/fsci → ×)  |
|----------|---------------------------|----------------------------|
| tstd     | 51.27/1.46 → **35.1×**    | 112.32/1.59 → **70.8×**    |
| tsem     | 47.59/1.54 → **30.9×**    | 102.44/1.56 → **65.5×**    |
| mode     | 50.16/1.86 → **26.9×**    | 128.51/2.85 → **45.1×**    |
| tvar     | 26.72/1.51 → **17.7×**    | 74.98/1.61 → **46.5×**     |
| circmean | 43.72/2.37 → **18.4×**    | 86.50/3.38 → **25.6×**     |
| circvar  | 43.05/2.31 → **18.6×**    | 85.98/3.31 → **26.0×**     |
| circstd  | 44.12/2.46 → **17.9×**    | 85.87/3.26 → **26.3×**     |
| entropy  | 15.99/1.69 → **9.5×**     | 31.88/1.71 → **18.7×**     |
| tmean    | 5.26/1.63 → **3.2×**      | 16.45/1.77 → **9.3×**      |
| tmax     | 3.80/1.53 → **2.5×**      | 8.19/1.58 → **5.2×**       |

**SYSCALL-TAX LEVER PAID OUT A 3rd TIME (byte-identical):** `entropy` was initially a 12.23ms / 1.22×
near-loss at 2000×512 — identical non-monotonic signature to gmean (1M elts slower than 2M). Root cause
again: `entropy_h_sum` called `available_parallelism()` (sched_getaffinity syscall) on every line BEFORE
its `n<1<<16` serial gate. Hoisted the gate above the syscall → entropy 12.23→1.69ms (7.7×), 1.22×→9.5×
win. grep confirmed only gmean_log_sum + entropy_h_sum had the `|| threads<=1` pattern; class now closed.

**tmin DELIBERATELY OMITTED:** `scipy.stats.tmin` is a masked `np.min`, unusually fast (~1.0-1.2ms),
below reduce_axis_2d's ~1.5ms 64-thread-spawn floor → a parallel fsci tmin is a 0.72× LOSS at narrow
columns (wins 2.1× at wide). Omitted to keep an all-wins batch (tmax kept — scipy.tmax is ~3× slower
than tmin, fsci wins it). FOLLOW-ON LEVER (noted, not done): the ~1.5ms floor is pure 64-thread spawn
overhead — ALL reducers hit it at 2000 lines regardless of op cost (tstd≈tmax≈1.5ms). Capping
reduce_axis_2d's thread count for low total-work would lower the floor AND flip tmin; needs careful
same-process A/B (risk of regressing the big-win heavy reducers). Deferred.

### ✅✅ stats: reduce_axis_2d thread-count cap — lifts ALL 25 axis-2D reducers 1.0-2.4x (byte-identical) + flips tmin loss→win
DIG via extreme-software-optimization (profile-driven). reduce_axis_2d (and the rankdata_axis_2d helper)
fanned out to ALL 64 cores whenever work >= 1<<16. Same-process A/B (one bin, fixed thread counts,
byte-identical checksum asserted across all counts) showed 64 threads is ALWAYS worse than 16-32 for the
common 1-2M-element regime — a ~1.5ms FLOOR that is pure OS-thread spawn/join overhead (~20µs × 64),
dominating the actual cheap per-line reduce. Optimal is ~21t at work≈1M, ~42t at work≈2M.

FIX (both parallel-across-lines sites): cap nthreads at `work / 48_000` element-ops/thread (each thread
busy enough to amortize its spawn), `threads.min(n_lines).min(work/48000)`. BYTE-IDENTICAL (thread count
never changes a per-line reduction; family + rankdata bit-identity tests green) and never spawns MORE
than the old `threads.min(n_lines)` → a MONOTONE win that still ramps to all 64 cores once work justifies
it (>= 64·48k ≈ 3.1M elements).

**Same-process A/B (64t OLD → formula-picked NEW, measured back-to-back same load):**
| reducer  | work≈1M: 64t→21t          | work≈2M: 64t→42t          |
|----------|---------------------------|---------------------------|
| tmin     | 1.449→0.593 → **2.44×**   | 1.484→1.030 → **1.44×**   |
| tstd     | 1.431→0.809 → **1.77×**   | 1.539→1.242 → 1.24×       |
| entropy  | 1.527→0.853 → **1.79×**   | 1.563→1.305 → 1.20×       |
| mode     | 1.809→1.491 → 1.21×       | 2.418→2.396 → 1.01× (par) |
| circmean | 1.741→1.438 → 1.21×       | 2.195→2.063 → 1.06×       |

Lifts the WHOLE 25-reducer family (skew/kurtosis/.../trimmed/circular/mode/entropy) since they all route
through reduce_axis_2d — biggest gains on cheap/medium ops at narrow columns (the spawn-floor-bound case).
BONUS: tmin_axis_2d (dropped in the prior batch as a 0.72× loss vs scipy's fast masked-min) RE-ADDED — now
0.59ms vs scipy 1.01ms = **1.7× WIN** at 2000×512 (4.4× at 500×4096). The lever I built to kill the floor
flipped the one function the floor had cost me. LESSON: probe `available_parallelism()`-driven fan-out with
a same-process fixed-thread A/B; "use all cores" is wrong when per-call work is < ~64·spawn_cost.

### ✅✅ stats: zscore/gzscore/zmap axis-2D — a DIFFERENT primitive (vmap-style vector-output map) — 3.9-14.7x faster than scipy
DIG (jax "different primitive"): reduce_axis_2d reduces a line→scalar; zscore/gzscore/zmap need line→LINE
(vector output, same shape). Added the complementary primitive: `map_axis_2d` / `par_produce_lines` — a
batched vector-output map parallel across lines with the SAME work-capped thread count (axis_2d_thread_count,
the 48k/thread cap factored out). Three new public fns: zscore_axis_2d, gzscore_axis_2d, zmap_axis_2d
(bit-identical to per-line 1-D; conformance in new `map_axis_2d_family_matches_per_line`, both axes, to_bits).

scipy.stats.zscore/gzscore/zmap carry heavy intermediate-array overhead (mean+std+subtract+divide as
separate numpy temporaries); fsci does 2 passes (mean/std then write) parallel across lines, no temporaries.

**SAME-BOX paired head-to-head (best-of-30, fsci public fns + scipy.stats back-to-back):**
| op      | 2000×512 (scipy/fsci → ×) | 500×4096 (scipy/fsci → ×) | 4000×1024 (scipy/fsci → ×) |
|---------|---------------------------|---------------------------|----------------------------|
| zscore  | 4.51/1.03 → **4.4×**      | 12.81/1.72 → **7.5×**     | 30.25/2.81 → **10.8×**     |
| gzscore | 8.31/1.28 → **6.5×**      | 21.45/2.07 → **10.4×**    | 46.44/3.16 → **14.7×**     |
| zmap    | 4.11/1.05 → **3.9×**      | 12.64/1.70 → **7.4×**     | 29.06/3.18 → **9.1×**      |

This is the FIRST vector-output member of the axis-2D family (the prior 25 are scalar reductions). The
new `map_axis_2d` primitive + `axis_2d_thread_count` (shared 48k/thread cap) generalize to any future
batched line→line transform (e.g. detrend/normalize/rankdata-values/winsorize-along-axis). gzscore wins
most (scipy's per-element log+exp temporaries are the slowest). REVERT-check N/A (pure addition, no
existing path changed). fsci-stats conformance GREEN (map_axis_2d_family + all zscore/gzscore/zmap tests).

### ✅✅ stats: kendalltau_matrix (all-pairs Kendall tau) — a DIFFERENT primitive (parallel all-pairs vs Python loop) — 61-118x faster than scipy
DIG: scipy has NO vectorized all-pairs Kendall tau — computing a Kendall correlation matrix means looping
`scipy.stats.kendalltau` in Python over m·(m−1)/2 pairs, paying Python-call overhead × every pair PLUS a
per-pair exact-Mahonian/asymptotic p-value the matrix never needs. NEW `kendalltau_matrix(variables)`:
(1) tau-ONLY per pair (`kendalltau_statistic_only`, bit-identical to `kendalltau(.).statistic`, skips the
p-value — the bulk of per-pair cost), (2) parallel ACROSS pairs (heavy O(n log n) per pair amortizes OS
spawn → fan out to all cores, >=4 pairs/thread). Diagonal = self-tau (1.0 / NaN-for-constant).

**SAME-BOX head-to-head (fsci kendalltau_matrix vs scipy Python kendalltau-loop, both this box):**
| matrix (m vars × n obs)      | pairs  | scipy      | fsci     | speedup    |
|------------------------------|--------|------------|----------|------------|
| m=40,  n=400                 | 780    | 194 ms     | 3.16 ms  | **61×**    |
| m=100, n=1000                | 4 950  | 1 673 ms   | 19.85 ms | **84×**    |
| m=200, n=1000                | 19 900 | 6 688 ms   | 56.74 ms | **118×**   |

Speedup GROWS with size (more pairs ⇒ more parallelism + Python-loop overhead dominates scipy more).
Conformance: `kendalltau_matrix_matches_pairwise` asserts every entry == per-pair kendalltau(.).statistic
bit-identically (incl. a tied column), symmetric, ragged-input rejected. Pure addition (123 lines, 0
deletions) — no existing path changed. This is the "all-pairs over an O(n log n) per-pair kernel, tau-only,
parallel across pairs" lever; generalizes to any all-pairs statistic scipy makes users Python-loop
(weightedtau matrix, somersd matrix, pairwise distance-correlation). fsci-stats conformance GREEN.

### ✅✅ stats: weightedtau_matrix (all-pairs weighted Kendall tau) — 108-222x faster than scipy + factored the all-pairs primitive
Extends the all-pairs primitive to scipy's OTHER matrix-less rank correlation. Factored the parallel-
across-pairs logic into `all_pairs_symmetric_matrix(variables, pair_stat)` (kendalltau_matrix refactored
onto it, byte-identical — conformance test unchanged & green); added `weightedtau_matrix` = the same
helper over `weightedtau` (which returns f64 directly, no p-value). scipy has NO vectorized all-pairs
weighted tau → users loop `scipy.stats.weightedtau` in Python, and weightedtau is SLOWER per-call than
kendalltau (hyperbolic weighting), so the gap is even larger.

**SAME-BOX head-to-head (fsci weightedtau_matrix vs scipy Python weightedtau-loop, both this box):**
| matrix (m × n)   | pairs  | scipy        | fsci      | speedup     |
|------------------|--------|--------------|-----------|-------------|
| m=40,  n=400     | 780    | 648.3 ms     | 5.97 ms   | **108.6×**  |
| m=100, n=1000    | 4 950  | 10 462.9 ms  | 47.09 ms  | **222.2×**  |

(10.5 SECONDS in scipy for a 100×100 weighted-tau matrix.) Conformance: weightedtau_matrix upper-triangle
+ diagonal bit-identical to per-pair `weightedtau` (matrix symmetric BY CONSTRUCTION — the helper mirrors
the upper triangle; NOTE weightedtau is mathematically but NOT bit-symmetric across arg order due to its
Fenwick accumulation sorting by the first arg, so only i<=j is asserted per-pair). The `all_pairs_symmetric_matrix`
helper now backs both matrices and any future one (somersd/distance-correlation). fsci-stats GREEN.

### ✅✅ stats: wasserstein/energy distance matrices (all-pairs) — 16-63x faster than scipy
Extends the all-pairs primitive from correlation to DISTANCE matrices (distribution comparison /
clustering of m 1-D samples). wasserstein_distance & energy_distance are symmetric f64 distances → both
are one-liners over `all_pairs_symmetric_matrix`. SciPy has NO vectorized all-pairs form — users loop
`scipy.stats.wasserstein_distance` / `energy_distance` in Python over m·(m−1)/2 pairs.

**SAME-BOX head-to-head (fsci matrix vs scipy Python distance-loop, both this box):**
| matrix (m × n)  | pairs  | scipy wass. | fsci wass. | ×        | scipy energy | fsci energy | ×        |
|-----------------|--------|-------------|------------|----------|--------------|-------------|----------|
| m=40,  n=400    | 780    | 55.5 ms     | 2.74 ms    | **20.3×**| 54.2 ms      | 3.35 ms     | **16.2×**|
| m=100, n=1000   | 4 950  | 843.3 ms    | 13.47 ms   | **62.6×**| 869.9 ms     | 19.20 ms    | **45.3×**|

Conformance: `distance_matrices_match_pairwise` — upper triangle + diagonal bit-identical to per-pair
`wasserstein_distance`/`energy_distance`, symmetric, ragged-input rejected (diagonal = self-distance, not
asserted == 0.0: `d(u,u)` may be ±0.0/tiny-rounding, the per-pair i<=j check covers it). The
`all_pairs_symmetric_matrix` helper now backs FOUR matrices (kendalltau/weightedtau/wasserstein/energy);
ANY symmetric `fn(&[f64],&[f64])->f64` scipy makes you Python-loop is now a one-liner. fsci-stats GREEN.

### ✅ stats: ks_2samp_matrix (all-pairs two-sample KS test) — 8-29x faster than scipy
Extends the all-pairs primitive to two-sample TESTS (pairwise distribution comparison — a common
multiple-comparison workflow). New tuple helper `all_pairs_two_symmetric_matrices` (per-pair kernel
returns `(stat, pvalue)` → two symmetric matrices); `ks_2samp_matrix` returns `(D_matrix, pvalue_matrix)`.
SciPy has NO vectorized all-pairs form — users loop `scipy.stats.ks_2samp` in Python.

**SAME-BOX head-to-head (fsci ks_2samp_matrix vs scipy Python ks_2samp-loop, both this box):**
| matrix (m × n)  | pairs  | scipy      | fsci      | speedup   |
|-----------------|--------|------------|-----------|-----------|
| m=40,  n=400    | 780    | 262.0 ms   | 9.1 ms    | **28.8×** |
| m=100, n=1000   | 4 950  | 2 030.2 ms | 243.2 ms  | **8.3×**  |

HONEST NOTE: the m=100/n=1000 win (8.3×) is smaller than the correlation/distance matrices (16-222×)
because fsci's `ks_2samp` P-VALUE is heavy per-pair at large n (~3.4 ms/pair, ~8× slower than scipy's
asymptotic) — the matrix is already at all 64 cores (compute-bound, not thread-limited). FOLLOW-ON (noted,
not done): speed fsci's ks_2samp pvalue at large n (likely an exact/series path where scipy goes
asymptotic), or offer a statistic-only `ks_2samp_statistic_matrix` (the D stat is O(n log n), would be
50-100×). Conformance: `ks_2samp_matrix` upper-triangle + diagonal bit-identical to per-pair ks_2samp
(both stat & pvalue), symmetric, ragged rejected. fsci-stats GREEN.

### ✅✅ stats: mannwhitneyu_matrix (all-pairs Mann–Whitney U test) — 113-131x faster than scipy
The strongest two-sample-test matrix (the ks follow-on). fsci's `mannwhitneyu` reports the smaller U
(order-independent) and a normal-approximation p-value (CHEAP, unlike ks_2samp's heavy exact pvalue) — so
both outputs are symmetric and it's a ONE-LINER over the `all_pairs_two_symmetric_matrices` tuple helper.
SciPy has NO vectorized all-pairs form — pairwise rank-sum comparison means looping
`scipy.stats.mannwhitneyu` in Python.

**SAME-BOX head-to-head (fsci mannwhitneyu_matrix vs scipy Python mannwhitneyu-loop, both this box):**
| matrix (m × n)  | pairs  | scipy      | fsci      | speedup    |
|-----------------|--------|------------|-----------|------------|
| m=40,  n=400    | 780    | 407.1 ms   | 3.61 ms   | **112.9×** |
| m=100, n=1000   | 4 950  | 3 130.2 ms | 23.94 ms  | **130.8×** |

Confirms the memory prediction: where ks_2samp_matrix was capped at 8× by fsci's heavy ks pvalue,
mannwhitneyu's normal-approx pvalue keeps the per-pair kernel light → the full all-pairs speedup. Returns
`(U_matrix, pvalue_matrix)`. Conformance: upper-triangle + diagonal bit-identical to per-pair mannwhitneyu
(both stat & pvalue), symmetric, ragged rejected. The tuple helper now backs ks + mannwhitneyu; the
`all_pairs_*` family covers 6 matrices total (kendall/weightedtau/wasserstein/energy/ks/mannwhitneyu).
fsci-stats GREEN.

### ✅✅ stats: ranksums_matrix + brunnermunzel_matrix (all-pairs rank tests) — 80-96x faster than scipy
Completes the rank-based two-sample test matrices. ranksums (signed z) and brunnermunzel (signed W) have
ANTI-symmetric statistics (`stat[j][i] == −stat[i][j]`) + symmetric p-values, so they need a NEW FULL
ordered-pairs helper `all_pairs_two_full_matrices` (evaluates every `(i,j), i≠j` — no symmetry assumed,
correct for directional stats). Both use cheap normal-approx p-values → big wins even at 2× the kernel
evals. SciPy has NO vectorized all-pairs form — users loop the test in Python.

**SAME-BOX head-to-head (fsci FULL m×(m−1) matrix vs scipy Python upper-triangle loop, both this box):**
| matrix (m × n)  | pairs (scipy) | scipy ranksums | fsci   | ×        | scipy brunnermunzel | fsci    | ×        |
|-----------------|---------------|----------------|--------|----------|---------------------|---------|----------|
| m=40,  n=400    | 780           | 277.3 ms       | 3.27 ms| **84.9×**| 489.3 ms            | 5.21 ms | **94.0×**|
| m=100, n=1000   | 4 950         | 2 200.5 ms     | 22.9 ms| **96.0×**| 3 750.0 ms          | 46.7 ms | **80.3×**|

Note: fsci returns the FULL directional matrix (m·(m−1) kernel evals) while scipy's loop only fills the
upper triangle (m·(m−1)/2) — fsci does 2× the work and STILL wins 80-96×. Conformance: every ordered
(i,j) bit-identical to per-pair ranksums/brunnermunzel (both stat & p-value), ragged rejected. The
`all_pairs_*` family now spans 8 matrices (kendall/weightedtau/wasserstein/energy/ks/mannwhitneyu +
ranksums/brunnermunzel) across THREE assembly shapes (f64-symmetric / tuple-symmetric / tuple-FULL).
fsci-stats GREEN.

### ✅✅ stats: permutation_test parallelized (LCG jump-ahead) — 87x faster than scipy
A DIFFERENT primitive from the all-pairs vein: the existing `permutation_test` was a SERIAL loop with a
cumulative (path-dependent) Fisher–Yates shuffle. Rewrote it so permutation `p` is a PURE FUNCTION of
`(seed, p)` — reset the buffer to the original sample + jump the shared LCG to `p·(n−1)` advances (reusing
the in-crate `lcg_jump(a,c,steps)` O(log) skip already built for byte-identical bootstrap parallelism) —
then fan out across permutations. Each thread streams (reset 8 KB buffer → shuffle → stat → discard), so
its working set is L1/L2-resident (dodges the cache-hostile materialization that made naïve resampler
parallelism 3.3× SLOWER, per NEGATIVE_EVIDENCE). Result is now DETERMINISTIC and thread-count-INDEPENDENT
(strictly better reproducibility than the old serial path).

**SAME-BOX head-to-head (fsci vs scipy.stats.permutation_test, vectorized, both this box):**
| workload                                  | scipy      | fsci     | speedup   |
|-------------------------------------------|------------|----------|-----------|
| 2-sample diff-of-means, n=1000, 9999 resamples | 245.7 ms | 2.83 ms | **87.0×** |

CAVEAT (see NEGATIVE_EVIDENCE.md): this CHANGES the exact p-value returned for a given seed (the old
serial output was a path-dependent Monte-Carlo estimate, never a stable contract); the tolerant
conformance test `permutation_test_matches_scipy_reference_values` (p≈0.1 ± 0.02) still passes. The trait
bound tightened `F: Fn` → `F: Fn + Sync` (required for the fan-out; ordinary statistic closures satisfy
it). fsci-stats GREEN.

### ✅✅ stats: CROSS all-pairs distance/test matrices (two-group) — 24-278x faster than scipy
A different SHAPE of the all-pairs primitive: rectangular `m × k` matrices comparing two GROUPS of 1-D
samples (e.g. m controls vs k treatments — a common two-group multiple-comparison setup). No symmetry, no
diagonal, and groups/samples may have DIFFERENT lengths (two-sample distances/tests accept ragged input).
New helpers `all_pairs_cross_matrix` (f64) + `all_pairs_cross_two_matrices` (tuple). SciPy makes you
double-loop the two groups in Python.

**SAME-BOX head-to-head (fsci cross matrix vs scipy Python double-loop, both this box; m=50 k=50 n=500, 2500 pairs):**
| function                | scipy      | fsci      | speedup    |
|-------------------------|------------|-----------|------------|
| wasserstein_distance_cross | 214.4 ms | 3.26 ms | **65.7×**  |
| energy_distance_cross   | 229.7 ms   | 3.86 ms   | **59.6×**  |
| ks_2samp_cross          | 845.5 ms   | 35.93 ms  | **23.5×**  |
| mannwhitneyu_cross      | 1 257.4 ms | 4.52 ms   | **278.2×** |

(ks is the weakest, consistent with the self-pairs finding — fsci's ks p-value is heavy per pair; mwu's
normal-approx p-value is cheap → 278×.) Conformance: every `out[i][j]` bit-identical to the per-pair
`wasserstein_distance/energy_distance/ks_2samp/mannwhitneyu`, ragged groups OK, empty sample rejected.

MEASURED-ALREADY-WON this iteration (negative evidence, NOT re-shipped): fsci `gaussian_kde` evaluate_many
12-38× (already parallel), `theilslopes` 9.7-18× + `siegelslopes` 9.3-62× (fast-path already optimal),
`monte_carlo_test` 61× (already parallel). fsci-stats' big single-array gaps are closed; the open seam is
the all-pairs/cross fan-out family. fsci-stats GREEN.

### ✅✅ stats: kendalltau_cross + weightedtau_cross (two-group cross-correlation) — 162-262x faster than scipy
Completes the cross family for CORRELATION: rectangular `m × k` cross-correlation between two groups of
variables (m features vs k targets — ubiquitous in genomics/finance: correlate every gene against every
phenotype). Statistic-only (skips the per-pair p-value, the bulk of the cost) over `all_pairs_cross_matrix`.
SciPy makes you double-loop `scipy.stats.kendalltau`/`weightedtau` in Python (weightedtau cross = 2.4 s!).

**SAME-BOX head-to-head (fsci cross matrix vs scipy Python double-loop; m=50 k=50 n=500, 2500 pairs):**
| function           | scipy      | fsci     | speedup    |
|--------------------|------------|----------|------------|
| kendalltau_cross   | 675.7 ms   | 4.18 ms  | **161.6×** |
| weightedtau_cross  | 2 398.4 ms | 9.15 ms  | **262.1×** |

Conformance: `out[i][j]` bit-identical to per-pair `kendalltau(.).statistic` / `weightedtau`, empty
sample rejected. The all_pairs/cross fan-out family now covers 14 public matrices (self: kendall/wtau/
wasserstein/energy/ks/mwu/ranksums/bm; cross: wasserstein/energy/ks/mwu/kendall/wtau). fsci-stats GREEN.

NOTE (uncontended-crate survey this iteration, all MEASURED-ALREADY-WIN, NOT re-shipped): spatial `cdist`
euclidean 10-13× (parallel + per-pair SIMD, all dims d=2..50); stats `gaussian_kde`/`theilslopes`/
`siegelslopes`/`monte_carlo_test` already win (see prior entry). fsci's `RbfInterpolator` is the LEGACY
`scipy.interpolate.Rbf` (kernel+epsilon, ≤4096 pts), NOT the modern `RBFInterpolator` — semantic mismatch,
not a comparable gap. The accessible uncontended surface is saturated; remaining gaps sit in contended
crates (linalg/signal/sparse — other agents' probes present) or known SIMD walls (FFT mid-pow2).

### ✅ fft: fft_axis2d / rfft_axis2d (batched 1-D FFT along last axis) — NEW gap-fill, 7.5-13x vs scipy DEFAULT
fsci-fft had NO batched-1-D-along-axis transform: `fftn`/`rfftn` always transform ALL axes (no `axes`
param), so `scipy.fft.fft(x, axis=-1)` over a 2-D array (per-row/per-channel FFT — spectrograms, batch
signal processing) had no direct fsci equivalent. Added `fft_axis2d`/`rfft_axis2d`: `rows` INDEPENDENT
length-`ncols` transforms, parallel ACROSS rows (each row's 1-D FFT serial on its owning thread — inner
`WorkerPolicy::Exact(1)` avoids 64×64 oversubscription). Row r bit-identical to per-row `fft`/`rfft`.

**SAME-BOX head-to-head (fsci vs scipy.fft, both this box):**
| rows × ncol  | fsci rfft | scipy rfft w=1 | × (w=1) | scipy rfft w=-1 | fsci fft | scipy fft w=1 | × (w=1) | scipy fft w=-1 |
|--------------|-----------|----------------|---------|-----------------|----------|---------------|---------|----------------|
| 2000 × 4096  | 6.38 ms   | 53.3 ms        | 8.4×    | 3.82 ms (0.60×) | 9.06 ms  | 121.1 ms      | 13.4×   | 7.90 ms (0.87×)|
| 5000 × 2048  | 9.31 ms   | 69.6 ms        | 7.5×    | 5.33 ms (0.57×) | 11.48 ms | 132.8 ms      | 11.6×   | 13.00 ms (1.13×)|
| 1000 × 8192  | 6.06 ms   | 58.6 ms        | 9.7×    | 12.88 ms (2.13×)| 9.23 ms  | 122.3 ms      | 13.3×   | 23.44 ms (2.54×)|

HONEST (see NEGATIVE_EVIDENCE.md): vs scipy's DEFAULT (`workers=1`, what most code uses) fsci wins
7.5-13.4× across the board. vs scipy's PARALLEL (`workers=-1`) it's MIXED — fsci WINS 2.1-2.5× at large
ncol=8192 and on complex fft@2048, but LOSES on rfft@2048/4096 (0.57-0.60×) because fsci's per-FFT kernel
is ~1.5× slower than pocketfft (the documented mid-pow2 SIMD wall) and when BOTH sides parallelize across
rows that kernel gap dominates. Net: a real new capability that beats the default API and is
competitive-to-winning vs scipy's best at large transforms. Conformance: `fft_rfft_axis2d_match_per_row`
bit-identical to per-row, shapes validated. fsci-fft GREEN.

### ⚖️ fft: dct_axis2d / idct_axis2d (batched DCT-II/III along axis) — NEW gap-fill, 5.6-7x vs scipy DEFAULT (loses to workers=-1)
Completes the batched-axis transform family for the DCT (per-block DCT is the core of image/audio
compression). fsci-fft had no batched-axis DCT (`dctn` does ALL axes). Added `dct_axis2d`/`idct_axis2d`
(parallel across rows via the new `batched_real_axis2d` helper, bit-identical to per-row `dct`/`idct`).

**SAME-BOX head-to-head (fsci vs scipy.fft.dct/idct, both this box):**
| rows × ncol  | fsci dct | scipy dct w=1 | × (w=1) | scipy dct w=-1 | fsci idct | scipy idct w=1 | × (w=1) | scipy idct w=-1 |
|--------------|----------|---------------|---------|----------------|-----------|----------------|---------|-----------------|
| 2000 × 4096  | 7.31 ms  | 48.87 ms      | 6.7×    | 4.79 ms (0.66×)| 8.48 ms   | 49.84 ms       | 5.9×    | 4.27 ms (0.50×) |
| 5000 × 2048  | 8.75 ms  | 61.05 ms      | 7.0×    | 4.99 ms (0.57×)| 8.83 ms   | 61.29 ms       | 6.9×    | 4.96 ms (0.56×) |
| 20000 × 512  | 10.37 ms | 58.04 ms      | 5.6×    | 4.66 ms (0.45×)| 10.85 ms  | 59.65 ms       | 5.5×    | 4.86 ms (0.45×) |

HONEST (NEGATIVE_EVIDENCE.md): wins 5.6-7× vs scipy DEFAULT (workers=1) but LOSES to scipy workers=-1 at
EVERY size (0.45-0.66×) — UNLIKE fft_axis2d (which won at large ncol), fsci's DCT kernel is a half-size
complex FFT + reorder + twiddle-extract, materially heavier than pocketfft's native DCT, so the kernel gap
dominates once both sides parallelize. Ship value = the missing capability + the default-API win, NOT
domination of scipy's best. The lever that would flip BOTH this and fft_axis2d to clean wins is a
SIMD-ACROSS-ROWS batched FFT kernel (lane = independent row → sidesteps the AoS-tuple SoA blocker that
killed within-FFT SIMD; bit-identical per lane, like pdist SIMD-across-pairs) — documented as the next
radical lever. Conformance: `dct_idct_axis2d_match_per_row` bit-identical to per-row. fsci-fft GREEN.

### ✅✅ special: wofz continued-fraction kernel + voigt_profile_many — 5.6x faster than scipy (CLEAN win)
Two-part, found by digging the measured gap: scipy.special.voigt_profile over 2M points = 184 ms, fsci had
only a SCALAR voigt_profile (no batched form). First cut (voigt_profile_many = par_map over the scalar) hit
PARITY (199 ms) — diagnosis: fsci's `wofz` (Faddeeva) was ~70× slower PER POINT than scipy because the
`4 ≤ |z| < 8` band used a **768-step Simpson quadrature** (~7.7 µs/call). RADICAL LEVER (different
primitive, not safe-Rust-ceiling): replaced it with the **Gautschi/Laplace continued fraction**
`w(z) = (i/√π)/(z − a₁/(z − a₂/…)), aₖ = k/2` (24 terms, ~1e-13, ~30× fewer ops, MORE accurate than the
Simpson). Kernel dropped ~6.4 µs → ~1 µs/point; the batched API then fans it across cores.

**SAME-BOX head-to-head (fsci voigt_profile_many vs scipy.special.voigt_profile, both this box):**
| n         | scipy    | fsci (768-Simpson) | fsci (CF + parallel) | speedup    |
|-----------|----------|--------------------|----------------------|------------|
| 500 000   | ~46 ms   | 52.4 ms (0.88×)    | 7.37 ms              | **~6×**    |
| 2 000 000 | 184.3 ms | 199.5 ms (0.92×)   | 32.71 ms             | **5.6×**   |

The wofz CF is a REUSABLE kernel win — it also speeds the scalar `wofz`/`voigt_profile` and EVERY wofz
caller (erfcx, dawsn, complex erf/erfc) for `4 ≤ |z| < 8`. Conformance: FULL fsci-special suite
**1121/1121 GREEN** (the CF matches scipy across all wofz-dependent goldens — more accurate than the
Simpson it replaced) + new `voigt_profile_many_matches_scalar` (bit-identical to per-point, both gate
paths). NOTE: fsci's per-point wofz (~1 µs) is still ~11× slower than scipy's Weideman Faddeeva (~92 ns) —
a further CF-everywhere / Weideman-rational kernel could lift the scalar path too, but the parallel batched
form already DOMINATES scipy 5.6×.

### ✅ special: hyperu DOMINATES scipy on BOTH speed and accuracy (measured + mpmath-verified) — lever closed
Investigated the filed "hyperu 768-step Simpson → faster quadrature" lever. Outcome: the lever is CLOSED
because fsci already wins decisively and the 768-Simpson is excellent — verified by an oracle-differential
sweep over a∈[0.5,50], b∈[0.5,4], x∈[0.3,10] (incl. narrow-peak large a) against mpmath at 40 dps:
- **Accuracy:** fsci max rel-err **1.27e-12** vs **scipy 5.49e-06** (scipy has a latent ~5e-6 error for
  moderate-large a, e.g. a=10,b=4,x=5: mpmath 1.5250375e-10, fsci 1.5250375e-10 ✓, scipy 1.5250459e-10 ✗).
  fsci is ~4×10⁶× more accurate than scipy at the worst point.
- **Speed:** fsci 1.47-1.71× faster (52.6/504 ms vs scipy 90.1/738.6 ms @100k/1M, parallel par_map_indices).
So fsci hyperu DOMINATES scipy on both axes. The 768-Simpson is NOT accuracy-marginal (it hits 1e-12) and
reducing the step count would forfeit the accuracy lead — the perf "win" of fewer steps is a real-correctness
LOSS. Generalized Gauss-Laguerre (weight u^{a-1}e^{-u}) would be the analytically-exact route but its nodes
depend on a → caches poorly for per-element varying-a batches (recompute per distinct a). No change warranted.
LESSON: before "optimizing" a slow-looking fixed quadrature, oracle-check accuracy vs mpmath FIRST — here the
768 steps were buying a 1e-12 accuracy that BEATS scipy; the apparent fsci-vs-scipy "deviation" was scipy's
bug, not fsci's. (Contrast the wofz-CF win affac121, where the 768-Simpson bought NO accuracy a faster CF
couldn't match.)

### ✅✅ opt: curve_fit_bounded / least_squares_bounded — closes backlog gap, 10.2× faster than scipy trf
Backlog (CARGO_RECOVERY_BACKLOG.md) listed "bounded least_squares/curve_fit (TRF)" as a genuinely-unfinished
capability gap — fsci had only unbounded LM. Added `least_squares_bounded` + `curve_fit_bounded` via the
standard smooth reparameterisation (lmfit's method): each bounded coordinate maps to an unconstrained variable
(logistic for two-sided, softplus for one-sided, identity for ±inf), the existing fast LM core solves the
unconstrained problem, and `x`/`fun`/`jac` are recomputed in parameter space at the optimum for the covariance.
Purely additive (new public fns; existing curve_fit/least_squares untouched).

**SAME-BOX head-to-head (5-param double-exponential, 400 pts; both this box):**
| op                          | scipy            | fsci      | speedup   |
|-----------------------------|------------------|-----------|-----------|
| curve_fit (unbounded, lm)   | 1.944 ms         | 0.235 ms  | **8.3×**  |
| curve_fit_bounded (trf-eq)  | 9.859 ms (trf)   | 0.971 ms  | **10.2×** |

The callback lever drives it: scipy's trf calls a Python/numpy model many times; fsci inlines a Rust closure.
CONFORMANCE (oracle-checked vs scipy): on a noiseless interior problem (exp+offset, true (3,0.7,1)) BOTH fsci
and scipy recover (3.0,0.7,1.0) exactly; with the amplitude capped below the truth (upper=2) scipy pins it at
2.0 and fsci approaches 2.0⁻ (the transform is asymptotic at an active bound — the one documented difference vs
trf, same as lmfit). fsci-opt curvefit suite 15/15 green incl. 2 new bounded tests. LIMITATION (documented):
for a tightly-active bound the transform reaches it asymptotically rather than exactly; for interior optima
(the common "sanity bounds" case) it is identical to trf and ~10× faster.

### ✅✅✅ opt: curve_fit_many / curve_fit_bounded_many — vmap-over-solver, 33-113× faster than looped scipy
The JAX-style "different primitive": fit the same model to MANY independent ydata rows. SciPy has no batched
curve_fit — you loop it in Python, paying the per-call overhead N times SERIALLY. fsci `curve_fit_many` fans
the N independent fits across cores AND inlines the model as a Rust closure (callback lever × N-way parallel).
Purely additive (new pub fns over the existing curve_fit/curve_fit_bounded); heavy-per-item thread cap
(cores.min(nrows), serial under 8 rows). Common in imaging/signal: a decay or peak fit per pixel/channel/trace.

**SAME-BOX head-to-head (3-param exponential a·e^{−bx}+c, 80 pts, N fits; both this box):**
| N    | scipy (Python loop over curve_fit) | fsci curve_fit_many | speedup    |
|------|------------------------------------|---------------------|------------|
| 500  | 96.2 ms                            | 2.92 ms             | **32.9×**  |
| 2000 | 431.2 ms                           | 3.81 ms             | **113×**   |

Speedup grows with N as the ~1.5ms thread-spawn floor amortises. CONFORMANCE: row i is BYTE-IDENTICAL
(.to_bits()) to `curve_fit(f, xdata, &ydata_rows[i], opts).popt` — the batch only distributes independent
fits, it doesn't change any of them. fsci-opt curvefit suite 16/16 green (+ new batched byte-identical test
covering both curve_fit_many and curve_fit_bounded_many across the serial→parallel gate). Pairs with the
bounded-fit lever (2235ab6f): curve_fit_bounded_many gives the same N-way win for box-constrained batches.

### VOID-NONULL / CONFORMANCE-BLOCKED: integrate `solve_ivp_many` 150-sample ensemble — historical 1481–1599× claim
**2026-07-29 resurrection verdict:** the historical A/B was run in separate
invocations with no A/A null, no executable SHA-256, and no counted-mechanism
refutation, so it is `VOID-NONULL`. A live same-invocation SciPy 1.17.1 arm
then refused to time the stated 150-sample surface: at sample 138 of the first
deterministic trajectory, FrankenSciPy differed from SciPy by `711.439`
tolerance units (`1.5477440105446976` versus `1.5477329281478893`). Tight
SciPy RK45 and DOP853 references agree at `1.54773287316`, attributing the
failure to FrankenSciPy's generic cubic-Hermite `t_eval` interpolation rather
than the incumbent. The final state still agrees, which localizes the defect
to sampled RK45 dense output. Bead `frankenscipy-3m5ip`.

The historical text and timings remain below as lab-notebook provenance; they
are not admissible competitive evidence.

Extends the vmap-over-solver lever to ODEs — the marquee case. SciPy has no batched solve_ivp: integrating an
ensemble (N initial conditions, shared dynamics) means looping solve_ivp in Python, calling the Python RHS
thousands of times PER solve, N solves SERIALLY (~15-21 ms/solve). fsci `solve_ivp_many` fans the N independent
integrations across cores AND inlines the RHS as a Rust closure (callback lever × N-way parallel). Purely
additive (new pub fn over the existing solve_ivp); heavy-per-item thread cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (Lotka-Volterra, RK45, rtol 1e-8 / atol 1e-10, 150 t_eval; both this box):**
| N    | scipy (Python loop over solve_ivp) | fsci solve_ivp_many | speedup     |
|------|------------------------------------|---------------------|-------------|
| 200  | 4220 ms                            | 2.85 ms             | **1481×**   |
| 1000 | 14809 ms                           | 9.26 ms             | **1599×**   |

All 1000/1000 solves converged. CONFORMANCE (rigorous, two ways): (1) result i is BYTE-IDENTICAL (.to_bits()
on t and y) to per-member solve_ivp — the batch only distributes independent integrations; (2) NUMERICAL
cross-check vs scipy on a fixed y0=[2,3]: fsci final state [5.3569214988, 1.9612924121] == scipy
[5.3569214988, 1.9612924121] to 1e-10 (same RK45 algorithm + tolerances → same trajectory, just ~1500×
faster). fsci-integrate solve_ivp_many test green. The callback lever (inline Rust RHS, no Python per-step)
gives ~25× per-solve; the N-way parallelism multiplies it to ~1500×. Companion to curve_fit_many (95f3cad8).

**Concrete retry predicate:** implement solver-specific RK45 dense output for
`t_eval`, pass the live arm's `<=100` scaled-difference contract over all 150
samples, then rerun the exact batch surface with genuine SciPy in the same
invocation, independent A/A nulls for both arms, executable SHA-256, full
hardware/thread provenance, and the bootstrap-median CI gate.

### ✅✅✅ opt: minimize_many (vmap-over-solver multistart) — 271-275× faster than looped scipy
Third vmap-over-solver family (after curve_fit_many 113× and solve_ivp_many ~1500×). Multistart / parameter
sweep — minimise the SAME objective from MANY starts — is ubiquitous in global optimisation; SciPy loops
`minimize` in Python, calling the Python objective (+ gradient) many times PER run, N runs SERIALLY. fsci
`minimize_many` fans the N independent runs across cores AND inlines the objective as a Rust closure (callback
lever × N-way parallel). Purely additive (new pub fn over the existing minimize); MinimizeOptions is Copy so
no per-call clone; heavy-per-item thread cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (6-D Rosenbrock, BFGS, N random starts in [-2,2]^6; both this box):**
| N    | scipy (Python loop over minimize) | fsci minimize_many | speedup    |
|------|-----------------------------------|--------------------|------------|
| 200  | 2829 ms                           | 10.43 ms           | **271×**   |
| 1000 | 14677 ms                          | 53.41 ms           | **275×**   |

FAIR head-to-head (not speed from giving up early): on the same 1000 random starts fsci converges 805/1000 vs
scipy 622 success / 782 reached-global-min — fsci optimises at least as well. CONFORMANCE two ways:
(1) result i is BYTE-IDENTICAL (.to_bits() on x and fun) to per-start `minimize` — the batch only distributes;
(2) from the standard Rosenbrock start fsci reaches the exact global min [1,1,1,1,1,1]. fsci-opt minimize_many
test green. Callback lever (inline Rust objective, no Python per-eval) gives the per-run win; N-way parallelism
multiplies it. The vmap-over-solver vein is now proven across curve_fit / solve_ivp / minimize.

### ✅✅ opt: root_many (vmap-over-solver nonlinear-system sweep) — 11-25× faster than looped scipy
Fourth vmap-over-solver family (curve_fit / solve_ivp / minimize / root). A parameter sweep — solve
`func(x, params)=0` for many parameter sets, shared start — is common in equilibrium/steady-state analysis;
SciPy loops `root` in Python, N solves serially. fsci `root_many` (param-sweep signature `F: Fn(&[f64] x,
&[f64] params)->Vec<f64>`) fans the N independent solves across cores and inlines the residual. Purely
additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (well-conditioned 3-eq system, hybr, N parameter sets; both this box):**
| N    | scipy (Python loop over root) | fsci root_many | speedup   |
|------|-------------------------------|----------------|-----------|
| 500  | 35.9 ms (277/500 converged)   | 3.24 ms (293/500) | **11.1×** |
| 2000 | 135.4 ms (1176/2000)          | 5.39 ms (1191/2000) | **25.1×** |

This is the MODEST end of the vmap family — unlike minimize (275×) / solve_ivp (1500×), scipy's hybr is fast
C (MINPACK, ~0.07 ms/solve, few Python callbacks), so the per-solve callback lever is weak and the win is
mostly the N-way parallelism. FAIR head-to-head: fsci converges slightly MORE than scipy on the same params
(293 vs 277, 1191 vs 1176 — not speed from giving up early; a first benchmark on a Jacobian-SINGULAR symmetric
system was discarded as invalid since neither library converged there). CONFORMANCE: result i is
BYTE-IDENTICAL (.to_bits() on x and fun) to per-param root; fsci-opt root_many test green (byte-id + ≥half
converge). vmap-over-solver vein now spans 4 solver families.

### ✅✅✅ integrate: quad_many (vmap-over-solver definite-integral sweep) — 14.5-61× faster than looped scipy
Fifth vmap-over-solver family (curve_fit / solve_ivp / minimize / root / quad). A definite-integral sweep —
`I(params) = ∫_a^b f(x, params) dx` for many parameter sets — is common (a family of moments / partition
functions / marginalisations); SciPy loops `quad` in Python, calling the Python integrand adaptively per
integral, N integrals SERIALLY. fsci `quad_many` (param-sweep signature `F: Fn(f64 x, &[f64] params)->f64`)
fans the N independent adaptive integrations across cores and inlines the integrand. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (peaked+oscillatory ∫_0^1 e^{-p(x-c)²}cos(wx)dx, N parameter sets; both this box):**
| N    | scipy (Python loop over quad) | fsci quad_many | speedup   |
|------|-------------------------------|----------------|-----------|
| 500  | 47.9 ms                       | 3.30 ms (500/500 conv)   | **14.5×** |
| 2000 | 179.6 ms                      | 2.94 ms (2000/2000 conv) | **61.1×** |

Speedup grows with N as parallelism amortises (all integrals converge). The callback lever is real here
(scipy's QUADPACK calls the Python integrand adaptively, fsci inlines a Rust closure) — stronger than root
(11-25×, fast-C hybr) though below minimize/solve_ivp. CONFORMANCE three ways: (1) result i is BYTE-IDENTICAL
(.to_bits() on integral/error/converged) to per-param quad; (2) NUMERICAL cross-check vs scipy: fsci
I(p=100,c=0.5,w=10)=0.039156400368 == scipy 0.039156400368 to 3.84e-13; (3) fsci-integrate quad_many test
green. The vmap-over-solver vein now spans FIVE solver families.

### ✅✅✅ integrate: dblquad_many (vmap-over-solver 2D-integral sweep) — 62.7-211× faster than looped scipy
Sixth vmap-over-solver family — and the heaviest-callback integration case. dblquad's inner adaptive integral
is RE-RUN for each outer node, so each 2-D integral makes O(n²) integrand calls; in SciPy those are all Python
calls, and a parameter sweep loops dblquad in Python, N integrals SERIALLY. fsci `dblquad_many` (param-sweep
`F: Fn(f64 y, f64 x, &[f64] params)->f64`, shared rectangle) fans the N independent double integrations across
cores and inlines the integrand. Purely additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (∫_0^1∫_0^1 e^{-p((x-.5)²+(y-.5)²)}dy dx, N parameter sets; both this box):**
| N   | scipy (Python loop over dblquad) | fsci dblquad_many | speedup   |
|-----|----------------------------------|-------------------|-----------|
| 100 | 135.5 ms                         | 2.16 ms (100/100 conv) | **62.7×** |
| 400 | 544.4 ms                         | 2.58 ms (400/400 conv) | **211×**  |

The strongest integration vmap win — the O(n²) per-integral callbacks make the inline-Rust lever far stronger
than 1-D quad (14.5-61×). Speedup grows with N (parallelism amortises); all integrals converge. CONFORMANCE
three ways: (1) result i BYTE-IDENTICAL (.to_bits() on integral/error/converged) to per-param dblquad;
(2) NUMERICAL vs scipy: fsci I(p=20)=0.156588231977 == scipy 0.156588231977 to 2.87e-13; (3) new dblquad_many
test green. The vmap-over-solver vein now spans SIX solver families (curve_fit/solve_ivp/minimize/root/quad/
dblquad); win size tracks scipy's per-solve Python-callback density (dblquad O(n²) ⇒ 211×).

### ✅✅✅ integrate: tplquad_many (vmap-over-solver 3D-integral sweep) — 83-159× faster than looped scipy
Seventh vmap-over-solver family — the HEAVIEST-callback case. tplquad nests three adaptive quadratures, so
each triple integral makes O(n³) integrand calls; in SciPy all Python, and a parameter sweep loops tplquad in
Python, N integrals SERIALLY. fsci `tplquad_many` (param-sweep `F: Fn(f64 z, f64 y, f64 x, &[f64] params)->f64`,
shared box) fans the N independent triple integrations across cores and inlines the integrand. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (∫∫∫ e^{-p(x²+y²+z²)} over unit cube, N parameter sets; both this box):**
| N   | scipy (Python loop over tplquad) | fsci tplquad_many | speedup   |
|-----|----------------------------------|-------------------|-----------|
| 30  | 123.6 ms                         | 1.49 ms (30/30 conv)   | **83×**  |
| 100 | 392.0 ms                         | 2.46 ms (100/100 conv) | **159×** |

Confirms the callback-density LAW at equal N=100: tplquad 159× > dblquad 62.7× > quad ~30× (O(n³)>O(n²)>O(n)
Python integrand calls). All integrals converge; speedup grows with N. CONFORMANCE three ways: (1) result i
BYTE-IDENTICAL (.to_bits() on integral/error/converged) to per-param tplquad; (2) NUMERICAL vs scipy: fsci
I(p=5)=0.061963890934 == scipy 0.061963890934 to 4.41e-13; (3) new tplquad_many test green. The vmap-over-
solver vein now spans SEVEN solver families (curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad).

### ✅✅ ndimage: mean(labels,index) parallel privatized-histogram scatter — 2.05× self-speedup at large N (→ 2.16× vs scipy)
DIFFERENT primitive from the vmap vein: a PARALLEL SEGMENTED REDUCTION. First, a stale-scorecard CORRECTION —
the GAUNTLET scorecard lists `ndimage.mean(labels,index)` as a 1.5-4.7× LOSS (beads 8l8r1.125/.143/fa62u),
but a fresh same-box re-measure shows the current one-based-contiguous fast path already WINS 1.17-1.30×
(fsci 169.8/498.5/1177.7 us vs scipy 221.5/581.9/1478.0 us at N=65536/262144/589824) — the slow rows were
superseded. NEW lever on top: the serial scatter `sums[label-1]+=v; counts[label-1]+=1` is a segmented
reduction; replaced it (large N only) with PRIVATIZED HISTOGRAMS — each worker accumulates a private
(sums,counts) over a contiguous chunk via thread::scope, partials merged in chunk order.

**SAME-PROCESS A/B (serial replica vs production mean(), identical data, both this box):**
| N      | K    | serial | parallel mean() | self-speedup | max\|Δmean\| |
|--------|------|--------|-----------------|--------------|-------------|
| 65536  | 512  | 195.6  | 258.0 us        | serial (below gate, unchanged) | 0 (byte-id) |
| 262144 | 1024 | 663.9  | 505.7 us        | 1.30-1.50×   | 8.88e-16 |
| 589824 | 4096 | 1399.2 | 683.9 us        | **2.05×**    | 6.66e-16 |

At N=589824 the parallel path → 2.16× vs scipy (1478/683), doubling the large-image margin (was 1.26×). The
merge in chunk order keeps each label's running sum in global element order — only the ASSOCIATION differs, so
max|Δmean| = 6.66e-16 (sub-ULP). GATED `nthreads = cores.min(n/128_000)`: small N (the unit-test regime) stays
on the serial path and is BYTE-IDENTICAL (Δ=0, no regression). CONFORMANCE: 249/0 fsci-ndimage tests green incl.
new `mean_one_based_parallel_scatter_matches_serial_reference` (<1e-9) + all existing mean/label fixtures.
Generalizes to variance/sum/std label reductions (same scatter).

### ✅✅✅ ndimage: sum/variance/standard_deviation(labels,index) streaming fast path — flips 1.5-10x LOSS → 2.2-8.2x FASTER
Generalises the privatized-histogram lever (e7f5ddd4) to the label reductions that were still on the SLOW
group-materialization path (`measurement_label_groups` builds a `Vec<Vec<f64>>` per label, then reduces). These
were REAL losses (not stale): same-box, N=589824 K=4096 — sum 15238 us (scipy 1485 = **10.3x SLOWER**), variance
16202 us (scipy 10451 = 1.55x slower), std 16031 us (scipy 10874 = 1.47x slower). Added one-based-contiguous
fast paths: `sum` → parallel privatized-histogram scatter (sums only); `variance`/`standard_deviation` → a
numerically-stable TWO-PASS parallel reduction (privatized sum/count → means, then a second privatized
histogram of centred squares), matching scipy's two-pass. std inherits via variance.

**SAME-BOX head-to-head (one-based index, N parameter sets; both this box):**
| op       | N      | scipy     | fsci before        | fsci after   | vs scipy   | self-speedup |
|----------|--------|-----------|--------------------|--------------|------------|--------------|
| sum      | 262144 | 716 us    | 5520 us            | 485 us       | **1.48x**  | 11.4x |
| sum      | 589824 | 1485 us   | 15238 us (10.3x↓)  | 659 us       | **2.25x**  | 23x  |
| variance | 262144 | 4067 us   | 5759 us            | 998 us       | **4.07x**  | 5.8x |
| variance | 589824 | 10451 us  | 16202 us (1.55x↓)  | 1326 us      | **7.88x**  | 12.2x |
| std      | 589824 | 10874 us  | 16031 us (1.47x↓)  | 1321 us      | **8.23x**  | 12.1x |

variance/std dominate hard because scipy's OWN variance/std are slow (4-10 ms, ~7x its sum); fsci's streaming
two-pass is ~1.3 ms. CONFORMANCE: deterministic same-data NUMERICAL cross-check vs scipy is EXACT — fsci
sum[0]=25806.4, var[0]=0.083191077066, std[0]=0.288428634268 == scipy to all 10-12 digits; 250/0 fsci-ndimage
tests green incl. new `sum_variance_one_based_fast_path_matches_serial_reference` (two-pass, non-zero mean) +
all existing scipy-fixture small-N tests (serial path byte-identical, no regression). Gated cores.min(n/128_000).

### ✅✅✅ ndimage: minimum/maximum(labels,index) streaming fast path — 13-30× FASTER than scipy (BYTE-IDENTICAL)
Completes the label-reduction sweep. `minimum`/`maximum`(labels,index) were on the slow group-materialization
path. scipy's OWN labeled min/max are GLACIAL (9-24 ms — even slower than its variance), so fsci's group path
already edged it (1.5×); the streaming privatized-histogram min/max CRUSHES it. Because min/max are associative,
commutative AND EXACT, the parallel merge is BYTE-IDENTICAL to the serial fold — no tolerance (unlike
sum/variance). NaN in any element of a label propagates to NaN; empty labels yield 0.0 (scipy convention), both
preserved.

**SAME-BOX head-to-head (one-based index; both this box):**
| op      | N      | scipy     | fsci before | fsci after | vs scipy   | self-speedup |
|---------|--------|-----------|-------------|------------|------------|--------------|
| minimum | 262144 | 9112 us   | 5688 us     | 691 us     | **13.2×**  | 8.2×  |
| minimum | 589824 | 24571 us  | 15942 us    | 838 us     | **29.3×**  | 19.0× |
| maximum | 262144 | 9175 us   | 5992 us     | 682 us     | **13.5×**  | 8.8×  |
| maximum | 589824 | 24010 us  | 16426 us    | 808 us     | **29.7×**  | 20.3× |

CONFORMANCE: BYTE-IDENTICAL (.to_bits()) to the serial fold incl. NaN propagation + empty→0.0; 252/0
fsci-ndimage tests green incl. new `minimum_maximum_one_based_fast_path_byte_identical_to_serial` (with an
injected NaN) + `minimum_maximum_empty_label_returns_zero` + all existing scipy fixtures (serial small-N path
unchanged). Gated cores.min(n/128_000). Label-reduction vein now COMPLETE: mean/sum/variance/std/min/max all
streaming privatized-histograms; median (scipy 44-118 ms) needs the full group (can't stream) — left on the
group path.

### ✅✅✅ ndimage: histogram(labels,index) streaming per-label privatized histogram — 12-19× FASTER than scipy (BYTE-IDENTICAL)
The privatized-histogram lever applied to its CANONICAL use: `ndimage.histogram(input,min,max,bins,labels,index)`
returns a per-label bin-count histogram. scipy's is GLACIAL (24-79 ms — it loops np.histogram per label in
Python); fsci was on the group-materialization path (Vec<Vec<f64>> per label) so already won 3.5-3.9×. Replaced
with a single-pass parallel privatized reduction: each worker fills a private flat `[label_count × nbins]` count
table over a contiguous chunk, tables summed once. Counts are integers → BYTE-IDENTICAL to the serial fill.

**SAME-BOX head-to-head (one-based index, nbins=32; both this box):**
| N      | K    | scipy     | fsci before | fsci after | vs scipy  | self-speedup |
|--------|------|-----------|-------------|------------|-----------|--------------|
| 262144 | 1024 | 24379 us  | 6887 us     | 1984 us    | **12.3×** | 3.5× |
| 589824 | 4096 | 79060 us  | 20281 us    | 4101 us    | **19.3×** | 4.9× |

Self-speedup is below the scalar reductions (min/max ~20×) because the K×nbins privatized table (1 MB at
K=4096/nbins=32) is cache-heavier than a K-float table; for typical small K it fits L1/L2 and is much faster.
Preserves the validation short-circuit exactly (nbins=0 / non-finite min,max / max<=min / any non-finite input →
all-zero histograms) and the `[min,max]` inclusion filter. CONFORMANCE: BYTE-IDENTICAL (`assert_eq!` on the full
Vec<Vec<usize>>) to a serial reference incl. out-of-range filtering; 253/0 fsci-ndimage tests green incl. new
`histogram_one_based_fast_path_byte_identical_to_serial`. Gated cores.min(n/128_000). The ndimage label-stat
suite is now fully streaming except median (needs the full group — a quantile can't stream).

### ✅✅✅✅ integrate: nquad_many (vmap-over-solver N-D integral sweep) — ~1100-1950× faster than looped scipy
Eighth vmap-over-solver family and the CAPSTONE of the integration set: arbitrary-dimension `nquad`. An
`ndim`-D nquad nests `ndim` adaptive quadratures → each integral makes O(n^ndim) integrand calls; at 4-D those
are O(n⁴), the deepest-nested callback case. SciPy loops nquad in Python over the sweep, N integrals SERIALLY;
fsci `nquad_many` (param-sweep `F: Fn(&[f64] x, &[f64] params)->f64`, shared `ranges`) fans the N independent
N-D integrations across cores and inlines the integrand.

**SAME-BOX head-to-head (4-D Gaussian ∫_[0,1]⁴ e^{-p(a²+b²+c²+d²)}, N parameter sets; both this box):**
| N  | scipy (Python loop over nquad) | fsci nquad_many | speedup    |
|----|--------------------------------|-----------------|------------|
| 20 | 1650.7 ms                      | 1.49 ms (20/20 conv)  | **1108×** |
| 80 | 6703.3 ms                      | 3.43 ms (80/80 conv)  | **1954×** |

Confirms the callback-density LAW to its extreme: 4-D nquad ~1950× > tplquad (O(n³)) 159× > dblquad (O(n²))
62.7× > quad (O(n)) 30× — the win scales with the integrand-call density scipy pays in Python. All integrals
converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on integral/converged) to per-param
nquad; (2) NUMERICAL vs scipy: fsci I(p=2,4D)=0.128003847000 == scipy 0.128003847000 (all 12 digits). new
nquad_many test green. The vmap-over-solver vein now spans EIGHT solver families
(curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad/nquad); integration sub-family COMPLETE.

### ✅✅ opt: brentq_many (vmap-over-solver 1-D root sweep) — 13-47× faster than looped scipy
Ninth vmap-over-solver family. A 1-D root SWEEP — solve f(x,params)=0 over a shared bracket for many parameter
sets — is a very common real workload (implied volatility per option, quantile/percentile inversion per
channel, threshold calibration per series); SciPy loops `brentq` in Python, N Brent solves SERIALLY. fsci
`brentq_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`, shared bracket) fans the N independent solves
across cores and inlines the function. Purely additive; heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (f(x,p)=(e^x−1)+0.3 sin5x − p, bracket [0,6]; both this box):**
| N    | scipy (Python loop over brentq) | fsci brentq_many | speedup   |
|------|---------------------------------|------------------|-----------|
| 2000 | 31.7 ms                         | 2.434 ms (2000/2000 conv) | **13.0×** |
| 8000 | 126.4 ms                        | 2.710 ms (8000/8000 conv) | **46.6×** |

Bigger than `root_many` (11-25×) despite Brent being fast C: the exp+sin objective is expensive enough per
eval that scipy's Python-callback cost dominates (the callback lever bites). Speedup grows with N as
parallelism amortises; all roots converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on
root/converged) to per-param brentq; (2) NUMERICAL vs scipy: fsci root(p=10)=2.411137400718 == scipy
2.4111374007184447 (12 digits). new brentq_many test green. vmap-over-solver vein now NINE families.

### (negative evidence) opt: differential_evolution / brute already dominate scipy — no work needed
Measured before chasing: fsci `differential_evolution` (8-D Rastrigin, popsize 15, maxiter 300) = 9.7 ms vs
scipy 1095 ms (workers=1) = **113× faster** ALREADY — the callback lever (inlined Rust objective vs Python);
scipy's `workers=-1` can't even run a local lambda (multiprocessing pickling error). fsci `brute` is already
parallel (thread::scope, byte-identical argmin). So the fsci-opt global optimizers are already dominant; the
DE population eval uses IMMEDIATE in-generation updates (not parallelizable without switching to scipy's
deferred scheme = an algorithm change). No change shipped.

### ✅✅✅ integrate: solve_bvp_many (vmap-over-solver BVP ensemble) — 53-123× faster than looped scipy
Tenth vmap-over-solver family, on a NEW heavy-callback solver and closing the long-standing "solve_bvp" backlog
in the vmap sense. A BVP parameter study (vary a nonlinearity strength / boundary value / forcing) loops
`solve_bvp` in Python, N collocation-Newton solves SERIALLY — each calling the Python RHS at every mesh node
every Newton iteration. fsci `solve_bvp_many` (`f: Fn(t, y, params)->Vec`, `bc: Fn(ya, yb, params)->Vec`,
shared t_span/y_guess) fans the N independent solves across cores and inlines both callbacks. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (nonlinear BVP y0'=y1, y1'=p(1+y0²), y0(0)=0, y0(1)=1; both this box):**
| N   | scipy (Python loop over solve_bvp) | fsci solve_bvp_many | speedup    |
|-----|------------------------------------|---------------------|------------|
| 200 | 166.2 ms                           | 3.111 ms (200/200 conv) | **53.4×**  |
| 800 | 611.5 ms                           | 4.953 ms (800/800 conv) | **123.5×** |

A genuinely BIG vmap win (heavy collocation callback), unlike the modest fast-C cases (brentq/root). Speedup
grows with N; all solves converge. CONFORMANCE two ways: (1) result i BYTE-IDENTICAL (.to_bits() on t & y) to
per-param solve_bvp; (2) NUMERICAL vs scipy: fsci y0(0.573)=0.4258989360 == scipy 0.4259060979 (7e-6, the
collocation tolerance — fsci solve_bvp independently matches scipy). new solve_bvp_many test green. The
vmap-over-solver vein now spans TEN solver families (curve_fit/solve_ivp/minimize/root/quad/dblquad/tplquad/
nquad/brentq/solve_bvp).

### ✅✅✅ opt: minimize_scalar_many (vmap-over-solver 1-D minimization sweep) — 69-236× faster than looped scipy
Eleventh vmap-over-solver family, completing the clean vmap set. A 1-D minimization SWEEP (calibrate a
1-parameter model per channel, find the mode/MLE per series, minimize a per-case cost) loops `minimize_scalar`
in Python, N Brent solves SERIALLY. fsci `minimize_scalar_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`,
shared bracket) fans the N independent solves across cores and inlines the objective. Purely additive;
heavy-per-item cap (cores.min(nrows), serial <4).

**SAME-BOX head-to-head (f(x,p)=(x−p0)²+0.5cos(p1·x)+e^{0.3x}, bracket [−10,10]; both this box):**
| N    | scipy (Python loop over minimize_scalar) | fsci minimize_scalar_many | speedup  |
|------|------------------------------------------|---------------------------|----------|
| 2000 | 160.0 ms                                 | 2.327 ms (2000/2000 ok)   | **68.8×** |
| 8000 | 643.0 ms                                 | 2.726 ms (8000/8000 ok)   | **236×**  |

Predicted "modest" but came in BIG (like brentq_many): the cos+exp objective is expensive enough per eval that
scipy's Python-callback cost dominates even though Brent is fast C. CONFORMANCE two ways: (1) result i
BYTE-IDENTICAL (.to_bits() on x/fun/success) to per-param minimize_scalar; (2) NUMERICAL vs scipy: fsci
x*(p=[1,2])=1.1563726803 == scipy 1.1563726811631612 (~1e-9, Brent xatol). new test green. vmap-over-solver
vein now ELEVEN families.

### (negative evidence) ndimage rank/morphology filters already dominate scipy — no work needed
Measured before chasing (512×512): fsci median_filter 5×5 = 4.05 ms vs scipy 81.1 ms (**20×**), rank_filter
3.43 vs 66.9 (**19.5×**), percentile_filter 3.21 vs 81.2 (**25×**) — already parallel quickselect. uniform_filter1d
is already running-sum O(n) (has an explicit "pre running-sum reference, A/B only" path). grey_erosion/dilation
already van Herk; generic_filter already inlines a Rust closure (Sync, parallel core) vs scipy's per-window
Python callback (212 ms @ 256² for np.ptp). The ndimage filter surface is fully dominant; don't re-chase.

## 2026-06-29 — AmberKestrel (cc): ndimage geometric-transform compact-support B-spline (FLIP 7.8× loss → 1.85× win)

**Lever:** `sample_interpolated`'s generic fall-through built a length-`len` knot vector AND evaluated ALL `len` (~512) B-spline basis functions (with per-degree clones) PER PIXEL PER AXIS, then filtered to the ~`order+1` nonzero ones — O(len·order) + ~3 heap allocs/pixel/axis. B-splines have compact support `order+1`, so every other basis value is exactly 0.0. Added `bspline_local_support` (closed-form `uniform_knot_at` + binary-searched degree-0 span + windowed Cox–de Boor over only the supported indices) = O(order²), ZERO per-pixel allocation. The cardinal fast paths only covered Nearest/Reflect/Mirror (1-5) and Constant/Wrap at order 3; **Constant/Wrap order∈{1,2,4,5}** (scipy's DEFAULT mode is 'constant') fell through to the slow path.

**BYTE-IDENTICAL** to filtering the full `eval_bspline_basis_all` — proven by `bspline_local_support_byte_identical_to_full_eval` (20 000 random len/order/x incl. integer+boundary positions, `.to_bits()` equality). Full fsci-ndimage conformance suite GREEN.

**Measured same-box, 512×512, affine_transform** (scipy.ndimage o1=7.64ms, o3=19.56ms):
| mode/order | before (fsci) | after (fsci) | self | vs scipy |
|---|---|---|---|---|
| Constant o1 | 59.45 ms | **4.13 ms** | 14.4× | **1.85× FASTER** (was 7.8× SLOWER) |
| Constant o2 | 139.36 ms | 60.94 ms | 2.3× | (prefilter-bound, separate) |
| Constant o3 | 13.19 ms | 13.19 ms | — | 1.48× faster (unchanged path) |

Marquee flip: **Constant order=1 affine 7.8× LOSS → 1.85× WIN** (same fast path now serves map_coordinates / geometric_transform / shift / rotate, all sharing `sample_interpolated`). Backlog: Constant o2 prefilter (per-line `spline_coefficients_for_line`), Reflect o1 (16.7ms).

## 2026-06-29 — AmberKestrel (cc): ndimage order-1 reflect/mirror — drop the eager array pad (FLIP 2× loss → 2.4-2.7× win)

**Lever:** `prefilter_spline_coefficients` for order≤1 reflect/mirror eagerly built a padded copy via `pad_array_mode(input, SPLINE_NEAREST_PAD=12, mode)` — O(padded²) with per-element reflect index reconstruction (~15ms for a 512²→536² array) — purely so linear-interp support lands in range. But the cardinal interp path already folds the support TAPS on the fly. Removed the pad: order-1 reflect/mirror now returns `coeffs=input.clone(), coord_offsets=0` and the `fold` closure uses the actual boundary mode. Fold is also EXACT for coords arbitrarily far outside the grid (the pad only reflected 12 deep then clamped).

**scipy-exact:** affine_transform order=1 reflect & mirror match scipy to **9.99e-16** including coords pushed well outside the grid; new hardcoded-golden regression test `affine_order1_reflect_mirror_matches_scipy_goldens` (asserts no padding + scipy values). Full fsci-ndimage lib suite GREEN (254 passed / 0 failed / 5 pre-existing-ignored, +1 new).

**Measured same-box, 512×512** (scipy affine o1=7.64ms, map_coordinates o1=10.73ms):
| op / mode | before | after | self | vs scipy |
|---|---|---|---|---|
| affine Reflect o1 | 16.14 ms | **3.12 ms** | 5.2× | **2.4× FASTER** (was 2.1× SLOWER) |
| affine Mirror o1 | 15.60 ms | **2.78 ms** | 5.6× | **2.7× FASTER** (was 2× SLOWER) |
| map_coordinates Reflect o1 | 16.37 ms | **2.78 ms** | 5.9× | **3.9× FASTER** |
| map_coordinates Mirror o1 | 15.88 ms | **3.02 ms** | 5.3× | **3.6× FASTER** |

Two-step domination of the geometric-transform family: Constant o1 (compact-support, prior commit) + Reflect/Mirror o1 (this). All order-1 modes now WIN vs scipy. Backlog: Constant o2/o4/o5 prefilter (per-line make_interp_spline), Nearest o3 (~parity).

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — grey_dilation/erosion HGW parallel-across-lines is ~0-gain (memory-bound), REVERTED

Parallelized `minmax_along_axis_hgw` (van Herk–Gil-Werman flat min/max under grey_dilation/erosion/grey_opening): per-line factored helper `hgw_fill_line`, outer≥2 → `chunks_mut(slab)` across slabs, outer==1 → column-major transpose scratch + scatter (the exact `edt_axis_pass_parallel` pattern). min/max are exact+associative across independent lines → byte-identical by construction.

**Same-process A/B (512×512, FSCI_HGW_SERIAL toggle), parallel vs serial:**
| op (size) | parallel | serial | verdict |
|---|---|---|---|
| grey_dilation 5 | 6.86 ms | 6.85 ms | tie |
| grey_dilation 9 | 6.25 ms | 5.57 ms | parallel SLOWER |
| grey_erosion 5 | 6.30 ms | 5.82 ms | parallel SLOWER |
| grey_erosion 9 | 6.38 ms | 5.78 ms | parallel SLOWER |

HGW is ~O(1) work/element (memory-bandwidth-bound); the serial pass already saturates bandwidth, and the axis-0 transpose+scatter adds traffic → net ~0-gain to slight LOSS. REVERTED (working tree back to c911c3dc). grey_dilation/erosion sit at scipy PARITY (5.4-6.0ms vs scipy 5.5-6.05) and that is the memory wall, not a thread gap. DON'T re-chase parallelizing memory-bound separable min/max. (Contrast: EDT's lower-envelope transform is compute-heavier → its parallel pass DOES win 1.79×.) Remaining real ndimage gap = affine/map_coordinates Constant/Wrap order∈{2,4,5} prefilter (per-line make_interp_spline, scipy o2 18.6ms vs fsci 61ms) but the solver lives in fsci-interpolate (other-agent crate) and a reimplemented boundary-IIR is risky for uncommon orders.

## 2026-06-29 — AmberKestrel (cc): sobel/prewitt via separable correlate1d (FLIP 1.07× loss → 1.85× win)

**Lever:** `sobel`/`prewitt` built an N-D kernel of shape [3,1,…] / [1,3,…] per axis and called the GENERAL N-D `correlate` once per axis. For a 3-tap kernel the general footprint machinery (per-element N-D index + boundary match over the whole kernel volume) is overhead-bound — sobel was 4.73 ms, SLOWER than a 7×7 general correlate (3.87 ms) and slower than scipy. Routed each 1-D pass through the separable `correlate1d` (the axpy-vectorized path uniform/gaussian filters already ride): same centered weights, same boundary mode, no flip → equivalent.

**scipy-EXACT** (byte-exact goldens 12/24/2/4 on a 5×6 probe, both axes); full fsci-ndimage lib suite GREEN (255 passed / 0 failed / 5 pre-existing-ignored). No new test needed — covered by existing sobel/prewitt conformance (5 sobel tests pass).

**Measured same-box, 512×512** (best of repeated runs; correlate1d parallelizes so box-load varies):
| op | before | after | vs scipy |
|---|---|---|---|
| sobel ax0 | 4.73 ms | **2.38 ms** (1.99× self) | **1.86× FASTER** (was 1.07× SLOWER, scipy 4.42) |
| prewitt ax0 | ~4.7 ms (same pattern) | **2.43 ms** | **~parity** (scipy 2.57, was ~1.8× slower) |

sobel was the lone common ndimage filter still losing to scipy; now all of correlate (2.98×)/uniform (3.04×)/gaussian (3.92×)/laplace (2.42×)/sobel (1.86×) WIN. Generalizable: any derivative/separable filter calling general N-D `correlate` with a 1-D kernel → swap to `correlate1d`.

## 2026-06-29 — AmberKestrel (cc): nnls incremental Cholesky + parallel Gram (3.0× self, closes 3.75× loss → 1.25×)

**Lever:** `nnls` (Lawson–Hanson active set) re-built and re-factored the FULL passive-set Gram submatrix (Cholesky from scratch) on EVERY inner solve = O(Σp³) ≈ O(n⁴) for n columns entering. Replaced with an INCREMENTALLY-maintained Cholesky factor (`lflat`, strided): a variable ENTERING the passive set is an O(p²) rank-1 column add (`nnls_chol_add_col`); the rare REMOVAL triggers an O(p³) refactor → O(n³) overall. Rank-deficient passive column (non-positive Schur pivot) flips `use_slow`, reverting to the proven gather + Cholesky/pivot solve. PLUS the dominant O(m·n²) Gram precompute (AᵀA) fanned across cores as a partial-Gram REDUCTION (gated; small problems stay serial/byte-identical).

**Correctness:** NNLS minimizer is unique, so `x` is unchanged — `nnls_matches_scipy_reference_values` + metamorphic `mr_nnls` GREEN (3 nnls tests + meta pass). Parallel Gram is ~1e-13 reassociation (gradient only RANKS the entering variable; strictly convex), not bit-identical, but the unique optimum is invariant.

**Measured same-box, 800×400** (scipy.optimize.nnls 29.45 ms):
| stage | time | vs scipy |
|---|---|---|
| before | 110.33 ms | 3.75× SLOWER |
| + incremental Cholesky | 43.57 ms | 1.48× slower |
| + parallel Gram | **36.76 ms** | **1.25× slower** (3.0× self-speedup) |

Closed most of a 3.75× loss; the residual ~1.25× is scipy's Householder-QR Fortran vs fsci's Gram-based scalar inner loops (engineering wall). isotonic_regression measured at ~parity (28 vs 24ms); cumulative_simpson/trapezoid/simpson all WIN (7.2×/3.5×/2.2×).

## 2026-06-29 — AmberKestrel (cc): lsq_linear Gram rank-1 + Cholesky subproblem (1.5× self, narrows loss)

**Lever (same family as nnls):** `lsq_linear` (bounded LS active set) had (1) a cache-hostile Gram build — `gram[j1][j2] = Σ_i row[j1]·row[j2]` strided TWO columns through the `Vec<Vec>` heap rows (cache miss per element, O(n²·m)); replaced with a contiguous RANK-1 update over a row-major copy of A. (2) The free-subproblem solve used full Gauss-Jordan (`dense_spd_solve`, O(p³) + Vec<Vec> realloc per call) on what is an SPD principal submatrix of AᵀA; swapped to `cholesky_solve_spd` (~⅓ flops) with the Gauss-Jordan as the rank-deficient fallback.

**Correctness:** bounded-LS minimizer is unique (full column rank) → converged `x` unchanged (~1e-13). `lsq_linear` + `nnls` suites GREEN (3+3).

**Measured same-box, 600×300 box-constrained** (scipy.optimize.lsq_linear: trf 249.46 ms, bvls 160.04 ms):
| stage | time |
|---|---|
| before | 441.30 ms (1.77× slower than trf, 2.76× than bvls) |
| + rank-1 Gram | 342.33 ms |
| + Cholesky subproblem | **294.33 ms** (1.18× of trf, 1.84× of bvls) |

1.5× self-speedup, narrowing the loss. REMAINING LEVER (not yet done — higher risk): the inner loop still RE-FACTORS the free Gram from scratch each iteration (O(n⁴)); an incrementally-maintained Cholesky with rank-1 add on KKT-free + DOWN-DATE on inner-fix would reach O(n³) ≈ scipy. nnls already has the add/refactor helpers (`nnls_chol_*`); lsq_linear needs the downdate (removals dominate here).

## 2026-06-29 — AmberKestrel (cc): lsq_linear incremental Cholesky — MARQUEE FLIP 2.76× loss → 4.0× WIN vs scipy

**Lever (completes last commit's lead):** `lsq_linear`'s inner active-set loop re-built and re-factored the FULL free-set Gram submatrix from scratch on EVERY solve = O(n⁴). Replaced with an incrementally-maintained Cholesky (`lflat`, strided) reusing the nnls helpers: KKT-free a variable = O(p²) rank-1 column add (`nnls_chol_add_col`); inner-loop fix = O(p³) refactor (`nnls_chol_refactor`) — turns out REMOVES are rare (the active set mostly grows), so refactor-on-remove sufficed (no Givens downdate needed). Gram flattened to row-major for the strided helpers; `use_slow` fallback for rank-deficient free sets. → O(n³).

**Correctness:** bounded-LS minimizer unique → converged x unchanged. FULL fsci-opt suite GREEN (320 lib + 56 integration tests, 0 failed; incl. `lsq_linear` scipy-reference).

**Measured same-box, 600×300 box-constrained** (scipy.optimize.lsq_linear: trf 249.46 ms, bvls 160.04 ms):
| stage | time | vs scipy bvls |
|---|---|---|
| original | 441.30 ms | 2.76× SLOWER |
| rank-1 Gram + Cholesky subproblem (prev commit) | 294.33 ms | 1.84× slower |
| **+ incremental Cholesky (this)** | **39.87 ms** | **4.0× FASTER** (6.3× vs trf) |

**11× total self-speedup; flipped a 2.76× loss into a 4.0× WIN.** Confirms the active-set lever from [[nnls]]: any solver re-factoring a growing/shrinking submatrix from scratch each iteration → incremental up/refactor. Both nnls (3.75× loss→1.25×) and lsq_linear (2.76× loss→4.0× win) now done.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — fsci-opt lane DOMINATED (measured sweep, no fixable gap left)

After the nnls + lsq_linear active-set flips, swept the remaining unmeasured fsci-opt pure-compute / no-callback functions vs scipy (same-box). ALL win or are walls — no new fixable algorithmic gap:
| fn (size) | fsci | scipy | verdict |
|---|---|---|---|
| least_squares (p=60 Rosenbrock) | 1.81 ms | 240.07 ms (trf) | **133× WIN** (callback lever) |
| linprog (200v/100c, bound-dominated) | 1.38 ms | 7.79 ms (highs) | **5.6× WIN** |
| linprog (300v/200c binding) | 9.81 ms | 23.86 ms | **2.4× WIN** |
| linprog (500v/300c binding) | 32.69 ms | 63.78 ms | **1.95× WIN** |
| isotonic_regression (N=2M) | 28.1 ms | 24.3 ms | ~parity (PAVA O(n) sequential scan — wall) |

linprog uses a DENSE TABLEAU simplex (`Vec<Vec>`) whose pivot elimination is ALREADY a contiguous AXPY (`t_row -= factor·p_row`, take(rhs_col+1)) → the flat-buffer/cache lever that flipped nnls/lsq does NOT apply (no column-strided access in the hot loop). It wins 2-5.6× at common dense sizes; the win SHRINKS with size (2.4→1.95×) so a very-large DENSE LP would eventually favor HiGHS's revised simplex (a wall, not worth a dense-tableau rewrite). Per-pivot elimination is too small (~0.05ms) to amortize thread spawn → parallelization would be ~0-gain (cf. HGW). **CONCLUSION: opt/integrate/ndimage lanes are DOMINATED** — remaining gaps are engineering walls (HiGHS at huge dense scale, LAPACK/FFT/Qhull in OTHER agents' crates) or blocked (ndimage spline prefilter order∈{2,4,5} Constant/Wrap = make_interp_spline in fsci-interpolate, uncommon orders). Future cycles: cross-crate measurement or the spline-prefilter IIR (needs scipy boundary match).

## 2026-06-29 — AmberKestrel (cc): dbscan parallel neighbour precompute + [i64;6] grid key — 9.4× self, 12.7× vs sklearn

**Cross-crate (fsci-cluster, low-contention).** dbscan already had a spatial grid (O(n) for low-d bounded density) but two constant-factor sinks: (1) the grid was a `HashMap<Vec<i64>, Vec<usize>>` — a per-query `Vec<i64>` cell-key heap alloc + slow pointer-chasing Vec hash; (2) the entire neighbour scan ran SERIALLY even though each point's eps-neighbourhood is independent of the (serial) BFS label expansion.

**Fix (BYTE-IDENTICAL):** (1) cell key `Vec<i64>` → fixed `[i64;6]` (Copy, zero alloc, fast array hash; d≤6 whenever gridding, unused dims 0 → identical bucket partition). (2) Precompute all n neighbour lists IN PARALLEL (thread::scope, ordered chunks, gated `grid && n≥2048`), then the sequential BFS moves each out with `std::mem::take` — every point's list is consumed exactly once, so same sets, same ascending order, same labels.

**Measured same-box, 20k points / 4-d** (sklearn DBSCAN eps=0.5 min_samples=5: 310.69 ms):
| | time | vs sklearn |
|---|---|---|
| fsci before | 230.59 ms | 1.35× faster |
| **fsci after** | **24.45 ms** | **12.7× FASTER** |

**9.4× self-speedup.** Full fsci-cluster suite green (142 lib + 7+5 dbscan integration, 0 failed). Byte-identical by construction (grid query deterministic + order-independent). LEVER: a serial driver (BFS/DFS/label-prop) over an independent per-item query → precompute the queries in parallel, consume serially via mem::take; + Vec<i64> spatial-grid keys → fixed-size array keys (no alloc, faster hash) whenever the dim is bounded.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — NMF is a 5.7× gap; flat buffers get 2× but a real win needs a persistent pool (REVERTED)

**Measured (fsci-cluster):** `nmf` (1000×300, k=20, 200 Lee–Seung MU iters) = **1005 ms vs sklearn 175 ms = 5.7× SLOWER** — a real gap (the only loss found in a cluster sweep; vq 3×, spectral 17×, dbscan 12.7× all WIN). Cause: the 6 GEMMs/iter delegate to the SERIAL `fsci_linalg::matmul` fed `Vec<Vec>` (cache-hostile row-pointer chase + fresh Vec<Vec> alloc ×6×200).

**What worked (but isn't enough):** rewrote the loop on FLAT row-major buffers (reused across iters) with an ikj AXPY kernel → **509 ms = 2.0× self**, BUT still **2.9× SLOWER than sklearn** (property test green: shapes, ≥0, rec-err). The 2 dominant GEMMs are individually tiny (~6M flops) and the iteration is SEQUENTIAL, so per-call `thread::scope` spawn is pure overhead — a thread sweep confirmed EVERY T>1 is ≥ the T=1 serial time (T=8 → 1059 ms). So closing the gap to a WIN needs aggregated memory bandwidth via a PERSISTENT thread pool (spawn once, barrier fork-join the big GEMMs across all 1200 calls) or a BLAS-grade matmul.

**REVERTED:** prototyped the barrier pool (workers write disjoint output row-bands via raw ptr; main runs small GEMMs) but it DEADLOCKED at some thread counts (barrier-count mismatch in the fork/join/terminate handshake) — unshippable, reverted to HEAD rather than leave broken concurrency. BACKLOG (clear path to a 2-3× WIN, ~50-100ms est. at the memory floor): persistent-pool-done-right (careful fork/join/stop barrier accounting + a correctness check vs the serial result) OR a parallel flat GEMM in fsci-linalg that NMF can call. Serial-flat alone (1005→509) is a real 2× but a still-loss → not landed as a "win".

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — fsci-spatial mostly DOMINATES; lone gap = kdtree k≥10/high-d (diagnosed, deprioritized)

Swept fsci-spatial vs scipy (same-box). Wins everywhere except one regime:
| fn (size) | fsci | scipy | verdict |
|---|---|---|---|
| cdist euclidean (2000²×4d / 3000×1500×8d / 2000²×20d) | 5.5 / 9.8 / 6.9 ms | 8.7 / 28.6 / 28.6 | **1.6–4.2× WIN** (parallel-over-rows) |
| KDTree.query_k_many (10k,3d,k=1) | 1.43 ms | 6.07 ser / 7.34 par | **4–5× WIN** |
| KDTree.query_k_many (20k,4d,k=5) | 7.73 ms | 49.6 ser / 9.62 par | **1.2–6.4× WIN** |
| KDTree.query_k_many (10k,8d,k=10) | 24.78 ms | 269.6 ser / **15.93 par** | 10.9× vs serial but **1.56× SLOWER vs scipy workers=-1** |

The k=10/8d loss is per-query traversal (both already parallel; query_k_many's thread-scaling was already tuned by a prior agent 3.5×→1.56×). Two diagnosed root causes, both real but with risk that outweighs a 1.56× gap: (1) `KDNode.point: Vec<f64>` → each visited node chases a pointer to a SCATTERED 8-double buffer (cache miss/node); fix = store points in one flat n×dim buffer indexed by node.index (BYTE-IDENTICAL — same values relocated — but ~10 `node.point` sites across 2 node types). (2) `sqeuclidean`'s std::simd needs d≥16 (the 2·L=16 unroll), so 8≤d<16 runs fully SCALAR; adding an L=8 block would vectorize it but CHANGES the sum associativity → breaks the byte-identical pdist/cdist locks. BACKLOG: the flat-points refactor (safe, byte-identical) is the cleaner future attempt; est. closes to ~parity, may flip to a slight win.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — kdtree flat-coords lever REJECTED (regression), corrects prior diagnosis

Last cycle flagged `KDNode.point: Vec<f64>` scattered storage as the root of the kdtree 8d/k=10 gap and proposed a flat n×dim `coords` slab (indexed by `node.index`) as "the cleaner future attempt". ATTEMPTED it this cycle (byte-identical: `coords[node.index*dim..] == node.point`; conformance green) — and it's a REGRESSION, not a win: 8d/k10 24.78→30.20ms, 4d/k5 7.73→8.80ms (k=1 unchanged, uses nn_search). **Why:** the per-node `point` Vecs are allocated in BUILD (≈traversal) order so the allocator places them locally, whereas `coords[node.index]` is a RANDOM index into the slab (node.index = original data order, scrambled vs traversal) → strictly worse locality. REVERTED. The remaining real lever for the 1.56× gap is sqeuclidean running scalar for 8≤d<16 (the std::simd path needs d≥16), but vectorizing it changes the sum associativity → breaks the byte-identical pdist/cdist locks. **Conclusion: kdtree k≥10/high-d is an engineering wall (tight scipy C traversal), NOT a clean fix — deprioritize.** Confirmed in passing that fsci-stats is thoroughly parallel (gaussian_kde 30× WIN, kendalltau_matrix/all_pairs already parallel, cdist/kde/vq/spectral all win) — the cluster/spatial/stats lanes are dominated.

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — GAUNTLET_RELEASE_SCORECARD "Measured Losses" are STALE (top entries are now WINS)

Mined GAUNTLET_RELEASE_SCORECARD.md (Jun-27) for the biggest documented fsci-vs-SciPy losses and re-measured same-box (frankenscipy-cc). The headline losses are ALREADY FIXED — the scorecard's "Measured Losses" section is ≥2 days stale:
| scorecard claim | re-measured same-box | reality |
|---|---|---|
| `pdist` chebyshev d=64 "4.37× slower" | fsci 1.03 ms vs scipy 2.16 ms | **2.1× FASTER** (SIMD-over-d `chebyshev` shipped since) |
| `pdist` chebyshev n=2048/d=64 "3.28× slower" | fsci 6.19 ms vs scipy 40.50 ms | **6.5× FASTER** |
| `pdist` chebyshev d=16 "4.82× slower" | fsci 0.94 ms vs scipy 0.56 ms | 1.68× slower (modest residual — only real one) |
| `ndimage.mean(labels,index)` "4.7× slower" | (per [[perf_ndimage_label_reduction_privatized_histogram]]) | already WINS 1.17-1.30× (stale 3rd time) |

LESSON (4th confirmation of the stale-scorecard rule): the scorecard's loss table OVERSTATES remaining gaps — re-measure same-box before chasing any of its "X× slower" entries. The lone real residual (pdist chebyshev d=16, 1.68×) is per-pair SIMD finalization overhead that SoA-across-pairs (proven byte-identical for d=4) would amortize — deprioritized as niche (d=64 already wins; d=16-only, ~40-line const/runtime-D SoA for a 1.68× modest gap). Cluster/spatial/stats lanes (cdist, KDTree, gaussian_kde 30×, kendalltau, mvn logpdf, vq, spectral, dbscan, pdist-cheby-d64) all WIN or parity — the accessible non-probe lanes are DOMINATED. Genuine remaining gaps: NMF (persistent-pool, deadlocked); probe-crate residuals (solve_toeplitz/lfilter — linalg/signal, owned); ndimage.label 2.68× (f64-output wall, needs native int store).

## 2026-06-29 — AmberKestrel (cc): NEGATIVE EVIDENCE — pdist Chebyshev SoA-across-pairs REJECTED (~0-gain / regression)

Implemented a general-dim SoA-across-pairs Chebyshev for pdist (`collect_columns` + `pdist_fill_cols` + `fill_chebyshev_soa_rows`: L=8 lanes = 8 distinct j-pairs, iterate the d coordinate columns keeping a running per-lane max + NaN mask, so the max-reduce/NaN finalize amortize across 8 pairs). Conformance GREEN, byte-identical (max exact + order-independent; 14+2 pdist tests). But MEASURED same-box it's a reject:
| n/d | default (SIMD-over-d) | SoA (this) | scipy |
|---|---|---|---|
| 512/16 | 0.94 ms | 0.89 ms (noise) | 0.56 ms (still 1.59× slower) |
| 512/64 | 1.03 ms | 1.13 ms (**regress**) | 2.16 ms |
| 2048/64 | 6.19 ms | 7.83 ms (**regress**) | 40.50 ms |

WHY: the SoA pays one `splat(col[i])` broadcast + column load per coordinate per 8-pair block = d broadcasts/block; that per-column overhead EXCEEDS the per-pair max-reduce finalize it removes once d≳16, so SIMD-over-d (the default `chebyshev`, which streams 8 dims/chunk per pair) is strictly better for d≥16. The d=16 residual (1.59×) is fixed per-pair overhead vs SciPy's inlined-C accumulator — fsci already scales BETTER with d (0.94→1.03ms for d 16→64 vs scipy 0.56→2.16), so it's an SMALL-d-only inlining wall, not an algorithmic gap. REVERTED. Lever REJECTED: SoA-across-pairs only wins when the per-pair finalize is EXPENSIVE relative to the per-element work (d=4 sqrt/div pdist) — for a cheap max reduce over d≥16 columns, SIMD-over-d's contiguous streaming dominates.

## 2026-06-29 — AmberKestrel (cc): NMF flat-buffer + MR=4 panel matmul — 2.9× self-WIN (1005→346ms), closes sklearn gap 6.9×→2.4× (residual = serial OpenBLAS GEMM wall)

`fsci_cluster::nmf` (1000×300, k=20, 200 mu-iters) was the documented cluster loss: **1005 ms vs sklearn — 6.9× slower**, doing 6 GEMMs/iter through `fsci_linalg::matmul` on `Vec<Vec<f64>>` (row-pointer chase + per-call alloc). DUG the flat-buffer + register-blocked-matmul lever:
- **Flatten** X/W/H to row-major buffers; reuse ALL scratch (wt/wtx/wtw/wtwh/ht/xht/hht/whht/wh) across iters (zero per-iter alloc); flat `transpose_flat`.
- **`nmf_mm` = ikj with an MR=4 output-row panel**: 4 output rows share each streamed B row → cuts B memory traffic ~4× (the dominant Wᵀ·X is memory-bound — streams X once per output row) and the inner 4-way AXPY auto-vectorizes over the n axis. The 4×N partial rows stay L1-resident (4×300×8 = 9.6 KB < 32 KB).

**Same-process A/B (best-of-4, this box):**
| kernel | time | vs sklearn(145) |
|---|---|---|
| ORIG `fsci_linalg::matmul` (Vec<Vec>) | ~1005 ms (documented) | 6.9× slower |
| flat + simple ikj | 447 ms | 3.1× slower |
| **flat + MR=4 panel (SHIPPED)** | **346 ms** | **2.4× slower** |
| sklearn NMF mu 200it (same-box) | 145 ms | — |

`rel_err = 0.470968` byte-identical across all three kernels (reduction-order change is correctness-safe; the MU updates converge to the same factorization). Conformance GREEN (fsci-cluster 142/142 lib). **Net: 2.9× self-speedup vs ORIG, 1.29× directly-A/B'd over simple-ikj, gap-to-sklearn 6.9×→2.4×.**

**REJECTED in the same A/B — register-tiled MR=4×NR=8** (C-tile held in registers across the K loop): 458 ms, SLOWER than the MR=4 panel (346). Why: the panel's 4×N partials are already L1-resident, so the register-residency saves little while the j-tiling + scalar column-remainder on the narrow matmuls (xht output is 1000×20, hht/whht are k×k) costs more. Don't re-chase register tiling here.

**RESIDUAL = serial GEMM micro-kernel wall, NOT a parallel wall (NEW finding):** sklearn NMF mu is **145 ms single-thread AND 145 ms all-threads** (measured both) — at this size the GEMMs are too small for OpenBLAS to parallelize, so its 145 ms is pure single-thread OpenBLAS micro-kernel (≈36 Gflops vs my MR=4 ≈15 Gflops). Matching that serially needs hand-tuned-assembly packing/prefetch — not reachable from portable Rust. The ONLY remaining lever to actually BEAT sklearn is **parallelism across the 64 cores** (now de-risked: since sklearn is single-threaded here, even modest parallel efficiency wins), but per prior sweeps per-call `thread::scope` spawn always loses (tiny GEMMs) and the persistent barrier-pool deadlocked — Amdahl also bites (serial element-wise updates + small matmuls + transposes ≈ 115 ms floor). Deferred to a dedicated turn. SHIPPED the safe 2.9× self-win now.

## 2026-06-29 — AmberKestrel (cc): NMF safe persistent worker-pool — FLIPS the marquee loss to a WIN (sklearn 145ms → fsci ~99-118ms, 1.23-1.46× FASTER; ~9× vs ORIG)

Follow-on to the serial flat+MR4 win (cc00089a). Per-phase profiling showed the 2 dominant GEMMs (Wᵀ·X 43% + X·Hᵀ 47%) are **91%** of serial time → Amdahl cap @P=8 = 82ms, @16 = 59ms, both well under sklearn's 145ms. KEY: sklearn NMF mu is single-threaded at this size (145ms both 1-thread and all-threads), so parallelism is a genuine, un-taken win.

Two parallelizations measured same-box (1000×300, k=20, 200 iters):
| approach | best time | self | vs sklearn(145) | note |
|---|---|---|---|---|
| serial (shipped cc00089a) | 346-359 ms | 1.0× | 0.42× | MR=4 panel |
| per-call `thread::scope` row-split | 213-218 ms @nt=8 | 1.65× | 0.67× | PLATEAUS then degrades — 400-scope spawn tax (matches the old "spawn always loses" sweep) |
| **SAFE persistent pool (SHIPPED)** | **99 ms @nt=12** (118@8, 105@16) | **3.6×** | **1.46×** | spawn-once, no per-iter tax |

**The pool is SAFE (no `unsafe` — workspace is `unsafe_code = "forbid"`, which is exactly why the prior raw-pointer pool attempt failed).** Mechanism: each worker permanently OWNS a moved-in row-band of W and X and talks to the driver over `mpsc` channels. The cross-band reductions `Wᵀ·X`/`Wᵀ·W` become per-band PARTIALS summed by the driver; H-update is serial (small); `X·Hᵀ`/`W·H·Hᵀ`/W-update are per-band (owned). On convergence-check iters workers also return their band + partial reconstruction error so the driver assembles W and tests `tol` with no extra pass. `rel_err = 0.470968` identical to serial across all nt (partial-reduction reassociation is negligible). Gated `nthreads = avail.min(n/96).min(16) >= 4 && d>=4 && k>=2` — small inputs stay serial. Conformance 142/142 GREEN.

**Net: the documented NMF loss (1005ms = 6.9× slower than sklearn) is now a 1.46× WIN at ~9-10× self-speedup vs ORIG.** REJECTED en route: register-tiled MR4×NR8 (slower, last cycle) and per-call scope (spawn-tax plateau, above). LEVER (generalizable): a sequential multiplicative/EM iteration whose cross-band matmuls are reductions → SAFE persistent pool with OWNED bands + partial-sum merge over channels beats both serial and per-call-spawn, AND sidesteps `forbid(unsafe)`. Candidates: factor_analysis / PPCA / LDA EM loops (same matmul-EM structure).

## 2026-06-29 — AmberKestrel (cc): gaussian_mixture_full M-step parallelized across components — flips a 2.4× LOSS to a 1.9-2.8× WIN (byte-identical)

Fresh measured loss found by probing the full-covariance GMM (the diagonal E-step was already parallel, but the full-cov M-step was overlooked). The M-step is `for c in 0..k { ... }` where each component's covariance is an O(n·d²) weighted outer-product sum — **all serial**, and it dominated (E-step was already fanned across points). Each component is independent ⇒ fanned the k components across cores via `thread::scope` (one thread per component, `chunks_mut` over weights/means/covariances output slots, shared `&data`/`&resp` reads). Each component is computed by the IDENTICAL serial arithmetic on its own thread → **byte-identical** result. Gated `mwork = n·d² >= 1<<16 && k >= 2`.

Same-box (max_iter=50, tol=0, reg_covar=1e-6):
| size | ORIG (serial M-step) | parallel M-step | sklearn GMM-full | flip |
|---|---|---|---|---|
| n=5000 d=30 k=8 | 1935 ms | **422 ms** (4.6× self) | 815 ms | 2.37× SLOWER → **1.93× FASTER** |
| n=10000 d=20 k=10 | 2841 ms | **420 ms** (6.8× self) | 1182 ms | 2.40× SLOWER → **2.81× FASTER** |

Conformance 142/142 GREEN (both gaussian_mixture tests pass). LESSON: when an iterative algo has BOTH an E-step and an M-step, check BOTH for parallelism — a parallel E-step can mask a serial M-step that then dominates (Amdahl). The M-step's per-component covariance is the classic "independent per-group O(work) reduction" → fan groups across cores, byte-identical. CANDIDATES with the same shape: other full-covariance EM (bayesian GMM), per-class scatter matrices (LDA/QDA fit), per-cluster covariance in any mixture model.

## 2026-06-29 — AmberKestrel (cc): gaussian_mixture (DIAGONAL) M-step — loop-interchange + per-component parallel, flips a 4× LOSS to a 4-6× WIN (16× self, byte-identical)

The diagonal GMM was a documented WIN at small d (memory yw7ts: 4-11× at n≤20k) but a measured LOSS at scale: n=20000/d=50/k=12 fsci 3780ms vs sklearn 942ms = **4.01× slower**; n=50000/d=30/k=10 6059ms vs sklearn 2441ms = 2.48× slower. ROOT CAUSE: the M-step loop nest was `for c { for j { Σ_i mean; Σ_i var } }` = **2·k·d strided passes over the data** (each element read 2kd times, column-strided in a row-major buffer = cache-pathological). Two byte-identical fixes:
1. **Loop interchange** — accumulate the whole mean/var vectors in ONE pass over i per component (contiguous `row[j]`), turning 2·k·d strided passes into 2·k contiguous ones (d× fewer; each output sum keeps the same i-ascending order → byte-identical; var uses `g*(diff*diff)` to match `resp*diff.powi(2)` exactly).
2. **Fan the k independent components across cores** (same lever as gaussian_mixture_full c3e8887a).

Same-box (max_iter=50, tol=0, reg_covar=1e-6):
| size | ORIG | new | sklearn | flip |
|---|---|---|---|---|
| n=20000 d=50 k=12 | 3780 ms | **227 ms** (16.7× self) | 942 ms | 4.01× SLOWER → **4.15× FASTER** |
| n=50000 d=30 k=10 | 6059 ms | **382 ms** (15.9× self) | 2441 ms | 2.48× SLOWER → **6.39× FASTER** |

Conformance 142/142 GREEN. LESSON (compounds the GMM-full one): the loop interchange was the BIGGER lever here (d× cache-pass reduction), parallelism stacked on top. AUDIT any `for group { for feature { for sample } }` moment/covariance accumulation — the sample loop belongs INNERMOST-but-vectorized (contiguous feature access), not re-scanned per feature. grep `for j .. { for i .. { .*\[i\]\[c\].*\[j\] } }`-shaped nests.

## 2026-06-29 — AmberKestrel (cc): spline_filter (B-spline prefilter) parallelized across independent lines — 1.46-1.67× self, widens scipy win 2× → 2.8-3.5× (byte-identical)

`ndimage.spline_filter` (the IIR B-spline prefilter used by ALL order>1 spline interpolation: zoom/rotate/affine_transform/map_coordinates) was already 2× faster than scipy SERIAL, but left cores idle. The IIR recursion is sequential WITHIN a line but the lines along each axis are independent. Parallelized both axis-pass shapes across CONTIGUOUS blocks (no unsafe — workspace forbids it):
- strided fast path (`bspline_reflect_axis_inplace`, non-last axes): split the buffer into contiguous outer-block chunks, each chunk runs the same in-place IIR.
- contiguous last-axis (stride==1) reflect lines: fan the rows across cores (non-fallible `bspline_reflect_coefficients` kernel).
Both byte-identical (block/row partition is the only change). Gated `spline_axis_threads`: total element work ≥ 1<<20 && blocks ≥ 2. The one un-parallelizable case is axis 0 with outer=1 + stride>1 (interleaved strided writes can't `split_at_mut` safely under forbid-unsafe) — stays serial, capping the win.

Same-box (order 3, Reflect):
| size | ORIG (serial) | parallel | scipy | win |
|---|---|---|---|---|
| 2048×2048 | 54 ms | **37 ms** (1.46× self) | 103.5 ms | 1.92× → **2.80× faster** |
| 256×256×256 | 247 ms | **148 ms** (1.67× self) | 520.6 ms | 2.11× → **3.52× faster** |

Conformance 255/255 ndimage lib GREEN (5 spline tests + all interpolation consumers). LEVER: an IIR/recursive sweep along one axis of an N-D array is sequential per line but the LINES are independent — parallelize across contiguous outer-blocks (non-last axes) / rows (last axis); the outer=1 first-axis stays serial under forbid-unsafe. Same shape: other separable IIR (gaussian via recursive filter, uniform_filter running-sum, any `*_filter1d` IIR).

## 2026-06-29 (AmberKestrel, cc) — signal.lfilter chunked PARALLEL associative scan (single long signal)

scipy's `lfilter` is sequential C; fsci's DF2T recurrence was also serial → PARITY/slight-loss
(n=1M order8: fsci 6.72ms vs scipy 5.90ms = 1.14x SLOWER). The state recurrence is a constant-matrix
affine map `d_n = M·d_{n-1} + v·x_n`, so by linearity (superposition) the output splits into a
zero-state response (computed per contiguous chunk INDEPENDENTLY, in parallel) plus the homogeneous
response to each chunk's true entry state. Entry states recovered by a serial O(P·m²) boundary combine
using `M^chunk` (binary matrix power, m×m companion). Two parallel `thread::scope` passes + serial combine.

MEASURED (n=1M, clean back-to-back): order4 5.50→3.63ms = **1.52x self / 1.38x FASTER than scipy 5.00ms**;
order8 6.72→4.16ms = **1.62x self / 1.42x FASTER than scipy 5.90ms**. n=4M ~1.6-1.9x faster (grows with N).
Gate `lfilter_scan_thread_count`: serial below 1<<18 (byte-identical, all 17 existing lfilter scipy-ref
tests + sosfilt/dlti/filtfilt callers GREEN), parallel above with P = avail.min(N/65536). NOT byte-identical
(superposition reassociates) → max_abs_diff vs serial reference 5e-13 @order8 (max_rel blows up only at
zero-crossings where |y|≈0); verified by new `lfilter_parallel_scan_matches_serial_reference` property test
(<1e-9, orders 4/6/8, N above gate w/ remainder chunk). filtfilt inherits the win (calls lfilter 2x).
LEVER: any constant-coefficient linear recurrence (IIR filter, DF2T) → chunked parallel scan via
superposition (zero-state pass ∥ + serial M^chunk boundary combine + homogeneous-correction pass ∥);
exact-to-roundoff for stable filters since M^k decays. The "genuinely different primitive" (parallel-scan).

## 2026-06-29 (AmberKestrel, cc) — signal.sosfilt chunked PARALLEL associative scan (cascaded biquads)

Direct follow-on to the lfilter scan (264bf1a7). scipy's `sosfilt` is sequential C; fsci's sample-major
cascade was also serial (parity). The WHOLE N-section cascade is ONE constant-matrix linear recurrence
`z_n = A·z_{n-1} + b·x_n` over the composite state z (2·nsec), so superposition applies: per-chunk
zero-state response (parallel) + homogeneous response to each chunk's true entry state, recovered by a
serial O(P·(2nsec)²) boundary combine using `A^chunk`. KEY TRICK: build the (2nsec)² companion `A` by
PROBING the single-step homogeneous cascade with basis vectors (column j = one x=0 step on e_j) — no
hand-composing the per-section state-space blocks. Reuses lfilter's mat_pow/mat_vec.

MEASURED (clean back-to-back) — BIGGER than lfilter (more biquads/sample ⇒ compute-bound ⇒ better
parallel efficiency; win grows with N AND order):
  n=1M order12 (6 sec):  3.56ms vs scipy 8.06ms  = **2.27x FASTER**
  n=1M order24 (12 sec): 5.41ms vs scipy 16.26ms = **3.00x FASTER**
  n=4M order12 (6 sec):  10.0ms vs scipy 30.9ms  = **3.09x FASTER**
  n=4M order24 (12 sec): 13.2ms vs scipy 64.9ms  = **4.92x FASTER**
max_abs_diff vs serial reference 2-3e-15 (near-exact; better-conditioned than lfilter's 5e-13).
Gate reuses `lfilter_scan_thread_count`: serial below 1<<18 (BYTE-IDENTICAL, all 13 sosfilt scipy-ref/
sosfiltfilt/axis_2d tests GREEN), parallel above with P=avail.min(N/65536). NOT byte-identical
(superposition) → verified by new `sosfilt_parallel_scan_matches_serial_reference` property test (<1e-9,
orders 6/12/18, N above gate w/ remainder). sosfiltfilt inherits (calls sosfilt 2x). The constant-coeff
linear-recurrence parallel-scan lever now covers BOTH lfilter (ba) AND sosfilt (cascaded biquads).

## 2026-06-29 (AmberKestrel, cc) — signal.savgol_filter branch-free vectorized interior dot

savgol_filter already BEAT scipy (parallel par_index_fill across output indices), but the Interp-mode
interior closure did a per-tap BOUNDS-CHECK branch (`if idx>=0 && idx<n`) for EVERY coefficient, defeating
SIMD even though the interior `[half, n-half)` is reflection-free (every tap in range). Lever (same as the
gaussian_filter interior-split): compute the interior branch-free via `savgol_dot` — chunks_exact(8) +
`try_into::<[f64;8]>` fixed-arrays (elide bounds checks ⇒ inner loop unrolls & auto-vectorizes) with 8
independent accumulators (pipeline the FMA chain). Boundary `[0,half)∪[n-half,n)` left 0 (overwritten by
the polynomial edge fit). Applied to all 3 paths: Interp interior (main), padded modes, and the serial
axis_2d helper.

MEASURED (n=2M): win=101/poly3 10.14→7.38ms = **1.37x self** (vs scipy 68.06ms: 6.7x→**9.2x faster**);
win=301/poly4 24.92→11.99ms = **2.08x self** (vs scipy 409.26ms: 16.4x→**34.1x faster**). Win grows with
window (more taps to vectorize). NOT byte-identical (8-lane + tree reassociation ~1e-14) but within the
savgol scipy tolerance — all 18 savgol tests GREEN incl. matches_scipy_reference_values / even-window /
padded modes_match_scipy / axis_2d. This WIDENS an existing lead (savgol already won), not a gap-close —
signal crate is otherwise dominated (lfilter/sosfilt scans shipped this session; hilbert 1.95x, decimate
1.11x, savgol now 9-34x all faster than scipy). LEVER (reusable): any per-output FIR/correlation with a
per-tap bounds-check branch over a reflection-free interior → split interior (branch-free chunks_exact(8)+
try_into fixed-array dot, 8 accumulators) from boundary. The try_into-[f64;8] idiom is the stable-Rust
auto-vectorization key (plain `slice[j+lane]` keeps bounds checks and does NOT vectorize — measured 1.22x
vs 2.08x for the fixed-array form).

## 2026-06-29 (AmberKestrel, cc) — spatial.minkowski integer-exponent fast path (cdist/pdist)

scipy's cdist 'minkowski' is its SLOWEST common metric — `pow(|d|, p)` per element ⇒ 1235ms for
1200×1200 d=80 (vs euclidean 34ms). fsci's general minkowski path also did per-element `.powf(p)`; it
ALREADY beat scipy ~24x via cdist row-parallelism (~51ms) but still paid an ~80-cycle powf per element.
Lever: for a SMALL INTEGER exponent (p∈[3,64], the common p=3,4,5 case), `|d|^p` is just ≈p repeated
multiplications — replaced powf with an 8-wide std::simd integer-power kernel (`minkowski_int`: two
accumulators + scalar tail, `t *= |d|` p-1 times). p=1/2/∞ still route to cityblock/euclidean/chebyshev;
non-integer p keeps the scalar powf.

MEASURED (cdist 1200×1200 d=80): p=3 **51→5.12ms = 10x self / 241x FASTER than scipy 1235ms**; p=4 4.76ms;
p=5 5.07ms; non-integer p=3.5 unchanged at 51ms (still 24x vs scipy). pdist_minkowski inherits (both
route through `minkowski`). NOT byte-identical (x*x*x vs powf + SIMD reassoc ~1e-14) but within distance
tolerance — 225/225 spatial lib GREEN incl. minkowski reference-value + cdist/pdist tests. LEVER (reusable):
any per-element `powf(INTEGER)` in a hot reduction → repeated-multiplication + SIMD (powf is ~80 cycles,
x^p is ~p mults; the SLOWEST scipy distance metric becomes near-free). grep `.powf(` over possibly-integer
exponents. SIGNAL crate done this session (lfilter/sosfilt/savgol); spatial euclidean/cosine/cityblock/
canberra/chebyshev already SIMD — minkowski was the last scalar-powf hole.

## 2026-06-29 (AmberKestrel, cc) — spatial.minkowski_distance integer-exponent fast path (follow-on)

Follow-on to the cdist/pdist minkowski win (74ec55a9): the batched row-wise `minkowski_distance` /
`minkowski_distance_p` (scipy.spatial API) had its OWN inline per-element `.powf(p)` loop (separate from
the per-pair `minkowski`). Refactored the integer-power SIMD kernel out of `minkowski_int` into
`minkowski_pow_sum(a,b,p:u32)` (8-wide std::simd `Σ|Δ|^p` by repeated multiplication) and routed the
`minkowski_rowwise` else-branch through it for integer p∈[2,64]; non-integer p keeps scalar powf.

MEASURED (minkowski_distance n=400k d=8): p=3 **8.52ms vs scipy 100.66ms = 11.8x FASTER** (~5x self vs the
p=3.5 powf path at 44ms); p=2 8.30ms vs scipy 36.21ms = **4.4x**; p=4 8.90ms. correctness diff 0.0 vs scalar
powf reference (row 0). Within tolerance, 225/225 spatial lib GREEN incl. minkowski_distance_batched_matches_scipy.
Both minkowski_distance and minkowski_distance_p share `minkowski_rowwise` ⇒ both inherit. The `powf(integer)
→ repeated-mult + SIMD` lever now covers every minkowski surface (cdist/pdist per-pair + batched rowwise).
Left serial (row loop) — the kernel alone removes the powf bottleneck; row-parallel is a future follow-on.

## 2026-06-29 (AmberKestrel, cc) — interpolate.RbfInterpolator build: route dense solve to optimized blocked-LU

RbfInterpolator::new solved the n×n RBF system Φw=values via a LOCAL naive serial Gaussian elimination
(`solve_dense_system_flat`). Profiled n=2000 d=3: matrix fill 46ms, **naive solve ~1147ms** = 96% of the
1193ms build. fsci-interpolate already depends on fsci-linalg, whose `solve()` has a multithreaded
blocked-LU fast path (n≥1000). Routed the build's solve through it (flat phi → Vec<Vec> rows →
`fsci_linalg::solve`, default Strict/General options).

MEASURED (n=2000 d=3): build 1193→842ms = **1.42x self** (linalg solve 800ms vs naive 1147ms); closes the
build gap vs scipy 131ms from 9.1x→6.4x. NOT byte-identical (blocked-LU pivoting/blocking vs naive GE,
~1e-12) but within RBF tolerance — interpolate lib 178/178 GREEN incl. all rbf_* tests. HONEST FRAMING:
RBF is NOT a real end-to-end gap — eval_many is already 57x FASTER than scipy (16.4ms vs 950ms @ nq=20000),
so fsci wins the full build+eval workflow (856ms vs scipy 1081ms = 1.26x) even before this. The residual
6.4x build gap is the fsci-linalg DENSE SOLVE wall (~3.3 GFLOP/s vs LAPACK ~40 — n=2000 solve is 800ms),
a separate-crate multi-session target, not cheaply closable here. This ship = reuse the optimized solver
instead of a naive one (good hygiene + 1.42x), not a wall-break. LEVER: grep for LOCAL naive dense solvers
(`solve_dense_system*`, hand-rolled GE) in non-linalg crates → route to fsci_linalg::solve's blocked path.

## 2026-06-29 (AmberKestrel, cc) — ndimage.spline_filter1d: route Reflect to the fast IIR prefilter (17.6x→1.86x flip)

`spline_filter1d` computed each axis-line's spline coefficients via `spline_coefficients_for_line`, which
called `make_interp_spline(0..n, line, order)` — building and solving a FULL n-point interpolation system
PER LINE. For a single long 1-D array that is pathological: n=4M order3 = **865ms = 17.6x SLOWER than
scipy 49ms** (order5 1123ms = 17.8x). The N-D `spline_filter`/`prefilter_spline_coefficients` already use
the fast exact O(n) recursive IIR (`bspline_reflect_coefficients`, Unser/Thévenaz, scipy-conformant) — but
`spline_filter1d` was on the slow `make_interp_spline` arm. Routed Reflect mode (order 2..=5, axis>order)
through `bspline_reflect_coefficients`; Nearest/short-axis keep the general path.

MEASURED (1-D n=4M): order3 865→**26.4ms = 32.8x self / 1.86x FASTER than scipy** (was 17.6x slower);
order5 1123→40.8ms = 27.5x self / **1.55x faster**. Multi-line spline_filter1d (2D/3D along one axis)
inherits the per-line speedup. SCIPY-VERIFIED (mode='reflect'): order2/3 EXACT, order4/5 ~1e-7 (within
tolerance); 255/255 ndimage lib GREEN. LEVER (recurring): a function computing spline/IIR prefilter
coefficients by SOLVING a full linear system per line → route to the recursive IIR prefilter the
sibling/N-D path already uses. grep `make_interp_spline` / dense-solve calls inside per-line coefficient
loops. (Same family as the lfilter/sosfilt scans: the fast way to "solve" a cardinal-spline interpolation
system IS the recursive IIR, not a banded/dense solve.)

## 2026-06-29 (AmberKestrel, cc) — interpolate.RectBivariateSpline build: chunked-parallel tensor product (5x flip)

RectBivariateSpline::new built the tensor-product spline by fitting 1-D `make_interp_spline` along each
row (ny) then each column (nx) — SERIALLY. Each row/col spline is INDEPENDENT and the per-call cost is
mostly fixed overhead (knot build + banded solve setup), so 1600 serial calls @ ~92µs dominated: 800×800
= **147ms = 5.2x SLOWER than scipy 28ms** (400² 4.6x). scipy's RectBivariateSpline is single-threaded
FITPACK ⇒ parallelism is pure domination. Fanned BOTH passes across cores via a new chunked
`par_chunk_try_map` (one thread::scope spawn per CHUNK, not per row — a prior per-column per-spawn attempt
was reverted for over-spawn, see line ~2658). Column pass assembles via transpose to avoid a cross-thread
write race on the row-major output.

MEASURED: 400² 35.1→**6.8ms = 5.2x self / 1.13x FASTER than scipy** (was 4.6x slower); 800²
147→**16.0ms = 9.2x self / 1.77x faster** (was 5.2x slower); win grows with grid. BYTE-IDENTICAL
(order-preserved chunked map ⇒ identical coefficients to serial), 178/178 interpolate lib GREEN incl. all
rect/smooth bivariate tests. LEVER: a tensor-product / per-line build looping an independent banded/spline
solve per row & column → chunked-parallel both passes (transpose-assemble the second pass). Reusable
`par_chunk_try_map` (fallible, order-preserved, chunk-spawn) added for Vec-valued parallel maps.

## 2026-06-29 (AmberKestrel, cc) — NEGATIVE EVIDENCE: interpolate builders / sparse / CloughTocher all DOMINATED

After flipping spline_filter1d (17.6x→1.86x) and RectBivariateSpline (5.2x→1.77x) via the "measure the
public builder directly" lever, swept the adjacent surface for more slow-arm/serial-builder gaps — measured
same-box vs scipy 1.17.1, all WIN or parity (do NOT re-chase):
- interpolate CubicSpline build n=500k: fsci 11.3ms vs scipy 39.6ms = **3.5x faster**
- interpolate Akima1D build n=500k: 10.8ms vs 41.7ms = **3.9x faster**
- interpolate PchipInterpolator build n=500k: 20.2ms vs 34.7ms = **1.7x faster**
- interpolate RegularGridInterpolator eval nq=200k (80³): 6.0ms vs 67.4ms = **11.2x faster**
- interpolate CloughTocher2D eval nq=50k: 7.5ms vs 44.8-96ms = **5.9-12.8x faster**; build ~parity
  (npt=2000 1.05x faster; npt=8000 1.17x slower — marginal, cost is the global gradient solve =
  known-hard backlog, NOT a cheap parallelization)
- interpolate SmoothBivariateSpline: FITPACK surfit (adaptive knots = sequential), scipy only 2.34ms@n=5000 — not a target
- sparse spsolve tridiag N=200k: 11.2ms vs scipy 85.4ms = **7.6x faster** (already special-cases banded)
- sparse spmm (CSR@CSR): already parallel Gustavson (fanned across rows, gated by work) — not a gap
CONCLUSION: the interpolate-1D-builder / sparse-core / CT-eval surfaces are DOMINATED. The "serial builder
over independent units" + "wrapper on naive arm" levers are now mined out across interpolate/ndimage/sparse;
the 2 genuine flips this session (spline_filter1d, RectBivariateSpline) were the payoff. Remaining gaps =
the known WALLS (fsci-linalg dense solve ~10x vs LAPACK, FFT non-pow2 SIMD, Qhull/HiGHS) + hot-crate
collision zones (stats/integrate axis-2d, other agents). Next dig should target a WALL or an unmeasured crate
(special/fft batched), not these.

---
## 2026-06-29 (AmberKestrel, cc) — DCT-IV / DST-IV Type-IV core: 2N-FFT → split into 2 parallel N-FFTs

DIG into fsci-fft (untouched 16h, low collision). Measured fsci vs scipy 1.17 dct/dst at n=2^20 and
n=1,000,000 (both single-threaded scipy pocketfft). Found the Type-IV transforms were the biggest gap:
  dst-IV n=2^20 79.4ms vs scipy 17.5 (4.8x slow);  n=1M 58.5ms vs 7.7 (7.5x slow)
  dst-III ~3x; dst-II/dct-II ~1.3-2.7x (pure FFT SIMD wall, documented, not chased)
  dst-I: fsci ALREADY 1.1-2.4x FASTER than scipy (scipy dst-I 306-349ms; not a gap)

ROOT CAUSE: `dct4_core_fft` (shared by dct_iv + dst_iv) ran ONE 2N-point COMPLEX FFT of a zero-padded
length-N complex sequence u (u[n]=x[n]e^{-iπn/2N}); only the first N bins are used. The 2N complex FFT
both doubles the length and thrashes cache (2x16MB at n=2^20). Measured building block: fft 2N complex
62.5ms vs fft N complex 18.7ms — superlinear cache blowup.

LEVER (exact Cooley-Tukey decimation-in-frequency, NOT byte-identical, ~1e-14): split the 2N transform by
OUTPUT PARITY into two independent N-point FFTs —
  U[2m]   = FFT_N(u)[m]
  U[2m+1] = FFT_N(u')[m],   u'[n] = u[n]·e^{-iπn/N}
verified vs scipy to 2.3e-10 abs (~1e-13 rel) across n=1..1024. The two N-FFTs run CONCURRENTLY on 2
threads above gate N>=1<<16 (scipy is single-threaded here so the 2nd core is free). A/B serial-vs-parallel
confirms parallelism is the bulk of the win (pow2 case is memory-bound; cache benefit of the split alone
is small at pow2, large for the 5-smooth 1M). New cached split-twiddle table e^{-iπk/N}.

RESULT (ratio vs ORIG fsci):
  dst-IV / dct-IV  n=2^20  79.4 -> 46.5ms = 1.71x self
                   n=1M    58.5 -> 31.7ms = 1.85x self
Both Type-IV transforms lifted (shared core). 236/236 fsci-fft tests GREEN (metamorphic self-inverse +
all). Still 2.7-4x vs scipy at these n (the per-FFT-point pocketfft SIMD wall, documented hard wall) — but
a clean algorithmic self-win flipping fsci's own 2N-complex baseline. Commit: this one.

GENERALIZABLE: any transform computing the first N bins of a 2N-point FFT of a zero-padded length-N input
(DCT/DST type IV cores, some chirp/Bluestein setups) → decimate into 2 N-point FFTs by output parity and
run them in parallel. The 2N->2*N split is ALWAYS >= as cheap (less work + cache-resident) and the pair is
embarrassingly parallel against scipy's single-threaded core.

### 2026-06-30 follow-on (AmberKestrel, cc) — idct twiddle cache (byte-identical), lifts dct-III/dst-III
The deferred follow-on from the Type-IV dig: `idct` (even-N path) recomputed its N/2 twiddle table
`(cos(πk/2N), sin(πk/2N))` with inline cos/sin on EVERY call — ~6-8ms of stray transcendentals at N=2^20.
That table = conj of the already-cached DCT-II twiddle (`get_or_compute_dct2_twiddles`); cos even + sin odd
make conj(stored (cos(-θ),sin(-θ))) == (cosθ,sinθ) BIT-IDENTICALLY (verified to_bits across 5.6e5 k/N, 0
mismatches). Reused the cache via `complex_conj`. BYTE-IDENTICAL output → zero conformance risk, 236/236
green. Lifts idct AND its dct_iii/dst_iii callers:
  idct     n=2^20 20.6 -> 15.9ms = 1.30x;  n=1M 21.0 -> 16.6ms = 1.27x
  dst-III  n=2^20 24.3 -> 17.3ms = 1.41x;  n=1M 22.5 -> 17.3ms = 1.30x
  dct-III  n=2^20 -> 15.8ms;               n=1M -> 16.3ms
dst-III now 2.16x vs scipy (was 3.0x). LEVER (generalizable): grep inline `angle.cos()/angle.sin()` in
per-element transform hot loops where a sibling already caches the same (or conj/neg) table — reuse it.

### 2026-06-30 (AmberKestrel, cc) — N-D dctn alloc-free fiber kernel: TESTED, ~0-gain, REVERTED
Swept rest of fsci-fft post Type-IV: hfft/ihfft parity-or-faster, fht 1.1-1.23x, dst-I fsci faster. Only gap
= dctn/dstn 2D (1.58-1.74x vs scipy serial). apply_dct_along_axis already 64-thread parallel but no better
than serial floor → hypothesized per-fiber alloc contention. Built alloc-free `dct_ii_gather_into` +
per-worker reused DctIIScratch + fiber-contiguous output (bit-identical). Clean A/B = ~0-gain (14-15 vs
14-16ms @1024^2; OLD faster @2048^2). REVERTED. Real wall = fsci 1-D dct 2x scipy per-call (FFT-SIMD wall) +
strided-axis bandwidth; needs cache-blocked transpose + faster FFT kernel, not an N-D lever. See
docs/NEGATIVE_EVIDENCE.md.

### 2026-07-01 (AmberKestrel, cc) — mmread_sparse: sparse-COO MatrixMarket read, ~10x faster than dense mmread (scipy parity)
DIG into fsci-io (13d stale, low collision). Measured all readers vs numpy/scipy on large files: loadtxt 14x
FASTER, read_csv 12x FASTER, mmread(dense) 1.46x FASTER — all wins. ONE gap: **mmread on a SPARSE (coordinate)
file was 9.8x SLOWER than scipy** (136ms vs scipy 14ms, 4000^2 @1% ~160k nnz). ROOT CAUSE (profiled): mmread
materializes a DENSE rows*cols buffer (128MB for this file) even for coordinate format; the ~120ms is almost
entirely first-touch page faults across that 128MB of mostly-zeros (parse-to-COO alone = 13ms = scipy
parity; the dense buffer is intrinsic to mmread's dense return type — can't be sped up in place). scipy
returns sparse COO in 14ms. LEVER: added `mmread_sparse` (+ `MmSparse` COO struct) = parse coordinate format
to (row_indices, col_indices, values) triplets, NO dense materialization; symmetric/skew/hermitian expand the
stored triangle to both off-diagonals (negate for skew), so scattering the triplets with `+=` reproduces
mmread's dense `data` BIT-FOR-BIT (verified: byte_mismatches=0 across general/symmetric/skew/duplicate/pattern
+ permanent test mmread_sparse_matches_dense_mmread). RESULT: **15.1ms vs mmread dense 123.6ms = 8.2x self;
9.8x scipy-loss -> ~PARITY (scipy 14ms)**. Purely additive (206 insertions, 0 deletions), matches scipy.io
sparse-return behavior for the format's primary (sparse) use case. fsci-io mmread family 20/20 green (my new
test passes). NOTE: pre-existing UNRELATED red `mmwrite_complex_output_format` fails on HEAD too (test asserts
0-based coordinate indices; the emitter correctly produces 1-based per MM spec) — another agent's test bug,
left untouched per own-files.

### 2026-07-01 (AmberKestrel, cc) — loadmat_v5 fused decode+transpose: 8.7x self, flips 8.2x scipy LOSS to WIN
Continued fsci-io sweep. Measured writers/readers vs scipy/numpy: savemat 1.55x FASTER, savetxt 3.2x FASTER,
mmwrite(dense) 5.6x SLOWER (float-format wall, deferred), **loadmat 8.2x SLOWER** (25.3ms vs scipy 3.1ms,
300000x8 v5 .mat). ROOT CAUSE (profiled): loadmat_v5 decoded the column-major disk payload into an
intermediate `column_major` Vec (`chunks_exact(8).map(from_le_bytes).collect()`) THEN did a SEPARATE strided
transpose into row-major `data` — one extra full 19MB alloc + two extra passes over R*C*8 bytes. A/B:
decode.collect+transpose 25.8ms vs FUSED single pass 6.4ms. LEVER: `decode_v5_numeric_rowmajor` fuses the
byte-decode and the column->row transpose in one pass (c outer = sequential disk read, strided row-major
write), dropping the intermediate buffer; handles all MI_ numeric types; error messages preserved
(loadmat_rejects_wrong_element_count green). RESULT: loadmat **25.3 -> 2.90ms = 8.7x self; flips 8.2x scipy
LOSS -> 1.07x WIN** (beats scipy's 3.10ms). Byte-identical (values == scipy exactly; 116/116 pre-existing
loadmat/mat tests green — sole red is the unrelated pre-existing mmwrite_complex 0-vs-1-based test). Net
-13 lines. LEVER (generalizable): binary readers that decode-to-Vec THEN transpose/reorder → fuse into one
pass writing the final layout directly; the intermediate buffer is pure alloc + memory-traffic overhead.
mmwrite float-format gap (5.6x) left as deferred (needs fast f64 formatter/dep — byte-id risk).

### 2026-07-01 (AmberKestrel, cc) — mmwrite parallel formatting: 8.7x self, flips 5.6x scipy LOSS to 1.56x WIN
The last fsci-io gap: mmwrite(dense) 5.6x SLOWER than scipy (276ms vs 49.8ms, 1500x1500). PROFILED: the f64
Display FORMATTING is the wall (~175ms/2.25M values = 78ns each) — NOT allocation (with_capacity gave 0-gain)
and NOT writeln! overhead (reused-buffer write! 0-gain); the strided column-major read adds ~100ms. No
byte-identical single-thread win without a float-formatter dep. BUT scipy's mmwrite is SINGLE-THREADED and
formatting is embarrassingly parallel. LEVER: each worker formats a contiguous slice of the column-major value
stream (value k → col k/rows, row k%rows → data[row*cols+col]) into a private String; concat parts in order =
BIT-FOR-BIT the serial output (verified all nthreads==serial + round-trip test). Serial gate n<1<<16 BEFORE
the available_parallelism syscall (per-call-syscall-tax lesson); nthreads = avail.min(n/16384). RESULT:
mmwrite **276 -> 31.8ms = 8.7x self; flips 5.6x scipy LOSS -> 1.56x WIN** (beats scipy 49.8ms). Byte-identical
(117/117 mmwrite/mm tests green incl. new mmwrite_parallel_path_matches_serial_and_roundtrips; sole red is
the unrelated pre-existing mmwrite_complex 0-vs-1-based test). fsci-io vein now: mmread_sparse (10x),
loadmat (8.7x), mmwrite (8.7x) all shipped; loadtxt/read_csv/savemat/savetxt already faster. LEVER
(generalizable): when the wall is a serial std formatter/parser and the peer lib is single-threaded, PARALLEL
format-into-private-buffers + ordered concat is byte-identical and wins.

### 2026-07-01 (AmberKestrel, cc) — wav_read parallel per-sample decode: 7.3x self (byte-identical)
Swept remaining fsci-io readers: read_arff 3.1x FASTER than scipy (1064ms vs 339ms, no gap); wav_read 22ms.
The wav scipy comparison is APPLES-TO-ORANGES (scipy.io.wavfile.read returns raw int16 zero-copy in 0.33ms;
fsci returns normalized f64, doing real conversion work scipy skips) — NOT a scipy gap, but a self-improvement:
the per-sample i16→f64 decode was a serial `chunks_exact.map.collect` that does NOT auto-vectorize (measured:
the chunks16+[u8;16] idiom gave ~0-gain, reverted) and is compute-bound (~4ns/sample). LEVER: parallelize the
per-sample decode across threads (generic `decode_wav_samples(bytes, stride, conv)` covering 8/16/24/32-bit,
serial gate n<1<<18 before the available_parallelism syscall, byte-identical — each worker runs the same conv
on a disjoint sample range). RESULT: wav_read **22.09 -> 3.02ms = 7.3x self**, byte-identical (new test
wav_read_parallel_decode_matches_serial + 118/118 wav/io tests green; sole red = unrelated pre-existing
mmwrite_complex). NOTE honestly: still ~9x scipy's 0.33ms because fsci produces f64 not raw int16 — a semantic
difference, not a perf bug. fsci-io scipy-comparable surface now fully swept/dominated.

### 2026-07-01 (AmberKestrel, cc) — KDTree knn flat point-slab: d=8 query 1.46x self, closes 1.54x scipy-parallel loss to parity
PIVOT off io (fully swept) to fsci-spatial KDTree. Measured vs scipy.spatial.cKDTree: fsci build 1.46-1.9x
FASTER; d=3 query_k_many 8.1ms BEATS scipy workers=-1 (11.5ms); BUT **d=8 query_k_many 270.6ms = 1.54x SLOWER
than scipy workers=-1 (175.9ms)** (36x faster than scipy SERIAL — scipy's single-thread d=8 query is 9.9s).
query_k_many already parallel+tuned, so the residual is the per-query knn_search KERNEL. ROOT CAUSE: `KDNode`
stores `point: Vec<f64>` — a SEPARATE heap alloc per node (200k scattered Vecs); knn_search's
`sqeuclidean(query, &node.point)` chases a pointer to scattered memory (cache miss per visited node, and d=8
backtracking visits MANY). sqeuclidean is already std::simd, so the wall is the scattered coord reads. The
BUILD already flattened coords (partition), but build_kdtree cloned each point back into a per-node Vec.
LEVER (flat-buffer, [[perf_equal_hardware_artifact_and_flatbuffer_lever]]): add a node-ordered contiguous
slab `points: Vec<f64>` (node i at points[i*dim..]) and route knn_search coord reads through it instead of
node.point. RESULT: d=8 query_k_many **270.6 -> 184.9ms = 1.46x self; flips 1.54x scipy-parallel loss to 1.05x
(PARITY)**; d=3 unchanged (already cache-friendly); build +1ms (the slab copy, negligible). Byte-identical
(slab is a node-order copy of node.point; 225/225 spatial tests green incl. kdtree_query_matches_scipy_
reference_values). Minimal 25/-7 diff (kept node.point for the non-hot ball/nn/pairs paths). FOLLOW-ON: remove
node.point entirely + route ball_search/nn_search/query_ball_* through the slab (would shrink KDNode 56->32B →
more nodes/cache-line, help ball queries too).

### 2026-07-01 (AmberKestrel, cc) — KDTree single-NN (query_many/nn_search): 2.2x self, closes d=8 3.9x scipy loss to 1.78x
Follow-on to the knn flat-slab (822eabd7). Measured fsci KDTree ball/nn/pairs vs scipy cKDTree workers=-1:
ball_point_many 2.4-12x FASTER, query_pairs faster, BUT **d=8 query_many (single-NN k=1) 103.3ms = 3.9x SLOWER
than scipy workers=-1 (26.4ms)**. TWO causes (both the levers already applied to query_k_many but NOT
query_many): (1) thread ceiling hard-capped at `cores.min(16)` — query_k_many's comment already noted the
16-cap left d=8 ~3.5x slow and lifted it for work-heavy batches, but query_many never got the fix; (2)
nn_search still read scattered `node.point`. FIX: thread_ceiling = if dim>=6 { cores } else { cores.min(16) }
(single-NN backtracking is compute-heavy at higher dim → use all cores; low-dim stays capped to avoid
over-spawn) + route nn_search through the node-ordered flat point slab. RESULT: d=8 query_many **103.3 ->
46.9ms = 2.2x self; closes 3.9x scipy-parallel loss to 1.78x**; d=3 2.3->1.8ms (flat-slab, still 4.8x faster
than scipy). Byte-identical (same coords, independent deterministic per-query; 225/225 spatial green incl.
kdtree_query_many_matches_per_query + nearest_neighbors_kdtree_matches_brute_force_bitwise). Also lifts single
`query` (uses nn_search). RESIDUAL 1.78x = nn_search kernel + median-split vs scipy's sliding-midpoint tree
(deeper rewrite, deferred). LEVER (reusable): when a batched-query method wins but its SINGLE/other-k sibling
loses, diff their thread-gate + kernel — the fix often already exists on the fast sibling.

### 2026-07-01 (AmberKestrel, cc) — BSpline.eval_many parallel de Boor: 11.5x self, 10.4x FASTER than scipy splev
PIVOT (no collision anywhere on frankenscipy — last 8h commits all mine). Measured fsci-interpolate spline eval
on nq=2M sorted queries (n=2000 knots): CubicSplineStandalone.eval_many 9.1ms = 1.68x FASTER than scipy
CubicSpline (15.3ms) — win. BUT **BSpline.eval_many 59.9ms was ~PARITY with scipy splev (54.3ms)** and it was
SERIAL. scipy's splev is single-threaded and per-point de Boor is independent, so PARALLELIZE. For sorted
input each worker re-seeds its knot-span pointer mu by advancing from k to its chunk start (O(#knots) ≪
#queries) then merge-advances within the chunk — the span reached for any x depends only on x+knots, so
BIT-IDENTICAL to the single serial pointer walk (verified byte_mismatches=0 sorted AND unsorted, 2M pts; new
test bspline_eval_many_parallel_matches_per_point). Serial gate n<1<<15 before the available_parallelism
syscall. RESULT: BSpline.eval_many **59.9 -> 5.2ms = 11.5x self; flips ~parity to 10.4x FASTER than scipy
splev** (14.7x vs scipy BSpline). Byte-identical, 179/179 interpolate tests green. Lifts all
BSpline.eval_many callers. LEVER (proven again): a serial per-element eval where the scipy peer is
single-threaded → chunk-parallel with per-chunk pointer re-seed (byte-id for sorted-monotone state).

### 2026-07-01 (AmberKestrel, cc) — rankdata(ordinal) stable→unstable sort: 1.36x self (byte-identical anomaly fix)
Swept fsci-stats sort/selection fns vs scipy (N=2M): rankdata(avg) 5.3x FASTER, wasserstein 6.6x, energy 4.2x
— all win. SELF-ANOMALY: rankdata(ordinal) 114.6ms was 1.9x SLOWER than rankdata(average) 61.0ms — ordinal
(argsort+assign 1..n) should be SIMPLER than average (tie-grouping). CAUSE: rankdata_ordinal used STABLE
`sort_by` while the tie methods use faster `sort_unstable_by`. The ordinal comparator (value, then original
index) is a STRICT TOTAL ORDER (unique indices → no equal elements), so an unstable sort yields the IDENTICAL
permutation as stable (nothing for stability to disambiguate). FIX: sort_by → sort_unstable_by. RESULT:
ordinal **114.6 -> 84.2ms = 1.36x self** (now 3.2x vs scipy 269ms, was 2.3x). BYTE-IDENTICAL (byte_mism=0 vs
stable-sort reference INCLUDING tied data; 9/9 rankdata tests green incl. rankdata_ordinal_matches_scipy_
reference + rankdata_with_ties). Applied the same provably-safe transform to multiscale_graphcorr's per-row
distance ranking (34173, comment-confirmed total order). Did NOT touch the 2 correlation sorts that tiebreak
by y-VALUES (not a guaranteed total order → stability matters). 2001/2001 stats green. LEVER: grep
`.sort_by(...total_cmp...then...index)` — a unique-index tiebreak makes it a total order → sort_unstable_by is
a free byte-identical speedup.

### 2026-07-01 (AmberKestrel, cc) — SURFACE: fsci-sparse competitive; spmm symbolic-pass removal = ~4.5% (reverted)
No clean sparse gap: matvec ≈parity (bandwidth), spmm 1.3x FASTER than scipy (already parallel). The spmm
parallel path's redundant symbolic-counts pass LOOKED like a 2x lever but same-binary atomic A/B = only ~4.5%
(symbolic pass buys exact-alloc + cache-warm, not pure waste) → reverted (near-zero + dead-code churn). Noise
lesson: cross-worker spmm = 84-107ms for IDENTICAL code; same-binary atomic toggle MANDATORY for <20% sparse
claims. See docs/NEGATIVE_EVIDENCE.md.

### 2026-07-01 (AmberKestrel, cc) — ndimage van Herk min/max: total_cmp → f64::max/min for clean data (1.6-2x, flips loss to win)
Measured fsci-ndimage vs scipy (2000², size 5): sobel/laplace/gaussian_gradient_magnitude WIN 1.4-2.1x, BUT
**grey_dilation 108.6ms = 1.49x SLOWER, grey_erosion 128.7ms = 1.81x SLOWER** than scipy (72.7/71.0). The van
Herk (Gil-Werman) HGW kernel is already the default (MINMAX_FILTER_HGW=true), so the wall is its HOT OP:
`tc_max`/`tc_min` use `f64::total_cmp` (~6 integer ops, bit-flip + i64 cmp) purely for scipy total-order
tie-breaks — called ~24M×/filter. `f64::max`/`f64::min` are FASTER and byte-identical to the total order
EXCEPT in exactly two spots: NaN (total_cmp propagates, f64::max drops) and the {+0.0,-0.0} pair
(f64::max(+0,-0)==-0 but total order gives +0). LEVER: probe the input ONCE for NaN-or-(-0.0); clean data
(the common case) runs f64::max/min, else the safe tc_max/tc_min. min/max of clean values can't MINT a NaN or
-0.0, so cleanliness holds through every separable axis pass. RESULT: grey_dilation **108.6->67.0ms = 1.62x
self, flips 1.49x loss to 1.09x WIN**; grey_erosion 128.7->76.3ms = 1.69x self (1.81x loss → parity);
morphological_gradient 193.7->98.4ms = 1.97x self, now **1.94x FASTER** than scipy. BYTE-IDENTICAL: 0
mismatches vs the deque/total_cmp reference across 5 boundary modes × 4 sizes × dil+ero, for BOTH ±0-injected
finite AND NaN-injected data; 255/255 ndimage tests green. Lifts the whole family (min/max filter, grey_open/
close, morphological_gradient, tophat). LEVER (reusable): grep hot-loop `total_cmp`-based max/min → gate on
NaN/-0.0 and use f64::max/min for clean data (byte-identical, ~2x on the op).

### 2026-07-01 (AmberKestrel, cc) — SURFACE: ndimage cdt(chessboard) offset-split ~1.10x, doesn't beat scipy (reverted)
median/percentile filters 20-35x FASTER (wins). Gaps: binary_erosion 1.35x, distance_transform_cdt(chessboard)
1.51x. cdt interior chamfer iterated all 8 offsets w/ sign branch (used 4); pre-split fwd/bwd → byte-identical
~1.10x (same-binary A/B) but STILL 1.33x vs scipy (offset iter not dominant; strided neighbour reads + raster
stepping = C-chamfer/memory wall). Reverted (near-zero + no scipy beat). See docs/NEGATIVE_EVIDENCE.md.

### 2026-07-01 (AmberKestrel, cc) — eigvals_banded: wrong-tool Lanczos → dense values-only, 4.3x self (fixes pathological anomaly)
Broad scipy sweep found the biggest gap vs ORIG: **eigvals_banded(1500,bw3) = 3097ms, 90x SLOWER than scipy
(34.3ms) AND 3.5x SLOWER than fsci's own eig_banded (896ms, values+VECTORS)** — an absurd inversion.
ROOT CAUSE: eig_banded's eigvals_only path called symmetric_lower_band_lanczos_eigenvalues — LANCZOS to compute
ALL n eigenvalues (Lanczos targets k≪n; all-n drives full reorthogonalization → pathologically slow + less
accurate; wrong tool). FIX: route eigvals_only through the SAME dense reduction the eigenvector path uses,
minus eigenvector accumulation (nalgebra symmetric_eigenvalues, ascending total_cmp = scipy convention).
RESULT: **3097 → 713.6ms = 4.3x self**; anomaly fixed (713 < eig_banded 874, values-only now correctly
cheaper). Conformance GREEN (10/10 eig_banded/eigvals_banded tests incl. eigvals_banded_matches_scipy_
pentadiagonal + eig_banded_lanczos_values_match_dense_reference; matches dense/scipy to tolerance). Dead
Lanczos cluster (8 fns) marked #[allow(dead_code)] (still exercised by the values-match test). STILL 20x vs
scipy (dense O(n³) vs banded O(n²·bw)) — the scipy-PARITY follow-on is a real band→tridiagonal reduction
(dsbtrd), a dedicated numerical task. This ships the bug fix + 4.3x; dsbtrd flagged for a future cycle.

### 2026-07-01 (AmberKestrel, cc) — kv half-integer closed-form: 37x self, flips 1.94x scipy LOSS to 19x WIN
Broad scipy.special sweep (2M pts): fsci jv 13.9x/iv 15.3x/gammainc 9x FASTER than scipy, BUT **kv(1.5) 194.3ms
= 1.94x SLOWER** than scipy (100.3) — and ~8x slower than fsci's OWN jv/iv (21-24ms). ROOT CAUSE: kv_scaled_
value routes NON-INTEGER order to kv_integral_scaled = a ~96-point Gauss-48 quadrature (×1-2, split at t*),
~hundreds of cosh/exp per call (the slow-quadrature-kernel pattern, cf. [[perf_special_quadrature_to_continued_
fraction]]). LEVER: half-integer order (v=n+1/2, the spherical-Bessel family) has a CLOSED FORM — K_{1/2}·e^z =
sqrt(π/(2z)), upward recurrence K_{ν+1}=K_{ν-1}+(2ν/z)K_ν (identical for the e^z-scaled values). Added a
half-integer fast path (v_abs.fract()==0.5) before the integral. RESULT: **kv(1.5) 194.3 → 5.2ms = 37x self;
flips 1.94x LOSS to 19x FASTER than scipy**. ANALYTICALLY EXACT — max rel err vs scipy 3.9e-16 across
v=0.5..5.5 × z=0.3..50 (more accurate than the quadrature it replaces). 1121/1121 special tests green. Lifts
all half-integer kv AND kve (shares kv_scaled_value). General non-integer v still uses the quadrature (a K_v
continued-fraction is the follow-on, cf. the wofz CF lever). LEVER: half-integer Bessel (jv/yv/iv/kv/spherical)
→ closed-form recurrence beats general quadrature; grep special kernels routing to fixed-step quadrature.

### 2026-07-01 (AmberKestrel, cc) — struve_many / modstruve_many: vectorized Struve gapfill, 1206x faster than scipy
Broad scipy.special sweep confirmed fsci DOMINATES everywhere (yv 8.5-91x, hyp 4.6-6.3x, gammaincinv 18x,
betaincinv 15x, lambertw 39x, airy 6.5x, ellip 14-19x, erfcx/dawsn 6-8x FASTER). ONE gap: **fsci had only
SCALAR struve** (`struve(v,x)->f64`) while scipy.special.struve is a vectorized ufunc — and scipy's is
pathologically slow (**10490ms for 2M points = 5.2µs/point**, per-point series/integral). fsci's struve_scalar
is ~47ns/point and correct (6.9e-10 vs scipy). GAPFILL: added `struve_many(v, x)` + `modstruve_many` — fan the
scalar kernel across cores via the crate's order-preserving par_map_indices (non-breaking; the bare `struve`
scalar + internal callers untouched). RESULT: struve_many(1) 2M **8.7ms vs scipy 10490.8ms = 1206x FASTER**;
byte-identical to serial struve (0 mismatches), matches scipy to 6.9e-10. 1121/1121 special tests green.
LEVER: a scalar-only fsci special fn whose scipy peer is a SLOW vectorized ufunc → add a parallel vectorized
wrapper (par_map over the fast scalar kernel) = huge gapfill win. grep fsci-special for `_scalar`-only fns
with no tensor sibling.

### 2026-07-01 (AmberKestrel, cc) — vectorized Kelvin ber/bei/ker/kei(+primes)_many: ~172x faster than scipy (8-fn gapfill)
Continuing the scalar-only-fn gapfill vein (after struve 1206x). Broad scipy.special speed sweep of fsci's
scalar-only public fns found the KELVIN family the slowest scipy peers: **ber/bei/ker/kei ~1272-1419ms/2M
(~640-710 ns/pt)**. fsci had them SCALAR-ONLY, correct (max rel err 7.2e-11 vs scipy across all 8) and fast
(ber ~33ns/pt, ~19x faster than scipy even SERIAL). GAPFILL: added ber_many/bei_many/ker_many/kei_many +
berp/beip/kerp/keip_many — par_map_indices over the scalar kernels (non-breaking; scalars + re-exports
untouched). RESULT: ber_many(2M) **7.4ms vs scipy 1272ms = 172x FASTER** (on a load-47 box; true ratio higher).
Byte-identical to serial (0 mismatches across all 8), 1121/1121 special tests green. LEVER (same as struve):
grep fsci-special `_scalar`-only fns whose scipy peer is a slow ufunc → add parallel `*_many` wrapper. Remaining
slow-peer scalar-only candidates for follow-on: itj0y0/iti0k0 (~580-601ms), expn (440ms), shichi (288ms),
fresnel (165ms), sici/poch (~90ms) — all vectorizable the same way.

### 2026-07-11 (ScarletChapel, cc) — labeled_comprehension parallel-across-groups: 1.55-1.96x, byte-identical
Re-swept the lane after a context reset; almost everything is harvested (griddata already grid-accelerated,
welch/cwt/csd + all `*_axis_2d` filters parallel, dctn fiber-parallel, special `_many` vein done, io shipped).
The one clean-file un-parallelized reduction left was `ndimage::labeled_comprehension`: it had NO `Sync` bound
(unlike its already-parallel sibling `generic_filter`) and mapped the per-group reducer SERIALLY. The N-D
`median`/`generic_filter1d` parallelizations landed the same day but this generic sibling was missed. LEVER:
add `+ Sync` to the reducer closure and fan the independent per-group `func` calls across cores (chunked
`thread::scope`, work-gated by `ndimage_filter_thread_count(total, 8).min(groups)`), toggled by
`NDIMAGE_LABELED_COMPREHENSION_FORCE_SERIAL` for the same-binary A/B. BYTE-IDENTICAL: each group's reducer is
independent → its own output slot, results collected in group order, `func` deterministic. scipy.ndimage.
labeled_comprehension runs a Python callback per label single-threaded, so this compounds Rust's per-call speed
with real parallelism on the realistic per-region-statistic workload. MEASURED (strict-remote release
`+avx2,+fma`, hz2/vmi1227854, paired median vs A/A NULL control; `p90` = per-region 90th percentile over 4M px):
16 labels 135.9→68.2ms = **1.955x** (NULL [0.876,1.127]); 64 labels 127.3→69.6ms = **1.639x** (NULL
[0.853,1.175]); 1024 labels 150.9→96.7ms = **1.546x** (NULL [0.918,1.098]); **bitmism=0** all three, DECIDED.
Full `fsci-ndimage` lib suite 272 passed/0 failed incl. `labeled_comprehension_matches_scipy_fixtures`. The
serial label value/position gather stays serial (O(N)), so the win is bounded by the reducer's fraction —
coarser regions (heavier per-region sort) win more. bin `perf_labeled_comprehension`. LEVER (reusable): grep
ndimage label-reduction fns whose closure lacks `+ Sync` — they map the reducer serially while the O(N) gather
is shared; the fan-out is byte-identical because groups are independent.

### 2026-07-11 (ScarletChapel, cc) — parallel label value/position GATHER: extrema ~1.4-1.7x, byte-identical
Follow-on to the labeled_comprehension map-parallel: the shared `measurement_label_value_positions` gather
(behind `extrema`/`labeled_comprehension` + 2 more) did a per-element std-HashMap (SipHash) lookup over all N
elements SERIALLY — the dominant cost of the gather-based path (~2/3 of `extrema`'s time). LEVER: split the
flat-index range into contiguous chunks across cores, each thread bucketing into PRIVATE per-label buckets,
then merge in thread (= flat) order. BYTE-IDENTICAL: thread t owns a strictly lower flat range than t+1 and
pushes stay flat-ascending within a thread, so the ordered concatenation reproduces the exact serial group
contents/order — even `extrema`'s argmin/argmax tie-breaking. Work-gated + a `t*num_groups <= n` guard against
many-label bucket blowup; toggled by `NDIMAGE_LABEL_GATHER_FORCE_SERIAL`. MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, via `extrema`, 4M px / 64 labels), DECIDED twice: 15 iters
47.51->31.74ms **1.357x** (null [0.829,1.243]); 31 iters 41.90->23.39ms **1.687x** (null [0.665,1.173]);
**bitmism=0** both (all 4 extrema outputs incl. positions). Consistently-positive byte-identical ~1.4-1.7x;
magnitude noisy under heavy box contention, memory-bound by the serial merge-copy so caps ~2x. Full
`fsci-ndimage` lib suite 272 passed / 0 failed. bin `perf_label_gather`. NOTE: the RCH admission for the heavy
test build stalled ~15min behind a stuck build on vmi1227854 (`hard_preflight=1`); landed once a slot freed
(never went local). LEVER (reusable): the sibling value-only gather `measurement_label_groups` (median +5
callers) is the identical next follow-on.

### 2026-07-11 (ScarletChapel, cc) — parallel value-only label GATHER (measurement_label_groups): 1.6-1.7x, byte-identical
Completes the label-gather family (value-only sibling of 7d624045). `measurement_label_groups` — behind
`median` (all cases, no streaming path) + `histogram`/`sum`/`variance`/`minimum`/`maximum` FALLBACK when the
index is not one-based-contiguous — bucketed per-element serially with the same per-element SipHash lookup.
Parallelized identically: private per-thread buckets over contiguous flat-chunks + thread-order merge =
BYTE-IDENTICAL (each group's ascending-flat-order value list reproduced exactly); reuses
`NDIMAGE_LABEL_GATHER_FORCE_SERIAL` + the `t*num_groups<=n` guard. MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, via `sum` with a REVERSED index — bypasses `sum`'s
`measurement_one_based_scatter` streaming path to force the gather, gather-dominated), DECIDED twice: 21 iters
50.58->28.71ms **1.607x** (null [0.792,1.474]); 31 iters 51.63->26.44ms **1.735x** (null [0.706,1.258]);
**bitmism=0** both. Full `fsci-ndimage` lib suite 272/0. bin `perf_label_groups_gather`. SCOPE: lifts `median`
(all cases) + non-contiguous-index label stats; the contiguous-index common case uses the unaffected
`measurement_one_based_scatter`. The label-stat parallelization vein is now SATURATED (map +
`labeled_comprehension`, both shared gathers done); no clean sibling remains in it.

### 2026-07-11 (ScarletChapel, cc) — otsu_threshold parallel min/max + histogram: 5.71x, byte-identical
Broke OUT of the label-stat vein into a compute-bound global reduction. `otsu_threshold` (image
binarization; peer = skimage `threshold_otsu`, single-threaded) ran 3 serial O(N) passes: a min fold, a max
fold, and a 256-bin histogram with a per-pixel divide+floor (the divide is COMPUTE-bound). LEVER: parallelize
all three — chunked NaN-propagating min/max reductions (associative + NaN-propagating → byte-identical) + a
privatized per-thread 256-bin histogram merged by summing counts (order-independent integer counts →
byte-identical). Same per-pixel bin assignment, so `best_thresh` is bit-identical. Gated by
`ndimage_filter_thread_count`; toggled by `NDIMAGE_OTSU_FORCE_SERIAL`. MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, 16M px): 90.06->13.15ms = **5.710x** (null [0.834,1.165]),
**bitmism=0**. cargo check compiles; the serial arm is the ORIG code verbatim + perf-bin bitmism=0 proves the
parallel arm is bit-identical, and the existing `diff_ndimage_otsu_threshold` conformance test validates that
serial path vs Python — so it transitively covers the parallel path. bin `perf_otsu`. LEVER (reusable, BIG):
a GLOBAL histogram/threshold fn with a per-pixel divide is compute-bound → privatized-parallel bincount is a
~5x byte-identical win (vs the memory-bound label gathers' ~1.5x — the per-pixel DIVIDE is what makes it 5x).
NEXT: `histogram`'s global (labels=None) path is the identical pattern (+drops a full-data clone).

### 2026-07-11 (ScarletChapel, cc) — global histogram direct privatized-parallel bincount: 5.78x, byte-identical
Direct follow-on to otsu (same compute-bound bincount pattern). `ndimage::histogram` with `labels=None` cloned
the ENTIRE array into one group via `measurement_label_groups` THEN serial-binned it. LEVER: for the global
case, bin `input.data` DIRECTLY with a privatized per-thread parallel bincount — BYTE-IDENTICAL (one group =
every element; the `< min || > max` range filter + per-value bin assignment are unchanged; integer counts sum
order-independently) AND it drops the full-data clone. Gated by `ndimage_filter_thread_count`; toggled by
`NDIMAGE_HISTOGRAM_FORCE_SERIAL` (serial arm = ORIG clone+group path). MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, 16M px / 256 bins): 95.13->12.49ms = **5.777x** (null [0.753,1.202]),
**bitmism=0**. Full `fsci-ndimage` lib suite 272/0 with the change present. bin `perf_histogram`. CONFIRMS the
otsu lever generalizes: per-pixel-divide bincount → privatized-parallel = ~5.8x byte-identical (the DIVIDE is
what lifts it from the memory-bound ~1.5x to ~5.8x).

### 2026-07-11 (ScarletChapel, cc) — global min/max: drop clone + parallel fold: 7.21x, byte-identical
Clone-removal follow-on (biggest of the session). `minimum`/`maximum` with `labels=None` fell through to
`measurement_label_groups(None)` = a full-data CLONE (`input.data.clone()`, ~128MB at 16M px) THEN a serial
fold. LEVER: a global fast path folds `input.data` DIRECTLY via a chunked parallel NaN-propagating reduction
(new `global_minmax_reduce` helper, shared by both) — BYTE-IDENTICAL (min/max associative + NaN propagates
through the combine) AND drops the clone. Gated by `ndimage_filter_thread_count`; toggled by
`NDIMAGE_MINMAX_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null, 16M px):
77.71->7.31ms = **7.213x** (null [0.890,1.147]), **bitmism=0**. The 128MB clone was the dominant cost, so this
beats even otsu/histogram. Compiles (worker vmi1149989); byte-identical so every existing min/max test passes
unchanged. bin `perf_minmax`. LEVER (reusable, clone-removal): EVERY global label-stat clones the whole array
via `measurement_label_groups(None)` then reduces — for order-independent reductions (min/max/count) a direct
parallel fold drops the clone for a big win; even float-sum reductions can drop the clone with a serial fold.
INFRA NOTE: worker `ovh-b` SIGILLs the num-traits build script (fleet `+avx2` rustflags applied to build
scripts on a non-AVX2 CPU); retried the build until rch routed to a healthy worker.

### 2026-07-11 (ScarletChapel, cc) — global sum: drop clone: 7.21x, byte-identical
Pure clone-removal (no parallelization needed). `sum` with `labels=None` cloned the whole array via
`measurement_label_groups(None)` (~128MB at 16M px) THEN serial-summed it. Fast path sums `input.data`
DIRECTLY — BYTE-IDENTICAL (same increasing-flat-index order; float add is non-associative so it stays a single
serial fold, NOT parallelized) — and drops the clone. Toggled by `NDIMAGE_SUM_FORCE_SERIAL`. MEASURED
(strict-remote release `+avx2,+fma`, paired median vs A/A null, 16M px): 57.88->8.03ms = **7.212x** (null
[0.748,1.171]), **bitmism=0**. The 128MB clone was ~85% of the time — a serial fold with no clone is 7x. bin
`perf_sum`. TAKEAWAY: the `measurement_label_groups(None)` clone is the single biggest ndimage global-label-stat
inefficiency (~7x each). Remaining identical follow-ons: `mean`/`variance`/`standard_deviation` global (same
clone; variance/std keep the serial 2-pass so still byte-identical, just no clone).

### 2026-07-11 (ScarletChapel, cc) — global variance (+std) drop clone: 4.30x, byte-identical
Closes the clone-removal family. `variance` with `labels=None` cloned the whole array via
`measurement_label_groups(None)` THEN ran its serial two-pass (mean, then Σ(x-mean)²). Fast path computes over
`input.data` DIRECTLY — BYTE-IDENTICAL (same `mean_of_values`, same increasing-flat-index Σ order; float
two-pass stays serial) — no clone. Toggled by `NDIMAGE_VARIANCE_FORCE_SERIAL`; also lifts
`standard_deviation` (= `variance(..).sqrt()`). MEASURED (strict-remote release `+avx2,+fma`, paired median vs
A/A null, 16M px): 68.14->16.27ms = **4.299x** (null [0.902,1.119]), **bitmism=0**. Lower than sum's 7.21x
because variance's TWO passes over the data make the clone a smaller fraction. bin `perf_variance`.
CLONE-REMOVAL FAMILY COMPLETE: min/max 7.21x, sum 7.21x, histogram 5.78x, variance/std 4.30x; `mean` was
already clone-free (`measurement_label_mean` has a direct global path). `measurement_label_groups(None)`'s
clone is no longer on any global-stat hot path.

### 2026-07-11 (ScarletChapel, cc) — global extrema fused single pass: 6.66x, byte-identical
The last global-label-stat allocation. `extrema` with `labels=None` allocated ALL `(value, position)` pairs
via `measurement_label_value_positions(None)` (~256MB at 16M px) THEN ran two separate scans
(`minimum_value_position` + `maximum_value_position`). LEVER: a single fused pass over `input.data` tracking
min/max + their positions — BYTE-IDENTICAL (identical strict-`<`/`>` first-occurrence tie-break, identical NaN
"last wins" update `value.is_nan()→update`, identical increasing-flat-index order; min/max are independent
trackers so fusing the two scans is exact) — no pair allocation, and 2 scans → 1. Toggled by
`NDIMAGE_EXTREMA_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null, 16M px):
101.05->15.83ms = **6.662x** (null [0.706,1.145]), **bitmism=0** (all 4 outputs incl. positions). bin
`perf_extrema_global`. ALLOCATION-REMOVAL FAMILY now FULLY CLOSED: the `measurement_label_groups(None)` clone
(min/max/sum/variance/std/histogram) and the `measurement_label_value_positions(None)` pair-alloc (extrema) are
off every global-stat hot path; `mean` was already direct. 9 ndimage wins this session (1.5-7.2x, all byte-id).

### 2026-07-11 (ScarletChapel, cc) — global labeled_comprehension direct reducer: 29.25x, byte-identical
The last global-label-stat allocation. `labeled_comprehension` with `labels=None` + `pass_positions=false`
allocated ALL `(value, position)` pairs via `measurement_label_value_positions(None)` (~256MB at 16M px) AND a
per-group value-extraction copy (~128MB) before calling the reducer. LEVER: hand `input.data` to the reducer
DIRECTLY — BYTE-IDENTICAL (the single group is every value in flat order; `func(&input.data, None)` matches the
group path's `func(&values, None)` where `values == input.data`; `func` deterministic). Toggled by
`NDIMAGE_LABELED_COMPREHENSION_GLOBAL_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median
vs A/A null, 16M px, light reducer): 121.95->4.08ms = **29.25x** (null [0.846,1.946], noisy cv~35% but 29x
dwarfs it), **bitmism=0**. CAVEAT: 29x is for a LIGHT reducer (the ~384MB alloc/copy dominates); a heavy
reducer amortizes the alloc so the win shrinks — but the alloc-removal is unconditional. bin
`perf_labcomp_global`. This closes the last global-label-stat allocation on a hot path. 10 ndimage wins this
session (1.5x → 29x, all byte-identical).

### 2026-07-11 (ScarletChapel, cc) — global median redundant-clone removal: IN-FLOOR (1.07x), NOT a win
`median` with `labels=None` DOUBLE-clones: `measurement_label_groups(None)` clones the whole array THEN
`median_of_values` clones AGAIN (`.to_vec()`) for its sort. A global fast path calls
`median_of_values(input.data)` directly, dropping the first clone. Byte-identical (same values, same sort;
bitmism=0). MEASURED (strict-remote release, paired median vs A/A null, 16M px): 675.91->605.03ms = **1.070x**,
INSIDE the A/A null band [0.798, 1.074] → **IN-FLOOR, not a decidable win**. ROOT CAUSE: median is
SORT-dominated (16M-element `total_cmp` sort ≈ 600ms), so the removed 128MB clone (~50ms) is only ~7% — below
the noise floor. Kept as a monotone byte-identical CLEANUP (a redundant clone removal can't be slower) but
explicitly NOT a perf win; toggled by `NDIMAGE_MEDIAN_GLOBAL_FORCE_SERIAL`. bin `perf_median_global`. LESSON:
clone-removal only pays when the clone is a large FRACTION — it dominated min/max/sum (cheap reduction, ~7x) but
is negligible under a full sort. This confirms the global-label-stat vein is DRY (median's remaining cost is
the inherent sort).

### 2026-07-11 (ScarletChapel, cc) — spatial::geometric_slerp parallel across t-values: 2.12x, byte-identical
First win OUTSIDE ndimage (crate restriction lifted; cod owns linalg/sparse). `spatial::geometric_slerp`
(spherical linear interpolation over an array of t-values; scipy's is serial) computed each interpolated point
in a SERIAL loop. Each point is INDEPENDENT — `a=sin((1-t)ω)/sinω`, `b=sin(tω)/sinω`, `point=a·start+b·end`
(two `sin` + `d` FMAs, compute-bound) — with the angle `ω` computed once up front. LEVER: fan the t-values
across cores (chunked `thread::scope`, results concatenated in t order). BYTE-IDENTICAL (identical per-t
arithmetic, order preserved; only the owning core changes). Work-gated (`n_t·(d+8) ≥ 2^16`); toggled by
`SPATIAL_SLERP_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
n_t=500k, d=64): 107.89->49.05ms = **2.118x** (null [0.900,1.166]), **bitmism=0**. Capped ~2x by the serial
256MB `Vec<Vec<f64>>` output allocation. bin `perf_slerp`. LEVER (reusable): grep for serial per-point
transcendental interpolation/mapping loops over an INPUT array (independent points) — a byte-identical
parallel win. CLEAN cc crates left: fsci-fft (kernel wall), fsci-spatial, fsci-stats.

### 2026-07-11 (ScarletChapel, cc) — spatial::cdist_func parallel across rows: 4.47x, byte-identical
`cdist_func(xa, xb, metric)` (the CUSTOM-metric cdist; scipy's callable-metric cdist is serial) mapped each
`xa` row to its distances-to-all-`xb` SERIALLY. Each row is INDEPENDENT → fan contiguous row-chunks across
cores (chunked `thread::scope`, rows concatenated in order). BYTE-IDENTICAL (identical per-pair `metric` calls,
order preserved) — added `+ Sync` to the metric bound (only caller is a conformance test passing Sync fns).
Work-gated (`m·n ≥ 2^14`); toggled by `SPATIAL_CDIST_FUNC_FORCE_SERIAL`. MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, m=n=1000, d=64, euclidean metric): 22.28->5.20ms = **4.468x** (null
[0.885,1.208]), **bitmism=0**. Better scaling than slerp (small m×n output alloc). Conformance
`diff_spatial_pdist_cdist_func` (cdist_func(euclidean)≡cdist_metric) unaffected (byte-identical). bin
`perf_cdist_func`. FOLLOW-ON: `pdist_func` (condensed, same pattern, trickier disjoint-range output indexing).
12 cc wins this session (10 ndimage + 2 spatial).

### 2026-07-11 (ScarletChapel, cc) — spatial::pdist_func parallel across rows: 3.65x, byte-identical
Follow-on to cdist_func. `pdist_func(data, metric)` (custom-metric CONDENSED pairwise; scipy's is serial)
computed the condensed vector via a serial double loop. Row `i` contributes the contiguous condensed block
[(i,i+1)..(i,n-1)]; blocks are independent and the condensed vector is exactly block_0 ++ block_1 ++ …, so
split contiguous i-ranges across cores and concatenate per-chunk blocks in i-order. BYTE-IDENTICAL (identical
per-pair `metric` calls + identical order); `+ Sync` on the metric (no non-test callers). Reuses
`SPATIAL_CDIST_FUNC_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
n=1400, d=64, euclidean): 22.78->6.05ms = **3.649x** (null [0.862,1.254]), **bitmism=0**. Slightly below
cdist_func (4.47x) due to the load imbalance of contiguous i-chunks (early rows carry more pairs) — a
balanced-by-cumulative-pairs split would recover it but breaks the simple order-preserving concat. bin
`perf_pdist_func`. 13 cc wins this session (10 ndimage + 3 spatial: slerp 2.12x, cdist_func 4.47x, pdist_func
3.65x). LEVER (reusable): serial closure-based map/double-loop over independent rows/pairs/points = byte-id
parallel win; add `+ Sync`, chunk contiguously to preserve output order.

### 2026-07-11 (ScarletChapel, cc) — opt::approx_derivative parallel across Jacobian columns: 5.99x, byte-identical
First win in fsci-opt. `approx_derivative` (the PUBLIC finite-difference Jacobian; scipy's is serial) computed
each Jacobian column in a SERIAL loop. Each column perturbs ONLY component `i` and evaluates `fun`
independently — so fan the columns across cores (chunked `thread::scope`, columns concatenated in order).
BYTE-IDENTICAL (identical per-column FD arithmetic, each `jt[i]` written once). Added `+ Sync` to the user
function bound (no internal callers — only Sync-closure tests; the parallel `fd_jacobian` benchmarked by
`perf_fd_jacobian_parallel` is a DIFFERENT internal fn, so this public one was missed). Toggled by
`OPT_APPROX_DERIV_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
n=48 params, m=48 outputs, expensive `fun`, ThreePoint): 829.36->127.59ms = **5.990x** (null [0.888,1.075]),
**bitmism=0**. A finite-difference Jacobian is used precisely when `fun` is expensive (ODE solve/simulation),
so the per-column evals dominate → this pays. Committed via a WORKTREE (opt/lib.rs had interleaved peer WIP in
`pub use`/`nnls_chol_refactor`/tests — pathspec commit would have swept it). bin `perf_approx_derivative`.
14 cc wins this session (10 ndimage + 3 spatial + 1 opt). LEVER (reusable): grep public callback-based fns
that map a user closure over independent items (FD Jacobians, custom-metric distances, comprehensions) and
lack `+ Sync` — parallelize byte-identically.

### 2026-07-11 (ScarletChapel, cc) — opt::approx_fprime parallel across gradient components: 2.55x, byte-identical
Direct sibling of approx_derivative — the FD GRADIENT (scalar objective; scipy's is serial). Each component's
forward difference perturbs ONLY `xk[index]` and evaluates `f` independently → fan the components across cores
(each thread a PRIVATE perturb buffer). BYTE-IDENTICAL: identical `(f(xk+ε·eᵢ)-f0)/ε` per component in index
order, AND the non-finite error reports the SAME lowest index the serial loop hits first (each component
returns `Result<f64,usize>`; the merge scans in index order for the first `Err`). Added `+ Sync` to `f` — the
only internal caller is `check_grad` (leaf, test-only callers), so the cascade is bounded (added `+ Sync`
there too). Toggled by `OPT_APPROX_FPRIME_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired
median vs A/A null, n=64 components, expensive `f`): 116.14->42.16ms = **2.547x** (null [0.860,1.095]),
**bitmism=0**. Lower than approx_derivative (5.99x) — the scalar `f` is lighter so parallel overhead + the
Result-collection are a bigger fraction. Committed via WORKTREE (opt/lib.rs peer WIP). bin `perf_approx_fprime`.
15 cc wins this session (10 ndimage + 3 spatial + 2 opt).

### 2026-07-11 (ScarletChapel, cc) — stats::jackknife parallel across leave-one-out replicates: 4.61x, byte-identical
First win in fsci-stats. `jackknife` (leave-one-out resampling; DETERMINISTIC — no RNG, unlike bootstrap)
computed each replicate's `statistic(data-minus-i)` in a SERIAL map. Each replicate is INDEPENDENT → fan across
cores (chunked `thread::scope`, replicates concatenated in `i` order). BYTE-IDENTICAL (identical per-replicate
subset + `statistic` call; the downstream jack_mean/bias/se operate on the same i-ordered replicates). Added
`+ Sync` to the statistic (callers are concrete Sync-closure wrappers — no cascade). Toggled by
`STATS_JACKKNIFE_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
n=3000, median statistic): 119.18->22.19ms = **4.612x** (null [0.869,1.316]), **bitmism=0**. stats crate was
CLEAN but the LEDGER was behind origin/main (opt entries landed via worktree), so committed via WORKTREE too.
bin `perf_jackknife`. 16 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats). KEY: jackknife is
byte-id-parallelizable (deterministic); bootstrap/permutation are NOT (RNG-order-dependent).

### 2026-07-11 (ScarletChapel, cc) — signal::group_delay_from_ba parallel across frequencies: 5.69x, byte-identical
Public-straggler lever. `group_delay_from_ba(b, a, n_freqs)` (the convenience group-delay sweep) computed each
frequency's delay in a SERIAL `for k in 0..n_freqs` loop, while its scipy-named sibling `group_delay` was ALREADY
parallel (both route the per-ω kernel through the shared `freqz_par_collect` helper). Each frequency's delay is a
PURE function of its index `ω_k=π·k/n`: `group_delay_at_frequency` reads only the immutable `b`/`a` and does two
`eval_weighted_poly_on_unit_circle` sweeps — a `cos`+`sin` PER COEFFICIENT (Σ k·c[k]·e^{-jkω}, the derivative
polynomial that plain Horner can't produce) — so per-ω work is O(len(b)+len(a)) transcendentals: compute-bound.
LEVER: fan the sweep across disjoint contiguous ω-chunks via `freqz_par_collect` (index-aligned, kernel pure) →
byte-identical to the serial loop; gated by `freqz_response_thread_count(n_freqs, len(b)+len(a))` (serial below
n_freqs<4096 or work<2^16). ORTHOGONAL to the prior Horner-routing of this fn (that cut the per-ω magnitude evals
to 1 cos+sin; the DERIVATIVE evals stay per-coefficient — which is exactly why the sweep is compute-bound and
parallelizes cleanly). Toggled by `GROUP_DELAY_FROM_BA_FORCE_SERIAL`. MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, order=1024 / n_freqs=16384): 264.50->41.54ms = **5.687x** (null
[0.885,1.172], serial cv 4.0%), **bitmism=0**. (A smaller order=256/nf=8192 probe was IN-FLOOR on a contended
worker — the raw ~2x was swamped by 43% parallel-arm cv; larger work amortised the scheduling jitter.) bin
`perf_group_delay_from_ba`. 17 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats + 1 signal). Sibling
stragglers `magnitude_response`/`phase_response` share the pattern but their per-ω kernel is Horner (1 cos+sin,
memory-bound) — lighter payoff, left for a follow-on. LEVER (reusable): grep public vs scipy-named sibling pairs
where the sibling routes a per-item kernel through a parallel helper but the convenience fn still loops serially.

### 2026-07-11 (ScarletChapel, cc) — signal::phase_response parallel across frequencies: 5.79x, byte-identical
Sibling straggler to `group_delay_from_ba` (same public-straddler vein). `phase_response(b, a, n_freqs)`
computed each frequency's phase in a SERIAL `for k in 0..n_freqs` loop, while the scipy-named `freqz`/
`group_delay` sweeps already route their per-ω kernel through the shared `freqz_par_collect` helper. Each
frequency's phase is a PURE function of its index `ω_k=π·k/n`: two Horner `eval_poly_on_unit_circle` sweeps
(O(len(b)+len(a)) complex MACs + a cos/sin each) plus two `atan2` — compute-bound at high filter order
(coeffs stay in L1, reused across all ω; the bottleneck is the per-ω arithmetic, not memory). LEVER: fan the
sweep across disjoint contiguous ω-chunks via `freqz_par_collect` (index-aligned, kernel pure) → byte-identical
to the serial loop; gated by `freqz_response_thread_count(n_freqs, len(b)+len(a))`. Toggled by
`PHASE_RESPONSE_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
order=2048 / n_freqs=16384): 68.76->10.40ms = **5.792x** (null [0.934,1.310], serial cv 7.0%), **bitmism=0**.
Needed order=2048 (2x group_delay's 1024) to clear the noise floor — the Horner kernel is ~10x lighter per-ω
than group_delay's per-coefficient trig, so the serial baseline is smaller (68ms vs 264ms). bin
`perf_phase_response`. 18 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats + 2 signal). REMAINING
straggler `magnitude_response` shares the pattern but is lighter still (1 sqrt vs 2 atan2) — likely IN-FLOOR
without a very high order; left on the frontier. VEIN NOW EXHAUSTED for the signal response family (freqz/
group_delay already parallel; group_delay_from_ba + phase_response landed; magnitude_response too light).

### 2026-07-11 (ScarletChapel, cc) — signal::magnitude_response parallel across frequencies: 3.51x, byte-identical
Last straggler of the signal response family (same public-straddler vein as group_delay_from_ba/phase_response).
`magnitude_response(b, a, n_freqs)` computed each frequency's |H| in a SERIAL `for k in 0..n_freqs` loop while
scipy-named `freqz`/`group_delay` already route their per-ω kernel through `freqz_par_collect`. Kernel = two
Horner `eval_poly_on_unit_circle` sweeps (O(len(b)+len(a)) complex MACs + cos/sin each) + a `sqrt` — pure per-ω
function of the index. LEVER: fan across disjoint contiguous ω-chunks via `freqz_par_collect` (index-aligned,
pure kernel) → byte-identical to the serial loop; gate `freqz_response_thread_count(n_freqs, len(b)+len(a))`,
toggle `MAGNITUDE_RESPONSE_FORCE_SERIAL`. `magnitude_response_db` WRAPS this fn (calls it then maps log10) so it
inherits the speedup for free. MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
order=3072 / n_freqs=16384): 102.41->22.39ms = **3.509x** (null [0.937,1.105], serial cv 2.6%), **bitmism=0**.
Lower than phase_response's 5.79x — the `sqrt` tail is lighter than phase's 2 atan2, so a larger fraction of the
parallel time is fixed thread overhead (hence order=3072 to clear the floor). bin `perf_magnitude_response`.
19 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats + 3 signal). SIGNAL RESPONSE FAMILY FULLY
EXHAUSTED: freqz/group_delay already parallel; group_delay_from_ba (5.69x) + phase_response (5.79x) +
magnitude_response (3.51x)/magnitude_response_db (inherits) landed. No more per-ω response stragglers.

### 2026-07-11 (ScarletChapel, cc) — signal::dfreqresp parallel across frequencies: 6.47x, byte-identical
Fresh straggler beyond the freqz-kernel family: `dfreqresp(num, den, dt, w)` (discrete-time complex frequency
response H(e^{jω})=num/den at an explicit ω-grid) did a SERIAL `w.iter().map(...).collect()` while its ANALOG
sibling `bode` (10569) already routes the identical shape through `freqz_par_collect`. Each ω is independent:
cos/sin + two Horner `eval_poly_complex` sweeps (O(len(num)+len(den)) complex MACs) + a complex divide — pure
per-ω function of the index. LEVER: fan across disjoint contiguous ω-chunks via `freqz_par_collect` (index-
aligned, pure kernel) → byte-identical to the serial map; gate `freqz_response_thread_count(w.len(),
2·(len(num)+len(den)))` (mirrors bode's work estimate), toggle `DFREQRESP_FORCE_SERIAL`. `dbode` (10599) calls
`dfreqresp` so it inherits the speedup for free. MEASURED (strict-remote release `+avx2,+fma`, paired median vs
A/A null, order=3072 / n_freqs=16384): 101.91->14.19ms = **6.474x** (null [0.948,1.068], serial cv 2.5%),
**bitmism=0**. HIGHER than the magnitude/phase siblings (6.47x vs 3.51/5.79x) — the kernel is pure Horner +
complex divide with NO atan2/sqrt tail, so the per-ω compute is cleaner and thread overhead is a smaller
fraction. bin `perf_dfreqresp`. 20 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats + 4 signal).
AUDIT NOTE: found by grepping the analog-vs-digital response sibling pair (bode parallel, dfreqresp/dbode serial)
— the same public-straddler heuristic across a DIFFERENT kernel (eval_poly_complex, not eval_poly_on_unit_circle).

### 2026-07-11 (ScarletChapel, cc) — signal::Lti::freqresp parallel across frequencies: 5.99x, byte-identical
Method straggler (public-straddler vein, on a struct method rather than a free fn). `Lti::freqresp(&self, w)`
(continuous-time transfer-function frequency response H(jω)=num(jω)/den(jω); scipy `lti.freqresp`) looped
`for &omega in w` SERIALLY while the free-fn `bode`/`dfreqresp` sweeps already route the identical shape through
`freqz_par_collect`. `Lti` is a `Vec<f64>` struct (Send+Sync) so `&self` is shareable; each ω is independent:
`eval_at(0,ω)` = two Horner `poly_eval_complex` sweeps (O(len(num)+len(den)) MACs) + complex divide + sqrt +
atan2 — pure per-ω function of the index. LEVER: fan across disjoint contiguous ω-chunks via `freqz_par_collect`
(index-aligned, pure kernel; the closure captures `&self` + `w`, both Sync) → byte-identical to the serial loop;
gate `freqz_response_thread_count(w.len(), 2·(len(num)+len(den)))`, toggle `FREQRESP_METHOD_FORCE_SERIAL` (shared
with the Dlti method — same lever). MEASURED (strict-remote release `+avx2,+fma`, paired median vs A/A null,
order=3072 / n_freqs=16384): 148.09->22.09ms = **5.994x** (null [0.931,1.038], serial cv 2.8%), **bitmism=0**.
bin `perf_lti_freqresp`. 21 cc wins this session (10 ndimage + 3 spatial + 2 opt + 1 stats + 5 signal). FOLLOW-ON
(next turn): `Dlti::freqresp` (18818) is the IDENTICAL serial method (uses `eval_at_freq`) and already reads the
shared `FREQRESP_METHOD_FORCE_SERIAL` gate name — same one-line routing through `freqz_par_collect`.

### 2026-07-11 (ScarletChapel, cc) — signal::Dlti::freqresp parallel across frequencies: 5.31x, byte-identical
Discrete-time sibling of `Lti::freqresp` (identical method straggler, shares the `FREQRESP_METHOD_FORCE_SERIAL`
gate already on origin). `Dlti::freqresp(&self, w)` (discrete-time transfer-function frequency response
H(e^{jωdt})=num/den; scipy `dlti.freqresp`) looped `for &omega in w` SERIALLY while the free-fn `bode`/`dfreqresp`
sweeps already route the identical shape through `freqz_par_collect`. `Dlti` is a `Vec<f64>`+`dt` struct
(Send+Sync); each ω is independent: `eval_at_freq(ω)` = two Horner `poly_eval_complex` + complex divide + sqrt +
atan2. LEVER: fan across disjoint contiguous ω-chunks via `freqz_par_collect` (index-aligned, pure kernel;
closure captures `&self`+`w`, both Sync) → byte-identical to the serial loop; gate
`freqz_response_thread_count(w.len(), 2·(len(num)+len(den)))`, toggle the SHARED `FREQRESP_METHOD_FORCE_SERIAL`
(commit is METHOD-ONLY — the atomic already landed with Lti::freqresp). MEASURED (strict-remote release
`+avx2,+fma`, paired median vs A/A null, order=3072 / n_freqs=16384): 103.56->15.47ms = **5.313x** (null
[0.956,1.038], serial cv 1.2%), **bitmism=0**. bin `perf_dlti_freqresp`. 22 cc wins this session (10 ndimage +
3 spatial + 2 opt + 1 stats + 6 signal). SIGNAL FREQUENCY-RESPONSE SURFACE NOW FULLY EXHAUSTED: free fns
(freqz/freqz_zpk/sosfreqz/freqs/freqs_zpk/group_delay/bode + group_delay_from_ba/phase_response/
magnitude_response/dfreqresp) AND methods (Lti/Dlti::freqresp) all parallel. No per-ω response stragglers remain.

### 2026-07-11 (ScarletChapel, cc) — signal::freqs parallel across frequencies: 4.83x, byte-identical
23rd win — REOPENS a straggler the consolidation below mis-listed as done. LEDGER-DRIFT CAUGHT: the "surface
fully exhausted" line above lists `freqs`/`freqs_zpk` as already-parallel, but reading ORIGIN SOURCE (not the
ledger prose) showed the ANALOG `freqs(b, a, w)` still looped `for &omega in w` SERIALLY — its sibling `bode`
(10620) already routes the identical `(ω, |H|, ∠H)` shape through `freqz_parallel_fill`. Each ω is independent:
two Horner `eval_analog_poly` sweeps (O(len(b)+len(a)) complex MACs) + a complex divide + a sqrt/atan2 tail —
pure per-ω function of the index. LEVER: fan across disjoint contiguous ω-chunks via `freqz_parallel_fill`
(index-aligned, pure kernel reading only immutable b/a/w) → byte-identical to the serial push loop; gate
`freqz_response_thread_count(w.len(), 2·(len(b)+len(a)))` (mirrors bode's work estimate), toggle new
`FREQS_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma` on vmi1293453, paired median vs A/A null,
order=3072 / n_freqs=16384): 954.84->183.27ms = **4.832x** (null median 1.005x range [0.938,1.143], serial cv
4.4%), **bitmism=0** (w + h_mag + h_phase all bit-identical across arms). bin `perf_freqs`. LESSON (reinforced):
VERIFY an "already parallel" claim against origin SOURCE before trusting exhaustion — the ledger's own summary
line drifted. FOLLOW-ON (queued, identical vein): `freqs_zpk(zpk, w)` (10415) is the STILL-SERIAL zpk twin
(kernel = k·Π(jω−z)/Π(jω−p) → (mag,phase)) — same one-fn routing through `freqz_parallel_fill`, separate
`FREQS_ZPK_FORCE_SERIAL` gate + bin.

### 2026-07-11 (ScarletChapel, cc) — ndimage::exp_array + log_array parallel transcendental maps: 2.00x / 3.05x, byte-identical
27th+28th wins — the compute-bound elementwise-map vein (opened by power_array) applied to the sibling
transcendental maps. `exp_array` did serial `data.iter().map(|&v| v.exp()).collect()`; `log_array` did
`.map(|&v| if v>0 { v.ln() } else { NEG_INFINITY })`. Both `exp`/`ln` are heavy per-element transcendentals
(~20-40 cycles) → COMPUTE-bound → routed through the same work-gated `fill_pixels_parallel(&mut out, 16, …)`
(byte-id, pure per-index; log's `v>0.0` branch is per-element so still pure). Toggles
`NDIMAGE_EXP_ARRAY_FORCE_SERIAL` / `NDIMAGE_LOG_ARRAY_FORCE_SERIAL`, bin `perf_explog_array`. MEASURED
(strict-remote release `+avx2,+fma` on vmi1149989, same-binary paired median vs A/A null, 4M elements):
`exp_array` 13.61→5.87ms = **2.002x DECIDED** (null [0.703,1.517] — marginal under heavy box contention,
parallel cv 37% but cand outside band), **bitmism=0**; `log_array` 24.63→6.14ms = **3.048x DECIDED** (null
[0.823,1.206] clean), **bitmism=0**. NOTE the speedup ORDERING confirms the discriminator: log (24.6ms serial,
ln+branch, heaviest) 3.05x > exp (13.6ms, plain exp) 2.00x > (add/mul, bandwidth-bound, WASH — not shipped).
The lighter the transcendental, the closer to the bandwidth floor and the lower the parallel multiple — but exp/ln
are both still compute-bound enough to DECIDE. `sqrt_array` (~1 instruction) is left serial (bandwidth-bound).
ndimage elementwise compute-bound-map vein now harvested (power/exp/log); the remaining unary ops (sqrt/neg/abs)
and the binary ops (add/mul/sub) are bandwidth-bound rejects.

### 2026-07-11 (ScarletChapel, cc) — ndimage::power_array parallel powf map: 3.61x, byte-identical
26th win — a NEW vein: compute-bound elementwise transcendental map (distinct from the serial-straggler-with-
parallel-sibling vein, now exhausted). `power_array(input, exponent)` did a serial
`input.data.iter().map(|&v| v.powf(exponent)).collect()`. `powf` is a heavy per-element transcendental
(~50-100 cycles), so this map is COMPUTE-bound — unlike the bandwidth-bound `add_arrays`/`multiply_arrays`
(x+y / x*y), which an Explore near-miss pass correctly flagged as likely-wash. LEVER: route the map through the
existing work-gated `fill_pixels_parallel(&mut output, kernel_work=16, |flat,_| input.data[flat].powf(exponent))`
helper — byte-identical (each output = `input.data[flat].powf(exponent)` in flat order, pure per-index), gated so
small arrays stay serial. Toggle `NDIMAGE_POWER_ARRAY_FORCE_SERIAL`. MEASURED (strict-remote release `+avx2,+fma`
on vmi1293453, same-binary paired median vs A/A null, bin `perf_power_array`, 4M elements, exponent=2.4):
38.51→9.84ms = **3.607x DECIDED** (null median 0.997x range [0.734,1.414] — wide under box contention but the
3.61x candidate is far outside), **bitmism=0** (all 4M elements bit-identical). Peer: numpy `np.power`,
single-threaded C. KEY: this REOPENS the compute-bound-map vein for ndimage elementwise ops — the DISCRIMINATOR is
per-element kernel weight: heavy transcendental (powf ✓ compute-bound → parallel wins) vs light arithmetic
(add/mul/sub ✗ bandwidth-bound → wash). FOLLOW-ON candidates (measure separately — lighter kernels, may be
borderline): `exp_array` (~20-40 cyc), `log_array` (~20-40 cyc); `sqrt_array` is a near-single-instruction →
bandwidth-bound, skip.

### 2026-07-11 (ScarletChapel, cc) — signal::gauss_spline parallelize the exp map: 2.67x, byte-identical
44th win — Explore follow-on. `gauss_spline(x, n)` (scipy.signal.gauss_spline) did serial `x.iter().map(|&xi| coef *
(-xi*xi/(2·signsq)).exp()).collect()` — one `exp` per element (coef/signsq hoisted), the whole function body. LEVER:
route through the order-preserving `par_index_fill` (the same helper the signal waveform gens/windows use) → BYTE-
IDENTICAL to the serial map. Toggle `GAUSS_SPLINE_FORCE_SERIAL`, bin `perf_gauss_spline`. MEASURED (strict-remote
release `+avx2,+fma` on vmi1167313, same-binary paired median vs A/A null, 8M elts, order=3): 51.25→17.52ms =
**2.668x DECIDED** (null [0.749,1.233] — robust), **bitmism=0** (full output). Clean single-exp elementwise map,
like ndimage exp_array. TEST-GATE: bin build served (compile verified); byte-id (bitmism=0) → median-gate ship.
FOLLOW-ON: spectral_flatness (signal:4654, serial Σln reduction — the last Explore candidate).

### 2026-07-11 (ScarletChapel, cc) — stats::GenNorm::logpdf_many parallelize the powf map: 3.08x, byte-identical
43rd win — found by an Explore sweep for "dominant serial heavy compute": `GenNorm::logpdf_many` (23269, scipy
`gennorm.logpdf` batch) was the LONE distribution `_many` method still serial — `xs.iter().map(|&x| lead -
x.abs().powf(b)).collect()` — while its sibling `pdf_many` (23281) AND every other distribution's `logpdf_many`
already use `par_continuous_map_min`. LEVER: route through `par_continuous_map_min(xs, 65536, |x| lead - x.abs()
.powf(b))` (the same helper+gate the sibling uses) → BYTE-IDENTICAL (order-preserving map, `lead` hoisted, pure
per-element powf). Toggle `GENNORM_LOGPDF_FORCE_SERIAL`, bin `perf_gennorm_logpdf`. MEASURED (strict-remote release
`+avx2,+fma` on vmi1167313, same-binary paired median vs A/A null, 8M elts, β=1.5): 116.40→32.33ms = **3.080x
DECIDED** (null [0.761,1.117] — robust, serial cv 6.3%), **bitmism=0** (full output vector). Clean single-powf
elementwise map (no reduction tax) → high 3.08x, like power_array/power_mean. TEST-GATE: bin build served (compile
verified); byte-id (bitmism=0) → shipped on median gate. LESSON: the "one sibling left serial" straggler recurs —
when N methods of a family are parallel and 1 isn't, that 1 is a byte-id win (freqs, SmoothBiv, filtfilt, now this).

### 2026-07-11 (ScarletChapel, cc) — stats::boxcox_llf parallelize the transform + Σln passes: 2.33x, byte-identical
42nd win — the Box-Cox log-likelihood objective (public + what `boxcox_normmax` optimizes over lambda). `boxcox_llf`
had TWO serial heavy passes: the transform (`(x^λ-1)/λ` or `ln x` per element, materialized) + `Σ ln(data)`. LEVER:
parallelize BOTH byte-identically — `par_map_inline(data, xform)` (order-preserving, the SAME helper `boxcox`'s own
transform uses → the compiler inlines/vectorizes the closure per-thread identically to the serial map) for the
transform, and `par_continuous_map(data, ln).iter().sum()` for the log-sum; the mean/variance passes over the
materialized `transformed` are unchanged. Toggle `BOXCOX_LLF_FORCE_SERIAL`, bin `perf_boxcox_llf`. MEASURED
(strict-remote release `+avx2,+fma` on vmi1149989, same-binary paired median vs A/A null, 8M elts, λ=0.5):
147.04→55.28ms = **2.333x DECIDED** (null [0.839,1.751], 33% margin), **bitmism=0** (result -1138300.236697203
both). Two heavy passes parallelized → higher than the single-pass geometric_mean (1.46x). Speeds up the whole
`boxcox_normmax` lambda search (calls llf ~20-50x). Drained the queue on rch recovery (the lever was code-complete
in stash@{0} after the prior rch-degraded surface). FOLLOW-ON: yeojohnson_llf (transform already parallel via
yeojohnson; its `Σ signum·ln` log_term is still serial → byte-id parallelize). CONTEXT: boxcox/yeojohnson TRANSFORMS
were already parallel; the LLF objectives were the last serial reduction passes in that family. TEST-GATE: bin build
served (compile verified) but heavy stats test compile refused (no admissible workers ×8) → shipped on MEDIAN gate
(BYTE-IDENTICAL bitmism=0 → no value regression possible + lib compiles; prior stats-suite runs 2023/0).

### 2026-07-11 (ScarletChapel, cc) — stats::cross_entropy parallelize the ln reduction: 2.33x, WITHIN-ULP
41st win — the cross_entropy sibling-straggler to kl_divergence (both left serial while `entropy` was parallel).
`cross_entropy` summed `-pᵢ·ln(qᵢ)` serially → added `ce_sum` mirroring `kl_sum`/`entropy_h_sum` EXACTLY (chunked,
4-way-unrolled), toggle `CROSS_ENTROPY_FORCE_SERIAL`, bin `perf_cross_entropy`. qi==0&pi>0 term → +INF (ln(0)=-INF,
·(-pi<0)=+INF) preserving scalar INFINITY. MEASURED (strict-remote release `+avx2,+fma` on vmi1149989, same-binary
paired median vs A/A null): 8M 2.004x (marginal, contended) → 24M 162.09→69.56ms = **2.330x DECIDED** (null
[0.534,1.345], 73% margin). WITHIN-ULP: rel drift **9.4e-14** (458 raw ULP on ~17.3) = same reorder entropy ships +
within scipy tol. TEST-GATE (mandatory for within-ULP): **fsci-stats --lib 2023/0** (cross_entropy scipy-ref tests
pass). Shipped on the FRESH SYNCED checkout (post 33-behind resync) with push-after-commit. ENTROPY-FAMILY REDUCTIONS
NOW FULLY PARALLEL: entropy (prior) + kl_divergence (c5b3351f7) + cross_entropy (this). The SIMD-reject was
single-thread-only — cross-core parallelization of all three wins 2-2.6x. Confirms the re-open-SIMD-rejects lesson.

### 2026-07-11 (ScarletChapel, cc) — stats::kl_divergence parallelize the ln reduction: 2.58x, WITHIN-ULP (first ULP-tolerant ship)
40th win, and the FIRST within-ULP (not byte-identical) ship this campaign — the operator authorized "byte-identical
OR within per-op ULP tolerance." `kl_divergence` did a serial `Σ pᵢ·ln(pᵢ/qᵢ)` (2 divides + a heavy `ln` per element
≈ 50-80 cyc → COMPUTE-bound) while its SIBLING `entropy` was already parallel via `entropy_h_sum`. LEVER: added
`kl_sum` mirroring `entropy_h_sum` EXACTLY (chunked across cores, 4-way-unrolled), toggle `KL_DIVERGENCE_FORCE_SERIAL`,
bin `perf_kl_divergence`. The `qi==0 && pi>0` term naturally yields `+INF` (pi/0=+INF, ln=+INF) → preserves the
scalar's INFINITY result. MEASURED (strict-remote release `+avx2,+fma` on vmi1227854, same-binary paired median vs A/A
null, 8M elts): 68.78→25.31ms = **2.578x DECIDED** (null [0.642,1.336]) — the HEAVIEST-kernel reduction win (2 div +
ln beats the single-transcendental means). **ULP DRIFT: rel 6.08e-14 (498 raw-bit ULP on a ~0.45 value)** — this is
the SAME 4-way-unroll+chunk reorder `entropy` ALREADY ships (so it's the codebase's accepted standard for this family)
AND within scipy's own pairwise-sum tolerance. **TEST-GATE MANDATORY for a within-ULP change (a byte-lock/tight-tol
test could break) — ran it: fsci-stats --lib 2023/0, incl. `kl_divergence_matches_scipy_reference_values` ✓ and
`entropy_kl_divergence_match_scipy` ✓** → the reordered result STILL matches scipy's references within tolerance =
proof the drift is within per-op ULP tolerance. Corrects the SIBLING-STRADDLER (entropy parallel, kl serial) and
CONTRADICTS the stale [[perf_stats_entropy_ln_reduction_reject]] (that was SIMD-ln ~1.15x single-thread; this is
PARALLELIZATION across cores = a different lever). FOLLOW-ON (same pattern): `cross_entropy` (28x/300 serial, sibling).
LESSON: a "rejected" reduction may have been rejected for SIMD (irreducible ln single-thread) — PARALLELIZATION across
cores is orthogonal and wins (2.58x here).

### 2026-07-11 (ScarletChapel, cc) — stats::geometric_mean parallelize the ln reduction: 1.46x, byte-identical
39th win — a SEPARATE public geometric-mean fn from `gmean` (which uses the already-parallel gmean_log_sum).
`geometric_mean(data)` did fused serial `log_sum = data.iter().map(ln).sum()` then `exp(log_sum/n)`. LEVER:
parallelize ONLY the ln map via order-preserving `par_continuous_map`, sum stays index-ordered → BYTE-IDENTICAL
(same values, same left-fold). Toggle `GEOMETRIC_MEAN_FORCE_SERIAL`, bin `perf_geometric_mean`. MEASURED (strict-
remote release `+avx2,+fma` on vmi1227854, same-binary paired median vs A/A null): 8M 32.27→22.18ms = 1.351x DECIDED
(marginal, 7% margin) → RE-MEASURED 24M: 93.90→69.90ms = **1.463x DECIDED** (null [0.825,1.270] 15% margin, cv 5-8%),
**bitmism=0** (result 1.866… both). MODEST (1.46x, below gstd's 1.6x) because the up-front serial validation pass
`data.iter().any(|&x| x<=0.0)` + the serial sum run in BOTH arms and cap the win (only the ln map parallelizes).
Lifts geometric_mean + its 5 internal callers (power_mean p→0 etc.). TEST-GATE: bin build served (compile verified) but heavy stats test compile refused (no admissible workers x8) -> shipped on MEDIAN gate (byte-id + lib compiles; prior stats-suite runs are 2023/0). NOTE: this is a DIFFERENT symbol from `gmean` (25230) — fsci has two geometric-mean
APIs; gmean was long-parallel (gmean_log_sum unrolled+chunked), geometric_mean (45287) was still serial.

### 2026-07-11 (ScarletChapel, cc) — signal::bode/dbode parallelize the mag/phase post-processing: 1.77x, byte-identical
38th win — pivot OUT of stats to signal's `bode_from_complex` post-processing (shared by `bode` + `dbode`). After
the parallel `freqz_par_collect` computes the complex response `h`, this helper did TWO serial heavy maps:
`mag = h.iter().map(|&(re,im)| 20·re.hypot(im).log10())` (hypot+log10) + `raw = h.iter().map(|&(re,im)| im.atan2(re))`
(atan2). With a LOW-ORDER filter + MANY frequencies (dense Bode plot), `h` is cheap and this post-processing DOMINATES.
LEVER: fan the two independent maps across cores via the order-preserving `freqz_par_collect` (the same helper `h`
uses) → BYTE-IDENTICAL (index order preserved); `unwrap_phase` stays serial (cumulative scan, not independent).
Gate `freqz_response_thread_count(n, 8)`, toggle `BODE_POST_FORCE_SERIAL`, bin `perf_bode_post`. MEASURED (strict-remote
release `+avx2,+fma` on vmi1149989, same-binary paired median vs A/A null, low-order H(jω)=1/(1+0.5jω)): 500k freqs
1.483x (fragile, contended cv 34%) → RE-MEASURED at 2M freqs: 91.56→43.62ms = **1.768x DECIDED** (null [0.812,1.202]
— 47% margin, serial cv 6.7%), **bitmism=0** (mag+phase). REGIME NOTE: gated at n_freqs≥8192, so typical few-point
Bode plots stay serial (no change/regression); the win is for DENSE frequency sweeps on low-order filters. LESSON:
after parallelizing the EXPENSIVE stage of a pipeline (h via freqz_par_collect), the POST-PROCESSING tail becomes the
new serial bottleneck for regimes where the expensive stage is cheap — parallelize the tail too. TEST-GATE: fsci-signal
--lib **674 passed / 0 failed** (rch served the signal test compile). Signal post-processing tail now parallel
(bode/dbode).

### 2026-07-11 (ScarletChapel, cc) — stats::gzscore/gzscore_ddof/gzscore_weighted parallelize the materialized ln map: 1.54x, byte-identical
37th win — the materialize-then-reduce sub-pattern (from gstd) applied to the geometric z-scores.
`gzscore_ddof(data)` = `zscore_ddof(ln(data))`; `gzscore_weighted` = `zscore_weighted(ln(data))` — both materialize
`logged = data.iter().map(ln).collect()` then `zscore` reduces it (mean, std, per-element output). LEVER: shared
`gzscore_ln_vec(data)` swaps the serial `.map(ln).collect()` for order-preserving `par_continuous_map(data, |x|
x.ln())` → BYTE-IDENTICAL (same ln values, index order), zscore unchanged. ONE lever lifts all three (gzscore→ddof).
Toggle `GZSCORE_FORCE_SERIAL`, bin `perf_gzscore`. MEASURED (strict-remote release `+avx2,+fma` on vmi1149989,
same-binary paired median vs A/A null, 8M elts, full-Vec bitmism): 91.80→52.59ms = **1.538x DECIDED** (null
[0.892,1.259] — 22% margin, robust), **bitmism=0**. ~Same magnitude as gstd (1.6x) — the ln map parallelizes, the
zscore mean/std/output passes stay serial and cap it. TEST-GATE: bin build served (compile verified) but heavy stats
test compile refused (no admissible workers ×10) → shipped on MEDIAN gate (byte-id + lib compiles); next stats-suite
confirms. MATERIALIZE-THEN-REDUCE sub-pattern now: gstd (1.60x) + gzscore family (1.54x) done. FOLLOW-ONS: any other
`map(heavy).collect()` feeding a multi-pass reducer (boxcox/yeojohnson log-transform are embedded in opt loops → not
this clean).

### 2026-07-11 (ScarletChapel, cc) — stats::gstd parallelize the materialized ln map: 1.60x, byte-identical
36th win — a SUB-PATTERN of reduction-map: parallelize a MATERIALIZED heavy transcendental map that feeds TWO
downstream reductions. `gstd` (geometric std = `exp(sqrt(var(ln(data))))`) built `logs = data.iter().map(ln).collect()`
then computed mean_log + var_log over it. LEVER: swap the serial `.map(ln).collect()` for the order-preserving
`par_continuous_map(data, |x| x.ln())` (par_continuous_map IS a parallel collect) → BYTE-IDENTICAL (identical ln
values in index order), so the serial mean/variance passes are unchanged. Toggle `GSTD_FORCE_SERIAL`, bin `perf_gstd`.
MEASURED (strict-remote release `+avx2,+fma` on vmi1227854, same-binary paired median vs A/A null): 4M 24.63→16.11ms
= 1.455x DECIDED (marginal, null [0.698,1.374], 6% margin under cv 16%) → RE-MEASURED at 16M for robustness:
153.25→90.33ms = **1.601x DECIDED** (null [0.776,1.167] — 37% margin, cv 13.9%), **bitmism=0** both (result
1.98…both). MODEST because only the ONE ln map parallelizes while the TWO serial reduction passes (mean, then
variance over `logs`) stay serial and cap the win — the win = parallelize the ln, the reductions are unchanged. This
is the "materialize-then-reduce-twice" sub-pattern: distinct from the fused map-sum (pmean) because the values are
NEEDED TWICE (mean, variance) so the Vec is materialized either way — the only change is the collect goes parallel.
LESSON: parallelize the `.map(heavy).collect()` when a later reduction needs the values ≥2× (can't fuse). TEST-GATE: bin build
served (compile verified) but heavy stats test compile refused (no admissible workers x10) -> shipped on MEDIAN gate
(byte-id + lib compiles); next stats-suite run confirms. FOLLOW-ONS: any `map(heavy).collect()` feeding
multiple reductions (2-pass mean/variance of a transformed array).

### 2026-07-11 (ScarletChapel, cc) — stats::rayleightest reuse the parallel sin/cos reduction: 4.17x, byte-identical
35th win — the circular-sin/cos vein extends to the directional TESTS. `rayleightest(samples)` (scipy.stats
Rayleigh test of circular uniformity) did serial `sum_cos = map(cos).sum()` + `sum_sin = map(sin).sum()` — two heavy
transcendentals, and NOTHING else heavy (r_bar/z/pvalue are O(1)). LEVER: one-line reuse of the existing shared
`circular_sincos_sums(samples)` helper (from the circmean win b79404bcc) → BYTE-IDENTICAL (each sum is a par map +
index-ordered sum; the cos-before-sin vs sin-before-cos order is irrelevant — independent sums). Shares the
`CIRC_FORCE_SERIAL` gate; bin `perf_rayleightest`. MEASURED (strict-remote release `+avx2,+fma` on vmi1149989,
same-binary paired median vs A/A null, 4M elts): 123.74→24.62ms = **4.172x DECIDED** (null [0.916,1.097] TIGHT, serial
cv 3.7%), **bitmism=0** (z AND pvalue both bit-identical). HIGHEST reduction-map win of the campaign — because
rayleightest is PURE two-transcendental sums with ZERO weighted-sum tax and NO other heavy work, so the two parallel
maps dominate completely and reach ~4x on a good box. LESSON: the cleanest reduction-map targets are the ones whose
ENTIRE cost is the transcendental map-sum (no weighting, no downstream heavy math) — rayleightest is the archetype.
TEST-GATE: bin build served (compile verified) but heavy stats test compile refused (no admissible workers ×10) →
shipped on MEDIAN gate (byte-id → no value regression + lib compiles); next stats-suite run confirms. Circular/
directional sin/cos surface: circmean/circvar/circstd (2.08x) + weighted (2.99x) + rayleightest (4.17x) all DONE via
ONE shared helper.

### 2026-07-11 (ScarletChapel, cc) — stats::circmean_weighted/circvar_weighted/circstd_weighted sin/cos reduction: 2.99x, byte-identical
34th win — the weighted circular family, direct follow-on to the unweighted circular win (b79404bcc). Same lever:
shared `circular_weighted_sincos_sums(data, weights)` parallelizes the sin/cos maps via order-preserving
`par_continuous_map`, keeps the weighted sums `w·sin[i]`/`w·cos[i]` index-ordered → BYTE-IDENTICAL (`w[i]·s[i]` =
`w[i]·x[i].sin()`, same left-fold). ONE lever lifts all three (circstd_weighted→circvar_weighted). Toggle
`CIRC_WEIGHTED_FORCE_SERIAL`, bin `perf_circmean_weighted`. MEASURED (strict-remote release `+avx2,+fma` on
vmi1227854/1293453, same-binary paired median vs A/A null, 4M elts): 131.68→39.19ms = **2.990x DECIDED** (null
[0.945,1.038] — TIGHT, serial cv 2.2% on a quiet box), **bitmism=0** (result 1.60296191416282 both). HIGHER than the
unweighted circmean's 2.08x — SAME lever, just a quieter box (that run was cv 40.9%). CONFIRMS the weighted circular
does NOT go marginal like gmean_weighted (1.17x): TWO transcendentals (sin+cos) dominate the weighted-sum tax where
gmean_weighted's single `ln` did not → the weighted variant is compute-bound iff ≥2 heavy transcendentals. CIRCULAR
STATISTICS FAMILY NOW FULLY DONE (unweighted b79404bcc + weighted this). TEST-GATE: bin build served (compile
verified) but heavy stats test compile refused (no admissible workers ×10) → shipped on MEDIAN gate (byte-id → no
value regression + lib compiles); next stats-suite run confirms. LESSON REINFORCED: weighted heavy-reduction ships robustly when the
per-element transcendental count is ≥2 (circular) but is marginal at 1 (gmean_weighted).

### 2026-07-11 (ScarletChapel, cc) — stats::circmean/circvar/circstd parallelize the sin/cos reduction: 2.08x, byte-identical
33rd win — a fresh public scipy-named family via the reduction-map-parallel vein. `circmean`/`circvar` did serial
`Σsin = data.iter().map(|&x| x.sin()).sum()` + `Σcos = ...cos().sum()` — TWO heavy transcendentals per element
(~20-40 cyc each) → the most compute-bound reduction of the mean family. `circstd` calls `circvar`. LEVER: extract a
shared `circular_sincos_sums(data) -> (sin_sum, cos_sum)` that parallelizes ONLY the sin/cos maps via order-preserving
`par_continuous_map`, keeping the sums index-ordered → BYTE-IDENTICAL (independent sin/cos values, same left-fold from
0.0). Toggle `CIRC_FORCE_SERIAL`, bin `perf_circmean`. ONE lever lifts all THREE (circmean/circvar directly, circstd
via circvar). MEASURED (strict-remote release `+avx2,+fma` on vmi1167313, same-binary paired median vs A/A null, 4M
elts): 189.35→70.70ms = **2.081x DECIDED** (null [0.876,1.623] — parallel cv 40.9% under box contention but the 2.08x
cand clears the ceiling by 28%), **bitmism=0** (result 1.1438146371745255 both). Serial is a big 189ms BECAUSE two
transcendentals/element → the reduction dwarfs the light double-sum tax → cleaner win than the single-kernel means
(pmean 1.9x). FOLLOW-ONS (same lever, weighted `w·sin`/`w·cos`): circmean_weighted/circvar_weighted (heavy sin/cos,
should ship ~2x — measure); von_mises fit sin/cos sum (embedded). TEST-GATE: rch served the bin build (compilation
VERIFIED) but refused the heavy stats test compile (no admissible workers ×10) → shipped on the MEDIAN gate (byte-id
→ no value regression + lib compiles); next stats-suite run confirms (as power_mean's 2023/0 confirmed pmean).
LESSON: circular stats do TWO transcendentals → highest compute:memory ratio of the mean family → robustly DECIDES.

### 2026-07-11 (ScarletChapel, cc) — stats::pmean_weighted parallelize the powf map inside the weighted reduction: 1.93x, byte-identical
32nd win — the reduction-map-parallel lever on the WEIGHTED sibling (overlooked when I did unweighted pmean/power_mean;
corrects the "reduction vein exhausted" claim — WEIGHTED variants are a whole parallel class). `pmean_weighted(data,p,
weights)` did serial `data.iter().zip(weights).map(|(&x,&w)| w*x.powf(p)).sum()`. LEVER: parallelize ONLY the `powf`
map via order-preserving `par_continuous_map`, then the light weighted sum `w·powed[i]` stays index-ordered →
BYTE-IDENTICAL (`w[i]·powed[i]` = `w[i]·x[i].powf(p)`, same left-fold from 0.0). Toggle `PMEAN_WEIGHTED_FORCE_SERIAL`,
bin `perf_wmean`. MEASURED (strict-remote release `+avx2,+fma` on vmi1293453, same-binary paired median vs A/A null,
4M elts p=2.5): 54.92→25.03ms = **1.931x DECIDED** (null [0.786,1.327]), **bitmism=0** (result 2.637476288765629 both).
SIBLING NOT SHIPPED: `gmean_weighted` (`w·x.ln()` sum) measured only **1.172x** (null [0.717,1.144] — barely DECIDED,
19% parallel cv) — the lighter `ln` kernel means the parallel benefit is mostly eaten by the weighted-sum tax → too
marginal/noisy to ship a robust win → REVERTED to serial, held for a quiet-box re-measure. TEST-GATE: rch gave workers
for the bin builds (compilation VERIFIED clean) but refused the heavier test compile (no admissible workers ×18) —
shipped on the MEDIAN gate (byte-id → no value regression possible + lib compiles); the next stats-suite run will
retroactively confirm (as power_mean's 2023/0 confirmed pmean). RCH FLEET intermittent — light builds land, test
compiles refused. LESSON (reinforced): when you parallelize `foo`, also grep `foo_weighted` — a whole sibling class.

### 2026-07-11 (ScarletChapel, cc) — stats::power_mean parallelize the powf map inside the reduction: 3.15x, byte-identical
31st win — the `pmean` reduction-map-parallel lever applied to its sibling `power_mean(data, p)` (a separate public
generalized-mean; `p→0` geometric, `p=-1` harmonic, `p=1` arithmetic). Same fused serial `data.iter().map(|&x|
x.powf(p)).sum()` → parallelize ONLY the `powf` map via order-preserving `par_continuous_map`, keep the sum in index
order → BYTE-IDENTICAL. Toggle `POWER_MEAN_FORCE_SERIAL`, bin `perf_power_mean`. MEASURED (strict-remote release
`+avx2,+fma` on vmi1227854, same-binary paired median vs A/A null, 4M elts p=2.5): 41.50→12.52ms = **3.154x DECIDED**
(null median 0.988x range [0.748,1.282]), **bitmism=0** (result 2.637360238656237 both arms). HIGHER than pmean's
1.94x purely because this run landed a quieter/faster worker (parallel arm 12.52ms vs pmean's 21.81ms on the more
contended vmi1149989) — SAME lever, SAME fixture; the win magnitude is worker-dependent, the byte-identity and the
DECIDE are not. TEST-GATE: `fsci-stats --lib` **2023 passed / 0 failed** — and since this worktree also carries the
pmean change (74a98a212), this GREEN suite RETROACTIVELY CONFIRMS the pmean win whose gate was rch-blocked last turn.
REDUCTION-MAP-PARALLEL VEIN across stats means: pmean + power_mean landed; `gmean` (ln-sum,
lighter) is the remaining follow-on; hmean (1/x) bandwidth-bound=skip.

### 2026-07-11 (ScarletChapel, cc) — stats::pmean parallelize the powf map inside the reduction: 1.94x, byte-identical
30th win — the "reduction vein" the operator opened by authorizing within-ULP changes, but landed BYTE-IDENTICAL
(zero ULP risk). `pmean(data, p)` computed `let power_sum = data.iter().map(|&x| x.powf(p)).sum()` — a fused serial
`powf`-map + left-fold. KEY INSIGHT: parallelizing the SUM would reorder the fold (a ULP change), but the `powf`
map dominates (~50-100 cyc vs ~1 cyc/add), so parallelize ONLY the map (via the order-preserving
`par_continuous_map`, byte-id) and keep the sum in index order → `par_continuous_map(data, |x| x.powf(p)).iter().sum()`
is BYTE-IDENTICAL to the fused `map(powf).sum()` (same independent values, same left-fold from 0.0 over the same
sequence). Toggle `PMEAN_FORCE_SERIAL`, bin `perf_pmean`. MEASURED (strict-remote release `+avx2,+fma` on vmi1149989,
same-binary paired median vs A/A null, 4M elts p=2.5): 42.80→21.81ms = **1.940x DECIDED** (null median 0.967x range
[0.771,1.283]), **bitmism=0** (result 2.637360238656237 both arms). LOWER than power_array's 3.61x because the
serial ordered sum adds an O(N) read+add pass + the Vec materialization the fused version avoids — that tax is the
cost of staying byte-identical. FOLLOW-ONS (same byte-id pattern, measure): `power_mean` (45173, IDENTICAL powf-sum
duplicate ~same 1.94x), `gmean` (25230, ln-sum, lighter kernel ~1.5x). `hmean`/`harmonic` (1/x reciprocal-sum) is
LIGHT → bandwidth-bound → skip. LESSON (reusable): a `map(heavy_transcendental).sum()` reduction has a BYTE-IDENTICAL
parallelization — parallelize the map, keep the ordered sum — whenever the transcendental dominates the add; no ULP
change needed. The ULP chunked-partial-sum variant (fuses away the serial-sum pass for a bigger win) was NOT taken:
byte-identical 1.94x is a clean ship, and the chunked sum would move last bits (surface-if-beyond-tolerance per the
operator's gate). TEST-GATE NOTE: the full `fsci-stats --lib` suite could NOT run — rch was persistently saturated
("no admissible workers" ×14 attempts) and strict-remote policy forbids local. Shipped anyway on the MEDIAN gate:
the change is byte-identical (bitmism=0 → no value assertion can regress) and the lib COMPILES clean (the perf_pmean
bin built against the full fsci-stats lib). RETRY: re-run `cargo test -p fsci-stats --lib` when rch recovers (expect
green; it is a pure map-parallelization behind a work-gate).

### 2026-07-11 (ScarletChapel, cc) — signal::filtfilt_axis_2d hoist lfilter_zi out of per-line loop: 1.66x, byte-identical
29th win — the shared-predictor HOIST vein (compute-once-reuse), found by a 2nd Explore fan-out for the
recompute-inside-loop anti-pattern (distinct from the parallelization straggler hunt). `filtfilt_axis_2d(b,a,x,axis)`
calls `filtfilt(b,a,line)` per line; each `filtfilt` recomputes `lfilter_zi(b,a)` — an **O(order³) dense linear
solve** (`fsci_linalg::solve` on an (order-1)² matrix + `Vec<Vec>` alloc) that is query-INDEPENDENT (depends only
on b/a). Over L lines that's L redundant solves. LEVER: factor `filtfilt_with_padtype_zi(b,a,x,padtype,zi_pre)`
(the old body, taking an optional precomputed zi; `filtfilt_with_padtype` = it with `None` → single-call path
byte-identical, same validation/error order); in `filtfilt_axis_2d` solve `lfilter_zi` ONCE, thread it into the
per-line closure. FALLBACK: on any `lfilter_zi` error (e.g. order-1 filter) or the `FILTFILT_AXIS_HOIST_DISABLE`
knob, revert to the exact per-line `filtfilt` path → error behaviour unchanged. BYTE-IDENTICAL: zi is deterministic
in (b,a); each line still scales it by its own first sample. MEASURED (strict-remote release `+avx2,+fma`,
same-binary paired median vs A/A null, bin `perf_filtfilt_axis_hoist`): order=14/6000×350 1.637x IN-FLOOR (box
contended, null blew to [0.709,2.258], cv 42% BOTH arms — the null jittered, cand consistent); order=20/10000×260
(heavier+quieter) **1.657x DECIDED** (null [0.752,1.343], cv 15.8%), **bitmism=0** both. WIN RATIO ≈ order²/len, so
DECIDES for HIGH-ORDER filters (sharp elliptic/Chebyshev) on many modest-length lines; marginal for low order/long
lines. MEASURE LESSON (reconfirmed 3rd time): contention blows out the A/A NULL, not the candidate — scale work up
(bigger order+more rows = longer call) to amortize scheduling jitter, don't accept the IN-FLOOR. The hoist vein is
otherwise SATURATED (2nd Explore fan-out: this was the sole non-marginal remaining site; sosfiltfilt/welch/csd/
spectrogram/coherence _axis_2d window-rebuilds are all rebuild≪per-line → left as documented near-misses).

### 2026-07-11 (ScarletChapel, cc) — SmoothBivariateSpline::eval_many hoist+parallel: 1.78x, byte-identical
25th win — a fresh vein OUTSIDE the exhausted signal-response family, found by re-running the freqs-class
"serial straggler with a parallel sibling" audit across the accessible crates (Explore fan-out). The pointwise
`SmoothBivariateSpline::eval_many(x, y)` (scattered (x,y) pairs) did a SERIAL `x.iter().zip(y).map(|(&xv,&yv)|
self.eval(xv,yv)).collect()` while BOTH the sibling `RectBivariateSpline::eval_many` (8859, commit d380511db
3.0-25.2x) AND this struct's own `eval_grid` (9287, commit b9c6ee6b5 5.1-8.4x) had already been given the
shared-predictor hoist + a parallel driver — the pointwise variant was the last straggler of the trio.
LEVER (same recipe as the sibling): `self.eval`→`eval_impl` rebuilds the ny x-direction BSplines
(`coeffs.chunks(nx_coeffs)`, cloning tx + a coeff row) on EVERY query though they depend only on the spline;
build them ONCE, then a per-query `eval_one` (finite-guard + bbox clamp + shared x-spline evals + one
query-dependent y-spline build/eval — mirrors `eval`'s exact order, so byte-identical) fans across cores via the
existing `par_query_map(&pairs, work_per_query=coeffs.len(), …)`. Toggle `SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR`;
x-spline build failure or the knob falls back to the exact serial `self.eval` map. MEASURED (strict-remote release
`+avx2,+fma`, same-binary paired median vs A/A null, bin `perf_smoothbispline_evalmany`): 30x30 samples/20k
queries **1.776x DECIDED** (null [0.914,1.190]); 45x45/40k **1.785x DECIDED** (null [0.488,1.426]); 60x60/100k
1.645x IN-FLOOR (null band blew out to [0.705,1.689] under box contention — candidate median STILL 1.65x, i.e.
the null jittered, not the candidate). **bitmism=0 all three runs.** Candidate median rock-stable ~1.78x across
sizes → shipped on the two DECIDED runs. MODEST (1.78x, not the sibling's 3-25x) because at these smoothing
factors the SmoothBiv fit has fewer coeffs than a full RectBiv tensor grid, so the hoisted per-query x-spline
rebuild is a smaller share; the bound is memory/dispatch, not compute. AUDIT LESSON: the freqs-class straggler
audit (serial public per-item loop + a parallel sibling) generalizes BEYOND signal — one member of a sibling
TRIO (RectBiv-evalmany / SmoothBiv-evalgrid / SmoothBiv-evalmany) was left serial after the other two shipped.

### 2026-07-11 (ScarletChapel, cc) — signal::freqs_zpk parallel across frequencies: 5.60x, byte-identical
24th win — completes the analog-response straggler pair opened by `freqs` (23rd). `freqs_zpk(zpk, w)` (10415)
looped `for &omega in w` SERIALLY while the sibling `bode`/`freqs` sweeps already route the identical
`(ω, |H|, ∠H)` shape through `freqz_parallel_fill`. Each ω is independent: two factored-product sweeps
(num = k·Π(jω−z), den = Π(jω−p) via the local `cmul`) over the immutable zero/pole lists + a complex divide +
a sqrt/atan2 tail — pure per-ω function of the index (`ZpkCoeffs` is a `Vec<f64>`+gain struct, Send+Sync, so
the kernel closure captures `&zpk`+`w`, both Sync). LEVER: fan across disjoint contiguous ω-chunks via
`freqz_parallel_fill` (index-aligned, pure kernel) → byte-identical to the serial push loop; gate
`freqz_response_thread_count(w.len(), 2·(zeros+poles))`, toggle new `FREQS_ZPK_FORCE_SERIAL`, `cmul` moved
inside the kernel (pure local, no arithmetic change). MEASURED (strict-remote release `+avx2,+fma` on
vmi1227854, same-binary paired median vs A/A null, order=1024 zeros=poles / n_freqs=16384): 58.03->8.40ms =
**5.595x** (null median 1.011x range [0.960,1.046], serial cv 1.9%), **bitmism=0** (w+h_mag+h_phase all
bit-identical). bin `perf_freqs_zpk`. SIGNAL FREQUENCY-RESPONSE SURFACE NOW GENUINELY EXHAUSTED (verified against
ORIGIN SOURCE, not ledger prose): every free fn (freqz/freqz_zpk/sosfreqz/freqs/freqs_zpk/group_delay/bode +
group_delay_from_ba/phase_response/magnitude_response/dfreqresp) AND both methods (Lti/Dlti::freqresp) now route
per-ω through a parallel helper. No serial per-ω response fn remains. 24 cc wins across the campaign.

---

## SESSION CONSOLIDATION — 2026-07-11 (ScarletChapel, cc): 22 byte-identical wins, then FRONTIER+HOLD
Roll-up of the byte-identical (bitmism=0, median-gated) parallelization campaign. All wins strict-remote release
`+avx2,+fma`, paired median vs A/A null, same-binary `*_FORCE_SERIAL` toggle.
- **ndimage ×10** (1.5–29x): global label-stat clone-drops (`measurement_label_groups(None)` 128 MB clone) +
  privatized parallel histograms (labeled_comprehension, label gathers, otsu, histogram, min/max, sum, variance/std,
  extrema, labcomp-global; median-global was IN-FLOOR cleanup).
- **signal ×6** (public-straddler vein): group_delay_from_ba 5.69x, phase_response 5.79x, magnitude_response 3.51x,
  dfreqresp 6.47x, Lti::freqresp 5.99x, Dlti::freqresp 5.31x. Frequency-response surface now fully parallel.
- **spatial ×3** (callback-map + Sync): geometric_slerp 2.12x, cdist_func 4.47x, pdist_func 3.65x.
- **opt ×2**: approx_derivative, approx_fprime (callback-map + Sync, first-error preserved).
- **stats ×1**: jackknife 4.61x (deterministic replicates; bootstrap/permutation excluded — RNG-order-dependent).
FRONTIER (full detail + retry conditions in docs/NEGATIVE_EVIDENCE.md → "FRONTIER SUMMARY — cc byte-identical
parallelization lane"): the accessible byte-id parallelization/hoist/structural surface is SATURATED. Known-ready
but unshipped: `freqs`/`freqs_zpk` (analog response stragglers of the already-parallel `bode`) — byte-id code
complete, blocked only on rch capacity (fleet too contended 2026-07-11 to obtain a median). Structural byte-id
primitives are walled (FFT SoA-SIMD behind forbid(unsafe); cache-blocking rejected; tolerance rewrites owner-gated).
HOLD until a retry condition is met.

### 2026-07-11 (cc) — signal::gauspuls parallel 3-output fill: 8.9x, byte-identical
Fresh win from the "one sibling left serial in an otherwise-parallel family" vein. `gauspuls(t, fc, bw, bwr)`
(the i/q/envelope form of `scipy.signal.gauspuls`, signal lib.rs:13183) looped its per-sample kernel SERIALLY
while its direct sibling `gausspulse` (2362, the real-part-only form) was already parallel via `par_index_fill`.
The kernel is the HEAVIEST of the campaign — 3 fused transcendentals per element: `e = exp(-a·t²)`, then
`e·cos(2π·fc·t)` and `e·sin(2π·fc·t)`. Because it writes THREE output arrays (i/q/envelope) there is no
single-output helper to reuse, so I added a dedicated work-gated 3-output fill: factor the kernel into ONE
shared closure (both arms run identical arithmetic → byte-identical by construction), preallocate the 3 vecs,
and in the parallel arm fan disjoint contiguous chunks of all three across cores via `thread::scope` +
triple-`chunks_mut` zip (`out[i]` is a pure function of `t[i]`, so bit-identical to the serial push loop; only
the owning core changes). Gate `>=4096 samples/thread` (same as the sibling), toggle `GAUSPULS_FORCE_SERIAL`,
bin `perf_gauspuls`. MEASURED (release `+avx2,+fma`, same-binary paired median vs A/A null, 4M samples/fc=1000,
21 iters, twice): serial 185.40→parallel 19.88ms = **8.861x DECIDED** (null [0.927,1.060]) and 177.78→19.46ms =
**8.939x DECIDED** (null [0.911,1.056]); **bitmism=0** across all three output vectors both runs. fsci-signal
--lib 674/0 (incl. all 5 gauspuls tests + `gauspuls_zero_center_frequency_matches_scipy`). HIGHEST reduction/
map-parallel ratio of the campaign — 3 heavy transcendentals fused per element gives the largest compute:memory
ratio, so the parallel fill dominates the light writes. Peer scipy.signal.gauspuls is single-threaded numpy.
LEVER (reusable): a MULTI-OUTPUT serial waveform generator (i/q/envelope, real/imag) parallelizes byte-id with a
dedicated N-buffer chunked fill even when no single-output helper fits — factor the kernel into one shared closure
so serial and parallel arms are provably identical. AUDIT: grep waveform/window/wavelet generators for the lone
serial member whose sibling is already `par_index_fill` (gausspulse→gauspuls; nuttall/bohman/morlet2 remain).

### 2026-07-11 (cc) — signal::morlet/morlet2 parallel tuple fill: 7.9x, byte-identical
Next straggler from the same waveform/wavelet-generator vein. The complex Morlet wavelet generators `morlet(m,w,s,
complete)` (3108) and `morlet2(m,s,w)` (3153, the default cwt wavelet) each filled their `Vec<(f64,f64)>` output
SERIALLY with a per-sample `exp` + `sin_cos` (3 transcendentals/element — same weight as gauspuls). Added a
tuple-valued twin of `par_index_fill`, **`par_index_fill_pairs`** (same >=4096-idx/thread work gate, order-preserving,
byte-identical), and routed BOTH generators through it — one shared helper + one shared toggle `MORLET_FORCE_SERIAL`
lifts the whole family. Each per-sample kernel is factored into ONE closure `|i| -> (f64,f64)` reused by the serial
`(0..m).map(kernel).collect()` and the parallel arm → provably identical arithmetic → byte-id (each pair is a pure
function of `i`). bin `perf_morlet2`. MEASURED (release `+avx2,+fma`, same-binary paired median vs A/A null, m=4M,
w=5, s=m/16, 21 iters, ×2): serial 93.33→parallel 11.42ms = **7.931x DECIDED** (null [0.904,1.070]) and 95.78→11.64ms
= **7.793x DECIDED** (null [0.909,1.123]); **bitmism=0** across all (re,im) pairs both runs. fsci-signal --lib 674/0
(incl. morlet2_optimized_matches_naive / morlet_optimized_matches_naive / morlet2_centering_matches_scipy_golden).
`m` is the wavelet length — large in cwt (scipy caps at min(10·width, len(data))) and for standalone long wavelets;
gated so short wavelets stay serial (no regression). Peer scipy.signal.morlet/morlet2 is single-threaded numpy.
NEW REUSABLE PRIMITIVE: `par_index_fill_pairs` (the tuple twin of `par_index_fill`) unlocks every complex/paired-
output generator. QUEUED (same vein, single-output `par_index_fill` drop-ins): `nuttall_window` (3×cos, sibling
blackmanharris par), `bohman_window` (cos+sin, sibling barthann par); stats `vtest` (Σcos+Σsin shifted, circular family).

### 2026-07-11 (cc) — signal::nuttall_window + bohman_window parallel fill: 11.0x / 6.4x, byte-identical
Drained the two queued single-output window stragglers. `nuttall_window(m)` (3×`cos`/sample) and `bohman_window(m)`
(`cos`+`sin`/sample) each filled `Vec<f64>` with a serial `(0..m).map(...).collect()` while their direct siblings
`blackmanharris` and `barthann` were ALREADY `par_index_fill`. Trivial precedent-backed drop-in: factor each kernel
into a closure, route through the order-preserving `par_index_fill` (byte-id), gate the same-binary A/B on a shared
`WINDOW_FORCE_SERIAL`, bin `perf_windows` (measures both). MEASURED (release `+avx2,+fma`, same-binary median vs A/A
null, m=4M, 21 iters, ×2): nuttall serial 98.96→8.47ms = **10.964x DECIDED** (null [0.916,1.063]) and 97.98→8.26ms =
**10.948x DECIDED** (null [0.932,1.081]); bohman 44.84→6.50ms = **6.390x DECIDED** (null [0.887,1.110]) and
44.78→6.64ms = **6.575x DECIDED** (null [0.895,1.105]); **bitmism=0** all four runs. fsci-signal --lib 674/0 (incl.
nuttall_window_matches_scipy_reference / bohman_window_matches_scipy_reference / nuttall_window_symmetric). nuttall's
11x is the HIGHEST single-output map win of the campaign (3 cos, ~99ms serial → the parallel 8.5ms rides the full core
count). `m`=window length, large for long-FFT segments (nperseg 2^16+); gated so short windows stay serial. Peer
scipy.signal.windows.nuttall/.bohman is single-threaded numpy. SIGNAL WINDOW GENERATORS now all parallel (hann/hamming/
blackman/blackmanharris/barthann/flattop/tukey/kaiser/... + nuttall/bohman landed). WAVEFORM/WINDOW/WAVELET VEIN this
session: gauspuls 8.9x + morlet/morlet2 7.9x + nuttall 11.0x + bohman 6.4x = 5 fns, 3 commits. Remaining serial gen
straggler: `general_hamming` (799, 1×cos — light, likely marginal). Cross-crate: stats `vtest` (circular family).

### 2026-07-11 (cc) — stats::vtest parallel circular reduction: 5.4x, byte-identical
The LAST serial member of the directional sin/cos-sum family (circmean/circvar/circstd/rayleightest + weighted all
already route their Σsin/Σcos through the parallel `par_continuous_map`; `vtest` was missed). `vtest(samples, mu)`
(V-test of circular uniformity with a known mean direction, stats:26224) did serial `Σcos(x−mu)` + `Σsin(x−mu)` —
two heavy transcendentals/element, everything else O(1) (the circular sweet spot). Parallelized ONLY the two maps via
the order-preserving `par_continuous_map` (`|x| (x−mu).cos()` / `.sin()`, sums kept index-ordered) → BYTE-IDENTICAL to
the serial `map(cos).sum()`/`map(sin).sum()` (same left-fold from 0.0); shares `CIRC_FORCE_SERIAL` with the family.
bin `perf_vtest`. MEASURED (release `+avx2,+fma`, same-binary median vs A/A null, 4M/mu=1, 21 iters, ×2): serial
113.42→19.46ms = **5.444x DECIDED** (null [0.932,1.044]) and 113.20→19.80ms = **5.463x DECIDED** (null [0.890,1.042]);
**bitmism=0** (v AND pvalue bit-identical, v=0.9938637087169686 both) both runs. fsci-stats --lib **2023/0** (incl.
vtest_returns_valid_results). ~5.4x = pure two-transcendental sums with no downstream heavy math (rayleightest-class);
higher than circmean's 2.08x (which has atan2+rem_euclid tails). Peer: astropy.stats.vtest (scipy has no vtest) →
byte-identical self-speedup. **DIRECTIONAL sin/cos-sum FAMILY NOW FULLY EXHAUSTED.** 49 cc wins total this campaign.

### 2026-07-11 (cc) — REJECT: special::softmax parallel exp/divide map — IN-FLOOR (bandwidth-bound)
`special::softmax` (scipy.special.softmax, convenience.rs:4269) was flagged as a serial `exp`-map straggler that
bypasses the elementwise family's `par_map_light`. Parallelized the two per-element maps (exp, divide) via
`par_map_light` (byte-identical, gated LIGHT_KERNEL_PAR_MIN), max/sum kept serial. MEASURED (release `+avx2,+fma`,
same-binary median vs A/A null, 8M, 21 iters, ×2): serial 107.86→104.16ms = **1.034x IN-FLOOR** (null [0.581,1.114])
and 107.52→100.85ms = **1.059x IN-FLOOR** (null [0.912,1.118]); **bitmism=0**. ROOT CAUSE = MEMORY-BANDWIDTH-BOUND:
softmax streams FOUR O(N) passes over 64MB Vecs (max reduce / exp map / sum reduce / divide map) — the `exp` is only
~25% of the 107ms serial time (a single 8M exp map ≈ 27ms cf. ndimage exp_array), so the two light reductions + the
memory streaming dominate, AND `par_map_light` adds a serial per-chunk concat copy. Under a bandwidth-contended box the
parallel arm can't reclaim time (bandwidth is a SHARED resource — unlike compute contention, which the median-null gate
absorbs). Quiet-box ceiling is only ~2-2.5x even in theory. REVERTED (no ship). **RULE (extends the elementwise-map
discriminator): a HEAVY-transcendental map WINS parallelized only when the transcendental DOMINATES the per-element
cost AND the fn is few-pass; a multi-pass streaming fn (softmax/log_softmax = max+exp+sum+divide) DILUTES the
transcendental below the memory-traffic floor → WASH, even at 8M.** Same applies to the queued log_softmax/logsumexp
(exp buried among light passes) — SKIP. `jensenshannon` (spatial:517, 2 ln/elt) is heavier per-element but a
single-(p,q)-pair scalar reduction (byte-id parallel-sum awkward) → only large-D, deprioritized.

### 2026-07-25 (cc/CopperFalcon) — KEEP / **SELF-SPEEDUP** (BIT-IDENTICAL): BDF exact-diagonal structured Newton solve — 1.91x @n=32 → 109.37x @n=512 (97.68x after the later fused-construction row, which sped the dense baseline too)
**Result class:** `SELF-SPEEDUP`.
**LEDGER RESURRECTION, rank #1** (`docs/LEDGER_RESURRECTION.md`, campaign `perf-campaign-20260725` Meta-Lever #1).
Entry `.165` (2026-07-23, `docs/progress/perf-negative-results.md:151`) measured this exact lever at **17.384x /
17.069x / 19.283x / 18.964x** across four runs against A/A nulls of **1.002-1.020**, proved exact full-`SolveIvpResult`
identity and 16/16 focused BDF conformance — and was **REJECTED anyway on a `cv < 5%` ceiling**. That gate is
unreachable on this hardware (frankenmermaid calibration: floor ~12%), so the row is **VOID**: the harness was
rejected, not the lever. Re-decided here under the §2 harness contract.
**Result class: SELF-SPEEDUP.** Both arms are OUR code (`BDF_FORCE_DENSE_NEWTON` on vs off); no SciPy
arm was measured, so this ratio is MAINTENANCE and must not be quoted as a competitive claim. Live-arm
work: `frankenscipy-0bu5p`.
**LEVER (one).** `BdfSolver`'s Newton matrix `I − c·J` is factorized by an unconditional dense `nalgebra` LU
(`bdf.rs`, 80.09% self-cycles in the `.165` profile, dense solve another 5.16%; independently re-confirmed on
MY binary — `perf record -F 499 --call-graph=dwarf`, n=256: `LU::new` **67.44% self**, `LU::solve_mut` 7.14%,
`solve_upper_triangular_mut` 7.13%, i.e. the frames this lever deletes are 81.7% of the profile the bench
actually executes). When the finite-difference Jacobian
is EXACTLY diagonal — componentwise dynamics `y_j' = f_j(t, y_j)`: decoupled stiff relaxation, per-species decay,
method-of-lines with only local terms — `I − c·J` is diagonal too, so the O(n³) factorization plus two O(n²)
substitutions collapse to `n` scalar divisions. New `enum NewtonFactor { Dense(LU), Diagonal(Vec<f64>) }`;
`exact_diagonal_newton()` is a column-major scan that bails on the FIRST non-zero off-diagonal, so a genuinely dense
Jacobian pays O(1), not O(n²). scipy does NOT exploit this (it always `lu_factor`s the dense system unless the user
hands it a `jac_sparsity`), so the win is against scipy as well as against our own prior path.
**BIT-IDENTICAL, not "equivalent."** With every off-diagonal exactly `0.0` and every `1 − c·J[j][j]` finite and
non-zero: partial pivoting selects row `j` strictly (`|d_j| > 0`) so the permutation is the identity; unit-lower
forward substitution computes `b[k] -= 0.0 * b[i]`, leaving finite `b` unchanged; back substitution is exactly
`b[j] / (1 − c·J[j][j])` — same IEEE division, same operands, same order, and the diagonal entry is formed with the
same expression the dense arm uses. The ONE divergence risk is non-finite intermediates (dense substitution multiplies
zeros against `±inf`/`NaN` and spreads `NaN` across components the diagonal arm keeps independent): the solve detects a
non-finite rhs or quotient and reconstructs the dense LU for that iteration, i.e. runs the dense computation itself.
`nlu`/`njev`/`nfev` counters are unchanged by construction. Unit test `bdf_diagonal_newton_is_bit_identical_to_dense_lu`
compares full trajectories + counters as raw bits across three system shapes (diagonal / coupled / mixed zero-row) with
`BDF_DIAG_NEWTON_HITS` as the EXECUTION PROOF (a coupled Jacobian must score zero hits; a diagonal one must score > 0 —
without that assert a broken predicate would make the test pass vacuously, which is exactly the failure mode that
voided a third of this repo's REJECT ledger).
**MEASURED** (`perf_bdf_diag_newton`, §2 contract: self-reported `elf_sha256=17f7355509ea7fa9a6117f2474ed2110a230236be5881145e2b19db7146cf3b9`
== the shell-side sha of the shipped binary,
`paired(base,base)` A/A null then `paired(base,cand)` in ONE invocation, arms interleaved with per-round alternation,
statistic = median of per-round ratios, `min_of=3` inner replicates, gate = candidate CI-low above `1 + 2*(null_edge−1)`,
`cv` reported as provenance only; thinkstation1 5975WX, `taskset -c 2`, load 15-16, build remote on ovh-a):

| n | base p50 | cand p50 | **ratio_p50** | cand ci95 | A/A null ci95 | gate | bitmism |
|---:|---:|---:|---:|---|---|---|---:|
| 32 | 30.88 ms | 16.16 ms | **1.912x** | [1.802, 2.033] | [0.971, 1.085] | DECIDED | 0 |
| 64 | 44.33 ms | 15.23 ms | **2.908x** | [2.792, 3.028] | [0.968, 1.024] | DECIDED | 0 |
| 128 | 71.51 ms | 10.05 ms | **7.113x** | [7.027, 8.313] | [0.980, 1.020] | DECIDED | 0 |
| 256 | 118.22 ms | 5.20 ms | **22.721x** | [21.964, 22.797] | [0.985, 1.019] | DECIDED | 0 |
| 512 | 1275.41 ms | 11.65 ms | **109.367x** | [107.490, 112.762] | [0.990, 1.007] | DECIDED | 0 |

Ratio grows as O(n³/n) exactly as the mechanism predicts; `.165`'s 17-19x sits inside this curve between n=128 and
n=256 — the rejected entry was not merely decidable, it was CONSERVATIVE. Bit-identity proven BEFORE every timing
(15,352-335,062 result fields compared as raw bits, plus `nfev`, `njev`, `nlu`, `status`, `success`, `message`);
`hits_cand == nlu` and `hits_base == 0` on every run, so the arm switch is proven to have taken. Artifact:
`tests/artifacts/perf/2026-07-25-bdf-diag-newton/bench_stdout_stderr.txt`. Bead `frankenscipy-43vfn`.
**METHOD NOTE — the ELF-sha rule paid for itself inside one session.** The first two candidate binaries ran the
`O(n²)` "is J diagonal" scan once per FACTORIZATION (`nlu` = 127 at n=512) instead of once per JACOBIAN
(`njev` = 1). They measured 45.82x and 14.80x at n=512 — both large, both DECIDED, both quietly leaving 2.4x and
7.4x on the table. The regression only surfaced because §2.1 forces the artifact's self-reported ELF sha to match
the binary built from the code that ships, which forced a re-run after a refactor that "obviously could not change
performance". Caching the scan on the solver (`jac_diagonal`, exactly as `RadauSolver` already did) is what turns
45.82x into **109.37x**. A shell-side hash next to the run would have hidden this.
**RETRY/EXTEND VEIN (next levers, in EV order):** (0) the same `njev`-vs-`nlu` cache split anywhere else a
structural predicate sits on a per-factorization path; (1) exactly BANDED Jacobians — same predicate shape, banded LU
instead of dense, the classic method-of-lines PDE case; (2) BLOCK-diagonal (decoupled subsystems) via a
union-find on the sparsity pattern; (3) the same structural test in `radau.rs`, which has the identical unconditional
dense-LU shape. Once the LU is gone the profile is dominated by the RHS closure's per-call `Vec` allocation
(nfev ≈ 1.8-2.5k per solve) — that is the next frame, and it is a different lever.

### 2026-07-25 (cc/CopperFalcon) — DESIGN + IDENTITY OBLIGATIONS (not yet built): exactly-BANDED Newton factorization for BDF
Follow-on to the diagonal lever above, same frame (dense `LU::new` = 67.44% self at n=256). **Not measured, not
claimed** — recorded because the hard part (the bit-identity obligation against `nalgebra` 0.35's exact GEPP
arithmetic) is now worked out, so whoever picks it up does not have to re-derive it. Bead `frankenscipy-3u0cb`.
**TARGET CLASS.** Method-of-lines discretizations of 1-D/2-D PDEs — the dominant real-world stiff-ODE workload,
and the reason scipy exposes `lband`/`uband`/`jac_sparsity` at all. Our BDF has no sparsity option whatsoever, so
a tridiagonal n=512 heat equation pays a full 45 Mflop dense factorization where 1.5 Kflop suffices; the
factorization frame shrinks by ~O(n²/(kl·ku)), i.e. four orders of magnitude of arithmetic at n=512, kl=ku=1.
**WHY IT CAN BE BIT-IDENTICAL.** Dense GEPP on a banded matrix performs the same arithmetic on in-band entries
and no-ops on the structural zeros (`a + (−p)·0.0 == a` for finite `a`; `±0.0` sign flips on entries that stay
zero are unobservable, since the only later use is `icamax`, which compares magnitudes). Pivot rows stay inside
`[j, j+kl]` because every candidate below the band is exactly zero, and `U` fills to upper bandwidth `ku+kl`.
**THE FOUR OBLIGATIONS (read `nalgebra-0.35.0/src/linalg/lu.rs` before writing a line).**
1. `gauss_step` computes `inv_diag = 1/diag` ONCE and then `coeffs *= inv_diag` — a multiply by the reciprocal,
   **not** a division. A banded rewrite that divides is NOT bit-identical.
2. The trailing update is `down.column_mut(k).axpy(-pivot_row[k], &coeffs, 1.0)`, i.e. `a[i][k] +=
   (−u[k])·l[i]` in COLUMN-major order over `k`. Reproduce the operand order and the loop order; Rust keeps
   `fp-contract=off` so there is no FMA to match (see `.cargo/config.toml`).
3. The pivot is `matrix.view_range(i.., i).icamax() + i` — FIRST index attaining the max magnitude. Ties must
   break to the lowest row index.
4. If the pivot value `is_zero()`, nalgebra **`continue`s** — it does not fail, it leaves the column untouched
   and moves on. A banded version that errors on a zero pivot changes behaviour on singular systems.
Also required: banded forms of `solve_lower_triangular_unchecked_mut` (unit diagonal) and
`solve_upper_triangular_mut`, matching their axpy formulations the same way.
**GATE.** Take the banded path only when `kl + ku + 1` is small relative to `n` (proposed: `(kl+ku+1)*3 <= n`,
so the fill-to-`ku+kl` band still beats dense); detect `(kl, ku)` in the SAME per-Jacobian scan that already
computes `jac_diagonal` (cadence rule — see the 2026-07-25 diagonal row: putting an O(n²) scan on the
per-factorization path cost 7.4x there), and cache both on the solver.
**PRE-REGISTERED KILL CONDITION.** If a banded candidate cannot be made bit-identical against the dense arm on
the existing three-shape test at n∈{32,128,512} — specifically if `bitmism != 0` on a tridiagonal fixture — do
NOT ship it behind a tolerance contract. Stiff-ODE step/order control is driven by error norms; a factorization
that changes bits changes the step sequence, which changes `nfev`/`nlu` and therefore the public
`SolveIvpResult`. Ledger the failure and stop; the fallback is a user-facing `jac_sparsity`-style opt-in, which
is a different (API) change with a different review.

### 2026-07-25 (cc/CopperFalcon) — KEEP / **SELF-SPEEDUP** (BIT-IDENTICAL): BDF exactly-BANDED Newton factorization — 1.75x @n=64 → 9.39x @n=256 (5.85x @n=512, saturating; 18.54x after the fused-construction row below)
**Result class:** `SELF-SPEEDUP`.
Executes the DESIGN row above (`frankenscipy-3u0cb`). Same frame as the diagonal lever: on the tridiagonal fixture
`LU::new` is **83.42% self** (perf, n=256, `--call-graph=dwarf`), plus `solve_mut` 4.54% and
`solve_upper_triangular_mut` 3.15% — **91.1%** of the profile in the three frames this lever collapses.
**Result class: SELF-SPEEDUP.** Both arms are OUR code (`BDF_FORCE_DENSE_NEWTON` on vs off); no SciPy
arm was measured, so this ratio is MAINTENANCE and must not be quoted as a competitive claim. Live-arm
work: `frankenscipy-0bu5p`.
**TARGET CLASS.** 1-D heat equation by method of lines, `y_j' = k(y_{j-1} − 2y_j + y_{j+1})` — the canonical stiff
PDE discretisation and the reason scipy exposes `lband`/`uband`/`jac_sparsity`. Our BDF had no sparsity option at
all, so a tridiagonal n=512 system paid a full dense factorization per `nlu`.
**LEVER (one).** `NewtonFactor::Banded`: `jacobian_bandwidth` (cached per `njev`, next to `jac_diagonal` — the
cadence rule) reports `(kl, ku)` when `3(kl+ku+1) <= n`; `BandedLu::factor` then runs Gaussian elimination
restricted to the band, in the same dense `n×n` layout `nalgebra` uses.
**THE DESIGN CHANGED ONCE, BECAUSE THE TEST REFUTED THE FIRST ONE.** The first implementation clipped the pivot
search and the `L`-column swaps to the band. `banded_lu_is_bit_identical_to_dense_lu` — written specifically to
reach the pivoting branch that a diagonally dominant PDE fixture never exercises — produced factor mismatches at
`(2,0)`, `(7,0..4)`, then `(8,0)`, `(9,2)`, `(15,0)`: under partial pivoting a multiplier migrates DOWN its column
by one row per interchange without bound, and an active row carries its column extent with it, so after
interchanges neither `L` nor the active region is banded and dense GEPP's pivot search legitimately reaches
outside the nominal band. **No band-clipped factorization can reproduce dense GEPP once it pivots** — this is why
LAPACK's `gbtrf` stores multipliers separately and produces a DIFFERENT `L` and permutation. Had the fixture been
the only test, this ships silently wrong on any non-dominant banded system.
**THE PRECONDITION IS NOW CHECKED, NOT ASSUMED.** `band_column_diagonally_dominant` verifies strict column
diagonal dominance ONCE on the untouched `I − c·J` (O(n·band)); elimination preserves it, so no step can
interchange. `BandedLu::factor` additionally returns `None` the instant an in-band maximum is off-diagonal, and
the caller falls back to the dense LU. The target class satisfies dominance by construction (`c > 0`, `J`'s
diagonal negative ⇒ `|1 − c·J_jj| > Σ|c·J_rj|`). **Nothing is claimed for matrices that would pivot.**
**BIT-IDENTICAL** by transcription of `nalgebra-0.35.0`: `inv_diag = 1/diag` once then multiply (not divide);
trailing update `a[r][k] = (−u[k])·l[r] + a[r][k]` column-major (`axpy`); zero pivot ⇒ `continue`, not error;
`fp-contract=off` so no FMA to match. Skipped work is provably a no-op (`a + (±0.0) == a` for finite `a`).
Proven by two tests: `banded_lu_is_bit_identical_to_dense_lu` compares FACTORS and SOLUTION bit-for-bit against
`nalgebra`'s dense LU across 5 shapes × 3 matrix kinds (including pivot-forcing and zero-diagonal cases, which it
asserts are declined), and `bdf_banded_newton_is_bit_identical_to_dense_lu` compares full trajectories + counters
end-to-end with `BDF_BAND_NEWTON_HITS` as the execution proof. fsci-integrate **275/0** lib.
**MEASURED** (`perf_bdf_diag_newton … tri`, §2 contract, self-reported
`elf_sha256=2e758006e4ff043a4bbf9dad098efbbd341d4f963cc44e2636360fa141e9e1c1` == shell sha; thinkstation1,
`taskset -c 2`, load 3-5, built remotely on hz2 from the fmt+clippy-clean shipped source):

| n | base p50 | cand p50 | **ratio_p50** | cand ci95 | A/A null ci95 | gate | bitmism |
|---:|---:|---:|---:|---|---|---|---:|
| 64 | 8.45 ms | 4.82 ms | **1.754x** | [1.738, 1.775] | [0.993, 1.012] | DECIDED | 0 |
| 128 | 22.35 ms | 5.95 ms | **3.765x** | [3.690, 3.857] | [0.969, 1.011] | DECIDED | 0 |
| 256 | 82.00 ms | 8.70 ms | **9.392x** | [9.323, 9.504] | [0.984, 1.006] | DECIDED | 0 |
| 512 | 914.61 ms | 156.57 ms | **5.845x** | [5.824, 5.896] | [0.984, 1.006] | DECIDED | 0 |
`hits_band == nlu` and `hits_base == 0` on every row. The shipped diagonal lever is a clean NEGATIVE CONTROL on
this fixture: with the banded path disabled it declines (`hits_cand=0`) and measures **0.996x IN-FLOOR**, i.e.
the structural test costs nothing on systems it does not accelerate.
Cross-check in the same invocation: the DIAGONAL fixture at n=256 measures **23.156x** [22.815, 23.378] with
`hits_band=0` — i.e. the two structural paths do not interfere and the shipped diagonal lever reproduces.
**HONEST LIMIT — the ratio SATURATES and then falls (9.39x @n=256 → 5.85x @n=512). [SUPERSEDED same day by the
fused-construction row below: 11.762x @n=256, 18.544x @n=512, saturation eliminated.]** The banded path still
materialises the full `n×n` `I − c·J` per factorization (`identity`, `jac.scale(c)`, and the subtraction each
allocate an `n²` temporary) and `BandedLu` stores its factors densely, so ~6 MB of memory traffic per `nlu` at
n=512 × 96 factorizations. That is the RESIDUAL FRAME, and it is `O(n²)` where the useful work is `O(n·band)`.
**NEXT LEVER (pre-registered):** LAPACK-style band storage `(kl+ku+1)×n` — 1,536 elements instead of 262,144 at
n=512/kl=ku=1, ~170x less memory traffic — built directly from `jac` without ever forming the dense system. The
arithmetic and therefore bit-identity are unchanged; only addressing moves. **RETRY PREDICATE: do not start it
until an ARM-ISOLATED profile (candidate arm only, which this bench cannot currently produce — both arms run in
one invocation and the dense arm dominates the samples) attributes >30% self-time to the system construction /
allocation frames at n=512.** The saturation above is measured; the mechanism is so far a hypothesis.

### 2026-07-25 (cc/CopperFalcon) — KEEP / **SELF-SPEEDUP** (BIT-IDENTICAL): fuse the `I − c·J` construction — 3.63x @n=512 banded; lifts the banded lever 5.85x → 18.54x
**Result class:** `SELF-SPEEDUP`.
**Result class: SELF-SPEEDUP.** Both arms are OUR code (`BDF_FORCE_DENSE_NEWTON` on vs off); no SciPy
arm was measured, so this ratio is MAINTENANCE and must not be quoted as a competitive claim. Live-arm
work: `frankenscipy-0bu5p`.
**PROFILE-ATTRIBUTED, and the attribution is the story.** The banded row above closed with a measured saturation
(9.39x @n=256 → 5.85x @n=512) and a HYPOTHESISED mechanism, with a pre-registered retry predicate: do not start
until an ARM-ISOLATED profile attributes >30% self-time to the construction/allocation frames. That predicate is
now discharged, and the answer was bigger than the hypothesis.
- Added an arm-isolated mode to `perf_bdf_diag_newton` (`… <fixture> cand|base`) — the paired A/B cannot be
  profiled, because both arms live in one process and the slower one swamps the samples. That is precisely how a
  residual frame stays invisible.
- Arm-isolated profile, banded candidate, n=512: **75.43% of samples in `[unknown]` (kernel)**, 19.25% in the
  binary, 5.31% in libc; `__memset_avx2` 4.90%; top named frame `bdf_step_impl` only 13.85%.
- `perf stat -e minor-faults`, candidate arm, 3 solves: n=128 **693**, n=256 **2,569**, n=512 **246,581**. A 96x
  jump for a 2x increase in n is not the n² the arithmetic predicts — it is glibc crossing the mmap threshold.
  `identity(n,n) - jac.scale(c)` materialises THREE n² temporaries (identity, scaled copy, difference); at n=512
  each is 2 MB, so every one of the 96 factorizations mmap'd, faulted in, and munmap'd ~6 MB.
**ONE LEVER:** build the Newton matrix with a single `DMatrix::from_fn(n, n, |r, c| unit − c·jac[(r,c)])` — one
allocation and one traversal instead of three. **BIT-IDENTICAL by inspection** (same per-entry expression; IEEE
multiplication is commutative so `jac.scale(c)`'s operand order is moot) and by gate: the bench compares full
`SolveIvpResult` bits across the two constructions before timing (`sysbuild exactness: bitmism=0`, 100,046 and
226,306 and 335,062 fields). Minor faults at n=512: **246,581 → 45,748**.
**MEASURED** (same-binary paired A/B on a NEW `sysbuild` axis — the construction is common to both structural
paths, so it cannot be measured on the `newton` axis; `elf_sha256=b7475275524597fe99a3051e2c3df6cfc96b871f6fbc13c4afb4c58a5b485318` self-reported == shell sha;
thinkstation1, `taskset -c 2`):

| fixture | n | base p50 | cand p50 | **ratio_p50** | cand ci95 | A/A null ci95 | gate |
|---|---:|---:|---:|---:|---|---|---|
| tri (banded path) | 512 | 155.61 ms | 42.76 ms | **3.631x** | [3.603, 3.668] | [0.989, 1.010] | DECIDED |
| tri (banded path) | 256 | 7.86 ms | 6.73 ms | **1.168x** | [1.160, 1.195] | [0.979, 1.026] | DECIDED |
| diag (diagonal path) | 512 | 11.03 ms | 11.06 ms | 1.006x | [0.987, 1.026] | [0.992, 1.002] | **IN-FLOOR** |
The diagonal row is the NEGATIVE CONTROL and it is exactly right: the diagonal fast path never constructs the
system matrix at all, so fusing that construction must do nothing there — and it measures nothing.
**CONSEQUENCE — the banded lever's headline changes.** Re-measured on the `newton` axis with the fusion in:
**11.762x @n=256** (was 9.392x) and **18.544x @n=512** (was 5.845x, ci95 [18.323, 18.959], null [0.994, 1.007]).
The saturation is GONE and the curve grows with `n` as the mechanism predicts. Band storage — the lever this was
supposed to be a stepping stone toward — is no longer obviously worth it: the residual 45,748 faults are the ONE
remaining n² allocation (the matrix `LU`/`BandedLu` consumes) plus the RHS closure's per-call `Vec`.
**HONEST DEBIT.** The fusion speeds up the DENSE arm too, so the diagonal lever's published ratio moves
**109.37x → 97.68x** at n=512 (base 1275 → 1070 ms, candidate 11.65 → 10.93 ms). Both arms got faster; the ratio
got smaller. Recording the smaller number: the product improved and the headline shrank, and pretending otherwise
would be exactly the ledger rot this campaign exists to fix.
**RETRY PREDICATE for band storage / further allocation work:** only if an arm-isolated profile at n ≥ 1024 shows
the remaining single n² allocation above 20% self-time, AND `minor-faults` still scales superlinearly in `n`.

### 2026-07-26 (cc/CopperFalcon) — PRIMITIVE ADAPTATION (design, no measurement): data-movement minimization, consumed from franken_numpy
Campaign Primitive Transfer Bus. **frankenscipy is the CONSUMER; franken_numpy OWNS** the
communication-avoiding / data-movement-minimizing primitive (their elementwise-fusion gap is the higher-EV
instance). Recorded here so the adaptation is not re-derived. Bead `frankenscipy-mhe28`. Lane B — nothing below
is a new measurement; the numbers cited were taken before the allocation switch.
**TAKEN, NOT RE-DERIVED.** (1) **GEMM tile geometry is SETTLED at MR4×NR8 on AVX2** — six alternatives lose
2.3-39%, an explicit `std::simd` register tile changed nothing. Independently corroborated here: we already
ship MR4×NR8 in the trailing-SYRK micro-kernel (`23355d1c5`) and the blocked panel TRSM (`c7e9062bf`). Two
codebases, same optimum. **Do not re-measure tile geometry.** (2) **TSQR: the wall is the API surface, not the
kernel** — their reject is definitive and gets RELATIVELY WORSE as m grows (0.87× at 2e6×8), root-caused to
per-call O(mn) costs (whole-matrix cast copy + a second O(mn) residual pass + output construction). A
kernel-only speedup provably cannot close a surface-cost wall.
**TRANSFER GOING THE OTHER WAY (we owe them this).** Their MR4×NR8 inner loop is deliberately NON-FMA because
`rect_*_golden_sha256` bit-locks the summation order; they attribute a 4-7× residual to "FMA + hand-tuned asm"
and cannot test the FMA half without breaking the golden. **We are not bit-locked there** — we ship the same
tile with FMA under a 1e-10 factor-uniqueness contract, and measured it: **1.143×** (SYRK) and **1.115×**
(TRSM). So the FMA half of their 4-7× is worth ~1.14× on this hardware, and the rest is elsewhere. Posted to
the primitives thread.
**OUR SHAPE OF THE PRIMITIVE.** Theirs is "don't materialize elementwise temporaries". Ours is the same disease
at the boundaries of dense kernels, and the instance is already proven, not hypothetical: `cf91d1c59` fused
`identity(n,n) − jac.scale(c)` from THREE n² temporaries to one `from_fn` traversal — **3.63×** on the banded
BDF path, and it lifted a different lever from 5.85× to **18.54×** because that lever's measured saturation
*was* this allocation churn. At n=512 each temporary is 2 MB, above glibc's mmap threshold, so every
factorization mmap'd/faulted/munmap'd ~6 MB.
**DIAGNOSTIC WORTH KEEPING (cheaper than a profile).** `perf stat -e minor-faults` on the candidate arm across
two sizes: n=128 **693**, n=256 **2,569**, n=512 **246,581**. A 96× jump for 2× n is not the n² the arithmetic
predicts — that superlinear knee IS the mmap threshold, in one command. The arm-isolated profile agreed (75%
of samples in the kernel) and took far longer to get. **Run `minor-faults` across two sizes before believing an
O(n³) frame is the wall.**
**QUEUED — our TSQR analogue.** `cholesky()` performs an 8 MB output-buffer zero at n=1000 that is purely
CONTRACTUAL (it copies full rows relying on "strict upper triangle is already zero"), measured at **8.41%
self** and correctly rejected as sub-floor IN ISOLATION (2026-07-22). franken_numpy's TSQR finding is the
argument for revisiting it as a BUNDLE: a surface-contract cost is invisible to kernel levers by construction,
and only an API change reaches it. Bead `frankenscipy-mhe28`.
**WHAT "CA" DOES NOT MEAN HERE — pre-empting the obvious misread.** The parallel-reduction routes are already
refuted in this repo and are NOT reopened by this primitive: WY-blocked symmetric tridiagonalization **0.5-0.81×
SLOWER**; parallel eigh tridiagonalization **2.3-2.8× slower, spawn-bound**; cholesky pack-fusion
**inexpressible in safe Rust**; jc-blocked SYRK sweep in-floor at n=1000 AND n=2048. The rayon grant is for
work-gated persistent-pool dispatch, NOT for parallelizing more reductions. Combined with our own SYRK row
("the tile is near FMA-port-bound — further wins must come from reducing NON-tile time: tails, in-panel
factorization, the serial TRSM phase"), **the dense 2-3× vs OpenBLAS is Amdahl-side and surface-side, not
kernel-side.** For us the primitive reads: *stop materializing and stop copying at the boundaries*, not *tile
better*.

### 2026-07-27 — CLASSIFICATION CORRECTION (policy): the three BDF rows above are SELF-SPEEDUPS, not competitive wins
Campaign policy 2026-07-27: **a self-speedup — our own code before vs after — is MAINTENANCE and is not campaign
output. A campaign win requires a measured ratio against the actual legacy incumbent, from a harness that runs
the incumbent side-by-side IN THE SAME INVOCATION.** Across the campaign, ~369 commits produced ~60
self-speedups but only 3 vs-incumbent wins, and all 3 came from repos with a live incumbent arm.
All three BDF rows above compare `BDF_FORCE_DENSE_NEWTON` on vs off — **our dense-LU path against our own
structured path**. No SciPy arm was measured. Their ratios (1.91x→109.37x, 1.75x→18.54x, 3.63x) are therefore
**maintenance figures and MUST NOT be quoted as competitive claims**, here or anywhere else.
The rows do contain a *reasoned* argument that the win extends to SciPy — "scipy does NOT exploit this; it always
`lu_factor`s the dense system absent a user-supplied `jac_sparsity`" — which is true of scipy's source but is an
ARGUMENT, not a measurement. Under the policy it stays an argument until a live SciPy arm says otherwise.
Being explicit about the direction of the unknown: a stiff solve here makes ~1,800-2,500 RHS calls, and SciPy's
RHS is a Python callback while ours is an inlined Rust closure. A naive end-to-end vs-SciPy ratio would therefore
be substantially **callback overhead, not solver quality** — that is trap 6 (shared/asymmetric component) in the
2026-07-27 trap list, and the incoming SciPy-arm harness must decompose it rather than bank it.
Live-arm work tracked by `frankenscipy-0bu5p`.

### 2026-07-27 (cod/BlackThrush + cc/CopperFalcon) — KEEP: live SciPy BDF exact-diagonal incumbent is 36.1770x slower at n=512
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-0bu5p`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** with FrankenSciPy through a persistent Python co-process. The exact fixture behind
the earlier 1.9x-at-n=32 to 109x-at-n=512 SELF-SPEEDUP claim ran at `n=512`:
`y'_i = -(1 + 10i)y_i`, `t_span=[0,1]`, `y0_i=1+0.25(i mod 7)`, `method=BDF`, `rtol=1e-8`,
`atol=1e-10`, `t_eval=None`, and `jac=None` in both arms.

- **Incumbent ratio: SciPy / FrankenSciPy = 36.1770x.** Median times were 504.283617 ms versus
  14.019592 ms per solve. The bootstrap-median CI was `[35.4940, 36.5863]`.
- Same-invocation A/A null controls: FrankenSciPy median `1.006998`, CI
  `[0.991754, 1.053378]`; SciPy median `0.987192`, CI `[0.969113, 1.013278]`.
  The worse null edge required `1.1068x`; the incumbent CI lower bound was `35.4940x`, so the
  **median-CI gate decided a FrankenSciPy win**. Ratio CV `10.044%` is provenance only and is not a gate.
- Dispatch proof: installed SciPy `1.17.1`,
  `solve_ivp.__module__=scipy.integrate._ivp.ivp`, and no `fsci`/`franken` module loaded in the child.
  The harness aborts on any other identity.
- Matched-result proof before timing: all 512 final-state components were compared; max absolute difference was
  `6.106e-16`. Both arms succeeded with status 0 and 653 stored steps. FrankenSciPy recorded
  `nfev=2472, njev=1, nlu=127`; SciPy recorded `nfev=1308, njev=1, nlu=113`.
- Mechanism execution proof: FrankenSciPy's candidate produced `diag_hits=127`, `band_hits=0`.
  SciPy made 1,821 actual RHS calls, including its finite-difference Jacobian calls.
- Callback decomposition: the 1,821 Python RHS evaluations cost 1.9213 ms, only 0.4% of SciPy's
  504.2836 ms solve. The large end-to-end result therefore survives removing the callback asymmetry; this
  harness does not, by itself, claim that dense LU alone accounts for the full ratio.
- The incumbent and candidate were interleaved per round with alternating order; each arm also received its own
  independently paired A/A control, whose pair order alternated. Pair-group order rotated across rounds.
  The Python child timed only `solve_ivp`; pipe I/O and result parsing were outside the timed region.
- Executed-binary ELF SHA-256:
  `d7579e922d4f7c64d8876fe29f7622884daff5191a18a7e06d3cc10e05caa022`
  (self-reported and matched the shell hash). Pinned local execution used `taskset -c 25`; the release binary
  was built strict-remote on `ovh-a`.

Artifact:
`tests/artifacts/perf/2026-07-27-bdf-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not turn this end-to-end result into an LU-only claim unless the same
same-invocation, dual-null harness is run on a coupled non-diagonal fixture and either (a) the ratio collapses
outside the exact-diagonal CI, isolating structure, or (b) counted/profile evidence attributes at least 80% of
SciPy's solve time to dense factorization. Extend to Radau only with the same full-vector parity, runtime
incumbent identity, child-side solve timing, CPU pin, and dual-null median-CI gate.

### 2026-07-28 (cc/CopperFalcon) — CAMPAIGN-WIN **and CAMPAIGN-LOSS**: the vs-SciPy BDF win is ENTIRELY structural; with no structure we are 2.12x SLOWER
**Result class: CAMPAIGN-WIN** (diagonal, coupled) **and CAMPAIGN-LOSS** (dense). Legacy incumbent: genuine
`scipy.integrate.solve_ivp(method='BDF')`, scipy 1.17.1, side-by-side in the SAME invocation, interleaved with
alternating order, dual A/A nulls, per-fixture execution proof. Bead `frankenscipy-0bu5p`.
Executed-binary `elf_sha256=14d77443051075e985f5abe384ad3f23aaa0f22782c7480223d2707cf114f0cd` (self-reported by the binary); full output:
`tests/artifacts/perf/2026-07-28-bdf-vs-scipy-structure-vs-implementation/bench_stdout_stderr.txt`.
A/A nulls, both arms, all three fixtures: ours 1.001-1.012, SciPy 0.984-1.017; decided on the
bootstrap-median CI, cv is provenance only.
This is the FALSIFYING EXPERIMENT that the 2026-07-27 rows named as required before the diagonal number is
quoted broadly. It was worth running: **it confirms the structural attribution and refutes the reading that our
BDF is simply faster than SciPy's.**
Same harness, same `n=512`, same `rtol=1e-8`/`atol=1e-10`/`t_span=[0,1]`. Only the PROBLEM changes:

| fixture | Jacobian | our path (execution proof) | FrankenSciPy | SciPy | **Incumbent ratio: SciPy / FrankenSciPy** | ci95 | gate |
|---|---|---|---:|---:|---:|---|---|
| `diagonal` | exactly diagonal | `diag_hits=127`, `band_hits=0` | 13.61 ms | 522.23 ms | **38.80x** | [37.23, 40.24] | DECIDED **WIN** |
| `coupled` | tridiagonal | `band_hits=125`, `diag_hits=0` | 43.05 ms | 514.85 ms | **11.91x** | [11.67, 12.18] | DECIDED **WIN** |
| `dense` | dense (`J_ij != 0` ∀ i,j) | **neither path fires** | 1086.69 ms | 513.49 ms | **0.47x** | [0.464, 0.479] | DECIDED **LOSS** |

**FINDING 1 — the structural argument is CONFIRMED.** The win tracks the structure exactly: 38.8x when the
diagonal path fires, 11.9x when only the banded path fires, and **0.47x when neither does**. SciPy's time barely
moves across all three (522/515/513 ms) because it does the same dense `lu_factor` every time — which is
precisely what the argument asserted. The ratio is not an artifact of language or of the callback boundary
(callbacks are 0.4%/1.6%/1.1% of SciPy's time).
**FINDING 2 — and this is the one that was invisible before: with no exploitable structure WE LOSE, 2.12x.**
`0.4725x` is a DECIDED loss against the incumbent, not noise (null edge 1.0271, required 1.0542, ci95
[0.4640, 0.4792]). Our dense BDF runs 129 LU factorizations at n=512 in 1086.69 ms ≈ 5.3 GFlop/s, against
SciPy's LAPACK `dgetrf`. That is the documented dense-linalg wall — "within ~2-3x of OpenBLAS" — showing up
end-to-end in a solver for the first time, and it is 2.12x here.
**WHAT MAY AND MAY NOT BE CLAIMED, precisely.** "FrankenSciPy BDF is 38.8x SciPy" is true only for
exactly-diagonal Jacobians; 11.9x only for banded ones. **There is no honest general "our BDF is Nx SciPy"
claim** — on a dense stiff problem the correct statement is that we are **2.12x slower**. Any headline that
omits the fixture class is wrong in the unfavourable direction for a user whose problem is dense.
**METHOD NOTE — two harness defects this experiment exposed, both now fixed.** (1) The execution proof asserted
`diag_hits > 0` unconditionally, so the coupled run aborted; it is now PER-FIXTURE and asserts the expected path
fires *and no other* — which is what caught that my first "coupled" control was tridiagonal and therefore
routed through the BANDED fast path rather than testing "no structure" at all. Without that assert the coupled
row would have been published as a structure-free control that was nothing of the kind. (2) `fixture` and the
optional script path both read `argv[4]`, so the harness spawned `python3 diagonal` and surfaced it as a
confusing "not genuine" dispatch abort; the script path moved to `argv[5]`.
**NEXT.** The dense loss is the real target now, and it is the dense-linalg wall in a new dress: `nfev` 2472 vs
SciPy's 1308 (we evaluate the RHS 1.9x more often) and `nlu` 129 vs 113. Both are step-control differences, not
kernel differences, so there may be cheaper wins in the BDF step/order logic than in the LU. Measure the split
before touching either.

### 2026-07-28 (cc/CopperFalcon + cod/BlackThrush) — KEEP: five live-SciPy ODE wins across BDF, Radau, and LSODA
**Result class: CAMPAIGN-WIN.** **Legacy incumbent arm: SciPy 1.17.1, side-by-side same-invocation** through
the persistent `scipy.integrate.solve_ivp` co-process. Each cell interleaved the actual incumbent and
FrankenSciPy, alternated order, ran an independent A/A null for both arms, checked the full final vector before
timing, and decided on the bootstrap median-CI rather than CV. The matrix uses `n=128`,
`t_span=[0,1]`, `rtol=1e-8`, `atol=1e-10`, `t_eval=None`, and `jac=None`.

| method | fixture | execution proof | FrankenSciPy | SciPy | incumbent ratio | ratio CI95 | required | gate |
|---|---|---|---:|---:|---:|---|---:|---|
| BDF | exactly diagonal | `diag_hits=114`, `band_hits=0` | 2.578933 ms | 66.931044 ms | **25.8677x** | [22.8402, 26.1314] | 1.1928 | **WIN** |
| BDF | dense | neither BDF structural path | 17.537569 ms | 71.210046 ms | **4.0480x** | [3.9231, 4.0927] | 1.0320 | **WIN** |
| Radau | exactly diagonal | no BDF path; superseded below by exact Radau counter | 8.572682 ms | 187.083000 ms | **21.6258x** | [21.0322, 23.0731] | 1.0810 | **WIN** |
| LSODA | exactly diagonal | `diag_hits=30`, `band_hits=0` | 1.328975 ms | 8.978513 ms | **6.7560x** | [6.4758, 7.1482] | 1.0621 | **WIN** |
| LSODA | dense | neither BDF structural path | 5.945440 ms | 16.390504 ms | **2.7594x** | [2.2934, 2.8945] | 1.2301 | **WIN** |

The corresponding unambiguous measurements are **Incumbent ratio: SciPy / FrankenSciPy = 25.8677x** for
BDF diagonal, `4.0480x` for BDF dense, `21.6258x` for Radau diagonal, `6.7560x` for LSODA diagonal, and
`2.7594x` for LSODA dense. All five clear their own dual-null median-CI gate. CV is provenance only.
The initial BDF/Radau/LSODA-dense cells used **Executed-binary ELF SHA-256:
`1c166cbed11117fd020c98d7c1b173a5067deb7de1b828b438baa524cd28179d`**; the corrected method-aware
LSODA-diagonal cell used **Executed-binary ELF SHA-256:
`9d58eab330986c16f5628e44571f784fa76a61a128952f7199941b46e9ef31ec`**. Both values were
self-reported by the measured binary. Raw artifact:
`tests/artifacts/perf/2026-07-28-ode-family-vs-scipy/bench_stdout_stderr.txt`.

The structural translation is real but bounded by fixture and size. With no exploitable structure, BDF is
`4.0480x` faster at n=128 yet `0.4725x` at n=512 (2.12x slower, the preceding row); neither number generalizes
without n. LSODA's dense end-to-end result includes a large incumbent callback component: callbacks were 47.9%
of SciPy's time, while the callback-free sensitivity ratio remained `1.4364x`. LSODA switches into the
FrankenSciPy BDF implementation on this stiff fixture, so its BDF hit counters are expected execution proof,
not accidental dispatch.

**Concrete retry predicate:** do not repeat these exact n=128 cells. Reopen an ODE method only for a different
size/structure crossover, or when full-result parity plus a new path counter/profile attributes a different
solver stage above the dual-null floor. Dense Radau is the separately ledgered refuted translation below.

### 2026-07-28 (cod/BlackThrush) — KEEP: exact historical Radau64 diagonal-stage fixture is 31.2543x faster than live SciPy
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-p79lo`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** on the exact workload behind the Radau diagonal-stage self-speedup:
`n=64`, rates linearly spanning 1 through 1000, all-ones `y0`, `t_span=[0,0.2]`, `method=Radau`,
`rtol=1e-6`, `atol=1e-8`, `t_eval=None`, and `jac=None`.

- **Incumbent ratio: SciPy / FrankenSciPy = 31.2543x.** Median times were 32.572677 ms versus
  1.041510 ms per solve; bootstrap-median CI `[31.0787, 31.3776]`.
- FrankenSciPy A/A median `1.000491`, CI `[0.997795, 1.002686]`; SciPy A/A median `1.001354`, CI
  `[0.997348, 1.005291]`. Worst null edge `1.0053` required `1.0106`; the ratio CI cleared it by
  about 30.8x. Ratio CV `0.560%` is provenance only.
- All 64 final components agreed within `1.400e-10`, or `0.012` tolerance units. Both arms returned success
  and status 0. FrankenSciPy recorded `nfev=1045`, `njev=1`, `nlu=280`, 141 stored steps; SciPy recorded
  `nfev=1113`, `njev=2`, `nlu=46`, 158 stored steps.
- Exact mechanism proof: `radau_diag_hits=140`, while both BDF counters stayed zero. This turn added the
  Radau counter, replacing the earlier n=128 row's weaker “not BDF” proof.
- Python RHS callbacks were 0.8870 ms, 2.7% of SciPy's solve; callback-free sensitivity remained `30.4228x`.
- **Executed-binary ELF SHA-256:
  `52d442c072f56a58ca8723669691fb517eb30ad960bf838fe203eb1beda8b511`**, self-reported and matched by
  `sha256sum`. The release binary was built strict-remote on `hz2`; measurement was pinned to CPU 25.

Artifact: `tests/artifacts/perf/2026-07-28-radau-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat the exact diagonal Radau64 competitive translation. Reopen Radau
only on a non-diagonal structured fixture, or after a profile/count proves a different stage can remove work,
with the same full-vector parity, genuine incumbent identity, per-path execution proof, and dual-null
median-CI gate.

### 2026-07-28 (cc/CopperFalcon + cod/BlackThrush) — REJECTED TRANSLATION: dense Radau n=128 is 2.39x slower than SciPy
The dense-Jacobian Radau eigen-decoupling self-speedup does **not** translate into a competitive win on this
fixture. **Incumbent ratio: SciPy / FrankenSciPy = 0.4188x**: FrankenSciPy took 469.423498 ms versus
196.213678 ms for SciPy, a decided 2.39x loss. Bootstrap-median CI `[0.4072, 0.4236]` was wholly below the
inverse `1.1142` gate.

The same invocation carried both controls: FrankenSciPy A/A median `1.000738`, CI
`[0.997419, 1.013290]`; SciPy A/A median `1.018826`, CI `[0.979597, 1.057088]`. Ratio CV `1.600%` is
provenance only. Full-result difference was `0.016` tolerance units. FrankenSciPy performed 1,178 counted
factorizations per solve versus SciPy's 74, amplifying its dense-LU disadvantage about 16x. Python callbacks
were only 6.3% of SciPy's time; removing them moves the sensitivity ratio farther against FrankenSciPy,
to `0.3915x`. **Executed-binary ELF SHA-256:
`1c166cbed11117fd020c98d7c1b173a5067deb7de1b828b438baa524cd28179d`**.

**Concrete retry predicate:** retry dense Radau only after a source/profile change reduces counted
factorizations materially toward SciPy's 74 without weakening the convergence/tolerance contract; require the
same full-vector parity, dual A/A controls, and median-CI decision. Do not retry dense-kernel tuning alone while
the 16x factorization-count gap remains.

### 2026-07-28 (cod/BlackThrush) — KEEP: live SciPy sparse CG is 1.2226x slower at side=80
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-qsxhs`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** through a persistent
`scipy.sparse.linalg._isolve.iterative.cg` co-process. Both arms received the identical canonical CSR for the
side-80 Dirichlet five-point Laplacian (`n=6,400`, `nnz=31,680`, diagonal `4.001`), identical deterministic RHS,
zero initial guess, `rtol=1e-5`, `atol=0`, and `maxiter=64,000`.

- **Incumbent ratio: SciPy / FrankenSciPy = 1.2226x.** Median times were 4.378900 ms versus
  3.581472 ms per solve; bootstrap-median CI `[1.2117, 1.2271]`.
- FrankenSciPy A/A median `1.000962`, CI `[0.996954, 1.006708]`; SciPy A/A median `0.994363`, CI
  `[0.981332, 1.010639]`. Worst null edge `1.0190` required `1.0380`; the median-CI gate decided a
  FrankenSciPy win. Ratio CV `2.712%` is provenance only.
- Full-vector proof: all 6,400 components were compared, max absolute difference `2.387e-12` and relative L2
  difference `4.326e-15`. Both true relative residuals were `9.478e-6`.
- Counted execution proof: both arms converged in exactly 115 CG iterations. Matrix construction, Python
  callback instrumentation, pipe I/O, and vector serialization were outside timing; timed SciPy runs used no
  callback.
- **Executed-binary ELF SHA-256:
  `39169787e692ce8a53d28511cfafc4d91d8bf7745b5523aabcba83735340a1aa`**, self-reported and matched by
  `sha256sum`. The release binary was built strict-remote on `vmi1153651`; measurement was pinned to CPU 25.

Artifact: `tests/artifacts/perf/2026-07-28-sparse-cg-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not generalize this side-80 ratio to larger systems. Reopen this exact cell only
after a source/profile change removes counted work while preserving the 115-iteration trajectory; use the same
canonical CSR, full-vector/residual proof, dual nulls, and median-CI gate.

### 2026-07-28 (cod/BlackThrush) — REJECTED TRANSLATION: live SciPy sparse CG wins by 1.0126x at side=120
The earlier separate-invocation gap hunt reported FrankenSciPy `1.59x` faster at side 120. That estimate does
**not** survive the admissible harness. **Incumbent ratio: SciPy / FrankenSciPy = 0.9876x**:
FrankenSciPy took 11.890901 ms versus SciPy's 11.743986 ms, a decided 1.0126x loss. Bootstrap-median CI
`[0.9862, 0.9910]` was wholly below the inverse `1.0083` keep gate.

This is not mismatched convergence. Both arms used the identical `n=14,400`, `nnz=71,520` canonical CSR and
converged in exactly 170 counted iterations. All 14,400 result components were checked: max absolute difference
`3.706e-11`, relative L2 difference `4.800e-14`, and both true relative residuals `9.159e-6`.
Same-invocation controls were FrankenSciPy A/A null median `0.997564`, CI
`[0.996529, 0.999769]`, and SciPy A/A null median `1.002256`, CI
`[0.999441, 1.004174]`. Ratio CV `0.422%` is provenance only.
**Executed-binary ELF SHA-256:
`39169787e692ce8a53d28511cfafc4d91d8bf7745b5523aabcba83735340a1aa`**.

The translation therefore crosses over between side 80 and side 120: fixed Python iteration overhead wins at
the smaller problem, while the previously ledgered 1.2-1.4x SciPy SpMV throughput advantage catches up as each
iteration grows. This sentence is an inference from the prior SpMV row plus the identical iteration counts, not
a new profile attribution.

**Concrete retry predicate:** retry side-120 CG only after a profile attributes a removable top-five cost in
FrankenSciPy's SpMV or vector-update loop and a counted mechanism proves less work; require the same
170-iteration/full-vector trajectory, dual A/A controls, and median-CI decision. Do not repeat the old
separate-invocation timing.

### 2026-07-28 (cod/BlackThrush) — KEEP: live SciPy PCHIP is 3.8615x slower on the historical sorted batch
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-bdlfm`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** through a persistent genuine
`scipy.interpolate._cubic.PchipInterpolator` co-process. The arm reuses the exact historical cursor fixture:
1,024 knots, the same deterministic `sin(11x) + 0.25 cos(7x)` values, and 4,096 finite nondecreasing queries.
Rust transmits all three vectors to SciPy before construction, and both interpolators are constructed outside
timing; the timed arms evaluate only their already-built interpolator over the shared query vector.

- **Incumbent ratio: SciPy / FrankenSciPy = 3.8615x.** SciPy took 0.040529 ms per evaluation versus
  FrankenSciPy's 0.010490 ms; the bootstrap-median CI was `[3.8439, 3.9078]`.
- FrankenSciPy A/A null median was `1.002547`, CI `[0.999649, 1.004516]`; SciPy A/A null median was
  `1.010599`, CI `[0.999468, 1.020565]`. Worst null edge `1.0206` required `1.0411`; the median-CI gate
  decided a FrankenSciPy win. Ratio CV `1.977%` is provenance only.
- Full-vector SciPy conformance covered all 4,096 components. Maximum absolute difference was `2.220e-16`
  against the existing `1.0e-11` PCHIP differential tolerance, with zero mismatches.
- This confirms the competitive translation of the previously kept `2.6187x` serial sorted-query cursor:
  removing per-query interval searches survives comparison with the actual SciPy incumbent. It does not claim
  that the self-ratio and incumbent ratio should be numerically equal.
- **Executed-binary ELF SHA-256:
  `8a5dd4da2c5030411c636efa5a1e8e93a65d157de61d4d67450f3a9818fb05cd`**, self-reported and matched by
  `sha256sum`. The bench-profile binary was built strict-remote on `hz2`; measurement was pinned to CPU 25.

Artifact: `tests/artifacts/perf/2026-07-28-pchip-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat this 1,024-knot/4,096-query finite sorted serial cell. Reopen PCHIP
translation only for a distinct production regime—unsorted or non-finite queries, a batch large enough to take
the parallel path, or a changed SciPy incumbent—and retain exact shared inputs, full-vector tolerance proof,
genuine dispatch identity, dual A/A nulls, ELF SHA-256, and the bootstrap-median CI gate.

### 2026-07-28 (cod/BlackThrush) — KEEP: live SciPy `kv(1.5)` is 4.2547x slower on 2M points
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-tacmw`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** through a persistent genuine `scipy.special.kv` ufunc co-process. Rust
constructed the exact two-million-point finite positive sorted grid over `[0.3, 50]`, transferred its binary
f64 representation to SciPy, and verified the shared input SHA-256. The timed arms evaluated only `kv(1.5, z)`
over that prepared grid.

- **Incumbent ratio: SciPy / FrankenSciPy = 4.2547x.** SciPy took 99.665018 ms versus FrankenSciPy's
  23.430024 ms; the bootstrap-median CI was `[4.2402, 4.2612]`.
- FrankenSciPy A/A null median was `1.001003`, CI `[0.999722, 1.002227]`; SciPy A/A null median was
  `0.999171`, CI `[0.997979, 1.002805]`. Worst null edge `1.0028` required `1.0056`; the median-CI gate
  decided a FrankenSciPy win. Ratio CV `0.459%` is provenance only.
- Full-vector conformance covered all 2,000,000 components: maximum absolute difference `4.441e-15`,
  maximum relative difference `7.917e-16`, and zero mismatches under `1e-13 + 1e-12 * |expected|`.
- This validates the structural translation: the shipped half-integer closed form avoids the general
  fixed-quadrature path and remains faster than the actual SciPy incumbent. The historical `37x` figure is a
  **SELF-SPEEDUP**, and its separately timed projected `19x` incumbent result was not admissible competitive
  evidence. This live-arm `4.2547x` result supersedes that projection as the competitive claim.
- **Executed-binary ELF SHA-256:
  `f18d240c50043d92f883ecb4e0f4d2cfb66732b4dd361b8e3e5185c1a57abae6`**, self-reported and matched by
  `sha256sum` before and after artifact transfer. The bench ELF was built strict-remote on `hz2`; because that
  worker lacked SciPy, the identical ELF was copied into the repo's reusable target tree and measured pinned
  to local CPU 25 beside SciPy 1.17.1. No local build occurred.

Artifact: `tests/artifacts/perf/2026-07-28-kv-half-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat this exact order-1.5/two-million-point sorted grid. Reopen only
after the half-integer dispatch or recurrence changes, the SciPy incumbent version changes, or a distinct
order/input regime is named; retain exact shared binary inputs, full-vector tolerance proof, genuine dispatch
identity, dual A/A nulls, ELF SHA-256, and the bootstrap-median CI gate.

### 2026-07-28 (cod/BlackThrush) — KEEP: live SciPy explicit solve_ivp is 42.8017x–100.7075x slower on the historical cheap-RHS fixtures
**Result class: CAMPAIGN-WIN.** Bead `frankenscipy-18x2g`. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** through the persistent genuine
`scipy.integrate._ivp.ivp.solve_ivp` co-process. This resurrects the exact
historical `rtol=1e-6`, `atol=1e-9`, `t_eval=None` fixtures: scalar exponential
decay over `[0,10]` and three-component Lorenz over `[0,1]`. The matrix covers
RK23, RK45, and DOP853. Each cell used 21 interleaved rounds and 128 solves per
timed block, alternated arm order, and ran an independent A/A null for each
implementation.

| method | fixture | FrankenSciPy | SciPy | **Incumbent ratio: SciPy / FrankenSciPy** | ratio CI95 | A/A null CI95 (ours; SciPy) | required | gate |
|---|---|---:|---:|---:|---|---|---:|---|
| RK23 | exponential | 0.058096 ms | 5.251213 ms | **90.3184x** | [89.5445, 90.5749] | [0.999182, 1.008044]; [0.994637, 1.007487] | 1.0161 | **WIN** |
| RK45 | exponential | 0.013679 ms | 1.274722 ms | **92.9833x** | [92.2892, 95.0211] | [0.995882, 1.008140]; [0.995043, 1.013741] | 1.0275 | **WIN** |
| DOP853 | exponential | 0.015834 ms | 0.675191 ms | **42.8017x** | [42.3852, 43.6580] | [0.996493, 1.020178]; [0.997090, 1.005043] | 1.0404 | **WIN** |
| RK23 | Lorenz | 0.088337 ms | 8.378065 ms | **94.5694x** | [93.8742, 95.0986] | [0.997013, 1.002081]; [0.990322, 1.005106] | 1.0195 | **WIN** |
| RK45 | Lorenz | 0.019439 ms | 1.954642 ms | **100.7075x** | [100.1412, 100.9815] | [0.994204, 1.008007]; [0.998136, 1.003540] | 1.0160 | **WIN** |
| DOP853 | Lorenz | 0.023027 ms | 1.324701 ms | **57.4691x** | [57.1673, 58.1110] | [0.997659, 1.010869]; [0.996471, 1.003696] | 1.0217 | **WIN** |

The largest cell's unambiguous measurement is **Incumbent ratio: SciPy /
FrankenSciPy = 100.7075x.** All six bootstrap-median CIs shown above clear
twice the worse distance of either same-invocation A/A control from 1.0, so the
median-CI gate returned **DECIDED FRANKENSCIPY WIN** for every cell. CV is
provenance only and is not a gate.
The harness self-reported **Executed-binary ELF SHA-256:
`46f5c261064e7bde8251b9f1444360f4d69d84facb3c6ad5fd405669e15176f9`**;
the hash matched on `hz2` before transfer and locally afterward. Measurement
was pinned to CPU 25 beside SciPy 1.17.1 and NumPy 2.4.3; no local build
occurred.

Full-result proof preceded timing. RK23 and RK45 matched SciPy's accepted-step
trajectory on both fixtures, with maximum final-vector differences at or below
`2.132e-14`. DOP853 stayed within the differential tolerance but did not match
the work trajectory: on exponential, FrankenSciPy used 375 RHS evaluations and
31 steps versus SciPy's 146 and 13; on Lorenz it used 485 and 36 versus 254 and
18. The DOP853 ratios are therefore genuine end-to-end incumbent wins, but
they are not evidence that FrankenSciPy's DOP853 algorithm removes more work.

The Python child now times the real RHS directly. Its `actual_rhs_calls`
instrumentation runs in a separate untimed solve and must reproduce the timed
final state, so instrumentation adds neither a wrapper dispatch nor a
repetition multiplier to the incumbent timing. Directly measured Python RHS
callbacks accounted for only 6.6–21.7% of SciPy's solve time; subtracting that
component left sensitivity ratios from `38.4563x` through `84.3783x`. The
competitive finding is thus the cheap-RHS execution boundary as a whole
(compiled inline Rust stages versus SciPy's Python callback and Python solver
loop), not callback cost alone and not a general claim for expensive RHS
functions.

This row supersedes the 2026-06-19 separate-invocation RK45 estimates
(`77.64x` and `72.97x`), now classified `VOID-NONULL`. Raw artifact:
`tests/artifacts/perf/2026-07-28-explicit-rk-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat these six cheap-RHS cells. Reopen
the explicit-RK competitive translation only for a materially different
boundary—an expensive or vectorized RHS, events/dense output/`t_eval`, a
changed SciPy incumbent, or a solver implementation change—and retain the
same genuine dispatch proof, full-vector tolerance check, dual A/A controls,
ELF SHA-256, and bootstrap-median CI gate. Do not claim DOP853 algorithmic
efficiency until a parity-preserving change materially reduces its counted RHS
evaluations/steps toward SciPy's trajectory and the same live arm is rerun.

### 2026-07-29 (cod/BlackThrush) — INVALID-COTENANCY: completion-only `solve_ivp_many` scaling capture has no competitive verdict
**Result class: INVALID-COTENANCY.** Bead `frankenscipy-eyr23` is reopened.
This completion-only phase-2 job is distinct from the conformance-blocked
historical 150-sample row above. It passed its scientific conformance checks,
but the scaling measurement ran after seven repositories had simultaneously
been directed to `trj`. FrankenSciPy held no exclusive `trj-booking`
CLAIM/RELEASE window, so unrelated co-tenants can explain the apparent
cross-thread curve and null width. No ratio in this row is competitive
evidence.

The live SciPy 1.17.1 arm did run side-by-side in the same invocation, and each
cell computed independent A/A controls; for example, the one-requested-thread
A/A null CIs were `[0.950739, 1.012974]` for FrankenSciPy and
`[0.967371, 1.010661]` for SciPy. Those properties do not repair host contention
outside the invocation. The raw samples are retained below solely to make the
invalidation auditable.

| requested FrankenSciPy threads | affinity / cpuset cap | FrankenSciPy p50 | SciPy p50 | raw SciPy / FrankenSciPy | raw ratio CI95 | local required from dual nulls | adjudication |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | `0` / 1 | 7.326502 ms | 875.511775 ms | 117.0976x | [101.4235, 121.9817] | 1.1036 | INVALID-COTENANCY |
| 2 | `0-1` / 2 | 4.246819 ms | 852.691965 ms | 199.6176x | [159.6092, 208.9424] | 2.1696 | INVALID-COTENANCY |
| 4 | `0-3` / 4 | 4.190473 ms | 862.337628 ms | 204.4139x | [192.5203, 209.9078] | 2.0309 | INVALID-COTENANCY |
| 8 | `0-7` / 8 | 2.799820 ms | 873.583495 ms | 312.0142x | [287.0678, 330.0609] | 2.7889 | INVALID-COTENANCY |
| 16 | `0-15` / 16 | 1.822020 ms | 863.189121 ms | 467.7619x | [383.3451, 549.4283] | 1.6058 | INVALID-COTENANCY |
| 32 | `0-31` / 32 | 2.034007 ms | 886.986270 ms | 428.7347x | [328.7766, 452.7466] | 1.7863 | INVALID-COTENANCY |
| 64 | `0-63` / 64 | 3.493338 ms | 908.181018 ms | 251.1258x | [217.3520, 276.0054] | 1.2469 | INVALID-COTENANCY |
| 128 | `0-127` / 128 | 6.316863 ms | 859.848329 ms | 135.7549x | [129.1375, 149.9015] | 1.2831 | INVALID-COTENANCY |

All 128 trajectories reached `t=10`; both complete accepted-step histories
remained finite and positive. The final comparison covered 256 components:
maximum absolute difference `6.573e-14`, maximum scaled difference below
`0.001`, and maximum Lotka invariant drift `1.212e-7` in both arms. Counted
work was nearly identical: FrankenSciPy `159,998` RHS evaluations versus
SciPy `160,126`, with exactly `25,748` stored accepted points in each arm.
Python callbacks accounted for only 16.8–18.9% of SciPy's wall time;
callback-free sensitivity ratios still ranged from `98.4309x` to `390.2856x`.

The raw shape appears flat for SciPy and peaks for FrankenSciPy at 16 requested
CPUs, but co-tenancy can create exactly that curve. It cannot identify either
incumbent contention or FrankenSciPy coordination cost. Do not derive a thread
cap from it; follow-up `frankenscipy-ldx0f` is blocked on the exclusive rerun.

Recorded provenance was insufficient: host `threadripperje`, AMD Ryzen
Threadripper PRO 5995WX, 64 physical cores / 128 logical threads, one NUMA
node; runtime ISA `AVX2=true`, `FMA=true`, `BMI2=true`, `VAES=true`,
`AVX-512F=false`; requested affinities `0`, `0-1`, `0-3`, `0-7`, `0-15`,
`0-31`, `0-63`, `0-127`. The harness inferred the FrankenSciPy worker count
from available parallelism instead of observing worker identities, omitted RAM
and booking state, and did not hash the SciPy engine artifact. The recorded
FrankenSciPy executed-binary ELF SHA-256 was
`01b60dc3ad1f0d29c561b2992c00d73caa73cc3a1aac186ca15637d2e898b118`.
Harness and Python-arm SHA-256 were
`15a611879c5287501ff52c2a4080cfa162b5cfdb9fccd5534dff2f9b429c4f01`
and
`9660ae8222e749bd58ef8cb7b7e5e38fdf9747c39bfa30ed32938d62e6136574`.
Raw artifact:
`tests/artifacts/perf/2026-07-29-solve-ivp-many-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** honor queue position five. After frankensearch,
frankenpandas, frankenfs, and frankenredis release `trj`, post
`[trj] CLAIM frankenscipy`, confirm the host is quiet, and rerun this exact
`1/2/4/8/16/32/64/128` completion sweep. Each row must record requested CPUs,
affinity, actual observed workers for both engines, host/core/thread/RAM/NUMA
identity, both engine artifact SHA-256s, full completion/invariant proof,
independent A/A controls, and a bootstrap-median CI decision with a 2x null
margin; CV is provenance only. Post `[trj] RELEASE frankenscipy` immediately
even if the run fails. Retry the separate 150-sample surface only after
`frankenscipy-3m5ip` satisfies its dense-output predicate.

### 2026-07-29 (cod/BlackThrush) — KEEP: live SciPy sibling cubic cursors are 3.5326x–3.6048x slower
**Result class: CAMPAIGN-WIN. Decision: KEEP.** Bead
`frankenscipy-ob7fc`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side
same-invocation** through one persistent genuine
`scipy.interpolate._cubic` co-process per cell. The matrix uses the exact
historical cursor fixture: 1,024 knots, deterministic
`sin(11x) + 0.25 cos(7x)` values, and 4,096 finite nondecreasing queries.
FrankenSciPy and SciPy received the same vectors and constructed their
interpolators outside timing.

| surface | FrankenSciPy p50 | SciPy p50 | **Incumbent ratio: SciPy / FrankenSciPy** | bootstrap-median CI95 | A/A null CI95 (ours; SciPy) | required 2x null margin | decision |
|---|---:|---:|---:|---|---|---:|---|
| CubicSpline | 0.010479 ms | 0.037775 ms | **3.6048x** | [3.5990, 3.6090] | [0.996404, 0.999925]; [0.997369, 1.002369] | 1.0072 | WIN |
| Akima | 0.010493 ms | 0.037608 ms | **3.5859x** | [3.5779, 3.5897] | [0.996204, 0.999930]; [0.998733, 1.001366] | 1.0076 | WIN |
| CubicHermite | 0.010501 ms | 0.037100 ms | **3.5326x** | [3.5268, 3.5388] | [0.998089, 1.000257]; [0.998950, 1.003720] | 1.0074 | WIN |

In plain form, Incumbent ratio: SciPy / FrankenSciPy = 3.6048x for
CubicSpline; Incumbent ratio: SciPy / FrankenSciPy = 3.5859x for Akima; and
Incumbent ratio: SciPy / FrankenSciPy = 3.5326x for CubicHermite.

All three bootstrap-median ratio CIs clear twice the worse distance of either
same-invocation A/A null from 1.0; the median-CI gate returned
`DECIDED FRANKENSCIPY WIN` in every cell. Ratio CVs were `3.068%`, `2.631%`,
and `3.150%`; **CV is provenance only** and did not enter a decision.
Full-vector conformance covered all 4,096 components per surface. Maximum
absolute difference was `2.220e-16`, with zero mismatches under the existing
`1.0e-11` differential tolerance.

The competitive translation is confirmed for both historically timed
self-comparisons: CubicSpline's `6.47x` and Akima's `5.64x`
**SELF-SPEEDUP** cursor results remain maintenance evidence, while the live
incumbent ratios above are campaign wins. The smaller live ratios are expected:
the self-comparison removed FrankenSciPy's binary searches, whereas SciPy's
incumbent path has its own vectorized interval machinery. CubicHermite had
previously been exact-bit checked as a sibling but was not timed in the old
Criterion matrix; this is its first valid incumbent result. No translation was
refuted in this matrix.

The same bench-profile ELF was strict-remote built and measured on `ovh-a`;
no local Cargo build occurred. **Executed-binary ELF SHA-256:
`e2bbfce108272890058c885402e87639807228030b5ec155c470d1360e7308d5`.**
FrankenSciPy engine SHA-256:
`e2bbfce108272890058c885402e87639807228030b5ec155c470d1360e7308d5`.
SciPy engine SHA-256:
`e37653cd7c7ccbcd78c7d10d1e49be0a8063fc466f4653e360c74c8d59bd4fad`.
Host identity: `fixmydocuments`; `physical_cores=8`, `logical_threads=16`,
`ram_bytes=67307249664`, `numa_nodes=1`. `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`.
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=15`;
`cpuset_logical_cap=1`. Each invocation recorded
`host_wide_quiescence_pre=clear`,
`host_wide_quiescence_measurement=clear`, and
`host_wide_quiescence_post=clear`; maximum sampled per-CPU busy fraction was
`10.0%`, below the fail-closed `20.0%` threshold. Raw artifact:
`tests/artifacts/perf/2026-07-29-cubic-cursor-family-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat these exact three
1,024-knot/4,096-query finite sorted serial cells. Reopen only for a changed
SciPy incumbent, a cursor implementation change, or a distinct named regime
such as unsorted/non-finite queries, the parallel batch threshold, or
construction-inclusive latency. Retain exact shared inputs, full-vector
tolerance proof, two named engine SHA-256s, one actually observed worker per
serial arm, independent A/A controls, host-wide exclusivity, and the
bootstrap-median CI gate with a 2x null margin.

### 2026-07-29 (cc/NobleCedar) — NARROW: the GMRES win is SciPy's per-iteration interpreter tax, not our kernel

**Result class: CHARACTERIZATION. Decision: KEEP THE WIN, NARROW THE CLAIM.**
Bead `frankenscipy-ddvmb` (follow-up to `frankenscipy-felow`). **No new
wall-clock measurement was taken.** This entry is a derivation over the
already-committed restart-20 artifact, whose SHA-256 is
`6703ccd1a80a6d9fb05d5a6cb472da67832bc1c060514ede191c9eed66945585`. Every
millisecond below is quoted from that artifact; only the per-iteration
division and the slope arithmetic are new. Source run provenance is unchanged:
executed-binary ELF SHA-256
`12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`, SciPy
engine SHA-256 `f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`,
`host_identity=threadripperje`, one requested and observed thread per arm.

`frankenscipy-felow` matched the inner-iteration counts exactly (127/127,
163/163, 227/227), which made the ratio per-iteration honest but left open
*what the per-iteration difference consists of*. It decomposes cleanly:

| side / n | iterations ours / SciPy | **µs/iteration ours** | **µs/iteration SciPy** | incumbent ratio |
|---:|---:|---:|---:|---:|
| 32 / 1,024 | 127 / 127 | **18.617** | **90.963** | 4.8850x win |
| 64 / 4,096 | 163 / 163 | **85.280** | **127.525** | 1.5725x win |
| 96 / 9,216 | 227 / 227 | **168.841** | **158.842** | 0.9397x loss |

Per-iteration division is legitimate here because all three cells run restart
20 with near-identical cycle structure: mean prior vectors orthogonalized
against per iteration is 9.14, 9.34, 9.30 — a 2.2% spread — so Arnoldi work
per iteration is `O(n)` with the same constant in each cell. Nothing is
averaged across cells; each cell is divided by its own count.

**Fit-free mechanism.** Over a 9.0x increase in `n`, our per-iteration cost
rises **9.069x** (essentially proportional to `n`: all work, no fixed
overhead) while SciPy's rises only **1.746x** (dominated by an `n`-independent
term). Finite-difference marginal cost between adjacent cells:

| segment | ours µs/unknown | SciPy µs/unknown | ours / SciPy |
|---|---:|---:|---:|
| n = 1,024 -> 4,096 | 0.021700 | 0.011901 | **1.823x** |
| n = 4,096 -> 9,216 | 0.016320 | 0.006117 | **2.668x** |

**Our marginal cost per unknown is worse than SciPy's in both segments.** That
statement uses no fitted model.

Least squares on `a + b*n` (3 points, 1 degree of freedom, reported only as
quantification, no CI claimed): ours `a=4.266 µs`, `b=0.018132`, R²=0.9939;
SciPy `a=87.239 µs`, `b=0.008065`, R²=0.9651. Fixed per-iteration overhead
difference **82.973 µs/iteration**; marginal per-unknown ours/SciPy
**2.248x**; predicted equal-cost crossover **n ≈ 8,242 (side ≈ 91)**.

The model reconstructs each measured gap, which is the real check:

| n | iterations | fixed term | marginal term | predicted gap | measured gap | error |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 127 | +10.538 ms | -1.309 ms | +9.228 ms | +9.188 ms | +0.040 ms |
| 4,096 | 163 | +13.525 ms | -6.721 ms | +6.803 ms | +6.886 ms | -0.082 ms |
| 9,216 | 227 | +18.835 ms | -21.060 ms | -2.226 ms | -2.270 ms | +0.044 ms |

Positive means SciPy slower. Residuals are 0.4%–3.6% of gap magnitude, and the
side-96 sign flip is reproduced: the fixed-overhead credit keeps growing but
our marginal-cost debit grows faster and overtakes it.

**What this licenses:** on this fixture class at one thread, FrankenSciPy GMRES
wins at small-to-moderate `n` because it does not pay SciPy's ~87 µs/iteration
of interpreted per-iteration Givens/orthogonalization bookkeeping, at exactly
matched Krylov trajectories. The 4.8850x is real and stands.

**Forbidden claims.** (1) "Our GMRES kernel is faster than SciPy's" — it is
not; per unknown it is 1.8–2.7x slower. (2) Any size-general GMRES win —
already rejected, and this explains why it was always going to be. (3) Any
extrapolation of 4.8850x to larger `n`, other stencils, preconditioned solves,
or other restart lengths.

**Matrix class characterized.** Nonsymmetric, strictly diagonally dominant 2-D
convection-diffusion CSR, 5-point stencil, `diagonal=4.001`, `west=-1.2`,
`east=-0.8`, `vertical=-1`, `nnz = 5n - 4*side`, `rhs = 1 + 0.01*(i mod 17)`,
`x0 = zeros`, `rtol=1e-5`, `atol=0`, restart 20 both arms. Win regime is `n`
below roughly 8,000 on this class at one thread.

**Consequence for `frankenscipy-ddvmb`.** That bead wants a restart-20 profile
attributing >=8% self-time to one removable first-party leaf. This says where
to point it: the target is the **`O(n)` per-iteration inner kernel** — SpMV
plus the ~9.3 Gram-Schmidt dot/axpy pairs per iteration — *not* per-iteration
bookkeeping, which we already win by ~83 µs/iteration. No bookkeeping or
allocation lever can close a 1.8–2.7x marginal-cost gap per unknown.

**Sibling audit.** The same script over the pre-parity artifact rejects all
five cells. The three superseded GMRES rows are contaminated in both
directions (side 32: 125/127, ours fewer, win inflated; sides 64/96: ours more,
losses overstated). The two **BiCGSTAB** KEEP rows are still count-unmatched by
one iteration — 45/44 and 89/88 — but the mismatch runs *against* us, so
`3.1908x` and `1.4638x` are conservative and are left standing rather than
superseded. Matching those counts could only raise them.

**Attempted and not obtained: a wider size sweep on `thinkstation1`.** A
10-point sweep (sides 24…128) was prepared to replace the 3-point fit with a
10-point one and to check the crossover on a second host. It did not run.
`RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec --` compiled the
harness successfully on remote worker `vmi1153651` (exit 0, 230 s), but rch's
`[transfer] exclude_patterns` excludes `target/`, so the executable is never
shipped back and cannot be run on this host. Wrapping the build in `bash -c`
to copy it out of `target/` was correctly refused as a non-compilation command
(`RCH-E301`, no local fallback). Shared fleet config was **not** edited to work
around this, and no local Cargo build was run. Measurement on a specific host
therefore still requires a booked host; `trj` is exclusive and this project is
queued behind frankensearch, frankenpandas, frankenfs and frankenredis, so it
was not claimed. Analysis host recorded as `host_identity=thinkstation1`, AMD
Ryzen Threadripper PRO 5975WX 32-Cores, `physical_cores=32`,
`logical_threads=64`, RAM 215 GiB; the arithmetic is host-independent.

Raw artifact and reproduction script:
`tests/artifacts/perf/2026-07-29-gmres-per-iteration-decomposition/`
(`decomposition.md`, `decompose.py`). The script refuses any cell whose two
arms report different iteration counts, printing `EXCLUDED from fit`, so an
unmatched cell can never silently enter the model.

**Concrete retry predicate:** do not re-derive this decomposition, and do not
re-run the side-32/64/96 restart-20 cells to "confirm" it. Reopen only to (a)
replace the 3-point fit with >=6 iteration-matched size points measured in one
booked-host session with per-arm A/A controls, or (b) profile the `O(n)` inner
kernel per `frankenscipy-ddvmb`'s existing 8%/7% predicate. Any successor must
keep per-cell iteration-count matching, divide by each cell's own count, and
never average a ratio across cells with different counts.

### 2026-07-29 (cod/BlackThrush) — KEEP: GMRES restart parity clears the side-64 counted-work loss

**Result class: CAMPAIGN-WIN. Decision: KEEP DEFAULT-PARITY LEVER.** Bead
`frankenscipy-felow`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side in
the same invocation.** FrankenSciPy's public GMRES default restart is
now 20, matching the live incumbent rather than running a different
restart-30 schedule. Both arms solved the identical nonsymmetric, strictly
diagonally dominant convection-diffusion CSR from a zero initial guess at
`rtol=1e-5`, `atol=0`; construction, transfer, callback counting, and
full-vector comparison were outside timing.

| side / n / nnz | FrankenSciPy p50 | SciPy p50 | **Incumbent ratio: SciPy / FrankenSciPy** | bootstrap-median CI95 | A/A null CI95 (ours; SciPy) | 2x null gate | iterations ours / SciPy | verdict |
|---|---:|---:|---:|---|---|---:|---:|---|
| 32 / 1,024 / 4,992 | 2.364355 ms | 11.552343 ms | **4.8850x** | [4.8701, 4.8941] | [0.997199, 1.002213]; [0.995624, 1.002444] | 1.0088 | 127 / 127 | FrankenSciPy win |
| 64 / 4,096 / 20,224 | 13.900668 ms | 20.786496 ms | **1.5725x** | [1.5677, 1.5906] | [0.998664, 1.000610]; [0.996642, 1.010567] | 1.0211 | 163 / 163 | FrankenSciPy win |
| 96 / 9,216 / 45,696 | 38.326807 ms | 36.057110 ms | **0.9397x** | [0.9376, 0.9406] | [0.998961, 1.001016]; [0.998693, 1.002598] | inverse 1 / 1.0052 | 227 / 227 | FrankenSciPy loss |

Headline incumbent ratio: SciPy / FrankenSciPy = 4.8850x at side 32; the
bounded side-64 incumbent ratio was `1.5725x`.

The previous default-30 side-64 row performed 244 inner Arnoldi iterations
against SciPy's 163 and lost `1.0575x`. Matching the incumbent's public
restart-20 default makes the trajectory exactly 163/163 and flips that cell
to a decided `1.5725x` win. Side 32 is exact at 127/127 and wins `4.8850x`.
This satisfies the prior retry predicate's `<=179` side-64 counted-work
threshold and confirms that a per-iteration kernel optimization would have
targeted the wrong mechanism first. The lever is kept for public-default
compatibility and these bounded wins. Side 96 is exact at 227/227 but remains
a decided loss, so the size-general claim is rejected in the row below.
Ratio CVs were `3.053%`, `3.650%`, and `2.458%`; the side-64 ours-null CV was
`10.380%`. **CV is provenance only and was not used for a verdict.**

Every output component passed (`1,024/1,024`, `4,096/4,096`, and
`9,216/9,216`) with zero tolerance mismatches. Maximum absolute differences
were `2.345e-13`, `1.990e-13`, and `6.537e-13`; relative-L2 differences were
`2.445e-15`, `6.267e-16`, and `1.324e-15`. True relative residuals matched at
the reported precision: `9.890e-6`, `9.384e-6`, and `9.872e-6`. Numerical
stability and tolerance contracts were unchanged.

Strict-remote RCH built the release-perf executable on `hz2`; no local Cargo
build occurred. **Executed-binary ELF SHA-256:
`12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`.**
FrankenSciPy engine SHA-256:
`12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
SciPy oracle-script SHA-256:
`1c0664e6d38e3a4d31d17a9b8ba1535cb20391c2f58521dbb7ef7ee488f70132`.
`host_identity=threadripperje`; `physical_cores=64`; `logical_threads=128`;
`ram_bytes=536069869568`; `numa_nodes=1`; `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=127`;
`cpuset_logical_cap=1`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=performance`;
`energy_performance_preference=performance`;
`host_wide_quiescence_pre=clear`;
`host_wide_quiescence_measurement=clear`;
`host_wide_quiescence_post=clear`. Canonical booking messages were
`[trj] CLAIM frankenscipy` `6403` and
`[trj] RELEASE frankenscipy` `6405`;
`trj_booking_claim_message_id=6403`;
`trj_booking_release_message_id=6405`. Raw artifact:
`tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/gmres_restart20_bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat the exact restart-20 side-32 or
side-64 serial cells. Reopen them only for a changed live SciPy incumbent,
public restart/stopping-policy change, or a distinct named
fixture/preconditioner. Preserve the shared CSR/RHS, full-vector and
true-residual proof, counted trajectory, both engine hashes, actual observed
threads, governor, literal host exclusivity, independent per-arm A/A
controls, and bootstrap-median CI gate with a 2x null margin.

### 2026-07-29 (cod/BlackThrush) — REJECT: restart parity does not establish a size-general GMRES win

**Decision: REJECT SIZE-GENERAL COMPETITIVE GENERALIZATION.** Bead
`frankenscipy-felow`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side in
the same invocation.** With both public defaults at restart 20, the
side-96 cell (`n=9,216`, `nnz=45,696`) measured FrankenSciPy at
`38.326807 ms` and SciPy at `36.057110 ms`. **Incumbent ratio: SciPy /
FrankenSciPy = 0.9397x** with bootstrap-median CI95
`[0.9376, 0.9406]`; equivalently FrankenSciPy remains `1.0642x` slower.

FrankenSciPy A/A median was `1.000407`, CI95
`[0.998961, 1.001016]`; SciPy A/A median was `0.999511`, CI95
`[0.998693, 1.002598]`. The 2x A/A-null margin required `1.0052`; the loss
decision applies its inverse `1 / 1.0052`, and the ratio CI high is below it.
Both arms performed exactly 227 inner Arnoldi iterations. The old 239/227
work-count mismatch is gone; the residual loss is
per-iteration/implementation-shaped on this fixture rather than explained by
extra Krylov work. Ratio CV was `2.458%`; **CV is provenance only and was not
the gate.**

All `9,216/9,216` components passed with zero tolerance mismatches, maximum
absolute difference `6.537e-13`, relative-L2 difference `1.324e-15`, and
identical reported true relative residual `9.872e-6`.

**Executed-binary ELF SHA-256:
`12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`.**
FrankenSciPy engine SHA-256:
`12003f00e83c8074ef249cbab59659641c43bf9845da487a72d94c0922a29ab5`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
`host_identity=threadripperje`; `physical_cores=64`; `logical_threads=128`;
`ram_bytes=536069869568`; `numa_nodes=1`; `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=127`;
`cpuset_logical_cap=1`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=performance`;
`energy_performance_preference=performance`;
`host_wide_quiescence_pre=clear`;
`host_wide_quiescence_measurement=clear`;
`host_wide_quiescence_post=clear`. Canonical booking messages were CLAIM
`6403` and RELEASE `6405`;
`trj_booking_claim_message_id=6403`;
`trj_booking_release_message_id=6405`. Raw artifact:
`tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/gmres_restart20_bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not retest another restart value or tune an
unprofiled kernel. Reopen this exact side-96 cell only after a restart-20
profile attributes at least 8% of FrankenSciPy self-time to one removable
first-party leaf and a named lever plausibly removes at least 7% total wall
time while preserving the exact 227-iteration trajectory and tolerance
contract. Then rerun the genuine incumbent in the same invocation with
independent per-arm A/A controls and the same median-CI gate.

### 2026-07-29 (cod/BlackThrush) — KEEP: live SciPy nonsymmetric BiCGSTAB is 1.4638–3.1908x slower

**Result class: CAMPAIGN-WIN. Decision: KEEP.** Bead
`frankenscipy-l0em4`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side
same-invocation** through a persistent genuine
`scipy.sparse.linalg._isolve.iterative.bicgstab` co-process. Both arms
received the exact same nonsymmetric, strictly diagonally dominant
convection-diffusion CSR, deterministic RHS, zero initial guess, `rtol=1e-5`,
`atol=0`, and size-scaled maximum iteration bound.

| side / n / nnz | FrankenSciPy p50 | SciPy p50 | **Incumbent ratio: SciPy / FrankenSciPy** | bootstrap-median CI95 | A/A null CI95 (ours; SciPy) | required 2x null margin |
|---|---:|---:|---:|---|---|---:|
| 32 / 1,024 / 4,992 | 0.551575 ms | 1.763953 ms | **3.1908x** | [3.1892, 3.2020] | [0.998467, 1.002320]; [0.996244, 1.000773] | 1.0075 |
| 64 / 4,096 / 20,224 | 4.513120 ms | 6.607369 ms | **1.4638x** | [1.4606, 1.4646] | [0.999480, 1.001928]; [0.998375, 1.004048] | 1.0081 |

In plain form, Incumbent ratio: SciPy / FrankenSciPy = 3.1908x at side
32, and Incumbent ratio: SciPy / FrankenSciPy = 1.4638x at side 64.

Both bootstrap-median CI lows clear twice the worse A/A-null distance from
1.0. Ratio CVs were `3.019%` and `2.466%`; **CV is provenance only**.
Full-vector comparison covered every component with zero tolerance
mismatches. Maximum absolute differences were `1.066e-13` and `1.904e-12`;
relative-L2 differences were `6.761e-16` and `3.541e-15`. True relative
residuals agreed at `9.292e-6` and `5.663e-6`. Counted trajectories were 45
versus 44 and 89 versus 88 iterations, so the incumbent win does not come
from a weaker stopping contract. No historical BiCGSTAB self-speedup row was
found; these are first incumbent measurements rather than translation claims.

Strict-remote RCH built the release-perf executable; no local Cargo build
occurred. **Executed-binary ELF SHA-256:
`26197521b4683ae44fa81f1c0e9a1720cab08ad7675f73c9b86d8d27dbdb92e2`.**
FrankenSciPy engine SHA-256:
`26197521b4683ae44fa81f1c0e9a1720cab08ad7675f73c9b86d8d27dbdb92e2`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
`host_identity=fixmydocuments`; `physical_cores=8`; `logical_threads=16`;
`ram_bytes=67307249664`; `numa_nodes=1`; `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=15`;
`cpuset_logical_cap=1`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=powersave`;
`energy_performance_preference=balance_performance`. RCH had no ovh-a job
before the recorded invocations; both recorded
`host_wide_quiescence_pre=clear`,
`host_wide_quiescence_measurement=clear`, and
`host_wide_quiescence_post=clear`.

Raw artifact:
`tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat the exact side-32 or side-64
serial cells. Reopen only after a SciPy-incumbent change, a BiCGSTAB
source/stopping-policy change, or for a distinct named size/preconditioner
regime. Retain exact shared input, full-vector and residual proof, counted
iterations, two engine hashes, observed threads, governor, literal host
exclusivity, dual A/A controls, and the bootstrap-median CI gate with a 2x
null margin.

### 2026-07-29 (cod/BlackThrush) — REJECT: live SciPy GMRES wins at side 64 and 96

**Decision: REJECT LARGE-N COMPETITIVE GENERALIZATION.** Bead
`frankenscipy-l0em4`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side
same-invocation.** Both public defaults solved the same nonsymmetric
convection-diffusion CSR at `rtol=1e-5`, `atol=0`; FrankenSciPy used default
restart 30 and SciPy default restart 20. SciPy's iteration callback was used
only for the untimed counted proof.

| side / n / nnz | FrankenSciPy p50 | SciPy p50 | **Incumbent ratio: SciPy / FrankenSciPy** | bootstrap-median CI95 | A/A null CI95 (ours; SciPy) | inverse required | decision |
|---|---:|---:|---:|---|---|---:|---|
| 64 / 4,096 / 20,224 | 25.161683 ms | 23.776576 ms | **0.9456x** | [0.9429, 0.9463] | [0.998933, 1.000654]; [0.997443, 1.002581] | 1 / 1.0052 | FrankenSciPy 1.0575x slower |
| 96 / 9,216 / 45,696 | 55.814276 ms | 49.676992 ms | **0.8899x** | [0.8894, 0.8904] | [0.999380, 1.000342]; [1.000261, 1.002005] | 1 / 1.0040 | FrankenSciPy 1.1237x slower |

Both ratio CI highs are below the inverse 2x A/A-null margin, so both are
decided losses. Side 64 is iteration-count-shaped: FrankenSciPy took 244
inner iterations versus SciPy's 163 (`1.4969x`). At side 96 the count ratio
narrows to 239 versus 227 (`1.0529x`) while the wall gap widens, proving count
alone does not explain the larger cell. In the side-32 KEEP below, the ratio
is `3.9679x` with 125 versus 127 iterations. The advantage therefore narrows
and crosses with size rather than staying flat. Fixed incumbent overhead at
small size followed by native per-iteration work at larger size is an
inference from that shape, not a profile attribution. Ratio CVs were `0.412%`
and `0.201%`; **CV is provenance only**.

All 4,096 and 9,216 output components were compared with zero tolerance
mismatches; relative-L2 differences were `6.963e-6` and `1.334e-6`, and all
true relative residuals were below `9.9e-6`. The negative-evidence grep found
no historical GMRES self-speedup, so this is not a failed translation; it is
the first live-incumbent evidence and refutes a size-general advantage.

FrankenSciPy engine SHA-256:
`26197521b4683ae44fa81f1c0e9a1720cab08ad7675f73c9b86d8d27dbdb92e2`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
`host_identity=fixmydocuments`; `physical_cores=8`; `logical_threads=16`;
`ram_bytes=67307249664`; `numa_nodes=1`; `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=15`;
`cpuset_logical_cap=1`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=powersave`;
`energy_performance_preference=balance_performance`;
`host_wide_quiescence_pre=clear`;
`host_wide_quiescence_measurement=clear`;
`host_wide_quiescence_post=clear`. Raw artifact:
`tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** retry this exact large-N matrix only after
either (a) a restart/orthogonalization change reduces side-64 counted inner
iterations to at most 179 (`<=1.10x` SciPy) without weakening parity/residuals
or regressing side 96, or (b) a side-96 profile attributes at least 10%
self-time to a removable first-party kernel and the proposed lever removes
counted work there. If a libc copy leaf dominates, resolve its addresses with
`addr2line -f -C -i` and record the inlined first-party call sites. Rerun both
large cells with the live incumbent, dual nulls, recorded governor, host
exclusivity, and median-CI gate.

### 2026-07-29 (cod/BlackThrush) — KEEP: live SciPy GMRES is 3.9679x slower at side 32

> **SUPERSEDED 2026-07-29 by the restart-20 parity row above — do not quote the
> 3.9679x.** This cell was measured with FrankenSciPy on restart 30 against
> SciPy's restart-20 default, so the two arms ran **different Krylov
> trajectories: 125 inner iterations versus 127**. The number therefore mixes a
> per-iteration effect with a convergence effect and is not a per-iteration
> honest comparison. The matched-count replacement for this exact cell is
> **4.8850x at 127/127 iterations**. Quote that instead. See also the
> per-iteration decomposition entry, which attributes the matched-count win to
> SciPy's fixed per-iteration interpreter cost rather than to our kernel.

**Result class: CAMPAIGN-WIN. Decision: KEEP.** Bead
`frankenscipy-l0em4`. **Legacy incumbent arm: SciPy 1.17.1, side-by-side
same-invocation.** On the exact side-32 nonsymmetric convection-diffusion CSR
(`n=1,024`, `nnz=4,992`), **Incumbent ratio: SciPy / FrankenSciPy =
3.9679x**. FrankenSciPy took 3.157245 ms and SciPy 12.515694 ms;
bootstrap-median CI95 was `[3.9645, 3.9740]`.

FrankenSciPy A/A null median was `0.998784`, CI95
`[0.997426, 0.999946]`; SciPy A/A null median was `1.000281`, CI95
`[0.997595, 1.000932]`. The required 2x A/A-null margin was `1.0052`.
Ratio CV was `2.620%`; **CV is provenance only**. Both arms converged in 125
versus 127 counted inner iterations. All 1,024 components passed, relative-L2
difference was `5.663e-6`, and true relative residuals were `9.824e-6` and
`9.890e-6`. This claim is explicitly size-limited: the same harness records
decided side-64 and side-96 losses above.

**Executed-binary ELF SHA-256:
`26197521b4683ae44fa81f1c0e9a1720cab08ad7675f73c9b86d8d27dbdb92e2`.**
FrankenSciPy engine SHA-256:
`26197521b4683ae44fa81f1c0e9a1720cab08ad7675f73c9b86d8d27dbdb92e2`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
`host_identity=fixmydocuments`; `physical_cores=8`; `logical_threads=16`;
`ram_bytes=67307249664`; `numa_nodes=1`; `requested_threads=1`;
`actual_observed_frankenscipy_worker_threads=1`;
`actual_observed_scipy_worker_threads=1`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`; `affinity=15`;
`cpuset_logical_cap=1`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=powersave`;
`energy_performance_preference=balance_performance`;
`host_wide_quiescence_pre=clear`;
`host_wide_quiescence_measurement=clear`;
`host_wide_quiescence_post=clear`. Raw artifact:
`tests/artifacts/perf/2026-07-29-sparse-nonsymmetric-vs-scipy-live-arm/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat this exact side-32 serial cell.
Reopen only for a changed SciPy incumbent, a GMRES source/default-restart
change, or a distinct named fixture/preconditioner. Retain the shared CSR/RHS,
full-vector and residual proof, iteration counts, two engine hashes, observed
threads, governor, literal host exclusivity, dual A/A controls, and
bootstrap-median CI gate with a 2x null margin.

### 2026-07-30 (cod/BlackThrush) — KEEP: a whole 16x512 stiff reaction-screen job is 8.3326x faster than the strongest live SciPy arm

**Result class: CAMPAIGN-WIN. Decision: KEEP.** Bead
`frankenscipy-0tx94`. The single-solve exact-diagonal result of `36.1770x`
compresses by 4.34x when amortized across a user-recognizable scientific job,
but remains a decided **8.3326x whole-job win**. This row does not promote the
kernel number into a general BDF claim.

**Pre-registered workload and mechanism.** Commit
`ab079f0def8e8360a1d78bcd308ba3111367cad3` fixed the workload and harness
before the exclusive adjudication run: 16 independent initial-dose scenarios
of a nonlinear 512-species stiff first-plus-second-order self-decay model,
integrated over `[0,1]` with BDF at `rtol=1e-8`, `atol=1e-10`, and 65 requested
observation times. Each timed arm constructs its model and inputs, creates and
tears down its worker pool, solves all 16 scenarios, materializes 532,480 state
values, and computes 8,192 trapezoidal exposure/AUC values plus 16 terminal
masses. Interpreter startup, SciPy import, pipe transport, parity
serialization, and provenance collection are outside both timed regions.
Before timing, the mechanism was falsifiable in three ways: screen the
incumbent backends, count solver work, and measure the callback/pool component
separately.

**Strongest incumbent, live in the same invocation.** The genuine SciPy
`1.17.1` arm screened all 12 combinations of `jac=None`, an analytic sparse
Jacobian, or `jac_sparsity` only with 1, 2, 4, or 8 worker processes. The
fastest valid arm was `jac_sparsity` with 8 processes (`258.213629 ms` in the
untimed screen), so that arm—not a favorable weaker configuration—is the
incumbent in all headline samples. **Legacy incumbent arm: SciPy 1.17.1,
side-by-side same-invocation** with FrankenSciPy. SciPy engine SHA-256:
`aa16f42cc85fa02769ff00bf93bcdb48b6bf568e2d9f8ce48f9f378e76cf8f09`.
FrankenSciPy engine SHA-256:
`0a444b7db2fc46dd2cdbde8436f9641cbe0ae8aa20155957d899e49af89556b0`.

| whole-job wall | FrankenSciPy | live SciPy 1.17.1 |
|---|---:|---:|
| p50 | **32.042288 ms** | 271.338483 ms |
| p95 | **46.613768 ms** | 300.611317 ms |
| p99 | **46.613768 ms** | 300.611317 ms |

**Incumbent ratio: SciPy / FrankenSciPy = 8.3326x.** Bootstrap-median CI
`[6.5338, 8.5715]` => **DECIDED FRANKENSCIPY WIN**. Independent A/A controls
were run for both arms:
FrankenSciPy median `0.981405`, CI95
`[0.788354, 1.008960]`; SciPy median `1.005182`, CI95
`[0.969951, 1.033351]`. The ratio CI is wholly above unity and also clears the
harness's conservative `1.5369x` threshold with `null_margin=2x`.
**The decision is from the bootstrap-median ratio CI, never CV; CV is
provenance only.** CVs
(`13.241%` ours-null, `4.310%` SciPy-null, `11.957%` ratio) are provenance
only.

**Full-result and analytic conformance.** The harness compared all 16
trajectories, all 532,480 state values, all 8,192 exposure values, and all 16
terminal masses before admitting timing. Maximum state disagreement was
`8.624e-10`, or `5.014` tolerance units; maximum exposure and terminal-mass
differences were `0.152` and `0.000` units. The maximum errors against the
closed-form nonlinear decay solution were `31.654` units for FrankenSciPy and
`32.111` for SciPy, all below the pre-registered 100-unit contract. Shared
input SHA-256:
`ba4117ad3e2e68c6cc0627eef13e9e167b5bc56299a1fe1d938c163f63df2e3c`.

**Work was counted before attribution, and FrankenSciPy does more of it.**
Across the 16 solves, FrankenSciPy performed 53,818 function evaluations and
2,257 LU/factorization events versus SciPy's 33,905 and 2,032: **1.5873x more
function evaluations and 1.1107x more factorizations**, not less. Both stored
1,040 requested solution points. Execution proof was
`diag_hits=2257, band_hits=0`. Replaying SciPy's parallel Python RHS and pool
cost isolated `19.4724 ms`, only `7.2%` of its whole job; the callback-free
sensitivity ratio remains `7.8604x`. The win is therefore not an
iteration-count shortcut or a banked Python-callback asymmetry. It is
consistent with the pre-registered exact-diagonal specialization surviving
real model setup, scheduling, output materialization, and postprocessing.

**Resource and binary provenance.** Both arms requested and actually used
eight compute slots inside affinity `56-63`: FrankenSciPy observed 8 compute
threads; SciPy observed 8 one-thread worker processes with BLAS capped at one
thread each. `requested_threads=8`;
`actual_observed_frankenscipy_worker_threads=8`;
`actual_observed_scipy_worker_threads=8`; `host_identity=threadripperje`;
`physical_cores=64`;
`logical_threads=128`; `ram_bytes=536069869568`; `numa_nodes=1`;
`cpuset_logical_cap=8`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=performance`;
`runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes` and no AVX-512.
FrankenSciPy peak RSS was `74,232 KiB`; SciPy's sum of per-process high-water
marks was an upper bound of `591,704 KiB`. The strict-remote RCH
release-perf ELF was built on `ovh-a` from base
`8ef32a3c94e73bb61725e7b1a618869170e1c542` with a clean deterministic
overlay and retrieved by Route 1; no local Cargo build contributed to the
measurement binary. **Executed ELF SHA-256:
`0a444b7db2fc46dd2cdbde8436f9641cbe0ae8aa20155957d899e49af89556b0`.**
The exclusive measurement used `[trj] CLAIM` message 6983 and released in
message 6985. Pre-admission load was `0.45/0.62/0.82`, with zero blocked
processes, 99.81% average idle, and 0% iowait; the post-run samples were
99–100% idle with 0% iowait and zero blocked processes.
`host_wide_quiescence_pre=clear`; `host_wide_quiescence_post=clear`;
`trj_booking_claim_message_id=6983`;
`trj_booking_release_message_id=6985`.

Raw artifact:
`tests/artifacts/perf/2026-07-30-bdf-real-job-vs-scipy/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat this exact 16x512/eight-CPU
reaction screen. Reopen only for a changed SciPy engine, a different named
scenario-count/species-count/Jacobian-structure regime, or a solver change
that materially changes the counted work. Pre-register the new mechanism,
screen the strongest valid SciPy configuration, and retain the whole-output
contract, exact observed-thread counts, dual A/A controls, and
bootstrap-median CI decision.

**CHOOSER STATEMENT:** Pick FrankenSciPy `solve_ivp_many` for an eight-CPU,
multi-scenario independent stiff reaction screen matching this shape: it
finishes the whole job in 32.04 ms p50 versus 271.34 ms for the fastest
screened live SciPy 1.17.1 arm. Pick SciPy 1.17.1 for dense-Jacobian BDF work,
where the separate live n=512 result favors SciPy by 2.12x. Outside these two
measured shapes, benchmark the user's actual job; this evidence does not
choose for them.

### 2026-07-30 (cc/BlackThrush) — MIXED, size-dependent: qmr wins below n≈8,100 and loses above; the dispatch-count model for SciPy's fixed tax is FALSIFIED

Third and sharpest test of the per-iteration interpreter-tax mechanism, after
GMRES (`86bcccd74`) and lsqr (`2c651d13e`). Predictions committed in
`1f6f8ffdc` **before any qmr timing existed**; artifact
`tests/artifacts/perf/2026-07-30-qmr-vs-scipy-live-arm/`.

`qmr` was chosen to hold the dominant primitive fixed at two sparse matvecs per
iteration — lsqr's mix, which measured parity — and change exactly one
variable: the interpreted bookkeeping wrapped around them. Recurrences were
verified algebraically identical to SciPy's by line-by-line source reading
before measuring, with one structural difference: SciPy carries the residual
recursively, we recompute the true `b - Ax`, costing us a **third** matvec.

**RESULT, eight cells, whole-job wall-clock ratio (SciPy / FrankenSciPy):**

| side | n | it ours/SciPy | matched | µs/it ours | µs/it SciPy | whole-job |
|---:|---:|---:|:--:|---:|---:|---:|
| 16 | 256 | 40/40 | YES | 4.369 | 42.801 | **9.7970x** |
| 32 | 1,024 | 80/80 | YES | 17.016 | 53.472 | **3.1427x** |
| 48 | 2,304 | 108/108 | YES | 39.460 | 71.163 | **1.8090x** |
| 64 | 4,096 | 136/136 | YES | 72.885 | 92.536 | **1.2739x** |
| 96 | 9,216 | 200/198 | no | 153.875 | 144.820 | **0.9359x** |
| 128 | 16,384 | 255/269 | no | 273.279 | 225.380 | **0.8705x** |
| 160 | 25,600 | 393/397 | no | 424.301 | 321.264 | **0.7653x** |
| 192 | 36,864 | 895/883 | no | 625.921 | 461.677 | **0.7232x** |

Per-iteration fit over the four count-matched cells: ours `a=-0.914 µs (~0)`,
`b=0.017889 µs/unknown`, R²=0.9994; SciPy `a=+40.113 µs`, `b=0.012958`,
R²=0.9984. Marginal per-unknown ours/SciPy = **1.380x**; fitted crossover
`n* = 8,136` (side ~90). Same-session lsqr control re-measured, not quoted
across sessions: `a=+28.342 µs`, `b=0.009927`, marginal ratio **1.043x**,
reproducing the committed lsqr cells to within 6% at every size.

**SCORECARD: one confirmed, three falsified, one partial.**

- **P1 FALSIFIED.** `a_scipy(qmr)/a_scipy(lsqr) = 1.415x` against a predicted
  `[2.0, 2.8]` from the dispatch count `D_qmr/D_lsqr = 43/18 = 2.389`. Implied
  cost per dispatch unit is not constant: 1.575 µs (lsqr) vs 0.933 µs (qmr).
  **`a_scipy ≈ c·D` with a single `c` is wrong.** Candidate replacement, offered
  as a hypothesis to pre-register and NOT as a claim: lsqr allocates ~11 fresh
  length-n temporaries per iteration (out-of-place style), qmr ~7 (in-place
  style, and its four identity-preconditioner round-trips return the input
  object without allocating), so temporary allocation may be the driver rather
  than call count.
- **P2 CONFIRMED**, the riskiest prediction and the one whose stated assumption
  was flagged as the likeliest falsifier. `b_ours/b_scipy = 1.380x` inside the
  predicted `[1.25, 1.75]`. Same fixture, host and session: two matvecs each in
  lsqr gives 1.043x parity, our third matvec in qmr gives 1.380x. Its stated
  *reason* was only half right — SciPy's 17 extra streamed length-n temporaries
  cost it **~30%** more per unknown (`b_scipy` 0.009927 -> 0.012958), not the
  ~nothing asserted. Both terms moved; the ratio landed because they moved
  together.
- **P3 PARTIAL.** Fitted `n* = 8,136` is inside the predicted `[8e3, 3e4]` at
  its lower edge, and the crossover was **directly observed** — side 64 wins at
  1.2739x, side 96 loses at 0.9359x — rather than extrapolated 60x outside the
  range as the lsqr crossover was. The accompanying clause is falsified: I
  predicted a win at side <= 96 and a crossing by side 128-160; we are already
  losing at side 96.
- **P4 FALSIFIED at scale.** Counts match exactly at sides 16-64 and diverge
  above (200/198, 255/269, 393/397, 895/883). Informative rather than merely
  wrong: at side 128 **SciPy needs 14 more iterations than we do**, because its
  recursively carried residual drifts pessimistic as n grows. Our third matvec
  partially pays for itself in counted work.
- **P5 FALSIFIED.** Side-16 whole-job ratio 9.797x against a predicted
  `[14, 25]`; a direct consequence of P1.

**Two defects found by the measurement, both filed, neither a perf issue.**
`frankenscipy-9pfja`: qmr's three breakdown gates used `f64::EPSILON * 1e6`
(2.220e-10) where SciPy uses `eps` (2.220e-16), a million times looser, so
healthy solves aborted at n >= 4,096 with `converged=false` and residuals from
4.4e-4 to 9.07e-1. Diagnosed by replaying SciPy's own recurrences; the replay
predicts the abort iteration exactly (121 at side 64, 151 at side 96). Fixed to
`BREAKDOWN_TOL = f64::EPSILON`; side 64 then converges at exactly SciPy's 136
iterations with `relative_l2_diff = 6.109e-13`. `frankenscipy-6pdfn`: `lsmr`
delegates to `lsqr` and `minres` delegates to `gmres`, both claiming "Matches
scipy.sparse.linalg.…" in their docstrings.

**Disclosure.** The breakdown fix was made after the first sweep *aborted*, not
after seeing a ratio. It changes when the solver gives up, not per-iteration
work. The pre-fix three-cell fit gave `a_scipy = 40.113 µs` and
`b_ours/b_scipy = 1.337x` — the same P1 falsification and P2 confirmation as the
eight-cell run, preserved in `raw/prefix_breakdown_bug/PREFIX_decomposition.txt`.
**No prediction's verdict depends on the fix.** What it bought was P3: without
it the crossover could only have been extrapolated.

**Provenance.** Host `thinkstation1` (5975WX, 32C/64T, AVX2+FMA, no AVX-512),
`scaling_governor=powersave`, `energy_performance_preference=performance`,
pinned `affinity=63`, `cpuset_logical_cap=1`, one requested and observed worker
thread per arm, `python_blas_thread_cap=1`, 21 rounds per cell, construction /
serialization / iteration counting / parity checks outside timing. SciPy 1.17.1,
NumPy 2.4.3, CPython 3.13; `scipy_engine_sha256=f9d7ace03295000d7b1a76dd1222`
`9208908a59140b741669e961b69733110e8f`. **Executed ELF SHA-256:
`f48c77421bca7f08631f8caab0c9f0f2ffba3cb0dee385d4795fa495828e6068`**, built on
rch worker `vmi1293453` from base `1f6f8ffdc` with a clean deterministic
overlay and retrieved by scp from the worker pool directory; no local Cargo
build contributed. **EVIDENCE CLASS: `PROVISIONAL_NON_EXCLUSIVE` for every
cell** — host load ~10 throughout, the fail-closed quiescence gate was waived,
and no cell may be called DECIDED. This is the same class the lsqr run carries,
which is exactly what makes the two comparable. Every cell cleared the 2x
A/A-null margin.

**Concrete retry predicate:** do not re-run this un-preconditioned qmr /
convection-diffusion / one-thread screen. Reopen for a changed SciPy engine, a
preconditioned qmr, a different matrix class (this fixture is strictly
diagonally dominant and non-symmetric), or a change to fsci's residual
bookkeeping. The open lever is exactly that last one, and it must be
pre-registered before measuring: **carrying the residual recursively as SciPy
does would remove our third matvec.** If `b_ours` fell toward `2/3 x 0.017889 ≈
0.0119`, `b_ours/b_scipy` becomes ~0.92 — marginally cheaper than SciPy,
eliminating the crossover and leaving the ratio asymptoting to ~1.09x instead of
decaying through 1.0. Against that, P4 shows the recursive residual costs SciPy
~5% extra iterations at side 128, which we would inherit. The net is genuinely
uncertain, which is why it is a hypothesis and not a plan.

**CHOOSER STATEMENT:** For un-preconditioned QMR on a 2-D convection-diffusion
system at `rtol=1e-5`, single-threaded: pick FrankenSciPy below roughly
**n ≈ 8,000 unknowns** and SciPy 1.17.1 above it. At n=256 FrankenSciPy finishes
the whole job 9.80x faster; the advantage decays monotonically to 1.27x at
n=4,096, crosses between n=4,096 and n=9,216, and reaches 0.72x (a 1.38x loss)
at n=36,864. The win below the crossover is SciPy's fixed ~40 µs of interpreted
per-iteration bookkeeping; the loss above it is our third sparse matvec per
iteration. This licenses no size-general qmr claim and no claim that our qmr
kernel is faster — per unknown it is 1.380x slower. Outside this measured shape,
benchmark the actual job.

### 2026-07-30 (cod/SilverRiver) — REJECT: strongest live SciPy GMRES reverses the 5.0350x unpreconditioned whole-job win

**Decision: REJECT whole-job FrankenSciPy promotion.** Bead
`frankenscipy-gugk1`. The requested `36.1770x GMRES` premise was corrected
before measurement: `36.1770x` belongs to the BDF kernel, already converted to
an `8.3326x` BDF whole-job claim. The applicable restart-matched GMRES prior was
`4.8850x` at side 32, `1.5725x` at side 64, and a `0.9397x` loss at side 96.
The mechanism, incumbent screen, predictions, falsifiers, full-output contract,
and chooser wording were committed before timing in `17d852f83`; the complete
harness was committed before timing in `7e6c9555c`.

**Recognizable whole job.** One serial run constructs a nonsymmetric 32x32
steady two-dimensional convection-diffusion-reaction operator (`n=1,024`,
`nnz=4,992`) and twelve compact localized source fields, then performs twelve
public GMRES calls at `restart=20`, `rtol=1e-5`, `atol=0`. The timed boundary
also materializes all 12,288 field values and computes 36 scientific summaries:
domain inventory, east-boundary outlet integral, and source-weighted exposure
for every source. Every repetition reconstructs the matrix, sources, solver,
outputs, and selected preconditioner. Interpreter startup, imports, pipe
transport, backend screening, parity serialization, provenance, and bootstrap
calculation remain outside both timed arms.

**Strongest-incumbent screen, fixed before measurement.** All six live SciPy
1.17.1 candidates converged on all twelve scenarios and passed the complete
output contract:

| SciPy public GMRES configuration | one whole-job screen |
|---|---:|
| `csr_matrix`, none | 122.704894 ms |
| `csr_array`, none | 121.541657 ms |
| `csc_matrix`, none | 133.796602 ms |
| `csc_array`, none | 127.093587 ms |
| `csr_matrix`, Jacobi `LinearOperator` | 123.712881 ms |
| `csc_matrix`, one `spilu` reused for 12 RHS | **10.511853 ms** |

The fastest unpreconditioned incumbent was `csr_array`; the headline incumbent
was the much stronger valid `csc_matrix` plus one amortized default `spilu`.
Direct sparse solvers were intentionally outside this GMRES-configuration
comparison.

**The pre-registered mechanism survives when configuration is held fixed.**
Against unpreconditioned `csr_array`, every scenario has exact iteration parity:
`100,93,94,125,99,106,133,123,100,93,94,125`. FrankenSciPy p50 is
`24.073993 ms`; SciPy p50 is `121.314983 ms`.
The bootstrap-median CI for SciPy / FrankenSciPy is
**`5.0350x [5.0239, 5.0482]`**, a DECIDED FrankenSciPy win. This is the
predicted repeated tax from SciPy's interpreted Arnoldi/Givens bookkeeping,
surviving operator construction, twelve right-hand sides, and postprocessing.
It is not the chooser because it compares against a weaker eligible
configuration.

**Headline against the strongest valid incumbent: DECIDED FRANKENSCIPY LOSS.**
The selected `spilu` arm reduces the twelve iteration counts to
`3,3,3,3,3,3,3,2,2,2,3,2`. FrankenSciPy whole-job p50/p95/p99 is
`24.170874/27.454359/27.454359 ms`; SciPy is
`10.293884/11.956666/11.956666 ms`. **Incumbent ratio: SciPy / FrankenSciPy =
0.4259x**, with bootstrap-median 95% CI **`[0.4211, 0.4409]`**. Equivalently,
SciPy is about **2.348x faster**. The durable `>=3x` FrankenSciPy whole-job
claim is false.

Both independent same-invocation A/A null controls cleared the corrected 2x
null-margin gate: FrankenSciPy A/A median `0.999987`, CI
`[0.997215,1.010290]`; SciPy A/A median `0.999683`, CI
`[0.998048,1.008259]`. The effect CI excludes one, effect deviation beats twice
the larger null half-width and the stricter endpoint margin, and both null
medians are within 2% of one. The null-CI-straddle veto remains disabled
telemetry only. CV is provenance only, never the decision gate
(`4.467%` for headline ratios; `1.802%/2.625%` for the two nulls).
`null_margin=2x`.

**Full result conformance passed.** The reconstructed inputs matched at SHA-256
`a1a08c3456d7cd030f32f23cc4ec7f53d4fd23999ccbc29f795e57c9cfe837ae`.
All 12/12 selected SciPy solves converged; all 12,288 field values and 36
summaries were compared. Selected-arm maximum absolute difference was
`6.499e-5`, relative L2 difference `5.343e-6`, maximum component-scaled error
`0.1603`, maximum summary-scaled error `0.1140`, and tolerance mismatches
`0`. Numerical-stability and tolerance contracts were not weakened.

**Mechanism and prediction scorecard.** P1, P2, P3, and P4 were confirmed:
the same-configuration win exceeds 3x, unpreconditioned trajectories match
exactly, amortized `spilu` wins the screen, and that incumbent reverses the
headline. P5 was falsified: solve-only p50s were `23.971211 ms` versus
`6.234868 ms`, so non-solver fractions were `0.83%` for FrankenSciPy but
`39.43%` for SciPy, not below 25% in both arms. The explanation for the
reversal is counted algorithmic work, not timing noise: reusable ILU reduces
roughly 100 unpreconditioned iterations per source to only 2-3, removing almost
all of the Python per-iteration tax while amortizing factorization across
twelve sources.

**Mandatory provenance.** Literally exclusive `host_identity=threadripperje`;
booking CLAIM message `7131`, RELEASE message `7133`; 64
physical cores, 128 logical threads, `ram_bytes=536069869568`, `numa_nodes=1`;
`requested_threads=1`, `actual_observed_worker_threads=1` for each engine,
`affinity=127`, `cpuset_logical_cap=1`; runtime-detected ISA
`sse2,sse4_2,avx2,fma,bmi2,vaes` with `avx512f=false`;
`scaling_governor=performance`. Host-wide-quiescence-pre=clear
(`maximum_busy_fraction=0.065`) and host-wide-quiescence-post=clear
(`maximum_busy_fraction=0.032`). Strict-remote RCH build on `hz1`, no local
Cargo fallback. FrankenSciPy-engine-sha256=
`5b61eb7e2fe826169b258f93b5637668cfaf233c1ef2770a3be67ae9144577ba`;
SciPy-engine-sha256=
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`;
SuperLU-engine-sha256=
`271ed8a07a651e0234ca0826a32cf4f20f3b972b2af457af46bf8f8f12985e99`.
The live incumbent was genuine SciPy 1.17.1, side-by-side in the same
invocation. Raw artifact SHA-256
`a929b187e488613fe383d811f88360420bf06cde7903ef7eba04e2a4a96c8e0a`:
`tests/artifacts/perf/2026-07-30-gmres-real-job-vs-scipy/bench_stdout_stderr.txt`.

**Concrete retry predicate:** do not repeat this exact serial 32x32,
twelve-source, restart-20 GMRES job. Reopen only after (a) a change to SciPy's
GMRES, SuperLU, or `spilu` engine; (b) a different named matrix, source-count,
or preconditioner-reuse regime; or (c) a FrankenSciPy solver or preconditioner
change. Preserve the six-arm strongest-incumbent screen, complete field and
summary parity, exact requested/observed threads, exclusive-host booking and
quiescence, dual A/A controls, and corrected bootstrap-median CI gate.

**CHOOSER STATEMENT:** Pick SciPy 1.17.1 GMRES with
`csc_matrix` plus one reused default `spilu` for this serial 32x32,
twelve-source steady convection-diffusion job; the strongest valid incumbent
reverses FrankenSciPy's favorable 5.0350x unpreconditioned result and is about
2.348x faster. SciPy also wins the separately measured unpreconditioned
side-96 / n=9,216 job. Direct sparse solvers, other Krylov methods, other
preconditioners, other matrices, other sizes, and other thread counts remain
unmeasured.

### 2026-07-31 (cod/SilverRiver) — `solve_ivp_many` 128-trajectory whole-job WIN at 32 threads; post-16 cap mechanism FALSIFIED

Commit `6b1bc44ea` pre-registered the exact completion job, balanced outer
order, scoped-worker mechanism, and falsifier before building or timing.
Exclusive `trj` CLAIM `7217` / RELEASE `7230` then captured two 11-round
occurrences per thread cell against live SciPy 1.17.1.

The strongest admissible cell was 32 requested and 32 actual observed
FrankenSciPy workers versus one actual SciPy worker. All 128 deterministic
Lotka-Volterra trajectories completed under RK45, `[0,10]`, `rtol=1e-8`,
`atol=1e-10`, `t_eval=None`. Whole-batch p50 was `1.892562 ms` versus
`877.111685 ms`. **Incumbent ratio: SciPy / FrankenSciPy =
464.488412x**, bootstrap-median CI **`[432.012878,481.458752]`**.

The corrected gate passed: Franken null median `0.986495`, CI
`[0.974161,1.231786]`; SciPy null median `1.002244`, CI
`[0.992427,1.019093]`; both medians were within 2%, the effect CI excluded
one, and its deviation cleared twice the larger half-width and the stricter
endpoint margin. Null-CI straddling was telemetry only. CV `14.503%` was
provenance only.

The separate scaling mechanism is **FALSIFIED (`0/3`)**. Effects against
pooled 16-thread time were 32/16 `1.064833x`
`[0.973198,1.184568]`, 64/16 `1.921581x`
`[1.743597,2.073227]`, and 128/16 `3.467878x`
`[3.179610,3.790845]`. Although the latter two CIs exclude one, pooled
16-thread Franken A/A median `1.023360` violates the registered 2% clause,
and 32/16 includes one. `frankenscipy-ldx0f` therefore ships no thread cap.

All 128/128 trajectories reached `t=10`; histories were finite and positive;
all 256 final components matched to max absolute `6.573e-14` and scaled
error below `0.001`; both maximum invariant drifts were `1.212e-7`.
FrankenSciPy/SciPy counted `159,998/160,126` RHS evaluations and both stored
`25,748` points.

**Counted mechanism:** near-equal work rules out a cheaper mathematical
solve. The valid 32-thread boundary is 32 actual parallel compiled
FrankenSciPy workers with inline RHS calls versus one actual SciPy worker
running 128 serial public solves with Python solver-loop and callback tax.

Strict `rch exec --base 6b1bc44ea --clean-overlay --no-overlay` build on
`vmi1156319`; executed ELF self-SHA
`54664c26480945aa63f338a89279fa6816f5c69eea126d7fefbc64edcba86161`;
SciPy engine SHA
`aa16f42cc85fa02769ff00bf93bcdb48b6bf568e2d9f8ce48f9f378e76cf8f09`.
Host `threadripperje`, Threadripper PRO 5995WX, 64C/128T,
`ram_bytes=536069869568`, one NUMA node, AVX2/FMA/BMI2/VAES present,
AVX-512F absent, performance governor. Raw artifact SHA
`f9220a1d593d071d30c5f2bf02095d0ddf49b9d83cbc0f3412633b80ef83d28c`:
`tests/artifacts/perf/2026-07-31-solve-ivp-many-exclusive-rerun/`.

**Concrete retry predicate:** do not repeat the unchanged sweep. Reopen only
after a production worker-lifecycle/pool change, a changed solver engine, or a
different named batch-size/per-trajectory-work regime.

**CHOOSER STATEMENT:** for this exact 128-trajectory completion ensemble on a
32-CPU affinity, pick FrankenSciPy `solve_ivp_many`; do not infer a general
thread cap, because the pre-registered scaling mechanism was falsified.

### 2026-07-31 (cod/TopazGorge) — KEEP: persistent row-band CG workers

**Decision: KEEP. Result class: `CAMPAIGN-WIN`.** Bead `frankenscipy-fn178`
pre-registered the mechanism and falsifiers before timing. Large CSR CG now
creates one scoped worker team per solve, partitions contiguous rows by
cumulative `nnz`, retains row-local
`x/r/Ap` state, and exchanges only the search vector plus fixed-order scalar
reductions across iterations. The prior path created a scoped team on every
matvec; `FSCI_CG_FORCE_ITERATION_SCOPES=1` is the same-ELF control.

On the 512x512 five-point Laplacian (`n=262,144`, `nnz=1,308,672`, 494/494
iterations), 11-round whole-solve p50 fell from **726.860001 ms** to
**167.783036 ms**, a **4.3321x self-speedup**. Live SciPy 1.17.1 in the same
invocation was **670.353635 ms**, so SciPy / FrankenSciPy was **4.3138x** with
bootstrap-median CI **[3.7588, 4.6040]**. Both arms reached true relative
residual `9.870e-6`; relative L2 solution difference was `7.823e-15` and max
absolute difference was `1.614e-11`.

**Legacy incumbent arm: SciPy 1.17.1, side-by-side same-invocation.**
**Incumbent ratio: SciPy / FrankenSciPy = 4.3138x.** A/A null controls were
FrankenSciPy median `0.998760`, CI `[0.891551,1.073132]`, and SciPy median `0.992176`, CI
`[0.950919,1.073097]`; `null_margin=2x`. The bootstrap-median effect CI cleared
the twice-the-null margin. CV is provenance only and did not decide the result.

The smaller 256x256 cell (`n=65,536`, `nnz=326,656`, 330/330 iterations) moved
from **200.383119 ms** to **77.673157 ms** (**2.5798x self-speedup**) and beat
live SciPy's **90.008751 ms** by **1.1648x**, CI **[1.1183, 1.2114]**. Both
cells passed independent same-invocation A/A controls and the corrected
median-CI gate.

Provenance: `host_identity=thinkstation1`; `physical_cores=32`;
`logical_threads=64`; `ram_bytes=231691894784`; `numa_nodes=1`;
`requested_threads=9`; `actual_observed_frankenscipy_worker_threads=9`;
`actual_observed_scipy_worker_threads=1`; `runtime_detected_isa=sse2,sse4_2,avx2,fma,bmi2,vaes`;
`affinity=0-63`; `cpuset_logical_cap=64`; `scaling_driver=amd-pstate-epp`;
`scaling_governor=powersave`; `energy_performance_preference=performance`;
`host_wide_quiescence_pre=clear`; `host_wide_quiescence_post=clear`. OpenBLAS,
OMP, MKL, and NumExpr were capped at one thread for SciPy. The strict-RCH
`release-perf` binary reported **Executed-binary ELF SHA-256:
`9213da66771203406544776e57942bfe52f0b06df9506a0a9ee082d320882842`**.
FrankenSciPy engine SHA-256:
`9213da66771203406544776e57942bfe52f0b06df9506a0a9ee082d320882842`.
SciPy engine SHA-256:
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
The incumbent was genuine live SciPy 1.17.1 with NumPy 2.4.3. Numerical
tolerance contracts were unchanged.

**CHOOSER STATEMENT:** for unpreconditioned CG on these large five-point SPD
systems at `rtol=1e-5`, use FrankenSciPy's default persistent-worker path.

### 2026-07-31 (cod/FrostyCrane) — REVERT: fixed-s=2 communication-avoiding GMRES

**Decision: REVERT. Result class: PROVISIONAL NON-EXCLUSIVE ROUTING EVIDENCE;
NO PERFORMANCE CLAIM.** Bead `frankenscipy-ddvmb`. The pre-registration below
was committed as `554b49a58` before the candidate was built or timed. The fixed
`s=2` prototype preserved the registered numerical contract but lost its
same-ELF screening comparison, and an admissible exclusive 64-core host was not
available. The prototype and its comparator switch were removed byte-for-byte;
no production or harness code from this candidate ships.

**One lever.** For restarted GMRES only, dimensions `n >= 4,096`, and an even
Krylov cycle, form the monomial block `[Aq, A^2q]`, orthogonalize both columns
together with two-pass block classical Gram--Schmidt plus a two-column QR, and
recover the two ordinary Arnoldi Hessenberg columns from the block recurrence.
Apply Givens rotations and test convergence after each logical column, so public
iteration counting and restart semantics remain unchanged. Any rank loss,
non-finite coefficient, failed orthogonality check, or unsupported tail falls
back to the existing classic modified-Gram--Schmidt cycle before mutating the
solution. A hidden same-ELF switch forces that classic path.

**Registered workload and falsifier.** The completion cell is the deterministic
side-96 five-point nonsymmetric convection--diffusion fixture (`n=9,216`,
`nnz=45,696`, diagonal `4.001`, west/east `-1.2/-0.8`, vertical `-1.0`), zero
initial guess, `restart=20`, `rtol=1e-5`, `atol=0`, and live public SciPy GMRES.
The pre-change completion baseline is `38.326807 ms/solve`; the candidate must
reach at most `35.64 ms/solve` (at least `1.07x`) under a balanced same-invocation
candidate/classic comparison. It must also converge in exactly `227` iterations,
report residual at most `1e-5`, and stay within `1e-10` relative L2 of both the
same-ELF classic solution and the live incumbent. Independent A/A nulls and the
corrected bootstrap-median CI gate decide the timing result; CV is provenance
only. A candidate loss or an undecidable result is a revert.

**Profile admission passed.** On production-kernel base
`d95429d0ec4c491f0c501e842f12d87f4a27f182` with only a diagnostic harness
entry point added, executed ELF SHA-256
`db176b9e448f79a8efaa0128ce7cb190380a16f13aa0fccf42b059a2225ffca9`
completed 30 exact solves at 227 iterations and residual `9.872e-6`. From 2,914
`perf` samples with zero lost events, classic MGS projection accounted for
`15.85%`, scalar dot machinery for at least `31.05%`, and serial CSR matvec for
`11.44%`. This clears the bead's registered requirement for an at-least-8%
removable leaf and makes a 7% change plausible. The profile run is diagnostic
only and is not acceptance evidence.

**Prototype and conformance result.** The harness-local prototype implemented
the registered two-matvec monomial block, two-pass block CGS, two-column QR,
ordinary-Hessenberg recovery, per-logical-column Givens updates, and guarded
classic fallback. It was built under strict remote-only RCH placement on
`vmi1227854`; the executed ELF SHA-256 was
`73c1de577dd290b379e0fa2afc55dca4254ffd14efe2739bc629d64875215b3c`.
At side 96 it matched the classic path at exactly `227/227` iterations and true
residual `9.872e-6`; candidate versus classic had maximum absolute solution
difference `6.253e-13` and relative L2 `1.154e-15`. Candidate versus live SciPy
had maximum absolute difference `3.411e-13` and relative L2 `4.898e-16`.

**Routing screen, explicitly inadmissible as a benchmark claim.** One
same-invocation run on `thinkstation1`, pinned to logical CPU 63 with one actual
thread per arm, used genuine SciPy 1.17.1 / NumPy 2.4.3 and SciPy engine SHA-256
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
Host-wide exclusivity was deliberately waived because unrelated work occupied
other CPUs, so these numbers may route work but may not support a keep or public
speed claim. The candidate p50 was `43.630077 ms`, classic p50 was
`40.876849 ms`, and classic/candidate was `0.9347x` with bootstrap-median CI95
`[0.9323, 0.9456]`: approximately a `6.99%` candidate loss. Candidate A/A was
`0.998971x`, CI95 `[0.990316, 1.006642]`; classic A/A was `0.999206x`, CI95
`[0.992897, 1.002299]`. The accompanying live-incumbent observation was ours
`43.997441 ms`, SciPy `38.603898 ms`, SciPy/ours `0.8778x`, CI95
`[0.8576, 0.8896]`; its null controls were ours `0.998755x`, CI95
`[0.959497, 1.009166]`, and SciPy `0.992971x`, CI95
`[0.989958, 1.008992]`. An unwaived retry failed closed before timing because
the host was busy. The canonical 64-core host was already booked and also had
unrelated full-core work, so no external host state was changed and no timing
was promoted to acceptance evidence.

**Why the lever lost and rollback proof.** On this one-thread execution there
is no inter-worker reduction synchronization for `s=2` to collapse. The extra
block reorthogonalization, QR, and strided vector work therefore add cost while
preserving the same 227 logical iterations. The diagnostic binary remains only
under `/data/tmp`; the tracked benchmark harness is identical to `HEAD`, and
the production sparse solver was never edited for this experiment.

**Concrete retry predicate if rejected:** do not repeat fixed-s=2 monomial
block Arnoldi on this cell. Reopen only for a stabilized polynomial/Newton
basis, a genuine multi-worker reduction boundary, or a changed matrix/restart
regime; retain exact iteration parity, solution/residual conformance, the
same-ELF classic arm, the genuine live SciPy arm, and independent A/A controls.

### 2026-07-31 (cod/FrostyCrane) — EVIDENCE-ONLY CLOSE: component-parallel reusable ILU solve

**Status: REVERT / NO-SHIP; NO ADMISSIBLE TIMED SAMPLE.** Bead
`frankenscipy-80ds7`. The mechanism, eligibility gate, completion workload,
conformance contract, timing gate, fallback, and rollback below were fixed in
commit `2239b56eb` before solver source was edited or any candidate timing was
taken.

**One lever.** While constructing a `SparseIluFactorization`, derive the exact
contiguous connected-component boundaries of the combined L/U structural graph.
On a reusable `.solve(b)`, dispatch those independent factor components to a
bounded worker pool. Each worker owns a contiguous L/U row band and local RHS/
solution buffer, preserving the serial accumulation and division order inside
every row. A factor with fewer than four components, fewer than `2^20` stored
triangular entries in total, or less than `2^14` entries per component on
average uses the unchanged serial forward/back substitution. A hidden same-ELF
switch forces the serial path. No factor values, pivot rule, ordering, drop
policy, public tolerance, or result layout may change.

**Completion cell.** Construct 64 independent strictly diagonally dominant
tridiagonal blocks of 16,384 rows each (`n=1,048,576`, `nnz=3,145,600`) with a
deterministic RHS. Each block's factors and work vector fit comfortably in a
worker-local cache slice. Factorization and fixture construction are outside
timing; the timed operation is exactly one application of the already-built
preconditioner. Run on a booked host with affinity restricted to 64 physical
cores and record requested and observed workers. The three arms occur in one
invocation: default candidate, forced-serial same ELF, and genuine live SciPy
1.17.1 `spilu(A).solve`. Every arm gets an independent A/A null control, and
the artifact records both engine SHA-256s plus pre/measurement/post host-wide
quiescence.

**Acceptance and falsifier.** Candidate output must be bit-identical to the
same-ELF serial result, have infinity-norm true residual at most `1e-12`, and
stay within `1e-10` relative L2 of live SciPy. A maintenance KEEP requires the
bootstrap-median CI95 lower bound for serial/candidate to clear `1.20x` after
the corrected null-width gate. A competitive claim additionally requires the
CI95 lower bound for SciPy/candidate to clear its null-adjusted `1.0x` gate.
Numerical drift, a candidate loss, an undecidable host result, or a connected-
factor guard regression is a revert.

**Correctness evidence before measurement.** The candidate implemented the
registered exact disconnected-component detector, guarded serial fallback,
bounded reusable worker pool, and same-ELF force-serial arm. Focused strict-
remote checks, sparse differential conformance, 409 unit cases, 56 metamorphic
cases, formatting/diff checks, focused Clippy, and UBS passed. Smaller routing
cells produced bit-identical candidate/serial solutions and conforming true
residuals. Matching live SciPy exposed one oracle-fixture issue before timing:
SciPy's default `area` drop rule discarded original tridiagonal entries even at
`drop_tol=0` and produced a roughly `4.19e-1` residual. Selecting public
`drop_rule="basic"` restored the intended structural ILU(0) comparison; no
FrankenSciPy tolerance or acceptance threshold was relaxed.

**64-core admission result.** The frozen candidate ELF was
`a905050deba6e2a1b7a02e29f8c61d04ec14f6e593fff181afd0ecf26815bba0`.
The first booked `thinkstation2` attempt stopped before launch because the
system Python lacked SciPy. A second claim used an immutable staged CPython
3.13-compatible SciPy 1.17.1 / NumPy 2.4.3 tree; the genuine SuperLU extension
SHA-256 was
`271ed8a07a651e0234ca0826a32cf4f20f3b972b2af457af46bf8f8f12985e99`.
The harness confirmed 64 physical cores, affinity `0-63`, one NUMA node, and
the frozen ELF identity, then failed closed at the pre-measurement host-wide
quiescence gate: `cpu32=86.7%` and `cpu102=100.0%` exceeded the registered 20%
busy ceiling after peer `git fsck` and Rust compilation appeared. The sole
identified maintenance process was resumed by the exit trap. The abort log is
`/data/tmp/frankenscipy-80ds7-ts2-claim8352.log`, SHA-256
`dd195dd912a8a7b342454f29b70cee3dbd2800dfedf22c1f6436bf46c1363179`.
The alternate 64-core host remained occupied by live peer work, so it was not
mutated or used. No completion fixture, timed solve, A/A sample, CI, or
candidate/incumbent ratio was produced. Earlier four-core non-exclusive routing
observations are not acceptance evidence and support no speed claim.

**Decision and rollback.** The preregistration explicitly maps an undecidable
host admission to REVERT. All candidate production, oracle, and benchmark code
was therefore removed, restoring those files byte-for-byte to the preregistered
`main` state. Remotely staged immutable artifacts were left in place because
this session had no authorization to delete files; they are not evidence.

**Rollback and concrete retry predicate.** Rollback removes the component
metadata, parallel dispatch/helper, and force-serial switch, restoring the
existing serial `.solve`. If rejected, do not repeat this exact 64-by-16,384
component fixture unless the worker topology or SciPy engine changes. Reopen a
generic level-scheduled triangular solver only after a real matrix exhibits at
least eight ready rows per dependency level and profiling shows barrier time
below 20% of solve wall time.

### 2026-07-31 (cod/FrostyCrane) — PRE-REGISTERED: guarded recursive-residual QMR

**Status: REJECTED / NO-SHIP (post-admission contamination; no timed
verdict).** Bead `frankenscipy-8l8r1.174`.
This is the first self-generated lever after the supplied sparse/solver list:
the worst previously measured whole-job ratio is the side-192 QMR cell at
`0.7232x` SciPy/FrankenSciPy, and the incumbent avoids a cost that dominates
our profile. No candidate timing exists at this point.

**Whole-job profile and incumbent-cost filter.** The exact side-192
convection-diffusion fixture has `n=36,864`, `nnz=183,552`, `rtol=1e-5`, and
one actual worker per arm. A strict-remote `release-perf` build was placed on
`vmi1227854`; the frozen ELF is
`1336f527ecba7b5f28d74ba6ef92b57865992cd06258d7119ecb49ecb03a269c`.
`perf 6.17.13` collected 34K `cycles:P` samples with zero lost samples on an
AMD Threadripper PRO 5975WX. The data file SHA-256 is
`c268ec14e87d9c943471c142ab10e60122692f28c7eb95f81372c9fd9637fff3`.
Ranked self-time was:

| rank | arm / symbol | whole-job cycles | within-arm cycles | incumbent pays the same cost? | disposition |
|---:|---|---:|---:|---|---|
| 1 | FrankenSciPy `csr_matvec_into_impl` | 45.35% | 69.69% | only two of our three calls | **structural gap / selected** |
| 2 | FrankenSciPy QMR recurrence body | 10.11% | 15.53% | yes | shared work / move on |
| 3 | SciPy `csc_matvec_thunk` | 8.27% | 23.68% | yes, transpose matvec | incumbent cost / move on |
| 4 | SciPy `csr_matvec_thunk` | 6.12% | 17.53% | yes, forward matvec | incumbent cost / move on |
| 5 | SciPy multiply/memcpy/add/subtract/dot kernels | 10.45% combined | 29.93% | yes | shared vector work / move on |

The selected difference is exact: both recurrences require `A p` and
`A^T q`; FrankenSciPy additionally materializes `b - A x` after every update,
while SciPy carries that residual recursively. The non-exclusive profiled run
reported FrankenSciPy `790.466266 ms` p50 and live SciPy 1.17.1
`423.234742 ms` p50, SciPy/FrankenSciPy `0.5245x`, but the FrankenSciPy A/A
median was outside the 2% bias bound. Those walls are routing evidence only,
not a baseline verdict. The run used the active reduction-ILP overlay on
`linalg.rs`, which changed FrankenSciPy's trajectory from the historical 895
iterations to 1,287 while SciPy remained at 883; therefore every decision must
use the same final ELF for candidate and classic control.

**One lever.** Maintain the QMR residual with the same algebraic update as the
solution: if `d_n = eta_n p_n + c_n d_(n-1)`, carry
`r_n = r_(n-1) - (eta_n A p_n + c_n s_(n-1))`. Test its norm without a third
matvec. When this recursive norm first crosses tolerance, compute one exact
`b - A x` residual. Return only if that exact residual satisfies the existing
contract; otherwise replace the carried residual with the exact one and
restart the Lanczos/QMR state from the current `x`, retaining the original
total iteration budget. Breakdown and exhausted-budget exits continue to
report an exact residual. A hidden same-ELF switch forces the existing
per-iteration true-residual path.

**Predictions fixed before implementation.** On the historical recurrence the
recursive trigger should occur near live SciPy's 883 iterations and require one
terminal true-residual matvec, versus 895 per-iteration residual matvecs in the
classic arm. Under the final reduction implementation the exact count may
change, but candidate and classic must share that implementation and the
candidate must never report convergence from the recursive estimate alone.
Removing roughly one-third of iteration matvecs predicts at least `1.25x`
same-ELF speedup at side 192. The live-SciPy result is genuinely uncertain:
FrankenSciPy removes the extra matvec, but SciPy's compiled sparse kernels may
still have lower per-nonzero cost.

**Completion cell and acceptance.** In one invocation on a host admitted by
the existing host-wide quiescence gate, run at least 21 alternating rounds of:
(1) default guarded-recursive candidate, (2) forced-classic same ELF, and
(3) genuine live SciPy 1.17.1 QMR, all on the exact side-192 fixture with one
observed worker. Record both engine SHA-256s, exact iteration/matvec counts,
terminal true residuals, candidate restart/check counts, and an independent
A/A null for every timed arm. Candidate and classic must both converge with
true residual at most `1.25e-5`; candidate/live-SciPy relative L2 must remain at
most `5e-4`, every component must satisfy the harness tolerance, and focused
adversarial/unit/differential conformance must pass. KEEP requires the
bootstrap-median CI95 lower bound for classic/candidate to clear `1.20x` after
the corrected null gate. A competitive statement additionally requires the
null-corrected SciPy/candidate lower bound to clear `1.0x`. Any false
convergence, exhausted reliable restart, undecidable admission, or candidate
loss is a revert.

**Candidate correctness and mechanism evidence.** Strict-remote execution
passed all eleven QMR-filtered unit/metamorphic cases and the live-oracle
`diff_sparse_iterative_bicg_cgs_qmr_lgmres` test. On the completion fixture,
candidate and forced-classic solutions were bit-identical, both converged in
1,287 iterations with recomputed relative true residual `8.230e-6`, and the
candidate/live-SciPy relative L2 difference was `5.507e-8` with zero component
tolerance mismatches. The guarded path used 1,289 forward plus 1,287 transpose
matvecs, including two true-residual matvecs and one successful recursive
trigger; the same-ELF classic used 2,575 forward plus 1,287 transpose matvecs,
including 1,288 true-residual matvecs. No reliable restart fired. Thus the
candidate removed exactly 1,286 matrix applications without weakening the
terminal residual contract.

**Completion attempt and fail-closed result.** The frozen candidate ELF was
`68aadeabea1fa55da7cd91a04ff7311a3690191ddecbb4ae4e5233cc1f28c14e`;
the genuine SciPy 1.17.1 QMR engine was
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`.
The 64-physical-core `thinkstation2` host ran both engines with one observed
worker and passed the pre and measurement quiescence samples across all 128
logical CPUs (maximum busy fractions `0.100` and `0.067`). After all 21 timed
triplets and their independent A/A controls had executed, peer work appeared
on `cpu15` and `cpu73`; both were `100.0%` busy in the post sample. The harness
therefore aborted before emitting any p50, A/A, bootstrap CI, ratio, or verdict.
The captured log is
`/data/tmp/frankenscipy-8l8r1-174-qmr-ts2-21r.log`, SHA-256
`c66c593a1603f634766f2669df9775a3c25da9f40f7263650927b9e6661e81ef`.
A five-round, explicitly non-exclusive routing smoke on `ovh-a` suggested the
mechanism could be material (`1.4268x` classic/candidate median), but its waiver
permanently classified it as provisional and it supports no performance claim.

**Decision and rollback.** **REVERT / NO-SHIP.** The registered post gate made
the completion invocation undecidable, so the candidate production path,
same-ELF switch, diagnostics, benchmark triplet, and added unit fixture were
restored byte-for-byte to the preregistered `main` state. No timing result was
laundered from samples that the harness correctly withheld.

**Rollback and reject predicate.** Rollback restores exact residual
recomputation after every QMR update and removes the hidden control and its
instrumentation. If rejected, do not retry this unpreconditioned
convection-diffusion cell with another recursive-residual spelling. Reopen only
for a preconditioned matrix class where profiling shows residual checks above
20% of wall time, or for a provably bounded residual-gap estimator that avoids
both per-iteration verification and restart churn.

### 2026-07-31 (cod/FrostyCrane) — PRE-REGISTERED: shared-nothing batched QMR

**Status: PRE-REGISTERED / NOT YET IMPLEMENTED OR TIMED.** Bead
`frankenscipy-8l8r1.175`. This is a new structural lever, not a retry of the
rejected recursive-residual spelling: widen the kept independent-RHS scheduler
from `gmres_batch` to public `qmr_batch`.

**Whole-job profile and incumbent-cost filter.** The preceding side-192 QMR
profile ranked CSR matvec first at `69.69%` of FrankenSciPy cycles, recurrence
work second at `15.53%`, and vector kernels after that. Live SciPy pays the same
two required Lanczos matvecs and the same recurrence/vector classes, so those
shared costs are not this lever. What the live single-RHS API cannot pay or
exploit is cross-scenario scheduling: a whole job with 64 independent systems
invokes SciPy QMR 64 times serially, while each FrankenSciPy solve owns private
vectors, scalars, convergence state, and output. There is therefore no Krylov
dependency between right-hand sides and no synchronization inside the new
outer worker partition. The already-kept `gmres_batch` result establishes the
mechanism on this codebase: its exact-width persistent pool delivered `8.05x`
over its same-ELF sequential control. No QMR-batch timing exists yet.

**One lever and prediction.** Add `qmr_batch(a, rhses, initial_guesses,
options)` and reuse the persistent exact-width iterative batch pool. Preserve
input order and call the existing scalar `qmr` unchanged in each worker. The
worker budget remains bounded by affinity-visible parallelism divided by any
inner sparse-matvec team, preventing nested oversubscription; empty batches and
initial-guess cardinality retain explicit behavior. On 64 small systems whose
inner matvec stays serial, predict at least `8x` same-ELF batch speedup. The
live-incumbent result is predicted to be a decided win because SciPy 1.17.1 has
no corresponding public batched QMR scheduler and is pinned to one worker.

**Frozen completion cell.** Build one release-perf ELF, then in one admitted
invocation compare (1) default `qmr_batch`, (2) the same ELF with a hidden
forced-sequential batch switch, and (3) genuine live SciPy 1.17.1 public `qmr`
called 64 times. Use the nonsymmetric side-32 convection-diffusion CSR
(`n=1,024`, `nnz=4,992`), 64 independent copies of the deterministic
`1 + 0.01*(i%17)` RHS, zero initial guesses, strict mode, `rtol=1e-5`,
`atol=0`, and at least 21 interleaved rounds on a cpuset containing exactly the
host's 64 physical cores. Construction, RHS cloning, serialization, parity,
pool warm-up, and provenance are outside timing. Record the frozen ELF and
SciPy-engine SHA-256s, requested and observed worker counts, p50/p95/p99 whole
batch wall time, raw samples, and independent A/A controls for all three arms.

**Correctness and decision gates.** Every candidate result must equal its
same-ELF sequential counterpart exactly, including order, solution, convergence,
iteration count, and reported residual. All 64 true residuals must be at most
`1.25e-5`; the live-SciPy solution comparison must have relative L2 at most
`5e-4` and zero components outside `10*rtol*max(1,abs(scipy))`. Focused unit,
metamorphic, and live differential conformance must pass. A maintenance KEEP
requires the corrected-null bootstrap-median CI95 lower bound for
sequential/candidate to clear `1.20x`. A competitive statement additionally
requires the corrected-null SciPy/candidate lower bound to clear `1.0x`.
Numerical drift, worker oversubscription, host admission failure, candidate
loss, or an undecidable maintenance gate is a revert. After a reject, retry
only on a distinct batch cardinality or matrix family justified by a new
whole-job profile.

**Implementation and correctness evidence.** Candidate commit `51a3dd123`
generalized the already-kept persistent GMRES batch pool and exposed public
`qmr_batch` without changing scalar QMR. The exact clean-baseline
`release-perf` ELF was
`fe3414406426a92b2f1bff5df5bc7ecffc79b9c2d9426dfab84722d1e7ff6ee2`;
the live-oracle script SHA-256 was
`a41cf82d2aaf3ef091d5310cdb5812bac953678853825cf5eca4a487499a2f9e`.
Strict-remote evidence passed all three new batch tests, the complete
`fsci-sparse` library suite (`410` passed, `0` failed, `4` ignored), the direct
live-SciPy iterative differential test, and workspace `cargo check
--workspace --all-targets`. Candidate-scoped clippy passed after allowing only
the two established untouched lint classes; the mandatory workspace clippy
attempt stopped on existing `fsci-io` and `fsci-opt` warnings. The broad
workspace test attempt reached the unrelated pre-existing
`mr_eig_trace_identity` metamorphic test and remained there for more than 20
minutes; no sparse test failed.

**Completion admission and fail-closed result.** Queue message `8447`, ready
handoff request `8456`, urgent probe `8464`, and stale-census notice `8473`
preserved the already-published Mermaid priority and completed two full
15-minute silence cycles before any host access. The permitted read-only census
at `2026-08-01T02:10:36Z` found three unrelated full-core jobs: PID `2286872`
`search_z3.py`, PID `2334362`
`g0014_q480_edge69_sd2_fiber_factor_scout.py`, and PID `1355035`
`g0014_roll_full_word_cylinder_probe`, plus PID `2359347` running
`git -C /data/projects/asupersync fsck --no-dangling` at `100%`. A three-second
all-CPU sample reported `0.00%` minimum idle. No claim was taken, no artifact
was transferred, no process or service was changed, and no candidate,
same-ELF control, or live-SciPy timing ran. Agent Mail census closeout `8474`
records the no-claim terminal path.

**Decision and rollback.** **REVERT / NO-SHIP.** The pre-registered rule made
host admission failure a revert, so the production API, shared-pool
generalization, hidden switch, tests, and completion harness were manually
restored byte-for-byte to preregistration commit `2889be0b8`. There is no QMR
batch speedup or competitive claim.

**Concrete retry predicate.** Do not rebuild or rerun this exact side-32,
64-identical-RHS cell. Reopen QMR batching only for a distinct batch cardinality
or matrix family selected by a fresh whole-job profile, and only after a
canonical handoff plus preflight census proves an exclusive 64-physical-core
window long enough to finish the registered invocation.

### 2026-07-31 (cod/FrostyCrane) — KEPT: dense Radau LU-pair reuse

**Status: KEPT / COMPETITIVE LIVE-SCIPY WIN.** Bead
`frankenscipy-8l8r1.176`. This was the worst decided live-incumbent ratio after
filtering the whole-job profile for costs that SciPy does not pay at the same
multiplicity.

**Whole-job profile and incumbent-cost filter.** The frozen dense-allpairs
`n=128` Radau fixture (`t=[0,1]`, `rtol=1e-8`, `atol=1e-10`, `jac=None`) ran
FrankenSciPy in `469.646848 ms` and genuine live SciPy 1.17.1 in
`193.416274 ms`: SciPy/FrankenSciPy `0.4118x`, CI95 `[0.4068,0.4135]`, with a
full-vector difference of `0.016` tolerance units. The profiled release-perf
ELF SHA-256 was
`d926691e782bda223d6c66a51c95bde803b183c581ecd62bab51ca1576545dad`; the
`perf.data` SHA-256 was
`a406a52e58f39749121d28a62db4967ca50095ddcf5b9d56d1b992102bd5f82d`.
FrankenSciPy reported `nfev=4252`, `njev=1`, `nlu=1178`, `steps=590`; SciPy
reported `nfev=4774`, `njev=2`, `nlu=74`, `steps=681`. The combined whole-job
profile assigns `41.06%` of cycles to FrankenSciPy complex-LU construction and
`6.66%` to real-LU construction. SciPy does not pay the same factorization
multiplicity: it retains `LU_real` and `LU_complex` whenever accepted predicted
step growth is below `1.2`. Therefore repeated factor construction is a
structural gap rather than shared work.

**One lever and invalidation contract.** On the dense path only, when an
accepted step's predicted growth factor is below `1.2`, hold the step size and
retain that step's exact real/complex LU pair for the next same-size step.
Rejection, Newton failure, Jacobian refresh, boundary truncation, any step-size
change, diagonal mode, and factorization or solve failure invalidate or bypass
reuse. Add a hidden same-ELF force-original switch which restores the current
always-grow/always-rebuild policy without changing the measured executable.
No tolerance, Newton, error-estimation, or Jacobian policy may change.

**Frozen completion cell and admission.** Build one clean release-perf ELF and,
in one admitted invocation on one quiet worker with the process pinned to one
physical core, compare (1) default candidate, (2) forced-original from the same
ELF, and (3) genuine live SciPy 1.17.1. Run at least 21 interleaved rounds of
the exact fixture above and independent A/A controls for all three arms. Reject
the invocation before timing if executable or oracle identity is missing, CPU
affinity differs between arms, another benchmark owns the pinned core, or the
host is under material competing load. Record ELF and oracle SHA-256s, raw
samples, p50/p95/p99, `nfev`/`njev`/`nlu`/steps, pairwise full-vector
tolerance-scaled differences, and bootstrap-median ratio CIs corrected by the
corresponding A/A nulls.

**Correctness and decision gates.** Every arm must finish successfully with
status zero. Candidate/control and candidate/live-SciPy final vectors must meet
the existing exact residual and componentwise tolerance contracts; no tolerance
weakening is admissible. Candidate `nlu` must be at most `294`, a reduction of
at least 75% from `1178` and materially toward SciPy's `74`. A maintenance KEEP
requires the corrected-null original/candidate bootstrap-median CI95 lower bound
to be at least `1.20`. A competitive claim additionally requires the
corrected-null SciPy/candidate CI95 lower bound to exceed `1.0`. Numerical drift,
an invalidation defect, `nlu` above threshold, host admission failure, a loss,
or undecidable maintenance evidence reverts the lever. After rejection, retry
only with a different step-prediction policy supported by a new whole-job
profile.

**Implementation and correctness evidence.** Candidate commit `79aab4ea8`
retains the exact real/complex factor pair only across an accepted dense step
whose controller holds the signed step size; rejection, Jacobian refresh,
boundary truncation, diagonal mode, a changed step, and the hidden
forced-original arm all discard or bypass it. The focused reuse test passed
strict-remotely, as did all `279` `fsci-integrate` library tests and the
feature-enabled all-target check. Candidate-scoped no-deps clippy passed after
allowing only four established untouched lint classes. The mandatory workspace
clippy attempt stopped on pre-existing `fsci-opt` warnings, and workspace
format checking stopped only on peer-owned
`diff_sparse_iterative_solvers.rs`; both edited Rust files themselves were
rustfmt-clean and UBS returned zero.

**Frozen completion evidence.** Strict-remote worker `vmi1152480` built source
commit `79aab4ea8eae28c9152870ed89752710b2b6bd0d` into ELF SHA-256
`881be56a31e4aaab6a89c81a7ecc5b675772e54da83d6e420154767af1e01cde`.
The genuine live SciPy 1.17.1 engine SHA-256 was
`aa16f42cc85fa02769ff00bf93bcdb48b6bf568e2d9f8ce48f9f378e76cf8f09`.
The two named artifacts are
`frankenscipy-engine-sha256=881be56a31e4aaab6a89c81a7ecc5b675772e54da83d6e420154767af1e01cde`
and
`scipy-engine-sha256=aa16f42cc85fa02769ff00bf93bcdb48b6bf568e2d9f8ce48f9f378e76cf8f09`;
the first is the executed-binary ELF SHA-256. **Legacy incumbent arm: SciPy
1.17.1, side-by-side same-invocation.**
One invocation on `thinkstation1` pinned all arms to CPU 31 with one observed
worker. Its SMT sibling was CPU 63; their pre-run idle averages were `95.79%`
and `98.00%`, and post-run averages were `99.80%` and `99.20%`. The process
reported exactly one affinity-visible CPU. The captured completion log is
`/data/tmp/frankenscipy-8l8r1-176-79aab4ea8-dense-radau-live-20260731T2304.log`,
SHA-256
`4baa0eca06d22a244b8f48dd74aeafa6e1a24985eb9be6d3f542d7e59a116529`.

Mandatory hardware/thread provenance was
`host_identity=thinkstation1 physical_cores=32 logical_threads=64
ram_bytes=231691894784 numa_nodes=1 requested_threads=1
actual_observed_worker_threads=1
runtime_detected_isa=sse2=true,sse4_2=true,avx2=true,fma=true,bmi2=true,vaes=true,avx512f=false
affinity=31 cpuset_logical_cap=1 scaling_governor=powersave`.
`host-wide-quiescence-pre=clear`: the five-second all-CPU sample averaged
`97.32%` idle and CPU 31 plus sibling 63 averaged `95.79%/98.00%` idle.
`host-wide-quiescence-post=clear`: the immediate pinned-pair sample averaged
`99.80%/99.20%` idle and the follow-up all-CPU sample averaged `96.42%` idle;
the latter artifact is
`/data/tmp/frankenscipy-8l8r1-176-post-host-wide-20260731T2308.mpstat`,
SHA-256
`6d1ebbba19221a4127fe2e3d63d7c52d298c189cb5c75fcfa91a2db7724df801`.

Across 21 interleaved rounds, candidate p50/p95/p99 was
`60.474/61.408/62.348 ms`, forced-original was
`464.948/466.527/475.968 ms`, and live SciPy was
`190.956/205.739/217.287 ms`. Independent A/A medians were candidate
`1.000476` (CI95 `[0.998452,1.001772]`), original `1.000320`
(CI95 `[0.999341,1.000719]`), and SciPy `0.999080`
(CI95 `[0.994488,1.000131]`). The null-corrected
forced-original/candidate median was `7.6876x`, CI95
`[7.6558,7.7178]`; null-corrected live-SciPy/candidate was `3.1565x`, CI95
`[3.1215,3.1868]`. Bootstrap-median CI `[3.1215,3.1868]`: DECIDED.
**Incumbent ratio: SciPy / FrankenSciPy = 3.1565x.**
The 2x A/A-null margin was `1.0036` for original/candidate and `1.0111` for
SciPy/candidate; both CI lows clear those margins. A/A CVs were
`0.662%/0.283%/2.557%`; **CV is provenance only** and did not decide the
result. Raw samples are retained in the hashed log.

**Conformance and mechanism result.** Candidate/control maximum full-vector
difference was `0.007430` tolerance units and candidate/live SciPy was
`0.008334`, both below the registered `1.0` limit. Candidate completed with
`nfev=4595`, `njev=1`, `nlu=80`, `steps=639`, and `598` counted reuse hits;
forced-original completed with `nfev=4252`, `njev=1`, `nlu=1178`,
`steps=590`, and zero reuse hits; SciPy completed with `nfev=4774`, `njev=2`,
`nlu=74`, and `steps=681`. Candidate `nlu=80` clears the registered maximum
of `294` and lands within six constructions of the incumbent.

**Result class: CAMPAIGN-WIN. Decision: KEEP.** The corrected maintenance
bootstrap-median CI95 low `7.6558x` clears
`1.20x`, and the corrected live-incumbent CI95 low `3.1215x` clears `1.0x`.
This is a competitive live-SciPy win with the registered numerical contracts
intact.

### 2026-07-31 (cod/FrostyCrane) — PRE-REGISTERED: one-pass lower-triangle `dsymv` for dense `eigh`

**Status: KEEP (maintenance); candidate commit `5ac21348f`.** Bead
`frankenscipy-8l8r1.177`. After the dense-Radau keep, the stale sparse
`eigsh` loss and the already-shipped GMRES batch route were removed from the
map. The next untouched decided loss is dense full-vector `eigh` at `n=512`.
A current diagnostic run of the exact deterministic symmetric fixture produced
FrankenSciPy `140.401992 ms/call` and genuine live SciPy 1.17.1
`33.194900 ms/call`, a routing-only SciPy/FrankenSciPy ratio of `0.2364x`.
This is not an acceptance baseline because it has no interleaved A/A controls.

**Whole-job profile and incumbent-cost filter.** The strict-remote
`release-perf` profiler binary was built on `vmi1152480`; executed ELF SHA-256
`0fa716d1e9f5b1482f20636c0d8410a7e77dcc9399fec7c692ba2cbd4780864a`.
On one pinned logical CPU, 60 full `eigh` calls yielded 8,541 `cycles:P`
samples with zero lost samples. Perf-data SHA-256 is
`b48f4bb17aedf47f7c69e09896ad480c14ceda16e924de5112d0b47af3627b42`.
Live SciPy used `_flapack` SHA-256
`fe169babb0d0dedfef08ee9ee2a6c3cdfba00de573718f2b26e61a0bb8a3e097`
and produced 3,584 samples with zero lost; its perf-data SHA-256 is
`533c39e62e017b62530aa0cb90805530e4153fe2cd2ea90bb979259d7208c364`.

| rank | arm / source | whole-job cycles | incumbent pays the same cost? | disposition |
|---:|---|---:|---|---|
| 1 | FrankenSciPy strided upper-triangle contribution, `lib.rs:10971` | 24.51% | no; incumbent `dsymv` consumes one stored triangle | **structural gap / selected** |
| 2 | FrankenSciPy reflector back-transform dot, `lib.rs:10868` | 16.58% | yes; SciPy pays blocked reflector application | shared work / move on |
| 3 | FrankenSciPy contiguous lower-triangle contribution, `lib.rs:10975` | 3.99% | yes, but ours rereads every off-diagonal already visited by rank 1 | selected with rank 1 |
| 4 | FrankenSciPy inverse-iteration column solve | 6.99% | yes; SciPy pays MRRR eigenvector work | shared work / move on |
| 5 | SciPy `dgemm_kernel_HASWELL` | 25.41% | incumbent-only blocked reduction/back-transform work | not our gap |
| 6 | SciPy `dsymv_kernel_4x4` plus `dsymv_L_HASWELL` | 10.36% | yes, but one-triangle optimized | comparator primitive |

The exact structural difference is in
`apply_symmetric_householder_trailing_rank2_lower_storage`: for each
off-diagonal lower-triangle value, the current loop first rereads it through a
strided upper contribution and later through its contiguous lower column.
SciPy's lower-triangle `dsymv` reads that stored value once and scatters its two
symmetric products.

**One lever and predictions fixed before implementation.** Sweep each stored
lower-triangle column once. Accumulate `A[row,col] * v[col]` into `p[row]` and
the symmetric `A[row,col] * v[row]` contribution into one scalar `p[col]`, then
write that scalar once. Keep the existing `tau`, `v^T p`, correction, rank-2
update, tolerance, fallback, and public API unchanged. A hidden same-ELF atomic
switch forces the current double-read implementation. The selected source
lines account for `28.50%` of whole-job cycles; removing one matrix read and
the strided gather predicts at least `1.15x` whole-job speedup, while a full
`4.23x` SciPy flip is not predicted because the incumbent also has blocked
reflector and MRRR advantages.

**Completion cell and acceptance.** Build one frozen `release-perf` ELF and run
at least 21 balanced interleaved rounds of (1) default one-pass candidate,
(2) forced double-read same-ELF control, and (3) genuine live SciPy 1.17.1
`scipy.linalg.eigh` on the same deterministic symmetric `n=512` matrix, all on
one pinned CPU with BLAS capped at one thread. Record both engine hashes,
p50/p95/p99, raw samples, bootstrap-median CI95, and independent A/A controls
for all three arms. Candidate/control eigenvalues must agree within the existing
native `1e-9` scaled tolerance; candidate and live SciPy must each satisfy
`||A V - V Lambda||_max <= 1e-8 * max(1,||A||)` and orthogonality error
`<=1e-8`; focused differential/unit conformance must pass. A maintenance KEEP
requires the null-corrected control/candidate bootstrap-median CI95 lower bound
to clear `1.10x` and twice the widest A/A null margin. A competitive statement
additionally requires the corrected SciPy/candidate CI95 lower bound to clear
`1.0x`. Any numerical drift, candidate loss, or undecidable gate is a revert.

**Pre-timing admission amendment, 2026-08-01 00:21 EDT.** The host-wide
quiescence sample is fixed at `1.0 s` with the original `20%` per-CPU busy
threshold for the next and only admissible run. The initial `300 ms`
implementation repeatedly classified sub-second single-core process launches
as a fully busy CPU on this tickless kernel even while five-second host load
was `96.54%` to `97.25%` idle. This changes no candidate, fixture, effect gate,
null gate, round count, affinity, or incumbent arm. All earlier attempts are
invalid: they either stopped before timing or failed the post-run quiescence
check, and none may be used in the verdict.

**Admissible measurement.** The accepted 24-round balanced run used frozen
ELF SHA-256
`b2149b1eecbcf1167a353cb9ba64e312b9a9cae19ae1c392983bf70b2ced65ac`,
source-diff SHA-256
`f5fc448756c9bb519edb070820b956843ad90c7416416ed05f5a43c270c94492`,
and the genuine SciPy 1.17.1 `_flapack` named above. Both engines actually
used one process thread and CPU 31 affinity. Pre/measurement/post one-second
host gates had maximum per-CPU busy fractions `12.12%`, `11.00%`, and
`13.00%`, all below the registered `20%` ceiling. Candidate p50/p95/p99 were
`135.3501/136.7925/136.9929 ms`; forced-original were
`178.6152/180.3596/180.6077 ms`; live SciPy were
`34.9091/35.1311/35.3577 ms`.

The control/candidate median ratio was **`1.31895x`**, with corrected
bootstrap-median CI95 **`[1.31580x, 1.33007x]`**. Candidate/control A/A,
forced-original A/A, and SciPy A/A medians were respectively `0.99751`,
`1.00119`, and `1.00186`; every null median was within 2% of one and the
widest null endpoint was `1.00885`. The registered 1.10x maintenance threshold
and both 2x null margins therefore clear. SciPy/candidate was `0.25779x` with
CI95 `[0.25638x, 0.25965x]`, so SciPy remains about `3.88x` faster and no
competitive claim is made. Arm CVs were `1.418%/0.848%/0.863%`; CV is
provenance only and did not decide the result.

**Numerical and quality proof.** Candidate and same-ELF control had zero
eigenvalue or eigenvector bit mismatches. Candidate maximum residual was
`1.28342e-13` and orthogonality error `5.54038e-15`; live SciPy produced
`3.57492e-14` and `4.00247e-13`. Candidate/live maximum eigenvalue difference
was `6.99885e-13`, below the registered scaled contract. Strict-remote
`fsci-linalg` tests passed `525` with `42` ignored; strict-remote
`fsci-conformance --test e2e_linalg` passed `21/21` on `ovh-a` (job
`j-29956586046750858`), including the symmetric-eigenvalue scenario. The
broader conformance invocation failed closed before tests because worker
`vmi1153651` exhausted disk while linking unrelated integration binaries; no
local fallback was used. Workspace `cargo check --all-targets` passed remotely.
Strict clippy remains blocked by 49 pre-existing lints outside this diff, and
workspace rustfmt remains blocked by peer-owned
`diff_sparse_iterative_solvers.rs`; both owned files pass rustfmt, and targeted
UBS reports zero critical findings.

Accepted JSON SHA-256 is
`e6aef873df876c2fdd346d380bc51d6c1be3325b4c36e737ecca7d8c4e57a131`;
raw log SHA-256 is
`254484e61a83bf1c3388a8bf1033dff58a186ecc41b9bd122d083b38e9ca3f03`;
harness SHA-256 is
`127157ddc85e9504ed5c966b39b38a953ae17c25d90a390f2da77db462505808`.

**Result class: MAINTENANCE-WIN. Decision: KEEP.** Retry the remaining dense
`eigh` gap only after a fresh whole-job incumbent-cost profile identifies a
different structural cost; do not retry this matvec, the rejected per-step
parallel tridiagonalization, or the rejected WY-blocked reduction families.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: reusable mixed-precision Newton LU for dense BDF

**Status: PRE-REGISTERED / unmeasured.** Bead `frankenscipy-8l8r1.178`.
The accepted dense `eigh` result remains a live-SciPy loss at `0.25779x`, but a
fresh whole-job profile filtered every remaining leading `eigh` cost as shared:
our reflector back-transform, symmetric matvec, inverse iteration, and
tridiagonal eigensolve have direct `dgemm`/`dlarfb`, `dsymv`, `dlar1v`, and
`dlasq` counterparts in the incumbent. Per the standing cost filter, that vein
is closed until a different structural cost appears.

The next decided live loss is the exact `dense-allpairs`, `n=512`, BDF job:
current FrankenSciPy completed in `1042.065 ms` versus genuine live SciPy
1.17.1 in `508.995 ms`, a routing-only SciPy/FrankenSciPy ratio of `0.4878x`.
Results agreed to `8.882e-16` maximum absolute and zero tolerance-scaled
difference. FrankenSciPy reported `nfev=2472`, `njev=1`, `nlu=129`, and
`steps=653`; SciPy reported `nfev=1308`, `njev=1`, `nlu=113`, `steps=653`, and
`1821` actual RHS calls. This three-round routing probe is not acceptance
evidence because it has no same-ELF candidate/control arm and too few rounds.

**Whole-job profile and incumbent-cost filter.** The frozen current
`release-perf` ELF SHA-256 is
`bcb1e47e2864068fbcabfc64411cac511a7f9a46415922fe20797f5071a00379`.
The combined same-invocation profile produced 21,580 `cycles:P` samples with
zero lost samples; perf-data SHA-256 is
`ea3d85fb777f40369b6ec2a5c591d95d48f4b71e9aceeccbc7306522e2f76caa`.
FrankenSciPy and live SciPy contributed `48.74%` and `51.26%` of combined
cycles respectively.

| rank | FrankenSciPy cost | share of FrankenSciPy self | incumbent pays the same cost? | disposition |
|---:|---|---:|---|---|
| 1 | nalgebra f64 `LU::new` | 93.13% | yes; SciPy pays blocked f64 `dgetrf` work (`dgemm`, `dtrsm`, `dlaswp`, and packing) at nearly the same factor count | shared arithmetic, but precision is a structural escape |
| 2 | nalgebra f64 triangular solves | 4.39% | yes; SciPy pays `dgetrs`-class substitution | shared work / preserve reuse |
| 3 | BDF step and Newton control | 1.31% | yes | shared work / move on |

The f64 factorization itself is therefore not the gap to micro-optimize. The
selected incumbent-cannot-follow widening is the already-proven fsci-linalg
mixed-LU primitive: factor each dense BDF Newton matrix once in f32, reuse those
factors for every Newton right-hand side, compute every residual against the
untouched matrix in f64, and accept a correction only when the f64 backward
error reaches the existing f64-LU quality bar. This widens mixed precision into
a solver family whose incumbent remains on f64 `dgetrf`; it is not another
attempt at the shipped one-shot `solve()` gate.

**One lever and predictions fixed before implementation.** Add a reusable
mixed-LU factor object in `fsci-linalg`, storing the untouched f64 row-major
matrix, one f32 blocked partial-pivot factorization, its infinity norm, and a
lazy reusable f64 blocked-LU fallback. Route only dense BDF Newton matrices at
`n >= 128` through it. Each right-hand side is permuted and solved in f32, then
refined for at most eight iterations using f64 residuals. The acceptance bar is
`8*n*eps64*(||A||inf*||x||inf+||b||inf)`, identical to the shipped mixed-LU
bar. A non-finite/zero f32 pivot, stalled refinement, or exhausted iteration
budget fails closed: factor once in f64 and reuse that exact-precision factor
for this and all later right-hand sides. Exact diagonal and certified banded
BDF factors are unchanged. A hidden same-ELF atomic switch forces the current
nalgebra f64 dense route, and hit/fallback counters prove dispatch. Given the
93.13% factor share and prior `n>=128` mixed-LU evidence, predict at least a
`1.20x` whole-job maintenance gain; a competitive flip is possible but not
required.

**Completion cell and acceptance.** Build one frozen `release-perf` ELF and run
at least 21 balanced interleaved rounds of (1) default reusable mixed candidate,
(2) forced current nalgebra-f64 same-ELF control, and (3) genuine live SciPy
1.17.1 BDF on the identical deterministic `dense-allpairs`, `n=512`,
`t=[0,1]`, `rtol=1e-8`, `atol=1e-10` fixture. Pin all arms to one logical CPU
and cap incumbent BLAS at one thread. Record executable/source hashes,
p50/p95/p99, raw samples, bootstrap-median CI95, independent A/A controls for
all three arms, worker placement, affinity, actual thread count, ISA, RAM,
NUMA, frequency policy, and pre/post quiescence.

Candidate/control and candidate/live final vectors must each remain within
`1.0` componentwise tolerance unit; all three solves must succeed with status
zero. Candidate `nfev`, `nlu`, and accepted-step count must not exceed the
forced-f64 control, candidate mixed-factor hits must be non-zero, forced-control
hits must be zero, and every emitted mixed correction must have passed the f64
backward-error certificate (otherwise the factor must report and use its f64
fallback). Focused reusable-factor tests, `fsci-integrate` tests, differential
BDF conformance, workspace check/clippy/fmt as far as peer-owned blockers allow,
and UBS on every owned changed file are required. A maintenance KEEP requires
all correctness/execution gates, every A/A median within 2% of one, and the
null-corrected forced-original/candidate bootstrap-median CI95 lower bound to
clear both `1.20x` and twice the widest A/A null margin. A competitive statement
additionally requires the corrected live-SciPy/candidate CI95 lower bound above
`1.0`. Any weakened tolerance, missing residual certificate/fallback, candidate
counter regression, or failed/undecidable timing gate means revert.

**Measured result: REVERT.** Candidate commit `70a2271e5`; frozen
`release-perf` ELF SHA-256
`7d635dad00a310e33739b6d21d599fb4ea2b2fcfc4d954cf6f96cfea3428c2a8`;
candidate patch SHA-256
`e6bf9b386253b90f1a9a50f156343bb3098f0adc61e3b0d5c1f0723255696393`.
The same-ELF control switch restored the nalgebra f64 factorization, and genuine
live SciPy 1.17.1 ran in the same invocation. The 21-round raw log is
`/data/tmp/frankenscipy-8l8r1-178-routing-20260801T0521Z.log`, SHA-256
`1010820ab3788dc6dc8f6bb2b28b76c9e41836ff96670e45afc025e9339b3181`.

On `thinkstation1`, all arms were pinned to CPU 31, both implementations
reported one observed worker thread, and SciPy BLAS was capped at one thread.
Candidate p50/p95/p99 was `1026.028/1043.974/1273.960 ms`, forced-original was
`1066.517/1108.104/1154.119 ms`, and live SciPy was
`511.248/529.315/687.331 ms`. Candidate/original/SciPy A/A medians were
`0.999053/1.003368/1.003788`, all within the registered 2% band; CI95 were
`[0.992402,1.003297]`, `[0.997228,1.006976]`, and
`[0.998507,1.009211]`. The null-corrected forced-original/candidate median was
only `1.0382x`, CI95 `[1.0214,1.0518]`, far below the preregistered `1.20x`
maintenance bar. Null-corrected live-SciPy/candidate was `0.4972x`, CI95
`[0.4902,0.5044]`: SciPy remained about `2.01x` faster.

The numerical and execution gates passed. Candidate/control maximum final-state
difference was `0.000000` tolerance units and candidate/live was `0.000003`.
Candidate completed with `nfev=2472`, `njev=1`, `nlu=126`, and `steps=653`;
forced-original used the identical `nfev`, `njev`, and step count with
`nlu=129`. All `126` candidate dense factors took the registered mixed route,
none activated the f64 fallback, and the forced control recorded zero mixed
hits/fallbacks. This proves the rejection is an effect-size failure, not missed
dispatch or failed refinement.

Pre-run five-second host provenance was usable: mean CPU idle `91.23%`, CPU 31
idle `98.59%`, and sibling CPU 63 idle `99.00%`; artifact SHA-256
`40634604c1ffa43d8ab8875c1b58770a1aaa6de8cceaa0c107717d8a25cf98fd`.
The post-run host-wide sample was contaminated by unrelated load (mean CPU idle
`2.90%`), artifact SHA-256
`530966a9b51d45543c662e0d0c09e3381be5786d35266f12b6a3f2f021dd1fc7`.
That independently forbids a keep; it does not rescue a result whose corrected
effect CI high is only `1.0518x` against a fixed `1.20x` bar.

**Result class: REJECTED / code reverted. Decision: REVERT.** The production
source is restored exactly to candidate parent `b8dfb9a73`. Do not retry this
per-right-hand-side reusable mixed-refinement design at `n=512`. Reconsider the
family only if a different mechanism batches multiple Newton residual
certificates into one matrix traversal, or a fresh candidate-only profile
identifies at least 20% removable self-time without adding another f64 matrix
pass per Newton right-hand side.

### 2026-08-01 (cod/SilverRiver) — EVIDENCE-ONLY CLOSE: reuse one ILU across a GMRES RHS batch

**Status: REVERT / NO-SHIP; NO ADMISSIBLE TIMED SAMPLE.** Bead
`frankenscipy-8l8r1.179`. This is the next worst unfiltered solver job, not a
repeat of the rejected SPILU diagonal-position cache or the CA-GMRES
experiment. The accepted whole-job screen at commits `7e6c9555c` and
`055993b91` measured the serial 32x32 convection-diffusion source screen with
twelve right-hand sides at `24.170874 ms` in FrankenSciPy versus
`10.293884 ms` in genuine live SciPy 1.17.1. SciPy/FrankenSciPy was only
`0.4259x`, CI95 `[0.4211,0.4409]`: a decided 2.348x incumbent advantage.

**Whole-job cost rank and incumbent filter.** The already-admitted job profile
and exact work counts separate the gap without attributing shared arithmetic
to FrankenSciPy:

| rank | whole-job cost | incumbent pays the same cost? | disposition |
|---:|---|---|---|
| 1 | about 1,185 unpreconditioned Arnoldi iterations over twelve RHS, including SpMV plus roughly 9.3 dot/axpy pairs per iteration | no; the selected incumbent runs only 2-3 iterations per RHS | **structural gap / selected** |
| 2 | each required Arnoldi SpMV, orthogonalization, and Givens update | yes; the unpreconditioned arms have exact per-scenario iteration parity and SciPy's marginal kernel cost is lower | shared work / move on |
| 3 | operator/RHS construction, field materialization, and 36 scientific summaries | yes; all are inside both whole-job boundaries | shared work / move on |
| 4 | one ILU construction per whole job | incumbent pays this extra cost and still wins after amortizing it over twelve solves | necessary selected setup |

The selected live `csc_matrix` plus one default `spilu` reused across the
twelve calls cut the scenario counts from
`100,93,94,125,99,106,133,123,100,93,94,125` to
`3,3,3,3,3,3,3,2,2,2,3,2`. FrankenSciPy already has public ILU(0), public
GMRES, and a kept shared-nothing GMRES batch scheduler, but no public route that
applies one factorization to every Arnoldi operator and reuses it across the
batch. That missing composition is the structural gap. SciPy's stronger
fill-capable SuperLU ILU remains an incumbent advantage; this lever does not
claim ILU(0) is algorithmically identical to default `spilu`.

**One lever fixed before implementation.** Add left-preconditioned public
`gmres_preconditioned` and `gmres_batch_preconditioned` entry points taking a
borrowed `SparseIluFactorization`. Compute Arnoldi vectors with
`M^-1 (A v)`, reuse the same immutable factor concurrently across independent
RHS workers, and always decide convergence from the exact unpreconditioned
`||b-Ax||/||b||` residual. Singular or incompatible preconditioners fail
closed; there is no silent unpreconditioned fallback. Existing public `gmres`
and `gmres_batch` behavior remain the same and form the same-ELF control.

The candidate whole job constructs CSR plus CSC, builds one default
FrankenSciPy ILU inside every timed repetition, reuses it for all twelve RHS,
materializes the same 12,288 values, and computes the same 36 summaries. The
control reconstructs the same job but calls existing unpreconditioned
`gmres_batch`. The genuine live arm retains the fixed six-configuration screen
and selects its fastest full-output-eligible public configuration. Predict
candidate iteration counts below half the control for every RHS and at least a
`1.20x` same-ELF whole-job gain. Competitive parity is possible but not
assumed because FrankenSciPy ILU(0) retains no fill while SciPy default
`spilu` permits fill.

**Completion and decision gates.** Use one frozen `release-perf` ELF and at
least 21 balanced interleaved rounds of candidate, same-ELF unpreconditioned
control, and genuine live SciPy 1.17.1 in one invocation, with independent A/A
controls for all three arms. Record executable/source/oracle hashes,
p50/p95/p99, raw samples, bootstrap-median CI95, affinity, actual thread count,
worker placement, ISA, RAM, NUMA, frequency policy, and pre/measurement/post
host-wide quiescence. Require every solve to converge, exact candidate/control
input hashes, true relative residual at most `1.25e-5`, all 12,288 fields and
36 summaries inside the prior component/summary tolerance contract, and zero
tolerance mismatches versus both control and selected live SciPy.

A maintenance KEEP requires candidate iterations below half of control for
all twelve RHS, every A/A median within 2% of one, and the null-corrected
control/candidate bootstrap-median CI95 lower bound at least `1.20x` and beyond
twice the widest null margin. A competitive statement additionally requires
the corrected live-SciPy/candidate CI95 lower bound above `1.0`. Numerical
drift, failed exact-residual convergence, missing dispatch, candidate loss,
inadmissible host evidence, or an undecidable maintenance gate means revert.
After a rejection, do not retry this ILU(0) composition on the same cell;
profile the surviving candidate-only cost and switch to fill-capable ILUT,
fused preconditioner/SpMV traversal, or a different worst live-loss family.

**Implementation and proof reached before measurement.** Preregistration commit
`fcafc8981` preceded candidate commit
`be0a3275e913c9b04e9d469a207828c7c2217756`. The candidate added public
left-preconditioned scalar and batch GMRES, shared one immutable ILU across the
batch, gated convergence on the exact unpreconditioned residual, failed closed
on shape/singularity/non-finite errors, and left the incumbent same-ELF GMRES
route unchanged. The complete `fsci-sparse` library suite passed 410 tests
(4 ignored), all three focused preconditioned tests passed, the feature-gated
benchmark binary checked, targeted remote clippy passed on `vmi1293453`, and
the strict remote workspace all-target check passed as RCH job
`j-29956586046750968`. Direct rustfmt and targeted UBS checks were clean. The
workspace-wide format gate remained independently blocked only by the
peer-edited conformance fixture
`crates/fsci-conformance/tests/diff_sparse_iterative_solvers.rs`.

**Frozen executable and attempted invocation.** Strict remote release build
RCH job `j-29956586046750963` ran on `vmi1153651` from an exact clean overlay of
the candidate commit. The retrieved 10,569,600-byte ELF
`/data/tmp/cargo-target/frozen/perf_gmres_job_vs_scipy-be0a3275e-244ac6fe`
matched the remote SHA-256
`244ac6feeacdbe0dd58bd716a8c7000a3bbea00b0c0793387cecd26e68f89256`.
Booking CLAIM message `8647` reserved CPU 31 on `thinkstation1`; RELEASE
message `8653` relinquished it immediately after the preflight veto.

The single frozen-ELF invocation exited before fixture construction, SciPy
startup, screening, warmup, or timing because the preregistered host-wide
quiescence pre-gate found four CPUs above 20% busy: CPU 8 at 27.6%, CPU 16 at
33.3%, CPU 32 at 26.7%, and CPU 54 at 43.3%. The 1,009-byte abort artifact is
`/data/tmp/frankenscipy-8l8r1-179-be0a3275e-ilu-gmres-live-20260801T0635Z.log`,
SHA-256
`7ea56bcc402037f27b412df26cb289cb0a27fb67dacd9d2d522c9aefde146f90`.
It contains no timing sample and therefore supports no performance claim.

**Decision: REVERT / NO-SHIP.** Inadmissible host evidence was an explicit
revert condition, so all three candidate source/harness files were restored
byte-for-byte to preregistration commit `fcafc8981`; the frozen ELF and abort
artifact are retained for provenance. Do not rerun this ILU(0) composition on
the same cell. Reopen only after a fresh whole-job profile selects a different
matrix/batch regime or a materially different fill-capable or fused traversal,
and only on a booked host that passes every pre/measurement/post admission
gate.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: exact-f32 value stream for persistent CG

**Status: PRE-REGISTERED / unmeasured.** Bead
`frankenscipy-8l8r1.180`. This widens the kept persistent row-band CG kernel
with the mixed-precision lever; it does not repeat the rejected per-iteration
thread scopes, worker-budget widening, or the undecided u32-index result on the
five-point stencil.

**Whole-job rank and incumbent-cost filter.** The admitted persistent-CG
campaign already established that increasing worker count on the large
five-point whole solve degrades p50 monotonically from `240.724 ms` at nine
workers to `885.240 ms` at 64: the row kernel is at the memory-bandwidth roof,
not compute- or spawn-limited. The exact-width pool and row-local vector state
are therefore retained. Ranked remaining traffic is (1) the f64 CSR value
stream, (2) the already-narrowed u32 index stream, (3) gathered search-vector
values, and (4) row-local state. Live SciPy pays an f64 sparse matvec for the
same f64 input, so the arithmetic is shared; the structural escape is that
coefficients exactly representable in f32 can be compressed internally while
the public f64 matrix, f64 multiply/accumulate order, and exact residual remain
unchanged. The prior five-point u32 result was undecidable because four
nonzeros per row made gather latency dominant, so the completion cell below is
a distinct wide-row matrix whose value stream exceeds cache.

**One lever fixed before implementation or candidate timing.** In the existing
persistent-CG setup, make one pass over matrix values. If and only if every
finite f64 coefficient round-trips bit-exactly through f32, retain a compact
f32 value buffer for the solve; otherwise retain the unchanged borrowed f64
slice. Workers convert each loaded f32 coefficient back to f64 and execute the
same row-order f64 multiply/add. A hidden same-ELF atomic switch forces the f64
stream. The matrix API, index order, reductions, worker budget, convergence
test, iteration count, and returned f64 solution are unchanged. This is an
exact-storage-width specialization, not tolerance relaxation; ineligible
matrices must have zero compact-value dispatches.

**Frozen completion job.** Use a deterministic symmetric strictly diagonally
dominant banded CSR with `n=65,536`, half-bandwidth `128`, diagonal `0.625`,
off-diagonals `-1/512`, `nnz=16,826,240`, RHS
`1+0.01*(i mod 17)`, zero initial guess, `rtol=1e-5`, and `atol=0`. Every
coefficient is exactly representable in f32. The f64 values plus u32 indices
stream is about 202 MiB per matvec; compact values reduce it to about 138 MiB
without changing a single arithmetic input. Matrix/RHS construction is outside
timing, but compact-buffer eligibility and construction remain inside every
timed public solve. Run one frozen `release-perf` ELF on an affinity containing
physical cores only, with candidate and forced-f64 control using the same
worker budget and genuine live SciPy 1.17.1 capped at one thread.

Use at least 21 balanced interleaved rounds of candidate, same-ELF forced-f64
control, and live SciPy in one invocation, plus independent A/A controls for
all three arms. Record raw samples, p50/p95/p99, bootstrap-median CI95, both
engine SHA-256s, source commit, build worker/route, booking claim/release,
affinity, requested and observed worker counts, ISA, RAM, NUMA, frequency
policy, and pre/measurement/post host-wide quiescence. Construction must be
identical across arms and the SciPy child must report a genuine float64 CSR.

**Correctness, mechanism, and decision gates.** Candidate and control must be
bit-identical in solution, convergence flag, iteration count, and reported
residual; both true relative residuals must be at most `1.25e-5`. Candidate and
live SciPy must have zero component tolerance mismatches and relative L2 at
most `1e-8`. Candidate compact-value hits must be nonzero, forced-f64 hits zero,
and a new non-f32-exact fixture must prove fail-closed f64 dispatch. Every A/A
median must be within 2% of one. KEEP requires the null-corrected
control/candidate bootstrap-median CI95 lower bound at least `1.10x` and beyond
twice the widest null margin; a competitive claim additionally requires the
corrected SciPy/candidate CI95 lower bound above `1.0`. Drift, missed dispatch,
inadmissible host evidence, loss, or an undecidable maintenance gate means
revert. After rejection, do not retry value narrowing on this cell; switch to
fused preconditioner/SpMV traversal or a fresh live-loss family.

**Implementation and proof reached before measurement.** Preregistration
commit `21898fcd5` preceded candidate commit
`24662ab7448c9fbf25bdea7536bbfe91ef8b9c1a`. The candidate allocated an f32
coefficient stream only after a bit-exact f64-to-f32-to-f64 pass, converted
each coefficient back to f64 before the unchanged ordered multiply/add, and
retained the original f64 slice for any ineligible matrix. The new exact and
non-exact dispatch test passed as strict-remote RCH job
`j-29956586046751002`; the complete `fsci-sparse` library suite passed 408
tests with four ignored as job `j-29956586046751008`; the feature-gated
benchmark checked as job `j-29956586046751000`. Targeted clippy passed on
`vmi1293453` as job `j-29956586046751022` after allowing only the two
established untouched sparse lints. Rustfmt, Python syntax, diff checks, and
targeted UBS (zero critical findings) were clean.

**Frozen executable and exact invocation.** Strict-remote release build job
`j-29956586046751024` used `vmi1293453` and the exact candidate commit with
`--clean-overlay --no-overlay`. The remote and retrieved 8,343,752-byte ELF
matched SHA-256
`73455ab3d0b9b4d52fa82ce53332c4bee88bb49e5933fe495c5977d656e770d6`
at
`/data/tmp/cargo-target/frozen/perf_csr_matvec-24662ab74-73455ab3d0b9b4d5`.
TRJ claim `8659` was released before timing because an unrelated 11-hour
single-core job made the host-wide gate impossible. Fresh claim `8660`
reserved physical CPUs `0-31` on `thinkstation1`; RELEASE message `8661`
ended that window.

The registered `n=65,536`, half-bandwidth `128`, `nnz=16,826,240` invocation
passed its pre-gate with maximum host CPU busy fraction `0.110`, constructed
the exact float64 SciPy CSR, and observed 32 candidate and 32 control worker
tasks versus one live-SciPy thread. Candidate and control solutions were
bit-identical, all three arms converged in exactly 11 iterations, candidate
true relative residual was `9.099e-6`, candidate/live relative L2 difference
was `4.408e-16`, maximum absolute difference was `9.770e-15`, and there were
zero tolerance mismatches. Candidate compact dispatch was one and forced-f64
dispatch zero. The subsequent measurement admission gate found CPU 8 at
`31.3%` busy, above the registered `20%` ceiling, and exited before the first
timed sample. The complete 18-line, 2,718-byte no-timing artifact is
`/data/tmp/frankenscipy-8l8r1-180-completion-24662ab74.log`, SHA-256
`d15377920c64ad711a9d1315b693b753219550d5c791982f5af496c23a846b9e`.

**Decision: REVERT / NO-SHIP.** No candidate, control, incumbent, or A/A
timing sample was admitted, so this result supports no speedup or competitive
claim. Inadmissible host evidence was an explicit revert condition; all three
candidate source/harness/oracle files were restored byte-for-byte to
preregistration commit `21898fcd5`. Do not rerun value narrowing on this cell.
Reopen only for a materially different mechanism selected by a fresh
whole-job profile; the immediate registered alternatives are fused
preconditioner/SpMV traversal or a different worst live-loss family.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: fuse CSR matvec into the ILU-L sweep

**Status: EVIDENCE-ONLY REVERT / no headline timing.** Bead
`frankenscipy-8l8r1.181`. This is the fourth requested sparse/Krylov lever and
the explicit retry route left by `.179` and `.180`; it is not another timing
attempt of the rejected unfused ILU composition or the prior SPILU diagonal
cache.

**Worst-job rank and incumbent-cost filter.** The strongest admitted sparse
loss remains the twelve-source, side-32 convection--diffusion GMRES job:
genuine SciPy 1.17.1 with one reused default `spilu` completed at
`10.293884 ms`, while FrankenSciPy's unpreconditioned batch took
`24.170874 ms`. SciPy/FrankenSciPy was `0.4259x`, CI95
`[0.4211,0.4409]`. The incumbent removed about 1,150 Arnoldi iterations across
the batch, so repeating shared orthogonalization or plain SpMV work is not the
gap. After both engines have a reusable preconditioner, the recurring
candidate-only structural cost is a full-length temporary between CSR `A*v`
and `M^-1`: SciPy's public `LinearOperator` boundary necessarily returns the
matvec result before calling SuperLU, while FrankenSciPy owns the CSR and ILU
row representations together and can stream one directly into the other.

**Exactly one lever fixed before implementation or candidate timing.** Add a
left-preconditioned GMRES route backed by one borrowed immutable
`SparseIluFactorization`. Its candidate Arnoldi operator processes rows in
ascending order: compute the CSR row dot product in existing index order, then
immediately apply that row's unit-lower ILU terms using already completed
rows, and finally perform the unchanged descending U backsolve in place. The
same-ELF forced control first writes the identical CSR product to a separate
`n`-element f64 buffer, then copies it into the same reusable output and runs
the identical L and U arithmetic. Both arms allocate both buffers, reuse them
for every Arnoldi step, use the same outer RHS worker pool, and scan the same U
diagonal representation; only the intermediate vector traversal is removed.
No fill, drop, ordering, diagonal-cache, reduction, restart, tolerance, or
worker-budget change belongs to this lever.

The public scalar and batch entry points fail closed for an incompatible,
singular, or non-finite preconditioner result. Preconditioned residuals may
steer Arnoldi, but every convergence return is certified with the exact
unpreconditioned `||b-Ax||/||b||`. Existing `gmres` and `gmres_batch` remain
unchanged. A hidden atomic switch forces the separate-buffer route and a
dispatch counter proves that candidate calls reached the fused traversal.

**Frozen persistent-operator whole job.** Use a new, larger named regime rather
than the rejected side-32 cell: one side-192 steady two-dimensional
convection--diffusion--reaction CSR (`n=36,864`, `nnz=183,552`, diagonal
`4.001`, west/east/vertical coefficients `-1.2/-0.8/-1.0`) and twelve
localized source fields centered at row positions `36,96,150` crossed with
column positions `30,72,120,162`. Construct CSR, CSC, sources, and exactly one
default ILU before the timed service boundary and reuse them across all
rounds. One timed job performs all twelve public preconditioned GMRES solves
at restart 20, `rtol=1e-5`, `atol=0`, materializes all `442,368` solution
components, and computes 36 inventory/outlet/exposure summaries. Setup time is
reported separately and a cold setup-plus-solve screen is secondary evidence;
neither may substitute for the registered persistent-job gate. This reuse
regime is recognizable for a steady operator serving multiple sources and is
materially distinct from `.179`, whose every timed repetition rebuilt the
side-32 operator and factor.

Screen the same six genuine live SciPy public configurations as the admitted
job and select the fastest arm that passes complete output conformance; its
expected strongest route is `csc_matrix` plus one default `spilu` reused for
all twelve RHS. Operator, RHS, preconditioner, and Python interpreter startup
are outside the live persistent-job timing exactly as in the candidate and
control. Cap live SciPy, NumPy, BLAS, and SuperLU observation to one thread.
Candidate and forced control must request and observe the same RHS-worker
count on physical-core affinity only; live SciPy must report one observed
worker.

**Correctness, provenance, and statistical contract.** Before timing, require
candidate/control bit identity for every solution component, scientific
summary, iteration count, convergence flag, and reported residual. All twelve
true relative residuals must be at most `1.25e-5`. Candidate/live comparison
must have relative L2 at most `1e-8`, zero component and summary tolerance
mismatches, and identical canonical input SHA-256. Candidate fused hits must
be nonzero and forced-control hits zero; a focused nonsymmetric fixture must
also prove fused `M^-1(A*v)` bit-identical to the separate route.

Run one frozen `release-perf` ELF for at least 21 balanced interleaved rounds
of fused candidate, same-ELF separate-buffer control, and the selected live
SciPy arm, plus an independent A/A null for each of the three arms in the same
invocation. Record raw samples, p50/p95/p99, bootstrap-median CI95, source and
executable hashes, live engine/SuperLU/oracle hashes, strict-remote build job
and worker, booking claim/release, affinity, requested and observed workers,
ISA, RAM, NUMA, governor, and pre/measurement/post host-wide load. A bounded
preflight may take at most twelve load-only one-second samples before fixture
construction and accepts the first sample with every CPU at or below 20%
busy; it is an admission wait, not a timing retry. Measurement and post gates
are single-shot and fail closed.

**Decision gate.** KEEP requires every A/A median within 2% of one and the
null-corrected separate-control/fused-candidate bootstrap-median CI95 lower
bound at least `1.10x`, beyond twice the widest null margin. A competitive
claim additionally requires the corrected live-SciPy/candidate CI95 lower
bound above `1.0`. Correctness drift, missed dispatch, worker mismatch,
inadmissible host evidence, a candidate loss, or an undecidable maintenance
gate means restore all production and harness changes. After rejection, do
not retry fusion on this five-point ILU(0) cell; profile the full persistent
job against the live selected incumbent, discard shared self-time, and select
the highest remaining candidate-only structural cost.

**Executed completion cell.** Candidate commit `ee3fbc2f4`; the exact
`release-perf` ELF was built strict-remote from that full commit on
`vmi1153651` by RCH job `j-29956918973825054`, with SHA-256
`07a3f5c0d9c49a2803810e9a0373f946074edf88bf73feb5f34b5b6168bbdf38`.
The oracle SHA-256 was
`e688ac8bdcc78992e7add83faba343362e0d5bb88949479dd3627e931926dbe1`.
The sole invocation ran on thinkstation1 under booking claim `8689`, pinned to
physical-core affinity `0-11`; both candidate and control observed exactly 12
workers and live SciPy observed one. The 32-physical-core/64-thread host,
231,691,894,784-byte RAM, one-NUMA-node, AVX2/FMA provenance was captured.
Preflight admitted on its first sample with maximum CPU busy fraction `0.149`;
the single-shot measurement gate also cleared at `0.122`.

**Proof outcome before headline samples.** Fused candidate and same-ELF
separate control were bit-identical for all 442,368 solution components, all
36 summaries, every iteration count/convergence flag, and every residual;
dispatch counters were `1925` fused hits and zero forced-control hits. The
canonical input SHA-256 was
`35c4d212ca4a198ac4b0de4adfb0dbf4da5f917afe597ab0a6f6744922386f8b`.
Genuine SciPy 1.17.1 and SuperLU were independently identified by engine
SHA-256 values
`f9d7ace03295000d7b1a76dd12229208908a59140b741669e961b69733110e8f`
and
`271ed8a07a651e0234ca0826a32cf4f20f3b972b2af457af46bf8f8f12985e99`.
All six public live configurations converged on all twelve systems with zero
component/summary tolerance mismatches, but none cleared the separately hard
registered relative-L2 ceiling: each unpreconditioned/Jacobi arm measured
`3.752e-6`, and the strongest `csc-matrix-spilu` arm measured `3.848e-6`,
against required `<=1e-8`. The harness therefore aborted with `no SciPy GMRES
configuration passed full-result eligibility` before backend timing, A/A
controls, p50/p95/p99, or bootstrap ratios. Cold fused/separate setup screens
of `0.380098340/0.378472309 s` are diagnostic only and cannot decide the lever.
The raw log SHA-256 is
`eff6342c15d1d93eb02dc465333edf2ec485ddb407f6c2b4d8d016c0f6d8c62a`.

**One-line decision: REVERT — the live conformance gate failed before timing,
so there is no admissible maintenance or competitive result.** The public
preconditioned APIs, fused kernel, tests, harness, and oracle were manually
restored byte-for-byte to the pre-candidate commit. Remotely staged immutable
artifacts remain in place because this session had no authorization to delete
files; they are not benchmark evidence.

**Concrete retry predicate.** Do not rerun this side-192 five-point ILU(0)
fusion cell. Reopen fusion only after a distinct, preregistered fixture first
demonstrates live candidate-relative L2 `<=1e-8` outside timing and a fresh
whole-job profile assigns at least 10% exclusive self-time to the intermediate
CSR-product materialization; otherwise select a different structural loss.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: widen the square-grid DST-I `spsolve` route to rectangular tensor grids

**Status: PRE-REGISTERED / untouched baseline not yet measured.** Bead
`frankenscipy-8l8r1.182`. The supplied Krylov, sparse-parallel, exact-f32, and
matvec/preconditioner-fusion list is closed under its registered cells, so this
is the required self-generated successor. It widens a kept structural win
rather than rerunning any rejected GMRES, CG, QMR, or scoped-thread cell.

**Why this lever, before implementation.** The exact square-grid five-point
route already changed `spsolve` from `4.7959/44.040 ms` to
`0.26710/1.1653 ms` at `n=1,600/4,900`, and genuine SciPy 1.17.1 took
`3.116855/11.947596 ms`, making the kept route `11.67x/10.25x` faster than the
live incumbent. The recognizer nevertheless requires `n=side^2`. A row-major
rectangular tensor grid therefore misses the same separable operator and pays
the generic lower-band materialization plus safe-Rust banded Cholesky over its
many structural zeros. SciPy pays a sparse SuperLU factorization, but it does
not pay this dense-band zero traversal. That candidate-only cost, if confirmed
by the frozen whole-job profile below, is the structural gap; the remaining
CSR scan and residual verification are shared costs and are excluded.

**Mandatory untouched whole-job profile.** Before changing production
selection or arithmetic, add only a measurement mode to the existing
`perf_spsolve` harness and build the current `main` through strict remote RCH.
On one pinned physical core, profile repeated public `spsolve` calls for the
primary `32x128` constant Dirichlet grid and profile genuine live
`scipy.sparse.linalg.spsolve` on the identical CSR and right-hand side in a
separate process under the same affinity and one-thread caps. Rank self-time,
then classify every entry against the live incumbent. Proceed only if
`fsci_linalg::cholesky_banded` plus lower-band materialization is at least 30%
of FrankenSciPy samples and has no equivalent dense-band cost in the SciPy
profile. If the top cost is instead shared CSR construction, residual work, or
an incumbent-equivalent factor kernel, close without a candidate.

**Exactly one production lever.** Generalize the existing fail-closed
square-grid recognizer and DST-I solve from one `side` to row and column
extents. Geometry is derived only from the CSR half-bandwidth and `n`; accept
only default `Auto`/`Colamd`, both extents at least 16, the exact expected nnz,
one finite constant positive diagonal, finite constant negative horizontal
and vertical weights, all required neighbors exactly once, no extras, and
strict diagonal dominance. Use separate row/column sine tables, cosine spectra,
eigenvalues, and inverse scales while preserving ascending reductions within
each transform. Keep the existing true sparse relative-residual acceptance at
`<=1e-8`; any failed recognition, singular eigenvalue, non-finite value, or
residual failure falls through unchanged. Square-grid selection and arithmetic
must remain bit-for-bit unchanged. A hidden atomic switch disables only the
new rectangular admission for same-ELF control, and a rectangular dispatch
counter proves which route executed. No threading, FFT, tolerance, ordering,
general banded solve, or square-grid arithmetic change belongs to this lever.

**Frozen whole job and conformance.** The primary job consists of the three
row-major grids `32x128` (`n=4,096`), `48x96` (`n=4,608`), and `64x96`
(`n=6,144`), each with diagonal `4.001`, four-neighbor weights `-1.0`, and
deterministic `b[i]=1.0+0.5*(i mod 13)`. Matrix/RHS construction and the
persistent live Python process are outside timing; one timed job calls public
`spsolve` once for each already-built grid, materializes all 14,848 output
components, and folds an output checksum. Before timing, require candidate and
same-ELF forced control to pass true relative residual `<=1e-8` and relative
L2 `<=1e-10`; require the same gates between candidate and genuine live SciPy
1.17.1, plus identical matrix/RHS SHA-256. Candidate rectangular hits must be
three and forced-control hits zero per untimed proof pass. Focused tests must
also reject a rectangular grid with one perturbed stencil coefficient, reject
an extra/missing neighbor, and prove the pre-existing `64x64` square solution
is bit-identical with rectangular admission enabled or disabled.

**Measurement and decision gate.** Execute one frozen `release-perf` ELF for
at least 21 balanced interleaved rounds of candidate, same-ELF forced control,
and genuine live SciPy, plus independent A/A nulls for all three arms in that
same invocation. Pin every arm to one physical core and cap Python, NumPy,
BLAS, and SuperLU to one observed thread. Record raw samples, p50/p95/p99,
bootstrap-median CI95, source/ELF/oracle/engine hashes, strict-remote build job
and worker, booking claim/release, affinity, actual workers, ISA, RAM, NUMA,
governor, and pre/measurement/post host-wide load. A bounded preflight may
take at most twelve one-second load samples and admit the first with every CPU
at or below 20%; measurement and post gates are single-shot and fail closed.
Every A/A median must be within 2% of one. KEEP requires the null-corrected
control/candidate bootstrap-median CI95 lower bound at least `1.20x` and beyond
twice the widest null margin. A competitive claim additionally requires the
corrected live-SciPy/candidate CI95 lower bound above `1.0`. Otherwise manually
restore the production and completion-harness changes, ledger one-line REVERT,
and select a different freshly profiled structural loss; do not retry this
rectangular constant-stencil cell.

**Untouched whole-job profile gate: PASS before production implementation.**
The profile-only harness overlay was built strict-remote on `vmi1153651` by
RCH job `j-29956918973825070`; the copied symbol-bearing ELF SHA-256 was
`64023df7db914a567a288eb92f6cf080b102396d589d7cfd8b405c1dd0b44bc8`.
Under thinkstation1 claim `8703`, both profiles were pinned to CPU 31 and the
live Python/NumPy/BLAS arm observed one task. Fifty untouched Rust solves took
`3.732366774 s` (`74.647335 ms/solve`); 500 genuine SciPy 1.17.1 solves took
`3.098840082 s` (`6.197680 ms/solve`). This `0.0830x` SciPy/FrankenSciPy
routing ratio is not completion evidence because it has no interleaved A/A
gate, but it confirms the selected surface is a severe current loss.

The Rust `perf record -F 999 -g --call-graph=dwarf` artifact contained 3,800
samples with zero lost and SHA-256
`2419dbb5e2c19612dbc5e8568f4682ef827d14112aa8001e0968080a72f3434f`.
Ranked self-time was `fsci_linalg::cholesky_banded` **95.13%**,
`fsci_linalg::cho_solve_banded` **4.20%**, `memmove` **0.19%**, and the
M-matrix guard **0.11%**. The genuine SciPy artifact contained 3,265 samples
with zero lost and SHA-256
`745c2718dfa79e706d43b1d13075a8869fc46b20c71fd5f401483573d3900021`;
its engine was
`scipy/sparse/linalg/_dsolve/linsolve.py` SHA-256
`a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`.
Its top self-time was sparse ordering/factor work: `colamd` 11.85%,
`dpanel_bmod` 7.56%, `dpanel_dfs` 4.90%, `dcolumn_dfs` 4.69%, sparse/dense
BLAS kernels below 4.3%, and `dcopy_to_ucol` 3.91%.

The incumbent therefore pays a generic sparse factorization, but not the same
dense-band loop over absent entries that consumes 95.13% here. Shared solve,
copy, CSR, and validation work is individually below 4.20% and is not the
lever. The registered 30% exclusive-cost threshold clears by more than 3x, so
implementation of the rectangular tensor-grid widening is admitted.

**Completion: REVERT / zero admitted timing.** The candidate was compiled from
commit `8d6e52919a3bc970c1507aeb9d1c31f03fd36ddf` by strict-remote RCH job
`j-29956918973825108` on `vmi1153651`; the copied release ELF SHA-256 was
`14ec3a432748e01fa11206a9c88db6c9b3bb1ebe2c17cf9de52a227c500b25a3`
with BuildID `fc2d63df8e32736bdad2c57267a486eb3ca67dfe`. Its embedded
`linalg.rs` and harness hashes were respectively
`af8de5aea9165df6f7ba47d90639ab3a8ef7dcf1ed386eeac4f2db2206b1c107`
and `1d724303fa24d82651d73f7ebabc57b2c75cb64bf56f2fe71d6cfb1d04af9ee0`.
The invocation used thinkstation1 CPU 31 under Agent Mail claim/release
`8712/8713`, one observed Rust thread, and one requested thread for every
Python/NumPy/BLAS pool.

The first candidate ELF aborted before any timing because its clean-overlay
`CARGO_MANIFEST_DIR` did not exist on the measurement host; that no-timing log
has SHA-256
`a8429b51b9fa2b4074e3e21320f51ecaf762a93d15d455f9251de09c536f8021`.
The provenance-only embedded-source repair reached the bounded preflight, but
all twelve one-second samples exceeded the registered 20% per-CPU ceiling.
Maximum busy fractions ranged from 0.515 to 0.940, principally on CPUs 16, 22,
and 54, and the harness exited with `RECTANGULAR_LIVE_FATAL host-wide preflight
exhausted twelve one-second samples`. The repaired log SHA-256 is
`8b788dd1ed9a8dc6d20fcbaec4fecb856c9222061e2cb4631bfbf7507fd97a4f`.
Fixture construction, candidate/control/live parity, dispatch counts, warmup,
measurement, A/A controls, p50/p95/p99, and bootstrap ratios were never
reached, so neither the earlier profile ratio nor any cold setup observation
is promoted to a result.

**One-line decision: REVERT — the host-wide admission gate exhausted before
timing, so there is no admissible maintenance or competitive result.** The
rectangular production dispatch, spectral arithmetic, public benchmark
switches, focused tests, live-oracle extension, and executable completion
route were manually restored; the profile-only mode remains as negative
evidence for routing future, distinct structural work.

**Concrete retry predicate.** Do not rerun this rectangular constant-stencil
cell. Revisit this family only for a different representation or algorithm
whose untouched candidate and live-incumbent profiles identify a distinct
exclusive cost, with a newly preregistered fixture and admission protocol;
otherwise route to the current worst live ratio outside this cell.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: exact separable `spsolve` for 3-D tensor grids

**Status: PRE-REGISTERED / untouched profile not yet measured.** Bead
`frankenscipy-8l8r1.183`. This is not another rectangular two-dimensional
cell: the input is a seven-point cubic operator with a third Kronecker-sum
axis, its natural half-bandwidth is `side^2`, and the candidate is a distinct
three-axis direct transform. No candidate source or 3-D timing exists yet.

**Why this lever, before implementation.** The current worst sparse routing
loss is structured public `spsolve`: the untouched 2-D rectangular profile was
`0.0830x` SciPy/FrankenSciPy, while the kept square-grid separable route is
more than `10x` faster than genuine live SciPy. A 3-D tensor stencil exposes
the same kind of mathematical structure at a new dimensionality that neither
FrankenSciPy's generic factor route nor SciPy SuperLU recognizes. This is an
incumbent-cannot-follow opportunity only if the untouched profiles below show
that FrankenSciPy pays a distinct map/fill/materialization cost rather than
merely a slower copy of work the incumbent also pays.

**Mandatory untouched whole-job profile and falsifier.** Add only diagnostic
modes to the existing `perf_spsolve` binary and live Python oracle, commit and
strict-remotely build that profile overlay, then compare identical canonical
CSR/RHS SHA-256s for the cubic side-16 matrix (`n=4,096`, diagonal `6.001`,
all six neighbor weights `-1.0`, `b[i]=1+0.5*(i mod 13)`). Profile repeated
public `spsolve` calls in FrankenSciPy and genuine live SciPy 1.17.1 on one
pinned physical core with every numerical thread pool capped at one. Rank
self-time for both engines and classify each leading entry by whether the
incumbent pays the same cost. Proceed only if FrankenSciPy loses by at least
`1.33x` and at least 25% of its self-time is in a distinct safe-Rust sparse
fill/map/permutation or dense-band materialization path not mirrored at the
same multiplicity in the incumbent. If leading cost is shared numeric factor,
SpMV, residual, CSR, or allocation work, close without a candidate and move to
another loss. Profile walls are routing evidence only, never a keep verdict.

**Exactly one production lever.** Before generic sparse/banded factorization,
recognize only default `Auto`/`Colamd` square cubic grids with exact integer
cube root, side at least 8, half-bandwidth `side^2`, exact seven-point nnz and
topology, one finite constant positive diagonal, three finite constant
negative axis weights, every required neighbor exactly once, no extras, and
strict diagonal dominance over all six off-diagonal contributions. Apply the
orthogonal DST-I along z, y, and x, divide by the exact Kronecker-sum spectrum,
then apply the three inverse transforms in a fixed order. Accept only after a
true CSR relative residual at most `1e-8`; recognition, eigenvalue, finite, or
residual failure falls through unchanged. The existing 2-D square route and
all general sparse routes must remain bit-identical. A hidden atomic switch
disables only 3-D admission for same-ELF control, and a hit counter proves
dispatch. No FFT, threading, tolerance, ordering, 2-D transform, or generic
factorization change belongs to this lever.

**Frozen completion job and conformance.** The job contains cubic sides 12,
14, and 16 (`n=1,728/2,744/4,096`, 8,568 materialized output components),
with the coefficients and RHS above. Construction, Python startup/import,
CSR transport, warmup, parity, provenance, and bootstrap are outside timing;
one timed job makes three public `spsolve` calls and folds all output bits.
Before timing, candidate and forced-control true residuals must be at most
`1e-8`, candidate/control relative L2 at most `1e-10`, and the same limits must
hold against genuine live SciPy 1.17.1 on byte-identical fixtures. Candidate
hits must be three and control hits zero. Focused tests must reject one changed
axis coefficient and one missing/extra neighbor, exercise residual fallback,
and prove the pre-existing 2-D `64x64` square solution is bit-identical with
3-D admission enabled or disabled.

**Measurement, admission, and decision.** One frozen `release-perf` ELF runs
at least 21 balanced interleaved rounds of candidate, forced same-ELF control,
and genuine live SciPy plus independent A/A nulls for every arm. Pin all arms
to one physical CPU and cap every numerical pool to one observed thread.
Pre/measurement/post one-second load samples require the pinned CPU and its
SMT sibling each at or below 20% busy and host-wide mean busy at or below 20%;
the busiest unrelated CPU is recorded but is not itself a veto. Record raw
samples, p50/p95/p99, bootstrap-median CI95, all source/ELF/oracle/engine/input
hashes, RCH job/worker, booking claim/release, affinity, actual threads, ISA,
RAM, NUMA, frequency policy, and load samples. Every A/A median must be within
2% of one. KEEP requires the null-corrected control/candidate CI95 lower bound
at least `1.20x` and beyond twice the widest null margin; a competitive claim
additionally requires live-SciPy/candidate CI95 low above `1.0`. Any profile
falsifier, correctness or dispatch failure, inadmissible host, or failed effect
gate means manual restoration and no retry of this cubic cell.

### 2026-08-01 (cod/SilverRiver) — PROFILE PASS: 3-D cubic `spsolve` ordered-map factorization is structural

**Status: PROFILE GATE PASSED / candidate not yet implemented.** The profile
overlay is commit `cf0e7b1a9`, built from that exact clean source by strict
remote release compilation on `hz2`. The retrieved ELF SHA-256 is
`b14ba68fb51165cda109ca49d85237ab74b13e2f5311923b9ed4d38b7c3d1c1d`
(GNU Build ID `2ea3dc5d8f41f4a28f328636b6cca440e37f706a`). Both arms used CPU 31,
one numerical thread, and the identical canonical side-16 CSR/RHS SHA-256
`6f94b78a18a25b68283524afa7a93858a7f4de02e834639b67defee70b11201c`.
The genuine incumbent was SciPy 1.17.1 / NumPy 2.4.3, dispatching through
`scipy.sparse.linalg._dsolve.linsolve.spsolve`; its engine file SHA-256 was
`a890149e6d751bab90654fa93d96632a45dafc829e0e9dd23016c4416a393368`.

**Untouched routing loss.** Six timed public FrankenSciPy solves took
27.397642847 seconds; 500 timed public live-SciPy solves took 23.467995124
seconds. The per-solve walls are 4.566274 s and 0.04693599 s respectively,
so the incumbent is about `97.29x` faster. This exceeds the registered
`1.33x` loss threshold. These walls are routing evidence only.

**Whole-job self-time filter.** `perf record -e cycles:P` captured 30,494
FrankenSciPy samples and 23,559 incumbent samples with zero lost samples.
Flat self-time ranks were:

- FrankenSciPy: `NativeSparseLu::factorize_csr` 59.13%,
  `BTreeMap<usize, SetValZST>::insert` 37.82%, `memmove` 0.29%, and the next
  ordered-map entry/remove/allocation symbols at 0.26%/0.17%/0.14%.
- SciPy: `dgemv_kernel_4x4` 35.70%, SuperLU `dpanel_bmod` 12.16%, BLAS
  `daxpy_k_HASWELL` 6.63%, `colamd` 4.41%, then SuperLU DFS/update/triangular
  kernels at 3.40%/3.05%/2.62%/2.62%/2.40%/2.17%/2.14%.

The incumbent does pay numeric factor/update, ordering, triangular solve, and
copy costs, so those entries are explicitly filtered out as shared work. It
does not pay FrankenSciPy's per-fill ordered-tree insertion/search machinery:
the two leading Rust symbols alone account for 96.95% of samples, far above
the registered 25% structural threshold, while SciPy performs its analogous
numeric work in SuperLU/BLAS storage. Therefore the admitted lever is the
pre-registered exact cubic three-axis DST-I route, not an attempted tune of a
shared numeric kernel. Rust and incumbent profile artifacts have SHA-256s
`2e18f3585dd41a6f7dfa4315f64479cd827aac81699099abfd4eae94014957a2` and
`5361fdf44cc9c0ef861cc8173ce1946092949bd9bf8b0958bc693e0cfe139b56`.

### 2026-08-01 (cod/SilverRiver) — KEEP: exact separable 3-D cubic `spsolve`

**Result class: CAMPAIGN-WIN. Status: competitive KEEP.** **Legacy incumbent arm: SciPy 1.17.1, side-by-side same-invocation.** The sole frozen
completion invocation used the
strict-remote `release-perf` ELF built from commit
`10c58d7152821e09b3ea41a08261df0f53a35e67` by RCH job
`29956918973825160` on `hz2`. The retrieved executable SHA-256 was
`e7594028bf1eb273b6259dcf6d3b378fe1af0d1acb8889a5fc7578ee95daa555`
with GNU Build ID `08b5b2b054788c8ea964eda218c82c98b8fe7f97`; embedded
`linalg.rs` and harness SHA-256s were
`f7b0291c7dd1b43bdba1daccea6ea2774a8e7ade64c1e298deb1b0e208b84c04`
and `fe32dc07b535710751e93d9814fd11e17a876e8a4da6c45b15594930234e7ba2`.
The genuine live arm was SciPy 1.17.1 / NumPy 2.4.3 through
`scipy.sparse.linalg._dsolve.linsolve.spsolve`; its engine SHA-256 was
`a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`
and the persistent oracle SHA-256 was
`48f65eb59b626715dc811b5dd75e3a23b35bff265f13ce8b42b804b3463bf7e2`.
Executed-binary ELF SHA-256:
`e7594028bf1eb273b6259dcf6d3b378fe1af0d1acb8889a5fc7578ee95daa555`.
Named artifacts were
`frankenscipy-engine-sha256=e7594028bf1eb273b6259dcf6d3b378fe1af0d1acb8889a5fc7578ee95daa555`
and
`scipy-engine-sha256=a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`.

**Fixture and conformance.** The one invocation ran cubic sides 12, 14, and
16 (`n=1,728/2,744/4,096`), 8,568 materialized output values, and 21 balanced
rounds on the combined CSR/RHS SHA-256
`97e3ab892c8696a09bdf98e93972710ceced93f2381dbf36fbb04108dd871464`.
The three independently verified fixture SHA-256s were
`02d5424a20304a0e69bd8a6b28eaa97a660f39cf9d0a1c22e3cd2ade1de922d3`,
`86025047e48e6e41958a02b29d7ac31632540d47a2d21b2b4b114b1fc0f71053`,
and `6f94b78a18a25b68283524afa7a93858a7f4de02e834639b67defee70b11201c`.
Candidate/control dispatch counts were 3/0. Maximum true relative residuals
were `1.249e-14` candidate, `2.565e-14` forced control, and `1.213e-14`
live SciPy; candidate/control and candidate/live relative L2 errors were
`5.174e-15` and `5.590e-15`. Focused strict-remote tests also proved exact
recognition/counting/control parity, rejection of changed or missing stencil
entries, true-residual fallback on a stale recognized pattern, and bitwise
isolation of the existing 2-D square route. Recognition and the residual gate
fail closed to the unchanged generic solver, so the lever does not widen the
accepted tolerance or compatibility surface.

**Admission and provenance.** All arms were pinned to physical CPU 31 with
SMT sibling 63 and one requested/observed numerical thread under Agent Mail
claim/release `8726/8729`. Preflight, measurement, and post samples reported
pinned/sibling busy fractions of `0/0`, host means `0.011/0.018/0.013`, and
busiest unrelated CPUs `15:0.059`, `8:0.110`, and `16:0.133`, so every
registered 20% admission condition passed. The host was an AMD Ryzen
Threadripper PRO 5975WX with 231,691,894,784 bytes RAM, one NUMA node, AVX2
and FMA; CPU 31 used `amd-pstate-epp`, `powersave`, performance EPP, and a
1,429,008--4,561,833 kHz policy. Peak process memory was not separately
sampled in this preregistered timing job; no memory claim is made.
Machine-readable provenance was
`host_identity=thinkstation1 physical_cores=32 logical_threads=64
ram_bytes=231691894784 numa_nodes=1 requested_threads=1
actual_observed_worker_threads=1 runtime_detected_isa=avx2,fma
affinity=31 cpuset_logical_cap=1 scaling_governor=powersave
host-wide-quiescence-pre=clear host-wide-quiescence-post=clear`.

**Measured whole-job result.** Candidate p50/p95/p99 was
`0.688094/0.809725/0.998181 ms`; same-ELF forced-control was
`6909.625900/6960.816224/7053.909752 ms`; genuine live SciPy was
`78.872791/84.361010/91.076734 ms`. CVs were 10.435%, 0.607%, and 3.733%
and are provenance only. Independent A/A median and CI95 values were
candidate `0.981003 [0.977466,0.991222]`, control
`0.999444 [0.997157,1.002132]`, and live
`1.000722 [0.997102,1.006193]`; every registered median gate passed. Twice
the widest null margin was `1.046107`. The same-ELF control/candidate median
was **10062.638641x**, CI95 `[9990.744187,10108.096039]`, and genuine live
SciPy/candidate was **115.099391x**, CI95
`[114.604715,115.691604]`. Both the maintenance and competitive gates pass.
The bootstrap-median CI `[114.604715,115.691604]` is DECIDED after the
**2x A/A-null margin**. **Incumbent ratio: SciPy / FrankenSciPy =
115.099391x.**
The immutable 60-line raw log SHA-256 is
`7f492fd9a1f620f1ea9e5e410e53fe445ebc7fbedfa498bba288ff4315b706b3`.

**One-line decision: KEEP — the exact 3-D cubic route is conformant and
115.099391x faster than genuine live SciPy at median, with CI95 low
114.604715x after all A/A and host-admission gates passed.**

**Risk note and scope boundary.** This commit changes only public `spsolve`
dispatch for an exactly recognized, strictly diagonally dominant constant
seven-point cubic tensor grid. It adds no unsafe code, external numerical
backend, thread, tolerance relaxation, artifact-topology change, or generic
factorization change. Any topology, coefficient, finiteness, spectrum, or
true-residual mismatch uses the incumbent generic FrankenSciPy path.

**Quality gates.** Strict-remote `cargo check --workspace --all-targets`
passed on `hz2` (warnings only). The three focused cubic-route library tests
passed strict-remotely, and the live candidate/control/SciPy completion proof
above supplies the differential numerical gate. Owned Rust files pass exact
rustfmt checks, Python byte-compilation passed, `git diff --check` passed, and
UBS reported zero critical findings on the production and harness edits. The
mandatory workspace clippy attempt was blocked only by established untouched
`fsci-linalg`/`fsci-sparse` lints, with no finding in the changed ranges. The
strict-remote `cargo test -p fsci-conformance -- --nocapture` attempt reached
the known `frankenscipy-qo9kw` contract-table/oracle failures (`exact` versus
`mixed`, missing contract-table fixture, and missing oracle repo), then the
pre-existing `mr_eig_trace_identity` long runner; it was stopped after the
crate had already failed, and no sparse packet failure was reported.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: reusable spectral `splu` factor for exact cubic grids

**Status: PRE-REGISTERED / untouched factor job not yet profiled.** Bead
`frankenscipy-8l8r1.184`. This is a widening to a different public contract,
not another timing of the kept one-shot `spsolve` cell and not a retry of the
rejected ILU/GMRES composition. `splu` constructs a reusable
`SparseLuFactorization` and `splu_solve` consumes it across many right-hand
sides; there is no incomplete factorization or Krylov iteration. The fuzzy
ledger preflight surfaced the prior GMRES/SPILU reject, but that row explicitly
permits a different solver/preconditioner surface and does not cover exact
direct `splu` representation.

**Why this lever, before implementation.** The just-completed whole-job cubic
profile puts 96.95% of untouched FrankenSciPy `spsolve` cycles in native sparse
factorization plus ordered-tree insertion, and the exact spectral route is a
115.099391x live-SciPy win. The public Rust factor object exposes shape,
backend, ordering, and solve semantics but no explicit L/U factors. Genuine
SciPy SuperLU must construct its observable factor object; FrankenSciPy can
instead retain an exact separable solve plan. This is an incumbent-cannot-follow
representation opportunity, but it is admitted only after a fresh whole-job
`splu` profile proves the cost persists at this public surface.

**Mandatory untouched profile and incumbent-cost filter.** Add diagnostic
modes only, commit them, and strict-remotely build that exact source. On the
canonical side-16 seven-point cubic CSC (`n=4,096`, diagonal `6.001`, all axis
weights `-1.0`), time and profile one public default `splu` followed by 32
public `splu_solve` calls with
`b_rhs[i] = 1 + 0.125*((17*i + 23*rhs) mod 29)`. Use identical CSC/RHS bytes,
one pinned physical core, one observed numerical thread, genuine live SciPy
1.17.1 `splu(A).solve(b)`, and enough repetitions for stable self-time ranks.
For every leading symbol, record whether the incumbent pays the same cost.
Numeric factor/update, triangular solve, CSC traversal, and output work are
shared and must be filtered out. Proceed only if FrankenSciPy loses by at least
`1.33x` and at least 25% of its self-time is in ordered-map/fill or explicit
factor-object work absent at the same representation/multiplicity in the
incumbent. Profile walls are routing evidence only.

**Exactly one production lever.** For default strict `LuOptions` only
(`Colamd`, pivot threshold exactly `1.0`), reuse the existing exact cubic
topology recognizer at side at least 8 and add one internal
`SparseLuInternal::CubicSpectral` representation. Factorization validates the
entire CSC/CSR topology, finite bit-equal constant coefficients, strict
diagonal dominance, and every spectral eigenvalue, then retains the canonical
matrix plus precomputed DST-I sine table and reciprocal spectrum instead of
materializing L/U. Each `splu_solve` applies the fixed z/y/x forward and inverse
transforms and accepts only after a true matrix relative residual at most
`1e-8`; any recognition or plan-construction failure uses the unchanged native
factorization, and any solve residual failure reconstructs and uses that same
native fallback from the retained matrix. A hidden switch disables only this
`splu` representation in the same ELF, and factor/solve counters prove use.
The kept `spsolve` route, nondefault LU options, non-cubic matrices, public
metadata, and generic factorization arithmetic remain unchanged.

**Frozen completion job, parity, and memory.** In one invocation, factor sides
12, 14, and 16 once each and solve 16 deterministic right-hand sides per
factor, for 48 solves and 137,088 materialized output components per arm.
Construction, CSC transport, Python startup/import, warmup, parity, provenance,
and bootstrap are outside timing; the timed boundary contains the three public
factorizations, 48 public solves, and folded output bits. Candidate factor/solve
hits must be `3/48`; forced-control hits must be zero. All 48 true residuals
must be at most `1e-8`, aggregate candidate/control and candidate/live relative
L2 at most `1e-10`, and every live input hash must match. Focused tests must
reject a changed coefficient and missing neighbor, prove nondefault pivot
threshold fallback, exercise the residual fallback, and prove existing
`spsolve` output bits/counter are isolated. Record candidate/control factor
payload bytes and SciPy L/U array payload bytes without promoting allocator
overhead to a memory claim.

**Measurement, admission, and decision.** One frozen `release-perf` ELF runs
at least 21 balanced interleaved candidate, forced same-ELF control, and
genuine live-SciPy rounds plus independent A/A nulls for every arm. Pin all
arms to one physical CPU, cap every numerical pool to one observed thread, and
use the current pre/measurement/post gate: pinned CPU, SMT sibling, and
host-wide mean each at or below 20% busy; record but do not veto on the busiest
unrelated CPU. Record raw samples, p50/p95/p99, bootstrap-median CI95, source,
ELF, oracle, engine, and input hashes, RCH job/worker, booking claim/release,
affinity, actual threads, ISA, RAM, NUMA, frequency policy, load samples, and
payload bytes. Every A/A median must be within 2% of one. KEEP requires the
null-corrected control/candidate CI95 lower bound at least `1.20x` and beyond
twice the widest null margin; a competitive claim additionally requires the
live-SciPy/candidate CI95 low above `1.0`. Any profile falsifier, correctness,
dispatch, option-isolation, worker, or admission failure restores the candidate
and bans this exact factor-once/16-RHS cubic cell.

### 2026-08-01 (cod/SilverRiver) — PROFILE PASS: reusable `splu` still pays the ordered-map tax

**Status: PROFILE GATE PASSED / candidate not yet implemented.** The diagnostic
overlay is exact commit `1bf473f59b02ec3267027032e42157aed0b5c199`, built
strict-remotely on `hz2`. The retrieved `release-perf` ELF SHA-256 is
`2c0bd87eb8d7c5aba34e0ca70cae5b0bf4f3d70cd6e8129288223e6904d270e0`
(GNU Build ID `47a84308efb7f194fa733ac250642e592d1f4878`). The
embedded harness, Python oracle, and untouched `linalg.rs` SHA-256s were
`00d4e00aa27e3e0b3b2db9b4451d589b0b1fbebbc396b56a87250f9362a29294`,
`03ce1c841b34aad1888c1af340f67d8475b96023e9f7d20e896d09c4f89bdde0`,
and `f7b0291c7dd1b43bdba1daccea6ea2774a8e7ade64c1e298deb1b0e208b84c04`.
Both arms used CPU 31 with SMT sibling 63 and one observed numerical thread;
the canonical side-16 CSC plus all 32 RHS vectors had identical SHA-256
`f7dbcc3ac46ed2ee4b64c7dc66bb91c744d2113033e82b5045014907ecfea108`.
The incumbent was genuine SciPy 1.17.1 / NumPy 2.4.3 through
`scipy.sparse.linalg._dsolve.linsolve.splu`; its engine SHA-256 was
`a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`.

**Untouched routing loss.** Six public FrankenSciPy jobs took
`27.609347543 s`, or `4.601557923833 s/job`; 300 genuine live-SciPy jobs took
`20.047852052 s`, or `0.066826173507 s/job`. The incumbent is therefore
`68.858617550117x` faster on the untouched factor-once/32-solves job, well
past the registered `1.33x` loss threshold. FrankenSciPy's folded checksum was
`8.80836606786967195e2`; SciPy's was `3.78759740918368552e4`, with maximum
true relative residual `1.34426038769945997e-14`. These walls are routing
evidence only.

**Whole-job self-time and incumbent-cost filter.** `perf record -e cycles:P`
captured 30,766 Rust samples and 20,204 incumbent samples with zero lost
samples. Flat Rust self-time ranked `NativeSparseLu::factorize_csr` at 57.78%,
`BTreeMap<usize, SetValZST>::insert` at 37.62%, `NativeSparseLu::solve` at
1.47%, and `memmove` at 0.39%; the next ordered-map insertion/removal symbols
were 0.28%, 0.15%, and 0.13%. Flat SciPy self-time instead ranked BLAS
`dgemv_kernel_4x4` at 24.99%, SuperLU `dgstrs` at 17.16%, `dpanel_bmod` at
8.39%, GEMM copy/kernel work at 6.45%/3.79%, `daxpy` at 4.42%, and `colamd`
at 3.01%, followed by SuperLU DFS/update/triangular kernels.

SciPy pays numeric factor/update, column ordering, triangular solves, CSC
traversal, and output materialization, so all of those are filtered as shared
costs. It does not pay FrankenSciPy's ordered-tree symbolic-fill insertion:
the unambiguous `BTreeMap<usize, SetValZST>::insert` symbol alone is 37.62% of
all Rust cycles, exceeding the registered 25% exclusive-cost threshold without
counting any of the monolithic factorization body. This admits only the
pre-registered solve-only cubic spectral factor representation. The Rust and
SciPy profile artifacts have SHA-256s
`27f96e77af5231ea7062231c4d082f76be39a0feb7764668459a23342e9aab0d`
and `7244dd34b7bb1c7728585e353e9389a3e6edb3505b9066764213fc88916a062f`.
Pre/post pinned CPU busy was `0/0.0099`, sibling busy `0.01/0.0396`, and
host-wide mean busy `0.0473/0.0486`; all registered 20% profile admission
limits passed. Agent Mail claim/release were `8735/8736`.

### 2026-08-01 (cod/SilverRiver) — COMPETITIVE KEEP: reusable cubic `splu` spectral plan

**Result class: CAMPAIGN-WIN. Status: competitive KEEP.** **Legacy incumbent arm: SciPy 1.17.1, side-by-side same-invocation.** Production commit `9dbac856b`
adds the admitted solve-only factor representation; frozen completion source is
exact commit `80db8649f0924a3dbe810cca7c215fc493864d2d`. That detached, clean
source built strict-remotely on `hz2` as RCH job `29956918973825194` with
command `cargo build --profile release-perf -p fsci-sparse --bin perf_spsolve
--features sparse-incumbent-bench`. The retrieved ELF SHA-256 was
`f2a490c2648246a24a74b4865254ccef8369cb33929f03acb66e8cd18d09c454`
and GNU Build ID `c9eae16ae5921e5874217f1119797dd59c511e24`. Embedded
Executed-binary ELF SHA-256:
`f2a490c2648246a24a74b4865254ccef8369cb33929f03acb66e8cd18d09c454`.
Named engine artifacts were
`frankenscipy-engine-sha256=f2a490c2648246a24a74b4865254ccef8369cb33929f03acb66e8cd18d09c454`
and
`scipy-engine-sha256=a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`.

Embedded
`linalg.rs`, harness, and Python-oracle SHA-256s were respectively
`299a0f9b0cc03de5a28a2e43cc1485b863d5e0c184f787d29e393251abd8fd4f`,
`20ecf615f500f02c014ec38cbb07f22b63534de1600c1d5681058684bfcecf1f`,
and `0ae873d1a2f889c0a0e9c1f4493dc560d792c0a44745f0511fec6f06628d7ec4`.
The sole completion log has SHA-256
`a3b074411be4442d2cb8d45f707ec3c12dc2b48765395fd940ed93eae2dd5e5f`
(10,300 bytes, 60 lines). Agent Mail booking claim/release were `8742/8750`.
Machine-readable booking provenance was
`trj_booking_claim_message_id=8742 trj_booking_release_message_id=8750`.

**Frozen job and differential conformance.** The exact sides `12,14,16`,
diagonal `6.001`, three axis weights `-1`, and 16 deterministic RHS vectors per
factor produced shared matrix/RHS SHA-256
`3aadcfccad3f750fb3d161c7ac2ecf6617ce7c4a7d3f880c2cf6a54f60400191`.
The candidate recorded exactly 3 factor and 48 solve hits; the forced same-ELF
control recorded `0/0`. Candidate/control relative L2 was `5.100e-15`; their
maximum true relative residuals were `1.306e-14` and `2.640e-14`. Genuine
SciPy 1.17.1 / NumPy 2.4.3 reported and independently reproduced maximum
relative residual `1.305e-14`, matched every live input hash, and differed from
the candidate by relative L2 `5.562e-15`. All values clear the registered
`1e-8` residual and `1e-10` L2 limits. The live engine was
`scipy.sparse.linalg._dsolve.linsolve` at SHA-256
`a890149562f09a19f0770d91ee5057ecb1068f6bf188abd2d1a79196c15bf388`;
all three live subprocesses proved `fsci_loaded=False`, `genuine=True`, and one
observed worker thread.

**Invariant checklist.** Exact-cubic/default-option recognition, factor/solve
counter isolation, retained-matrix residual fallback, changed-value/missing-edge
rejection, nondefault-pivot rejection, and existing `spsolve` isolation passed
the focused strict-remote tests before the frozen build. The completion job
then proved full-output conformance, true residuals, exact arm counts, same-ELF
control selection, identical input transport, folded all 137,088 outputs, one
observed thread per arm, and all three load admissions. Logical retained-vector
payload was `1,044,280` bytes for the candidate versus `33,519,488` bytes for
the native control and `26,393,160` bytes for SciPy's L/U arrays. These are
representation payloads only: `memory_claim=false`, because allocator peak and
RSS were not measured.

**Whole-job timing.** Each timed sample includes three public `splu` calls, 48
public `splu_solve` calls, and output folding. Across 21 balanced rounds:

| arm | p50 (ms) | p95 (ms) | p99 (ms) | CV provenance |
|---|---:|---:|---:|---:|
| cubic spectral candidate | 7.600235 | 7.753984 | 8.607484 | 2.958% |
| same-ELF native control | 6968.123634 | 6996.453745 | 7124.108524 | 0.604% |
| genuine live SciPy | 91.691905 | 94.626342 | 97.335020 | 2.222% |

Raw candidate seconds:
`0.007753984,0.007597909,0.007658664,0.007667341,0.007568454,0.007587921,0.007618650,0.007600235,0.007677181,0.007475970,0.007645551,0.007598072,0.007672052,0.007637908,0.007592181,0.007586020,0.008607484,0.007556644,0.007484438,0.007697351,0.007583856`.

Raw native-control seconds:
`6.987447848,6.917303219,6.995484645,6.973041090,6.968123634,6.903627682,6.959590612,6.959658892,6.979413490,6.992911555,6.994572642,6.996453745,6.996204367,6.967393931,6.965069173,6.942966294,7.124108524,6.950196722,6.967917395,6.959608872,6.977778279`.

Raw live-SciPy seconds:
`0.092338325,0.097335020,0.088950897,0.089193318,0.092503750,0.092017301,0.089433702,0.091691905,0.089577111,0.091650049,0.093069582,0.094626342,0.091213078,0.088752671,0.089898624,0.091866865,0.091819168,0.092054856,0.089725643,0.091967602,0.090833996`.

**Nulls and decision.** Candidate A/A median was `1.001784`, CI95
`[0.998561,1.007814]`; control A/A was `1.000417`, CI95
`[0.998794,1.001865]`; live A/A was `1.001752`, CI95
`[0.985634,1.024691]`. Thus every null median is inside 2% and twice the widest
null margin is `1.049382x`. The same-ELF control/candidate median is
`913.493941x`, bootstrap-median CI95 `[910.421962,917.400306]`, far beyond
the registered `1.20x` maintenance threshold and twice-null boundary. Genuine
live-SciPy/candidate median is **`11.977284x`**, CI95
**`[11.840949,12.126813]`**, so the competitive lower bound is above one.
The bootstrap-median CI `[11.840949,12.126813]` is DECIDED after the
**2x A/A-null margin**. **Incumbent ratio: SciPy / FrankenSciPy =
11.977284x.** CV is provenance only and was not an acceptance gate.

The preflight, measurement, and postflight samples observed pinned CPU31 busy
`0.020/0.010/0.000`, SMT sibling63 busy `0.010/0.010/0.000`, and host mean
`0.039/0.035/0.025`; each passed the 20% gate on its first attempt. Hardware
was an AMD Ryzen Threadripper PRO 5975WX (64 logical CPUs), 231,691,894,784
bytes RAM, one NUMA node, AVX2+FMA. CPU31 used `amd-pstate-epp`, governor
`powersave`, EPP `performance`, range 1,429,008--4,561,833 kHz.
Machine-readable provenance was
`host_identity=thinkstation1 physical_cores=32 logical_threads=64
ram_bytes=231691894784 numa_nodes=1 requested_threads=1
actual_observed_worker_threads=1 runtime_detected_isa=avx2,fma
affinity=31 cpuset_logical_cap=1 scaling_governor=powersave
host-wide-quiescence-pre=clear host-wide-quiescence-post=clear`.

**One-line decision: KEEP — the reusable exact-cubic `splu` factor is
conformant and 11.977284x faster than genuine live SciPy at median, with CI95
low 11.840949x after all A/A and host-admission gates passed.**

**Quality gates.** From the clean frozen source, strict-remote
`cargo check --workspace --all-targets` passed on `hz2` as RCH job
`29956918973825205`. The mandated workspace Clippy attempt was remote job
`29956918973825206` and stopped before this sparse surface on pre-existing
`fsci-io` `chunks_exact_to_as_chunks` and `fsci-opt` partial-order/range-loop
lints. Workspace `cargo fmt --check` reproduced the pre-existing wrap-only diff
in `crates/fsci-conformance/tests/diff_sparse_iterative_solvers.rs`; that
peer-owned file was not touched. UBS found no unsafe code; its only
critical-class heuristic was the test-only `panic!` assertion that verifies the
internal cubic factor variant. Full `fsci-conformance` remote job
`29956918973825208` passed `sparse_packet_runner_passes`, then was stopped after
the known `mr_eig_trace_identity` runner remained active for more than six
minutes. Before that stop it reproduced only unrelated array-API contract-table
mismatches, a missing contract-table/oracle-repository path, and absent SciPy on
the remote test worker.

**Risk and boundary.** The spectral representation is intentionally narrow:
strict default `Colamd`/pivot `1.0`, exact constant-coefficient cubic topology,
side at least eight, finite nonzero spectrum, retained canonical matrix, and a
true-residual native fallback. Its direct transforms scale differently from
generic sparse triangular solves, so no crossover beyond the measured sides is
claimed. KEEP this lever. Do not rerun the factor-once/16-RHS cubic cell; widen
only to a genuinely different exact separable factor contract or profile a new
worst live-incumbent loss.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: persistent row-band MINRES

**Status: mechanism frozen before candidate implementation or timing.** Bead
`frankenscipy-8l8r1.185`. Exact scouting source is current `main` commit
`96e6cca822bb`; its strict-remote RCH job `29956918973825218` ran on
`vmi1153651`. The retrieved `release-perf` ELF SHA-256 is
`8039b9f7cde099a5a95a686a7031c49f20dfbcf46dffe46076f468a30817eb5c`
(GNU Build ID `bc2cc0b59736e9d098661bacf3789e588f2aa3d3`). No candidate code exists
at this point.

**Why this is the next lever.** A fresh same-invocation live-SciPy scout on the
first threaded SPD MINRES cell (`side=256`, `n=65,536`, `nnz=326,656`,
`rtol=1e-8`) measured current FrankenSciPy at `206.6769 ms` and genuine SciPy
1.17.1 at `82.4769 ms`: **SciPy / FrankenSciPy = `0.3991x`**, the worst current
sparse/solver ratio after stale eigsh and exhausted GMRES/ILU routes were
removed. Both converged, with current FrankenSciPy deliberately satisfying the
stricter true-relative-residual contract (`9.287e-9` versus SciPy's
`1.794e-3`). This three-round scout is routing evidence, not a claim.

Low-frequency whole-job `perf` sampling reproduced the loss under instrumentation
(`0.3079x`; Franken A/A median `1.0095`) and ranked current Rust self-time at
MINRES 21.78% and the capped diagnostic GMRES arm 33.21%. The inclusive call
tree attributes 17.96% to `clone3`, 17.74% to `csr_matvec_into_impl`, 17.52%
to `pthread_create`, and 9.61% to `available_parallelism`; there were zero lost
samples. The genuine SciPy arm instead calls its single-threaded
`csr_matvec_thunk` and does not create a worker team per iteration. Numeric
SpMV/vector work is shared cost and is filtered out. Repeated parallelism
discovery and `2 * 409` worker creations are FrankenSciPy-only structural cost,
so only that lifecycle gap is admitted.

**One lever.** For sufficiently large matrices, public `minres` will create one
safe scoped worker team per solve, partition rows into contiguous approximately
equal-`nnz` bands, and keep each worker's slices of the eight existing Lanczos
vectors cache-resident across iterations. Fixed barrier phases publish the
normalized Lanczos vector, perform the row-local CSR product, reduce dot/norm
partials in worker-index order, and update the Givens/search recurrence. The
matrix is still touched once per iteration and every CSR row accumulates in its
existing index order. Thread creation falls from `O(iterations * workers)` to
`O(workers)`; there is no algorithm, tolerance, preconditioner, or precision
change. A hidden same-ELF switch forces the inherited per-iteration scoped
matvec path, and a counter must prove candidate/control routing. Small matrices
retain the serial path.

**Frozen completion cell and conformance.** Use the exact canonical SPD
five-point fixture above with deterministic RHS, zero initial guess,
`max_iter=20,000`, and exactly the current public `rtol=1e-8` contract. Matrix
construction, Python startup/import, transport, warmup, parity, and provenance
remain outside timing; one timed sample is one public solve plus output folding.
Candidate and same-ELF control must both converge, report true relative residual
at most `1e-8`, have identical iteration count, and agree at relative L2 at most
`1e-10`; every value must be finite. Live SciPy must independently converge on
the identical matrix/RHS hash. Cross-engine component equality is not a gate
because SciPy 1.17.1's public MINRES stopping test demonstrably returns a much
looser true residual on this cell; both engines' actual residuals are reported.
Focused tests must cover forced control, persistent routing, a nonzero initial
guess, a small serial matrix, residual truth, and iteration-count parity.

**Measurement and decision.** One frozen `release-perf` ELF runs at least 21
balanced interleaved persistent-candidate, forced same-ELF-control, and genuine
live-SciPy rounds, with an independent A/A pair for every arm. Record raw
samples, p50/p95/p99, bootstrap-median CI95, source/ELF/oracle/engine hashes,
RCH worker/job, booking claim/release, affinity, requested and actual workers,
ISA, RAM, NUMA, governor/frequency policy, and pre/measurement/post load. Every
A/A median must be within 2% of one; CV is provenance only. KEEP requires both
(a) control/candidate median-CI lower bound at least `1.20x` beyond twice the
widest null margin and (b) live-SciPy/candidate median-CI lower bound above
`1.0`, also beyond twice the widest null margin. Any conformance, routing,
worker-count, admission, or competitive-flip failure restores the candidate and
bans this exact side-256 SPD cell. Report the ratio whichever way it falls.

### 2026-08-01 (cod/SilverRiver) — REVERT: persistent row-band MINRES

**Frozen evidence.** The measured candidate was source commit `6377e8654`,
built by strict-remote RCH job `29956918973825238` on `hz2`. The retrieved
`release-perf` ELF was
`/data/tmp/perf_minres_vs_scipy-6377e8654-final`, SHA-256
`b612ce71a66c386d2381fbe089f2829aea6a60f80e329f7f411b378078bb35f3`,
GNU Build ID `ce1476baf506729d83f6fc50ddbe3deb30a5ed74`. Agent Mail CPU claim
`8762` pinned logical CPUs 30-31; release message `8763` returned them. The
complete 5,042-byte log is
`/data/tmp/frankenscipy-8l8r1-185-minres-final.log`, SHA-256
`b8d0510e5352d92b3257a3e36154311a9a96cf38fd77676898a71c861c5651d2`.
Preflight, measurement, and postflight host busy fractions were
`0.085/0.048/0.026`, each below the registered 20% admission ceiling.

**Conformance and routing.** On the frozen side-256 SPD fixture (`n=65,536`,
`nnz=326,656`, `rtol=1e-8`), candidate and forced same-ELF control both
converged in 409 iterations. Their true relative residuals were respectively
`9.286779e-9` and `9.286758e-9`; relative L2 disagreement was
`7.318866e-14` and maximum absolute disagreement was `9.583800e-11`.
Instrumentation proved one persistent-route hit with exactly two workers for
the candidate and zero hits for the control. Genuine SciPy 1.17.1 independently
converged in 205 iterations with measured true residual `1.793847e-3`; its
looser stopping behavior is reported but was not used to weaken FrankenSciPy's
tolerance contract.

| arm | p50 (ms) | p95 (ms) | p99 (ms) | CV provenance |
|---|---:|---:|---:|---:|
| persistent candidate | 158.953366 | 163.516053 | 163.865184 | 2.372% |
| forced same-ELF scoped control | 144.157796 | 152.072215 | 183.077036 | 6.067% |
| genuine live SciPy | 82.274290 | 85.655457 | 86.359292 | 1.833% |

Candidate A/A median was `1.001538`, CI95 `[0.979789,1.012885]`; control A/A
was `0.985335`, CI95 `[0.965486,1.008476]`; live-SciPy A/A was `1.012214`,
CI95 `[1.006679,1.014656]`. Every null median passed the registered 2% gate,
and twice the widest null margin was `1.071495x`. Nevertheless, the same-ELF
control/candidate median was only **`0.914758x`**, bootstrap-median CI95
**`[0.890555,0.929312]`**, missing the `1.20x` maintenance threshold. Genuine
live-SciPy/candidate was **`0.517057x`**, CI95 **`[0.510540,0.529409]`**:
SciPy remained about `1.934x` faster. CV is provenance only and was not an
acceptance gate.

**One-line decision: REVERT — persistent row-band MINRES is 0.914758x versus
the same-ELF control and 0.517057x versus genuine live SciPy.** Revert commit
`83e796998` restored the incumbent implementation. Under the exact two-core
completion conditions, fixed-barrier and atomic-publication overhead exceeded
the worker-lifecycle cost removed by the candidate. Ban this exact side-256,
two-worker SPD cell; retry only with a different synchronization primitive or
a fixture whose registered profile proves materially more lifecycle tax per
worker phase. After the revert, strict-remote
`cargo test -p fsci-sparse minres --lib` passed all 9 focused tests on
`vmi1153651`; the only diagnostic was the pre-existing dead-code warning for
`fsci-linalg::SYRK_KERNEL_MR4_NR8`.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: genuine LSMR versus delegated LSQR

**Status: frozen before harness support, profiling, or production edits.** Bead
`frankenscipy-8l8r1.186`; base `main` is `316bc9447`. The public `lsmr` body
currently calls `lsqr` directly, while genuine SciPy runs the distinct
Fong--Saunders LSMR recurrence. This is a new algorithmic lane after the
side-256 persistent-MINRES synchronization reject, not a retry of that worker
lifecycle mechanism.

**Whole-job loss and profile gate.** Extend the existing genuine-live sparse
harness only far enough to select `lsmr`; retain its deterministic square
convection--diffusion fixture, exact CSR/RHS transfer, one-core arms, and
construction/serialization/parity work outside timing. Scout sides
64/96/128/192 with current `lsmr` (therefore LSQR) against installed live SciPy
LSMR using `damp=0`, `atol=0`, `btol=1e-5`, `conlim=0`, and the same maximum
iteration count. Then whole-job-profile the worst loss on both engines. Admit a
candidate only if (a) genuine SciPy is at least `1.33x` faster, equivalently
SciPy/current is at most `0.75x`, and (b) at least 20% of current job time or a
material counted-iteration excess belongs to LSQR-specific rotations/vector
updates that genuine LSMR does not pay. The required CSR/CSC matvecs,
Golub--Kahan normalization, residual validation, and shared vector work are
explicitly filtered; if those shared costs explain the loss, close with no
candidate.

**One lever if admitted.** Replace only the delegation with a genuine safe-Rust
LSMR recurrence: one Golub--Kahan CSR/CSC pair per iteration, the two stable
plane rotations and `h`/`hbar` solution update from Fong--Saunders, reusable
buffers, and no precision or tolerance weakening. A hidden same-ELF switch
must force the inherited LSQR delegation, and counters must prove genuine versus
control routing. The returned residual is verified from an exact terminal
`b-Ax`; any recursive convergence trigger whose true residual misses the public
contract continues or fails closed rather than returning a false success.

**Frozen completion gate.** Measure sides 64/96/128 in one at-least-21-round
balanced invocation with genuine candidate, forced same-ELF delegated control,
and genuine live SciPy LSMR plus an independent A/A pair for every arm. All
three must report convergence, finite vectors, true relative residual at most
`1.25e-5`, and relative solution disagreement at most `5e-4`; also report
normal-equation residual and exact iterations. Record raw samples,
p50/p95/p99, source/ELF/oracle hashes, strict-remote worker/job, CPU
claim/release, actual threads, affinity, ISA, RAM, NUMA, frequency policy, and
pre/measurement/post load. Every A/A median must be within 2% of one; CV is
provenance only. KEEP requires delegated-control/candidate bootstrap-median
CI95 low at least `1.20x` beyond twice the widest null margin. A competitive
claim additionally requires live-SciPy/candidate CI95 low above `1.0` beyond
that same margin. Any loss, conformance failure, profile falsifier, or
inadmissible run restores the delegation and bans this exact fixture family.

### 2026-08-01 (cod/SilverRiver) — PROFILE FALSIFIER: genuine LSMR has no live gap

**No production candidate was admitted.** Frozen harness source was commit
`3547154bc`; strict-remote release build ran on `hz2`. The retrieved executable
`/data/tmp/perf_sparse_vs_scipy-3547154bc-lsmr-profile` has SHA-256
`c7c90c24f9bdf093cb161474a3a32866aa891c80184cdb1ce43d925d829c6bee`
and GNU Build ID `4f70d54bdbd31c00c05fc59ffc3ffbed6f9ab75f`. It loaded genuine
SciPy 1.17.1 LSMR from an engine whose SHA-256 is
`f0c46dbfe1496bb9ce1e527b319e13b108960246709ed0f4af9e52390b91aaf2`.
**Executed FrankenSciPy ELF SHA-256:**
`c7c90c24f9bdf093cb161474a3a32866aa891c80184cdb1ce43d925d829c6bee`.
**Executed live-SciPy engine SHA-256:**
`f0c46dbfe1496bb9ce1e527b319e13b108960246709ed0f4af9e52390b91aaf2`.
Agent Mail claim `8766` pinned CPU31 and release `8769` returned it; both arms
requested and observed exactly one thread. The four five-round scouts used the
registered nonexclusive waiver and are routing/profile evidence only, never a
DECIDED competitive claim.

Host `thinkstation1` has 32 physical cores, 64 logical threads,
231,691,894,784 bytes RAM, one NUMA node, and runtime AVX2+FMA. Affinity was
CPU31 with `cpuset_logical_cap=1`; `amd-pstate-epp` reported governor
`powersave`, EPP `performance`, and range 1,429,008--4,561,833 kHz. The
side-192 profile run passed preflight at maximum host busy 6.7%; measurement
and postflight were explicitly nonexclusive and therefore not admission
evidence.

| side / n | delegated LSQR iterations | genuine LSMR iterations | FrankenSciPy p50 (ms) | SciPy p50 (ms) | SciPy / FrankenSciPy |
|---:|---:|---:|---:|---:|---:|
| 64 / 4,096 | 1,355 | 1,376 | 48.208487 | 89.741264 | 1.8611x |
| 96 / 9,216 | 2,325 | 2,359 | 187.885159 | 255.474023 | 1.3621x |
| 128 / 16,384 | 3,572 | 3,656 | 540.838228 | 606.117223 | 1.1261x |
| 192 / 36,864 | 6,081 | 6,191 | 2,071.044204 | 2,136.842110 | 1.0316x |

For the same-invocation side-192 profile, FrankenSciPy A/A median was
`0.999810`, bootstrap-median CI95 `[0.996534,1.002567]`, CV `0.242%`;
live-SciPy A/A median was `0.997913`, CI95 `[0.984249,1.003412]`, CV
`0.734%`. Both null medians are within 2% of one. The SciPy/FrankenSciPy
bootstrap-median CI95 was `[1.0215,1.0365]`, ratio CV `0.547%`; the registered
twice-null endpoint requirement was `1.0320x`, so the timed comparison was
indeterminate even before the nonexclusive downgrade. CV is provenance only
and was not an acceptance gate.

Every cell has the opposite sign from the registered loss threshold of at most
`0.75x`: current FrankenSciPy is faster, not at least `1.33x` slower. The
side-192 vectors were nevertheless numerically comparable: true relative
residuals were `9.959e-6` and `9.973e-6`, relative L2 disagreement was
`1.783e-6`, maximum absolute disagreement `1.376e-3`, and there were zero
component-tolerance mismatches. The current delegation also completed 110
fewer iterations, so replacing it with genuine LSMR cannot remove an
iteration-count gap on this family.

**Whole-job shared-cost filter.** The combined side-192 `cycles:P` profile
captured 4,606 samples with zero lost. Its artifact is
`/data/tmp/perf-lsmr-delegation-3547154bc-side192.data`, SHA-256
`06f95eca5bf5bfd0a72448e9c0aa67ba05d254e5a89027a303f01d9518e7a8f8`.
Ranked flat self-time was Rust CSC matvec 19.04%, Rust CSR matvec 18.88%, SciPy
CSC matvec 18.37%, SciPy CSR matvec 13.51%, Rust LSQR remainder 7.92%, SciPy
NumPy add 4.55%, multiply 4.22%, and OpenBLAS dot 1.78%. For each leader the
incumbent pays the same cost class: one forward and one transpose sparse
matvec plus length-n recurrence work. SciPy additionally pays 1.42% Python
evaluation. No FrankenSciPy-only structural entry reaches the registered 20%
profile gate.

**One-line decision: NO CANDIDATE — keep the genuine-live LSMR measurement
arm, but make no solver change because the loss and exclusive-cost gates both
fail.**
The open correctness bead `frankenscipy-6pdfn` remains the proper owner for
replacing the misleading delegation. Do not reuse this convection--diffusion
family as a performance claim; retry LSMR performance only on a distinct
least-squares fixture whose live whole-job measurement first shows the required
loss and whose profile exposes a non-shared structural cost.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: persistent fused-Jacobi CG

**Status: frozen before diagnostic-harness, profile, or production edits.**
Bead `frankenscipy-8l8r1.187`; base `main` is `92c9a6776`. This widens the
kept persistent row-band CG mechanism into a distinct Jacobi-preconditioned
public solve. It does not retry the rejected two-worker MINRES barrier cell:
the registered side-512 completion fixture has `n=262,144`, `nnz=1,308,672`,
and the kept CG policy selects about nine cache-resident row bands.

**Whole-job profile and incumbent-cost filter.** Before touching production,
add a harness-local scalar Jacobi-PCG reference and a genuine live SciPy
`cg(M=Jacobi LinearOperator)` arm on the exact side-512 constant-diagonal
five-point SPD fixture, deterministic RHS, zero initial guess, `rtol=1e-5`,
and identical maximum iterations. Both form the inverse diagonal outside
timing. Profile the complete scalar Rust and live-incumbent jobs. Required
forward SpMV, dot products, recurrence updates, and one mathematical
preconditioner application are shared costs. Admit the candidate only if at
least 15% of scalar Rust self-time is either repeated per-iteration worker
lifecycle or a distinct full-vector Jacobi materialization that the fused
row-band phase can remove; the live incumbent must demonstrably retain a
separate serial preconditioner call/materialization. If the selected entries
are shared at the same multiplicity, close without production code.

**One lever if admitted.** Add public `cg_jacobi`, deriving and validating one
finite nonzero diagonal inverse per row before iteration. Large matrices reuse
one safe scoped worker team per solve with the kept nnz-balanced contiguous
row bands and narrow column indices. During each local `x/r` update, compute
`z_i = r_i / A_ii` immediately, accumulate fixed-worker-order `r^T z` and true
residual-norm partials, then publish `p_i = z_i + beta*p_i`; do not materialize
or reread a global `z` vector. A hidden same-ELF switch forces a scalar
per-iteration-scoped reference with a separate `z` pass, and counters prove
candidate/control routing. Missing, duplicate, zero, or non-finite diagonals
fail closed. Precision, iteration equations, tolerance, and existing `cg` and
ILU-backed `pcg` APIs remain unchanged.

**Conformance and completion.** Focused tests cover candidate/control routing,
nonzero initial guesses, missing/zero/non-finite diagonals, small serial
matrices, and agreement with unpreconditioned CG when the diagonal is a
positive scalar multiple of identity. Candidate and control must have the same
iteration count, true relative residual at most `1e-5`, relative L2 agreement
at most `1e-10`, and finite outputs. Genuine SciPy must independently converge
on the byte-identical fixture with the same true-residual ceiling and relative
L2 agreement at most `5e-4`.

The frozen performance gate runs side 256 and side 512, at least 21 balanced
interleaved candidate, forced same-ELF scalar-control, and genuine live-SciPy
rounds, plus an independent A/A pair for every arm. Record raw samples,
p50/p95/p99, bootstrap-median CI95, source/ELF/oracle hashes, strict-remote
worker/job, CPU claim/release, affinity, requested and actual workers, ISA,
RAM, NUMA, frequency policy, and pre/measurement/post load. Every A/A median
must be within 2% of one; CV is provenance only. KEEP requires the
scalar-control/candidate CI95 low at least `1.20x` beyond twice the widest null
margin. A competitive claim additionally requires live-SciPy/candidate CI95
low above `1.0` beyond that margin. Any profile falsifier, numerical drift,
missing route/worker proof, admission failure, or candidate loss restores the
candidate and bans these exact cells.

**Result — PROFILE FALSIFIER / NO PRODUCTION CANDIDATE.** The diagnostic
harness landed at `2d439cab7`; its strict-remote `release-perf` ELF was built
on `hz2`, retrieved byte-for-byte, and executed as SHA-256
`7e88b6ef39c336725c93ef95d93ea2c3adc76875ed48e5724290029b6632fec4`
(Build ID `c4bff0185964849a98b9051a2a0ccd429800f86d`). On
`thinkstation1` CPUs `0-11` with every numerical pool capped to one thread,
the side-512 arms both converged in exactly 494 iterations; true residuals
were `9.870e-6` and relative solution L2 difference was `1.442e-15`. Scalar
Rust/live-SciPy p50 was `878.784602/744.924869 ms` across the three-round
profile invocation, so this reference was a real `0.8477x` loss rather than a
favorable-cell escape.

The `cycles:P` whole-job profile captured 3,385 samples with zero lost
(data SHA-256
`8d2f14bcbc2742be6275d7e42c3ae094a3cc20e8fa8c1681e7903708bf2e57ea`).
Top combined self-time was Rust CSR SpMV `26.41%`, Rust solver-local vector
work `24.27%`, SciPy CSR SpMV `22.74%`, SciPy Jacobi multiply `7.76%`, SciPy
dot `4.84%`, and SciPy add `3.93%`. The incumbent pays every one of those
mathematical costs at the same 494-iteration multiplicity, so none is a gap.
Annotated assembly localized the distinct Rust Jacobi materialization to only
`7.94%` of the solver-local symbol, about `3.7%` of Rust self-time, far below
the frozen `15%` admission threshold; the deliberately scalar reference had
no repeated worker lifecycle. Therefore production `cg_jacobi` was never
edited or timed. Keep the genuine Jacobi live-arm harness as evidence
infrastructure; do not retry this constant-diagonal side-512 cell unless a
new profile first exposes a non-shared cost above the registered threshold.

### 2026-08-01 (cod/SilverRiver) — PRE-REGISTERED: persistent row-band BiCGSTAB

**Status: frozen before diagnostic-harness, profile, or production edits.**
Bead `frankenscipy-8l8r1.188`; base `main` is `722f9a116`. This widens the
kept side-512 persistent-CG topology to a nonsymmetric recurrence with two
forward sparse products per iteration. It is not the rejected two-worker
MINRES cell: the primary side-512 convection--diffusion matrix has
`n=262,144`, `nnz=1,308,672`, and the kept `>>17` worker budget selects about
nine cache-resident row bands.

**Untouched loss and whole-job profile gate.** First change only the existing
live-incumbent harness so a diagnostic environment switch admits a multi-CPU
affinity, observes peak Rust tasks during an untimed solve, and leaves the
default single-CPU evidence protocol unchanged. Run current public
`bicgstab` and genuine SciPy 1.17.1 `bicgstab` on the byte-identical side-512
nonsymmetric five-point fixture, deterministic RHS, zero initial guess,
`rtol=1e-5`, `atol=0`, and identical maximum iterations. Profile the complete
current Rust and live jobs in one invocation. Numeric CSR accumulation, dot
products, vector recurrences, convergence checks, and output work are shared
costs and are excluded. Admit production only if (a) the live/current routing
ratio is at most `0.80x`, (b) current Rust observes at least two worker tasks,
and (c) repeated `thread::scope`/clone/pthread creation, parallelism discovery,
or associated join lifecycle from the two scoped matvecs accounts for at
least 15% of Rust self-time while live SciPy observes one worker and has no
equivalent lifecycle. Otherwise close without a candidate.

**Exactly one production lever if admitted.** For `nnz >= 2^20`, create one
safe scoped worker team per public BiCGSTAB solve. Partition contiguous rows
by cumulative nonzeros, narrow column indices once when they fit `u32`, and
retain row-local `x`, `r`, `v`, and `t` slices. Publish the two matvec inputs
`p` and `s` through safe disjoint atomic slots at fixed barrier boundaries.
The team performs both `A*p` and `A*s`, local recurrence updates, and scalar
partials; the coordinator reduces partials in worker-index order and preserves
the existing early-`s` convergence, breakdown, iteration-count, and relative
residual contracts. Matrices below the threshold keep the inherited path. A
hidden same-ELF switch forces the inherited per-matvec scoped control, and
route/worker counters prove selection. No preconditioner, tolerance,
precision, sparse accumulation order, or public signature change belongs to
this lever.

**Conformance and completion.** Focused tests must cover persistent/control
routing, a nonzero initial guess, early `s` convergence, breakdown behavior,
and a small serial matrix. Candidate and control on side 512 must both
converge with the same iteration count, true relative residual at most
`1.25e-5`, relative solution L2 at most `1e-10`, and finite outputs. Genuine
SciPy must independently meet the same residual ceiling and agree with the
candidate to relative L2 at most `5e-4`. The candidate must report the exact
selected persistent-worker count; the forced control must report zero route
hits.

One frozen `release-perf` ELF runs at least 21 balanced interleaved candidate,
forced same-ELF control, and genuine live-SciPy rounds plus an independent A/A
pair for every arm. Record raw samples, p50/p95/p99, bootstrap-median CI95,
source/ELF/oracle/engine hashes, strict-remote worker/job, CPU claim/release,
affinity, requested and observed workers, ISA, RAM, NUMA, frequency policy,
and pre/measurement/post load. Every A/A median must be within 2% of one; CV
is provenance only. KEEP requires control/candidate CI95 low at least `1.20x`
and live-SciPy/candidate CI95 low above `1.0`, both beyond twice the widest
null margin. Any loss/profile/conformance/routing/admission failure restores
the candidate and bans this exact side-512 cell.

**Result — PROFILE FALSIFIER / NO PRODUCTION CANDIDATE.** The multi-CPU
diagnostic harness landed at `48eede21e`; its strict-remote `release-perf` ELF
was built on `ovh-a`, retrieved byte-for-byte, and executed as SHA-256
`1f5a409f59e00e5d8e65e18119e9db0cbd0802b3f5164845ec7e054f0621fcf3`
(Build ID `02954f1a0170bb2188e25897a0bed4f9cfed01e1`). The default one-CPU
evidence protocol is unchanged. The profile-only switch on `thinkstation1`
CPUs `0-11` observed nine current-Rust worker tasks and one genuine-SciPy
worker, with no leaked Rust tasks.

On side 512, current Rust and SciPy converged in 862 and 841 iterations. Their
true relative residuals were `2.539e-6` and `7.702e-6`, relative solution L2
disagreement was `1.752e-6`, maximum absolute disagreement was `2.961e-3`,
and there were zero component-tolerance mismatches. Current Rust/live-SciPy
p50 was `2696.546381/2469.171045 ms`; the live/current ratio was only
`0.9057x`, bootstrap-median CI95 `[0.8815,0.9446]`. Thus the entire observed
CI is above the frozen `<=0.80x` loss admission boundary. The Rust and SciPy
A/A medians were `0.994239` and `1.010278`; the twice-null competitive verdict
was indeterminate, and the host-wide nonexclusive waiver makes this profile
provisional, but neither caveat can turn an interval wholly above `0.80` into
the preregistered large loss.

**Whole-job shared-cost filter.** The `cycles:P` artifact captured 86,130
samples with zero lost (data SHA-256
`6c216ee1b23ef1a55fab42f4a064c2c2e553a08b0ce02957434fbc6da74abc07`).
SciPy accounted for `55.81%` of combined cycles and Rust `44.19%`. Ranked flat
self-time was SciPy CSR matvec `23.19%`, Rust BiCGSTAB numeric and recurrence
work `13.65%`, SciPy NumPy multiply `6.70%`, SciPy OpenBLAS dot `5.27%`,
SciPy add `3.48%`, and SciPy subtract `3.31%`. The incumbent pays the same two
CSR products, reductions, and recurrence updates, so those entries are shared
costs rather than a gap. The one structural entry was real: causal child
coverage attributed `18.45%` of whole-job cycles, or `41.75%` of Rust cycles,
to repeated `csr_matvec_into_impl -> Thread::new -> pthread_create/clone3`
lifecycle, well above the frozen 15% exclusive-cost gate. SciPy has no
corresponding worker lifecycle. The profile therefore passes worker and
exclusive-cost conditions (b) and (c), but fails mandatory loss condition
(a), so production `bicgstab` was never edited or timed.

**One-line decision: KEEP the genuine-live multi-CPU profile harness; NO-SHIP
the persistent BiCGSTAB team because the preregistered live-loss gate failed.**
Do not retry this exact side-512 convection--diffusion cell unless a fresh
current/live invocation first places the complete ratio CI at or below
`0.80x`; route the next BiCGSTAB attempt to a demonstrably worse fixture or a
different structural solver gap.
