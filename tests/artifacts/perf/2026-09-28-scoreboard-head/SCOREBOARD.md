# FrankenSciPy performance scoreboard at HEAD vs live SciPy

Bead `frankenscipy-sw4p0.2`. Generated 2026-09-28T06:31:32+00:00 by `scripts/perf_scoreboard.py build` from the raw run logs in `raw/`. HEAD `54ee859e6c90c3ee8a0f909bfa6e61422a3c30da`, measured on `thinkstation1`.

**Ratio convention: ratio = SciPy time / fsci time. Above 1 means FrankenSciPy is faster.** Harnesses that print fsci/SciPy were inverted and their intervals swapped.

This file replaces the headline role of `docs/GAUNTLET_RELEASE_SCORECARD.md` (June 2026). The ledgers keep their history; nothing in them was rewritten.

## Headline

Primary rows only (one per invocation, case and incumbent variant; planted negative-case rows and superseded replicates are excluded).

| class | all | pinned (1 CPU) | unpinned |
|---|---|---|---|
| WIN | 44 | 30 | 14 |
| LOSE | 34 | 25 | 9 |
| UNRESOLVED | 69 | 20 | 49 |
| SELF_COMPARISON | 0 | 0 | 0 |
| INVALID | 8 | 4 | 4 |
| runs that produced no row (harness refused or failed), plus harnesses not built | 8 | 3 | 4 |

A WIN or LOSE needs: SciPy live in the same invocation (pinned 1.17.1 / numpy 2.4.3, `genuine=true`), a self-reported fsci ELF sha256 equal to the executed binary's, both arms' A/A nulls inside their band (centered: within +/-2%; spread max/min: <= 1.05), the host under the load ceiling of 20 on all three readings (1-min loadavg just before launch; median over the row's own time window of the 1-min loadavg net of the harness's own running threads; median over that window of foreign runnable tasks), no refusal from the harness's own gate, and an interval that does not touch 1.0.

Intervals come from the harness's own bootstrap CI where it prints one (`harness_bootstrap95`), else a bootstrap over the harness's per-round ratios (`builder_bootstrap95_over_rounds`), else the null envelope ratio/(nf*ns) .. ratio*(nf*ns) built from the two arms' own A/A nulls (`null_envelope`). A replicate of the same invocation replaces a row only when the earlier row was load-gated; the choice never looks at the ratio.

## LOSE rows, ranked by magnitude

| rank | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `perf_splu` | n=16384 nnz=81408 | scipy | pinned:5 | **0.484** | [0.465, 0.496] harness_bootstrap95 | 0.987 / 1.000 (centered) | 13.0 / 11.8 / 8.5 (raw la1 max 13.0) | 0.00 | 1/1 | `cdd4b18dc510` | `raw/splu-convection128-solve.pinned.log:30` |
| 2 | `perf_bdf_vs_scipy` | dense-allpairs n=512 method=BDF | scipy | pinned:5 | **0.541** | [0.537, 0.548] harness_bootstrap95 | 0.996 / 1.004 (centered) | 9.1 / 9.4 / 5.0 (raw la1 max 111.2) | 0.16 | 1/1 | `532f88447679` | `raw/bdf-dense512-bdf.pinned.r1.log:20` |
| 3 | `perf_special_vs_scipy` | n200000 op=erfcinv | scipy | unpinned | **0.559** | [0.550, 0.569] null_envelope | 1.009 / 1.008 (spread) | 8.9 / 8.0 / 7.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:123` |
| 4 | `perf_splu` | n=16384 nnz=81408 | scipy | unpinned | **0.565** | [0.546, 0.577] harness_bootstrap95 | 0.997 / 1.008 (centered) | 9.2 / 7.9 / 5.5 (raw la1 max 9.2) | - | 1/1 | `cdd4b18dc510` | `raw/splu-convection128-solve.unpinned.log:30` |
| 5 | `perf_special_vs_scipy` | n200000 op=erfcinv | scipy | pinned:5 | **0.576** | [0.573, 0.579] null_envelope | 1.002 / 1.003 (spread) | 12.8 / 11.4 / 5.0 (raw la1 max 12.4) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:123` |
| 6 | `perf_cluster_vs_scipy` | n1500d8 op=kmeans2 | scipy | pinned:5 | **0.598** | [0.590, 0.606] null_envelope | 1.004 / 1.009 (spread) | 9.3 / 7.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `29095f1270a1` | `raw/cluster.pinned.log:17` |
| 7 | `perf_fft_vs_scipy` | fft n=4194304 | scipy | unpinned | **0.659** | [0.640, 0.679] null_envelope | 1.010 / 0.981 (centered) | 14.4 / 11.2 / 7.0 (raw la1 max 12.2) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:16` |
| 8 | `perf_fft_vs_scipy` | fft n=262144 | scipy | pinned:5 | **0.672** | [0.658, 0.686] null_envelope | 1.011 / 0.990 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:8` |
| 9 | `perf_fft_vs_scipy` | fft n=2097152 | scipy | pinned:5 | **0.682** | [0.673, 0.691] null_envelope | 0.997 / 1.010 (centered) | 9.3 / 7.9 / 6.0 (raw la1 max 8.9) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:14` |
| 10 | `perf_fft_vs_scipy` | fft n=4194304 | scipy | pinned:5 | **0.694** | [0.691, 0.697] null_envelope | 0.999 / 1.004 (centered) | 9.3 / 7.6 / 6.0 (raw la1 max 8.7) | 0.01 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:16` |
| 11 | `perf_special_vs_scipy` | n200000 op=erfinv | scipy | unpinned | **0.695** | [0.692, 0.698] null_envelope | 1.003 / 1.001 (spread) | 8.9 / 8.1 / 7.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:106` |
| 12 | `perf_fft_vs_scipy` | rfft n=4194304 | scipy | pinned:5 | **0.699** | [0.692, 0.706] null_envelope | 1.005 / 1.005 (centered) | 9.3 / 7.7 / 6.0 (raw la1 max 8.8) | 0.01 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:15` |
| 13 | `perf_special_vs_scipy` | n200000 op=erfinv | scipy | pinned:5 | **0.704** | [0.702, 0.706] null_envelope | 1.001 / 1.002 (spread) | 12.8 / 11.4 / 8.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:106` |
| 14 | `perf_special_vs_scipy` | n200000 op=gammaln | scipy | pinned:5 | **0.728** | [0.726, 0.730] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 11.8 / 12.0 (raw la1 max 12.8) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:23` |
| 15 | `perf_special_vs_scipy` | n200000 op=gamma | scipy | pinned:5 | **0.769** | [0.756, 0.782] null_envelope | 1.016 / 1.001 (spread) | 12.8 / 12.0 / 10.0 (raw la1 max 13.0) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:57` |
| 16 | `perf_cluster_vs_scipy` | n1500d8 op=vq | scipy | pinned:5 | **0.776** | [0.768, 0.785] null_envelope | 1.002 / 1.009 (spread) | 9.3 / 8.3 / 4.0 (raw la1 max 9.3) | 0.00 | 1/1 | `29095f1270a1` | `raw/cluster.pinned.log:8` |
| 17 | `perf_ndimage_vs_scipy` | n512 op=median | scipy | pinned:5 | **0.777** | [0.772, 0.782] null_envelope | 1.002 / 1.004 (spread) | 9.1 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:13` |
| 18 | `perf_special_vs_scipy` | n200000 op=rgamma | scipy | pinned:5 | **0.787** | [0.781, 0.793] null_envelope | 1.004 / 1.004 (spread) | 12.8 / 11.8 / 9.5 (raw la1 max 12.8) | 0.01 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:74` |
| 19 | `perf_eigsh_vs_scipy` | lap2d_100_LM_k20 n=10000 k=20 which=LM | scipy | unpinned | **0.793** | [0.745, 0.844] null_envelope | 1.033 / 1.030 (spread) | 12.6 / 9.6 / 4.0 (raw la1 max 11.0) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:16` |
| 20 | `perf_eigsh_vs_scipy` | planted_20000_LM_k6 n=20000 k=6 which=LM | scipy | pinned:5 | **0.797** | [0.789, 0.805] null_envelope | 1.006 / 1.004 (spread) | 9.1 / 8.1 / 8.0 (raw la1 max 9.1) | 0.00 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:7` |
| 21 | `perf_special_vs_scipy` | n200000 op=spence | scipy | pinned:5 | **0.804** | [0.802, 0.806] null_envelope | 1.001 / 1.001 (spread) | 12.8 / 12.2 / 8.0 (raw la1 max 13.3) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:257` |
| 22 | `perf_special_vs_scipy` | n200000 op=j0 | scipy | unpinned | **0.808** | [0.764, 0.855] null_envelope | 1.033 / 1.024 (spread) | 8.9 / 8.0 / 13.0 (raw la1 max 9.0) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:145` |
| 23 | `perf_eigsh_vs_scipy` | lap2d_100_LM_k20 n=10000 k=20 which=LM | scipy | pinned:5 | **0.814** | [0.808, 0.821] null_envelope | 1.004 / 1.004 (spread) | 9.1 / 8.6 / 9.0 (raw la1 max 10.2) | 0.01 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:16` |
| 24 | `perf_special_vs_scipy` | n200000 op=digamma | scipy | pinned:5 | **0.822** | [0.805, 0.839] null_envelope | 1.017 / 1.004 (spread) | 12.8 / 12.0 / 12.0 (raw la1 max 13.0) | 0.03 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:40` |
| 25 | `perf_special_vs_scipy` | n200000 op=j0 | scipy | pinned:5 | **0.824** | [0.822, 0.826] null_envelope | 1.001 / 1.001 (spread) | 12.8 / 11.0 / 11.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:145` |
| 26 | `perf_eigsh_vs_scipy` | planted_20000_LM_k6 n=20000 k=6 which=LM | scipy | unpinned | **0.844** | [0.798, 0.893] null_envelope | 1.028 / 1.029 (spread) | 12.6 / 11.3 / 7.0 (raw la1 max 12.6) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:7` |
| 27 | `perf_splu` | n=13824 nnz=93312 | scipy | pinned:5 | **0.849** | [0.837, 0.865] harness_bootstrap95 | 1.004 / 0.983 (centered) | 11.4 / 8.7 / 6.0 (raw la1 max 11.9) | 0.07 | 1/1 | `cdd4b18dc510` | `raw/splu-default.pinned.log:30` |
| 28 | `perf_ndimage_vs_scipy` | n512 op=uniform | scipy | pinned:5 | **0.890** | [0.866, 0.914] null_envelope | 1.015 / 1.012 (spread) | 9.1 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:10` |
| 29 | `perf_splu` | n=13824 nnz=93312 | scipy | unpinned | **0.898** | [0.880, 0.907] harness_bootstrap95 | 1.002 / 1.002 (centered) | 11.3 / 4.0 / 2.0 (raw la1 max 11.7) | - | 1/1 | `cdd4b18dc510` | `raw/splu-default.unpinned.r1.log:30` |
| 30 | `perf_special_vs_scipy` | n200000 op=y1 | scipy | pinned:5 | **0.929** | [0.897, 0.962] null_envelope | 1.008 / 1.027 (spread) | 12.3 / 8.4 / 1.0 (raw la1 max 9.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.r1.log:172` |
| 31 | `perf_special_vs_scipy` | n200000 op=y0 | scipy | pinned:5 | **0.945** | [0.931, 0.959] null_envelope | 1.006 / 1.009 (spread) | 12.8 / 11.0 / 18.0 (raw la1 max 12.0) | 0.01 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:155` |
| 32 | `perf_eigsh_vs_scipy` | lap2d_100_LM_k1 n=10000 k=1 which=LM | scipy | unpinned | **0.952** | [0.936, 0.968] null_envelope | 1.013 / 1.004 (spread) | 12.6 / 10.6 / 5.0 (raw la1 max 11.6) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:13` |
| 33 | `perf_eigsh_vs_scipy` | lap2d_100_LM_k1 n=10000 k=1 which=LM | scipy | pinned:5 | **0.954** | [0.951, 0.957] null_envelope | 1.001 / 1.002 (spread) | 9.1 / 8.3 / 6.0 (raw la1 max 9.3) | 0.01 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:13` |
| 34 | `perf_signal_vs_scipy` | n1048576 op=lfilter | scipy | pinned:5 | **0.990** | [0.980, 1.000] null_envelope | 1.001 / 1.009 (spread) | 9.1 / 8.1 / 5.0 (raw la1 max 9.1) | 0.00 | 1/1 | `a80c0d861967` | `raw/signal.pinned.log:5` |

## WIN rows

| harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `perf_interpolate_vs_scipy` | n2000m100000g48 op=splev | scipy | unpinned | **29.437** | [28.350, 30.565] null_envelope | 1.024 / 1.014 (spread) | 12.1 / 11.5 / 6.0 (raw la1 max 12.5) | - | 1/1 | `1d4679491d94` | `raw/interpolate.unpinned.log:6` |
| `perf_bdf_vs_scipy` | exact-diagonal n=128 method=BDF | scipy | pinned:5 | **29.358** | [28.766, 29.685] harness_bootstrap95 | 1.008 / 1.004 (centered) | 12.4 / 10.8 / 5.0 (raw la1 max 12.4) | 0.04 | 1/1 | `532f88447679` | `raw/bdf-default.pinned.log:20` |
| `perf_opt_vs_scipy` | n256m256 op=linprog | scipy | pinned:5 | **14.078** | [13.911, 14.247] null_envelope | 1.009 / 1.003 (spread) | 9.3 / 8.1 / 5.0 (raw la1 max 9.3) | 0.00 | 1/32 | `3628d8c5a9e4` | `raw/opt.pinned.log:13` |
| `perf_interpolate_vs_scipy` | n2000m100000g48 op=splev | scipy | pinned:5 | **10.089** | [9.682, 10.514] null_envelope | 1.040 / 1.002 (spread) | 8.7 / 7.7 / 8.0 (raw la1 max 8.9) | 0.04 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:6` |
| `perf_stats_vs_scipy` | n200000t8 op=ks_2samp | scipy | pinned:5 | **3.450** | [3.389, 3.512] null_envelope | 1.010 / 1.008 (spread) | 9.5 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:15` |
| `perf_stats_vs_scipy` | n200000t8 op=spearmanr | scipy | pinned:5 | **2.709** | [2.599, 2.824] null_envelope | 1.033 / 1.009 (spread) | 9.5 / 8.5 / 7.5 (raw la1 max 9.5) | 0.01 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:9` |
| `perf_special_vs_scipy` | n200000 op=i0 | scipy | unpinned | **2.667** | [2.537, 2.804] null_envelope | 1.040 / 1.011 (spread) | 8.9 / 8.9 / 15.5 (raw la1 max 9.9) | - | 8/1 | `b8c631d60934` | `raw/special.unpinned.log:189` |
| `perf_signal_vs_scipy` | n1048576 op=convolve | scipy | unpinned | **2.310** | [2.236, 2.387] null_envelope | 1.022 / 1.011 (spread) | 12.6 / 11.6 / 3.0 (raw la1 max 12.6) | - | 1/1 | `a80c0d861967` | `raw/signal.unpinned.log:7` |
| `perf_special_vs_scipy` | n200000 op=erf | scipy | pinned:5 | **1.955** | [1.943, 1.967] null_envelope | 1.004 / 1.002 (spread) | 12.8 / 11.4 / 8.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:96` |
| `perf_spatial_vs_scipy` | n2000d8 op=kdtree | scipy | unpinned | **1.860** | [1.765, 1.960] null_envelope | 1.025 / 1.028 (spread) | 8.9 / 7.9 / 5.0 (raw la1 max 8.9) | - | 1/1 | `e966375cd0d2` | `raw/spatial.unpinned.log:7` |
| `perf_spatial_vs_scipy` | n2000d8 op=kdtree | scipy | pinned:5 | **1.764** | [1.709, 1.821] null_envelope | 1.015 / 1.017 (spread) | 13.2 / 12.2 / 4.0 (raw la1 max 13.2) | 0.00 | 1/0 | `e966375cd0d2` | `raw/spatial.pinned.r1.log:7` |
| `perf_ndimage_vs_scipy` | n512 op=edt | scipy | unpinned | **1.762** | [1.606, 1.933] null_envelope | 1.046 / 1.049 (spread) | 11.6 / 10.6 / 11.0 (raw la1 max 11.6) | - | 1/1 | `31bf88d2c340` | `raw/ndimage.unpinned.log:16` |
| `perf_special_vs_scipy` | n200000 op=gammaln | scipy | unpinned | **1.719** | [1.644, 1.797] null_envelope | 1.019 / 1.026 (spread) | 8.9 / 7.9 / 6.0 (raw la1 max 8.9) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:23` |
| `perf_special_vs_scipy` | n200000 op=erfc | scipy | pinned:5 | **1.718** | [1.701, 1.735] null_envelope | 1.004 / 1.006 (spread) | 12.8 / 11.4 / 20.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:101` |
| `perf_interpolate_vs_scipy` | n2000m100000g48 op=rgi | scipy | pinned:5 | **1.687** | [1.617, 1.760] null_envelope | 1.002 / 1.041 (spread) | 8.7 / 7.9 / 8.0 (raw la1 max 8.9) | 0.03 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:12` |
| `perf_special_vs_scipy` | n200000 op=digamma | scipy | unpinned | **1.647** | [1.527, 1.776] null_envelope | 1.037 / 1.040 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 9.3) | - | 7/1 | `b8c631d60934` | `raw/special.unpinned.log:40` |
| `perf_ndimage_vs_scipy` | n512 op=edt | scipy | pinned:5 | **1.577** | [1.492, 1.667] null_envelope | 1.032 / 1.024 (spread) | 9.1 / 8.3 / 4.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:16` |
| `perf_chol_vs_scipy` | n=1024 | scipy1 | pinned:5 | **1.517** | [1.502, 1.523] builder_bootstrap95_over_rounds | 1.013 / 1.006 (spread) | 8.9 / 8.0 / 7.0 (raw la1 max 9.1) | 0.00 | 1/4 | `ebc968513e35` | `raw/chol.pinned.log:34` |
| `perf_stats_vs_scipy` | n200000t8 op=rankdata | scipy | pinned:5 | **1.480** | [1.400, 1.564] null_envelope | 1.026 / 1.030 (spread) | 9.5 / 8.5 / 6.0 (raw la1 max 9.5) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:6` |
| `perf_eigsh_vs_scipy` | lap2d_60_SA_k6 n=3600 k=6 which=SA | scipy | unpinned | **1.479** | [1.416, 1.545] null_envelope | 1.026 / 1.018 (spread) | 12.6 / 9.1 / 5.0 (raw la1 max 10.1) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:19` |
| `perf_special_vs_scipy` | n200000 op=zeta | scipy | pinned:5 | **1.476** | [1.470, 1.482] null_envelope | 1.002 / 1.002 (spread) | 12.8 / 11.8 / 11.0 (raw la1 max 12.8) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:91` |
| `perf_eigsh_vs_scipy` | lap2d_60_SA_k6 n=3600 k=6 which=SA | scipy | pinned:5 | **1.418** | [1.375, 1.462] null_envelope | 1.008 / 1.023 (spread) | 9.1 / 9.2 / 7.5 (raw la1 max 10.2) | 0.02 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:19` |
| `perf_chol_vs_scipy` | n=512 | scipy1 | unpinned | **1.383** | [1.360, 1.415] builder_bootstrap95_over_rounds | 1.021 / 1.045 (spread) | 12.0 / 7.5 / 4.5 (raw la1 max 12.0) | - | 1/66 | `ebc968513e35` | `raw/chol.unpinned.log:25` |
| `perf_interpolate_vs_scipy` | n2000m100000g48 op=cubic | scipy | pinned:5 | **1.367** | [1.283, 1.457] null_envelope | 1.048 / 1.017 (spread) | 8.7 / 7.9 / 11.0 (raw la1 max 8.9) | 0.01 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:9` |
| `perf_chol_vs_scipy` | n=512 | scipy1 | pinned:5 | **1.365** | [1.361, 1.373] builder_bootstrap95_over_rounds | 1.005 / 1.005 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 8.9) | 0.01 | 1/3 | `ebc968513e35` | `raw/chol.pinned.log:24` |
| `perf_special_vs_scipy` | n200000 op=dawsn | scipy | pinned:5 | **1.291** | [1.282, 1.300] null_envelope | 1.002 / 1.005 (spread) | 12.8 / 11.0 / 6.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:140` |
| `perf_special_vs_scipy` | n200000 op=exprel | scipy | unpinned | **1.261** | [1.214, 1.309] null_envelope | 1.011 / 1.027 (spread) | 8.9 / 15.9 / 20.0 (raw la1 max 16.9) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:267` |
| `perf_signal_vs_scipy` | n1048576 op=convolve | scipy | pinned:5 | **1.208** | [1.182, 1.235] null_envelope | 1.019 / 1.003 (spread) | 9.1 / 7.1 / 7.0 (raw la1 max 9.1) | 0.00 | 1/1 | `a80c0d861967` | `raw/signal.pinned.log:7` |
| `perf_opt_vs_scipy` | n256m256 op=assignment | scipy | pinned:5 | **1.181** | [1.174, 1.188] null_envelope | 1.003 / 1.003 (spread) | 9.3 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `3628d8c5a9e4` | `raw/opt.pinned.log:6` |
| `perf_special_vs_scipy` | n200000 op=exprel | scipy | pinned:5 | **1.171** | [1.166, 1.176] null_envelope | 1.002 / 1.002 (spread) | 12.8 / 12.2 / 11.0 (raw la1 max 13.2) | 0.00 | 1/0 | `b8c631d60934` | `raw/special.pinned.log:267` |
| `perf_opt_vs_scipy` | n256m256 op=nnls | scipy | pinned:5 | **1.165** | [1.147, 1.184] null_envelope | 1.013 / 1.003 (spread) | 9.3 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `3628d8c5a9e4` | `raw/opt.pinned.log:9` |
| `perf_opt_vs_scipy` | n256m256 op=assignment | scipy | unpinned | **1.163** | [1.127, 1.201] null_envelope | 1.013 / 1.019 (spread) | 11.8 / 10.8 / 3.0 (raw la1 max 11.8) | - | 1/1 | `3628d8c5a9e4` | `raw/opt.unpinned.log:6` |
| `perf_eigsh_vs_scipy` | convdiff2d_60_LR_k6 n=3600 k=6 which=LR | scipy | pinned:5 | **1.149** | [1.130, 1.169] null_envelope | 1.005 / 1.012 (spread) | 9.1 / 9.0 / 7.0 (raw la1 max 10.0) | 0.03 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:22` |
| `perf_stats_vs_scipy` | n200000t8 op=kendalltau | scipy | pinned:5 | **1.145** | [1.110, 1.181] null_envelope | 1.009 / 1.022 (spread) | 9.5 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:12` |
| `perf_special_vs_scipy` | n200000 op=i1 | scipy | pinned:5 | **1.139** | [1.136, 1.142] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 12.3 / 10.0 (raw la1 max 13.3) | 0.09 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:206` |
| `perf_eigsh_vs_scipy` | lap1d_20000_sigma0.5_k6 n=20000 k=6 which=LM | scipy | pinned:5 | **1.123** | [1.098, 1.149] null_envelope | 1.012 / 1.011 (spread) | 9.1 / 8.1 / 10.0 (raw la1 max 9.1) | 0.24 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:10` |
| `perf_eigsh_vs_scipy` | lap1d_20000_sigma0.5_k6 n=20000 k=6 which=LM | scipy | unpinned | **1.097** | [1.056, 1.140] null_envelope | 1.029 / 1.010 (spread) | 12.6 / 11.0 / 5.0 (raw la1 max 12.0) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:10` |
| `perf_eigsh_vs_scipy` | convdiff2d_60_LR_k6 n=3600 k=6 which=LR | scipy | unpinned | **1.091** | [1.055, 1.128] null_envelope | 1.026 / 1.008 (spread) | 12.6 / 8.9 / 4.5 (raw la1 max 9.9) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:22` |
| `perf_fft_vs_scipy` | rfft n=524288 | scipy | pinned:5 | **1.064** | [1.052, 1.077] null_envelope | 0.996 / 1.008 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:9` |
| `perf_special_vs_scipy` | n200000 op=k0 | scipy | pinned:5 | **1.056** | [1.053, 1.059] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 12.3 / 18.5 (raw la1 max 13.3) | 0.05 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:223` |
| `perf_special_vs_scipy` | n200000 op=j1 | scipy | pinned:5 | **1.035** | [1.033, 1.037] null_envelope | 1.002 / 1.000 (spread) | 12.8 / 11.0 / 11.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:150` |
| `perf_fft_vs_scipy` | rfft n=262144 | scipy | pinned:5 | **1.034** | [1.016, 1.052] null_envelope | 0.985 / 1.003 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:7` |
| `perf_fft_vs_scipy` | rfft n=131072 | scipy | unpinned | **1.022** | [1.009, 1.035] null_envelope | 0.994 / 0.993 (centered) | 14.4 / 12.9 / 6.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:5` |
| `perf_chol_vs_scipy` | n=256 | scipy1 | pinned:5 | **1.015** | [1.003, 1.058] builder_bootstrap95_over_rounds | 1.011 / 1.007 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 8.9) | 0.01 | 1/3 | `ebc968513e35` | `raw/chol.pinned.log:14` |

## W7 epic standing losses, re-measured at HEAD

| named loss | prior figure | HEAD verdict | HEAD rows (mode, vs, class, ratio, interval) |
|---|---|---|---|
| add_coo 28672^2 | ~0.062 (perf_ledger_cc.md:13837-13870) | UNRESOLVED: no live-SciPy harness for add_coo exists at HEAD | - |
| splu convection 16-RHS solve stage | 0.442 (NEGATIVE_EVIDENCE.md:42600-42625) | LOSE | pinned:5 scipy LOSE 0.484 [0.465, 0.496] (`raw/splu-convection128-solve.pinned.log:30`); unpinned scipy LOSE 0.565 [0.546, 0.577] (`raw/splu-convection128-solve.unpinned.log:30`) |
| eigh n=768 | 0.276 (commit 932bd498a) | UNRESOLVED: perf_eigh_vs_scipy does not build at HEAD | - |
| dense BDF n=512 | 0.497 (perf_ledger_cc.md:5985) | LOSE | pinned:5 scipy LOSE 0.541 [0.537, 0.548] (`raw/bdf-dense512-bdf.pinned.r1.log:20`) |
| erfinv 200k | 0.728 (commit 42c898d99) | LOSE | pinned:5 scipy LOSE 0.704 [0.702, 0.706] (`raw/special.pinned.log:106`); unpinned scipy LOSE 0.695 [0.692, 0.698] (`raw/special.unpinned.log:106`) |
| erfcinv 200k | 0.721 (commit 42c898d99) | LOSE | pinned:5 scipy LOSE 0.576 [0.573, 0.579] (`raw/special.pinned.log:123`); unpinned scipy LOSE 0.559 [0.550, 0.569] (`raw/special.unpinned.log:123`) |

## Harness status

Every run is listed, replicates included. `peak threads` is the largest task count seen by the 1 Hz sampler over the whole run (fsci process / SciPy child); worker threads that live for less than a sampling interval can be missed, so it is a lower bound.

| invocation | family | mode | log | exit | rows | classes of primary rows | elapsed s | loadavg pre | peak threads | note |
|---|---|---|---|---|---|---|---|---|---|---|
| `perf_bdf_vs_scipy` | integrate | pinned:5 | `raw/bdf-default.pinned.log` | 0 | 1 | WIN 1 | 9 | 12.4 | 1/1 |  |
| `perf_bdf_vs_scipy` | integrate | unpinned | `raw/bdf-default.unpinned.log` | 2 | 0 | - | 0 | 16.4 | -/- | ABORT: pin ordinary solver cells to exactly one CPU; the batch shape cells require an explicit taskset affinity |
| `perf_bdf_vs_scipy 512 21 3 dense bdf` | integrate | pinned:5 | `raw/bdf-dense512-bdf.pinned.log` | 0 | 1 | - | 334 | 18.7 | 1/1 |  |
| `perf_bdf_vs_scipy 512 21 3 dense bdf` | integrate | pinned:5 | `raw/bdf-dense512-bdf.pinned.r1.log` | 0 | 1 | LOSE 1 | 327 | 9.1 | 1/1 |  |
| `perf_bdf_vs_scipy 512 21 3 dense bdf` | integrate | unpinned | `raw/bdf-dense512-bdf.unpinned.log` | 2 | 0 | - | 0 | 9.2 | -/- | ABORT: pin ordinary solver cells to exactly one CPU; the batch shape cells require an explicit taskset affinity |
| `perf_chol_vs_scipy` | linalg | pinned:5 | `raw/chol.pinned.log` | 0 | 6 | UNRESOLVED 3, WIN 3 | 7 | 8.9 | 1/4 |  |
| `perf_chol_vs_scipy` | linalg | unpinned | `raw/chol.unpinned.log` | 0 | 6 | UNRESOLVED 5, WIN 1 | 7 | 12.0 | 65/66 |  |
| `perf_cluster_vs_scipy` | cluster | pinned:5 | `raw/cluster.pinned.log` | 0 | 3 | LOSE 2, UNRESOLVED 1 | 2 | 9.3 | 1/1 |  |
| `perf_cluster_vs_scipy` | cluster | unpinned | `raw/cluster.unpinned.log` | 0 | 3 | UNRESOLVED 3 | 2 | 14.4 | 1/1 |  |
| `perf_eig_vs_scipy` | linalg | pinned:5 | `raw/eig.pinned.log` | 0 | 4 | INVALID 4 | 3 | 9.1 | 1/0 |  |
| `perf_eig_vs_scipy` | linalg | unpinned | `raw/eig.unpinned.log` | 0 | 4 | INVALID 4 | 3 | 12.0 | 1/0 |  |
| `perf_eigsh_vs_scipy` | sparse | pinned:5 | `raw/eigsh.pinned.log` | 0 | 7 | LOSE 3, UNRESOLVED 1, WIN 3 | 43 | 9.1 | 1/1 |  |
| `perf_eigsh_vs_scipy` | sparse | unpinned | `raw/eigsh.unpinned.log` | 0 | 7 | LOSE 3, UNRESOLVED 1, WIN 3 | 44 | 12.6 | 1/1 |  |
| `perf_fft_vs_scipy` | fft | pinned:5 | `raw/fft.pinned.log` | 0 | 14 | LOSE 4, UNRESOLVED 8, WIN 2 | 29 | 9.3 | 1/1 |  |
| `perf_fft_vs_scipy` | fft | unpinned | `raw/fft.unpinned.log` | 0 | 14 | LOSE 1, UNRESOLVED 10, WIN 1 | 32 | 14.4 | 1/1 |  |
| `perf_fft_vs_scipy` | fft | unpinned | `raw/fft.unpinned.r1.log` | 0 | 14 | UNRESOLVED 2 | 30 | 19.9 | 1/1 |  |
| `BINARY_BUILDER_IDENTITY=rch:hz4 BINARY_BUILD_ROUTE=rch-exec-job-release BINARY_SOURCE_COMMIT=54ee859e6 perf_gmres_job_vs_scipy` | sparse | pinned:5 | `raw/gmres_job.pinned.log` | 2 | 0 | - | 0 | 9.8 | -/- | ABORT: TRJ_BOOKING_CLAIM_MESSAGE_ID is required: send an addressed agent-mail message whose subject contains [MEASUREMENT BOOKING] and pass its id |
| `BINARY_BUILDER_IDENTITY=rch:hz4 BINARY_BUILD_ROUTE=rch-exec-job-release BINARY_SOURCE_COMMIT=54ee859e6 perf_gmres_job_vs_scipy` | sparse | unpinned | `raw/gmres_job.unpinned.log` | 2 | 0 | - | 0 | 9.2 | -/- | ABORT: TRJ_BOOKING_CLAIM_MESSAGE_ID is required: send an addressed agent-mail message whose subject contains [MEASUREMENT BOOKING] and pass its id |
| `perf_interpolate_vs_scipy` | interpolate | pinned:5 | `raw/interpolate.pinned.log` | 0 | 3 | WIN 3 | 5 | 8.7 | 1/1 |  |
| `perf_interpolate_vs_scipy` | interpolate | unpinned | `raw/interpolate.unpinned.log` | 0 | 3 | UNRESOLVED 2, WIN 1 | 9 | 12.1 | 1/1 |  |
| `perf_minres_vs_scipy` | sparse | pinned:5 | `raw/minres.pinned.log` | 0 | 1 | - | 3 | 9.8 | 1/1 |  |
| `perf_minres_vs_scipy` | sparse | pinned:5 | `raw/minres.pinned.r1.log` | 0 | 1 | UNRESOLVED 1 | 3 | 13.2 | 1/1 |  |
| `perf_minres_vs_scipy` | sparse | unpinned | `raw/minres.unpinned.log` | 0 | 1 | UNRESOLVED 1 | 3 | 9.2 | 1/1 |  |
| `perf_ndimage_vs_scipy` | ndimage | pinned:5 | `raw/ndimage.pinned.log` | 0 | 4 | LOSE 2, UNRESOLVED 1, WIN 1 | 4 | 9.1 | 1/1 |  |
| `perf_ndimage_vs_scipy` | ndimage | unpinned | `raw/ndimage.unpinned.log` | 0 | 4 | UNRESOLVED 3, WIN 1 | 4 | 11.6 | 5/1 |  |
| `perf_opt_vs_scipy` | optimize | pinned:5 | `raw/opt.pinned.log` | 0 | 3 | WIN 3 | 6 | 9.3 | 1/32 |  |
| `perf_opt_vs_scipy` | optimize | unpinned | `raw/opt.unpinned.log` | 0 | 3 | UNRESOLVED 2, WIN 1 | 5 | 11.8 | 1/32 |  |
| `perf_signal_vs_scipy` | signal | pinned:5 | `raw/signal.pinned.log` | 0 | 2 | LOSE 1, WIN 1 | 2 | 9.1 | 1/1 |  |
| `perf_signal_vs_scipy` | signal | unpinned | `raw/signal.unpinned.log` | 0 | 2 | UNRESOLVED 1, WIN 1 | 3 | 12.6 | 1/1 |  |
| `FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1 perf_sparse_vs_scipy` | sparse | pinned:5 | `raw/sparse-nonexclusive.pinned.log` | 0 | 1 | UNRESOLVED 1 | 4 | 9.7 | 1/1 |  |
| `perf_sparse_vs_scipy` | sparse | pinned:5 | `raw/sparse.pinned.log` | 2 | 0 | - | 0 | 9.7 | -/- | ABORT: host-wide benchmark exclusivity failed during pre; CPUs above 20.0% busy: cpu19=100.0%,cpu22=73.3%,cpu23=83.9%,cpu40=26.7%,cpu48=100.0%,cpu50=100.0%,cpu54=27.6% |
| `perf_sparse_vs_scipy` | sparse | pinned:5 | `raw/sparse.pinned.r1.log` | 2 | 0 | - | 0 | 3.4 | 1/0 | ABORT: host-wide benchmark exclusivity failed during pre; CPUs above 20.0% busy: cpu16=40.0% |
| `perf_sparse_vs_scipy` | sparse | unpinned | `raw/sparse.unpinned.log` | 2 | 0 | - | 0 | 8.9 | -/- | ABORT: pin this invocation to exactly one CPU with taskset |
| `perf_spatial_vs_scipy` | spatial | pinned:5 | `raw/spatial.pinned.log` | 0 | 2 | - | 1 | 9.5 | -/- |  |
| `perf_spatial_vs_scipy` | spatial | pinned:5 | `raw/spatial.pinned.r1.log` | 0 | 2 | UNRESOLVED 1, WIN 1 | 1 | 13.2 | 1/0 |  |
| `perf_spatial_vs_scipy` | spatial | unpinned | `raw/spatial.unpinned.log` | 0 | 2 | UNRESOLVED 1, WIN 1 | 1 | 8.9 | 1/1 |  |
| `perf_special_vs_scipy` | special | pinned:5 | `raw/special.pinned.log` | 0 | 21 | LOSE 9, UNRESOLVED 3, WIN 8 | 50 | 12.8 | 1/1 |  |
| `perf_special_vs_scipy` | special | pinned:5 | `raw/special.pinned.r1.log` | 0 | 21 | LOSE 1 | 50 | 12.3 | 1/1 |  |
| `perf_special_vs_scipy` | special | unpinned | `raw/special.unpinned.log` | 0 | 21 | LOSE 3, UNRESOLVED 11, WIN 4 | 55 | 8.9 | 47/1 |  |
| `perf_special_vs_scipy` | special | unpinned | `raw/special.unpinned.r1.log` | 0 | 21 | UNRESOLVED 3 | 56 | 6.3 | 58/1 |  |
| `FSCI_SPLU_STAGE=solve perf_splu 128 21 4 off convection` | sparse | pinned:5 | `raw/splu-convection128-solve.pinned.log` | 0 | 1 | LOSE 1 | 7 | 13.0 | 1/1 |  |
| `FSCI_SPLU_STAGE=solve perf_splu 128 21 4 off convection` | sparse | unpinned | `raw/splu-convection128-solve.unpinned.log` | 0 | 1 | LOSE 1 | 9 | 9.2 | 1/1 |  |
| `perf_splu` | sparse | pinned:5 | `raw/splu-default.pinned.log` | 0 | 1 | LOSE 1 | 233 | 11.4 | 1/1 |  |
| `perf_splu` | sparse | unpinned | `raw/splu-default.unpinned.log` | 0 | 1 | - | 250 | 16.4 | 1/1 |  |
| `perf_splu` | sparse | unpinned | `raw/splu-default.unpinned.r1.log` | 0 | 1 | LOSE 1 | 231 | 11.3 | 1/1 |  |
| `perf_stats_vs_scipy` | stats | pinned:5 | `raw/stats.pinned.log` | 0 | 4 | WIN 4 | 6 | 9.5 | 1/1 |  |
| `perf_stats_vs_scipy` | stats | unpinned | `raw/stats.unpinned.log` | 0 | 4 | UNRESOLVED 4 | 7 | 8.9 | 1/1 |  |
| `perf_eigh_vs_scipy` | - | - | - | not built | 0 | - | - | - | - | does not compile with its documented feature eigh-incumbent-bench: E0425/E0433 `ScipyIncumbent` not in scope at crates/fsci-linalg/src/bin/perf_eigh_vs_scipy.rs:186-189 (the feature-gated `mod bench` never imports fsci_runtime::scipy_incumbent::ScipyIncumbent); introduced by b63e2a698 (2026-09-01). Compiler output: build/perf_eigh_vs_scipy.compile_messages.txt. Not edited here; UNRESOLVED. |

Live-incumbent harness sources present in the tree but NOT run in this pass: `crates/fsci-integrate/src/bin/perf_dblquad_many_scipy.rs`, `crates/fsci-integrate/src/bin/perf_quad_many_scipy.rs`, `crates/fsci-integrate/src/bin/perf_tplquad_many_scipy.rs`, `crates/fsci-interpolate/src/bin/perf_griddata_scipy.rs`, `crates/fsci-interpolate/src/bin/perf_interpn_scipy.rs`, `crates/fsci-opt/src/bin/perf_curve_fit_many_scipy.rs`, `crates/fsci-opt/src/bin/perf_minimize_many_scipy.rs`, `crates/fsci-opt/src/bin/perf_newton_many_scipy.rs`, `crates/fsci-opt/src/bin/perf_root_many_scipy.rs`, `crates/fsci-stats/src/bin/perf_kde_scipy.rs`, `crates/fsci-stats/src/bin/perf_kruskal_scipy.rs`, `crates/fsci-stats/src/bin/perf_mwu_scipy.rs`, `crates/fsci-stats/src/bin/perf_normality_many_scipy.rs`, `crates/fsci-stats/src/bin/perf_truncweibull_scipy.rs`.

## Harness defects observed

- perf_eigh_vs_scipy does not build at HEAD (see the build-failure row), so the W7 eigh n=768 loss cannot be re-measured by the harness that produced it. The fix is a one-line `use` inside `mod bench`; it was not made here because this pass does not edit harnesses.
- perf_eig_vs_scipy has no SciPy arm despite its name: it times fsci's nalgebra-Schur arm, fsci's Francis-Schur arm and an fsci A/A, prints no ELF sha256 and no incumbent line. Every row it produces is INVALID as a vs-SciPy row.
- perf_minres_vs_scipy carries an A/A null for ours-minres only; the scipy-minres arm has none and its ratio is a quotient of unpaired arm medians. No row from it can be decided.
- perf_chol_vs_scipy and perf_eigh_vs_scipy measure a scipyN (default BLAS threads) arm with no A/A null of its own (chol gates scipyN on null_fsci and null_scipy1; eigh on the fsci null only). Every scipyN row is UNRESOLVED on the board by construction.
- perf_fft_vs_scipy runs a fixed 3 rounds of plain ABBA (never flipped) and its per-arm null is a first-half/second-half median over 3 rounds; at n <= 2^17 the SciPy null read 0.47-0.59 pinned, so the small sizes cannot be decided. Its two arms digest different algorithms' outputs, so agreement between fsci and SciPy is never checked.
- perf_signal_vs_scipy and perf_cluster_vs_scipy interleave plain ABBA every round (no BAAB flip), so the fsci arm always holds the outer slots; position bias is not cancelled.
- The case-style harnesses (cluster, interpolate, ndimage, opt, signal, spatial, special, stats, eigsh) print only per-case medians and a max/min spread null over 5 rounds, with no interval and no per-round samples. The board's interval for these rows is the null envelope ratio/(nf*ns) .. ratio*(nf*ns); CHECK lines report max_abs/max_rel but no harness gates on a tolerance.
- perf_sparse_vs_scipy prints the literal `solver_schedule: frankenscipy_restart=20 scipy_restart=default_20` while HEAD fsci GMRES uses `restart = n.min(30)` (crates/fsci-sparse/src/linalg.rs:7232). The felow change to 20 (f10be8e16, 2026-07-29) was reverted inside the formatting commit 1e12c2d6e (`fsci-sparse: rustfmt ...`, committed 2026-08-03). At HEAD fsci takes 244 inner iterations to SciPy's 163 on the side-64 cell (raw/sparse-nonexclusive.pinned.log). The provenance line is a hard-coded claim, not an observation.
- perf_sparse_vs_scipy requires host-wide quiescence (every one of 64 CPUs <= 20% busy over its sample) and refused on this shared host twice (raw/sparse.pinned.log, raw/sparse.pinned.r1.log, the second with loadavg 3.4 and a single CPU at 40%). Only its FSCI_SPARSE_ALLOW_NON_EXCLUSIVE waiver produced a ratio, which the harness itself labels PROVISIONAL and never DECIDED; the board keeps that row UNRESOLVED.
- perf_gmres_job_vs_scipy requires an agent-mail booking claim (TRJ_BOOKING_CLAIM_MESSAGE_ID resolving to a [MEASUREMENT BOOKING] message sent by AGENT_NAME). No booking was sent in this pass, so it produced no rows (raw/gmres_job.pinned.log, raw/gmres_job.unpinned.log).
- perf_bdf_vs_scipy and perf_sparse_vs_scipy refuse any affinity wider than one CPU by design, so they have no unpinned rows.
- perf_splu prints `quiescence=clear` on the same run whose host-wide quiescence it reports as NOT_CERTIFIED; its `quiescence` token means 'both balanced-square A/A nulls within 0.02', which a reader grepping for host quiescence will misread.
- add_coo: no live-SciPy harness for public add_coo exists at HEAD. The 2026-08-02 ledger row's diagnostic harness extension was removed by 87d6edd1b together with the reverted candidate, so the W7 add_coo loss cannot be re-measured without writing a new harness.
- Agreement flag (recorded, not gated): `perf_cluster_vs_scipy` n1500d8 op=vq pinned:5 reports `max_rel=1` against SciPy's output (`raw/cluster.pinned.log:8`: `CHECK max_abs=1.49011611938476562e-08 max_rel=1.00000000000000000e+00`).
- Agreement flag (recorded, not gated): `perf_cluster_vs_scipy` n1500d8 op=vq unpinned reports `max_rel=1` against SciPy's output (`raw/cluster.unpinned.log:8`: `CHECK max_abs=1.49011611938476562e-08 max_rel=1.00000000000000000e+00`).

## Observations

- Pinned and unpinned rows answer different questions. Unpinned, the fsci special arm reached 47-58 threads at n=200000 (raw/special.unpinned*.trace.tsv) while SciPy's ufuncs stayed on one thread, which is why gammaln/digamma/i0/rgamma flip from LOSE or UNRESOLVED pinned to WIN or UNRESOLVED unpinned; erfinv and erfcinv lose in both modes.
- In raw/opt.pinned.log's run the SciPy child held 32 threads while confined to one CPU (trace column child_threads); if those threads were active the SciPy arm was oversubscribing its own core. The linprog row wins 14.1x pinned and 12.5x unpinned, so the verdict does not hinge on it.
- Replicates: seven invocations had a load-gated row on the first pass and were re-run once (tags ending .r1); perf_sparse_vs_scipy, refused by its own host-quiescence gate, was also re-run once and refused again. A replicate replaced a row only where the earlier row was load-gated; every run is listed in the harness status table and kept under raw/.
- The perf_sparse_vs_scipy GMRES side-64 cell (PROVISIONAL, UNRESOLVED here) read 0.987 [0.985, 0.989] with fsci at 244 iterations to SciPy's 163; see the restart defect above.

## Negative cases (planted, never counted)

Each planted row is built from real HEAD run data (generator and inputs named in the row's `case`); `planted_rows.json` is the builder's input. The last column is what the same row would have been called with the source and executable-identity checks skipped, which is the failure each case exists to catch.

| planted row | ratio (SciPy/fsci) | class | reason | class without those checks |
|---|---|---|---|---|
| NEGATIVE CASE 1 (gmres): fsci p50 19.677214 ms live at HEAD (`raw/sparse-nonexclusive.pinned.log`), SciPy p50 20.786496 ms COPIED from docs/perf_ledger_cc.md:4467 (GMRES side=64 SciPy p50, frankenscipy-felow row, 2026-07-29) | 1.056 [1.054, 1.056] fsci_null_ci_propagated | **INVALID** | SciPy arm not live in the same invocation (source='cached:docs/perf_ledger_cc.md:4467') | WIN |
| NEGATIVE CASE 1 (bdf_dense512): fsci p50 923.388407 ms live at HEAD (`raw/bdf-dense512-bdf.pinned.r1.log`), SciPy p50 511.248 ms COPIED from docs/perf_ledger_cc.md:5979 (dense BDF n=512 live SciPy p50, 2026-08-01) | 0.554 [0.553, 0.561] fsci_null_ci_propagated | **INVALID** | SciPy arm not live in the same invocation (source='cached:docs/perf_ledger_cc.md:5979') | LOSE |
| NEGATIVE CASE 2: perf_splu fsci arm against the SAME fsci ELF (its own `NULL fsci/fsci` over 21 balanced-square rounds, `raw/splu-convection128-solve.pinned.log`), labelled as a SciPy row | 0.987 [0.962, 1.013] null_envelope | **SELF_COMPARISON** | incumbent executable sha256 equals the fsci ELF sha256: fsci-vs-fsci | UNRESOLVED |

## All primary rows

### cluster

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LOSE | `perf_cluster_vs_scipy` | n1500d8 op=kmeans2 | scipy | pinned:5 | **0.598** | [0.590, 0.606] null_envelope | 1.004 / 1.009 (spread) | 9.3 / 7.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `29095f1270a1` | `raw/cluster.pinned.log:17` |
| UNRESOLVED (fsci A/A null 1.073 outside its spread band; scipy A/A null 1.072 outside its spread band) | `perf_cluster_vs_scipy` | n1500d8 op=linkage | scipy | pinned:5 | **1.129** | [0.982, 1.299] null_envelope | 1.073 / 1.072 (spread) | 9.3 / 8.3 / 4.0 (raw la1 max 9.3) | 0.00 | 1/1 | `29095f1270a1` | `raw/cluster.pinned.log:12` |
| LOSE | `perf_cluster_vs_scipy` | n1500d8 op=vq | scipy | pinned:5 | **0.776** | [0.768, 0.785] null_envelope | 1.002 / 1.009 (spread) | 9.3 / 8.3 / 4.0 (raw la1 max 9.3) | 0.00 | 1/1 | `29095f1270a1` | `raw/cluster.pinned.log:8` |
| UNRESOLVED (fsci A/A null 1.119 outside its spread band; scipy A/A null 1.067 outside its spread band) | `perf_cluster_vs_scipy` | n1500d8 op=kmeans2 | scipy | unpinned | **0.637** | [0.534, 0.761] null_envelope | 1.119 / 1.067 (spread) | 14.4 / 13.4 / 16.0 (raw la1 max 14.4) | - | 1/1 | `29095f1270a1` | `raw/cluster.unpinned.log:17` |
| UNRESOLVED (scipy A/A null 1.104 outside its spread band) | `perf_cluster_vs_scipy` | n1500d8 op=linkage | scipy | unpinned | **1.139** | [0.989, 1.312] null_envelope | 1.043 / 1.104 (spread) | 14.4 / 13.4 / 17.0 (raw la1 max 14.4) | - | 1/1 | `29095f1270a1` | `raw/cluster.unpinned.log:12` |
| UNRESOLVED (fsci A/A null 1.055 outside its spread band; scipy A/A null 1.121 outside its spread band) | `perf_cluster_vs_scipy` | n1500d8 op=vq | scipy | unpinned | **0.079** | [0.067, 0.093] null_envelope | 1.055 / 1.121 (spread) | 14.4 / 13.4 / 17.0 (raw la1 max 14.4) | - | 1/1 | `29095f1270a1` | `raw/cluster.unpinned.log:8` |

### fft

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UNRESOLVED (fsci A/A null 1.020176 outside its centered band; scipy A/A null 0.946603 outside its centered band) | `perf_fft_vs_scipy` | fft n=1048576 | scipy | pinned:5 | **0.814** | [0.756, 0.878] null_envelope | 1.020 / 0.947 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:12` |
| UNRESOLVED (fsci A/A null 0.891404 outside its centered band; scipy A/A null 0.589004 outside its centered band) | `perf_fft_vs_scipy` | fft n=131072 | scipy | pinned:5 | **0.708** | [0.372, 1.349] null_envelope | 0.891 / 0.589 (centered) | 9.3 / 8.3 / 6.0 (raw la1 max 9.3) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:6` |
| LOSE | `perf_fft_vs_scipy` | fft n=2097152 | scipy | pinned:5 | **0.682** | [0.673, 0.691] null_envelope | 0.997 / 1.010 (centered) | 9.3 / 7.9 / 6.0 (raw la1 max 8.9) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:14` |
| LOSE | `perf_fft_vs_scipy` | fft n=262144 | scipy | pinned:5 | **0.672** | [0.658, 0.686] null_envelope | 1.011 / 0.990 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:8` |
| LOSE | `perf_fft_vs_scipy` | fft n=4194304 | scipy | pinned:5 | **0.694** | [0.691, 0.697] null_envelope | 0.999 / 1.004 (centered) | 9.3 / 7.6 / 6.0 (raw la1 max 8.7) | 0.01 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:16` |
| UNRESOLVED (scipy A/A null 0.593065 outside its centered band) | `perf_fft_vs_scipy` | fft n=524288 | scipy | pinned:5 | **0.646** | [0.380, 1.098] null_envelope | 0.992 / 0.593 (centered) | 9.3 / 8.2 / 5.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:10` |
| UNRESOLVED (scipy A/A null 0.574578 outside its centered band) | `perf_fft_vs_scipy` | fft n=65536 | scipy | pinned:5 | **0.630** | [0.360, 1.102] null_envelope | 0.995 / 0.575 (centered) | 9.3 / 8.3 / 6.0 (raw la1 max 9.3) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:4` |
| UNRESOLVED (fsci A/A null 1.075908 outside its centered band) | `perf_fft_vs_scipy` | rfft n=1048576 | scipy | pinned:5 | **1.073** | [0.984, 1.169] null_envelope | 1.076 / 0.987 (centered) | 9.3 / 8.2 / 7.0 (raw la1 max 9.2) | 0.01 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:11` |
| UNRESOLVED (fsci A/A null 0.939999 outside its centered band) | `perf_fft_vs_scipy` | rfft n=131072 | scipy | pinned:5 | **0.997** | [0.928, 1.072] null_envelope | 0.940 / 1.010 (centered) | 9.3 / 8.3 / 6.0 (raw la1 max 9.3) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:5` |
| UNRESOLVED (scipy A/A null 1.020195 outside its centered band) | `perf_fft_vs_scipy` | rfft n=2097152 | scipy | pinned:5 | **0.865** | [0.838, 0.893] null_envelope | 0.988 / 1.020 (centered) | 9.3 / 7.9 / 5.5 (raw la1 max 8.9) | 0.04 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:13` |
| WIN | `perf_fft_vs_scipy` | rfft n=262144 | scipy | pinned:5 | **1.034** | [1.016, 1.052] null_envelope | 0.985 / 1.003 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:7` |
| LOSE | `perf_fft_vs_scipy` | rfft n=4194304 | scipy | pinned:5 | **0.699** | [0.692, 0.706] null_envelope | 1.005 / 1.005 (centered) | 9.3 / 7.7 / 6.0 (raw la1 max 8.8) | 0.01 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:15` |
| WIN | `perf_fft_vs_scipy` | rfft n=524288 | scipy | pinned:5 | **1.064** | [1.052, 1.077] null_envelope | 0.996 / 1.008 (centered) | 9.3 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:9` |
| UNRESOLVED (fsci A/A null 0.947094 outside its centered band; scipy A/A null 0.475325 outside its centered band) | `perf_fft_vs_scipy` | rfft n=65536 | scipy | pinned:5 | **0.718** | [0.323, 1.594] null_envelope | 0.947 / 0.475 (centered) | 9.3 / 8.3 / 6.0 (raw la1 max 9.3) | 0.00 | 1/1 | `f619b6142bc7` | `raw/fft.pinned.log:3` |
| UNRESOLVED (scipy A/A null 0.964977 outside its centered band) | `perf_fft_vs_scipy` | fft n=1048576 | scipy | unpinned | **0.695** | [0.669, 0.722] null_envelope | 1.002 / 0.965 (centered) | 14.4 / 12.4 / 7.0 (raw la1 max 13.4) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:12` |
| UNRESOLVED (fsci A/A null 1.03864 outside its centered band; scipy A/A null 0.800176 outside its centered band) | `perf_fft_vs_scipy` | fft n=131072 | scipy | unpinned | **0.694** | [0.534, 0.900] null_envelope | 1.039 / 0.800 (centered) | 14.4 / 12.9 / 6.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:6` |
| UNRESOLVED (fsci A/A null 1.031336 outside its centered band; scipy A/A null 1.132489 outside its centered band) | `perf_fft_vs_scipy` | fft n=2097152 | scipy | unpinned | **0.701** | [0.601, 0.819] null_envelope | 1.031 / 1.132 (centered) | 14.4 / 12.0 / 8.0 (raw la1 max 13.0) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:14` |
| UNRESOLVED (fsci A/A null 1.024553 outside its centered band; scipy A/A null 0.948981 outside its centered band) | `perf_fft_vs_scipy` | fft n=262144 | scipy | unpinned | **0.732** | [0.678, 0.790] null_envelope | 1.025 / 0.949 (centered) | 14.4 / 12.9 / 8.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:8` |
| LOSE | `perf_fft_vs_scipy` | fft n=4194304 | scipy | unpinned | **0.659** | [0.640, 0.679] null_envelope | 1.010 / 0.981 (centered) | 14.4 / 11.2 / 7.0 (raw la1 max 12.2) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:16` |
| UNRESOLVED (scipy A/A null 0.616072 outside its centered band) | `perf_fft_vs_scipy` | fft n=524288 | scipy | unpinned | **0.646** | [0.396, 1.054] null_envelope | 0.995 / 0.616 (centered) | 19.9 / 18.9 / 1.0 (raw la1 max 19.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.r1.log:10` |
| UNRESOLVED (fsci A/A null 0.847516 outside its centered band; scipy A/A null 0.734562 outside its centered band) | `perf_fft_vs_scipy` | fft n=65536 | scipy | unpinned | **0.657** | [0.409, 1.055] null_envelope | 0.848 / 0.735 (centered) | 14.4 / 12.9 / 6.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:4` |
| UNRESOLVED (scipy A/A null 0.924621 outside its centered band) | `perf_fft_vs_scipy` | rfft n=1048576 | scipy | unpinned | **1.223** | [1.114, 1.343] null_envelope | 1.016 / 0.925 (centered) | 14.4 / 12.9 / 7.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:11` |
| WIN | `perf_fft_vs_scipy` | rfft n=131072 | scipy | unpinned | **1.022** | [1.009, 1.035] null_envelope | 0.994 / 0.993 (centered) | 14.4 / 12.9 / 6.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:5` |
| UNRESOLVED (fsci A/A null 1.025338 outside its centered band) | `perf_fft_vs_scipy` | rfft n=2097152 | scipy | unpinned | **0.858** | [0.833, 0.884] null_envelope | 1.025 / 1.005 (centered) | 14.4 / 12.4 / 8.0 (raw la1 max 13.4) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:13` |
| UNRESOLVED (fsci A/A null 0.785943 outside its centered band; scipy A/A null 1.076965 outside its centered band) | `perf_fft_vs_scipy` | rfft n=262144 | scipy | unpinned | **0.998** | [0.728, 1.368] null_envelope | 0.786 / 1.077 (centered) | 14.4 / 12.9 / 8.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:7` |
| UNRESOLVED (fsci A/A null 0.967908 outside its centered band; scipy A/A null 0.956993 outside its centered band) | `perf_fft_vs_scipy` | rfft n=4194304 | scipy | unpinned | **0.754** | [0.698, 0.814] null_envelope | 0.968 / 0.957 (centered) | 14.4 / 11.6 / 8.0 (raw la1 max 12.6) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:15` |
| UNRESOLVED (scipy A/A null 1.087488 outside its centered band) | `perf_fft_vs_scipy` | rfft n=524288 | scipy | unpinned | **1.009** | [0.922, 1.104] null_envelope | 1.006 / 1.087 (centered) | 19.9 / 18.9 / 1.0 (raw la1 max 19.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.r1.log:9` |
| UNRESOLVED (fsci A/A null 1.077836 outside its centered band; scipy A/A null 0.486831 outside its centered band) | `perf_fft_vs_scipy` | rfft n=65536 | scipy | unpinned | **0.668** | [0.302, 1.479] null_envelope | 1.078 / 0.487 (centered) | 14.4 / 12.9 / 6.0 (raw la1 max 13.9) | - | 1/1 | `f619b6142bc7` | `raw/fft.unpinned.log:3` |

### integrate

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LOSE | `perf_bdf_vs_scipy` | dense-allpairs n=512 method=BDF | scipy | pinned:5 | **0.541** | [0.537, 0.548] harness_bootstrap95 | 0.996 / 1.004 (centered) | 9.1 / 9.4 / 5.0 (raw la1 max 111.2) | 0.16 | 1/1 | `532f88447679` | `raw/bdf-dense512-bdf.pinned.r1.log:20` |
| WIN | `perf_bdf_vs_scipy` | exact-diagonal n=128 method=BDF | scipy | pinned:5 | **29.358** | [28.766, 29.685] harness_bootstrap95 | 1.008 / 1.004 (centered) | 12.4 / 10.8 / 5.0 (raw la1 max 12.4) | 0.04 | 1/1 | `532f88447679` | `raw/bdf-default.pinned.log:20` |

### interpolate

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_interpolate_vs_scipy` | n2000m100000g48 op=cubic | scipy | pinned:5 | **1.367** | [1.283, 1.457] null_envelope | 1.048 / 1.017 (spread) | 8.7 / 7.9 / 11.0 (raw la1 max 8.9) | 0.01 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:9` |
| WIN | `perf_interpolate_vs_scipy` | n2000m100000g48 op=rgi | scipy | pinned:5 | **1.687** | [1.617, 1.760] null_envelope | 1.002 / 1.041 (spread) | 8.7 / 7.9 / 8.0 (raw la1 max 8.9) | 0.03 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:12` |
| WIN | `perf_interpolate_vs_scipy` | n2000m100000g48 op=splev | scipy | pinned:5 | **10.089** | [9.682, 10.514] null_envelope | 1.040 / 1.002 (spread) | 8.7 / 7.7 / 8.0 (raw la1 max 8.9) | 0.04 | 1/1 | `1d4679491d94` | `raw/interpolate.pinned.log:6` |
| UNRESOLVED (fsci A/A null 1.07 outside its spread band) | `perf_interpolate_vs_scipy` | n2000m100000g48 op=cubic | scipy | unpinned | **1.390** | [1.258, 1.536] null_envelope | 1.070 / 1.033 (spread) | 12.1 / 11.5 / 4.0 (raw la1 max 12.5) | - | 1/1 | `1d4679491d94` | `raw/interpolate.unpinned.log:9` |
| UNRESOLVED (fsci A/A null 1.117 outside its spread band; scipy A/A null 1.125 outside its spread band) | `perf_interpolate_vs_scipy` | n2000m100000g48 op=rgi | scipy | unpinned | **1.896** | [1.509, 2.383] null_envelope | 1.117 / 1.125 (spread) | 12.1 / 11.0 / 4.0 (raw la1 max 12.0) | - | 1/1 | `1d4679491d94` | `raw/interpolate.unpinned.log:12` |
| WIN | `perf_interpolate_vs_scipy` | n2000m100000g48 op=splev | scipy | unpinned | **29.437** | [28.350, 30.565] null_envelope | 1.024 / 1.014 (spread) | 12.1 / 11.5 / 6.0 (raw la1 max 12.5) | - | 1/1 | `1d4679491d94` | `raw/interpolate.unpinned.log:6` |

### linalg

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_chol_vs_scipy` | n=1024 | scipy1 | pinned:5 | **1.517** | [1.502, 1.523] builder_bootstrap95_over_rounds | 1.013 / 1.006 (spread) | 8.9 / 8.0 / 7.0 (raw la1 max 9.1) | 0.00 | 1/4 | `ebc968513e35` | `raw/chol.pinned.log:34` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_chol_vs_scipy` | n=1024 | scipyN | pinned:5 | **1.549** | [1.529, 1.562] builder_bootstrap95_over_rounds | 1.013 / - (spread) | 8.9 / 8.1 / 7.0 (raw la1 max 9.1) | 0.00 | 1/4 | `ebc968513e35` | `raw/chol.pinned.log:34` |
| WIN | `perf_chol_vs_scipy` | n=256 | scipy1 | pinned:5 | **1.015** | [1.003, 1.058] builder_bootstrap95_over_rounds | 1.011 / 1.007 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 8.9) | 0.01 | 1/3 | `ebc968513e35` | `raw/chol.pinned.log:14` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_chol_vs_scipy` | n=256 | scipyN | pinned:5 | **1.020** | [1.000, 1.044] builder_bootstrap95_over_rounds | 1.011 / - (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 8.9) | 0.01 | 1/3 | `ebc968513e35` | `raw/chol.pinned.log:14` |
| WIN | `perf_chol_vs_scipy` | n=512 | scipy1 | pinned:5 | **1.365** | [1.361, 1.373] builder_bootstrap95_over_rounds | 1.005 / 1.005 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 8.9) | 0.01 | 1/3 | `ebc968513e35` | `raw/chol.pinned.log:24` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_chol_vs_scipy` | n=512 | scipyN | pinned:5 | **1.358** | [1.339, 1.366] builder_bootstrap95_over_rounds | 1.005 / - (spread) | 8.9 / 8.0 / 8.0 (raw la1 max 9.0) | 0.03 | 1/0 | `ebc968513e35` | `raw/chol.pinned.log:24` |
| UNRESOLVED (fsci A/A null 1.081 outside its spread band; harness voided/refused the row: gates=FAIL) | `perf_chol_vs_scipy` | n=1024 | scipy1 | unpinned | **1.844** | [1.739, 1.889] builder_bootstrap95_over_rounds | 1.081 / 1.033 (spread) | 12.0 / -4.0 / 4.0 (raw la1 max 12.0) | - | 65/66 | `ebc968513e35` | `raw/chol.unpinned.log:35` |
| UNRESOLVED (fsci A/A null 1.081 outside its spread band; no A/A null for the scipy arm; harness voided/refused the row: gates=FAIL) | `perf_chol_vs_scipy` | n=1024 | scipyN | unpinned | **1.439** | [1.278, 1.472] builder_bootstrap95_over_rounds | 1.081 / - (spread) | 12.0 / -4.0 / 7.0 (raw la1 max 12.0) | - | 65/66 | `ebc968513e35` | `raw/chol.unpinned.log:35` |
| UNRESOLVED (scipy A/A null 1.245 outside its spread band; harness voided/refused the row: gates=FAIL) | `perf_chol_vs_scipy` | n=256 | scipy1 | unpinned | **0.828** | [0.820, 1.273] builder_bootstrap95_over_rounds | 1.003 / 1.245 (spread) | 12.0 / 11.0 / 4.0 (raw la1 max 12.0) | - | 1/1 | `ebc968513e35` | `raw/chol.unpinned.log:14` |
| UNRESOLVED (no A/A null for the scipy arm; harness voided/refused the row: gates=FAIL) | `perf_chol_vs_scipy` | n=256 | scipyN | unpinned | **1.330** | [1.288, 1.503] builder_bootstrap95_over_rounds | 1.003 / - (spread) | 12.0 / 11.0 / 4.0 (raw la1 max 12.0) | - | 1/1 | `ebc968513e35` | `raw/chol.unpinned.log:14` |
| WIN | `perf_chol_vs_scipy` | n=512 | scipy1 | unpinned | **1.383** | [1.360, 1.415] builder_bootstrap95_over_rounds | 1.021 / 1.045 (spread) | 12.0 / 7.5 / 4.5 (raw la1 max 12.0) | - | 1/66 | `ebc968513e35` | `raw/chol.unpinned.log:25` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_chol_vs_scipy` | n=512 | scipyN | unpinned | **1.313** | [1.210, 1.333] builder_bootstrap95_over_rounds | 1.021 / - (spread) | 12.0 / -4.0 / 5.0 (raw la1 max 12.0) | - | 65/66 | `ebc968513e35` | `raw/chol.unpinned.log:25` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=128 | none (fsci arms only) | pinned:5 | **nan** | -  | - / - (centered) | 9.1 / 8.1 / 8.0 (raw la1 max 9.1) | 0.23 | 1/0 | `-` | `raw/eig.pinned.log:9` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=256 | none (fsci arms only) | pinned:5 | **nan** | -  | - / - (centered) | 9.1 / 8.1 / 6.5 (raw la1 max 9.1) | 0.15 | 1/0 | `-` | `raw/eig.pinned.log:12` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=32 | none (fsci arms only) | pinned:5 | **nan** | -  | - / - (centered) | 9.1 / 8.1 / 8.0 (raw la1 max 9.1) | 0.23 | 1/0 | `-` | `raw/eig.pinned.log:3` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=64 | none (fsci arms only) | pinned:5 | **nan** | -  | - / - (centered) | 9.1 / 8.1 / 8.0 (raw la1 max 9.1) | 0.23 | 1/0 | `-` | `raw/eig.pinned.log:6` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=128 | none (fsci arms only) | unpinned | **nan** | -  | - / - (centered) | 12.0 / 11.0 / 7.0 (raw la1 max 12.0) | - | 1/0 | `-` | `raw/eig.unpinned.log:9` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=256 | none (fsci arms only) | unpinned | **nan** | -  | - / - (centered) | 12.0 / 10.8 / 5.5 (raw la1 max 12.0) | - | 1/0 | `-` | `raw/eig.unpinned.log:12` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=32 | none (fsci arms only) | unpinned | **nan** | -  | - / - (centered) | 12.0 / 11.0 / 7.0 (raw la1 max 12.0) | - | 1/0 | `-` | `raw/eig.unpinned.log:3` |
| INVALID (no self-reported fsci ELF sha256; SciPy arm not live in the same invocation (source='absent'); no scipy_incumbent provenance line) | `perf_eig_vs_scipy` | n=64 | none (fsci arms only) | unpinned | **nan** | -  | - / - (centered) | 12.0 / 11.0 / 7.0 (raw la1 max 12.0) | - | 1/0 | `-` | `raw/eig.unpinned.log:6` |

### ndimage

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_ndimage_vs_scipy` | n512 op=edt | scipy | pinned:5 | **1.577** | [1.492, 1.667] null_envelope | 1.032 / 1.024 (spread) | 9.1 / 8.3 / 4.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:16` |
| UNRESOLVED (fsci A/A null 1.052 outside its spread band) | `perf_ndimage_vs_scipy` | n512 op=gaussian | scipy | pinned:5 | **3.206** | [3.041, 3.379] null_envelope | 1.052 / 1.002 (spread) | 9.1 / 8.1 / 7.0 (raw la1 max 9.1) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:7` |
| LOSE | `perf_ndimage_vs_scipy` | n512 op=median | scipy | pinned:5 | **0.777** | [0.772, 0.782] null_envelope | 1.002 / 1.004 (spread) | 9.1 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:13` |
| LOSE | `perf_ndimage_vs_scipy` | n512 op=uniform | scipy | pinned:5 | **0.890** | [0.866, 0.914] null_envelope | 1.015 / 1.012 (spread) | 9.1 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `31bf88d2c340` | `raw/ndimage.pinned.log:10` |
| WIN | `perf_ndimage_vs_scipy` | n512 op=edt | scipy | unpinned | **1.762** | [1.606, 1.933] null_envelope | 1.046 / 1.049 (spread) | 11.6 / 10.6 / 11.0 (raw la1 max 11.6) | - | 1/1 | `31bf88d2c340` | `raw/ndimage.unpinned.log:16` |
| UNRESOLVED (fsci A/A null 1.068 outside its spread band; scipy A/A null 1.085 outside its spread band) | `perf_ndimage_vs_scipy` | n512 op=gaussian | scipy | unpinned | **2.433** | [2.100, 2.819] null_envelope | 1.068 / 1.085 (spread) | 11.6 / 10.6 / 18.0 (raw la1 max 11.6) | - | 5/1 | `31bf88d2c340` | `raw/ndimage.unpinned.log:7` |
| UNRESOLVED (fsci A/A null 1.075 outside its spread band; scipy A/A null 1.131 outside its spread band) | `perf_ndimage_vs_scipy` | n512 op=median | scipy | unpinned | **0.783** | [0.644, 0.952] null_envelope | 1.075 / 1.131 (spread) | 11.6 / 10.6 / 11.0 (raw la1 max 11.6) | - | 1/1 | `31bf88d2c340` | `raw/ndimage.unpinned.log:13` |
| UNRESOLVED (fsci A/A null 1.223 outside its spread band; scipy A/A null 1.18 outside its spread band) | `perf_ndimage_vs_scipy` | n512 op=uniform | scipy | unpinned | **1.975** | [1.369, 2.850] null_envelope | 1.223 / 1.180 (spread) | 11.6 / 10.6 / 5.0 (raw la1 max 11.6) | - | 1/1 | `31bf88d2c340` | `raw/ndimage.unpinned.log:10` |

### optimize

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_opt_vs_scipy` | n256m256 op=assignment | scipy | pinned:5 | **1.181** | [1.174, 1.188] null_envelope | 1.003 / 1.003 (spread) | 9.3 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `3628d8c5a9e4` | `raw/opt.pinned.log:6` |
| WIN | `perf_opt_vs_scipy` | n256m256 op=linprog | scipy | pinned:5 | **14.078** | [13.911, 14.247] null_envelope | 1.009 / 1.003 (spread) | 9.3 / 8.1 / 5.0 (raw la1 max 9.3) | 0.00 | 1/32 | `3628d8c5a9e4` | `raw/opt.pinned.log:13` |
| WIN | `perf_opt_vs_scipy` | n256m256 op=nnls | scipy | pinned:5 | **1.165** | [1.147, 1.184] null_envelope | 1.013 / 1.003 (spread) | 9.3 / 8.3 / 5.0 (raw la1 max 9.3) | 0.00 | 1/1 | `3628d8c5a9e4` | `raw/opt.pinned.log:9` |
| WIN | `perf_opt_vs_scipy` | n256m256 op=assignment | scipy | unpinned | **1.163** | [1.127, 1.201] null_envelope | 1.013 / 1.019 (spread) | 11.8 / 10.8 / 3.0 (raw la1 max 11.8) | - | 1/1 | `3628d8c5a9e4` | `raw/opt.unpinned.log:6` |
| UNRESOLVED (fsci A/A null 1.124 outside its spread band) | `perf_opt_vs_scipy` | n256m256 op=linprog | scipy | unpinned | **12.480** | [10.697, 14.561] null_envelope | 1.124 / 1.038 (spread) | 11.8 / 10.8 / 3.0 (raw la1 max 11.8) | - | 1/32 | `3628d8c5a9e4` | `raw/opt.unpinned.log:13` |
| UNRESOLVED (fsci A/A null 1.103 outside its spread band; scipy A/A null 1.072 outside its spread band) | `perf_opt_vs_scipy` | n256m256 op=nnls | scipy | unpinned | **1.158** | [0.979, 1.369] null_envelope | 1.103 / 1.072 (spread) | 11.8 / 10.8 / 3.0 (raw la1 max 11.8) | - | 1/1 | `3628d8c5a9e4` | `raw/opt.unpinned.log:9` |

### signal

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_signal_vs_scipy` | n1048576 op=convolve | scipy | pinned:5 | **1.208** | [1.182, 1.235] null_envelope | 1.019 / 1.003 (spread) | 9.1 / 7.1 / 7.0 (raw la1 max 9.1) | 0.00 | 1/1 | `a80c0d861967` | `raw/signal.pinned.log:7` |
| LOSE | `perf_signal_vs_scipy` | n1048576 op=lfilter | scipy | pinned:5 | **0.990** | [0.980, 1.000] null_envelope | 1.001 / 1.009 (spread) | 9.1 / 8.1 / 5.0 (raw la1 max 9.1) | 0.00 | 1/1 | `a80c0d861967` | `raw/signal.pinned.log:5` |
| WIN | `perf_signal_vs_scipy` | n1048576 op=convolve | scipy | unpinned | **2.310** | [2.236, 2.387] null_envelope | 1.022 / 1.011 (spread) | 12.6 / 11.6 / 3.0 (raw la1 max 12.6) | - | 1/1 | `a80c0d861967` | `raw/signal.unpinned.log:7` |
| UNRESOLVED (fsci A/A null 1.101 outside its spread band) | `perf_signal_vs_scipy` | n1048576 op=lfilter | scipy | unpinned | **1.065** | [0.927, 1.224] null_envelope | 1.101 / 1.044 (spread) | 12.6 / 11.6 / 6.0 (raw la1 max 12.6) | - | 1/1 | `a80c0d861967` | `raw/signal.unpinned.log:5` |

### sparse

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UNRESOLVED (fsci A/A null 1.062 outside its spread band) | `perf_eigsh_vs_scipy` | convdiff2d_100_sigma0_k6 n=10000 k=6 which=LM | scipy | pinned:5 | **0.817** | [0.759, 0.880] null_envelope | 1.062 / 1.014 (spread) | 9.1 / 8.7 / 8.5 (raw la1 max 9.8) | 0.06 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:25` |
| WIN | `perf_eigsh_vs_scipy` | convdiff2d_60_LR_k6 n=3600 k=6 which=LR | scipy | pinned:5 | **1.149** | [1.130, 1.169] null_envelope | 1.005 / 1.012 (spread) | 9.1 / 9.0 / 7.0 (raw la1 max 10.0) | 0.03 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:22` |
| WIN | `perf_eigsh_vs_scipy` | lap1d_20000_sigma0.5_k6 n=20000 k=6 which=LM | scipy | pinned:5 | **1.123** | [1.098, 1.149] null_envelope | 1.012 / 1.011 (spread) | 9.1 / 8.1 / 10.0 (raw la1 max 9.1) | 0.24 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:10` |
| LOSE | `perf_eigsh_vs_scipy` | lap2d_100_LM_k1 n=10000 k=1 which=LM | scipy | pinned:5 | **0.954** | [0.951, 0.957] null_envelope | 1.001 / 1.002 (spread) | 9.1 / 8.3 / 6.0 (raw la1 max 9.3) | 0.01 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:13` |
| LOSE | `perf_eigsh_vs_scipy` | lap2d_100_LM_k20 n=10000 k=20 which=LM | scipy | pinned:5 | **0.814** | [0.808, 0.821] null_envelope | 1.004 / 1.004 (spread) | 9.1 / 8.6 / 9.0 (raw la1 max 10.2) | 0.01 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:16` |
| WIN | `perf_eigsh_vs_scipy` | lap2d_60_SA_k6 n=3600 k=6 which=SA | scipy | pinned:5 | **1.418** | [1.375, 1.462] null_envelope | 1.008 / 1.023 (spread) | 9.1 / 9.2 / 7.5 (raw la1 max 10.2) | 0.02 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:19` |
| LOSE | `perf_eigsh_vs_scipy` | planted_20000_LM_k6 n=20000 k=6 which=LM | scipy | pinned:5 | **0.797** | [0.789, 0.805] null_envelope | 1.006 / 1.004 (spread) | 9.1 / 8.1 / 8.0 (raw la1 max 9.1) | 0.00 | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.pinned.log:7` |
| UNRESOLVED (fsci A/A null 1.072 outside its spread band) | `perf_eigsh_vs_scipy` | convdiff2d_100_sigma0_k6 n=10000 k=6 which=LM | scipy | unpinned | **0.688** | [0.613, 0.772] null_envelope | 1.072 / 1.047 (spread) | 12.6 / 8.7 / 4.0 (raw la1 max 9.9) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:25` |
| WIN | `perf_eigsh_vs_scipy` | convdiff2d_60_LR_k6 n=3600 k=6 which=LR | scipy | unpinned | **1.091** | [1.055, 1.128] null_envelope | 1.026 / 1.008 (spread) | 12.6 / 8.9 / 4.5 (raw la1 max 9.9) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:22` |
| WIN | `perf_eigsh_vs_scipy` | lap1d_20000_sigma0.5_k6 n=20000 k=6 which=LM | scipy | unpinned | **1.097** | [1.056, 1.140] null_envelope | 1.029 / 1.010 (spread) | 12.6 / 11.0 / 5.0 (raw la1 max 12.0) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:10` |
| LOSE | `perf_eigsh_vs_scipy` | lap2d_100_LM_k1 n=10000 k=1 which=LM | scipy | unpinned | **0.952** | [0.936, 0.968] null_envelope | 1.013 / 1.004 (spread) | 12.6 / 10.6 / 5.0 (raw la1 max 11.6) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:13` |
| LOSE | `perf_eigsh_vs_scipy` | lap2d_100_LM_k20 n=10000 k=20 which=LM | scipy | unpinned | **0.793** | [0.745, 0.844] null_envelope | 1.033 / 1.030 (spread) | 12.6 / 9.6 / 4.0 (raw la1 max 11.0) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:16` |
| WIN | `perf_eigsh_vs_scipy` | lap2d_60_SA_k6 n=3600 k=6 which=SA | scipy | unpinned | **1.479** | [1.416, 1.545] null_envelope | 1.026 / 1.018 (spread) | 12.6 / 9.1 / 5.0 (raw la1 max 10.1) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:19` |
| LOSE | `perf_eigsh_vs_scipy` | planted_20000_LM_k6 n=20000 k=6 which=LM | scipy | unpinned | **0.844** | [0.798, 0.893] null_envelope | 1.028 / 1.029 (spread) | 12.6 / 11.3 / 7.0 (raw la1 max 12.6) | - | 1/1 | `8fa7f6d3d1e4` | `raw/eigsh.unpinned.log:7` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_minres_vs_scipy` | minres | scipy | pinned:5 | **0.973** | [0.953, 0.991] harness_bootstrap95 | 1.001 / - (centered) | 13.2 / 12.2 / 3.0 (raw la1 max 13.2) | 0.06 | 1/1 | `e481404dc2dd` | `raw/minres.pinned.r1.log:19` |
| UNRESOLVED (no A/A null for the scipy arm) | `perf_minres_vs_scipy` | minres | scipy | unpinned | **0.989** | [0.979, 0.992] harness_bootstrap95 | 1.001 / - (centered) | 9.2 / 8.2 / 3.5 (raw la1 max 9.2) | - | 1/1 | `e481404dc2dd` | `raw/minres.unpinned.log:19` |
| UNRESOLVED (harness voided/refused the row: PROVISIONAL FRANKENSCIPY LOSS (non-exclusive host; NOT DECIDED-class evidence)) | `perf_sparse_vs_scipy` | nonsymmetric-convection-diffusion-2d n=4096 method=gmres | scipy | pinned:5 | **0.987** | [0.985, 0.989] harness_bootstrap95 | 1.001 / 1.000 (centered) | 9.7 / 9.1 / 5.0 (raw la1 max 9.7) | 0.00 | 1/1 | `6f25167539bd` | `raw/sparse-nonexclusive.pinned.log:26` |
| LOSE | `perf_splu` | n=13824 nnz=93312 | scipy | pinned:5 | **0.849** | [0.837, 0.865] harness_bootstrap95 | 1.004 / 0.983 (centered) | 11.4 / 8.7 / 6.0 (raw la1 max 11.9) | 0.07 | 1/1 | `cdd4b18dc510` | `raw/splu-default.pinned.log:30` |
| LOSE | `perf_splu` | n=16384 nnz=81408 | scipy | pinned:5 | **0.484** | [0.465, 0.496] harness_bootstrap95 | 0.987 / 1.000 (centered) | 13.0 / 11.8 / 8.5 (raw la1 max 13.0) | 0.00 | 1/1 | `cdd4b18dc510` | `raw/splu-convection128-solve.pinned.log:30` |
| LOSE | `perf_splu` | n=13824 nnz=93312 | scipy | unpinned | **0.898** | [0.880, 0.907] harness_bootstrap95 | 1.002 / 1.002 (centered) | 11.3 / 4.0 / 2.0 (raw la1 max 11.7) | - | 1/1 | `cdd4b18dc510` | `raw/splu-default.unpinned.r1.log:30` |
| LOSE | `perf_splu` | n=16384 nnz=81408 | scipy | unpinned | **0.565** | [0.546, 0.577] harness_bootstrap95 | 0.997 / 1.008 (centered) | 9.2 / 7.9 / 5.5 (raw la1 max 9.2) | - | 1/1 | `cdd4b18dc510` | `raw/splu-convection128-solve.unpinned.log:30` |

### spatial

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_spatial_vs_scipy` | n2000d8 op=kdtree | scipy | pinned:5 | **1.764** | [1.709, 1.821] null_envelope | 1.015 / 1.017 (spread) | 13.2 / 12.2 / 4.0 (raw la1 max 13.2) | 0.00 | 1/0 | `e966375cd0d2` | `raw/spatial.pinned.r1.log:7` |
| UNRESOLVED (fsci A/A null 1.064 outside its spread band; scipy A/A null 1.061 outside its spread band) | `perf_spatial_vs_scipy` | n2000d8 op=pdist | scipy | pinned:5 | **2.458** | [2.177, 2.775] null_envelope | 1.064 / 1.061 (spread) | 13.2 / 12.2 / 4.0 (raw la1 max 13.2) | 0.00 | 1/0 | `e966375cd0d2` | `raw/spatial.pinned.r1.log:5` |
| WIN | `perf_spatial_vs_scipy` | n2000d8 op=kdtree | scipy | unpinned | **1.860** | [1.765, 1.960] null_envelope | 1.025 / 1.028 (spread) | 8.9 / 7.9 / 5.0 (raw la1 max 8.9) | - | 1/1 | `e966375cd0d2` | `raw/spatial.unpinned.log:7` |
| UNRESOLVED (scipy A/A null 1.141 outside its spread band) | `perf_spatial_vs_scipy` | n2000d8 op=pdist | scipy | unpinned | **2.373** | [2.033, 2.770] null_envelope | 1.023 / 1.141 (spread) | 8.9 / 7.9 / 5.0 (raw la1 max 8.9) | - | 1/1 | `e966375cd0d2` | `raw/spatial.unpinned.log:5` |

### special

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_special_vs_scipy` | n200000 op=dawsn | scipy | pinned:5 | **1.291** | [1.282, 1.300] null_envelope | 1.002 / 1.005 (spread) | 12.8 / 11.0 / 6.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:140` |
| LOSE | `perf_special_vs_scipy` | n200000 op=digamma | scipy | pinned:5 | **0.822** | [0.805, 0.839] null_envelope | 1.017 / 1.004 (spread) | 12.8 / 12.0 / 12.0 (raw la1 max 13.0) | 0.03 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:40` |
| WIN | `perf_special_vs_scipy` | n200000 op=erf | scipy | pinned:5 | **1.955** | [1.943, 1.967] null_envelope | 1.004 / 1.002 (spread) | 12.8 / 11.4 / 8.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:96` |
| WIN | `perf_special_vs_scipy` | n200000 op=erfc | scipy | pinned:5 | **1.718** | [1.701, 1.735] null_envelope | 1.004 / 1.006 (spread) | 12.8 / 11.4 / 20.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:101` |
| LOSE | `perf_special_vs_scipy` | n200000 op=erfcinv | scipy | pinned:5 | **0.576** | [0.573, 0.579] null_envelope | 1.002 / 1.003 (spread) | 12.8 / 11.4 / 5.0 (raw la1 max 12.4) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:123` |
| LOSE | `perf_special_vs_scipy` | n200000 op=erfinv | scipy | pinned:5 | **0.704** | [0.702, 0.706] null_envelope | 1.001 / 1.002 (spread) | 12.8 / 11.4 / 8.0 (raw la1 max 12.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:106` |
| UNRESOLVED (scipy A/A null 1.066 outside its spread band) | `perf_special_vs_scipy` | n200000 op=expit | scipy | pinned:5 | **0.969** | [0.907, 1.035] null_envelope | 1.002 / 1.066 (spread) | 12.8 / 12.2 / 11.0 (raw la1 max 13.2) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:262` |
| WIN | `perf_special_vs_scipy` | n200000 op=exprel | scipy | pinned:5 | **1.171** | [1.166, 1.176] null_envelope | 1.002 / 1.002 (spread) | 12.8 / 12.2 / 11.0 (raw la1 max 13.2) | 0.00 | 1/0 | `b8c631d60934` | `raw/special.pinned.log:267` |
| LOSE | `perf_special_vs_scipy` | n200000 op=gamma | scipy | pinned:5 | **0.769** | [0.756, 0.782] null_envelope | 1.016 / 1.001 (spread) | 12.8 / 12.0 / 10.0 (raw la1 max 13.0) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:57` |
| LOSE | `perf_special_vs_scipy` | n200000 op=gammaln | scipy | pinned:5 | **0.728** | [0.726, 0.730] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 11.8 / 12.0 (raw la1 max 12.8) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:23` |
| UNRESOLVED (fsci A/A null 1.06 outside its spread band; scipy A/A null 1.076 outside its spread band) | `perf_special_vs_scipy` | n200000 op=i0 | scipy | pinned:5 | **1.128** | [0.989, 1.287] null_envelope | 1.060 / 1.076 (spread) | 12.8 / 12.1 / 17.5 (raw la1 max 13.3) | 0.20 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:189` |
| WIN | `perf_special_vs_scipy` | n200000 op=i1 | scipy | pinned:5 | **1.139** | [1.136, 1.142] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 12.3 / 10.0 (raw la1 max 13.3) | 0.09 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:206` |
| LOSE | `perf_special_vs_scipy` | n200000 op=j0 | scipy | pinned:5 | **0.824** | [0.822, 0.826] null_envelope | 1.001 / 1.001 (spread) | 12.8 / 11.0 / 11.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:145` |
| WIN | `perf_special_vs_scipy` | n200000 op=j1 | scipy | pinned:5 | **1.035** | [1.033, 1.037] null_envelope | 1.002 / 1.000 (spread) | 12.8 / 11.0 / 11.0 (raw la1 max 12.0) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:150` |
| WIN | `perf_special_vs_scipy` | n200000 op=k0 | scipy | pinned:5 | **1.056** | [1.053, 1.059] null_envelope | 1.002 / 1.001 (spread) | 12.8 / 12.3 / 18.5 (raw la1 max 13.3) | 0.05 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:223` |
| UNRESOLVED (scipy A/A null 1.051 outside its spread band) | `perf_special_vs_scipy` | n200000 op=k1 | scipy | pinned:5 | **1.092** | [1.035, 1.152] null_envelope | 1.004 / 1.051 (spread) | 12.8 / 12.3 / 11.0 (raw la1 max 13.3) | 0.05 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:240` |
| LOSE | `perf_special_vs_scipy` | n200000 op=rgamma | scipy | pinned:5 | **0.787** | [0.781, 0.793] null_envelope | 1.004 / 1.004 (spread) | 12.8 / 11.8 / 9.5 (raw la1 max 12.8) | 0.01 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:74` |
| LOSE | `perf_special_vs_scipy` | n200000 op=spence | scipy | pinned:5 | **0.804** | [0.802, 0.806] null_envelope | 1.001 / 1.001 (spread) | 12.8 / 12.2 / 8.0 (raw la1 max 13.3) | 0.04 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:257` |
| LOSE | `perf_special_vs_scipy` | n200000 op=y0 | scipy | pinned:5 | **0.945** | [0.931, 0.959] null_envelope | 1.006 / 1.009 (spread) | 12.8 / 11.0 / 18.0 (raw la1 max 12.0) | 0.01 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:155` |
| LOSE | `perf_special_vs_scipy` | n200000 op=y1 | scipy | pinned:5 | **0.929** | [0.897, 0.962] null_envelope | 1.008 / 1.027 (spread) | 12.3 / 8.4 / 1.0 (raw la1 max 9.4) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.r1.log:172` |
| WIN | `perf_special_vs_scipy` | n200000 op=zeta | scipy | pinned:5 | **1.476** | [1.470, 1.482] null_envelope | 1.002 / 1.002 (spread) | 12.8 / 11.8 / 11.0 (raw la1 max 12.8) | 0.00 | 1/1 | `b8c631d60934` | `raw/special.pinned.log:91` |
| UNRESOLVED (fsci A/A null 1.075 outside its spread band) | `perf_special_vs_scipy` | n200000 op=dawsn | scipy | unpinned | **2.268** | [2.075, 2.480] null_envelope | 1.075 / 1.017 (spread) | 8.9 / 8.0 / 10.5 (raw la1 max 9.0) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:140` |
| WIN | `perf_special_vs_scipy` | n200000 op=digamma | scipy | unpinned | **1.647** | [1.527, 1.776] null_envelope | 1.037 / 1.040 (spread) | 8.9 / 7.9 / 7.0 (raw la1 max 9.3) | - | 7/1 | `b8c631d60934` | `raw/special.unpinned.log:40` |
| UNRESOLVED (fsci A/A null 1.146 outside its spread band; scipy A/A null 1.09 outside its spread band) | `perf_special_vs_scipy` | n200000 op=erf | scipy | unpinned | **1.956** | [1.566, 2.443] null_envelope | 1.146 / 1.090 (spread) | 8.9 / 8.1 / 13.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:96` |
| UNRESOLVED (fsci A/A null 1.086 outside its spread band; scipy A/A null 1.063 outside its spread band) | `perf_special_vs_scipy` | n200000 op=erfc | scipy | unpinned | **1.715** | [1.486, 1.980] null_envelope | 1.086 / 1.063 (spread) | 8.9 / 8.1 / 7.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:101` |
| LOSE | `perf_special_vs_scipy` | n200000 op=erfcinv | scipy | unpinned | **0.559** | [0.550, 0.569] null_envelope | 1.009 / 1.008 (spread) | 8.9 / 8.0 / 7.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:123` |
| LOSE | `perf_special_vs_scipy` | n200000 op=erfinv | scipy | unpinned | **0.695** | [0.692, 0.698] null_envelope | 1.003 / 1.001 (spread) | 8.9 / 8.1 / 7.0 (raw la1 max 9.1) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:106` |
| UNRESOLVED (fsci A/A null 1.072 outside its spread band; scipy A/A null 1.108 outside its spread band) | `perf_special_vs_scipy` | n200000 op=expit | scipy | unpinned | **0.950** | [0.800, 1.128] null_envelope | 1.072 / 1.108 (spread) | 8.9 / 15.9 / 20.0 (raw la1 max 16.9) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:262` |
| WIN | `perf_special_vs_scipy` | n200000 op=exprel | scipy | unpinned | **1.261** | [1.214, 1.309] null_envelope | 1.011 / 1.027 (spread) | 8.9 / 15.9 / 20.0 (raw la1 max 16.9) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:267` |
| UNRESOLVED (fsci A/A null 1.062 outside its spread band; scipy A/A null 1.201 outside its spread band) | `perf_special_vs_scipy` | n200000 op=gamma | scipy | unpinned | **0.718** | [0.563, 0.916] null_envelope | 1.062 / 1.201 (spread) | 8.9 / 8.3 / 7.0 (raw la1 max 9.3) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:57` |
| WIN | `perf_special_vs_scipy` | n200000 op=gammaln | scipy | unpinned | **1.719** | [1.644, 1.797] null_envelope | 1.019 / 1.026 (spread) | 8.9 / 7.9 / 6.0 (raw la1 max 8.9) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:23` |
| WIN | `perf_special_vs_scipy` | n200000 op=i0 | scipy | unpinned | **2.667** | [2.537, 2.804] null_envelope | 1.040 / 1.011 (spread) | 8.9 / 8.9 / 15.5 (raw la1 max 9.9) | - | 8/1 | `b8c631d60934` | `raw/special.unpinned.log:189` |
| UNRESOLVED (scipy A/A null 1.064 outside its spread band) | `perf_special_vs_scipy` | n200000 op=i1 | scipy | unpinned | **2.728** | [2.519, 2.955] null_envelope | 1.018 / 1.064 (spread) | 6.3 / -2.1 / 0.5 (raw la1 max 9.4) | - | 43/1 | `b8c631d60934` | `raw/special.unpinned.r1.log:206` |
| LOSE | `perf_special_vs_scipy` | n200000 op=j0 | scipy | unpinned | **0.808** | [0.764, 0.855] null_envelope | 1.033 / 1.024 (spread) | 8.9 / 8.0 / 13.0 (raw la1 max 9.0) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:145` |
| UNRESOLVED (fsci A/A null 1.152 outside its spread band; scipy A/A null 1.134 outside its spread band) | `perf_special_vs_scipy` | n200000 op=j1 | scipy | unpinned | **1.024** | [0.784, 1.338] null_envelope | 1.152 / 1.134 (spread) | 8.9 / 8.6 / 15.0 (raw la1 max 9.6) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:150` |
| UNRESOLVED (fsci A/A null 1.076 outside its spread band; scipy A/A null 1.111 outside its spread band) | `perf_special_vs_scipy` | n200000 op=k0 | scipy | unpinned | **2.533** | [2.119, 3.028] null_envelope | 1.076 / 1.111 (spread) | 6.3 / 7.8 / 9.0 (raw la1 max 9.4) | - | 7/1 | `b8c631d60934` | `raw/special.unpinned.r1.log:223` |
| UNRESOLVED (fsci A/A null 1.07 outside its spread band) | `perf_special_vs_scipy` | n200000 op=k1 | scipy | unpinned | **2.653** | [2.377, 2.961] null_envelope | 1.070 / 1.043 (spread) | 8.9 / 15.6 / 17.5 (raw la1 max 17.1) | - | 9/1 | `b8c631d60934` | `raw/special.unpinned.log:240` |
| UNRESOLVED (scipy A/A null 1.074 outside its spread band) | `perf_special_vs_scipy` | n200000 op=rgamma | scipy | unpinned | **2.328** | [2.078, 2.608] null_envelope | 1.043 / 1.074 (spread) | 8.9 / 8.2 / 7.0 (raw la1 max 9.3) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:74` |
| UNRESOLVED (fsci A/A null 1.076 outside its spread band; scipy A/A null 1.091 outside its spread band) | `perf_special_vs_scipy` | n200000 op=spence | scipy | unpinned | **2.455** | [2.091, 2.882] null_envelope | 1.076 / 1.091 (spread) | 6.3 / 11.1 / 9.0 (raw la1 max 12.1) | - | 6/1 | `b8c631d60934` | `raw/special.unpinned.r1.log:257` |
| UNRESOLVED (scipy A/A null 1.075 outside its spread band) | `perf_special_vs_scipy` | n200000 op=y0 | scipy | unpinned | **1.388** | [1.256, 1.534] null_envelope | 1.028 / 1.075 (spread) | 8.9 / 8.6 / 17.0 (raw la1 max 9.6) | - | 1/1 | `b8c631d60934` | `raw/special.unpinned.log:155` |
| UNRESOLVED (fsci A/A null 1.217 outside its spread band; scipy A/A null 1.058 outside its spread band) | `perf_special_vs_scipy` | n200000 op=y1 | scipy | unpinned | **1.418** | [1.101, 1.826] null_envelope | 1.217 / 1.058 (spread) | 8.9 / 8.6 / 17.0 (raw la1 max 9.6) | - | 4/1 | `b8c631d60934` | `raw/special.unpinned.log:172` |
| UNRESOLVED (fsci A/A null 1.068 outside its spread band) | `perf_special_vs_scipy` | n200000 op=zeta | scipy | unpinned | **1.114** | [1.012, 1.227] null_envelope | 1.068 / 1.031 (spread) | 8.9 / 3.2 / 7.0 (raw la1 max 9.2) | - | 7/1 | `b8c631d60934` | `raw/special.unpinned.log:91` |

### stats

| class | harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WIN | `perf_stats_vs_scipy` | n200000t8 op=kendalltau | scipy | pinned:5 | **1.145** | [1.110, 1.181] null_envelope | 1.009 / 1.022 (spread) | 9.5 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:12` |
| WIN | `perf_stats_vs_scipy` | n200000t8 op=ks_2samp | scipy | pinned:5 | **3.450** | [3.389, 3.512] null_envelope | 1.010 / 1.008 (spread) | 9.5 / 8.2 / 6.0 (raw la1 max 9.2) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:15` |
| WIN | `perf_stats_vs_scipy` | n200000t8 op=rankdata | scipy | pinned:5 | **1.480** | [1.400, 1.564] null_envelope | 1.026 / 1.030 (spread) | 9.5 / 8.5 / 6.0 (raw la1 max 9.5) | 0.00 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:6` |
| WIN | `perf_stats_vs_scipy` | n200000t8 op=spearmanr | scipy | pinned:5 | **2.709** | [2.599, 2.824] null_envelope | 1.033 / 1.009 (spread) | 9.5 / 8.5 / 7.5 (raw la1 max 9.5) | 0.01 | 1/1 | `dedffde5209f` | `raw/stats.pinned.log:9` |
| UNRESOLVED (fsci A/A null 1.144 outside its spread band; scipy A/A null 1.094 outside its spread band) | `perf_stats_vs_scipy` | n200000t8 op=kendalltau | scipy | unpinned | **1.191** | [0.952, 1.491] null_envelope | 1.144 / 1.094 (spread) | 8.9 / 7.7 / 3.5 (raw la1 max 8.7) | - | 1/1 | `dedffde5209f` | `raw/stats.unpinned.log:12` |
| UNRESOLVED (fsci A/A null 1.071 outside its spread band) | `perf_stats_vs_scipy` | n200000t8 op=ks_2samp | scipy | unpinned | **3.667** | [3.283, 4.096] null_envelope | 1.071 / 1.043 (spread) | 8.9 / 6.7 / 3.0 (raw la1 max 8.7) | - | 1/1 | `dedffde5209f` | `raw/stats.unpinned.log:15` |
| UNRESOLVED (fsci A/A null 1.1 outside its spread band) | `perf_stats_vs_scipy` | n200000t8 op=rankdata | scipy | unpinned | **1.615** | [1.413, 1.846] null_envelope | 1.100 / 1.039 (spread) | 8.9 / 7.9 / 5.0 (raw la1 max 8.9) | - | 1/1 | `dedffde5209f` | `raw/stats.unpinned.log:6` |
| UNRESOLVED (fsci A/A null 1.087 outside its spread band) | `perf_stats_vs_scipy` | n200000t8 op=spearmanr | scipy | unpinned | **4.138** | [3.675, 4.660] null_envelope | 1.087 / 1.036 (spread) | 8.9 / 7.9 / 6.0 (raw la1 max 8.9) | - | 1/1 | `dedffde5209f` | `raw/stats.unpinned.log:9` |

## Provenance

- `host`: thinkstation1
- `cpu_model`: AMD Ryzen Threadripper PRO 5975WX 32-Cores
- `logical_cpus`: 64
- `kernel`: 7.0.0-30-generic
- `governor`: powersavex64
- `scaling_driver`: amd-pstate-epp
- `epp`: balance_performance
- `isa`: sse4_2+avx+avx2+fma
- `build_target_features`: release profile (lto=true, codegen-units=1, opt-level=3) with .cargo/config.toml [build] rustflags `-C target-feature=+avx2,+fma`; built on rch worker hz4 from the worktree at 54ee859e6 with `cargo build -j 2 --keep-going --release` and features fsci-integrate/bdf-diag-bench, fsci-linalg/eigh-incumbent-bench, fsci-sparse/sparse-incumbent-bench, fsci-sparse/live-scipy-bench (the fsci-sparse features gate only sub_coo/laplacian A/B toggles in the library, none on a path these harnesses time); every bin artifact reported fresh=false (compiled in that invocation)
- `build_worker`: hz4
- `git_head`: 54ee859e6c90c3ee8a0f909bfa6e61422a3c30da
- `incumbent_interpreter`: /home/ubuntu/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu/bin/python3.13
- `incumbent_interpreter_sha256`: 263dd6cc0c5b61880e54abfb2b1b4ea9aef7ae64e5f0e0b9f3eb15b8e8b9ae4e
- `incumbent`: {'python': '/home/ubuntu/.local/bin/python3.13', 'scipy': '1.17.1', 'numpy': '2.4.3', 'fsci_loaded': 'false', 'genuine': True, 'blas': 'Haswell'}

## Regenerating

```
# per harness, from the worktree root, binaries built remotely and brought back:
python3.13 scripts/perf_scoreboard.py run --out-dir <dir>/raw --tag <tag> --harness <bin> --binary <path> [--pin <cpu>] [--env K=V ...] -- <harness args>
python3.13 scripts/perf_scoreboard.py plant --artifact-dir <dir> --gmres-tag <tag> --bdf-tag <tag> --splu-tag <tag>
python3.13 scripts/perf_scoreboard.py build --artifact-dir <dir> --out-json <dir>/scoreboard.json --out-md <dir>/SCOREBOARD.md --build-manifest <dir>/build/manifest_b1.txt
python3.13 scripts/perf_scoreboard.py --self-test
```

Per row, the JSON carries: raw log path and line, both ELF identities (self-reported and executed), the incumbent interpreter sha256 and SciPy engine sha256 where the harness prints one, the interval kind, both null values and their kind, the row's time window, its loadavg trace summary, foreign runnable median, pinned-CPU and SMT-sibling busy, peak observed thread counts of both arms, the interleaving schedule, and the classification reasons.

