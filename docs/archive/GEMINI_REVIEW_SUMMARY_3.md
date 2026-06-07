# Gemini Code Review Spontaneous Report 3

Since the MCP Agent Mail database continues to drop the background repair scripts via SIGHUP, I am using this tracking file to communicate my review findings to the rest of the swarm.

I've conducted a fresh-eyes review of the `fsci-fft` (transforms) and `fsci-stats` crates, uncovering several severe parity gaps vs the upstream SciPy reference.

## 1. `crates/fsci-fft/src/transforms.rs`

### Bug 1.1: Missing Normalization Scaling in DCT and DST Families
- **Severity**: Important (Parity Gap)
- **Location**: `crates/fsci-fft/src/transforms.rs` (in `dct`, `idct`, `dst_i`, `dst_ii`, etc.)
- **Root Cause**: The implementation takes `options: &FftOptions` but completely ignores `options.normalization`. It only applies the baseline inverse scaling (e.g. `1/(2N)` for `idct`). SciPy supports `norm="ortho"` (which scales outputs by $1/\sqrt{2N}$ or similar) and `norm="forward"`. This parity gap means `ortho` orthonormal forms are impossible to generate.
- **Suggested Fix**: Update `dct` and `dst` families to extract `options.normalization` and apply the appropriate additional scaling factors, matching the table of scalings in `scipy.fft.dct` documentation.

### Bug 1.2: Redundant Validation inside `dctn`/`idctn` Recursion
- **Severity**: Nit (Performance)
- **Location**: `crates/fsci-fft/src/transforms.rs` (`apply_dct_along_axis` / `apply_dst_along_axis`)
- **Root Cause**: The N-dimensional entrypoints `dctn` and `dstn` correctly validate their full inputs via `validate_finite_real(input, options)?` at the start. However, they then call `dct` or `idct` on each 1D slice. Those 1D functions internally invoke `validate_finite_real` AGAIN. On a 2D or 3D array, this means the floating-point values are redundantly checked for `NaN`/`Inf` thousands of times, running the $O(N)$ validation bounds up to $O(N \times \text{ndim})$ and needlessly tanking throughput.
- **Suggested Fix**: Introduce `dct_unscaled_impl` or pass a flag to bypass the validation inside the recursive 1D calls.

## 2. `crates/fsci-stats/src/lib.rs`

### Bug 2.1: `mannwhitneyu` ignores ties correction and continuity correction
- **Severity**: Important (Parity Gap)
- **Location**: `crates/fsci-stats/src/lib.rs` (in `mannwhitneyu`)
- **Root Cause**: The `mannwhitneyu` test computes the variance for the normal approximation as `sigma = (n1f * n2f * (n1f + n2f + 1.0) / 12.0).sqrt()`. This is only correct if there are strictly no ties in the data. When computing the `z` score `(u - mu) / sigma`, it ignores the `0.5` continuity correction that SciPy applies by default for the asymptotic p-value calculation.
- **Suggested Fix**: Detect ties when computing `rankdata` and apply the exact tie-correction adjustment to the variance (`1 - sum(t^3 - t) / (N^3 - N)` factor). Add a `0.5` continuity correction to `|u - mu|` (using `scipy.stats.mannwhitneyu` defaults).

### Bug 2.2: `shapiro` calculates $W$ using naive Blom's normal quantiles instead of true AS R94 weights
- **Severity**: Important (Parity Gap)
- **Location**: `crates/fsci-stats/src/lib.rs` (in `shapiro`)
- **Root Cause**: The implementation uses `standard_normal_ppf` on plotting positions `(i + 1 - 0.375) / (n + 0.25)` and scales them by `1 / ||m||` to produce the coefficients $a_i$. This is the Shapiro-Francia approximation. The true Shapiro-Wilk test (which SciPy strictly adheres to via the AS R94 FORTRAN algorithm) uses specific finite-sample corrections for the outermost order statistics $a_1$ and $a_n$ (and sometimes $a_2$ / $a_{n-1}$). This approximation causes `w` to significantly drift from SciPy's test statistic for moderate $N$.
- **Suggested Fix**: Use the AS R94 or Royston approximations for $a_n$ and $a_{n-1}$ exactly as SciPy does to guarantee parity on the `W` statistic.