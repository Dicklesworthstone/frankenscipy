# Latent Parity Audit Report (2026-04-19)

As no new commits have been introduced to `origin/main` in the recent window, I pivoted to reviewing the older `fsci-stats` algorithms for latent logic flaws and regressions.

## Audit Findings:

1. **Bug Found:** `ttest_ind` and `ttest_ind_welch` yield `NaN` when tested against identical/zero-variance input arrays. Unlike `ttest_rel` and `ttest_1samp`, these functions lack a check for a standard error (`se`) of exactly `0.0`.
   - Filed **frankenscipy-8l3l**.

2. **Bug Found:** `entropy` deviates significantly from SciPy on extreme edge cases. It currently returns `0.0` for entirely zero arrays (SciPy returns `NaN`) and clamps negative input vectors differently (SciPy returns `-inf`).
   - Filed **frankenscipy-r1zm**.

Both issues have been successfully tracked as `br` bug beads for immediate remediation. The repository continues to compile cleanly via `rch cargo clippy`.
