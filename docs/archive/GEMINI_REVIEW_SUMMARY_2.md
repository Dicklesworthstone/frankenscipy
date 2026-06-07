# Gemini Code Review Spontaneous Report 2

Since the MCP Agent Mail database is still failing with SIGHUP when the background repair script runs, I am continuing the fallback protocol and writing this review report to the file system for my Codex (`cod`) partner to pick up. 

I've conducted a fresh-eyes review of the `fsci-integrate` and `fsci-opt` crates, prioritizing mathematical kernels, error handling, and parity gaps vs SciPy.

## 1. `crates/fsci-integrate/src/rk.rs`

### Bug 1.1: Invalid FSAL Assumption for DOP853 (Critical Integration Failure)
- **Severity**: Critical
- **Location**: `crates/fsci-integrate/src/rk.rs`, in `rk_step` and `RkSolver::step_with`.
- **Root Cause**: The RK solver assumes that *all* provided Butcher tableaux are FSAL (First Same As Last), meaning the last stage of the current step evaluates the derivative at the new state `y_new`. It implements this by hardcoding `let f_new = k[tableau.n_stages - 1].clone();` and saving this derivative for the next step. While this holds true for RK45 and RK23, it is **fundamentally false for DOP853**. In DOP853, the last stage (`k[11]`) corresponds to an intermediate time step (`C[11] = 1.0`), but the `B` coefficients used to construct `y_new` do not match the `A11` coefficients used to construct the evaluation point for `k[11]`. Consequently, `f_new` is populated with the derivative at the wrong spatial point, propagating severe errors into the next integration step and completely breaking DOP853.
- **Suggested Fix**: Add an `is_fsal` boolean flag to `ButcherTableau`. If `is_fsal` is true, preserve the current logic. If false (like DOP853), explicitly evaluate `fun(t + h, y_new)` to compute the correct `f_new` for the next step, and ensure `self.nfev` is incremented by `n_stages` instead of `n_stages - 1`.

## 2. `crates/fsci-opt/src/minimize.rs`

### Bug 2.1: Missing Line Search in L-BFGS-B Steepest Descent Fallback
- **Severity**: Important
- **Location**: `crates/fsci-opt/src/minimize.rs`, inside `lbfgsb`.
- **Root Cause**: If the L-BFGS two-loop recursion produces a direction that is not a descent direction (`directional_deriv >= 0.0`), the algorithm correctly resets the L-BFGS history and falls back to steepest descent (`direction = -grad`). However, it sets `alpha = 1.0 / l2_norm(&direction).max(1.0)`, computes a new candidate point, and then unconditionally accepts this point (`x = projected_candidate`) and `continue`s to the next outer iteration **without performing an Armijo line search**. This can lead to the optimizer taking an arbitrarily large step that increases the objective function or pushes the state into a non-convergent regime.
- **Suggested Fix**: Remove the unconditional acceptance of the steepest descent step. Instead, simply update `direction` to `-grad`, reset the history, set `alpha = 1.0 / l2_norm(&direction).max(1.0)`, and allow the execution to fall through to the standard Armijo backtracking line search loop that immediately follows the `if` block.