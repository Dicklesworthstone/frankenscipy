# Gemini Code Review Final Report

*Note: MCP Agent Mail SQLite database is currently corrupted (`storage.sqlite3`). Because `am doctor repair --yes` receives SIGHUP in the local background execution environment and `send_message` errors out, I am dropping my review report into this tracking file as an alternative channel for the Codex implementer.*

## File: `crates/fsci-special/src/convenience.rs`

### 1. `gammaincinv` derivative computation intermediate overflow
- **Severity**: Critical
- **Root Cause**: The Newton-Raphson step computes `dpx = x.powf(a - 1.0) * (-x).exp() / ln_gamma_a.exp();`. For moderate-to-large values of `a` (e.g., `a > 171.0`), `ln_gamma_a.exp()` overflows to `f64::INFINITY`, driving `dpx` to `0.0`. This causes the Newton step to fail entirely, silently degrading to the bisection fallback which destroys performance and accuracy.
- **Suggested Fix**: Use numerically stable exponentiation, just as `betaincinv` already correctly does.
  Replace:
  ```rust
  let dpx = x.powf(a - 1.0) * (-x).exp() / ln_gamma_a.exp();
  ```
  With:
  ```rust
  let dpx = ((a - 1.0) * x.ln() - x - ln_gamma_a).exp();
  ```

### 2. `expi_scalar` missing asymptotic expansion for large `x`
- **Severity**: Important
- **Root Cause**: For `x > 0`, the implementation relies solely on the power series `x^k / (k * k!)`. At large `x` (e.g. `x > 25.0`), intermediate terms massively overflow `f64::MAX`, causing the function to return `INFINITY` or `NaN`, and destroying precision well before that due to catastrophic cancellation.
- **Suggested Fix**: Implement an asymptotic expansion for large positive `x` (e.g., `x > 25.0` or similar cutoff). The expansion is `Ei(x) ~ e^x / x * (1 + 1!/x + 2!/x^2 + 3!/x^3 + ...)`.

### 3. `ker` and `kei` diverge/NaN for large `x`
- **Severity**: Important
- **Root Cause**: The Kelvin functions `ker` and `kei` are implemented using their Maclaurin series expansions. This diverges wildly or hits `f64` overflow for `x > 10.0`. The codebase lacks the corresponding asymptotic expansions for large arguments.
- **Suggested Fix**: Add the standard asymptotic expansions for `ker(x)` and `kei(x)` for `x > 10.0` (typically involving `exp(-x / sqrt(2))` and trigonometric terms).

### 4. `owens_t` inaccurate for large `a`
- **Severity**: Important
- **Root Cause**: `owens_t` uses a fixed 10-point Gauss-Legendre quadrature on the interval `[0, a]`. If `a` is large, the peak of the integrand `1/(1+t^2)` is near `0` and is extremely narrow relative to the integration domain, rendering the 10-point fixed grid highly inaccurate.
- **Suggested Fix**: Use an adaptive quadrature, or split the domain, or apply the standard Owen's T rational approximations/asymptotics for large `a`. A quick patch could be a variable `n` or bounding the integration where the integrand effectively vanishes.

### 5. `log_ndtr` catastrophic cancellation for large `x > 6.0`
- **Severity**: Nit / Parity Gap
- **Root Cause**: For `x > 6.0`, the code computes `ndtr(x)` which effectively evaluates to `1.0` in `f64`, and then `1.0f64.ln()` yields exactly `0.0`. SciPy computes this tail to high relative precision using `log(1 - ndtr(-x)) ≈ -ndtr(-x)` or an asymptotic expansion for the log complement.
- **Suggested Fix**:
  Replace:
  ```rust
  if x > 6.0 {
      let t = ndtr(x);
      if t > 0.0 { t.ln() } else { 0.0 }
  }
  ```
  With an asymptotic approximation for the upper tail:
  ```rust
  if x > 6.0 {
      // log(1 - erfc(x/sqrt(2))/2) ≈ -erfc(x/sqrt(2))/2 ≈ -ndtr(-x)
      -ndtr(-x) 
  }
  ```