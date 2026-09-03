//! frankenscipy-x3kr0: do `symmetric_eigen` and `SVD::new` actually fail to converge, or
//! are they merely unbounded?
//!
//! nalgebra 0.33.3 builds both through `try_new(..., max_niter = 0)`, and the guard is
//! `if niter == max_niter { return None }`. With `max_niter == 0` the counter goes
//! 1, 2, 3, ... and never equals 0, so it can never fire. That is the same defect sez4r
//! documented for `Schur::new`, in the same dependency, and fsci reaches it from `eigh`,
//! `pinv`, the gram-matrix paths and the whole `svd` family -- a wider public surface than
//! `eig`.
//!
//! THAT IS ONE ARM. The source read establishes the loop CAN run forever; it does not
//! establish that any input makes it. Symmetric tridiagonal QL/QR with a Wilkinson shift is
//! globally convergent for symmetric input (Golub & Van Loan), so the must-hit may
//! genuinely not exist here even though it did for the unsymmetric Francis QR. Bounding on
//! the source read alone would be one-arm evidence and unjustified churn.
//!
//! ## The detector is a COUNT, not a timeout
//!
//! The obvious probe -- run the unbounded constructor under a wall-clock timeout -- is a bad
//! probe on a shared box. A case that trips a 5-second timeout under run queue 72 is
//! indistinguishable from a case that hangs, and the host has been at runq 72 today. So
//! this instead calls `try_new(m, f64::EPSILON, K)` and reads `None` as "needs more than K
//! iterations". That is a property of the matrix and the algorithm, identical on an idle and
//! a saturated host, and it also cannot hang: every call here is bounded by construction.
//!
//! Two bounds are used to separate two different findings:
//!   * `K_LAPACK = 30 * n.max(10)` -- the LAPACK-order bound sez4r adopted. A `None` here
//!     means bounding at this value WOULD change behaviour on this input.
//!   * `K_HUGE = 100 * K_LAPACK` -- a `None` here too means the iteration is not merely slow
//!     but is not converging at all, which is the real must-hit.
//!
//! A case that fails the first and passes the second is not a hang; it is evidence that
//! sez4r's constant is too tight for this iteration, which is a different bead.
//!
//! ## The must-miss arm
//!
//! Counting non-convergence proves nothing on its own unless the bounded call is also shown
//! to agree with the unbounded one where both finish. For every case that converges within
//! `K_LAPACK`, this compares the bounded result against `new()` BIT-FOR-BIT. Calling `new()`
//! is safe there and only there: the case has already been proven to converge in at most
//! `K_LAPACK` iterations, so the unbounded call cannot hang on it.
//!
//! Emits counts only. No timings, so nothing here needs the fleet build slot and nothing
//! here is invalidated by load.

use nalgebra::{DMatrix, SVD, SymmetricEigen};

const MARKER: &str = "unbounded-sym-svd-v1";
/// `SVD::new` passes nalgebra's default epsilon multiplied by five through
/// `new_unordered`; bounded comparisons must use the identical threshold.
const SVD_NEW_EPSILON: f64 = f64::EPSILON * 5.0;

/// Deterministic PRNG so every reported `(family, n, seed)` is reproducible without
/// carrying a fixture file around.
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Symmetrised form of sez4r's generator -- the family that exposed 230 non-convergent
/// cases for the unsymmetric path. Included so that a null result here is a statement about
/// the SYMMETRIC iteration rather than about the fixtures.
fn diag_dominant_sym(n: usize, seed: u64) -> DMatrix<f64> {
    let mut a = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let r =
                ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000) as f64 / 1000.0;
            let v = if i == j {
                (n as f64) * 2.0 + r
            } else {
                r - 0.5
            };
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    a
}

/// Graded spectrum spanning many orders of magnitude: `Q diag(10^-k) Qᵀ`. The classic hard
/// case for a shifted QR -- tiny eigenvalues sit below the deflation threshold of the large
/// ones, so the tail of the sweep is where a shift strategy earns its keep.
fn ill_conditioned(n: usize, seed: u64) -> DMatrix<f64> {
    let mut next = rng(seed);
    let q = orthogonal(n, &mut next);
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        let exponent = -16.0 * (i as f64) / ((n.max(2) - 1) as f64);
        d[(i, i)] = 10.0_f64.powf(exponent);
    }
    &q * d * q.transpose()
}

/// Tightly clustered eigenvalues -- repeated to within a few ulps. Clusters are what stall a
/// Wilkinson shift, because the shift cannot separate eigenvalues it cannot tell apart.
fn clustered(n: usize, seed: u64) -> DMatrix<f64> {
    let mut next = rng(seed);
    let q = orthogonal(n, &mut next);
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        let cluster = (i / 3) as f64;
        d[(i, i)] = 1.0 + cluster + (i % 3) as f64 * 1e-15;
    }
    &q * d * q.transpose()
}

/// Exactly-degenerate spectrum with a handful of distinct values, so most subdiagonals are
/// deflatable immediately and the iteration spends its time on the few that are not.
fn degenerate(n: usize, seed: u64) -> DMatrix<f64> {
    let mut next = rng(seed);
    let q = orthogonal(n, &mut next);
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = ((i % 4) + 1) as f64;
    }
    &q * d * q.transpose()
}

/// Gram-Schmidt on a random matrix. Good enough to be orthogonal to roundoff, which is all
/// these fixtures need -- the point is a realistic eigenvector basis, not an exact one.
fn orthogonal(n: usize, next: &mut impl FnMut() -> f64) -> DMatrix<f64> {
    let mut q: DMatrix<f64> = DMatrix::from_fn(n, n, |_, _| next());
    for j in 0..n {
        for k in 0..j {
            let dot: f64 = (0..n).map(|i| q[(i, j)] * q[(i, k)]).sum();
            for i in 0..n {
                q[(i, j)] -= dot * q[(i, k)];
            }
        }
        let norm: f64 = (0..n).map(|i| q[(i, j)] * q[(i, j)]).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..n {
                q[(i, j)] /= norm;
            }
        } else {
            // Degenerate draw: fall back to a unit vector so the family still produces a
            // usable orthogonal factor instead of silently returning a rank-deficient one.
            for i in 0..n {
                q[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    q
}

/// Rectangular fixture for the SVD arm, including rank-deficient columns.
fn rect(rows: usize, cols: usize, seed: u64, rank_deficient: bool) -> DMatrix<f64> {
    let mut next = rng(seed);
    let mut a: DMatrix<f64> = DMatrix::from_fn(rows, cols, |_, _| next());
    if rank_deficient && cols >= 2 {
        // Duplicate a column so the matrix has an exactly zero singular value -- the
        // condition that makes a bidiagonal sweep work hardest.
        for i in 0..rows {
            let v = a[(i, 0)];
            a[(i, cols - 1)] = v;
        }
    }
    a
}

struct Tally {
    cases: usize,
    over_lapack: Vec<String>,
    over_huge: Vec<String>,
    bit_mismatch: Vec<String>,
    compared: usize,
    /// Of the mismatches, how many are the SAME multiset of values in a different order.
    /// Separated out because "different order" and "different numbers" are different
    /// findings and only one of them is about convergence.
    order_only: usize,
}

impl Tally {
    fn new() -> Self {
        Self {
            cases: 0,
            over_lapack: Vec::new(),
            over_huge: Vec::new(),
            bit_mismatch: Vec::new(),
            compared: 0,
            order_only: 0,
        }
    }
}

fn probe_symmetric(tally: &mut Tally, label: &str, m: DMatrix<f64>, n: usize) {
    tally.cases += 1;
    let k_lapack = 30 * n.max(10);
    let k_huge = 100 * k_lapack;

    let bounded = SymmetricEigen::try_new(m.clone(), f64::EPSILON, k_lapack);
    if bounded.is_none() {
        tally.over_lapack.push(label.to_string());
        if SymmetricEigen::try_new(m.clone(), f64::EPSILON, k_huge).is_none() {
            tally.over_huge.push(label.to_string());
        }
        // Do NOT call `new()` here. This case has just been shown to need more than
        // k_huge iterations, which is exactly the input on which the unbounded
        // constructor would hang -- calling it is the bug, not the test.
        return;
    }

    // MUST-MISS arm. Safe precisely because the bounded call above converged: the
    // unbounded one cannot iterate longer than a bound it already finished inside.
    let bounded = bounded.expect("checked above");
    let unbounded = m.symmetric_eigen();
    tally.compared += 1;
    let same = bounded.eigenvalues.len() == unbounded.eigenvalues.len()
        && bounded
            .eigenvalues
            .iter()
            .zip(unbounded.eigenvalues.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !same {
        tally.bit_mismatch.push(label.to_string());
    }
}

fn probe_svd(tally: &mut Tally, label: &str, m: DMatrix<f64>) {
    tally.cases += 1;
    let n = m.nrows().max(m.ncols());
    let k_lapack = 30 * n.max(10);
    let k_huge = 100 * k_lapack;

    // EPS MUST MATCH THE ARM BEING REPLACED. `SVD::new` routes through `new_unordered`,
    // which passes `default_epsilon() * 5.0` -- NOT `default_epsilon()`. `SymmetricEigen::new`
    // and `Schur::new` pass it unscaled. Getting this wrong is not a rounding detail: the
    // first run of this probe used a bare `f64::EPSILON` here and reported 805 of 2400 SVD
    // cases as differing from the unbounded call. Every one was a deflation-threshold
    // difference of about one ulp (worst 7.2e-16), caused by the probe, not by nalgebra --
    // and the symmetric arm, where the eps DID match, showed 0 of 5600. That contrast is
    // what identified it.
    //
    // The finding matters beyond this probe: sez4r's bounding recipe
    // `try_new(m, f64::EPSILON, 30 * n.max(10))` is correct for Schur and symmetric_eigen
    // and WRONG for SVD. Anyone bounding an SVD site with it silently shifts results by an
    // ulp while believing the change is behaviour-preserving.
    let bounded = SVD::try_new(m.clone(), true, true, SVD_NEW_EPSILON, k_lapack);
    if bounded.is_none() {
        tally.over_lapack.push(label.to_string());
        if SVD::try_new(m.clone(), true, true, SVD_NEW_EPSILON, k_huge).is_none() {
            tally.over_huge.push(label.to_string());
        }
        return;
    }

    let bounded = bounded.expect("checked above");
    let unbounded = SVD::new(m, true, true);
    tally.compared += 1;
    let same = bounded.singular_values.len() == unbounded.singular_values.len()
        && bounded
            .singular_values
            .iter()
            .zip(unbounded.singular_values.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !same {
        // A raw index-by-index bit diff says the two disagree without saying HOW, and on
        // this project that shape of report has produced a phantom defect before: pairing
        // by position across two spectra is only meaningful when position means the same
        // thing in both. So quantify it three ways before believing anything.
        //
        //   * as MULTISETS (sorted by bits): if these agree, the values are identical and
        //     only the ORDER differs -- a tie-breaking difference, not a numerical one.
        //   * the worst RELATIVE gap position-by-position: sizes a real disagreement.
        //   * the count of positions differing at all.
        let mut a: Vec<u64> = bounded
            .singular_values
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut b: Vec<u64> = unbounded
            .singular_values
            .iter()
            .map(|v| v.to_bits())
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        let multiset_equal = a == b;
        let mut worst_rel = 0.0_f64;
        let mut positions = 0usize;
        for (x, y) in bounded
            .singular_values
            .iter()
            .zip(unbounded.singular_values.iter())
        {
            if x.to_bits() != y.to_bits() {
                positions += 1;
                let scale = x.abs().max(y.abs()).max(f64::MIN_POSITIVE);
                worst_rel = worst_rel.max((x - y).abs() / scale);
            }
        }
        tally.bit_mismatch.push(format!(
            "{label} multiset_equal={multiset_equal} positions={positions} worst_rel={worst_rel:.3e}"
        ));
        if multiset_equal {
            tally.order_only += 1;
        }
    }
}

fn report(name: &str, t: &Tally) {
    println!("--- {name} ---");
    println!("cases                          {}", t.cases);
    println!("bit-compared against unbounded {}", t.compared);
    println!("needed > 30*n iterations       {}", t.over_lapack.len());
    println!("needed > 3000*n iterations     {}", t.over_huge.len());
    println!("bit MISMATCH vs unbounded      {}", t.bit_mismatch.len());
    println!("  of those, ORDER ONLY         {}", t.order_only);
    println!(
        "  of those, VALUES differ      {}",
        t.bit_mismatch.len() - t.order_only
    );
    for label in t.over_lapack.iter().take(20) {
        println!("OVER_LAPACK {name} {label}");
    }
    for label in t.over_huge.iter().take(20) {
        println!("OVER_HUGE {name} {label}");
    }
    for label in t.bit_mismatch.iter().take(20) {
        println!("MISMATCH {name} {label}");
    }
}

fn main() {
    let seeds: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let nmax: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    println!("MARKER={MARKER} seeds={seeds} nmax={nmax}");
    emit_environment();

    // ---- MUST-HIT CONTROL, before any zero from this probe means anything -------------
    //
    // "0 non-convergent cases in 8000" is indistinguishable from "the detector never
    // fires" until the detector is shown firing. So squeeze the bound to a value the
    // iteration certainly exceeds and require `None` from BOTH constructors. If either of
    // these comes back `Some`, every zero reported below is uninterpretable and the run is
    // void rather than negative.
    let control_m = ill_conditioned(16, 7);
    let sym_fires = SymmetricEigen::try_new(control_m.clone(), f64::EPSILON, 1).is_none();
    let svd_fires = SVD::try_new(rect(16, 16, 7, false), true, true, SVD_NEW_EPSILON, 1).is_none();
    // And the must-MISS half of the same control: at a generous bound the SAME inputs must
    // come back `Some`, or the detector is simply always-None and equally blind.
    let sym_misses = SymmetricEigen::try_new(control_m, f64::EPSILON, 100_000).is_some();
    let svd_misses =
        SVD::try_new(rect(16, 16, 7, false), true, true, SVD_NEW_EPSILON, 100_000).is_some();
    println!(
        "CONTROL sym_fires_at_1={sym_fires} svd_fires_at_1={svd_fires} \
sym_converges_at_100k={sym_misses} svd_converges_at_100k={svd_misses}"
    );
    if !(sym_fires && svd_fires && sym_misses && svd_misses) {
        println!("CONTROL FAILED -- the sweep below is VOID, not negative");
        std::process::exit(2);
    }

    let mut sym = Tally::new();
    for n in [2usize, 3, 5, 8, 13, 21, nmax] {
        for seed in 0..seeds {
            probe_symmetric(
                &mut sym,
                &format!("diag_dominant n={n} seed={seed}"),
                diag_dominant_sym(n, seed),
                n,
            );
            probe_symmetric(
                &mut sym,
                &format!("ill_conditioned n={n} seed={seed}"),
                ill_conditioned(n, seed),
                n,
            );
            probe_symmetric(
                &mut sym,
                &format!("clustered n={n} seed={seed}"),
                clustered(n, seed),
                n,
            );
            probe_symmetric(
                &mut sym,
                &format!("degenerate n={n} seed={seed}"),
                degenerate(n, seed),
                n,
            );
        }
    }

    let mut svd = Tally::new();
    for (rows, cols) in [(4usize, 4usize), (8, 3), (3, 8), (16, 16), (32, 7), (7, 32)] {
        for seed in 0..seeds {
            svd_case(&mut svd, rows, cols, seed, false);
            svd_case(&mut svd, rows, cols, seed, true);
        }
    }

    report("symmetric_eigen", &sym);
    report("svd", &svd);

    // The verdict line, stated in the form the bead asks for: either a named must-hit or a
    // reportable negative WITH its sweep size. A count with no sweep size attached is not a
    // negative result, it is an unfinished one.
    let must_hit = sym.over_huge.len() + svd.over_huge.len();
    println!(
        "VERDICT must_hit={must_hit} sym_cases={} svd_cases={} sym_compared={} svd_compared={}",
        sym.cases, svd.cases, sym.compared, svd.compared
    );
}

/// Print host, load, clock and iowait from the process that produces the numbers.
///
/// This exists because the alternative -- wrapping the run in a shell that echoes
/// `uptime` -- is not available here: `rch exec` refuses non-compilation commands, so a
/// remote run cannot be bracketed from outside. Reading it from inside is better anyway.
/// Provenance collected in a DIFFERENT invocation than the measurement is provenance for a
/// different invocation, and on a shared box that distinction has teeth.
///
/// This row's numbers are counts and are load-independent by construction, so these values
/// are recorded for the ledger's environment gate rather than to qualify the result.
fn emit_environment() {
    let read = |path: &str| std::fs::read_to_string(path).unwrap_or_default();
    let host = read("/proc/sys/kernel/hostname").trim().to_string();
    let loadavg = read("/proc/loadavg").trim().to_string();
    let mhz = read("/proc/cpuinfo")
        .lines()
        .find(|l| l.starts_with("cpu MHz"))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    // Cumulative jiffies since boot: field 5 of the aggregate `cpu` line is iowait. A
    // cumulative figure is the honest one to report for a short run -- a sampled
    // instantaneous rate would describe a moment rather than the run.
    let stat = read("/proc/stat");
    let cpu_line = stat.lines().next().unwrap_or("").to_string();
    let fields: Vec<&str> = cpu_line.split_whitespace().collect();
    let (idle, iowait, total) = if fields.len() > 5 {
        let nums: Vec<u64> = fields[1..].iter().filter_map(|f| f.parse().ok()).collect();
        let total: u64 = nums.iter().sum();
        (
            nums.get(3).copied().unwrap_or(0),
            nums.get(4).copied().unwrap_or(0),
            total,
        )
    } else {
        (0, 0, 0)
    };
    let pct = |v: u64| {
        if total == 0 {
            0.0
        } else {
            100.0 * v as f64 / total as f64
        }
    };
    println!(
        "ENV host={host} loadavg=[{loadavg}] cpu_mhz={mhz} \
idle_pct_since_boot={:.1} iowait_pct_since_boot={:.2} logical_cpus={}",
        pct(idle),
        pct(iowait),
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(0)
    );
}

fn svd_case(tally: &mut Tally, rows: usize, cols: usize, seed: u64, deficient: bool) {
    let label = format!("rect {rows}x{cols} seed={seed} deficient={deficient}");
    probe_svd(tally, &label, rect(rows, cols, seed, deficient));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bounded_svd_matches_new(matrix: DMatrix<f64>) {
        let max_niter = 30 * matrix.nrows().max(matrix.ncols()).max(10);
        let bounded = SVD::try_new(matrix.clone(), true, true, SVD_NEW_EPSILON, max_niter)
            .expect("fixture must converge within the LAPACK-order bound");
        let unbounded = SVD::new(matrix, true, true);

        let bounded_bits: Vec<u64> = bounded
            .singular_values
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let unbounded_bits: Vec<u64> = unbounded
            .singular_values
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(bounded_bits, unbounded_bits);
    }

    #[test]
    fn bounded_svd_uses_the_same_epsilon_as_svd_new() {
        for matrix in [
            rect(4, 4, 7, false),
            rect(8, 3, 11, false),
            rect(3, 8, 13, false),
            rect(16, 16, 17, true),
        ] {
            assert_bounded_svd_matches_new(matrix);
        }
    }
}
