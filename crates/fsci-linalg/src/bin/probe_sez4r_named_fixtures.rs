//! frankenscipy-sez4r's own closing criterion, run rather than asserted.
//!
//! The bead's last open comment names nine fixtures and says they settle the matter:
//!
//!   MUST CONVERGE (they hung, or failed to converge, before the fix):
//!       (5,201) (5,213) (5,234) (6,319) (6,335)
//!   MUST STILL CONVERGE TO BIT-IDENTICAL VALUES (the regression control):
//!       (5,0) (6,0) (8,999) (8,42)
//!
//! Both arms are required and neither alone is worth anything. If only the first set is
//! checked, a change that "fixes" them by perturbing every result would pass. If only the
//! second, a no-op would pass. Together they say: the previously-failing cases now succeed
//! AND the previously-succeeding cases are untouched to the bit.
//!
//! The bit-identity arm is measured against the SAME binary with `EIG_FRANCIS_FALLBACK`
//! disabled, not against a remembered value — the fallback may only change behaviour on
//! inputs that previously returned `ConvergenceFailure`, and that is exactly what this
//! checks. Emits counts and bit comparisons only; no timing, so load cannot affect it.

use fsci_linalg::{DecompOptions, EIG_FRANCIS_FALLBACK, eig};
use std::sync::atomic::Ordering;

const MARKER: &str = "sez4r-named-fixtures-v1";

/// Copied VERBATIM from the conformance metamorphic suite, as `perf_francis_vs_nalgebra`
/// does, so the `(n, seed)` pairs in the bead address the same matrices they always did.
fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let r =
                ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000) as f64 / 1000.0;
            a[i][j] = if i == j { (n as f64) * 2.0 + r } else { r - 0.5 };
        }
    }
    a
}

fn spectrum_bits(a: &[Vec<f64>]) -> Option<Vec<(u64, u64)>> {
    eig(a, DecompOptions::default()).ok().map(|r| {
        r.eigenvalues_re
            .iter()
            .zip(r.eigenvalues_im.iter())
            .map(|(&re, &im)| (re.to_bits(), im.to_bits()))
            .collect()
    })
}

fn emit_environment() {
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
    println!(
        "ENV host={} loadavg=[{}] cpu_mhz={}",
        read("/proc/sys/kernel/hostname").trim(),
        read("/proc/loadavg").trim(),
        read("/proc/cpuinfo")
            .lines()
            .find(|l| l.starts_with("cpu MHz"))
            .and_then(|l| l.split(':').nth(1))
            .unwrap_or(" unknown")
            .trim()
    );
}

fn main() {
    println!("MARKER={MARKER}");
    emit_environment();

    const MUST_CONVERGE: [(usize, u64); 5] = [(5, 201), (5, 213), (5, 234), (6, 319), (6, 335)];
    const MUST_BE_UNCHANGED: [(usize, u64); 4] = [(5, 0), (6, 0), (8, 999), (8, 42)];

    // ---- arm 1: the previously-failing fixtures must now converge --------------------
    //
    // Also recorded: whether each one still FAILS with the fallback disabled. That is the
    // must-hit control for this arm — if a fixture converges with the fallback off, it was
    // not one of the failing cases any more and this arm is testing nothing on it. A
    // silently-stale fixture set is the failure mode that makes a green run meaningless.
    let mut converged_with = 0usize;
    let mut still_fails_without = 0usize;
    for (n, seed) in MUST_CONVERGE {
        let a = make_diag_dominant(n, seed);

        EIG_FRANCIS_FALLBACK.store(false, Ordering::Relaxed);
        let without = spectrum_bits(&a).is_some();

        EIG_FRANCIS_FALLBACK.store(true, Ordering::Relaxed);
        let with = spectrum_bits(&a).is_some();

        if with {
            converged_with += 1;
        }
        if !without {
            still_fails_without += 1;
        }
        println!("MUSTCONVERGE n={n} seed={seed} with_fallback={with} without_fallback={without}");
    }

    // ---- arm 2: the passing fixtures must be untouched, bit for bit ------------------
    let mut identical = 0usize;
    for (n, seed) in MUST_BE_UNCHANGED {
        let a = make_diag_dominant(n, seed);

        EIG_FRANCIS_FALLBACK.store(false, Ordering::Relaxed);
        let baseline = spectrum_bits(&a);

        EIG_FRANCIS_FALLBACK.store(true, Ordering::Relaxed);
        let shipped = spectrum_bits(&a);

        let same = baseline.is_some() && baseline == shipped;
        if same {
            identical += 1;
        }
        println!(
            "MUSTBEUNCHANGED n={n} seed={seed} baseline_converged={} bit_identical={same}",
            baseline.is_some()
        );
    }

    EIG_FRANCIS_FALLBACK.store(true, Ordering::Relaxed);

    let pass = converged_with == MUST_CONVERGE.len()
        && identical == MUST_BE_UNCHANGED.len()
        && still_fails_without == MUST_CONVERGE.len();
    println!(
        "VERDICT converged_with_fallback={converged_with}/{} still_fail_without_fallback={still_fails_without}/{} bit_identical={identical}/{} pass={pass}",
        MUST_CONVERGE.len(),
        MUST_CONVERGE.len(),
        MUST_BE_UNCHANGED.len()
    );
    if !pass {
        std::process::exit(1);
    }
}
