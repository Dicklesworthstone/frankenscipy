//! Is `gammaln`'s Lanczos→asymptotic crossover at x=100 justified, or merely conservative?
//!
//! WHY THIS EXISTS. `gammaln` is the worst tractable vs-SciPy cell in the suite (2.07x
//! slower), and its Lanczos kernel costs two `ln` calls and eight divisions per element
//! against the asymptotic kernel's one `ln` and no division chain. Lowering the crossover
//! would therefore be worth something — and would be an ILLEGITIMATE lever if justified by
//! the benchmark fixture happening to sit in the affected range.
//!
//! So this asks the accuracy question on its own terms, and asks it in a way that can
//! answer NO. For a grid of x spanning the whole disputed interval, it computes `gammaln`
//! twice — once with the crossover at its default 100, once forced low — and compares BOTH
//! against live SciPy 1.17.1. The grid is logarithmic over [8, 120] plus points placed hard
//! against each candidate crossover, and is deliberately NOT the benchmark fixture's
//! distribution.
//!
//! The verdict rule is stated up front so it cannot be adjusted after seeing the numbers:
//! the low crossover is justified only if its worst error over the disputed interval is no
//! worse than the default's. "About the same" is not good enough for a range where the
//! default kernel is the incumbent behaviour; ties go to the default.

use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;

use fsci_runtime::RuntimeMode;
use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use fsci_special::{GAMMALN_ASYMPTOTIC_MIN_OVERRIDE, SpecialTensor, gammaln};

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would
/// otherwise only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.special"];

/// The one live-SciPy incumbent this process compares against, resolved once and PROVEN
/// by running the import rather than by a path or a `PATH` name resolving.
///
/// This harness used to spawn a bare `python3`, which on `thinkstation1` is 3.14
/// with no SciPy at all, so the oracle died on its first write and the run read as a
/// flaky pipe rather than as a missing incumbent (frankenscipy-m5s54).
fn incumbent() -> &'static ScipyIncumbent {
    static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
    INCUMBENT.get_or_init(|| {
        let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
            .unwrap_or_else(|error| panic!("{error}"));
        println!("{}", resolved.provenance_line());
        resolved
    })
}

const PYTHON: &str = r#"
import sys
import numpy as np
import scipy
from scipy.special import gammaln

raw = sys.stdin.buffer.read()
xs = np.frombuffer(raw, dtype='<f8')
ref = np.ascontiguousarray(gammaln(xs), dtype='<f8')
sys.stdout.write(f"META {scipy.__version__} {np.__version__} {ref.size}\n")
sys.stdout.flush()
sys.stdout.buffer.write(ref.tobytes())
sys.stdout.buffer.flush()
"#;

fn ours(xs: &[f64], crossover: Option<f64>) -> Vec<f64> {
    GAMMALN_ASYMPTOTIC_MIN_OVERRIDE.store(
        crossover.map_or(0, f64::to_bits),
        std::sync::atomic::Ordering::Relaxed,
    );
    let tensor = SpecialTensor::RealVec(xs.to_vec());
    let out = gammaln(&tensor, RuntimeMode::Hardened).expect("fsci gammaln");
    GAMMALN_ASYMPTOTIC_MIN_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
    match out {
        SpecialTensor::RealVec(v) => v,
        other => panic!("gammaln returned {other:?}"),
    }
}

/// Worst absolute error and worst error in ULPs of the reference value.
fn worst(ours: &[f64], reference: &[f64]) -> (f64, f64, f64) {
    let mut max_abs = 0.0_f64;
    let mut max_ulp = 0.0_f64;
    let mut at = f64::NAN;
    for ((o, r), _) in ours.iter().zip(reference.iter()).zip(0..) {
        let d = (o - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        let ulp = if r.is_finite() && *r != 0.0 {
            d / (r.abs() * f64::EPSILON)
        } else {
            0.0
        };
        if ulp > max_ulp {
            max_ulp = ulp;
            at = *r;
        }
    }
    (max_abs, max_ulp, at)
}

fn main() {
    // Grid over the DISPUTED interval, chosen for the question and not for any benchmark:
    // dense near each candidate crossover (that is where an asymptotic series is weakest),
    // logarithmic across the rest, plus the exact boundary points.
    let mut xs: Vec<f64> = Vec::new();
    for i in 0..=1200 {
        let t = i as f64 / 1200.0;
        xs.push(8.0 * (120.0_f64 / 8.0).powf(t));
    }
    for anchor in [
        8.0, 12.0, 12.5, 13.0, 13.5, 14.0, 16.0, 20.0, 32.0, 64.0, 99.0, 100.0, 101.0,
    ] {
        for delta in [-1.0e-9, 0.0, 1.0e-9, 0.25, 0.5, 0.75] {
            let v = anchor + delta;
            if v >= 8.0 {
                xs.push(v);
            }
        }
    }
    xs.sort_by(f64::total_cmp);

    let mut child = incumbent()
        .command()
        .args(["-u", "-c", PYTHON])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn live scipy.special child");
    let mut stdin = child.stdin.take().expect("python stdin");
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for v in &xs {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    stdin.write_all(&bytes).expect("send grid");
    stdin.flush().expect("flush grid");
    drop(stdin);

    let mut out = BufReader::new(child.stdout.take().expect("python stdout"));
    let mut meta = String::new();
    out.read_line(&mut meta).expect("read meta");
    let fields: Vec<&str> = meta.split_whitespace().collect();
    assert_eq!(fields[0], "META", "unexpected reply: {meta:?}");
    let count: usize = fields[3].parse().expect("count");
    assert_eq!(
        count,
        xs.len(),
        "scipy returned {count} values for {} inputs",
        xs.len()
    );
    let mut raw = vec![0u8; count * 8];
    std::io::Read::read_exact(&mut out, &mut raw).expect("read reference");
    let reference: Vec<f64> = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let _ = child.wait();
    println!(
        "scipy={} numpy={} grid={} over [8, 120]",
        fields[1],
        fields[2],
        xs.len()
    );

    // The disputed interval is where the two configurations can differ at all.
    let disputed: Vec<usize> = (0..xs.len())
        .filter(|&i| xs[i] >= 13.0 && xs[i] < 100.0)
        .collect();
    let pick = |src: &[f64]| -> Vec<f64> { disputed.iter().map(|&i| src[i]).collect() };

    // Candidate crossovers come from argv so the question "how low can it go?" is answered
    // by sweeping rather than by re-editing the probe. Default sweep brackets the answer.
    let candidates: Vec<f64> = {
        let given: Vec<f64> = std::env::args()
            .skip(1)
            .filter_map(|a| a.parse().ok())
            .collect();
        if given.is_empty() {
            vec![13.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        } else {
            given
        }
    };

    let default_all = ours(&xs, None);
    let (base_abs, base_ulp, base_at) = worst(&pick(&default_all), &pick(&reference));
    println!("disputed_points={} (13 <= x < 100)", disputed.len());
    println!(
        "crossover=100 (default): worst_abs={base_abs:.6e} worst_ulp={base_ulp:.2} at lnGamma={base_at:.4}"
    );
    for &c in &candidates {
        let all = ours(&xs, Some(c));
        // Compare only where THIS candidate changes behaviour: x in [c, 100).
        let idx: Vec<usize> = (0..xs.len())
            .filter(|&i| xs[i] >= c && xs[i] < 100.0)
            .collect();
        let take = |src: &[f64]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
        let (a, u, at) = worst(&take(&all), &take(&reference));
        let (_, bu, _) = worst(&take(&default_all), &take(&reference));
        let ok = u <= bu;
        println!(
            "crossover={c:>5}: worst_abs={a:.6e} worst_ulp={u:.2} at lnGamma={at:.4}              default_here_ulp={bu:.2} points={} => {}",
            idx.len(),
            if ok { "JUSTIFIED" } else { "REJECTED" }
        );
    }

    let lowered_all = ours(&xs, Some(13.0));

    let (d_abs, d_ulp, d_at) = worst(&pick(&default_all), &pick(&reference));
    let (l_abs, l_ulp, l_at) = worst(&pick(&lowered_all), &pick(&reference));

    println!("disputed_points={} (13 <= x < 100)", disputed.len());
    println!(
        "crossover=100 (default): worst_abs={d_abs:.6e} worst_ulp={d_ulp:.2} at lnGamma={d_at:.4}"
    );
    println!(
        "crossover=13  (lowered): worst_abs={l_abs:.6e} worst_ulp={l_ulp:.2} at lnGamma={l_at:.4}"
    );

    // Outside the disputed interval the two must be IDENTICAL — if they are not, the
    // override is reaching further than it should and every comparison above is confounded.
    let mut outside_differs = 0usize;
    for i in 0..xs.len() {
        if !(xs[i] >= 13.0 && xs[i] < 100.0) && default_all[i].to_bits() != lowered_all[i].to_bits()
        {
            outside_differs += 1;
        }
    }
    println!("outside_disputed_differing={outside_differs} (must be 0)");
    assert_eq!(
        outside_differs, 0,
        "the crossover override changed values outside the interval it controls"
    );

    // Stated before the numbers were seen: ties go to the default.
    let verdict = if l_ulp <= d_ulp {
        "LOWERING IS JUSTIFIED ON ACCURACY"
    } else {
        "LOWERING IS NOT JUSTIFIED — the asymptotic kernel is less accurate here"
    };
    println!(
        "ratio_lowered_over_default_ulp={:.3}",
        l_ulp / d_ulp.max(f64::MIN_POSITIVE)
    );
    println!("VERDICT: {verdict}");
}
