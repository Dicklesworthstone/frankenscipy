//! `frankenscipy-921i0` — paired A/B for Wilcoxon's eliminated tie-resort.
//!
//! `resort` restores the historical clone + sort + tolerance tie walk; `pass` takes the exact
//! tie sum the ranking pass already computed. They are alternated ABBA in one process, with
//! exact-tie fixtures only, so the comparison prices redundant work without relaxing the SciPy
//! exact-equality semantics that ship by default.

use fsci_stats::{WILCOXON_FORCE_TIE_RESORT, wilcoxon};
use std::hint::black_box;
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(index).copied().unwrap_or(f64::NAN)
}

fn parse_args() -> Result<(usize, Vec<usize>), String> {
    let args: Vec<String> = std::env::args().collect();
    let reps = args.get(1).map_or(Ok(21), |value| {
        value
            .parse::<usize>()
            .map_err(|_| format!("reps must be an integer, got {value:?}"))
    })?;
    if reps < 9 {
        return Err("at least 9 ABBA replicates are required".to_string());
    }
    let sizes = if args.len() > 2 {
        args.iter()
            .skip(2)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("size must be an integer, got {value:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![20_000, 200_000, 2_000_000]
    };
    if sizes.iter().any(|&size| size < 14) {
        return Err(
            "every size must exceed 13 so the tie correction reaches the normal path".into(),
        );
    }
    Ok((reps, sizes))
}

/// Exact repeated magnitudes make the historic tolerance grouping and shipping exact grouping
/// equivalent, while alternating signs ensure both rank sums are nontrivial.
fn fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let alphabet = (n / 8).max(2) as u64;
    let x = (0..n)
        .map(|i| {
            let magnitude = ((i as u64).wrapping_mul(7_919) % alphabet + 1) as f64;
            if i % 2 == 0 { magnitude } else { -magnitude }
        })
        .collect();
    (x, vec![0.0; n])
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| {
            let fields: Vec<&str> = text.split_whitespace().take(3).collect();
            (fields.len() == 3).then(|| fields.join("/"))
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

fn self_elf_sha256() -> (String, String) {
    let exe = match std::fs::read_link("/proc/self/exe") {
        Ok(path) => path,
        Err(error) => return ("unavailable".to_string(), format!("<unresolved: {error}>")),
    };
    let digest = Command::new("sha256sum")
        .arg(&exe)
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()?
                .split_whitespace()
                .next()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unavailable".to_string());
    (digest, exe.display().to_string())
}

// `method="approx"` and `correction=False` pin the same normal-approximation contract as the
// Rust public API on this tied, n > 13 fixture. The fixture expression is intentionally repeated
// here rather than serialized from Rust, making both inputs auditable in the same invocation.
const PY: &str = r#"
import sys, time
import numpy as np
from scipy.stats import wilcoxon
n = int(sys.argv[1]); reps = int(sys.argv[2]); inner = int(sys.argv[3])
alphabet = max(n // 8, 2)
i = np.arange(n, dtype=np.uint64)
magnitude = ((i * np.uint64(7919)) % np.uint64(alphabet) + 1).astype(np.float64)
magnitude[1::2] *= -1.0
y = np.zeros(n, dtype=np.float64)
wilcoxon(magnitude, y, method="approx", correction=False, zero_method="wilcox")
times = []
for _ in range(reps):
    started = time.perf_counter()
    for _ in range(inner):
        result = wilcoxon(magnitude, y, method="approx", correction=False, zero_method="wilcox")
    times.append((time.perf_counter() - started) * 1e3 / inner)
times.sort()
print("%.6f %.17g %.17g" % (times[len(times)//2], result.statistic, result.pvalue))
"#;

fn run_python(python: &str, n: usize, reps: usize, inner: usize) -> Option<(f64, f64, f64)> {
    let output = Command::new(python)
        .arg("-c")
        .arg(PY)
        .arg(n.to_string())
        .arg(reps.to_string())
        .arg(inner.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!(
            "# scipy arm failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return None;
    }
    let fields: Vec<f64> = String::from_utf8(output.stdout)
        .ok()?
        .split_whitespace()
        .filter_map(|field| field.parse().ok())
        .collect();
    (fields.len() == 3).then(|| (fields[0], fields[1], fields[2]))
}

fn main() {
    let (reps, sizes) = parse_args().unwrap_or_else(|error| {
        eprintln!("usage: perf_wilcoxon_tie_resort [reps>=9] [size>13 ...]\n{error}");
        std::process::exit(2);
    });
    let require_scipy = std::env::var("FSCI_REQUIRE_SCIPY").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (elf_sha256, elf_path) = self_elf_sha256();
    println!(
        "# harness=perf_wilcoxon_tie_resort bead=frankenscipy-921i0 reps={reps} \\
         incumbent=scipy.stats.wilcoxon method=approx correction=false zero_method=wilcox"
    );
    println!("# elf_sha256={elf_sha256} elf_path={elf_path} require_scipy={require_scipy}");

    let mut failed = false;

    for n in sizes {
        let (x, y) = fixture(n);
        let inner = (1_000_000usize / n).max(1);
        let time_one = |resort: bool| {
            WILCOXON_FORCE_TIE_RESORT.store(resort, Ordering::Relaxed);
            let started = Instant::now();
            for _ in 0..inner {
                black_box(wilcoxon(black_box(&x), black_box(&y)));
            }
            WILCOXON_FORCE_TIE_RESORT.store(false, Ordering::Relaxed);
            started.elapsed().as_secs_f64() * 1e3 / inner as f64
        };

        WILCOXON_FORCE_TIE_RESORT.store(true, Ordering::Relaxed);
        let original = wilcoxon(&x, &y);
        WILCOXON_FORCE_TIE_RESORT.store(false, Ordering::Relaxed);
        let shipping = wilcoxon(&x, &y);
        let bitmism = usize::from(original.statistic.to_bits() != shipping.statistic.to_bits())
            + usize::from(original.pvalue.to_bits() != shipping.pvalue.to_bits());
        assert_eq!(
            bitmism, 0,
            "exact-tie fixture changed semantics across the A/B switch"
        );

        let _ = time_one(true);
        let _ = time_one(false);
        let (mut candidate, mut null) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
        let (mut resort_ms, mut pass_ms) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
        for _ in 0..reps {
            let a1 = time_one(true);
            let b1 = time_one(false);
            let b2 = time_one(false);
            let a2 = time_one(true);
            candidate.push((a1 + a2) / (b1 + b2));
            null.push(a1 / a2);
            resort_ms.push((a1 + a2) / 2.0);
            pass_ms.push((b1 + b2) / 2.0);
        }
        let candidate_median = percentile(&candidate, 0.5);
        let null_median = percentile(&null, 0.5);
        let candidate_p10 = percentile(&candidate, 0.10);
        let null_p90 = percentile(&null, 0.90);
        println!(
            "n={n} resort={:.3}ms pass={:.3}ms ratio(resort/pass)={candidate_median:.4}x \\
             candidate_p10={candidate_p10:.4} null_A/A_median={null_median:.4} \\
             null_A/A_p90={null_p90:.4} bitmism={bitmism} inner={inner}",
            percentile(&resort_ms, 0.5),
            percentile(&pass_ms, 0.5),
        );

        // The scipy arm brackets the shipping Rust arm inside one process invocation. Its A/A
        // ratio is the drift control: it must remain within 0.97--1.03 before a live ratio is
        // interpreted as a performance result.
        let load_scipy_1 = loadavg();
        let scipy_1 = run_python(&python, n, reps, inner);
        WILCOXON_FORCE_TIE_RESORT.store(false, Ordering::Relaxed);
        let mut shipping_times = Vec::with_capacity(reps);
        let mut shipping_result = wilcoxon(&x, &y);
        for _ in 0..reps {
            let started = Instant::now();
            for _ in 0..inner {
                shipping_result = black_box(wilcoxon(black_box(&x), black_box(&y)));
            }
            shipping_times.push(started.elapsed().as_secs_f64() * 1e3 / inner as f64);
        }
        let shipping_ms = percentile(&shipping_times, 0.5);
        let load_fsci = loadavg();
        let scipy_2 = run_python(&python, n, reps, inner);
        let load_scipy_2 = loadavg();

        match (scipy_1, scipy_2) {
            (Some(first), Some(second)) => {
                let scipy_ms = (first.0 + second.0) / 2.0;
                let scipy_null = first.0 / second.0;
                let statistic_gap =
                    ((shipping_result.statistic - first.1) / first.1.abs().max(1.0)).abs();
                let pvalue_gap =
                    ((shipping_result.pvalue - first.2) / first.2.abs().max(1e-300)).abs();
                let agree = statistic_gap < 1e-12 && pvalue_gap < 1e-9;
                println!(
                    "n={n} LIVE fsci={shipping_ms:.3}ms scipy={scipy_ms:.3}ms \\
                     ratio(scipy/fsci)={:.4}x scipy_A/A={scipy_null:.4} agree={agree} \\
                     stat_gap={statistic_gap:.3e} p_gap={pvalue_gap:.3e} inner={inner}",
                    scipy_ms / shipping_ms,
                );
                println!("    load scipy1={load_scipy_1} fsci={load_fsci} scipy2={load_scipy_2}");
                if !agree {
                    failed = true;
                }
            }
            _ => {
                println!("n={n} LIVE-SCIPY-ARM-MISSING");
                failed |= require_scipy;
            }
        }
    }

    if failed {
        std::process::exit(1);
    }
}
