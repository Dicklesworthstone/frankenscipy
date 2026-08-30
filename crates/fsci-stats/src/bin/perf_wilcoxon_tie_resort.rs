//! `frankenscipy-921i0` — paired A/B for Wilcoxon's eliminated tie-resort.
//!
//! `resort` restores the historical clone + sort + tolerance tie walk; `pass` takes the exact
//! tie sum the ranking pass already computed. They are alternated ABBA in one process, with
//! exact-tie fixtures only, so the comparison prices redundant work without relaxing the SciPy
//! exact-equality semantics that ship by default.

use fsci_stats::{WILCOXON_FORCE_TIE_RESORT, wilcoxon};
use std::hint::black_box;
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

fn main() {
    let (reps, sizes) = parse_args().unwrap_or_else(|error| {
        eprintln!("usage: perf_wilcoxon_tie_resort [reps>=9] [size>13 ...]\n{error}");
        std::process::exit(2);
    });
    println!("# harness=perf_wilcoxon_tie_resort bead=frankenscipy-921i0 reps={reps}");

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
    }
}
