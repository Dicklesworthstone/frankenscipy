//! Share + paired-A/B probe for the dense-`eigh` symmetric matvec (`frankenscipy-2o0vp`).
//!
//! Two jobs in ONE binary so both arms are the same ELF:
//!
//! * `share <n>`  — one `eigh` call and nothing else, for a `callgrind` instruction profile.
//! * `ab <n> <rounds>` — position-balanced paired A/B of `EIGH_DSYMV_PARALLEL_GATHER`
//!   against the shipping serial dsymv, with dual A/A nulls in the same window.
//!
//! WHY A PAIRED SCHEDULE AND NOT TWO RUNS: window drift on this host exceeds the effect
//! being looked for, so the arms are alternated inside one window and aggregated at the
//! replicate level. The quartet FLIPS with the round (ABBA even, BAAB odd) because ABBA
//! cancels drift but not POSITION — the first pass over a freshly built matrix is
//! systematically slower, and a fixed schedule charges that to whichever arm holds slot 1.

use std::sync::atomic::Ordering;
use std::time::Instant;

use fsci_linalg::{DecompOptions, eigh};

fn build(n: usize) -> Vec<Vec<f64>> {
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let v = next();
            a[i][j] = v;
            a[j][i] = v;
        }
        a[i][i] += n as f64;
    }
    a
}

/// One timed `eigh`, with the gather arm selected at the call boundary (never inside a loop).
fn timed(a: &[Vec<f64>], gather: bool) -> (f64, f64) {
    fsci_linalg::EIGH_DSYMV_PARALLEL_GATHER.store(gather, Ordering::Relaxed);
    let start = Instant::now();
    let result = eigh(a, DecompOptions::default()).expect("eigh");
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    fsci_linalg::EIGH_DSYMV_PARALLEL_GATHER.store(false, Ordering::Relaxed);
    (elapsed, result.eigenvalues.iter().sum())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "share".to_string());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);

    println!(
        "available_parallelism={}",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    );

    if mode == "share" {
        let force_scalar = std::env::var("EIGH_DSYMV_FORCE_SCALAR").as_deref() == Ok("1");
        fsci_linalg::EIGH_DSYMV_FORCE_SCALAR.store(force_scalar, Ordering::Relaxed);
        let force_double = std::env::var("EIGH_DSYMV_FORCE_DOUBLE_READ").as_deref() == Ok("1");
        fsci_linalg::EIGH_DSYMV_FORCE_DOUBLE_READ.store(force_double, Ordering::Relaxed);
        let a = build(n);
        let result = eigh(&a, DecompOptions::default()).expect("eigh");
        let checksum: f64 = result.eigenvalues.iter().sum();
        println!("n={n} force_scalar={force_scalar} force_double={force_double} checksum={checksum:.6}");
        return;
    }

    if mode == "stages" {
        // Stage timers report WALL time per stage, which is the right unit: two of the three
        // stages fan out across threads and one does not, so an instruction share cannot be
        // read as a time share. Default-off in the shipping arm, so this costs nothing there.
        let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
        let a = build(n);
        let _ = eigh(&a, DecompOptions::default()).expect("eigh"); // warm
        fsci_linalg::EIGH_NATIVE_STAGE_TIMING.store(true, Ordering::Relaxed);
        fsci_linalg::EIGH_REDUCE_SUBSTAGE_TIMING.store(true, Ordering::Relaxed);
        fsci_linalg::EIGH_SOLVE_SUBSTAGE_TIMING.store(true, Ordering::Relaxed);
        for slot in &fsci_linalg::EIGH_NATIVE_STAGE_NANOS {
            slot.store(0, Ordering::Relaxed);
        }
        for slot in &fsci_linalg::EIGH_REDUCE_SUBSTAGE_NANOS {
            slot.store(0, Ordering::Relaxed);
        }
        for slot in &fsci_linalg::EIGH_SOLVE_SUBSTAGE_NANOS {
            slot.store(0, Ordering::Relaxed);
        }
        let start = Instant::now();
        for _ in 0..reps {
            let _ = eigh(&a, DecompOptions::default()).expect("eigh");
        }
        let wall = start.elapsed().as_secs_f64() * 1e3;
        let stage: Vec<f64> = fsci_linalg::EIGH_NATIVE_STAGE_NANOS
            .iter()
            .map(|s| s.load(Ordering::Relaxed) as f64 / 1e6)
            .collect();
        let sub: Vec<f64> = fsci_linalg::EIGH_REDUCE_SUBSTAGE_NANOS
            .iter()
            .map(|s| s.load(Ordering::Relaxed) as f64 / 1e6)
            .collect();
        let solve: Vec<f64> = fsci_linalg::EIGH_SOLVE_SUBSTAGE_NANOS
            .iter()
            .map(|s| s.load(Ordering::Relaxed) as f64 / 1e6)
            .collect();
        let total = stage.iter().sum::<f64>();
        println!("mode=stages n={n} reps={reps} wall_total_ms={wall:.3} stage_sum_ms={total:.3}");
        let names = ["reduce(tridiagonalize)", "solve(tridiagonal)", "back_transform"];
        for (i, name) in names.iter().enumerate() {
            println!(
                "  stage[{i}] {name:<24} {:9.3} ms  {:5.2}% of stage_sum",
                stage[i] / reps as f64,
                100.0 * stage[i] / total
            );
        }
        println!(
            "  reduce.dsymv_gather      {:9.3} ms  {:5.2}% of reduce",
            sub[0] / reps as f64,
            100.0 * sub[0] / stage[0]
        );
        println!(
            "  reduce.rank2_update      {:9.3} ms  {:5.2}% of reduce",
            sub[1] / reps as f64,
            100.0 * sub[1] / stage[0]
        );
        println!(
            "  solve.eigenvalues        {:9.3} ms  {:5.2}% of solve",
            solve[0] / reps as f64,
            100.0 * solve[0] / stage[1]
        );
        println!(
            "  solve.eigenvectors       {:9.3} ms  {:5.2}% of solve",
            solve[1] / reps as f64,
            100.0 * solve[1] / stage[1]
        );
        fsci_linalg::EIGH_NATIVE_STAGE_TIMING.store(false, Ordering::Relaxed);
        fsci_linalg::EIGH_REDUCE_SUBSTAGE_TIMING.store(false, Ordering::Relaxed);
        fsci_linalg::EIGH_SOLVE_SUBSTAGE_TIMING.store(false, Ordering::Relaxed);
        return;
    }

    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let a = build(n);

    // Warm the allocator and the factor path so round 1 is not the cold outlier.
    let (_, warm) = timed(&a, false);

    let mut serial = Vec::new();
    let mut gather = Vec::new();
    let mut null_serial = Vec::new();
    let mut null_gather = Vec::new();
    let mut mismatches = 0usize;
    let mut reference: Option<u64> = None;

    for round in 0..rounds {
        // Position-balanced quartet: ABBA on even rounds, BAAB on odd.
        let order: [bool; 4] = if round % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        let mut s = Vec::new();
        let mut g = Vec::new();
        for &is_gather in &order {
            let (ms, checksum) = timed(&a, is_gather);
            let bits = checksum.to_bits();
            match reference {
                None => reference = Some(bits),
                Some(r) if r != bits => mismatches += 1,
                _ => {}
            }
            if is_gather { g.push(ms) } else { s.push(ms) }
        }
        // Effect: the two arms against each other. Null: each arm against ITSELF, from the
        // two same-arm samples the quartet already provides, so the null sees the identical
        // schedule and the identical window as the effect.
        serial.push(median(s.clone()));
        gather.push(median(g.clone()));
        null_serial.push(s[0] / s[1]);
        null_gather.push(g[0] / g[1]);
    }

    let ratios: Vec<f64> = serial
        .iter()
        .zip(&gather)
        .map(|(s, g)| s / g)
        .collect();

    println!("mode=ab n={n} rounds={rounds} warm_checksum={warm:.6}");
    println!("bit_mismatches={mismatches}  (0 required: the gather arm is documented BIT-IDENTICAL)");
    println!("serial_p50_ms={:.4}", median(serial.clone()));
    println!("gather_p50_ms={:.4}", median(gather.clone()));
    println!(
        "effect serial/gather p50={:.4} p10={:.4} p90={:.4}",
        median(ratios.clone()),
        pct(ratios.clone(), 0.10),
        pct(ratios.clone(), 0.90)
    );
    println!(
        "null_serial p50={:.4} p10={:.4} p90={:.4}",
        median(null_serial.clone()),
        pct(null_serial.clone(), 0.10),
        pct(null_serial.clone(), 0.90)
    );
    println!(
        "null_gather p50={:.4} p10={:.4} p90={:.4}",
        median(null_gather.clone()),
        pct(null_gather.clone(), 0.10),
        pct(null_gather.clone(), 0.90)
    );
}
