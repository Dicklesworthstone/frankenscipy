//! BDF exact-diagonal structured Newton solve — §2 harness-contract A/B.
//!
//! WHY THIS BINARY EXISTS. `docs/LEDGER_RESURRECTION.md` rank-#1: entry `.165`
//! (2026-07-23) measured this lever at **17.4–19.3×** across four runs against A/A
//! nulls of 1.002–1.020 and was REJECTED anyway, because a `cv < 5%` ceiling — which
//! frankenmermaid's calibration proved is unreachable on this fleet — was the gate.
//! This binary re-decides it under the campaign's harness contract
//! (`/data/projects/frankenmermaid/docs/CROSS_REPO_RECOMMENDATION_bench_harness_contract.md`):
//!
//!   1. self-reporting ELF SHA-256 as line 1 — proves WHICH binary produced the
//!      numbers, which a shell-side hash next to the run cannot;
//!   2. `paired()` runs both arms INTERLEAVED inside one round with the order
//!      alternating per round; the statistic is the MEDIAN OF PER-ROUND RATIOS;
//!   3. it is called twice — `paired(base, base)` (A/A null) then
//!      `paired(base, cand)` — in the SAME invocation, so the null and the candidate
//!      see the same machine state;
//!   4. the verdict gates on the median-CI against the null, never on `cv`;
//!      `cv` is printed as provenance only;
//!   5. `min_of` inner replicates (keep the minimum) rather than long samples.
//!
//! BOTH ARMS ARE THE SAME BINARY, switched by `BDF_FORCE_DENSE_NEWTON`. `hits`
//! reports how many Newton factorizations took the diagonal path — a candidate arm
//! with `hits = 0` never ran the code under test, which is the exact failure mode
//! that voided a third of this repo's REJECT ledger.
//!
//! EXACTNESS is proven BEFORE timing: the full `SolveIvpResult` of both arms is
//! compared field by field with `to_bits()` (trajectory, time points, status,
//! message, `nfev`/`njev`/`nlu`) and the run aborts on any mismatch.
//!
//! Run: `cargo run --release --bin perf_bdf_diag_newton --features bdf-diag-bench -- [n] [rounds] [solves]`

#[cfg(feature = "bdf-diag-bench")]
mod bench {
    use fsci_integrate::bdf::{BDF_DIAG_NEWTON_HITS, BDF_FORCE_DENSE_NEWTON};
    use fsci_integrate::{SolveIvpOptions, SolveIvpResult, SolverKind, ToleranceValue, solve_ivp};
    use fsci_runtime::RuntimeMode;
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// Diagonal stiff decay: `y_j' = -(1 + 10 j) y_j`. Componentwise dynamics, so the
    /// finite-difference Jacobian is EXACTLY diagonal and the structural predicate
    /// fires. This is the `.165` fixture class (decoupled stiff relaxation) — the
    /// shape a method-of-lines discretisation with only local terms produces.
    fn rhs(_t: f64, y: &[f64]) -> Vec<f64> {
        (0..y.len())
            .map(|j| -(1.0 + 10.0 * j as f64) * y[j])
            .collect()
    }

    fn options<'a>(y0: &'a [f64]) -> SolveIvpOptions<'a> {
        SolveIvpOptions {
            t_span: (0.0, 1.0),
            y0,
            method: SolverKind::Bdf,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            mode: RuntimeMode::Strict,
            ..Default::default()
        }
    }

    fn solve(y0: &[f64]) -> SolveIvpResult {
        solve_ivp(&mut |t: f64, y: &[f64]| rhs(t, y), &options(y0)).expect("bdf solve")
    }

    /// Every observable field of the result, as raw bits.
    fn result_bits(r: &SolveIvpResult) -> Vec<u64> {
        let mut bits: Vec<u64> = r.t.iter().map(|v| v.to_bits()).collect();
        for row in &r.y {
            bits.extend(row.iter().map(|v| v.to_bits()));
        }
        bits.push(r.nfev as u64);
        bits.push(r.njev as u64);
        bits.push(r.nlu as u64);
        bits.push(r.status as u64);
        bits.push(u64::from(r.success));
        bits.extend(r.message.bytes().map(u64::from));
        bits
    }

    /// One timed sample: `solves` integrations under `dense`, keeping the MINIMUM of
    /// `min_of` inner replicates (the dominant knob per §2.4).
    fn sample(y0: &[f64], solves: usize, dense: bool, min_of: usize) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..min_of {
            BDF_FORCE_DENSE_NEWTON.store(dense, Ordering::Relaxed);
            let start = Instant::now();
            for _ in 0..solves {
                black_box(solve(black_box(y0)));
            }
            let secs = start.elapsed().as_secs_f64();
            if secs < best {
                best = secs;
            }
        }
        best
    }

    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(f64::total_cmp);
        let n = xs.len();
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            0.5 * (xs[n / 2 - 1] + xs[n / 2])
        }
    }

    fn quantile(xs: &[f64], q: f64) -> f64 {
        let mut v = xs.to_vec();
        v.sort_by(f64::total_cmp);
        let idx = ((v.len() - 1) as f64 * q).round() as usize;
        v[idx]
    }

    struct Paired {
        p50_a: f64,
        p50_b: f64,
        ratios: Vec<f64>,
        ratio_p50: f64,
        ratio_lo: f64,
        ratio_hi: f64,
        cv: f64,
    }

    /// Interleave two arms inside each round, alternating which goes first, and
    /// report the MEDIAN OF PER-ROUND RATIOS with a 95% percentile interval.
    fn paired(
        y0: &[f64],
        solves: usize,
        rounds: usize,
        min_of: usize,
        dense_a: bool,
        dense_b: bool,
    ) -> Paired {
        let mut ta = Vec::with_capacity(rounds);
        let mut tb = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (a, b) = if round % 2 == 0 {
                let a = sample(y0, solves, dense_a, min_of);
                let b = sample(y0, solves, dense_b, min_of);
                (a, b)
            } else {
                let b = sample(y0, solves, dense_b, min_of);
                let a = sample(y0, solves, dense_a, min_of);
                (a, b)
            };
            ta.push(a);
            tb.push(b);
            ratios.push(a / b); // >1 means arm B (candidate) is faster
        }
        let ratio_p50 = median(ratios.clone());
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;
        Paired {
            p50_a: median(ta),
            p50_b: median(tb),
            ratio_lo: quantile(&ratios, 0.025),
            ratio_hi: quantile(&ratios, 0.975),
            ratio_p50,
            cv: var.sqrt() / mean,
            ratios,
        }
    }

    fn report(label: &str, p: &Paired) {
        println!(
            "{label:<10} p50_a={:.6}ms p50_b={:.6}ms ratio_p50={:.6} ci95=[{:.6},{:.6}] cv={:.3}% n={}",
            p.p50_a * 1e3,
            p.p50_b * 1e3,
            p.ratio_p50,
            p.ratio_lo,
            p.ratio_hi,
            p.cv * 100.0,
            p.ratios.len()
        );
    }

    pub fn run() {
        // §2.1 — the binary hashes its own ELF and prints it first.
        let exe = std::env::current_exe().expect("current_exe");
        let sha = {
            let bytes = std::fs::read(&exe).expect("read own ELF");
            let mut h = Sha256::new();
            h.update(&bytes);
            format!("{:x}", h.finalize())
        };
        println!("elf_sha256={sha}");
        println!("elf_path={}", exe.display());

        let args: Vec<String> = std::env::args().collect();
        let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
        let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
        let solves: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
        let min_of: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
        println!("n={n} rounds={rounds} solves={solves} min_of={min_of}");

        let y0: Vec<f64> = (0..n).map(|j| 1.0 + 0.25 * (j % 7) as f64).collect();

        // ── EXACTNESS BEFORE TIMING ──────────────────────────────────────────────
        BDF_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        BDF_FORCE_DENSE_NEWTON.store(true, Ordering::Relaxed);
        let base = solve(&y0);
        let hits_base = BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        BDF_FORCE_DENSE_NEWTON.store(false, Ordering::Relaxed);
        let cand = solve(&y0);
        let hits_cand = BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed) - hits_base;
        let (bb, cb) = (result_bits(&base), result_bits(&cand));
        let mismatches = if bb.len() != cb.len() {
            usize::MAX
        } else {
            bb.iter().zip(cb.iter()).filter(|(x, y)| x != y).count()
        };
        println!(
            "exactness: bitmism={mismatches} fields={} steps={} nfev={} njev={} nlu={} status={} \
             hits_base={hits_base} hits_cand={hits_cand}",
            bb.len(),
            base.t.len(),
            base.nfev,
            base.njev,
            base.nlu,
            base.status
        );
        if mismatches != 0 {
            eprintln!("ABORT: candidate is not bit-identical — no timing is admissible");
            std::process::exit(3);
        }
        if hits_base != 0 || hits_cand == 0 {
            eprintln!(
                "ABORT: arm switch did not take (hits_base={hits_base}, hits_cand={hits_cand}) \
                 — the A/B would measure nothing"
            );
            std::process::exit(4);
        }

        // ── A/A NULL, then A/B, same invocation ──────────────────────────────────
        let null = paired(&y0, solves, rounds, min_of, true, true);
        report("NULL", &null);
        let ab = paired(&y0, solves, rounds, min_of, true, false);
        report("CAND", &ab);

        // §2.3 — decide on the median-CI against the null, with a 2× margin.
        let null_edge = null.ratio_hi.max(1.0 / null.ratio_lo.max(f64::MIN_POSITIVE));
        let required = 1.0 + 2.0 * (null_edge - 1.0);
        let decided = ab.ratio_lo > required;
        println!(
            "gate: null_edge={null_edge:.6} required={required:.6} cand_ci_lo={:.6} => {}",
            ab.ratio_lo,
            if decided { "DECIDED" } else { "NOT DECIDED" }
        );
        println!(
            "verdict: {} ratio_p50={:.4}x",
            if decided { "WIN" } else { "IN-FLOOR" },
            ab.ratio_p50
        );
    }
}

#[cfg(feature = "bdf-diag-bench")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "bdf-diag-bench"))]
fn main() {
    eprintln!("perf_bdf_diag_newton requires --features bdf-diag-bench");
    std::process::exit(2);
}
