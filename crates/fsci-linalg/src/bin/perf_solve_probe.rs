use fsci_linalg::{
    DISABLE_FLAT_LU_FACTOR, DecompOptions, InvOptions, SolveOptions, cholesky,
    condition_diagnostics, inv, matmul, solve, solve_with_action, solve_with_audit,
    solve_with_casp, verify_solve_certificate,
};
use fsci_runtime::{AuditLedger, Fingerprinter, RuntimeMode, SolverAction, SolverPortfolio};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn mk(n: usize, s: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let v = (i as f64 * 0.013 + j as f64 * 0.007 + s).sin();
                    if i == j { v + n as f64 } else { v * 0.1 } // diagonally dominant
                })
                .collect()
        })
        .collect()
}
fn t(label: &str, mut f: impl FnMut()) {
    f();
    let st = Instant::now();
    f();
    println!("{label}: {:.1} ms", st.elapsed().as_secs_f64() * 1e3);
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// frankenscipy-u87cd: the CASP portfolio path with the blocked LU (the default) against
/// nalgebra's serial LU (`DISABLE_FLAT_LU_FACTOR`), in one process, interleaved and
/// position-balanced (ABBA, then BAAB), with `solve()` timed in the same rounds.
fn portfolio_lu_arms() {
    for n in [512usize, 1024, 2048] {
        let a = mk(n, 0.3);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        let run = |blocked: bool| {
            DISABLE_FLAT_LU_FACTOR.store(!blocked, Ordering::Relaxed);
            let ledger = AuditLedger::shared();
            let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 4);
            let start = Instant::now();
            let audited =
                solve_with_audit(&a, &b, SolveOptions::default(), &mut portfolio, &ledger).unwrap();
            let audit_ms = start.elapsed().as_secs_f64() * 1e3;
            let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 4);
            let start = Instant::now();
            black_box(solve_with_casp(&a, &b, SolveOptions::default(), &mut portfolio).unwrap());
            let casp_ms = start.elapsed().as_secs_f64() * 1e3;
            DISABLE_FLAT_LU_FACTOR.store(false, Ordering::Relaxed);
            (audit_ms, casp_ms, audited.x)
        };
        let (_, _, x_blocked) = run(true);
        let (_, _, x_nalgebra) = run(false);
        let scale = x_nalgebra.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let agreement = x_blocked
            .iter()
            .zip(&x_nalgebra)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max)
            / scale;
        let (mut audit, mut casp) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        let mut plain = Vec::new();
        for round in 0..6 {
            let order = if round % 2 == 0 {
                [true, false, false, true]
            } else {
                [false, true, true, false]
            };
            for blocked in order {
                let (audit_ms, casp_ms, _) = run(blocked);
                audit[usize::from(blocked)].push(audit_ms);
                casp[usize::from(blocked)].push(casp_ms);
            }
            let start = Instant::now();
            black_box(solve(&a, &b, SolveOptions::default()).unwrap());
            plain.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let (audit_nalgebra, audit_blocked) = (median(&mut audit[0]), median(&mut audit[1]));
        let (casp_nalgebra, casp_blocked) = (median(&mut casp[0]), median(&mut casp[1]));
        println!(
            "u87cd n={n}: solve_with_audit nalgebra {audit_nalgebra:.1} ms -> blocked \
             {audit_blocked:.1} ms ({:.2}x); solve_with_casp nalgebra {casp_nalgebra:.1} ms -> \
             blocked {casp_blocked:.1} ms ({:.2}x); solve() {:.1} ms; blocked/nalgebra x \
             agree to {agreement:.1e} (medians of 12 per arm, 6 of solve())",
            audit_nalgebra / audit_blocked,
            casp_nalgebra / casp_blocked,
            median(&mut plain),
        );
    }
}

/// frankenscipy-w8bjb: where the portfolio's symmetric path spends its time next to the LU
/// path, on one symmetric positive definite matrix per n: each action through
/// `solve_with_action` (both run the same solve diagnostics, so their difference is what the
/// symmetric action adds), a Cholesky factorization alone, and the public
/// `condition_diagnostics` (the `inv` detection, which also tests positive definiteness).
/// Arms interleave, rotating the order per round.
fn symmetric_path_components() {
    let threads = std::thread::available_parallelism().map_or(0, |p| p.get());
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
    for n in [128usize, 512, 1024] {
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let v = (0.01 * (i + j) as f64 + 0.3).sin();
                        if i == j { v + n as f64 } else { v * 0.1 }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        let arms: [(&str, &dyn Fn()); 5] = [
            ("lu_action", &|| {
                black_box(solve_with_action(&a, &b, SolverAction::DirectLU).unwrap());
            }),
            ("symmetric_action", &|| {
                black_box(solve_with_action(&a, &b, SolverAction::SymmetricFastPath).unwrap());
            }),
            ("inv_diagnostics", &|| {
                black_box(condition_diagnostics(&a).unwrap());
            }),
            ("cholesky", &|| {
                black_box(cholesky(&a, true, DecompOptions::default()).unwrap());
            }),
            // A/A null: the LU arm again; lu_null / lu must come out near 1.
            ("lu_null", &|| {
                black_box(solve_with_action(&a, &b, SolverAction::DirectLU).unwrap());
            }),
        ];
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (_, arm) in &arms {
            arm();
        }
        // Sub-millisecond arms need more rounds for the A/A null to settle.
        let rounds = if n <= 128 { 35 } else { 7 };
        for round in 0..rounds {
            for k in 0..arms.len() {
                let index = (k + round) % arms.len();
                let start = Instant::now();
                arms[index].1();
                times[index].push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        let medians: Vec<f64> = times.iter_mut().map(|t| median(t)).collect();
        println!(
            "w8bjb n={n} host={} threads={threads}: {} (medians of {rounds} ms); \
             symmetric/lu {:.2}; A/A null lu_null/lu {:.2}; symmetric - lu {:.2} ms vs \
             cholesky {:.2} ms",
            host.trim(),
            arms.iter()
                .zip(&medians)
                .map(|((name, _), ms)| format!("{name} {ms:.2}"))
                .collect::<Vec<_>>()
                .join(", "),
            medians[1] / medians[0],
            medians[4] / medians[0],
            medians[1] - medians[0],
            medians[3],
        );
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--symmetric") {
        symmetric_path_components();
        return;
    }
    portfolio_lu_arms();
    for n in [1024usize, 2048] {
        let a = mk(n, 0.3);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        let bb = mk(n, 1.1);
        t(&format!("matmul   n={n}"), || {
            black_box(matmul(&a, &bb).unwrap());
        });
        t(&format!("solve    n={n}"), || {
            black_box(solve(&a, &b, SolveOptions::default()).unwrap());
        });
        t(&format!("inv      n={n}"), || {
            black_box(inv(&a, InvOptions::default()).unwrap());
        });
    }

    // frankenscipy-3cu8u.1: what the audit fingerprint (BLAKE3 over the whole of `a` and `b`)
    // costs next to the audited solve it belongs to, interleaved in one run.
    let n = 1024;
    let a = mk(n, 0.3);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 4);
    let ledger = AuditLedger::shared();
    for rep in 0..5 {
        let start = Instant::now();
        black_box(
            Fingerprinter::new("fsci_linalg::solve")
                .rows(&a)
                .f64s(&b)
                .finish(),
        );
        let fingerprint = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let result =
            solve_with_audit(&a, &b, SolveOptions::default(), &mut portfolio, &ledger).unwrap();
        let audited = start.elapsed().as_secs_f64();
        println!(
            "audit fingerprint n={n} rep={rep}: fingerprint {:.3} ms, solve_with_audit {:.1} ms, \
             share {:.4}",
            fingerprint * 1e3,
            audited * 1e3,
            fingerprint / audited
        );
        // frankenscipy-7tb8d.6: checking the certificate is O(n²) next to the O(n³) solve.
        let certificate = result
            .certificate
            .expect("portfolio solves carry a certificate");
        let start = Instant::now();
        let report = black_box(verify_solve_certificate(&a, &b, &result.x, &certificate));
        let verify = start.elapsed().as_secs_f64();
        assert!(report.verified, "{}", report.reason);
        println!(
            "certificate n={n} rep={rep}: verify {:.3} ms, share of solve_with_audit {:.4}, \
             forward bound {:?} ({:?})",
            verify * 1e3,
            verify / audited,
            certificate
                .accuracy
                .as_ref()
                .and_then(|a| a.forward_error_bound),
            certificate
                .accuracy
                .as_ref()
                .map(|a| a.forward_bound_method)
        );
    }
}
