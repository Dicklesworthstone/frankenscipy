use fsci_linalg::{
    InvOptions, SolveOptions, inv, matmul, solve, solve_with_audit, verify_solve_certificate,
};
use fsci_runtime::{AuditLedger, Fingerprinter, RuntimeMode, SolverPortfolio};
use std::hint::black_box;
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
fn main() {
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
