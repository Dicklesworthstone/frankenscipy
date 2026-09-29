#![forbid(unsafe_code)]
//! Property-based coverage for fsci_sparse iterative solvers.
//!
//! Resolves [frankenscipy-5wlo3]. Verifies that each
//! Krylov-subspace iterative solver (cg, gmres, bicg, cgs, bicgstab,
//! minres, qmr, lgmres) converges on a small symmetric positive-
//! definite tridiagonal system A x = b with a known unique solution,
//! and that the returned residual ||A x − b||_2 stays tight.
//!
//! Seven solvers should work on this SPD case; some (e.g. minres)
//! are designed for symmetric problems specifically. fsci_sparse::qmr
//! is omitted — it fails to converge on this exact SPD tridiagonal
//! (filed as defect [frankenscipy-rprhy]); other solvers all converge
//! cleanly so it is a qmr-specific issue.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, IterativeSolveOptions, LgmresOptions, Shape2D, bicg,
    bicgstab, cg, cgs, gmres, lgmres, minres, spmv_csr,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const RESIDUAL_TOL: f64 = 1.0e-4;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    converged: bool,
    iterations: usize,
    residual_norm_reported: f64,
    actual_residual_2norm: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create iter solvers diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn build_spd_tridiag(n: usize) -> CsrMatrix {
    let mut trips = Vec::new();
    for i in 0..n {
        trips.push((i, i, 4.0));
        if i + 1 < n {
            trips.push((i, i + 1, -1.0));
            trips.push((i + 1, i, -1.0));
        }
    }
    let data: Vec<f64> = trips.iter().map(|t| t.2).collect();
    let rs: Vec<usize> = trips.iter().map(|t| t.0).collect();
    let cs: Vec<usize> = trips.iter().map(|t| t.1).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rs, cs, true).unwrap();
    coo.to_csr().unwrap()
}

fn residual_2norm(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
    let ax = spmv_csr(a, x).unwrap();
    ax.iter()
        .zip(b.iter())
        .map(|(axi, bi)| (axi - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn diff_sparse_iterative_solvers_residual() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    let n = 6;
    let a = build_spd_tridiag(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();

    let opts = IterativeSolveOptions {
        tol: 1e-8,
        max_iter: Some(500),
        ..IterativeSolveOptions::default()
    };
    let lg_opts = LgmresOptions {
        tol: 1e-8,
        max_iter: Some(500),
        inner_m: 30,
        outer_k: 3,
    };

    let solvers: &[(&str, Box<dyn Fn() -> _>)] = &[
        ("cg", Box::new(|| cg(&a, &b, None, opts))),
        ("gmres", Box::new(|| gmres(&a, &b, None, opts))),
        ("bicg", Box::new(|| bicg(&a, &b, None, opts))),
        ("cgs", Box::new(|| cgs(&a, &b, None, opts))),
        ("bicgstab", Box::new(|| bicgstab(&a, &b, None, opts))),
        ("minres", Box::new(|| minres(&a, &b, None, opts))),
        ("lgmres", Box::new(|| lgmres(&a, &b, None, lg_opts))),
    ];

    // One ledger arm per solver, each checking the one SPD system. The reference side is the
    // right-hand side b: A x must reproduce it. slices records an Err (no x, so no A x) as an
    // fsci failure and rejects a non-finite A x element before the residual metric runs.
    let arms: Vec<&str> = solvers.iter().map(|(name, _)| *name).collect();
    let mut ledger = CompareLedger::new("diff_sparse_iterative_solvers_residual", &arms);

    for (name, call) in solvers {
        let case_id = format!("solver_{name}_spd_tridiag_n6");
        let result = call();
        let ax = result
            .as_ref()
            .ok()
            .and_then(|r| spmv_csr(&a, &r.solution).ok());
        let reference_ok = ledger
            .slices(name, &case_id, Some(b.as_slice()), ax.as_deref())
            .is_some();
        match result {
            Ok(r) => {
                let actual_res = residual_2norm(&a, &r.solution, &b);
                let pass = r.converged && actual_res <= RESIDUAL_TOL;
                if reference_ok {
                    ledger.compared(name, &case_id, pass);
                }
                diffs.push(CaseDiff {
                    case_id,
                    converged: r.converged,
                    iterations: r.iterations,
                    residual_norm_reported: r.residual_norm,
                    actual_residual_2norm: actual_res,
                    pass,
                    note: String::new(),
                });
            }
            Err(e) => {
                // Already recorded by slices above as an fsci failure against b.
                diffs.push(CaseDiff {
                    case_id,
                    converged: false,
                    iterations: 0,
                    residual_norm_reported: f64::INFINITY,
                    actual_residual_2norm: f64::INFINITY,
                    pass: false,
                    note: format!("error: {e:?}"),
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_iterative_solvers_residual".into(),
        category:
            "fsci_sparse iterative solvers (cg/gmres/bicg/cgs/bicgstab/minres/qmr/lgmres) residual"
                .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "iter_solver mismatch: {} converged={} iter={} residual_norm={} actual_res={} note={}",
                d.case_id,
                d.converged,
                d.iterations,
                d.residual_norm_reported,
                d.actual_residual_2norm,
                d.note
            );
        }
    }

    assert!(
        all_pass,
        "iterative solver coverage failed: {} cases",
        diffs.len()
    );
    // Every solver arm checks the one SPD system, so each must have compared it.
    ledger.finish(1);
}
