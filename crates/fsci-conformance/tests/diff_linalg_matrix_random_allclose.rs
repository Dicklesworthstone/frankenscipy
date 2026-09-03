#![forbid(unsafe_code)]
//! Cover fsci_linalg::{mat_allclose, random_matrix, random_spd}.
//!
//! Resolves [frankenscipy-yk9wx].
//! mat_allclose semantics:
//!   * close within atol+rtol → true; outside → false
//!   * shape mismatch → false
//!   * NaN/NaN → true; Inf/Inf → true; mixed → false
//!
//! random_matrix:
//!   * shape correct
//!   * deterministic for same seed
//!   * different seeds produce different matrices
//!   * values in [0, 1)
//!
//! random_spd:
//!   * shape n × n
//!   * symmetric (A == A^T) within tol
//!   * positive definite (cholesky succeeds)

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, cholesky, mat_allclose, random_matrix, random_spd};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create matrand diff dir");
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

#[test]
fn diff_linalg_matrix_random_allclose() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === mat_allclose semantics ===
    {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0 + 1e-12, 2.0], vec![3.0 - 1e-12, 4.0]];
        check(
            "mat_allclose_within_tol_true",
            mat_allclose(&a, &b, 1e-9, 1e-9),
            String::new(),
        );

        let c = vec![vec![1.0, 2.0], vec![3.0, 99.0]];
        check(
            "mat_allclose_outside_tol_false",
            !mat_allclose(&a, &c, 1e-9, 1e-9),
            String::new(),
        );

        let d = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        check(
            "mat_allclose_shape_mismatch_rows_false",
            !mat_allclose(&a, &d, 1e-9, 1e-9),
            String::new(),
        );

        let e = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let f = vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]];
        check(
            "mat_allclose_shape_mismatch_cols_false",
            !mat_allclose(&e, &f, 1e-9, 1e-9),
            String::new(),
        );

        // NaN handling
        let nan_a = vec![vec![f64::NAN, 1.0]];
        let nan_b = vec![vec![f64::NAN, 1.0]];
        check(
            "mat_allclose_nan_nan_true",
            mat_allclose(&nan_a, &nan_b, 1e-9, 1e-9),
            String::new(),
        );

        // Inf-Inf
        let inf_a = vec![vec![f64::INFINITY, f64::NEG_INFINITY]];
        let inf_b = vec![vec![f64::INFINITY, f64::NEG_INFINITY]];
        check(
            "mat_allclose_inf_inf_true",
            mat_allclose(&inf_a, &inf_b, 1e-9, 1e-9),
            String::new(),
        );

        // Mixed NaN/finite
        let mixed_a = vec![vec![f64::NAN, 1.0]];
        let mixed_b = vec![vec![0.0, 1.0]];
        check(
            "mat_allclose_nan_vs_finite_false",
            !mat_allclose(&mixed_a, &mixed_b, 1e-9, 1e-9),
            String::new(),
        );
    }

    // === random_matrix ===
    {
        let m = random_matrix(4, 5, 42);
        check(
            "random_matrix_shape",
            m.len() == 4 && m.iter().all(|r| r.len() == 5),
            format!("rows={}", m.len()),
        );
        check(
            "random_matrix_values_in_unit_interval",
            m.iter().flatten().all(|&v| (0.0..1.0).contains(&v)),
            "range_check".to_string(),
        );
        // Determinism
        let m2 = random_matrix(4, 5, 42);
        check(
            "random_matrix_deterministic_same_seed",
            m == m2,
            String::new(),
        );
        // Different seeds → different matrices
        let m3 = random_matrix(4, 5, 99);
        check(
            "random_matrix_different_seed_different_matrix",
            m != m3,
            String::new(),
        );
    }

    // === random_spd ===
    {
        let n = 5;
        let s = random_spd(n, 123);
        check(
            "random_spd_shape",
            s.len() == n && s.iter().all(|r| r.len() == n),
            format!("rows={}", s.len()),
        );

        // Symmetric: s[i][j] == s[j][i] within tight tol
        let mut max_asym = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_asym = max_asym.max((s[i][j] - s[j][i]).abs());
            }
        }
        check(
            "random_spd_symmetric",
            max_asym <= 1e-12,
            format!("max_asym={max_asym}"),
        );

        // Positive definite: cholesky should succeed
        let chol = cholesky(&s, true, DecompOptions::default());
        check(
            "random_spd_cholesky_succeeds",
            chol.is_ok(),
            format!("chol_err={:?}", chol.err()),
        );

        // Determinism
        let s2 = random_spd(n, 123);
        check("random_spd_deterministic_same_seed", s == s2, String::new());
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_matrix_random_allclose".into(),
        category: "fsci_linalg::{mat_allclose, random_matrix, random_spd} coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("matrand mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(all_pass, "matrand coverage failed: {} cases", diffs.len());
}
