#![forbid(unsafe_code)]
//! Cover fsci_spatial::{pdist_func, cdist_func, distance_matrix}.
//!
//! Resolves [frankenscipy-2yqwd]. Closure-metric variants of pdist/
//! cdist accept a user-supplied `Fn(&[f64], &[f64]) -> f64` callback,
//! letting callers plug in any metric (custom, weighted, asymmetric).
//! distance_matrix(x, y) is a thin wrapper around the Euclidean cdist.
//!
//! Properties verified:
//!   * pdist_func(euclidean) == pdist(DistanceMetric::Euclidean) over
//!     all n*(n-1)/2 condensed pairs
//!   * cdist_func(euclidean) == cdist_metric(Euclidean) element-wise
//!   * distance_matrix(x, y) == cdist_metric(x, y, Euclidean)
//!   * cdist_func with a custom metric (here: max-abs Chebyshev) yields
//!     the expected hand-computed values

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_spatial::{
    DistanceMetric, cdist_func, cdist_metric, chebyshev, distance_matrix, euclidean, pdist,
    pdist_func,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

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
    fs::create_dir_all(output_dir()).expect("create pdist_func diff dir");
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

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn matrix_max_abs(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| max_abs(ra, rb))
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_spatial_pdist_cdist_func() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    // One arm per fixed comparison (each is a single case, recorded under its own name as the
    // case id). The reference side is the plain enum-metric call or the hand-computed values;
    // slices rejects a non-finite element that the max_abs fold would swallow. The pair-count
    // and shape checks are structural and stay plain checks.
    let mut ledger = CompareLedger::new(
        "diff_spatial_pdist_cdist_func",
        &[
            "pdist_func_eq_pdist_euclidean",
            "cdist_func_eq_cdist_metric_euclidean",
            "distance_matrix_eq_cdist_euclidean",
            "cdist_func_custom_chebyshev_metric",
            "pdist_func_custom_half_sq",
        ],
    );
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    let data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![1.0, 2.0, 0.0],
        vec![3.0, 4.0, 5.0],
    ];

    // === 1. pdist_func(euclidean) ≡ pdist(Euclidean) ===
    {
        let arm = "pdist_func_eq_pdist_euclidean";
        let from_func = pdist_func(&data, euclidean);
        let from_enum = pdist(&data, DistanceMetric::Euclidean).ok();
        if let Some((from_enum, from_func)) =
            ledger.slices(arm, arm, from_enum.as_deref(), Some(from_func.as_slice()))
        {
            let pass =
                from_func.len() == from_enum.len() && max_abs(from_func, from_enum) <= ABS_TOL;
            ledger.compared(arm, arm, pass);
            check(
                arm,
                pass,
                format!(
                    "func_len={} enum_len={} max_abs={}",
                    from_func.len(),
                    from_enum.len(),
                    max_abs(from_func, from_enum)
                ),
            );
        }
        // 5 points → 10 pairs
        check(
            "pdist_func_correct_pair_count",
            from_func.len() == 10,
            format!("len={}", from_func.len()),
        );
    }

    // === 2. cdist_func(euclidean) ≡ cdist_metric(Euclidean) ===
    {
        let xa: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![3.0, 4.0]];
        let xb: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
        let arm = "cdist_func_eq_cdist_metric_euclidean";
        let from_func = cdist_func(&xa, &xb, euclidean);
        let from_enum = cdist_metric(&xa, &xb, DistanceMetric::Euclidean).ok();
        let enum_flat = from_enum.as_ref().map(|m| m.concat());
        let func_flat = from_func.concat();
        // slices returns Some only when from_enum is Some.
        if let (Some(_), Some(from_enum)) = (
            ledger.slices(arm, arm, enum_flat.as_deref(), Some(func_flat.as_slice())),
            from_enum.as_ref(),
        ) {
            let pass = from_func.len() == from_enum.len()
                && matrix_max_abs(&from_func, from_enum) <= ABS_TOL;
            ledger.compared(arm, arm, pass);
            check(
                arm,
                pass,
                format!("max_abs={}", matrix_max_abs(&from_func, from_enum)),
            );
        }
        check(
            "cdist_func_shape_correct",
            from_func.len() == 3 && from_func.iter().all(|r| r.len() == 2),
            format!(
                "rows={} cols0={}",
                from_func.len(),
                from_func.first().map_or(0, |r| r.len())
            ),
        );
    }

    // === 3. distance_matrix(x, y) ≡ cdist_metric(Euclidean) ===
    {
        let x: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let y: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let arm = "distance_matrix_eq_cdist_euclidean";
        let dm = distance_matrix(&x, &y).ok();
        let cm = cdist_metric(&x, &y, DistanceMetric::Euclidean).ok();
        let dm_flat = dm.as_ref().map(|m| m.concat());
        let cm_flat = cm.as_ref().map(|m| m.concat());
        // slices returns Some only when both dm and cm are Some.
        if let (Some(_), Some(dm), Some(cm)) = (
            ledger.slices(arm, arm, cm_flat.as_deref(), dm_flat.as_deref()),
            dm.as_ref(),
            cm.as_ref(),
        ) {
            let pass = dm.len() == cm.len() && matrix_max_abs(dm, cm) <= ABS_TOL;
            ledger.compared(arm, arm, pass);
            check(arm, pass, format!("max_abs={}", matrix_max_abs(dm, cm)));
        }
    }

    // === 4. cdist_func with a custom metric (Chebyshev) ===
    {
        let xa: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        let xb: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, -4.0]];
        let arm = "cdist_func_custom_chebyshev_metric";
        let result = cdist_func(&xa, &xb, chebyshev);
        // Chebyshev: max(|x_i - y_i|)
        // (0,0)→(1,2): max(1, 2) = 2
        // (0,0)→(3,-4): max(3, 4) = 4
        let expected = [2.0, 4.0];
        let result_flat = result.concat();
        if ledger
            .slices(
                arm,
                arm,
                Some(expected.as_slice()),
                Some(result_flat.as_slice()),
            )
            .is_some()
        {
            let pass = result.len() == 1
                && (result[0][0] - 2.0).abs() < ABS_TOL
                && (result[0][1] - 4.0).abs() < ABS_TOL;
            ledger.compared(arm, arm, pass);
            check(arm, pass, format!("result={result:?}"));
        }
    }

    // === 5. pdist_func with custom closure (sq euclidean / 2) ===
    {
        let pts: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 8.0]];
        let half_sq = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                / 2.0
        };
        let arm = "pdist_func_custom_half_sq";
        let res = pdist_func(&pts, half_sq);
        // (0,0)→(3,4): (9+16)/2 = 12.5
        // (0,0)→(6,8): (36+64)/2 = 50.0
        // (3,4)→(6,8): (9+16)/2 = 12.5
        let expected = [12.5, 50.0, 12.5];
        if ledger
            .slices(arm, arm, Some(expected.as_slice()), Some(res.as_slice()))
            .is_some()
        {
            let pass = res.len() == 3
                && (res[0] - 12.5).abs() < ABS_TOL
                && (res[1] - 50.0).abs() < ABS_TOL
                && (res[2] - 12.5).abs() < ABS_TOL;
            ledger.compared(arm, arm, pass);
            check(arm, pass, format!("res={res:?}"));
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_spatial_pdist_cdist_func".into(),
        category: "fsci_spatial::{pdist_func, cdist_func, distance_matrix} closure-metric coverage"
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
            eprintln!("pdist/cdist_func mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "pdist/cdist_func coverage failed: {} cases",
        diffs.len()
    );
    // Every arm is one fixed case, so each must have compared it.
    ledger.finish(1);
}
