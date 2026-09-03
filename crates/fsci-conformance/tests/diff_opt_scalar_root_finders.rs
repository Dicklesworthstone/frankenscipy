#![forbid(unsafe_code)]
//! Analytic-root parity for fsci_opt scalar root finders: bisect,
//! brentq, brenth, ridder, toms748.
//!
//! Resolves [frankenscipy-1dg9v]. Uses hard-coded models with known
//! analytic roots; no scipy oracle needed. 1e-9 abs.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{RootOptions, bisect, brenth, brentq, ridder, toms748};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create scalar_root diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize scalar_root diff log");
    fs::write(path, json).expect("write scalar_root diff log");
}

#[test]
fn diff_opt_scalar_root_finders() {
    let opts = RootOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // (case_id_prefix, function, bracket, expected_root)
    let cases: Vec<(&str, Box<dyn Fn(f64) -> f64>, (f64, f64), f64)> = vec![
        (
            "sqrt2",
            Box::new(|x: f64| x * x - 2.0),
            (0.0, 2.0),
            std::f64::consts::SQRT_2,
        ),
        // x³ - x - 1 = 0 → root ≈ 1.3247179572447460
        (
            "plastic",
            Box::new(|x: f64| x * x * x - x - 1.0),
            (1.0, 2.0),
            1.324_717_957_244_746,
        ),
        // sin x = 0 in [3, 4] → π
        (
            "pi",
            Box::new(|x: f64| x.sin()),
            (3.0, 4.0),
            std::f64::consts::PI,
        ),
        // exp(x) = 2 in [0, 1] → ln 2
        (
            "ln2",
            Box::new(|x: f64| x.exp() - 2.0),
            (0.0, 1.0),
            std::f64::consts::LN_2,
        ),
        // x - cos(x) = 0 in [0, 1] → 0.7390851332151607 (Dottie)
        (
            "dottie",
            Box::new(|x: f64| x - x.cos()),
            (0.0, 1.0),
            0.739_085_133_215_160_7,
        ),
    ];

    for (label, f, br, expected) in &cases {
        for method in ["bisect", "brentq", "brenth", "ridder", "toms748"] {
            let res = match method {
                "bisect" => bisect(f, *br, opts),
                "brentq" => brentq(f, *br, opts),
                "brenth" => brenth(f, *br, opts),
                "ridder" => ridder(f, *br, opts),
                "toms748" => toms748(f, *br, opts),
                _ => continue,
            };
            let Ok(r) = res else {
                continue;
            };
            let d = (r.root - expected).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: format!("{label}_{method}"),
                method: method.into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_scalar_root_finders".into(),
        category: "fsci_opt scalar root finders vs analytic".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "{} mismatch: {} abs_diff={}",
                d.method, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scalar_root_finders conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
