#![forbid(unsafe_code)]
//! Cover fsci_ndimage::generic_filter.
//!
//! Resolves [frankenscipy-su57q]. generic_filter walks a sliding
//! window and applies a user-supplied closure to each
//! neighborhood. Verifies the closure receives the right values by
//! checking equivalence with the canonical filter primitives:
//!   * closure = max → equals maximum_filter
//!   * closure = min → equals minimum_filter
//!   * closure = sum / size² → equals uniform_filter
//!   * closure receives exactly size^ndim values
//!
//! Plus edge-case errors: size = 0, empty input.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, generic_filter, maximum_filter, minimum_filter, uniform_filter,
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
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create generic_filter diff dir");
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

#[test]
fn diff_ndimage_generic_filter() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // 5×5 test image with mixed positive/negative values
    let data: Vec<f64> = (0..25).map(|i| (i as f64 - 12.0) * 0.5).collect();
    let arr = NdArray::new(data, vec![5, 5]).expect("ndarray");

    let size = 3;
    let mode = BoundaryMode::Reflect;

    // === closure = max → equals maximum_filter ===
    {
        let g = generic_filter(
            &arr,
            |w| w.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            size,
            mode,
            0.0,
        )
        .expect("generic max");
        let m = maximum_filter(&arr, size, mode, 0.0).expect("maximum_filter");
        check(
            "max_closure_eq_maximum_filter",
            g.shape == m.shape && max_abs(&g.data, &m.data) <= ABS_TOL,
            format!("max_abs={}", max_abs(&g.data, &m.data)),
        );
    }

    // === closure = min → equals minimum_filter ===
    {
        let g = generic_filter(
            &arr,
            |w| w.iter().copied().fold(f64::INFINITY, f64::min),
            size,
            mode,
            0.0,
        )
        .expect("generic min");
        let m = minimum_filter(&arr, size, mode, 0.0).expect("minimum_filter");
        check(
            "min_closure_eq_minimum_filter",
            g.shape == m.shape && max_abs(&g.data, &m.data) <= ABS_TOL,
            format!("max_abs={}", max_abs(&g.data, &m.data)),
        );
    }

    // === closure = sum / size² → equals uniform_filter ===
    {
        let size_sq = (size * size) as f64;
        let g = generic_filter(&arr, |w| w.iter().sum::<f64>() / size_sq, size, mode, 0.0)
            .expect("generic uniform");
        let u = uniform_filter(&arr, size, mode, 0.0).expect("uniform_filter");
        check(
            "sum_closure_eq_uniform_filter",
            g.shape == u.shape && max_abs(&g.data, &u.data) <= ABS_TOL,
            format!("max_abs={}", max_abs(&g.data, &u.data)),
        );
    }

    // === Closure receives exactly size^ndim values per call ===
    {
        let expected_len = size * size; // 2D → size² values per neighborhood
        // generic_filter runs the closure in parallel across output cells, so the
        // collector must be Sync; Mutex<Vec> works because these checks (count +
        // all-equal) are order-independent.
        let observed_lengths = std::sync::Mutex::new(Vec::<usize>::new());
        let g = generic_filter(
            &arr,
            |w| {
                observed_lengths.lock().unwrap().push(w.len());
                w.iter().sum::<f64>()
            },
            size,
            mode,
            0.0,
        )
        .expect("generic capture");
        // Filter visits every output cell once
        let observed = observed_lengths.lock().unwrap();
        let calls_eq_size = observed.len() == arr.size();
        let all_lengths_correct = observed.iter().all(|&l| l == expected_len);
        check(
            "closure_called_per_output_cell",
            calls_eq_size,
            format!("calls={} expected={}", observed.len(), arr.size()),
        );
        check(
            "closure_neighborhood_length_correct",
            all_lengths_correct,
            format!(
                "expected={expected_len} actual={:?}",
                &observed[..5.min(observed.len())]
            ),
        );
        // sanity check the output is well-formed
        check(
            "generic_capture_shape_matches",
            g.shape == arr.shape,
            String::new(),
        );
    }

    // === Error: size = 0 ===
    {
        let r = generic_filter(&arr, |w| w[0], 0, mode, 0.0);
        check("size_zero_errors", r.is_err(), format!("res={r:?}"));
    }

    // === Error: empty input ===
    {
        let empty = NdArray::new(Vec::<f64>::new(), vec![0]).expect("empty ndarray");
        let r = generic_filter(&empty, |w| w.iter().sum(), 3, mode, 0.0);
        check("empty_input_errors", r.is_err(), format!("res={r:?}"));
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_ndimage_generic_filter".into(),
        category: "fsci_ndimage::generic_filter coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("generic_filter mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "generic_filter coverage failed: {} cases",
        diffs.len()
    );
}
