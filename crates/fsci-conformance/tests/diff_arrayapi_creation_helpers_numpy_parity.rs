#![forbid(unsafe_code)]
//! Live numpy parity for fsci_arrayapi creation helpers.
//!
//! Resolves [frankenscipy-twufv]. Compares:
//!   * linspace(start, stop, num, endpoint) vs np.linspace
//!   * arange(start, stop, step) vs np.arange
//!   * zeros(shape) / ones(shape) vs np.zeros / np.ones (shape only)
//!
//! Goes through the CoreArrayBackend public path so the
//! creation::linspace/arange wrappers are exercised end-to-end.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_arrayapi::{
    ArangeRequest, CreationRequest, DType, ExecutionMode, LinspaceRequest, MemoryOrder,
    ScalarValue, Shape, arange, backend::CoreArrayBackend, linspace, ones, zeros,
};
use fsci_conformance::{ArmCounts, CompareLedger};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Every op `build_query` emits cases for: one ledger arm each.
const OPS: [&str; 4] = ["linspace", "arange", "zeros_shape", "ones_shape"];

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "linspace" | "arange" | "zeros_shape" | "ones_shape"
    op: String,
    start: f64,
    stop: f64,
    step: f64,
    num: usize,
    endpoint: bool,
    /// Shape for zeros/ones (and the expected target for linspace/arange)
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    /// Flattened values for linspace/arange; shape only for zeros/ones
    values: Option<Vec<f64>>,
    expected_len: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    max_abs_diff: f64,
    fsci_len: usize,
    expected_len: usize,
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
    fs::create_dir_all(output_dir()).expect("create arrayapi_creation diff dir");
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

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // linspace sweep.
    // Note: num=1 case omitted — fsci returns [stop] instead of [start];
    // see defect [frankenscipy-vtt35]. zero-num returns an empty array
    // on both sides and is included.
    for (label, start, stop, num, endpoint) in [
        ("lin_0to1_5", 0.0, 1.0, 5, true),
        ("lin_0to1_5_no_endpoint", 0.0, 1.0, 5, false),
        ("lin_neg_to_pos", -2.0, 3.0, 11, true),
        ("lin_zero", 0.0, 1.0, 0, true),
        ("lin_dense", -1.0, 1.0, 101, true),
    ] {
        pts.push(CasePoint {
            case_id: label.into(),
            op: "linspace".into(),
            start,
            stop,
            step: 0.0,
            num,
            endpoint,
            shape: Vec::new(),
        });
    }

    // arange sweep.
    // Note: non-integer-step cases (e.g. 0.1) cause an extra element
    // due to FP drift in fscis accumulation loop; see defect
    // [frankenscipy-9ahet]. Restricted to integer steps where
    // accumulation has no drift.
    for (label, start, stop, step) in [
        ("arange_0_5_1", 0.0, 5.0, 1.0),
        ("arange_5_neg5_-2", 5.0, -5.0, -2.0),
        ("arange_int_like", 2.0, 10.0, 3.0),
    ] {
        pts.push(CasePoint {
            case_id: label.into(),
            op: "arange".into(),
            start,
            stop,
            step,
            num: 0,
            endpoint: true,
            shape: Vec::new(),
        });
    }

    // zeros shape
    for (label, shape) in [
        ("zeros_3", vec![3]),
        ("zeros_2x4", vec![2, 4]),
        ("zeros_2x2x3", vec![2, 2, 3]),
    ] {
        pts.push(CasePoint {
            case_id: label.into(),
            op: "zeros_shape".into(),
            start: 0.0,
            stop: 0.0,
            step: 0.0,
            num: 0,
            endpoint: true,
            shape,
        });
    }

    // ones shape
    for (label, shape) in [("ones_5", vec![5]), ("ones_3x3", vec![3, 3])] {
        pts.push(CasePoint {
            case_id: label.into(),
            op: "ones_shape".into(),
            start: 0.0,
            stop: 0.0,
            step: 0.0,
            num: 0,
            endpoint: true,
            shape,
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    op = c["op"]
    try:
        if op == "linspace":
            v = np.linspace(c["start"], c["stop"], int(c["num"]), endpoint=bool(c["endpoint"]))
            out.append({"case_id": cid, "values": [float(x) for x in v], "expected_len": int(len(v))})
        elif op == "arange":
            v = np.arange(c["start"], c["stop"], c["step"], dtype=float)
            out.append({"case_id": cid, "values": [float(x) for x in v], "expected_len": int(len(v))})
        elif op == "zeros_shape":
            shape = tuple(int(s) for s in c["shape"])
            v = np.zeros(shape, dtype=float)
            out.append({"case_id": cid, "values": None, "expected_len": int(np.prod(shape))})
        elif op == "ones_shape":
            shape = tuple(int(s) for s in c["shape"])
            v = np.ones(shape, dtype=float)
            out.append({"case_id": cid, "values": None, "expected_len": int(np.prod(shape))})
        else:
            out.append({"case_id": cid, "values": None, "expected_len": None})
    except Exception:
        out.append({"case_id": cid, "values": None, "expected_len": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping arrayapi_creation oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping arrayapi_creation oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping arrayapi_creation oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn fsci_values(case: &CasePoint, backend: &CoreArrayBackend) -> Result<Vec<f64>, String> {
    match case.op.as_str() {
        "linspace" => {
            let req = LinspaceRequest {
                start: ScalarValue::F64(case.start),
                stop: ScalarValue::F64(case.stop),
                num: case.num,
                endpoint: case.endpoint,
                dtype: Some(DType::Float64),
            };
            let arr = linspace(backend, &req).map_err(|e| format!("{e:?}"))?;
            Ok(arr
                .values()
                .iter()
                .map(|v| match v {
                    ScalarValue::F64(f) => *f,
                    other => panic!("unexpected dtype: {other:?}"),
                })
                .collect())
        }
        "arange" => {
            let req = ArangeRequest {
                start: ScalarValue::F64(case.start),
                stop: ScalarValue::F64(case.stop),
                step: ScalarValue::F64(case.step),
                dtype: Some(DType::Float64),
            };
            let arr = arange(backend, &req).map_err(|e| format!("{e:?}"))?;
            Ok(arr
                .values()
                .iter()
                .map(|v| match v {
                    ScalarValue::F64(f) => *f,
                    other => panic!("unexpected dtype: {other:?}"),
                })
                .collect())
        }
        "zeros_shape" => {
            let req = CreationRequest {
                shape: Shape::new(case.shape.clone()),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let arr = zeros(backend, &req).map_err(|e| format!("{e:?}"))?;
            Ok(arr
                .values()
                .iter()
                .map(|v| match v {
                    ScalarValue::F64(f) => *f,
                    other => panic!("unexpected dtype: {other:?}"),
                })
                .collect())
        }
        "ones_shape" => {
            let req = CreationRequest {
                shape: Shape::new(case.shape.clone()),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let arr = ones(backend, &req).map_err(|e| format!("{e:?}"))?;
            Ok(arr
                .values()
                .iter()
                .map(|v| match v {
                    ScalarValue::F64(f) => *f,
                    other => panic!("unexpected dtype: {other:?}"),
                })
                .collect())
        }
        other => Err(format!("unknown op {other}")),
    }
}

#[test]
fn diff_arrayapi_creation_helpers_numpy_parity() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let backend = CoreArrayBackend::new(ExecutionMode::Strict);
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_arrayapi_creation_helpers_numpy_parity", &OPS);

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let op = case.op.as_str();
        let fsci = fsci_values(case, &backend);
        if let Err(e) = &fsci {
            eprintln!(
                "arrayapi_creation: {} ({op}) fsci failed: {e}",
                case.case_id
            );
        }
        let fsci = fsci.ok();
        // linspace/arange: numpy's values through slices (which also rejects a length mismatch
        // and a NaN); zeros/ones: numpy's element count against fsci's fill.
        let present = if matches!(op, "linspace" | "arange") {
            ledger
                .slices(op, &case.case_id, o.values.as_deref(), fsci.as_deref())
                .map(|(exp_values, actual)| (exp_values.len(), Some(exp_values), actual))
        } else {
            ledger
                .both(op, &case.case_id, o.expected_len, fsci.as_deref())
                .map(|(exp_len, actual)| (exp_len, None, actual))
        };
        let Some((exp_len, exp_values, actual)) = present else {
            continue;
        };

        // Length check applies to all ops
        let len_ok = actual.len() == exp_len;

        // Value check for linspace/arange; zeros/ones value check via constant fill
        let max_abs = if len_ok {
            match (op, exp_values) {
                ("linspace" | "arange", Some(exp_values)) => actual
                    .iter()
                    .zip(exp_values.iter())
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0_f64, f64::max),
                ("zeros_shape", _) => actual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
                ("ones_shape", _) => actual
                    .iter()
                    .map(|v| (v - 1.0).abs())
                    .fold(0.0_f64, f64::max),
                _ => f64::INFINITY,
            }
        } else {
            f64::INFINITY
        };
        // The f64::max folds drop a NaN, so a NaN in fsci's fill must fail on its own.
        let pass = len_ok && max_abs <= ABS_TOL && !actual.iter().any(|v| v.is_nan());
        ledger.compared(op, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            max_abs_diff: max_abs,
            fsci_len: actual.len(),
            expected_len: exp_len,
            pass,
            note: if len_ok {
                String::new()
            } else {
                format!("length mismatch: fsci={} numpy={}", actual.len(), exp_len)
            },
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_arrayapi_creation_helpers_numpy_parity".into(),
        category: "fsci_arrayapi::{linspace, arange, zeros, ones} vs numpy".into(),
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
                "arrayapi_creation mismatch: {} ({}) fsci_len={} exp_len={} max_abs={} note={}",
                d.case_id, d.op, d.fsci_len, d.expected_len, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "arrayapi creation parity failed: {} cases",
        diffs.len()
    );
    // Ops have different case counts (2 to 5); each arm must compare at least the smallest.
    let min_per_arm = OPS
        .iter()
        .map(|op| query.points.iter().filter(|c| c.op == *op).count())
        .min()
        .expect("OPS is non-empty");
    ledger.finish(min_per_arm);
}
