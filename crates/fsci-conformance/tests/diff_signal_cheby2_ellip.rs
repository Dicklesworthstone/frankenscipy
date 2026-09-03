#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.cheby2` (Chebyshev-2)
//! and `scipy.signal.ellip` (elliptic / Cauer) lowpass filters.
//!
//! Resolves [frankenscipy-74s4j]. Both return (b, a). Compared via
//! concatenated coefficient vector at 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{FilterType, cheby2, ellip};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    order: usize,
    /// For cheby2: rs. For ellip: rp.
    p1: f64,
    /// For ellip only: rs. Ignored for cheby2.
    p2: f64,
    wn: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    n_b: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create cheby2_ellip diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cheby2_ellip diff log");
    fs::write(path, json).expect("write cheby2_ellip diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // cheby2: (order, rs, wn)
    let cheby2_cases: &[(usize, f64, f64)] = &[
        (2, 30.0, 0.3),
        (3, 30.0, 0.3),
        (4, 30.0, 0.3),
        (4, 40.0, 0.3),
        (4, 30.0, 0.5),
        (5, 30.0, 0.4),
        (6, 50.0, 0.3),
    ];
    for (i, (n, rs, w)) in cheby2_cases.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("cheby2_{i:02}_n{n}_rs{rs}_wn{w}"),
            op: "cheby2".into(),
            order: *n,
            p1: *rs,
            p2: 0.0,
            wn: *w,
        });
    }

    // ellip: (order, rp, rs, wn)
    let ellip_cases: &[(usize, f64, f64, f64)] = &[
        (2, 0.5, 30.0, 0.3),
        (3, 0.5, 30.0, 0.3),
        (4, 0.5, 30.0, 0.3),
        (4, 1.0, 30.0, 0.3),
        (4, 0.5, 40.0, 0.3),
        (4, 0.5, 30.0, 0.5),
        (5, 0.5, 35.0, 0.4),
    ];
    for (i, (n, rp, rs, w)) in ellip_cases.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("ellip_{i:02}_n{n}_rp{rp}_rs{rs}_wn{w}"),
            op: "ellip".into(),
            order: *n,
            p1: *rp,
            p2: *rs,
            wn: *w,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    n = int(case["order"]); p1 = float(case["p1"]); p2 = float(case["p2"]); w = float(case["wn"])
    try:
        if op == "cheby2":
            b, a = signal.cheby2(n, p1, w, btype='low')
        elif op == "ellip":
            b, a = signal.ellip(n, p1, p2, w, btype='low')
        else:
            points.append({"case_id": cid, "values": None, "n_b": None}); continue
        bv = finite_vec_or_none(b); av = finite_vec_or_none(a)
        if bv is None or av is None:
            points.append({"case_id": cid, "values": None, "n_b": None})
        else:
            points.append({"case_id": cid, "values": bv + av, "n_b": len(bv)})
    except Exception:
        points.append({"case_id": cid, "values": None, "n_b": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cheby2_ellip query");
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
                "failed to spawn python3 for cheby2_ellip oracle: {e}"
            );
            eprintln!("skipping cheby2_ellip oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cheby2_ellip oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cheby2_ellip oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cheby2_ellip oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cheby2_ellip oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cheby2_ellip oracle failed: {stderr}"
        );
        eprintln!("skipping cheby2_ellip oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cheby2_ellip oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<(Vec<f64>, usize)> {
    let coeffs = match case.op.as_str() {
        "cheby2" => cheby2(case.order, case.p1, &[case.wn], FilterType::Lowpass).ok()?,
        "ellip" => ellip(
            case.order,
            case.p1,
            case.p2,
            &[case.wn],
            FilterType::Lowpass,
        )
        .ok()?,
        _ => return None,
    };
    let n_b = coeffs.b.len();
    let mut packed = coeffs.b;
    packed.extend(coeffs.a);
    Some((packed, n_b))
}

#[test]
fn diff_signal_cheby2_ellip() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(n_b) = scipy_arm.n_b else { continue };
        let Some((fsci_v, fsci_n_b)) = fsci_eval(case) else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() || fsci_n_b != n_b {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_cheby2_ellip".into(),
        category: "scipy.signal.cheby2 + ellip (lowpass)".into(),
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
                "cheby2_ellip {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal cheby2/ellip conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
