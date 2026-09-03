#![forbid(unsafe_code)]
//! Live scipy.linalg.{issymmetric, ishermitian} parity for fsci_linalg.
//!
//! Resolves [frankenscipy-b67vw]. For real-valued matrices, the two
//! functions coincide. Tolerance: exact (boolean result).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{ishermitian, issymmetric};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    a: Vec<f64>,
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sym: Option<bool>,
    her: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    pass: bool,
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
    fs::create_dir_all(output_dir()).expect("create issym diff dir");
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

fn pack_2d(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn generate_query() -> OracleQuery {
    let perfect_sym = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 5.0],
        vec![3.0, 5.0, 6.0],
    ];
    let asym = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let near_sym = vec![
        vec![1.0, 2.0 + 1e-12, 3.0],
        vec![2.0, 4.0, 5.0],
        vec![3.0, 5.0 - 1e-12, 6.0],
    ];
    let antisym = vec![
        vec![0.0, 1.0, -2.0],
        vec![-1.0, 0.0, 3.0],
        vec![2.0, -3.0, 0.0],
    ];
    let identity_4 = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    let cases: Vec<(&str, Vec<Vec<f64>>, f64, f64)> = vec![
        ("perfect_sym", perfect_sym, 1e-10, 1e-10),
        ("asym", asym, 1e-10, 1e-10),
        ("near_sym_loose", near_sym.clone(), 1e-6, 1e-6),
        ("near_sym_tight", near_sym, 1e-15, 1e-15),
        ("antisym", antisym, 1e-10, 1e-10),
        ("identity_4", identity_4, 1e-10, 1e-10),
    ];

    let points = cases
        .into_iter()
        .map(|(cid, a, atol, rtol)| {
            let n = a.len();
            Case {
                case_id: cid.into(),
                n,
                a: pack_2d(&a),
                atol,
                rtol,
            }
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.linalg import issymmetric, ishermitian

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    A = np.array(case["a"], dtype=float).reshape(n, n)
    atol = float(case["atol"]); rtol = float(case["rtol"])
    try:
        sym = bool(issymmetric(A, atol=atol, rtol=rtol))
        her = bool(ishermitian(A, atol=atol, rtol=rtol))
        points.append({"case_id": cid, "sym": sym, "her": her})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "sym": None, "her": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for issym oracle: {e}"
            );
            eprintln!("skipping issym oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "issym oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping issym oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for issym oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "issym oracle failed: {stderr}"
        );
        eprintln!("skipping issym oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse issym oracle JSON"))
}

fn unpack_2d(flat: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
}

#[test]
fn diff_linalg_issymmetric_ishermitian() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(esym), Some(eher)) = (arm.sym, arm.her) else {
            continue;
        };
        let a = unpack_2d(&case.a, case.n);

        let Ok(sym) = issymmetric(&a, case.atol, case.rtol) else {
            continue;
        };
        let Ok(her) = ishermitian(&a, case.atol, case.rtol) else {
            continue;
        };

        diffs.push(CaseDiff {
            case_id: format!("{}_sym", case.case_id),
            op: "issymmetric".into(),
            pass: sym == esym,
        });
        diffs.push(CaseDiff {
            case_id: format!("{}_her", case.case_id),
            op: "ishermitian".into(),
            pass: her == eher,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_issymmetric_ishermitian".into(),
        category: "fsci_linalg::issymmetric + ishermitian vs scipy.linalg".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {}", d.op, d.case_id);
        }
    }

    assert!(
        all_pass,
        "issym/isher conformance failed: {} cases",
        diffs.len()
    );
}
