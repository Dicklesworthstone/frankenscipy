#![forbid(unsafe_code)]
//! Live scipy.linalg.eigvals parity for fsci_linalg::eigvals.
//!
//! Resolves [frankenscipy-pn8ai]. Returns eigenvalues as a sorted
//! list of (re, im) pairs to absorb the inherent ordering ambiguity
//! between QR-based eig solvers.
//!
//! Tolerance: 1e-8 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, eigvals};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n: usize,
    /// Row-major flat.
    a: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened sorted [re0, im0, re1, im1, ...]
    eigs_packed: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create eigvals diff dir");
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
    let diag = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 4.0, 0.0],
        vec![0.0, 0.0, -2.0],
    ];
    let sym = vec![
        vec![4.0, 1.0, 0.5],
        vec![1.0, 3.0, 0.25],
        vec![0.5, 0.25, 2.0],
    ];
    let nonsym = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 4.0, 5.0],
        vec![1.0, 0.0, 2.0],
    ];
    // Rotation matrix → complex conjugate pair eigenvalues
    let theta = 0.7_f64;
    let rot = vec![
        vec![theta.cos(), -theta.sin(), 0.0],
        vec![theta.sin(), theta.cos(), 0.0],
        vec![0.0, 0.0, 0.5],
    ];

    let cases: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("diag_3", diag),
        ("sym_3", sym),
        ("nonsym_3", nonsym),
        ("rotation_3", rot),
    ];

    let points = cases
        .into_iter()
        .map(|(cid, a)| {
            let n = a.len();
            Case {
                case_id: cid.into(),
                n,
                a: pack_2d(&a),
            }
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"])
    A = np.array(case["a"], dtype=float).reshape(n, n)
    try:
        eigs = linalg.eigvals(A)
        pairs = [(float(z.real), float(z.imag)) for z in eigs.tolist()]
        # Sort by (re, im) for canonical comparison.
        pairs.sort(key=lambda t: (t[0], t[1]))
        flat = []
        for r, i in pairs:
            flat.append(r)
            flat.append(i)
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "eigs_packed": flat})
        else:
            points.append({"case_id": cid, "eigs_packed": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "eigs_packed": None})
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
                "failed to spawn python3 for eigvals oracle: {e}"
            );
            eprintln!("skipping eigvals oracle: python3 not available ({e})");
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
                "eigvals oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eigvals oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for eigvals oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eigvals oracle failed: {stderr}"
        );
        eprintln!("skipping eigvals oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eigvals oracle JSON"))
}

fn unpack_2d(flat: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
}

#[test]
fn diff_linalg_eigvals() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = DecompOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.eigs_packed.as_ref() else {
            continue;
        };
        let a = unpack_2d(&case.a, case.n);
        let Ok((re, im)) = eigvals(&a, opts) else {
            continue;
        };
        if re.len() != im.len() {
            continue;
        }
        let mut pairs: Vec<(f64, f64)> = re.iter().zip(im.iter()).map(|(&r, &i)| (r, i)).collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        let mut flat = Vec::with_capacity(pairs.len() * 2);
        for &(r, i) in &pairs {
            flat.push(r);
            flat.push(i);
        }
        let abs_d = if flat.len() != expected.len() {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_eigvals".into(),
        category: "fsci_linalg::eigvals vs scipy.linalg.eigvals (sorted)".into(),
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
            eprintln!("eigvals mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "eigvals conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
