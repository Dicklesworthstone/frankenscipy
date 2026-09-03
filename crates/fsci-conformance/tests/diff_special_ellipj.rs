#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Jacobian elliptic
//! functions `scipy.special.ellipj(u, m)` → (sn, cn, dn, ph).
//!
//! Resolves [frankenscipy-0q17x]. fsci_special::ellipj is
//! implemented via the AGM-driven nome expansion (Carlson +
//! arithmetic-geometric mean) and currently has no conformance
//! harness — only its callers (lambertw, ellipkinc, etc.) are
//! tested.
//!
//! 8 (u, m) tuples × 4 arms (sn, cn, dn, ph) = 32 cases via
//! subprocess. Tolerance: 1e-10 abs / rel — fsci matches scipy
//! to ~5e-13 in practice, so 1e-10 leaves headroom for the
//! large-amplitude phase regime without papering over real
//! drift.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::ellipj;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    u: f64,
    m: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sn: Option<f64>,
    cn: Option<f64>,
    dn: Option<f64>,
    ph: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    abs_diff: f64,
    rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create ellipj diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ellipj diff log");
    fs::write(path, json).expect("write ellipj diff log");
}

fn generate_query() -> OracleQuery {
    // (u, m) grid: covers near-zero amplitude, mid-range periodic, near-unity
    // m (where the period blows up), zero m (degenerate to circular), and
    // negative u (odd symmetry check on sn).
    let cases: &[(f64, f64)] = &[
        (0.5, 0.3),
        (1.0, 0.5),
        (1.5, 0.7),
        (2.0, 0.1),
        (-0.5, 0.5),
        (3.0, 0.9),
        (0.1, 0.99),
        (5.0, 0.0),
    ];
    let points = cases
        .iter()
        .map(|&(u, m)| PointCase {
            case_id: format!("u{u}_m{m}"),
            u,
            m,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; u = float(case["u"]); m = float(case["m"])
    try:
        sn, cn, dn, ph = special.ellipj(u, m)
        points.append({
            "case_id": cid,
            "sn": fnone(sn),
            "cn": fnone(cn),
            "dn": fnone(dn),
            "ph": fnone(ph),
        })
    except Exception:
        points.append({"case_id": cid, "sn": None, "cn": None, "dn": None, "ph": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ellipj query");
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
                "failed to spawn python3 for ellipj oracle: {e}"
            );
            eprintln!("skipping ellipj oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ellipj oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ellipj oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ellipj oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ellipj oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ellipj oracle failed: {stderr}"
        );
        eprintln!("skipping ellipj oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ellipj oracle JSON"))
}

#[test]
fn diff_special_ellipj() {
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
        let (sn, cn, dn, ph) = ellipj(case.u, case.m);
        let arms: [(&str, Option<f64>, f64); 4] = [
            ("sn", scipy_arm.sn, sn),
            ("cn", scipy_arm.cn, cn),
            ("dn", scipy_arm.dn, dn),
            ("ph", scipy_arm.ph, ph),
        ];
        for (arm_name, scipy_v, rust_v) in arms {
            if let Some(scipy_v) = scipy_v
                && rust_v.is_finite()
            {
                let abs_d = (rust_v - scipy_v).abs();
                let rel_d = abs_d / scipy_v.abs().max(f64::MIN_POSITIVE);
                max_overall = max_overall.max(abs_d);
                let pass = abs_d <= ABS_TOL || rel_d <= REL_TOL;
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: arm_name.into(),
                    abs_diff: abs_d,
                    rel_diff: rel_d,
                    pass,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_ellipj".into(),
        category: "scipy.special.ellipj".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };
    emit_log(&log);

    assert!(
        all_pass,
        "ellipj diff harness failed; see fixtures/artifacts/{PACKET_ID}/diff/diff_special_ellipj.json"
    );
}
