#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.fisk` (the
//! log-logistic distribution).
//!
//! Resolves [frankenscipy-jlh96]. fsci-stats has TWO structs
//! that match scipy.stats.fisk: `Fisk` and `Loglogistic`. They
//! are algebraically identical (same closed forms with shape
//! parameter c). Neither had a dedicated diff harness.
//!
//! This harness drives 5 c-values × 7 x-values × 3 families
//! through scipy.stats.fisk and pins **both** fsci structs
//! against the oracle. If Fisk and Loglogistic ever drift
//! apart in maintenance, this catches it.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Fisk, Loglogistic};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    q: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    ppf: Vec<PpfCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
    sf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpfArm {
    case_id: String,
    ppf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    ppf: Vec<PpfArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    family: String,
    fsci_struct: String,
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
    fs::create_dir_all(output_dir()).expect("create fisk diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fisk diff log");
    fs::write(path, json).expect("write fisk diff log");
}

fn generate_query() -> OracleQuery {
    let cs = [0.5_f64, 1.0, 1.5, 2.0, 5.0];
    let xs = [0.05_f64, 0.3, 0.7, 1.0, 1.5, 3.0, 10.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &c in &cs {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("c{c}_x{x}"),
                c,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &c in &cs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("c{c}_q{q}"),
                c,
                q,
            });
        }
    }
    OracleQuery {
        points,
        ppf: ppf_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import fisk

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(fisk.pdf(x, c)),
            "cdf": float(fisk.cdf(x, c)),
            "sf":  float(fisk.sf(x, c)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; c = float(case["c"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(fisk.ppf(qv, c))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize fisk query");
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
                "failed to spawn python3 for fisk oracle: {e}"
            );
            eprintln!("skipping fisk oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fisk oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fisk oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fisk oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for fisk oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fisk oracle failed: {stderr}"
        );
        eprintln!("skipping fisk oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fisk oracle JSON"))
}

#[test]
fn diff_stats_fisk() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(oracle.ppf.len(), query.ppf.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let ppfmap: HashMap<String, PpfArm> = oracle
        .ppf
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let fisk = Fisk::new(case.c);
        let loglog = Loglogistic::new(case.c);

        // Tuples of (struct_name, pdf, cdf, sf) to drive both
        // implementations through the same oracle comparison.
        let triplets: [(&str, f64, f64, f64); 2] = [
            ("Fisk", fisk.pdf(case.x), fisk.cdf(case.x), fisk.sf(case.x)),
            (
                "Loglogistic",
                loglog.pdf(case.x),
                loglog.cdf(case.x),
                loglog.sf(case.x),
            ),
        ];

        for (struct_name, pdf_v, cdf_v, sf_v) in triplets {
            if let Some(spdf) = oracle.pdf {
                let d = (pdf_v - spdf).abs();
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: "pdf".into(),
                    fsci_struct: struct_name.into(),
                    abs_diff: d,
                    pass: d <= PDF_TOL,
                });
            }
            if let Some(scdf) = oracle.cdf {
                let d = (cdf_v - scdf).abs();
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: "cdf".into(),
                    fsci_struct: struct_name.into(),
                    abs_diff: d,
                    pass: d <= CDF_TOL,
                });
            }
            if let Some(ssf) = oracle.sf {
                let d = (sf_v - ssf).abs();
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: "sf".into(),
                    fsci_struct: struct_name.into(),
                    abs_diff: d,
                    pass: d <= CDF_TOL,
                });
            }
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            for (struct_name, val) in [
                ("Fisk", Fisk::new(case.c).ppf(case.q)),
                ("Loglogistic", Loglogistic::new(case.c).ppf(case.q)),
            ] {
                let d = (val - sppf).abs();
                let scale = sppf.abs().max(1.0);
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    family: "ppf".into(),
                    fsci_struct: struct_name.into(),
                    abs_diff: d,
                    pass: d <= PPF_TOL_REL * scale,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_fisk".into(),
        category: "scipy.stats.fisk".into(),
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
                "fisk[{}] {} mismatch: {} abs_diff={}",
                d.fsci_struct, d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.fisk conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
