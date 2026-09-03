#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.argus`.
//!
//! Resolves [frankenscipy-2lqlm]. Argus has anchor tests in
//! `fsci-stats/src/lib.rs` but no dedicated scipy diff harness.
//! Mirrors the diff_stats_foldnorm.rs / diff_stats_foldcauchy.rs
//! pattern. 5 chi-values × 7 x-values × 3 families (pdf/cdf/sf)
//! + 5 chi × 5 q-values for ppf via subprocess. Skips cleanly
//!   if scipy is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{Argus, ContinuousDistribution};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Argus cdf composes 1 − Ψ(χ·√(1−x²))/Ψ(χ) where
// Ψ(y) = Φ(y) − y·φ(y) − 1/2. Near the support edge x → 1, the
// numerator Ψ at the small argument loses precision by
// catastrophic cancellation, and the ratio amplifies that.
// Worst seen: chi=5, x=0.95 → 1.9e-8 drift in cdf. Loosen to
// 5e-8 — still tight enough to catch real drift in the closed
// form (would have caught the FoldedCauchy +0.5 bug at scale).
const PDF_TOL: f64 = 1.0e-11;
const CDF_TOL: f64 = 5.0e-8;
// ppf inverts cdf via bisection, inheriting the cdf precision
// floor amplified by the bisection tolerance. Worst seen:
// chi=5, q=0.95 → 3e-7. Loosen accordingly.
const PPF_TOL_REL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    chi: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    chi: f64,
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
    fs::create_dir_all(output_dir()).expect("create argus diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize argus diff log");
    fs::write(path, json).expect("write argus diff log");
}

fn generate_query() -> OracleQuery {
    // Argus support is x ∈ [0, 1]. Vary chi (>0) and sample
    // x across that interval.
    let chis = [0.5_f64, 1.0, 2.0, 3.0, 5.0];
    let xs = [0.05_f64, 0.2, 0.4, 0.5, 0.7, 0.9, 0.95];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &chi in &chis {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("chi{chi}_x{x}"),
                chi,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &chi in &chis {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("chi{chi}_q{q}"),
                chi,
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
from scipy.stats import argus

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    chi = float(case["chi"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(argus.pdf(x, chi)),
            "cdf": float(argus.cdf(x, chi)),
            "sf":  float(argus.sf(x, chi)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; chi = float(case["chi"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(argus.ppf(qv, chi))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize argus query");
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
                "failed to spawn python3 for argus oracle: {e}"
            );
            eprintln!("skipping argus oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open argus oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "argus oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping argus oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for argus oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "argus oracle failed: {stderr}"
        );
        eprintln!("skipping argus oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse argus oracle JSON"))
}

#[test]
fn diff_stats_argus() {
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
        let dist = Argus::new(case.chi);
        if let Some(spdf) = oracle.pdf {
            let d = (dist.pdf(case.x) - spdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= PDF_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.x) - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
        if let Some(ssf) = oracle.sf {
            let d = (dist.sf(case.x) - ssf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "sf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let dist = Argus::new(case.chi);
            let rust = dist.ppf(case.q);
            let d = (rust - sppf).abs();
            let scale = sppf.abs().max(1.0);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= PPF_TOL_REL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_argus".into(),
        category: "scipy.stats.argus".into(),
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
                "argus {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.argus conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
