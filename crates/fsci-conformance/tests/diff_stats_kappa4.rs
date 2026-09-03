#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.kappa4`.
//!
//! Resolves [frankenscipy-te545]. Kappa4 has anchor tests in
//! `fsci-stats/src/lib.rs` but no dedicated scipy diff harness.
//! 6 (h, k) tuples × 7 x-values × 3 families + 6 × 5 ppf cases
//! via subprocess.
//!
//! Kappa4 generalizes Gumbel/GenExtreme/GenLogistic/Logistic
//! depending on (h, k) — h, k = 0 ⇒ Gumbel; h = 0 ⇒ GenExtreme;
//! h = 1 ⇒ GenPareto-like; etc.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Kappa4};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-11;
const CDF_TOL: f64 = 1.0e-11;
const PPF_TOL_REL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    h: f64,
    k: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    h: f64,
    k: f64,
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
    fs::create_dir_all(output_dir()).expect("create kappa4 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kappa4 diff log");
    fs::write(path, json).expect("write kappa4 diff log");
}

fn generate_query() -> OracleQuery {
    // Pin (h, k) tuples that span the Gumbel / GenExtreme /
    // GenLogistic / GenPareto reductions plus general
    // off-axis combinations. Sample x from middle/tails of the
    // ppf range so we stay inside whatever support the (h, k)
    // pair imposes.
    let pairs: [(f64, f64); 6] = [
        (0.0, 0.0),   // Gumbel
        (0.0, 0.5),   // GenExtreme-like
        (1.0, 0.5),   // bounded both sides
        (-1.0, -0.5), // both unbounded, heavier
        (0.5, 0.2),   // moderate
        (0.5, -0.5),  // moderate, opposite skew
    ];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    let inner_qs = [0.1_f64, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9];
    for &(h, k) in &pairs {
        let dist = Kappa4::new(h, k);
        for (idx, &iq) in inner_qs.iter().enumerate() {
            let x = dist.ppf(iq);
            points.push(PointCase {
                case_id: format!("h{h}_k{k}_idx{idx}"),
                h,
                k,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &(h, k) in &pairs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("h{h}_k{k}_q{q}"),
                h,
                k,
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
from scipy.stats import kappa4

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    h = float(case["h"]); k = float(case["k"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(kappa4.pdf(x, h, k)),
            "cdf": float(kappa4.cdf(x, h, k)),
            "sf":  float(kappa4.sf(x, h, k)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; h = float(case["h"]); k = float(case["k"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(kappa4.ppf(qv, h, k))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize kappa4 query");
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
                "failed to spawn python3 for kappa4 oracle: {e}"
            );
            eprintln!("skipping kappa4 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open kappa4 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kappa4 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kappa4 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for kappa4 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kappa4 oracle failed: {stderr}"
        );
        eprintln!("skipping kappa4 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kappa4 oracle JSON"))
}

#[test]
fn diff_stats_kappa4() {
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
        let dist = Kappa4::new(case.h, case.k);
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
            let dist = Kappa4::new(case.h, case.k);
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
        test_id: "diff_stats_kappa4".into(),
        category: "scipy.stats.kappa4".into(),
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
                "kappa4 {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.kappa4 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
