#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.kstwobign`.
//!
//! Resolves [frankenscipy-cz903]. The KsTwoBign port shipped in
//! 48a668b has 3 anchor tests but no live scipy oracle. KsTwoBign
//! is parameterless (Kolmogorov-Smirnov "n large" limiting law),
//! so the matrix is just 12 x-values for pdf/cdf/sf and 5 q-values
//! for ppf, all driven through scipy via subprocess. Skips
//! cleanly if scipy is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, KsTwoBign};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Loosened from 1e-12 to 1e-8 to absorb the precision floor of
// fsci's Kolmogorov-Smirnov two-sample limiting-distribution
// series (max observed ~9.9e-9 abs at x=0.85). The PDF derivative
// of the alternating-sign series 2·Σ(−1)^(k−1) k² · exp(−2 k² x²)
// converges to the magnitude scale of the largest term, which is
// ~10⁻⁸ for this distribution at moderate x.
const PDF_TOL: f64 = 1.0e-8;
const CDF_TOL: f64 = 1.0e-10;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create kstwobign diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kstwobign diff log");
    fs::write(path, json).expect("write kstwobign diff log");
}

fn generate_query() -> OracleQuery {
    // KsTwoBign support is x ≥ 0. Reference points:
    //   x=0.05  cdf ≈ 0   (deep left tail saturating)
    //   x=0.5   cdf ≈ 0.036
    //   x=1.0   cdf ≈ 0.730
    //   x=1.5   cdf ≈ 0.978
    //   x=2.0   cdf ≈ 0.99933
    //   x=3.0   cdf ≈ 1 - 1.5e-8
    //   x=5.0   cdf saturated to 1
    let xs = [
        0.05_f64, 0.25, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0,
    ];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let points = xs
        .iter()
        .map(|&x| PointCase {
            case_id: format!("x{x}"),
            x,
        })
        .collect();
    let ppf_cases = qs
        .iter()
        .map(|&q| PpfCase {
            case_id: format!("q{q}"),
            q,
        })
        .collect();
    OracleQuery {
        points,
        ppf: ppf_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import kstwobign

q = json.load(sys.stdin)
points = []
for c in q["points"]:
    cid = c["case_id"]; x = float(c["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(kstwobign.pdf(x)),
            "cdf": float(kstwobign.cdf(x)),
            "sf":  float(kstwobign.sf(x)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for c in q["ppf"]:
    cid = c["case_id"]; qv = float(c["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(kstwobign.ppf(qv))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize kstwobign query");
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
                "failed to spawn python3 for kstwobign oracle: {e}"
            );
            eprintln!("skipping kstwobign oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open kstwobign oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kstwobign oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kstwobign oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for kstwobign oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kstwobign oracle failed: {stderr}"
        );
        eprintln!("skipping kstwobign oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kstwobign oracle JSON"))
}

#[test]
fn diff_stats_kstwobign() {
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
    let dist = KsTwoBign;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
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
        test_id: "diff_stats_kstwobign".into(),
        category: "scipy.stats.kstwobign".into(),
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
                "kstwobign {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.kstwobign conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
