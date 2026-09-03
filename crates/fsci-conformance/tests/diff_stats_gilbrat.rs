#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_stats::Gilbrat`.
//!
//! Resolves [frankenscipy-t2qia]. `scipy.stats.gilbrat` was removed
//! in scipy 1.17 in favour of the canonical one-arg lognormal
//! `scipy.stats.lognorm(s=1, scale=1)`. The Gilbrat struct is kept
//! in fsci-stats for backwards compatibility, so the conformance
//! oracle here uses lognorm with s=1 — by definition the same
//! distribution.
//!
//! 9 x-values × 3 families (pdf, cdf, sf) + 7 ppf cases via subprocess.
//! pdf/cdf/sf use closed forms (Φ(ln x), exp(-(ln x)²/2)/(x√2π))
//! ppf uses ndtri, all routed through `fsci_special` so we hold to 1e-12.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Gilbrat};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const SF_TOL: f64 = 1.0e-12;
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
    fs::create_dir_all(output_dir()).expect("create gilbrat diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gilbrat diff log");
    fs::write(path, json).expect("write gilbrat diff log");
}

fn generate_query() -> OracleQuery {
    // Gilbrat support is (0, ∞). Walk x across 9 decades around the
    // median (= 1) to exercise the small-x tail (where pdf is small)
    // and the large-x tail (where the log-normal fall-off matters).
    let xs = [0.01_f64, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 20.0];
    // q grid spans the body and both tails.
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];

    let points = xs
        .iter()
        .map(|&x| PointCase {
            case_id: format!("x{x}"),
            x,
        })
        .collect();
    let ppf = qs
        .iter()
        .map(|&q| PpfCase {
            case_id: format!("q{q}"),
            q,
        })
        .collect();
    OracleQuery { points, ppf }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    // scipy.stats.gilbrat was removed in scipy 1.17 — use the
    // canonical one-arg lognormal replacement (s=1, scale=1).
    let script = r#"
import json
import sys
from scipy.stats import lognorm

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(lognorm.pdf(x, 1.0)),
            "cdf": float(lognorm.cdf(x, 1.0)),
            "sf":  float(lognorm.sf(x, 1.0)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(lognorm.ppf(qv, 1.0))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize gilbrat query");
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
                "failed to spawn python3 for gilbrat oracle: {e}"
            );
            eprintln!("skipping gilbrat oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open gilbrat oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gilbrat oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gilbrat oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for gilbrat oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gilbrat oracle failed: {stderr}"
        );
        eprintln!("skipping gilbrat oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gilbrat oracle JSON"))
}

#[test]
fn diff_stats_gilbrat() {
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
    let dist = Gilbrat;

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
                pass: d <= SF_TOL,
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
        test_id: "diff_stats_gilbrat".into(),
        category: "scipy.stats.lognorm(s=1) ≡ gilbrat".into(),
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
                "gilbrat {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.lognorm(s=1) conformance failed for fsci_stats::Gilbrat: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
