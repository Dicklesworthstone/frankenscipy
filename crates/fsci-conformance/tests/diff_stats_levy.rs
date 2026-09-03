#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.levy` and
//! `scipy.stats.levy_l` (the right- and left-skew Lévy
//! distributions).
//!
//! Resolves [frankenscipy-vw3ai]. Both fsci-stats Levy and
//! LevyLeft (parameterless) had anchor tests but no dedicated
//! scipy diff harness. 9 x-values for each variant × 3 families
//! + 5 q-values for ppf via subprocess. Skips cleanly if scipy
//!   is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Levy, LevyLeft};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    variant: String, // "levy" or "levy_l"
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    variant: String,
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
    fs::create_dir_all(output_dir()).expect("create levy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize levy diff log");
    fs::write(path, json).expect("write levy diff log");
}

fn generate_query() -> OracleQuery {
    let xs_pos = [0.1_f64, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &x in &xs_pos {
        points.push(PointCase {
            case_id: format!("levy_x{x}"),
            variant: "levy".into(),
            x,
        });
    }
    for &x in &xs_pos {
        points.push(PointCase {
            case_id: format!("levyL_x{}", -x),
            variant: "levy_l".into(),
            x: -x,
        });
    }
    let mut ppf_cases = Vec::new();
    for &q in &qs {
        ppf_cases.push(PpfCase {
            case_id: format!("levy_q{q}"),
            variant: "levy".into(),
            q,
        });
    }
    for &q in &qs {
        ppf_cases.push(PpfCase {
            case_id: format!("levyL_q{q}"),
            variant: "levy_l".into(),
            q,
        });
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
from scipy.stats import levy, levy_l

q = json.load(sys.stdin)
DISTS = {"levy": levy, "levy_l": levy_l}
points = []
for case in q["points"]:
    cid = case["case_id"]
    d = DISTS[case["variant"]]
    x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(d.pdf(x)),
            "cdf": float(d.cdf(x)),
            "sf":  float(d.sf(x)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; d = DISTS[case["variant"]]; qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(d.ppf(qv))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize levy query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for levy oracle: {e}"
            );
            eprintln!("skipping levy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open levy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "levy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping levy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for levy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "levy oracle failed: {stderr}"
        );
        eprintln!("skipping levy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse levy oracle JSON"))
}

#[test]
fn diff_stats_levy() {
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

    let levy = Levy::new(0.0, 1.0);
    let levy_l = LevyLeft::new(0.0, 1.0);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let (pdf_v, cdf_v, sf_v) = match case.variant.as_str() {
            "levy" => (levy.pdf(case.x), levy.cdf(case.x), levy.sf(case.x)),
            "levy_l" => (levy_l.pdf(case.x), levy_l.cdf(case.x), levy_l.sf(case.x)),
            other => panic!("unknown variant: {other}"),
        };
        if let Some(spdf) = oracle.pdf {
            let d = (pdf_v - spdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: format!("{}.pdf", case.variant),
                abs_diff: d,
                pass: d <= PDF_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (cdf_v - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: format!("{}.cdf", case.variant),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
        if let Some(ssf) = oracle.sf {
            let d = (sf_v - ssf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: format!("{}.sf", case.variant),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let rust = match case.variant.as_str() {
                "levy" => levy.ppf(case.q),
                "levy_l" => levy_l.ppf(case.q),
                other => panic!("unknown variant: {other}"),
            };
            let d = (rust - sppf).abs();
            let scale = sppf.abs().max(1.0);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: format!("{}.ppf", case.variant),
                abs_diff: d,
                pass: d <= PPF_TOL_REL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_levy".into(),
        category: "scipy.stats.levy + scipy.stats.levy_l".into(),
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
                "levy {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.levy/levy_l conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
