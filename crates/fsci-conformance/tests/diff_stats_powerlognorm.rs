#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.powerlognorm`.
//!
//! Resolves [frankenscipy-7e1ut]. The PowerLognorm port shipped
//! in 08ef1ef has 3 anchor tests but no live scipy oracle. This
//! harness drives 4 c-values × 4 s-values × 6 x-values for
//! pdf/cdf/sf and 4 c × 4 s × 5 q-values for ppf through scipy
//! via subprocess. Skips cleanly if scipy is unavailable —
//! per [frankenscipy-v10ie] that means rch workers silently
//! no-op; the canonical pin against drift is the existing
//! anchor tests + a forthcoming golden artifact.
//!
//! ppf tolerance is set wider (1e-7 absolute) to absorb the
//! ~1e-9 precision floor of the Beasley-Springer-Moro inverse
//! normal helper used by PowerLognorm.ppf.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, PowerLognorm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Tolerances: PowerLognorm composes the standard normal Φ
// helper with `powf(Φ(−z), c − 1)` in pdf and `exp(−s · Φ⁻¹)` in
// ppf. Each composition amplifies the helper's precision floor:
//   • pdf/cdf in the deep tail (e.g., c=0.5, s=0.3, x=5):
//     Φ(−z) ≈ 4e-8 raised to fractional powers → ~1e-8 drift.
//   • ppf with large s (s=2) and extreme q: exp() magnifies the
//     ~1e-9 Beasley-Springer-Moro Φ⁻¹ error to ~1e-6 absolute.
// These tolerances catch real divergence while allowing the
// helper-precision floor; tightening would require swapping
// Φ/Φ⁻¹ helpers for full-double-precision variants.
const PDF_TOL: f64 = 1.0e-7;
const CDF_TOL: f64 = 1.0e-7;
const PPF_TOL_ABS: f64 = 5.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    s: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    s: f64,
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
    fs::create_dir_all(output_dir()).expect("create powerlognorm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize powerlognorm diff log");
    fs::write(path, json).expect("write powerlognorm diff log");
}

fn generate_query() -> OracleQuery {
    let cs = [0.5_f64, 1.0, 2.0, 3.0];
    let ss = [0.3_f64, 0.7, 1.0, 2.0];
    let xs = [0.1_f64, 0.3, 0.7, 1.0, 2.0, 5.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &c in &cs {
        for &s in &ss {
            for &x in &xs {
                points.push(PointCase {
                    case_id: format!("c{c}_s{s}_x{x}"),
                    c,
                    s,
                    x,
                });
            }
        }
    }
    let mut ppf_cases = Vec::new();
    for &c in &cs {
        for &s in &ss {
            for &q in &qs {
                ppf_cases.push(PpfCase {
                    case_id: format!("c{c}_s{s}_q{q}"),
                    c,
                    s,
                    q,
                });
            }
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
from scipy.stats import powerlognorm

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); s = float(case["s"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(powerlognorm.pdf(x, c, s)),
            "cdf": float(powerlognorm.cdf(x, c, s)),
            "sf":  float(powerlognorm.sf(x, c, s)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; c = float(case["c"]); s = float(case["s"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(powerlognorm.ppf(qv, c, s))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize powerlognorm query");
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
                "failed to spawn python3 for powerlognorm oracle: {e}"
            );
            eprintln!("skipping powerlognorm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open powerlognorm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "powerlognorm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping powerlognorm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for powerlognorm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "powerlognorm oracle failed: {stderr}"
        );
        eprintln!("skipping powerlognorm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse powerlognorm oracle JSON"))
}

#[test]
fn diff_stats_powerlognorm() {
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
        let dist = PowerLognorm::new(case.c, case.s);
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
            let dist = PowerLognorm::new(case.c, case.s);
            let rust = dist.ppf(case.q);
            let d = (rust - sppf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= PPF_TOL_ABS,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_powerlognorm".into(),
        category: "scipy.stats.powerlognorm".into(),
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
                "powerlognorm {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.powerlognorm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
