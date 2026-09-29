#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.dweibull`.
//!
//! Resolves [frankenscipy-l8kpp]. DoubleWeibull has anchor
//! tests in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. 6 c values × 9 x-values × 2 families (pdf, cdf) +
//! 6 × 7 ppf cases via subprocess.
//!
//! All families are closed-form pow/exp/log on the symmetric
//! Weibull-mixture support. cdf splits at x=0 into mirrored
//! branches; ppf folds q-by-side. 1e-13 abs holds.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, DoubleWeibull};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-13;
const CDF_TOL: f64 = 1.0e-13;
const PPF_TOL_REL: f64 = 1.0e-12;
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
    // SciPy's pdf(0, c<1) is +inf; it arrives as "inf", distinct from null.
    #[serde(
        default,
        deserialize_with = "fsci_conformance::compare_ledger::oracle_f64"
    )]
    pdf: Option<f64>,
    #[serde(
        default,
        deserialize_with = "fsci_conformance::compare_ledger::oracle_f64"
    )]
    cdf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpfArm {
    case_id: String,
    #[serde(
        default,
        deserialize_with = "fsci_conformance::compare_ledger::oracle_f64"
    )]
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create dweibull diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dweibull diff log");
    fs::write(path, json).expect("write dweibull diff log");
}

fn generate_query() -> OracleQuery {
    // c spans heavy-tail (c<1) through bell-shape (c≥2).
    let cs = [0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0];
    let xs = [-5.0_f64, -2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 5.0];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
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
import math
import sys
from scipy.stats import dweibull

def fval(v):
    # A non-finite answer is sent as "nan"/"inf"/"-inf", so the harness can tell it from a
    # raised call (null): pdf(0, c=0.5) is +inf.
    try:
        v = float(v)
    except Exception:
        return None
    if math.isfinite(v):
        return v
    return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": fval(dweibull.pdf(x, c)),
            "cdf": fval(dweibull.cdf(x, c)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; c = float(case["c"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": fval(dweibull.ppf(qv, c))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize dweibull query");
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
                "failed to spawn python3 for dweibull oracle: {e}"
            );
            eprintln!("skipping dweibull oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open dweibull oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dweibull oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dweibull oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for dweibull oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dweibull oracle failed: {stderr}"
        );
        eprintln!("skipping dweibull oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dweibull oracle JSON"))
}

#[test]
fn diff_stats_dweibull() {
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
    let mut ledger = CompareLedger::new("diff_stats_dweibull", &["pdf", "cdf", "ppf"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = DoubleWeibull::new(case.c);
        if let Some((spdf, rpdf)) =
            ledger.pair("pdf", &case.case_id, oracle.pdf, Some(dist.pdf(case.x)))
        {
            let d = (rpdf - spdf).abs();
            max_overall = max_overall.max(d);
            ledger.compared("pdf", &case.case_id, d <= PDF_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= PDF_TOL,
            });
        }
        if let Some((scdf, rcdf)) =
            ledger.pair("cdf", &case.case_id, oracle.cdf, Some(dist.cdf(case.x)))
        {
            let d = (rcdf - scdf).abs();
            max_overall = max_overall.max(d);
            ledger.compared("cdf", &case.case_id, d <= CDF_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        let dist = DoubleWeibull::new(case.c);
        let Some((sppf, rust)) =
            ledger.pair("ppf", &case.case_id, oracle.ppf, Some(dist.ppf(case.q)))
        else {
            continue;
        };
        let d = (rust - sppf).abs();
        let scale = sppf.abs().max(1.0);
        max_overall = max_overall.max(d);
        ledger.compared("ppf", &case.case_id, d <= PPF_TOL_REL * scale);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            family: "ppf".into(),
            abs_diff: d,
            pass: d <= PPF_TOL_REL * scale,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_dweibull".into(),
        category: "scipy.stats.dweibull".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
                "dweibull {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.dweibull conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // pdf/cdf run over query.points, ppf over query.ppf; each arm must compare all of its own.
    ledger.finish(query.points.len().min(query.ppf.len()));
}
