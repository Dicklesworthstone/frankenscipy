#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.f`.
//!
//! Resolves [frankenscipy-76dkp]. F-distribution has anchor
//! tests in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. 6 (dfn, dfd) pairs × 9 x-values × 2 families (pdf,
//! cdf) + 6 × 7 ppf cases via subprocess.
//!
//! pdf composes ln_beta into an exp; cdf uses the regularized
//! incomplete beta helper. Both sit on the ribeta series-
//! precision floor (~1e-12). ppf uses betaincinv + Newton
//! refinement, hold to 1e-9 rel.
//!
//! sf is intentionally omitted — fsci F-dist inherits the trait
//! default `1 - cdf` which catastrophically cancels in the
//! deep right tail. The diff is for correctness of the working
//! arms, not to verify a known-imprecise default.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, FDistribution};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    dfn: f64,
    dfd: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    dfn: f64,
    dfd: f64,
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
    fs::create_dir_all(output_dir()).expect("create f diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize f diff log");
    fs::write(path, json).expect("write f diff log");
}

fn generate_query() -> OracleQuery {
    // (dfn, dfd) pairs span small/medium/large ratios. Mode of
    // F is at (dfn-2)/(dfn) · dfd/(dfd+2) for dfn > 2; mean is
    // dfd/(dfd-2) for dfd > 2. Use multiplicative x-grid that
    // covers tail-tail span.
    let pairs: [(f64, f64); 6] = [
        (1.0, 1.0),
        (2.0, 5.0),
        (3.0, 10.0),
        (5.0, 5.0),
        (10.0, 30.0),
        (50.0, 100.0),
    ];
    let xs = [0.01_f64, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let mut points = Vec::new();
    for &(dfn, dfd) in &pairs {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("dfn{dfn}_dfd{dfd}_x{x}"),
                dfn,
                dfd,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &(dfn, dfd) in &pairs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("dfn{dfn}_dfd{dfd}_q{q}"),
                dfn,
                dfd,
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
from scipy.stats import f

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    dfn = float(case["dfn"]); dfd = float(case["dfd"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(f.pdf(x, dfn, dfd)),
            "cdf": float(f.cdf(x, dfn, dfd)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; dfn = float(case["dfn"]); dfd = float(case["dfd"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(f.ppf(qv, dfn, dfd))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize f query");
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
                "failed to spawn python3 for f oracle: {e}"
            );
            eprintln!("skipping f oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open f oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "f oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping f oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for f oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "f oracle failed: {stderr}"
        );
        eprintln!("skipping f oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse f oracle JSON"))
}

#[test]
fn diff_stats_f() {
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
        let dist = FDistribution::new(case.dfn, case.dfd);
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
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let dist = FDistribution::new(case.dfn, case.dfd);
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
        test_id: "diff_stats_f".into(),
        category: "scipy.stats.f".into(),
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
                "f {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.f conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
