#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.truncweibull_min`.
//!
//! Resolves [frankenscipy-f46xp]. The TruncWeibullMin port shipped in
//! 5557597 has 3 anchor tests but no live scipy oracle. This harness
//! drives 5 (c, a, b) shape triples × 7 x-values for pdf/cdf/sf and
//! 5 (c, a, b) × 5 q-values for ppf through scipy via subprocess at
//! tol 1e-12. Skips cleanly if scipy is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, TruncWeibullMin};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    a: f64,
    b: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    a: f64,
    b: f64,
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
    fs::create_dir_all(output_dir()).expect("create truncweibull diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize truncweibull diff log");
    fs::write(path, json).expect("write truncweibull diff log");
}

fn generate_query() -> OracleQuery {
    let triples: &[(f64, f64, f64)] = &[
        (1.0, 0.0, 5.0),
        (2.0, 0.5, 2.0),
        (0.5, 1.0, 10.0),
        (3.0, 1.0, 10.0),
        (1.5, 0.0, 3.0),
    ];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let xs_for = |a: f64, b: f64| -> Vec<f64> {
        let span = b - a;
        vec![
            a + 0.05 * span,
            a + 0.25 * span,
            a + 0.5 * span,
            a + 0.75 * span,
            a + 0.95 * span,
            a + 0.01 * span,
            a + 0.99 * span,
        ]
    };
    let mut points = Vec::new();
    for &(c, a, b) in triples {
        for x in xs_for(a, b) {
            points.push(PointCase {
                case_id: format!("c{c}_a{a}_b{b}_x{x:.4}"),
                c,
                a,
                b,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &(c, a, b) in triples {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("c{c}_a{a}_b{b}_q{q}"),
                c,
                a,
                b,
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
from scipy.stats import truncweibull_min

q = json.load(sys.stdin)
points = []
for c in q["points"]:
    cid = c["case_id"]
    cc = float(c["c"]); a = float(c["a"]); b = float(c["b"]); x = float(c["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(truncweibull_min.pdf(x, cc, a, b)),
            "cdf": float(truncweibull_min.cdf(x, cc, a, b)),
            "sf":  float(truncweibull_min.sf(x, cc, a, b)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for c in q["ppf"]:
    cid = c["case_id"]
    cc = float(c["c"]); a = float(c["a"]); b = float(c["b"]); qv = float(c["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(truncweibull_min.ppf(qv, cc, a, b))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize truncweibull query");
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
                "failed to spawn python3 for truncweibull oracle: {e}"
            );
            eprintln!("skipping truncweibull oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open truncweibull oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "truncweibull oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping truncweibull oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for truncweibull oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "truncweibull oracle failed: {stderr}"
        );
        eprintln!("skipping truncweibull oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse truncweibull oracle JSON"))
}

#[test]
fn diff_stats_truncweibull_min() {
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
        let dist = TruncWeibullMin::new(case.c, case.a, case.b);
        if let Some(spdf) = oracle.pdf {
            let d = (dist.pdf(case.x) - spdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.x) - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
        if let Some(ssf) = oracle.sf {
            let d = (dist.sf(case.x) - ssf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "sf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        if let Some(sppf) = oracle.ppf {
            let dist = TruncWeibullMin::new(case.c, case.a, case.b);
            let rust = dist.ppf(case.q);
            let d = (rust - sppf).abs();
            let scale = sppf.abs().max(1.0);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_truncweibull_min".into(),
        category: "scipy.stats.truncweibull_min".into(),
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
                "truncweibull {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.truncweibull_min conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
