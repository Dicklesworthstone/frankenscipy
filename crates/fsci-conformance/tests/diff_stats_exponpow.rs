#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.exponpow`.
//!
//! Resolves [frankenscipy-wepky]. The ExponPow port shipped in
//! 3efd9f9 has 3 closed-form anchor cases but no live scipy oracle.
//! This harness drives 5 b-values × 7 x-values for pdf/cdf/sf and
//! 5 b-values × 5 q-values for ppf through scipy via subprocess
//! and asserts byte-stable agreement at tol 1e-12 (1e-9 for sf in
//! the saturated tail). Skips cleanly if scipy/python3 is
//! unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, ExponPow};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL_TIGHT: f64 = 1.0e-12;
const ABS_TOL_LOOSE: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct ExpPowCase {
    case_id: String,
    b: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    b: f64,
    q: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    pdf_cdf_sf_cases: Vec<ExpPowCase>,
    ppf_cases: Vec<PpfCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PdfCdfSfArm {
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
    pdf_cdf_sf: Vec<PdfCdfSfArm>,
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
    fs::create_dir_all(output_dir()).expect("create exponpow diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize exponpow diff log");
    fs::write(path, json).expect("write exponpow diff log");
}

fn generate_query() -> OracleQuery {
    let bs = [0.5_f64, 1.0, 1.5, 2.0, 3.5];
    let xs = [0.05_f64, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut pdf_cdf_sf_cases = Vec::new();
    for &b in &bs {
        for &x in &xs {
            pdf_cdf_sf_cases.push(ExpPowCase {
                case_id: format!("b{b}_x{x}"),
                b,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &b in &bs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("b{b}_q{q}"),
                b,
                q,
            });
        }
    }
    OracleQuery {
        pdf_cdf_sf_cases,
        ppf_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import exponpow

q = json.load(sys.stdin)
pdfcdfsf = []
for c in q["pdf_cdf_sf_cases"]:
    cid = c["case_id"]
    b = float(c["b"])
    x = float(c["x"])
    try:
        pdfcdfsf.append({
            "case_id": cid,
            "pdf": float(exponpow.pdf(x, b)),
            "cdf": float(exponpow.cdf(x, b)),
            "sf":  float(exponpow.sf(x, b)),
        })
    except Exception:
        pdfcdfsf.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for c in q["ppf_cases"]:
    cid = c["case_id"]
    b = float(c["b"])
    qv = float(c["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(exponpow.ppf(qv, b))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"pdf_cdf_sf": pdfcdfsf, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize exponpow query");
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
                "failed to spawn python3 for exponpow oracle: {e}"
            );
            eprintln!("skipping exponpow oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open exponpow oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "exponpow oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping exponpow oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for exponpow oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "exponpow oracle failed: {stderr}"
        );
        eprintln!("skipping exponpow oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse exponpow oracle JSON"))
}

#[test]
fn diff_stats_exponpow() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    assert_eq!(oracle.pdf_cdf_sf.len(), query.pdf_cdf_sf_cases.len());
    assert_eq!(oracle.ppf.len(), query.ppf_cases.len());

    let pdf_map: HashMap<String, PdfCdfSfArm> = oracle
        .pdf_cdf_sf
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let ppf_map: HashMap<String, PpfArm> = oracle
        .ppf
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.pdf_cdf_sf_cases {
        let oracle = pdf_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let dist = ExponPow::new(case.b);
        if let Some(scipy_pdf) = oracle.pdf {
            let d = (dist.pdf(case.x) - scipy_pdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL_TIGHT,
            });
        }
        if let Some(scipy_cdf) = oracle.cdf {
            let d = (dist.cdf(case.x) - scipy_cdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL_TIGHT,
            });
        }
        if let Some(scipy_sf) = oracle.sf {
            let d = (dist.sf(case.x) - scipy_sf).abs();
            max_overall = max_overall.max(d);
            // sf can saturate near 0 in deep tail — looser tol.
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "sf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL_LOOSE,
            });
        }
    }

    for case in &query.ppf_cases {
        let oracle = ppf_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        if let Some(scipy_ppf) = oracle.ppf {
            let dist = ExponPow::new(case.b);
            let rust_ppf = dist.ppf(case.q);
            let d = (rust_ppf - scipy_ppf).abs();
            let scale = scipy_ppf.abs().max(1.0);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "ppf".into(),
                abs_diff: d,
                pass: d <= ABS_TOL_TIGHT * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_exponpow".into(),
        category: "scipy.stats.exponpow".into(),
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
                "exponpow {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.exponpow conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
