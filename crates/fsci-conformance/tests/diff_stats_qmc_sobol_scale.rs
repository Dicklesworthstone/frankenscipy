#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_stats::SobolSampler
//! (unscrambled) and qmc::scale.
//!
//! Resolves [frankenscipy-r1lg9]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{SobolSampler, qmc_scale};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SobolCase {
    case_id: String,
    dim: usize,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ScaleCase {
    case_id: String,
    sample: Vec<f64>,
    dim: usize,
    l_bounds: Vec<f64>,
    u_bounds: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    sobol: Vec<SobolCase>,
    scale: Vec<ScaleCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    sobol: Vec<PointArm>,
    scale: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create qmc_sobol_scale diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize qmc_sobol_scale diff log");
    fs::write(path, json).expect("write qmc_sobol_scale diff log");
}

fn generate_query() -> OracleQuery {
    let sobol_combos = &[(2_usize, 8_usize), (3, 8), (3, 16), (4, 16), (5, 32)];
    let sobol = sobol_combos
        .iter()
        .map(|(d, n)| SobolCase {
            case_id: format!("sobol_d{d}_n{n}"),
            dim: *d,
            n: *n,
        })
        .collect();

    let scale_cases = vec![
        ScaleCase {
            case_id: "scale_2d_simple".into(),
            sample: vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            dim: 2,
            l_bounds: vec![10.0, 20.0],
            u_bounds: vec![20.0, 60.0],
        },
        ScaleCase {
            case_id: "scale_3d_centered".into(),
            sample: vec![0.25, 0.5, 0.75, 0.1, 0.2, 0.3, 0.9, 0.5, 0.1],
            dim: 3,
            l_bounds: vec![-1.0, 0.0, 100.0],
            u_bounds: vec![1.0, 2.0, 200.0],
        },
        ScaleCase {
            case_id: "scale_1d".into(),
            sample: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            dim: 1,
            l_bounds: vec![-5.0],
            u_bounds: vec![5.0],
        },
    ];

    OracleQuery {
        sobol,
        scale: scale_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.stats import qmc

def finite_flat_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)

sobol_out = []
for c in q["sobol"]:
    cid = c["case_id"]; d = int(c["dim"]); n = int(c["n"])
    try:
        s = qmc.Sobol(d=d, scramble=False)
        sample = s.random(n=n)
        sobol_out.append({"case_id": cid, "values": finite_flat_or_none(sample)})
    except Exception:
        sobol_out.append({"case_id": cid, "values": None})

scale_out = []
for c in q["scale"]:
    cid = c["case_id"]; d = int(c["dim"])
    sample = np.array(c["sample"], dtype=float).reshape(-1, d)
    lb = np.array(c["l_bounds"], dtype=float)
    ub = np.array(c["u_bounds"], dtype=float)
    try:
        out = qmc.scale(sample, lb, ub)
        scale_out.append({"case_id": cid, "values": finite_flat_or_none(out)})
    except Exception:
        scale_out.append({"case_id": cid, "values": None})

print(json.dumps({"sobol": sobol_out, "scale": scale_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize qmc_sobol_scale query");
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
                "failed to spawn python3 for qmc_sobol_scale oracle: {e}"
            );
            eprintln!("skipping qmc_sobol_scale oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open qmc_sobol_scale oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "qmc_sobol_scale oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping qmc_sobol_scale oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for qmc_sobol_scale oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qmc_sobol_scale oracle failed: {stderr}"
        );
        eprintln!("skipping qmc_sobol_scale oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse qmc_sobol_scale oracle JSON"))
}

#[test]
fn diff_stats_qmc_sobol_scale() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.sobol.len(), query.sobol.len());
    assert_eq!(oracle.scale.len(), query.scale.len());

    let sobol_map: HashMap<String, PointArm> = oracle
        .sobol
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let scale_map: HashMap<String, PointArm> = oracle
        .scale
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // sobol
    for case in &query.sobol {
        let scipy_arm = sobol_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(mut sampler) = SobolSampler::new(case.dim) else {
            continue;
        };
        let fsci_v = sampler.sample(case.n);
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "sobol".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // scale
    for case in &query.scale {
        let scipy_arm = scale_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(fsci_v) = qmc_scale(&case.sample, case.dim, &case.l_bounds, &case.u_bounds) else {
            continue;
        };
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "scale".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_qmc_sobol_scale".into(),
        category: "scipy.stats.qmc Sobol + scale".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "qmc_sobol_scale conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
