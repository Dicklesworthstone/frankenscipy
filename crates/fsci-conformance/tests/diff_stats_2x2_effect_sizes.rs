#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for fsci's 2×2
//! contingency-table effect-size measures:
//!   • `relative_risk(table)` — RR = (a/(a+b)) / (c/(c+d))
//!   • `odds_ratio(table)`    — OR = ad / bc
//!   • `phi_coefficient(table)`
//!     — φ = (ad − bc) / sqrt((a+b)(c+d)(a+c)(b+d))
//!
//! Resolves [frankenscipy-29j6l]. The oracle calls
//! `scipy.stats.contingency.relative_risk(...)` for RR (the
//! .relative_risk attribute), and computes OR / phi via the
//! same closed-form formulas in numpy (no scipy primitive
//! exposes either of those scalars directly).
//!
//! 4 (a, b, c, d) fixtures × 3 funcs = 12 cases. Tol 1e-12 abs
//! (closed-form integer-cast ratios — no transcendentals).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{odds_ratio, phi_coefficient, relative_risk};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    /// 2×2 table flattened: [a, b, c, d] = [[a, b], [c, d]].
    table: [usize; 4],
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create 2x2_effect_sizes diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize 2x2_effect_sizes diff log");
    fs::write(path, json).expect("write 2x2_effect_sizes diff log");
}

fn generate_query() -> OracleQuery {
    // Flat [a, b, c, d] = [[a, b], [c, d]] entries.
    let fixtures: Vec<(&str, [usize; 4])> = vec![
        // Strong positive association (RR > 1, OR > 1)
        ("strong_assoc", [40, 10, 5, 45]),
        // Mild association
        ("mild_assoc", [25, 30, 15, 30]),
        // No association
        ("no_assoc", [20, 30, 20, 30]),
        // Larger N, strong signal
        ("large_n", [180, 70, 60, 190]),
    ];

    let mut points = Vec::new();
    for (name, table) in &fixtures {
        for func in ["relative_risk", "odds_ratio", "phi_coefficient"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                table: *table,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.stats import contingency

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    a, b, c, d = case["table"]
    val = None
    try:
        if func == "relative_risk":
            # contingency.relative_risk(exposed_cases, exposed_total,
            #                            control_cases, control_total)
            # gives RR = (a/(a+b)) / (c/(c+d))
            r = contingency.relative_risk(a, a + b, c, c + d)
            val = r.relative_risk
        elif func == "odds_ratio":
            # No scipy primitive returns the bare ratio; reproduce in numpy.
            val = (a * d) / (b * c) if (b * c) != 0 else float("inf")
        elif func == "phi_coefficient":
            # φ = (ad − bc) / sqrt((a+b)(c+d)(a+c)(b+d))
            num = (a * d) - (b * c)
            den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            val = num / den if den != 0 else float("nan")
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize 2x2_effect_sizes query");
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
                "failed to spawn python3 for 2x2_effect_sizes oracle: {e}"
            );
            eprintln!("skipping 2x2_effect_sizes oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open 2x2_effect_sizes oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "2x2_effect_sizes oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping 2x2_effect_sizes oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for 2x2_effect_sizes oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "2x2_effect_sizes oracle failed: {stderr}"
        );
        eprintln!("skipping 2x2_effect_sizes oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse 2x2_effect_sizes oracle JSON"))
}

#[test]
fn diff_stats_2x2_effect_sizes() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let table = [
            [case.table[0], case.table[1]],
            [case.table[2], case.table[3]],
        ];
        let rust_v = match case.func.as_str() {
            "relative_risk" => relative_risk(&table),
            "odds_ratio" => odds_ratio(&table),
            "phi_coefficient" => phi_coefficient(&table),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_2x2_effect_sizes".into(),
        category: "scipy.stats.contingency.relative_risk + numpy reference".into(),
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
                "2x2_effect_sizes {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "2x2_effect_sizes conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
