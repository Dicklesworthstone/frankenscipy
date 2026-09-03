#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_interpolate interpolation helpers:
//! chebyshev_nodes (first kind), chebyshev_nodes2 (second kind),
//! barycentric_weights, barycentric_eval, neville, hermite_interp.
//!
//! Resolves [frankenscipy-8aike]. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{
    barycentric_eval, barycentric_weights, chebyshev_nodes, chebyshev_nodes2, hermite_interp,
    neville,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct ChebCase {
    case_id: String,
    op: String, // "cheb1" | "cheb2"
    n: usize,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InterpCase {
    case_id: String,
    op: String, // "bary" | "neville" | "hermite"
    nodes: Vec<f64>,
    values: Vec<f64>,
    derivatives: Vec<f64>, // hermite only
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cheb: Vec<ChebCase>,
    interp: Vec<InterpCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChebArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct InterpArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    cheb: Vec<ChebArm>,
    interp: Vec<InterpArm>,
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
    fs::create_dir_all(output_dir()).expect("create cheb_bary diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let cheb = vec![
        ChebCase {
            case_id: "cheb1_n5_0_1".into(),
            op: "cheb1".into(),
            n: 5,
            a: 0.0,
            b: 1.0,
        },
        ChebCase {
            case_id: "cheb1_n7_n1_1".into(),
            op: "cheb1".into(),
            n: 7,
            a: -1.0,
            b: 1.0,
        },
        ChebCase {
            case_id: "cheb1_n10_n2_3".into(),
            op: "cheb1".into(),
            n: 10,
            a: -2.0,
            b: 3.0,
        },
        ChebCase {
            case_id: "cheb2_n5_0_1".into(),
            op: "cheb2".into(),
            n: 5,
            a: 0.0,
            b: 1.0,
        },
        ChebCase {
            case_id: "cheb2_n6_n1_1".into(),
            op: "cheb2".into(),
            n: 6,
            a: -1.0,
            b: 1.0,
        },
        ChebCase {
            case_id: "cheb2_n8_n5_5".into(),
            op: "cheb2".into(),
            n: 8,
            a: -5.0,
            b: 5.0,
        },
    ];

    // Use known nodes/values for interpolation; interior x.
    let nodes_a: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let vals_a: Vec<f64> = nodes_a.iter().map(|x| x * x - 2.0 * x).collect(); // y = x^2 - 2x
    let derivs_a: Vec<f64> = nodes_a.iter().map(|x| 2.0 * x - 2.0).collect(); // y' = 2x - 2

    let nodes_b: Vec<f64> = vec![-1.0, 0.0, 1.0, 2.0];
    let vals_b: Vec<f64> = nodes_b.iter().map(|x| (x * x * x) + x).collect(); // y = x^3 + x
    let derivs_b: Vec<f64> = nodes_b.iter().map(|x| 3.0 * x * x + 1.0).collect();

    let xs_a = [0.5_f64, 1.5, 2.5];
    let xs_b = [-0.5_f64, 0.5, 1.5];

    let mut interp = Vec::new();
    for op in ["bary", "neville", "hermite"] {
        for &x in &xs_a {
            interp.push(InterpCase {
                case_id: format!("{op}_quad_x{x}").replace('.', "p"),
                op: op.into(),
                nodes: nodes_a.clone(),
                values: vals_a.clone(),
                derivatives: derivs_a.clone(),
                x,
            });
        }
        for &x in &xs_b {
            interp.push(InterpCase {
                case_id: format!("{op}_cubic_x{x}")
                    .replace('.', "p")
                    .replace('-', "n"),
                op: op.into(),
                nodes: nodes_b.clone(),
                values: vals_b.clone(),
                derivatives: derivs_b.clone(),
                x,
            });
        }
    }

    OracleQuery { cheb, interp }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def cheb1_nodes(n, a, b):
    if n == 0: return []
    t = np.cos(np.pi * (2*np.arange(n) + 1) / (2.0*n))
    return ((a + b) / 2.0 + (b - a) / 2.0 * t).tolist()

def cheb2_nodes(n, a, b):
    if n == 0: return []
    if n == 1: return [(a + b) / 2.0]
    t = np.cos(np.pi * np.arange(n) / (n - 1))
    return ((a + b) / 2.0 + (b - a) / 2.0 * t).tolist()

def bary_weights(nodes):
    nodes = np.array(nodes, dtype=float)
    n = len(nodes)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (nodes[i] - nodes[j])
    return w

def bary_eval(nodes, values, weights, x):
    for k, xn in enumerate(nodes):
        if abs(x - xn) < 1e-15:
            return float(values[k])
    num = 0.0; den = 0.0
    for i in range(len(nodes)):
        t = weights[i] / (x - nodes[i])
        num += t * values[i]
        den += t
    return num / den

def neville(nodes, values, x):
    p = list(values)
    n = len(nodes)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            dx = nodes[i] - nodes[i - j]
            if abs(dx) < 1e-15:
                return float("nan")
            p[i] = ((x - nodes[i - j]) * p[i] - (x - nodes[i]) * p[i - 1]) / dx
    return p[-1]

def hermite_interp(nodes, values, derivs, x):
    # Standard Hermite basis interpolation
    n = len(nodes)
    result = 0.0
    for i in range(n):
        # Compute L_i(x) and L_i'(x_i)
        Li = 1.0
        Li_prime = 0.0
        for j in range(n):
            if i != j:
                Li *= (x - nodes[j]) / (nodes[i] - nodes[j])
        # L_i'(x_i) = sum_{k != i} 1/(x_i - x_k)
        for k in range(n):
            if k != i:
                Li_prime += 1.0 / (nodes[i] - nodes[k])
        # H_i(x) = [1 - 2*(x - x_i)*L_i'(x_i)] * L_i(x)^2
        Hi = (1.0 - 2.0 * (x - nodes[i]) * Li_prime) * Li * Li
        # K_i(x) = (x - x_i) * L_i(x)^2
        Ki = (x - nodes[i]) * Li * Li
        result += values[i] * Hi + derivs[i] * Ki
    return result

q = json.load(sys.stdin)
cheb = []
for case in q["cheb"]:
    cid = case["case_id"]; op = case["op"]
    n = int(case["n"]); a = float(case["a"]); b = float(case["b"])
    try:
        if op == "cheb1": v = cheb1_nodes(n, a, b)
        elif op == "cheb2": v = cheb2_nodes(n, a, b)
        else: v = None
        if v is None or any(not math.isfinite(x) for x in v):
            cheb.append({"case_id": cid, "values": None})
        else:
            cheb.append({"case_id": cid, "values": [float(x) for x in v]})
    except Exception as e:
        sys.stderr.write(f"cheb {cid}: {e}\n")
        cheb.append({"case_id": cid, "values": None})

interp = []
for case in q["interp"]:
    cid = case["case_id"]; op = case["op"]
    nodes = [float(x) for x in case["nodes"]]
    values = [float(x) for x in case["values"]]
    derivs = [float(x) for x in case["derivatives"]]
    x = float(case["x"])
    try:
        if op == "bary":
            w = bary_weights(nodes)
            v = bary_eval(nodes, values, w, x)
        elif op == "neville":
            v = neville(nodes, values, x)
        elif op == "hermite":
            v = hermite_interp(nodes, values, derivs, x)
        else:
            v = float("nan")
        if math.isfinite(v):
            interp.append({"case_id": cid, "value": float(v)})
        else:
            interp.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"interp {cid}: {e}\n")
        interp.append({"case_id": cid, "value": None})

print(json.dumps({"cheb": cheb, "interp": interp}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for cheb_bary oracle: {e}"
            );
            eprintln!("skipping cheb_bary oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cheb_bary oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cheb_bary oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cheb_bary oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cheb_bary oracle failed: {stderr}"
        );
        eprintln!("skipping cheb_bary oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cheb_bary oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_interpolate_chebyshev_barycentric() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let cheb_map: HashMap<String, ChebArm> = oracle
        .cheb
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let interp_map: HashMap<String, InterpArm> = oracle
        .interp
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.cheb {
        let Some(arm) = cheb_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let actual = match case.op.as_str() {
            "cheb1" => chebyshev_nodes(case.n, case.a, case.b),
            "cheb2" => chebyshev_nodes2(case.n, case.a, case.b),
            _ => continue,
        };
        let abs_d = vec_max_diff(&actual, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.interp {
        let Some(arm) = interp_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = match case.op.as_str() {
            "bary" => {
                let w = barycentric_weights(&case.nodes);
                barycentric_eval(&case.nodes, &case.values, &w, case.x)
            }
            "neville" => neville(&case.nodes, &case.values, case.x),
            "hermite" => hermite_interp(&case.nodes, &case.values, &case.derivatives, case.x),
            _ => continue,
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_chebyshev_barycentric".into(),
        category: "fsci_interpolate chebyshev/barycentric/neville/hermite vs numpy".into(),
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
        "cheb_bary conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
