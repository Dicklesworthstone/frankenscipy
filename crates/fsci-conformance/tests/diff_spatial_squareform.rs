#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_spatial squareform
//! and validation utilities.
//!
//! Resolves [frankenscipy-dvkls]. Diffs against:
//!   • scipy.spatial.distance.squareform — both directions
//!   • scipy.spatial.distance.num_obs_dm
//!   • scipy.spatial.distance.num_obs_y
//!   • scipy.spatial.distance.is_valid_dm
//!   • scipy.spatial.distance.is_valid_y
//!
//! 4 dimensional fixtures (n=3, 5, 8, 12) plus 4 invalid-DM
//! probes (asymmetric, non-zero diag, negative entry, NaN).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_spatial::{
    is_valid_dm, is_valid_y, num_obs_dm, num_obs_y, squareform_to_condensed, squareform_to_matrix,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-15;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    valid_dms: Vec<DmCase>,
    invalid_dms: Vec<DmCase>,
}

#[derive(Debug, Clone, Serialize)]
struct DmCase {
    case_id: String,
    matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ValidArm {
    case_id: String,
    condensed: Option<Vec<f64>>,
    matrix: Option<Vec<Vec<f64>>>,
    num_obs_dm: Option<i64>,
    num_obs_y: Option<i64>,
    is_valid_dm: Option<bool>,
    is_valid_y: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct InvalidArm {
    case_id: String,
    is_valid_dm: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    valid_dms: Vec<ValidArm>,
    invalid_dms: Vec<InvalidArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sub_check: String,
    pass: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create squareform diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize squareform diff log");
    fs::write(path, json).expect("write squareform diff log");
}

fn build_dm(n: usize, seed: f64) -> Vec<Vec<f64>> {
    // Symmetric n×n with zero diagonal; all entries distinct.
    // Use dyadic-rational arithmetic so values are exact-binary
    // and round-trip losslessly through JSON (multipliers 2.0 and
    // 0.5, plus an integer seed).
    let mut m = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = (i as f64 + 1.0) * 2.0 + (j as f64 + 1.0) * 0.5 + seed;
            m[i][j] = v;
            m[j][i] = v;
        }
    }
    m
}

fn generate_query() -> OracleQuery {
    let valid_dms = vec![
        DmCase {
            case_id: "n3".into(),
            matrix: build_dm(3, 0.0),
        },
        DmCase {
            case_id: "n5".into(),
            matrix: build_dm(5, 0.5),
        },
        DmCase {
            case_id: "n8".into(),
            matrix: build_dm(8, 1.0),
        },
        DmCase {
            case_id: "n12".into(),
            matrix: build_dm(12, 2.0),
        },
    ];
    let mut asymmetric = build_dm(4, 0.0);
    asymmetric[0][1] = 1.0;
    asymmetric[1][0] = 2.0; // breaks symmetry
    let mut nonzero_diag = build_dm(4, 0.0);
    nonzero_diag[2][2] = 5.0;
    let mut negative_entry = build_dm(4, 0.0);
    negative_entry[0][1] = -3.0;
    negative_entry[1][0] = -3.0;
    let mut nan_entry = build_dm(4, 0.0);
    nan_entry[0][2] = f64::NAN;
    nan_entry[2][0] = f64::NAN;

    let invalid_dms = vec![
        DmCase {
            case_id: "asymmetric".into(),
            matrix: asymmetric,
        },
        DmCase {
            case_id: "nonzero_diag".into(),
            matrix: nonzero_diag,
        },
        DmCase {
            case_id: "negative_entry".into(),
            matrix: negative_entry,
        },
        DmCase {
            case_id: "nan_entry".into(),
            matrix: nan_entry,
        },
    ];

    OracleQuery {
        valid_dms,
        invalid_dms,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.spatial.distance import (
    squareform, num_obs_dm, num_obs_y, is_valid_dm, is_valid_y,
)

def safe_float_list(arr):
    out = []
    for v in arr.tolist():
        f = float(v)
        if not math.isfinite(f):
            return None
        out.append(f)
    return out

def safe_float_2d(arr):
    out = []
    for row in arr.tolist():
        rrow = []
        for v in row:
            f = float(v)
            if not math.isfinite(f):
                return None
            rrow.append(f)
        out.append(rrow)
    return out

q = json.load(sys.stdin)

valid_results = []
for case in q["valid_dms"]:
    cid = case["case_id"]
    M = np.asarray(case["matrix"], dtype=np.float64)
    out = {
        "case_id": cid,
        "condensed": None,
        "matrix": None,
        "num_obs_dm": None,
        "num_obs_y": None,
        "is_valid_dm": None,
        "is_valid_y": None,
    }
    try:
        cond = squareform(M)
        sq = squareform(cond)
        out["condensed"] = safe_float_list(cond)
        out["matrix"] = safe_float_2d(sq)
        out["num_obs_dm"] = int(num_obs_dm(M))
        out["num_obs_y"] = int(num_obs_y(cond))
        out["is_valid_dm"] = bool(is_valid_dm(M))
        out["is_valid_y"] = bool(is_valid_y(cond))
    except Exception:
        pass
    valid_results.append(out)

invalid_results = []
for case in q["invalid_dms"]:
    cid = case["case_id"]
    M = np.asarray(case["matrix"], dtype=np.float64)
    out = {"case_id": cid, "is_valid_dm": None}
    try:
        # Use throw=False so scipy returns False instead of raising.
        out["is_valid_dm"] = bool(is_valid_dm(M, throw=False))
    except Exception:
        pass
    invalid_results.append(out)

print(json.dumps(
    {"valid_dms": valid_results, "invalid_dms": invalid_results},
    allow_nan=False,
))
"#;
    let query_json = serde_json::to_string(query).expect("serialize squareform query");
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
                "failed to spawn python3 for squareform oracle: {e}"
            );
            eprintln!("skipping squareform oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open squareform oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "squareform oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping squareform oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for squareform oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "squareform oracle failed: {stderr}"
        );
        eprintln!("skipping squareform oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse squareform oracle JSON"))
}

#[test]
fn diff_spatial_squareform() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.valid_dms.len(), query.valid_dms.len());
    assert_eq!(oracle.invalid_dms.len(), query.invalid_dms.len());

    let valid_map: HashMap<String, ValidArm> = oracle
        .valid_dms
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let invalid_map: HashMap<String, InvalidArm> = oracle
        .invalid_dms
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut cases = Vec::new();
    let mut ledger = CompareLedger::new(
        "diff_spatial_squareform",
        &[
            "squareform_to_condensed",
            "squareform_to_matrix",
            "num_obs_dm",
            "num_obs_y",
            "is_valid_dm_valid_input",
            "is_valid_y_valid_input",
            "is_valid_dm_invalid_input",
        ],
    );

    for case in &query.valid_dms {
        let scipy_arm = valid_map.get(&case.case_id).expect("validated oracle");
        let n = case.matrix.len();
        // fsci's condensed form feeds the round-trip, num_obs_y and is_valid_y arms; when it
        // fails, each of those arms records an fsci failure instead of skipping.
        let rust_cond_res = squareform_to_condensed(&case.matrix);
        let rust_cond = rust_cond_res.as_ref().ok();

        // squareform_to_condensed. slices rejects a length mismatch and a non-finite element.
        match (
            ledger.slices(
                "squareform_to_condensed",
                &case.case_id,
                scipy_arm.condensed.as_deref(),
                rust_cond.map(Vec::as_slice),
            ),
            &rust_cond_res,
        ) {
            (Some((scipy_cond, rust_cond)), _) => {
                let pass = rust_cond.len() == scipy_cond.len()
                    && rust_cond
                        .iter()
                        .zip(scipy_cond.iter())
                        .all(|(r, s)| (r - s).abs() <= ABS_TOL);
                ledger.compared("squareform_to_condensed", &case.case_id, pass);
                cases.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "squareform_to_condensed".into(),
                    pass,
                    detail: format!(
                        "rust_len={}, scipy_len={}",
                        rust_cond.len(),
                        scipy_cond.len()
                    ),
                });
            }
            (None, Err(e)) if scipy_arm.condensed.is_some() => cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "squareform_to_condensed".into(),
                pass: false,
                detail: format!("rust err: {e:?}"),
            }),
            _ => {} // the ledger recorded why this case was not compared
        }

        // squareform_to_matrix (round-trip on the condensed). Element values go through
        // slices on the row-major flattening; the row structure is checked below.
        let rust_mat_res = rust_cond.map(|c| squareform_to_matrix(c));
        let rust_mat = rust_mat_res.as_ref().and_then(|r| r.as_ref().ok());
        let scipy_mat_flat = scipy_arm.matrix.as_ref().map(|m| m.concat());
        let rust_mat_flat = rust_mat.map(|m| m.concat());
        let gate = ledger.slices(
            "squareform_to_matrix",
            &case.case_id,
            scipy_mat_flat.as_deref(),
            rust_mat_flat.as_deref(),
        );
        // gate is Some only when both matrices are present.
        match (gate, scipy_arm.matrix.as_ref(), rust_mat, &rust_mat_res) {
            (Some(_), Some(scipy_mat), Some(rust_mat), _) => {
                let pass = rust_mat.len() == scipy_mat.len()
                    && rust_mat.iter().zip(scipy_mat.iter()).all(|(rr, sr)| {
                        rr.len() == sr.len()
                            && rr
                                .iter()
                                .zip(sr.iter())
                                .all(|(r, s)| (r - s).abs() <= ABS_TOL)
                    });
                ledger.compared("squareform_to_matrix", &case.case_id, pass);
                cases.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "squareform_to_matrix".into(),
                    pass,
                    detail: format!("rust_n={}", rust_mat.len()),
                });
            }
            (_, Some(_), _, Some(Err(e))) => cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "squareform_to_matrix".into(),
                pass: false,
                detail: format!("rust err: {e:?}"),
            }),
            _ => {} // the ledger recorded why this case was not compared
        }

        // num_obs_dm
        if let Some((scipy_n, rust_n)) = ledger.both(
            "num_obs_dm",
            &case.case_id,
            scipy_arm.num_obs_dm,
            Some(num_obs_dm(&case.matrix) as i64),
        ) {
            ledger.compared("num_obs_dm", &case.case_id, rust_n == scipy_n);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "num_obs_dm".into(),
                pass: rust_n == scipy_n,
                detail: format!("rust={rust_n}, scipy={scipy_n}"),
            });
        }

        // num_obs_y
        if let Some((scipy_n, rust_n)) = ledger.both(
            "num_obs_y",
            &case.case_id,
            scipy_arm.num_obs_y,
            rust_cond.map(|c| num_obs_y(c) as i64),
        ) {
            ledger.compared("num_obs_y", &case.case_id, rust_n == scipy_n);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "num_obs_y".into(),
                pass: rust_n == scipy_n,
                detail: format!("rust={rust_n}, scipy={scipy_n}"),
            });
        }

        // is_valid_dm — both should agree (true on valid fixtures).
        if let Some((scipy_b, rust_b)) = ledger.both(
            "is_valid_dm_valid_input",
            &case.case_id,
            scipy_arm.is_valid_dm,
            Some(is_valid_dm(&case.matrix, ABS_TOL)),
        ) {
            ledger.compared("is_valid_dm_valid_input", &case.case_id, rust_b == scipy_b);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "is_valid_dm_valid_input".into(),
                pass: rust_b == scipy_b,
                detail: format!("rust={rust_b}, scipy={scipy_b}, n={n}"),
            });
        }

        // is_valid_y — both should agree on the condensed form.
        if let Some((scipy_b, rust_b)) = ledger.both(
            "is_valid_y_valid_input",
            &case.case_id,
            scipy_arm.is_valid_y,
            rust_cond.map(|c| is_valid_y(c)),
        ) {
            ledger.compared("is_valid_y_valid_input", &case.case_id, rust_b == scipy_b);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "is_valid_y_valid_input".into(),
                pass: rust_b == scipy_b,
                detail: format!("rust={rust_b}, scipy={scipy_b}"),
            });
        }
    }

    // The nan_entry matrix carries f64::NAN, which serde_json sends as null; the oracle's
    // np.asarray(dtype=float64) reads null back as NaN, so SciPy sees the intended input.
    for case in &query.invalid_dms {
        let scipy_arm = invalid_map.get(&case.case_id).expect("validated oracle");
        if let Some((scipy_b, rust_b)) = ledger.both(
            "is_valid_dm_invalid_input",
            &case.case_id,
            scipy_arm.is_valid_dm,
            Some(is_valid_dm(&case.matrix, ABS_TOL)),
        ) {
            ledger.compared(
                "is_valid_dm_invalid_input",
                &case.case_id,
                rust_b == scipy_b,
            );
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "is_valid_dm_invalid_input".into(),
                pass: rust_b == scipy_b,
                detail: format!("rust={rust_b}, scipy={scipy_b}"),
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_squareform".into(),
        category: "fsci_spatial squareform / num_obs / is_valid".into(),
        case_count: cases.len(),
        compared: ledger.counts().clone(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "squareform/util mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "squareform conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
    ledger.finish(query.valid_dms.len().min(query.invalid_dms.len()));
}
