#![forbid(unsafe_code)]
//! Live SciPy differential test for `fsci_linalg::funm` / `funm_with_error` on matrices with
//! complex eigenvalues (frankenscipy-szq1n.6). SciPy's funm runs the Schur-Parlett recurrence on
//! the complex Schur form, so `func` is evaluated at complex eigenvalues; fsci used to evaluate it
//! on the real Schur diagonal and returned NaN or a wrong value for every complex pair.
//!
//! Cases: the rotation [[0,-1],[1,0]]; the bead's 3x3 M with a complex pair; a seeded random 6x6
//! with at least two complex pairs (the first seed that has them, chosen at run time); a symmetric
//! 4x4. Functions: exp, sin, cos and z^3 - 2z + 1. Every result is compared with SciPy's funm AND
//! with SciPy's dedicated routine (expm, sinm, cosm, and the polynomial by matrix products). The
//! error estimate must flag exactly when SciPy's does (err > 1000 eps), and for the Jordan block
//! [[2,1],[0,2]], where the recurrence cannot resolve the repeated eigenvalue, it must be at least
//! SciPy's (1).

use std::io::Write;
use std::process::Stdio;

use fsci_linalg::{DecompOptions, eigvals, funm_with_error};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// max |fsci - reference| / max(1, max |reference|), against SciPy's funm and against SciPy's
/// dedicated routine.
const FUNM_REL_TOL: f64 = 1e-12;
const FUNCS: [&str; 4] = ["exp", "sin", "cos", "poly"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    a: Vec<Vec<f64>>,
    funcs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Row {
    case_id: String,
    func: String,
    complex_pairs: usize,
    funm: Option<Vec<Vec<f64>>>,
    funm_err: f64,
    reference: Vec<Vec<f64>>,
}

/// splitmix64 on [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn complex_pairs(a: &[Vec<f64>]) -> usize {
    let (_, im) = eigvals(a, DecompOptions::default()).expect("eigvals");
    im.iter().filter(|v| v.abs() > 1e-8).count() / 2
}

/// The first seeded random 6x6 with at least two complex-conjugate pairs.
fn random_with_two_pairs() -> (u64, Vec<Vec<f64>>) {
    (1..1000_u64)
        .find_map(|seed| {
            let mut rng = Rng(seed);
            let a: Vec<Vec<f64>> = (0..6)
                .map(|_| (0..6).map(|_| rng.next()).collect())
                .collect();
            (complex_pairs(&a) >= 2).then_some((seed, a))
        })
        .expect("a seed below 1000 gives a 6x6 with two complex pairs")
}

fn cases() -> Vec<Case> {
    let all: Vec<String> = FUNCS.iter().map(|s| (*s).to_string()).collect();
    let (seed, random) = random_with_two_pairs();
    println!(
        "random 6x6: seed {seed}, {} complex pairs",
        complex_pairs(&random)
    );
    vec![
        Case {
            case_id: "rotation".into(),
            a: vec![vec![0.0, -1.0], vec![1.0, 0.0]],
            funcs: all.clone(),
        },
        Case {
            case_id: "m3_complex_pair".into(),
            a: vec![
                vec![1.0, 2.0, 0.0],
                vec![-3.0, 1.0, 1.0],
                vec![0.0, 0.5, 2.0],
            ],
            funcs: all.clone(),
        },
        Case {
            case_id: format!("random6_seed{seed}"),
            a: random,
            funcs: all.clone(),
        },
        Case {
            case_id: "symmetric4".into(),
            a: vec![
                vec![4.0, 1.0, 0.0, 0.0],
                vec![1.0, 3.0, 1.0, 0.0],
                vec![0.0, 1.0, 2.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
            ],
            funcs: all,
        },
        Case {
            case_id: "jordan2".into(),
            a: vec![vec![2.0, 1.0], vec![0.0, 2.0]],
            funcs: vec!["exp".into()],
        },
    ]
}

fn fsci_funm(a: &[Vec<f64>], func: &str) -> (Vec<Vec<f64>>, f64) {
    let options = DecompOptions::default();
    let out = match func {
        "exp" => funm_with_error(a, |z| z.exp(), options),
        "sin" => funm_with_error(a, |z| z.sin(), options),
        "cos" => funm_with_error(a, |z| z.cos(), options),
        "poly" => funm_with_error(a, |z| z * z * z - z * 2.0 + 1.0, options),
        other => unreachable!("FUNCS has no {other}"),
    };
    out.expect("funm_with_error")
}

fn rel_diff(got: &[Vec<f64>], want: &[Vec<f64>]) -> f64 {
    assert_eq!(got.len(), want.len(), "shape");
    let scale = want
        .iter()
        .flatten()
        .fold(1.0_f64, |acc, v| acc.max(v.abs()));
    got.iter()
        .flatten()
        .zip(want.iter().flatten())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

fn scipy_rows(cases: &[Case]) -> Option<Vec<Row>> {
    let script = r#"
import json, sys
import numpy as np
from scipy.linalg import funm, expm, sinm, cosm

FUNCS = {"exp": np.exp, "sin": np.sin, "cos": np.cos, "poly": lambda z: z**3 - 2*z + 1}
REFS = {
    "exp": expm, "sin": sinm, "cos": cosm,
    "poly": lambda A: A @ A @ A - 2 * A + np.eye(A.shape[0]),
}
rows = []
for case in json.load(sys.stdin):
    A = np.asarray(case["a"], dtype=np.float64)
    pairs = int(np.sum(np.abs(np.linalg.eigvals(A).imag) > 1e-8) // 2)
    for name in case["funcs"]:
        F, err = funm(A, FUNCS[name], disp=False)
        F = np.asarray(F)
        real = None if np.max(np.abs(F.imag)) > 1e-10 else F.real.tolist()
        rows.append({
            "case_id": case["case_id"], "func": name, "complex_pairs": pairs,
            "funm": real, "funm_err": float(err),
            "reference": np.real(REFS[name](A)).tolist(),
        })
print(json.dumps(rows, allow_nan=False))
"#;
    let query = serde_json::to_string(cases).expect("serialize cases");
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
                "failed to spawn the funm oracle: {e}"
            );
            eprintln!("skipping funm oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for the funm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "funm oracle failed: {stderr}"
        );
        eprintln!("skipping funm oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse funm oracle JSON"))
}

#[test]
fn diff_linalg_funm_complex() {
    let cases = cases();
    let expected_rows: usize = cases.iter().map(|c| c.funcs.len()).sum();
    let Some(rows) = scipy_rows(&cases) else {
        return;
    };
    assert_eq!(
        rows.len(),
        expected_rows,
        "the oracle must answer every row"
    );

    let flag = 1000.0 * f64::EPSILON;
    let mut compared = 0;
    let mut failures = Vec::new();
    for row in &rows {
        let case = cases
            .iter()
            .find(|c| c.case_id == row.case_id)
            .expect("case");
        let (fsci, err) = fsci_funm(&case.a, &row.func);
        let scipy = row
            .funm
            .as_ref()
            .expect("SciPy's funm is real for these cases");
        let vs_funm = rel_diff(&fsci, scipy);
        let vs_reference = rel_diff(&fsci, &row.reference);
        println!(
            "{}/{}: pairs {} | vs SciPy funm {vs_funm:.2e} | vs SciPy dedicated {vs_reference:.2e} | err fsci {err:.2e} SciPy {:.2e}",
            row.case_id, row.func, row.complex_pairs, row.funm_err
        );
        if row.case_id == "jordan2" {
            // The recurrence cannot resolve it: agree with SciPy's funm (not expm), and say so.
            if vs_funm > FUNM_REL_TOL || err < row.funm_err || err <= flag {
                failures.push(format!("{}/{}", row.case_id, row.func));
            }
        } else if vs_funm > FUNM_REL_TOL
            || vs_reference > FUNM_REL_TOL
            || (err > flag) != (row.funm_err > flag)
        {
            failures.push(format!("{}/{}", row.case_id, row.func));
        }
        compared += 1;
    }
    let random = rows
        .iter()
        .find(|r| r.case_id.starts_with("random6"))
        .expect("random row");
    assert!(
        random.complex_pairs >= 2,
        "SciPy sees {} complex pairs in the random case",
        random.complex_pairs
    );
    assert_eq!(compared, expected_rows);
    assert!(
        failures.is_empty(),
        "funm disagrees with SciPy: {failures:?}"
    );
}
