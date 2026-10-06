#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.linalg.matrix_balance(a, permute, scale,
//! separate=True)` on sparse matrices, where LAPACK's `dgebal` isolates eigenvalues by
//! permutation.
//!
//! `dgebal` in LAPACK 3.12 (SciPy 1.17.1 ships it in OpenBLAS 0.3.30) scans the rows once per
//! pass and keeps scanning after each swap. fsci restarted the scan after every swap, which
//! visits rows in a different order and returns a different permutation and balanced matrix on
//! about 5% of random sparse matrices (215 of 4000 in a Python emulation of both orders against
//! `scipy.linalg.lapack.dgebal`). Permutations and the power-of-two scalings are exact, so every
//! output is compared for EXACT equality.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_linalg::matrix_balance;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    a: Vec<Vec<f64>>,
    permute: bool,
    scale: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    balanced: Option<Vec<Vec<f64>>>,
    scaling: Option<Vec<f64>>,
    perm: Option<Vec<usize>>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn corpus() -> Vec<Query> {
    let mut rng = Lcg(20_261_006);
    let mut out = Vec::new();
    for k in 0..400 {
        let n = 2 + (k % 6);
        let a: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        let keep = rng.next() < 0.35;
                        let v =
                            (rng.next() * 2.0 - 1.0) * 10f64.powi((rng.next() * 6.0) as i32 - 3);
                        if keep { v } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let (permute, scale) = match k % 4 {
            0 => (true, false),
            _ => (true, true),
        };
        out.push(Query {
            id: format!("m{k}_n{n}"),
            a,
            permute,
            scale,
        });
    }
    out
}

const ORACLE: &str = r#"
import json, sys
import numpy as np
from scipy.linalg import matrix_balance
out = []
for q in json.load(sys.stdin):
    arm = {"id": q["id"], "balanced": None, "scaling": None, "perm": None}
    try:
        b, (s, p) = matrix_balance(np.array(q["a"], dtype=float), permute=q["permute"],
                                   scale=q["scale"], separate=True)
        arm.update(balanced=b.tolist(), scaling=[float(x) for x in s], perm=[int(x) for x in p])
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;

#[test]
fn diff_linalg_matrix_balance_permutation() {
    let queries = corpus();
    let payload = serde_json::to_string(&queries).expect("serialize");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "spawn oracle: {e}"
            );
            eprintln!("skipping matrix_balance oracle: {e}");
            return;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write query");
    let output = child.wait_with_output().expect("oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping matrix_balance oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut ledger = CompareLedger::new(
        "diff_linalg_matrix_balance_permutation",
        &["matrix_balance"],
    );
    let mut failures = Vec::new();
    for (q, arm) in queries.iter().zip(&arms) {
        assert_eq!(q.id, arm.id);
        let fsci = matrix_balance(&q.a, q.permute, q.scale).ok();
        let Some((scipy, ours)) = ledger.both(
            "matrix_balance",
            &q.id,
            arm.balanced
                .as_ref()
                .zip(arm.scaling.as_ref())
                .zip(arm.perm.as_ref()),
            fsci.as_ref(),
        ) else {
            failures.push(format!("{}: missing value", q.id));
            continue;
        };
        let ((balanced, scaling), perm) = scipy;
        let same = ours.balanced == *balanced && ours.scaling == *scaling && ours.perm == *perm;
        if !same {
            failures.push(format!(
                "{}: fsci perm {:?} scaling {:?} vs SciPy perm {:?} scaling {:?}",
                q.id, ours.perm, ours.scaling, perm, scaling
            ));
        }
        ledger.compared("matrix_balance", &q.id, same);
    }
    for f in failures.iter().take(10) {
        println!("FAIL {f}");
    }
    println!(
        "{} matrices compared, {} differ from SciPy",
        queries.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "matrix_balance differs from SciPy");
    ledger.finish(queries.len());
}
