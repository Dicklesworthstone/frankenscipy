#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.linalg.funm_multiply_krylov(f, A, b, ...)`
//! against `fsci_sparse::funm_multiply_krylov`.
//!
//! Both sides run the same restarted Arnoldi (or Lanczos, `assume_a='hermitian'`) on the same
//! sparse matrix and apply `f` to the same growing block-Hessenberg matrix, with `f` one of
//! `expm` (`scipy.linalg.expm` / `fsci_linalg::expm`), `sqrtm` on SPD matrices, and the
//! polynomial `H ↦ H·H` (which `funm` reproduces exactly once the Krylov space holds `A²b`).
//! Cases cover nonsymmetric and symmetric matrices of sizes 8–150, `t` of either sign, short
//! restart cycles (`restart_every_m` 3–10) so several restarts happen, too few restarts to
//! converge (the partial answer must agree too), tight and loose `rtol`, positive `atol`,
//! `b = 0`, and the argument errors (negative `atol`, zero `restart_every_m` or
//! `max_restarts`).
//!
//! Agreement is at rounding level: the Gram–Schmidt inner products differ from numpy's BLAS
//! dot in the last bits, so `max|fsci − SciPy| ≤ 1e-11·max(1, max|SciPy|)`.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_linalg::DecompOptions;
use fsci_sparse::{
    CooMatrix, FormatConvertible, FunmKrylovOptions, Shape2D, SparseError, SparseResult,
    funm_multiply_krylov,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const TOL: f64 = 1.0e-11;

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    n: usize,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    f: String,
    hermitian: bool,
    t: f64,
    atol: f64,
    rtol: f64,
    restart_every_m: Option<usize>,
    max_restarts: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    y: Option<Vec<f64>>,
    error: Option<String>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A sparse test matrix: a 1-D Laplacian-like band plus random off-band entries, scaled to
/// norm ≈ `scale`; symmetric when `sym`, and shifted SPD when `spd`.
fn test_matrix(n: usize, sym: bool, spd: bool, scale: f64, rng: &mut Lcg) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = -2.0;
        if i + 1 < n {
            a[i][i + 1] = 1.0;
            a[i + 1][i] = 1.0;
        }
    }
    for _ in 0..2 * n {
        let i = ((rng.next() + 1.0) * 0.5 * n as f64) as usize % n;
        let j = ((rng.next() + 1.0) * 0.5 * n as f64) as usize % n;
        let v = 0.3 * rng.next();
        a[i][j] += v;
        if sym && i != j {
            a[j][i] += v;
        }
    }
    if spd {
        // -A shifted to be diagonally dominant: SPD.
        for (i, row) in a.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = -*x;
                if i == j {
                    *x += 1.0;
                }
            }
        }
        for i in 0..n {
            let off: f64 = (0..n).filter(|&j| j != i).map(|j| a[i][j].abs()).sum();
            a[i][i] = a[i][i].max(off + 0.5);
        }
    }
    let nrm = a
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    for row in &mut a {
        for x in row.iter_mut() {
            *x *= scale / nrm;
        }
    }
    a
}

fn corpus() -> Vec<Query> {
    let mut rng = Lcg(31_337);
    let mut out = Vec::new();
    let mut push = |id: String,
                    a: &Vec<Vec<f64>>,
                    b: Vec<f64>,
                    f: &str,
                    hermitian: bool,
                    opts: (f64, f64, f64, Option<usize>, usize)| {
        let (t, atol, rtol, restart_every_m, max_restarts) = opts;
        out.push(Query {
            id,
            n: a.len(),
            a: a.clone(),
            b,
            f: f.to_string(),
            hermitian,
            t,
            atol,
            rtol,
            restart_every_m,
            max_restarts,
        });
    };
    for &n in &[8usize, 20, 45, 100, 150] {
        for (kind, sym, spd) in [
            ("gen", false, false),
            ("sym", true, false),
            ("spd", true, true),
        ] {
            let a = test_matrix(n, sym, spd, 4.0, &mut rng);
            let b: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let fs: &[&str] = if spd {
                &["expm", "sqrtm", "square"]
            } else {
                &["expm", "square"]
            };
            for &f in fs {
                for &hermitian in &[false, true] {
                    if hermitian && !sym {
                        continue;
                    }
                    let tag = if hermitian { "her" } else { "gen" };
                    let variants: [(&str, (f64, f64, f64, Option<usize>, usize)); 6] = [
                        ("default", (1.0, 0.0, 1e-6, None, 20)),
                        ("t_neg", (-0.5, 0.0, 1e-10, None, 20)),
                        ("m5", (1.0, 0.0, 1e-10, Some(5), 20)),
                        ("m3_r2", (0.7, 0.0, 1e-12, Some(3), 2)),
                        ("m10_atol", (1.0, 1e-3, 0.0, Some(10), 20)),
                        ("m7_loose", (2.0, 0.0, 1e-3, Some(7), 5)),
                    ];
                    for (vname, opts) in variants {
                        if f == "sqrtm" && opts.0 < 0.0 {
                            continue;
                        }
                        push(
                            format!("{kind}{n}_{f}_{tag}_{vname}"),
                            &a,
                            b.clone(),
                            f,
                            hermitian,
                            opts,
                        );
                    }
                }
            }
        }
    }
    let a = test_matrix(12, false, false, 2.0, &mut rng);
    push(
        "zero_b".into(),
        &a,
        vec![0.0; 12],
        "expm",
        false,
        (1.0, 0.0, 1e-6, None, 20),
    );
    let b: Vec<f64> = (0..12).map(|_| rng.next()).collect();
    push(
        "err_atol".into(),
        &a,
        b.clone(),
        "expm",
        false,
        (1.0, -1.0, 1e-6, None, 20),
    );
    push(
        "err_m0".into(),
        &a,
        b.clone(),
        "expm",
        false,
        (1.0, 0.0, 1e-6, Some(0), 20),
    );
    push(
        "err_r0".into(),
        &a,
        b,
        "expm",
        false,
        (1.0, 0.0, 1e-6, None, 0),
    );
    out
}

const ORACLE: &str = r#"
import json, sys
import numpy as np
from scipy.linalg import expm, sqrtm
from scipy.sparse.linalg import funm_multiply_krylov
fs = {"expm": expm, "sqrtm": lambda h: np.real(sqrtm(h)), "square": lambda h: h @ h}
out = []
for q in json.load(sys.stdin):
    a = np.array(q["a"], dtype=float)
    b = np.array(q["b"], dtype=float)
    try:
        y = funm_multiply_krylov(fs[q["f"]], a, b,
                                 assume_a="hermitian" if q["hermitian"] else "general",
                                 t=q["t"], atol=q["atol"], rtol=q["rtol"],
                                 restart_every_m=q["restart_every_m"],
                                 max_restarts=q["max_restarts"])
        out.append({"id": q["id"], "y": [float(v) for v in y], "error": None})
    except Exception as e:
        out.append({"id": q["id"], "y": None, "error": f"{type(e).__name__}: {e}"})
print(json.dumps(out))
"#;

fn dense_fn(name: &str) -> impl FnMut(&[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> + '_ {
    move |h: &[Vec<f64>]| {
        let wrap = |e: fsci_linalg::LinalgError| SparseError::InvalidArgument {
            message: format!("{e:?}"),
        };
        match name {
            "expm" => fsci_linalg::expm(h, DecompOptions::default()).map_err(wrap),
            "sqrtm" => fsci_linalg::sqrtm(h, DecompOptions::default()).map_err(wrap),
            _ => Ok(h
                .iter()
                .map(|row| {
                    (0..h.len())
                        .map(|j| row.iter().zip(h).map(|(x, r)| x * r[j]).sum())
                        .collect()
                })
                .collect()),
        }
    }
}

fn fsci(q: &Query) -> SparseResult<Vec<f64>> {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (i, row) in q.a.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x != 0.0 {
                r.push(i);
                c.push(j);
                v.push(x);
            }
        }
    }
    let csr = CooMatrix::from_triplets(Shape2D::new(q.n, q.n), v, r, c, true)?.to_csr()?;
    funm_multiply_krylov(
        dense_fn(&q.f),
        &csr,
        &q.b,
        FunmKrylovOptions {
            hermitian: q.hermitian,
            t: q.t,
            atol: q.atol,
            rtol: q.rtol,
            restart_every_m: q.restart_every_m,
            max_restarts: q.max_restarts,
        },
    )
}

#[test]
fn diff_sparse_funm_multiply_krylov() {
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
            eprintln!("skipping funm_multiply_krylov oracle: {e}");
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
        eprintln!("skipping funm_multiply_krylov oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut ledger = CompareLedger::new(
        "diff_sparse_funm_multiply_krylov",
        &["funm_multiply_krylov"],
    );
    let (mut values, mut errors) = (0usize, 0usize);
    let mut worst = 0.0_f64;
    let mut failures = Vec::new();
    for (q, arm) in queries.iter().zip(&arms) {
        assert_eq!(q.id, arm.id);
        let ours = fsci(q);
        let pass = match (&arm.y, &ours) {
            (None, Err(_)) => {
                errors += 1;
                true
            }
            (Some(want), Ok(got)) => {
                values += 1;
                let scale = want.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
                let err = got
                    .iter()
                    .zip(want)
                    .map(|(g, w)| (g - w).abs() / scale)
                    .fold(0.0, f64::max);
                worst = worst.max(err);
                got.len() == want.len() && err <= TOL
            }
            _ => false,
        };
        if !pass {
            failures.push(format!(
                "{}: fsci {:?} vs SciPy {:?}",
                q.id,
                ours.as_ref()
                    .map(|y| y.iter().take(3).copied().collect::<Vec<_>>()),
                arm.error.clone().or_else(|| arm
                    .y
                    .as_ref()
                    .map(|y| format!("{:?}", &y[..y.len().min(3)])))
            ));
        }
        ledger.compared("funm_multiply_krylov", &q.id, pass);
    }
    for f in failures.iter().take(15) {
        println!("FAIL {f}");
    }
    println!(
        "{} cases: {values} values (worst rel err {worst:.2e}), {errors} matching errors; {} failed",
        queries.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "funm_multiply_krylov differs from SciPy"
    );
    ledger.finish(queries.len());
}
