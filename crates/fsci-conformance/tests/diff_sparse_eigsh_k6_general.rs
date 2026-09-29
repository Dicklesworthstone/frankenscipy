#![forbid(unsafe_code)]
//! Live `scipy.sparse.linalg.eigsh(A, k=6, which='LM')` parity for `fsci_sparse::eigsh` on
//! RUNTIME-GENERATED matrices (frankenscipy-szq1n.15).
//!
//! `eigsh` used a smaller Krylov window for exactly k == 6 and skipped its explicit residual
//! check for k <= 6, both tuned on one benchmark matrix (AGENTS #12). It now uses SciPy's
//! `ncv = min(n, max(2k+1, 20))` and checks every returned pair's residual. This test runs
//! k = 6 on 24 matrices generated from seeds that no development fixture uses: diagonally
//! dominant random sparse SPD matrices, Laplacians of random graphs, and a clustered top
//! spectrum under a sparse symmetric perturbation, n from 120 to 800.
//!
//! Rule: every matrix converges, and its six eigenvalues (sorted by |λ|) match SciPy's
//! (ARPACK, with implicit restarts) to `EIG_REL_TOL`. fsci restarts by thick-restart Lanczos
//! (frankenscipy-1ksfv.10); before that a single pass of the basis left relative residuals of
//! 4e-2 to 5e-1 on all 24 of these matrices, and an unconverged result is a failed case.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{CooMatrix, EigsOptions, FormatConvertible, Shape2D, eigsh};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const EIG_REL_TOL: f64 = 1.0e-8;
const K: usize = 6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct MatrixCase {
    case_id: String,
    n: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    eigvals_by_magnitude: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    n: usize,
    fsci_converged: bool,
    max_rel_diff: Option<f64>,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    converged_count: usize,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

/// xorshift64*: deterministic, and seeded here only.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

/// Symmetric triplets from strictly-upper off-diagonal entries plus a diagonal.
fn symmetric(n: usize, upper: &[(usize, usize, f64)], diag: &[f64]) -> MatrixCase {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for &(i, j, v) in upper {
        rows.extend([i, j]);
        cols.extend([j, i]);
        vals.extend([v, v]);
    }
    for (i, &d) in diag.iter().enumerate() {
        rows.push(i);
        cols.push(i);
        vals.push(d);
    }
    MatrixCase {
        case_id: String::new(),
        n,
        rows,
        cols,
        vals,
    }
}

fn random_upper(rng: &mut Rng, n: usize, per_row: usize, scale: f64) -> Vec<(usize, usize, f64)> {
    let mut seen = std::collections::BTreeSet::new();
    let mut upper = Vec::new();
    for i in 0..n {
        for _ in 0..per_row {
            let j = rng.below(n);
            let (lo, hi) = (i.min(j), i.max(j));
            if lo != hi && seen.insert((lo, hi)) {
                upper.push((lo, hi, scale * (2.0 * rng.next_f64() - 1.0)));
            }
        }
    }
    upper
}

fn generate_cases() -> Vec<MatrixCase> {
    let sizes = [120_usize, 200, 300, 400, 500, 600, 700, 800];
    let mut cases = Vec::new();
    for (idx, &n) in sizes.iter().enumerate() {
        let seed = 0xD1CE_5EED_0000_0000_u64 ^ ((idx as u64 + 1) * 0x9E37_79B9);
        // Diagonally dominant random sparse SPD.
        let mut rng = Rng(seed | 1);
        let upper = random_upper(&mut rng, n, 3, 1.0);
        let mut diag = vec![0.0; n];
        for &(i, j, v) in &upper {
            diag[i] += v.abs();
            diag[j] += v.abs();
        }
        for d in &mut diag {
            *d += 0.5 + 4.5 * rng.next_f64();
        }
        let mut spd = symmetric(n, &upper, &diag);
        spd.case_id = format!("spd_n{n}");
        cases.push(spd);

        // Laplacian of a random graph (average degree about 6).
        let mut rng = Rng((seed ^ 0xABCD_EF01) | 1);
        let edges = random_upper(&mut rng, n, 3, 1.0);
        let mut degree = vec![0.0; n];
        let upper: Vec<(usize, usize, f64)> = edges
            .iter()
            .map(|&(i, j, _)| {
                degree[i] += 1.0;
                degree[j] += 1.0;
                (i, j, -1.0)
            })
            .collect();
        let mut lap = symmetric(n, &upper, &degree);
        lap.case_id = format!("laplacian_n{n}");
        cases.push(lap);

        // A clustered top spectrum (six values within 5e-2 of 10) over a bulk in [0, 5],
        // perturbed by a sparse symmetric matrix of norm about 1e-2.
        let mut rng = Rng((seed ^ 0x1357_9BDF) | 1);
        let mut diag: Vec<f64> = (0..n).map(|_| 5.0 * rng.next_f64()).collect();
        for (slot, value) in [10.0, 9.99, 9.98, 9.97, 9.96, 9.95].into_iter().enumerate() {
            diag[(slot * 7919 + idx) % n] = value;
        }
        let upper = random_upper(&mut rng, n, 2, 1.0e-2);
        let mut clustered = symmetric(n, &upper, &diag);
        clustered.case_id = format!("clustered_n{n}");
        cases.push(clustered);
    }
    cases
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn scipy_oracle_or_skip(cases: &[MatrixCase]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, math, sys
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

out = []
for case in json.load(sys.stdin):
    arm = {"case_id": case["case_id"], "eigvals_by_magnitude": None}
    try:
        n = int(case["n"])
        A = csr_matrix((case["vals"], (case["rows"], case["cols"])), shape=(n, n))
        vals = eigsh(A, k=6, which="LM", return_eigenvectors=False)
        vals = sorted((float(v) for v in vals), key=lambda v: -abs(v))
        if all(math.isfinite(v) for v in vals):
            arm["eigvals_by_magnitude"] = vals
    except Exception as e:
        sys.stderr.write(f"oracle {case['case_id']}: {e}\n")
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(cases).expect("serialize eigsh k6 query");
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
                "failed to spawn python3 for the eigsh k6 oracle: {e}"
            );
            eprintln!("skipping eigsh k6 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open eigsh k6 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "eigsh k6 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping eigsh k6 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for eigsh k6 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "eigsh k6 oracle failed: {stderr}"
        );
        eprintln!("skipping eigsh k6 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse eigsh k6 oracle JSON"))
}

#[test]
fn diff_sparse_eigsh_k6_general() -> Result<(), String> {
    let cases = generate_cases();
    let Some(oracle) = scipy_oracle_or_skip(&cases) else {
        return Ok(());
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let mut diffs = Vec::new();
    let mut converged_count = 0_usize;
    let mut ledger = CompareLedger::new("diff_sparse_eigsh_k6_general", &["eigsh"]);
    for case in &cases {
        let scipy = arms
            .get(&case.case_id)
            .and_then(|arm| arm.eigvals_by_magnitude.as_deref());
        let csr = CooMatrix::from_triplets(
            Shape2D::new(case.n, case.n),
            case.vals.clone(),
            case.rows.clone(),
            case.cols.clone(),
            true,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = eigsh(&csr, K, EigsOptions::default());
        let fsci_converged = result.as_ref().is_ok_and(|r| r.converged);
        if fsci_converged {
            converged_count += 1;
        }
        let fsci_sorted = result.as_ref().ok().map(|r| {
            let mut v = r.eigenvalues.clone();
            v.sort_by(|x, y| y.abs().total_cmp(&x.abs()));
            v
        });
        println!(
            "{} n={} converged={fsci_converged} fsci={fsci_sorted:?} scipy={scipy:?} err={:?}",
            case.case_id,
            case.n,
            result.as_ref().err()
        );
        // An unconverged result is a failed case: it reaches the ledger as no fsci value.
        let fsci = fsci_sorted.filter(|_| fsci_converged);
        let Some((scipy, fsci)) = ledger.slices("eigsh", &case.case_id, scipy, fsci.as_deref())
        else {
            continue;
        };
        let worst = fsci
            .iter()
            .zip(scipy)
            .map(|(f, s)| (f - s).abs() / s.abs().max(1.0))
            .fold(0.0_f64, f64::max);
        let max_rel_diff = Some(worst);
        let pass = worst <= EIG_REL_TOL;
        ledger.compared("eigsh", &case.case_id, pass);
        println!("{} max_rel={max_rel_diff:?}", case.case_id);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            n: case.n,
            fsci_converged,
            max_rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_eigsh_k6_general".into(),
        category: "scipy.sparse.linalg.eigsh(k=6, which='LM') on runtime-generated matrices".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        converged_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create eigsh k6 diff dir");
    fs::write(
        output_dir().join("diff_sparse_eigsh_k6_general.json"),
        serde_json::to_string_pretty(&log).expect("serialize eigsh k6 log"),
    )
    .expect("write eigsh k6 log");

    println!("{converged_count} of {} converged", cases.len());
    assert_eq!(diffs.len(), cases.len(), "every matrix must be compared");
    assert_eq!(
        converged_count,
        diffs.len(),
        "every matrix must converge ({converged_count}/{})",
        diffs.len()
    );
    assert!(all_pass, "an eigsh result disagrees with SciPy");
    ledger.finish(cases.len());
    Ok(())
}
