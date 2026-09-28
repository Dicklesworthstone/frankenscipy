#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.sparse.linalg.spilu` as ILUTP
//! (frankenscipy-1ksfv.11).
//!
//! SciPy's `spilu` is SuperLU's supernodal `dgsitrf`; fsci's is a column-by-column factorization
//! that follows `dgsitrf`'s drop, fill and pivot rules (see `fsci_sparse::spilu`). Bit parity is
//! not the contract — the supernodal arithmetic is not reproduced — so this compares what the
//! rules produce:
//!
//! * `perm_c`: the column permutation, EXACTLY. fsci's `Colamd` is a port of SuperLU's COLAMD and
//!   its etree postorder, so under COLAMD and NATURAL both sides factor the same `A·Pc`;
//! * `lu_nnz`: `nnz(L) + nnz(U)` (SciPy `ilu.L.nnz + ilu.U.nnz`) within ±25% of SciPy's;
//! * `gmres`, `bicgstab`: the number of preconditioner applications SciPy's `gmres`/`bicgstab`
//!   need with `M = spilu(A)` against fsci's step-for-step ports with fsci's `spilu`; fsci must
//!   converge (true residual ≤ rtol) using at most 1.5× SciPy's count. The bound is one-sided:
//!   a factor that preconditions BETTER than SciPy's is not a defect.
//!
//! Every matrix is built here and sent to the oracle as triplets, so both sides factor the same
//! matrix. The bead's reference matrix is included and SciPy's counts on it are pinned to the
//! bead's (17326 / 31508 / 52188 / 59159), which proves the oracle saw that matrix. The case
//! SciPy refuses (the saddle-point matrix under NATURAL: "Factor is exactly singular") is an
//! expected raise that fsci's Strict mode must refuse too.
//!
//! Not compared: `MMD_ATA` / `MMD_AT_PLUS_A`. fsci runs its exact minimum degree there, not
//! SuperLU's `genmmd`, so `A·Pc` differs from SciPy's and so, widely, can the preconditioner.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, IluOptions, IterativeSolveOptions,
    PermutationOrdering, Shape2D, SparseIluFactorization, bicgstab_preconditioned,
    gmres_preconditioned, spilu,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-004";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// |fsci − SciPy| / SciPy on nnz(L+U).
const NNZ_REL_TOL: f64 = 0.25;
/// fsci's preconditioner applications may be at most this multiple of SciPy's.
const KRYLOV_RATIO_TOL: f64 = 1.5;
/// Relative residual the bead's negative case 1 must reach with preconditioned GMRES.
const SKEW_GMRES_TOL: f64 = 1.0e-10;
const ARMS: [&str; 4] = ["perm_c", "lu_nnz", "gmres", "bicgstab"];
/// SciPy 1.17.1 on the bead's matrix, fill_factor = 20 (bead frankenscipy-1ksfv.11).
const BEAD_SCIPY_NNZ: [(f64, usize); 4] =
    [(1e-1, 17326), (1e-2, 31508), (1e-4, 52188), (1e-6, 59159)];

#[derive(Debug, Clone, Serialize)]
struct MatrixSpec {
    n: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseSpec {
    case_id: String,
    matrix: String,
    permc_spec: String,
    drop_tol: f64,
    fill_factor: f64,
    rtol: f64,
    gmres_maxiter: usize,
    bicgstab_maxiter: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    matrices: BTreeMap<String, MatrixSpec>,
    cases: Vec<CaseSpec>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct KrylovRun {
    psolves: usize,
    converged: bool,
    residual: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    perm_c: Option<Vec<usize>>,
    lu_nnz: Option<usize>,
    raised: Option<String>,
    gmres: Option<KrylovRun>,
    bicgstab: Option<KrylovRun>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    cases: Vec<OracleCase>,
}

#[derive(Debug, Clone, Serialize)]
struct FsciRun {
    #[serde(skip)]
    perm_c: Vec<usize>,
    lu_nnz: usize,
    l_nnz: usize,
    u_nnz: usize,
    ordering_used: String,
    off_diagonal_pivots: usize,
    dropped_u_entries: usize,
    dropped_l_entries: usize,
    supernodes: usize,
    relaxed_supernodes: usize,
    gmres: KrylovRun,
    bicgstab: KrylovRun,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    scipy_lu_nnz: Option<usize>,
    scipy_raised: Option<String>,
    scipy_gmres: Option<KrylovRun>,
    scipy_bicgstab: Option<KrylovRun>,
    fsci: Option<FsciRun>,
    fsci_error: Option<String>,
    perm_c_equal: Option<bool>,
    nnz_rel_diff: Option<f64>,
    gmres_ratio: Option<f64>,
    bicgstab_ratio: Option<f64>,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create spilu ilutp diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize spilu ilutp diff log");
    fs::write(path, json).expect("write spilu ilutp diff log");
}

// ─── Matrices ────────────────────────────────────────────────────────────────────────────────

struct Triplets {
    n: usize,
    entries: BTreeMap<(usize, usize), f64>,
}

impl Triplets {
    fn new(n: usize) -> Self {
        Self {
            n,
            entries: BTreeMap::new(),
        }
    }

    fn add(&mut self, row: usize, col: usize, value: f64) {
        *self.entries.entry((row, col)).or_insert(0.0) += value;
    }

    fn spec(&self) -> MatrixSpec {
        MatrixSpec {
            n: self.n,
            rows: self.entries.keys().map(|&(r, _)| r).collect(),
            cols: self.entries.keys().map(|&(_, c)| c).collect(),
            vals: self.entries.values().copied().collect(),
        }
    }
}

/// Diagonal 0, sub-diagonal −1, super-diagonal +1: the bead's negative case 1.
fn skew_tridiagonal(n: usize) -> Triplets {
    let mut t = Triplets::new(n);
    for i in 0..n {
        if i > 0 {
            t.add(i, i - 1, -1.0);
        }
        if i + 1 < n {
            t.add(i, i + 1, 1.0);
        }
    }
    t
}

/// m×m convection–diffusion, central differences scaled by h²: 4 on the diagonal, −1 ∓ cx on
/// the x neighbours, −1 ∓ cy on the y neighbours. `(40, 0.4, 0.0)` is the bead's matrix
/// `kron(I, diags([−1.4, 4, −0.6])) + kron(diags([−1, −1], [−1, 1]), I)`.
fn convection_diffusion(m: usize, cx: f64, cy: f64) -> Triplets {
    let mut t = Triplets::new(m * m);
    for j in 0..m {
        for i in 0..m {
            let k = j * m + i;
            t.add(k, k, 4.0);
            if i > 0 {
                t.add(k, k - 1, -1.0 - cx);
            }
            if i + 1 < m {
                t.add(k, k + 1, -1.0 + cx);
            }
            if j > 0 {
                t.add(k, k - m, -1.0 - cy);
            }
            if j + 1 < m {
                t.add(k, k + m, -1.0 + cy);
            }
        }
    }
    t
}

/// −Δu + pe·(u_x + u_y) on the unit square, h = 1/(m+1): cell Péclet c = pe·h/2 both ways.
fn peclet_convection_diffusion(m: usize, peclet: f64) -> Triplets {
    let c = peclet / (m as f64 + 1.0) / 2.0;
    convection_diffusion(m, c, c)
}

/// A deterministic LCG in [0, 1), so the circuit is the same on every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn index(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize) % n
    }
}

/// Unsymmetric circuit-like nodal matrix: a ring of resistors plus two random links per node
/// (conductances spread over three decades), a weak leak to ground, and n/5 controlled sources
/// that break symmetry.
fn circuit_like(n: usize, seed: u64) -> Triplets {
    let mut rng = Lcg(seed);
    let mut t = Triplets::new(n);
    let mut diagonal = vec![0.0; n];
    let link = |t: &mut Triplets, diagonal: &mut [f64], i: usize, j: usize, g: f64| {
        if i != j {
            diagonal[i] += g;
            diagonal[j] += g;
            t.add(i, j, -g);
            t.add(j, i, -g);
        }
    };
    for i in 0..n {
        let g = 0.5 + rng.next();
        link(&mut t, &mut diagonal, i, (i + 1) % n, g);
        for _ in 0..2 {
            let j = rng.index(n);
            let g = 10f64.powf(-2.0 + 3.0 * rng.next());
            link(&mut t, &mut diagonal, i, j, g);
        }
    }
    for value in &mut diagonal {
        *value += 1e-3 * (1.0 + rng.next());
    }
    for _ in 0..n / 5 {
        let i = rng.index(n);
        let j = rng.index(n);
        if i != j {
            t.add(i, j, 2.0 * (rng.next() - 0.5));
        }
    }
    for (i, &value) in diagonal.iter().enumerate() {
        t.add(i, i, value);
    }
    t
}

/// [[A, Bᵀ], [B, 0]] with A the m×m 2-D Laplacian and B a forward-difference "divergence" onto
/// (m−1)·m cells: every constraint row has a zero diagonal.
fn saddle_point(m: usize) -> Triplets {
    let laplacian = convection_diffusion(m, 0.0, 0.0);
    let nu = m * m;
    let np = (m - 1) * m;
    let mut t = Triplets::new(nu + np);
    for (&(r, c), &v) in &laplacian.entries {
        t.add(r, c, v);
    }
    let mut p = 0;
    for j in 0..m {
        for i in 0..m - 1 {
            let k = j * m + i;
            t.add(nu + p, k, -1.0);
            t.add(nu + p, k + 1, 1.0);
            t.add(k, nu + p, -1.0);
            t.add(k + 1, nu + p, 1.0);
            p += 1;
        }
    }
    t
}

fn csc_of(spec: &MatrixSpec) -> CscMatrix {
    CooMatrix::from_triplets(
        Shape2D::new(spec.n, spec.n),
        spec.vals.clone(),
        spec.rows.clone(),
        spec.cols.clone(),
        false,
    )
    .expect("oracle matrix coo")
    .to_csc()
    .expect("oracle matrix csc")
}

// ─── Cases ───────────────────────────────────────────────────────────────────────────────────

fn generate_query() -> OracleQuery {
    let mut matrices = BTreeMap::new();
    matrices.insert("skew_tridiag_100".to_owned(), skew_tridiagonal(100).spec());
    matrices.insert(
        "bead_convdiff_40".to_owned(),
        convection_diffusion(40, 0.4, 0.0).spec(),
    );
    for peclet in [10.0, 100.0, 1000.0] {
        matrices.insert(
            format!("convdiff_pe{peclet}"),
            peclet_convection_diffusion(40, peclet).spec(),
        );
    }
    matrices.insert(
        "poisson_40".to_owned(),
        convection_diffusion(40, 0.0, 0.0).spec(),
    );
    matrices.insert("circuit_1000".to_owned(), circuit_like(1000, 7).spec());
    matrices.insert("saddle_20".to_owned(), saddle_point(20).spec());

    let mut cases = Vec::new();
    let mut push = |matrix: &str, permc: &str, drop_tol: f64, fill_factor: f64, rtol: f64| {
        cases.push(CaseSpec {
            case_id: format!("{matrix}/{permc}/dt{drop_tol:e}/ff{fill_factor}"),
            matrix: matrix.to_owned(),
            permc_spec: permc.to_owned(),
            drop_tol,
            fill_factor,
            rtol,
            gmres_maxiter: 50,
            bicgstab_maxiter: 1000,
        });
    };
    for permc in ["COLAMD", "NATURAL"] {
        push("skew_tridiag_100", permc, 1e-4, 10.0, SKEW_GMRES_TOL);
        for &(drop_tol, _) in &BEAD_SCIPY_NNZ {
            push("bead_convdiff_40", permc, drop_tol, 20.0, 1e-8);
        }
        push("bead_convdiff_40", permc, 1e-4, 10.0, 1e-8);
        for matrix in [
            "convdiff_pe10",
            "convdiff_pe100",
            "convdiff_pe1000",
            "poisson_40",
            "circuit_1000",
            "saddle_20",
        ] {
            push(matrix, permc, 1e-4, 10.0, 1e-8);
        }
    }
    OracleQuery { matrices, cases }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu

q = json.load(sys.stdin)
mats = {}
for mid, m in q["matrices"].items():
    n = m["n"]
    mats[mid] = sp.csc_matrix(
        (np.array(m["vals"], dtype=float), (np.array(m["rows"]), np.array(m["cols"]))),
        shape=(n, n),
    )
out = []
for c in q["cases"]:
    A = mats[c["matrix"]]
    n = A.shape[0]
    rec = {"case_id": c["case_id"], "perm_c": None, "lu_nnz": None, "raised": None,
           "gmres": None, "bicgstab": None}
    try:
        f = spilu(A, drop_tol=c["drop_tol"], fill_factor=c["fill_factor"], permc_spec=c["permc_spec"])
    except RuntimeError as e:
        rec["raised"] = str(e)
        out.append(rec)
        continue
    rec["perm_c"] = [int(v) for v in f.perm_c]
    rec["lu_nnz"] = int(f.L.nnz + f.U.nnz)
    b = np.ones(n)
    for method in ("gmres", "bicgstab"):
        calls = [0]

        def psolve(v, f=f, calls=calls):
            calls[0] += 1
            return f.solve(v)

        M = LinearOperator(A.shape, matvec=psolve, dtype=float)
        if method == "gmres":
            x, info = gmres(A, b, M=M, rtol=c["rtol"], atol=0.0, maxiter=c["gmres_maxiter"])
        else:
            x, info = bicgstab(A, b, M=M, rtol=c["rtol"], atol=0.0, maxiter=c["bicgstab_maxiter"])
        residual = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
        rec[method] = {"psolves": calls[0], "converged": bool(info == 0), "residual": residual}
    out.append(rec)
print(json.dumps({"cases": out}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize spilu ilutp query");
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
                "failed to spawn python3 for spilu ilutp oracle: {e}"
            );
            eprintln!("skipping spilu ilutp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open spilu ilutp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "spilu ilutp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping spilu ilutp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for spilu ilutp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "spilu ilutp oracle failed: {stderr}"
        );
        eprintln!("skipping spilu ilutp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse spilu ilutp oracle JSON"))
}

// ─── fsci side ───────────────────────────────────────────────────────────────────────────────

fn ordering_of(permc: &str) -> PermutationOrdering {
    match permc {
        "NATURAL" => PermutationOrdering::Natural,
        _ => PermutationOrdering::Colamd,
    }
}

/// fsci's preconditioned Krylov run with `M = ilu`, counting preconditioner applications.
fn krylov(
    csr: &CsrMatrix,
    ilu: &SparseIluFactorization,
    method: &str,
    case: &CaseSpec,
) -> Result<KrylovRun, String> {
    let n = csr.shape().rows;
    let b = vec![1.0; n];
    let calls = Cell::new(0usize);
    let psolve = |r: &[f64]| {
        calls.set(calls.get() + 1);
        ilu.solve(r)
    };
    let result = if method == "gmres" {
        gmres_preconditioned(
            csr,
            &b,
            psolve,
            None,
            None,
            IterativeSolveOptions {
                tol: case.rtol,
                max_iter: Some(case.gmres_maxiter),
                ..IterativeSolveOptions::default()
            },
        )
    } else {
        bicgstab_preconditioned(
            csr,
            &b,
            psolve,
            None,
            IterativeSolveOptions {
                tol: case.rtol,
                max_iter: Some(case.bicgstab_maxiter),
                ..IterativeSolveOptions::default()
            },
        )
    }
    .map_err(|e| format!("{method}: {e:?}"))?;
    // The TRUE residual, recomputed here rather than taken from the solver's report.
    let mut ax = vec![0.0; n];
    for row in 0..n {
        for idx in csr.indptr()[row]..csr.indptr()[row + 1] {
            ax[row] += csr.data()[idx] * result.solution[csr.indices()[idx]];
        }
    }
    let residual = ax
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
        / (n as f64).sqrt();
    Ok(KrylovRun {
        psolves: calls.get(),
        converged: result.converged && residual <= case.rtol,
        residual,
    })
}

fn fsci_case(csc: &CscMatrix, case: &CaseSpec) -> Result<FsciRun, String> {
    let ilu = spilu(
        csc,
        IluOptions {
            ordering: ordering_of(&case.permc_spec),
            drop_tol: case.drop_tol,
            fill_factor: case.fill_factor,
            ..IluOptions::default()
        },
    )
    .map_err(|e| format!("spilu: {e:?}"))?;
    let csr = csc.to_csr().map_err(|e| format!("csr: {e:?}"))?;
    let stats = ilu.statistics;
    Ok(FsciRun {
        perm_c: ilu.perm_c().to_vec(),
        lu_nnz: ilu.lu_nnz(),
        l_nnz: ilu.l().nnz(),
        u_nnz: ilu.u().nnz(),
        ordering_used: format!("{:?}", ilu.ordering_used),
        off_diagonal_pivots: stats.off_diagonal_pivots,
        dropped_u_entries: stats.dropped_u_entries,
        dropped_l_entries: stats.dropped_l_entries,
        supernodes: stats.supernodes,
        relaxed_supernodes: stats.relaxed_supernodes,
        gmres: krylov(&csr, &ilu, "gmres", case)?,
        bicgstab: krylov(&csr, &ilu, "bicgstab", case)?,
    })
}

/// fsci passes a Krylov arm when it converged within `KRYLOV_RATIO_TOL` × SciPy's
/// preconditioner applications; when SciPy itself did not converge there is no count to
/// bound, and fsci passes only by converging.
fn krylov_verdict(scipy: KrylovRun, fsci: KrylovRun) -> (bool, f64) {
    let ratio = fsci.psolves as f64 / scipy.psolves.max(1) as f64;
    let pass = fsci.converged && (!scipy.converged || ratio <= KRYLOV_RATIO_TOL);
    (pass, ratio)
}

#[test]
fn diff_sparse_spilu_ilutp() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.cases.len(), query.cases.len());
    let by_id: HashMap<String, OracleCase> = oracle
        .cases
        .into_iter()
        .map(|c| (c.case_id.clone(), c))
        .collect();
    let csc: BTreeMap<&str, CscMatrix> = query
        .matrices
        .iter()
        .map(|(id, spec)| (id.as_str(), csc_of(spec)))
        .collect();

    let start = Instant::now();
    let mut ledger = CompareLedger::new("diff_sparse_spilu_ilutp", &ARMS);
    let mut diffs = Vec::new();
    let mut expected = BTreeMap::from([
        ("perm_c", 0usize),
        ("lu_nnz", 0),
        ("gmres", 0),
        ("bicgstab", 0),
    ]);

    for case in &query.cases {
        let scipy = by_id
            .get(&case.case_id)
            .expect("oracle answered every case");
        let fsci = fsci_case(&csc[case.matrix.as_str()], case);
        let mut diff = CaseDiff {
            case_id: case.case_id.clone(),
            scipy_lu_nnz: scipy.lu_nnz,
            scipy_raised: scipy.raised.clone(),
            scipy_gmres: scipy.gmres,
            scipy_bicgstab: scipy.bicgstab,
            fsci: fsci.as_ref().ok().cloned(),
            fsci_error: fsci.as_ref().err().cloned(),
            perm_c_equal: None,
            nnz_rel_diff: None,
            gmres_ratio: None,
            bicgstab_ratio: None,
            pass: true,
        };
        *expected.get_mut("lu_nnz").expect("arm") += 1;
        if scipy.raised.is_some() {
            // SciPy: "Factor is exactly singular". Strict fsci must refuse as well.
            let refused = fsci
                .as_ref()
                .err()
                .is_some_and(|e| e.contains("SingularMatrix"));
            ledger.expected_raise("lu_nnz", &case.case_id, refused);
            diff.pass = refused;
            diffs.push(diff);
            continue;
        }
        *expected.get_mut("perm_c").expect("arm") += 1;
        let fsci_perm = fsci.as_ref().ok().map(|run| run.perm_c.as_slice());
        if let Some((s, f)) =
            ledger.both("perm_c", &case.case_id, scipy.perm_c.as_deref(), fsci_perm)
        {
            let equal = s == f;
            diff.perm_c_equal = Some(equal);
            diff.pass &= equal;
            ledger.compared("perm_c", &case.case_id, equal);
        } else {
            diff.pass = false;
        }
        let fsci_nnz = fsci.as_ref().ok().map(|run| run.lu_nnz as f64);
        let scipy_nnz = scipy.lu_nnz.map(|nnz| nnz as f64);
        if let Some((s, f)) = ledger.pair("lu_nnz", &case.case_id, scipy_nnz, fsci_nnz) {
            let rel = (f - s).abs() / s;
            diff.nnz_rel_diff = Some(rel);
            diff.pass &= rel <= NNZ_REL_TOL;
            ledger.compared("lu_nnz", &case.case_id, rel <= NNZ_REL_TOL);
        } else {
            diff.pass = false;
        }
        for arm in ["gmres", "bicgstab"] {
            *expected.get_mut(arm).expect("arm") += 1;
            let scipy_run = if arm == "gmres" {
                scipy.gmres
            } else {
                scipy.bicgstab
            };
            let fsci_run = fsci.as_ref().ok().map(|run| {
                if arm == "gmres" {
                    run.gmres
                } else {
                    run.bicgstab
                }
            });
            if let Some((s, f)) = ledger.both(arm, &case.case_id, scipy_run, fsci_run) {
                let (pass, ratio) = krylov_verdict(s, f);
                if arm == "gmres" {
                    diff.gmres_ratio = Some(ratio);
                } else {
                    diff.bicgstab_ratio = Some(ratio);
                }
                diff.pass &= pass;
                ledger.compared(arm, &case.case_id, pass);
            } else {
                diff.pass = false;
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_spilu_ilutp".into(),
        category: "scipy.sparse.linalg.spilu (SuperLU ILUTP) fill and preconditioner quality"
            .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        let (f_nnz, f_g, f_b, ordering, pivots) =
            d.fsci.as_ref().map_or((0, 0, 0, "-".to_owned(), 0), |run| {
                (
                    run.lu_nnz,
                    run.gmres.psolves,
                    run.bicgstab.psolves,
                    run.ordering_used.clone(),
                    run.off_diagonal_pivots,
                )
            });
        eprintln!(
            "SPILU {:<42} perm_c_equal={:?} scipy_nnz={:>6} fsci_nnz={:>6} ({:+6.1}%) \
             ordering={ordering} off_diag_pivots={pivots} gmres M-applications scipy={:>4} \
             fsci={:>4} bicgstab scipy={:>4} fsci={:>4} raised={:?} err={:?} pass={}",
            d.case_id,
            d.perm_c_equal,
            d.scipy_lu_nnz.unwrap_or(0),
            f_nnz,
            100.0
                * d.scipy_lu_nnz
                    .map_or(0.0, |s| (f_nnz as f64 - s as f64) / s as f64),
            d.scipy_gmres.map_or(0, |r| r.psolves),
            f_g,
            d.scipy_bicgstab.map_or(0, |r| r.psolves),
            f_b,
            d.scipy_raised,
            d.fsci_error,
            d.pass
        );
    }

    // Negative case 1: the zero-diagonal skew tridiagonal factors and preconditions GMRES to
    // 1e-10 (ILU(0), which this replaced, stops at a zero pivot in the first column).
    for d in diffs
        .iter()
        .filter(|d| d.case_id.starts_with("skew_tridiag_100/"))
    {
        let run = d
            .fsci
            .as_ref()
            .expect("fsci spilu factors the zero-diagonal matrix");
        assert!(
            run.gmres.converged && run.gmres.residual <= SKEW_GMRES_TOL,
            "{}: preconditioned GMRES must reach {SKEW_GMRES_TOL}, got {:?}",
            d.case_id,
            run.gmres
        );
        assert!(
            run.off_diagonal_pivots > 0,
            "{}: no pivoting happened",
            d.case_id
        );
    }

    // Negative case 2: SciPy's counts on the bead's matrix are the bead's (so the oracle saw the
    // bead's matrix), and fsci's nnz(L+U) grows strictly as drop_tol falls.
    let mut previous = 0usize;
    for &(drop_tol, bead_count) in &BEAD_SCIPY_NNZ {
        let id = format!("bead_convdiff_40/COLAMD/dt{drop_tol:e}/ff20");
        let d = diffs
            .iter()
            .find(|d| d.case_id == id)
            .expect("bead sweep case");
        assert_eq!(
            d.scipy_lu_nnz,
            Some(bead_count),
            "{id}: SciPy's count must be the bead's, or this is not the bead's matrix"
        );
        let fsci_nnz = d
            .fsci
            .as_ref()
            .expect("fsci factors the bead matrix")
            .lu_nnz;
        assert!(
            fsci_nnz > previous,
            "{id}: nnz(L+U) must grow as drop_tol falls: {fsci_nnz} after {previous}"
        );
        previous = fsci_nnz;
    }

    // fill_factor is respected (every case here has fill_factor >= 10).
    for (case, d) in query.cases.iter().zip(&diffs) {
        if let Some(run) = &d.fsci {
            let spec = &query.matrices[&case.matrix];
            let cap = case.fill_factor * spec.vals.len() as f64 + spec.n as f64;
            assert!(
                run.lu_nnz as f64 <= cap,
                "{}: nnz(L+U) = {} exceeds fill_factor·nnz(A) + n = {cap}",
                d.case_id,
                run.lu_nnz
            );
        }
    }

    assert!(
        all_pass,
        "spilu ILUTP conformance failed: {} of {} cases",
        diffs.iter().filter(|d| !d.pass).count(),
        diffs.len()
    );
    let counts = ledger.finish(expected.values().copied().min().unwrap_or(0));
    for (arm, cases) in expected {
        assert_eq!(
            counts[arm].compared_cases, cases,
            "arm `{arm}` must compare all {cases} of its cases"
        );
    }
}
