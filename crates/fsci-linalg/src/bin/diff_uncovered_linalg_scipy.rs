//! Differential probe for `fsci-linalg` entry points that SciPy names but the conformance
//! corpus never mentions (`frankenscipy-ivxx6`).
//!
//! WHY THIS EXISTS. `ivxx6` found 201 SciPy-named public entry points with no reference
//! anywhere in the conformance corpus, and states the consequence precisely: an uncovered
//! entry point CANNOT support a vs-SciPy row, because a ratio between two implementations
//! nobody has shown compute the same thing is not a measurement. The one member sampled so
//! far (`RbfInterpolator`, `frankenscipy-icozs`) turned out to implement SciPy's
//! non-default `degree=-1` variant under SciPy's default name. This samples five more from
//! the `fsci-linalg` block of that list.
//!
//! WHAT IT CHECKS. Each entry point is compared against LIVE SciPy 1.17.1 in the same
//! invocation, on fixtures chosen so that a wrong CONVENTION shows up rather than only a
//! wrong arithmetic:
//!
//!   * `cholesky_banded`   — banded storage is a convention (upper vs lower packing, which
//!                           diagonal sits in which row); a transposed packing still
//!                           produces a plausible triangular factor.
//!   * `eigvals_banded`    — same storage question, plus ordering: SciPy returns ascending.
//!   * `diagsvd`           — pure shape/placement; wrong only if m/n are swapped.
//!   * `pinvh`             — needs a matrix with a genuinely small eigenvalue, or the
//!                           cutoff logic is never exercised and any pinv looks right.
//!   * `orthogonal_procrustes` — the returned R is only unique up to the sign convention
//!                           SciPy's SVD picks; also returns a scale that is easy to define
//!                           differently.
//!
//! REPORTING. Every case prints `max_abs` and `max_rel` against SciPy's own output plus a
//! verdict, and the run exits non-zero if any case exceeds its tolerance, so this is a
//! probe that can fail rather than a report that always prints. Agreement here is NOT a
//! conformance test — it is evidence that one can now be written, which is what `ivxx6`
//! asks for.

use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.linalg"];

/// The one live-SciPy incumbent this process compares against, resolved once and PROVEN by
/// running the import rather than by a name resolving on `PATH`.
///
/// This harness used to spawn a bare `python3`. On `thinkstation1` that is 3.14 with no
/// SciPy at all, so the oracle died on its first write with `BrokenPipe` and the run read as
/// a flaky pipe rather than as a missing incumbent (frankenscipy-m5s54). Resolving names the
/// interpreter, and prints the scipy AND numpy versions it proved, before anything is timed.
fn incumbent() -> &'static ScipyIncumbent {
    static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
    INCUMBENT.get_or_init(|| {
        let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
            .unwrap_or_else(|error| panic!("{error}"));
        println!("{}", resolved.provenance_line());
        resolved
    })
}

use fsci_linalg::{
    DecompOptions, cholesky_banded, diagsvd, eigvals_banded, orthogonal_procrustes, pinvh,
};

const PYTHON: &str = r#"
import json, sys
import numpy as np
import scipy
from scipy.linalg import cholesky_banded, diagsvd, eig_banded, eigvals_banded
from scipy.linalg import orthogonal_procrustes, pinvh

req = json.loads(sys.stdin.readline())
out = {"scipy": scipy.__version__, "numpy": np.__version__}

# --- cholesky_banded: lower-form banded SPD ---
ab = np.array(req["ab"], dtype=np.float64)
out["cholesky_banded_lower"] = cholesky_banded(ab, lower=True).tolist()

# --- eigvals_banded: same storage, ascending eigenvalues ---
out["eigvals_banded_lower"] = eigvals_banded(ab, lower=True).tolist()

# --- diagsvd ---
s = np.array(req["s"], dtype=np.float64)
out["diagsvd"] = diagsvd(s, req["m"], req["n"]).tolist()

# --- pinvh on a Hermitian matrix with a small eigenvalue ---
h = np.array(req["h"], dtype=np.float64)
out["pinvh"] = pinvh(h).tolist()

# --- orthogonal_procrustes ---
a = np.array(req["pa"], dtype=np.float64)
b = np.array(req["pb"], dtype=np.float64)
r, scale = orthogonal_procrustes(a, b)
out["procrustes_r"] = r.tolist()
out["procrustes_scale"] = float(scale)

def emit(key, arr):
    a = np.atleast_1d(np.asarray(arr, dtype=np.float64)).ravel()
    print(key, a.size, " ".join(repr(float(v)) for v in a), flush=True)

print("META", scipy.__version__, np.__version__, flush=True)
emit("cholesky_banded_lower", out["cholesky_banded_lower"])
emit("eigvals_banded_lower", out["eigvals_banded_lower"])
emit("diagsvd", out["diagsvd"])
emit("pinvh", out["pinvh"])
emit("procrustes_r", out["procrustes_r"])
emit("procrustes_scale", out["procrustes_scale"])
print("END", flush=True)
"#;

fn flat(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

/// Worst absolute and relative deviation between two equally-shaped flat buffers.
fn deviation(ours: &[f64], theirs: &[f64]) -> (f64, f64) {
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / b.abs().max(f64::MIN_POSITIVE));
    }
    (max_abs, max_rel)
}

fn report(name: &str, ours: &[f64], theirs: &[f64], tol: f64, failures: &mut Vec<String>) {
    if ours.len() != theirs.len() {
        println!(
            "case={name} VERDICT=SHAPE_MISMATCH ours_len={} scipy_len={}",
            ours.len(),
            theirs.len()
        );
        failures.push(format!("{name}: shape {} vs {}", ours.len(), theirs.len()));
        return;
    }
    let (max_abs, max_rel) = deviation(ours, theirs);
    let ok = max_abs <= tol;
    println!(
        "case={name} max_abs={max_abs:.6e} max_rel={max_rel:.6e} tol={tol:.1e} \
         VERDICT={}",
        if ok { "AGREES" } else { "DIVERGES" }
    );
    if !ok {
        failures.push(format!("{name}: max_abs={max_abs:.6e} > {tol:.1e}"));
    }
}

fn main() {
    // Banded SPD in SciPy's LOWER form: row 0 is the main diagonal, row 1 the first
    // subdiagonal, padded with a trailing zero. Diagonally dominant so it is SPD.
    let n = 6usize;
    let diag: Vec<f64> = (0..n).map(|i| 4.0 + i as f64 * 0.25).collect();
    let sub: Vec<f64> = (0..n)
        .map(|i| {
            if i + 1 < n {
                -1.0 - i as f64 * 0.1
            } else {
                0.0
            }
        })
        .collect();
    let ab = vec![diag.clone(), sub.clone()];

    let s = vec![3.5, 2.0, 0.75];
    let (m_svd, n_svd) = (5usize, 3usize);

    // Hermitian with one deliberately tiny eigenvalue, so pinvh's cutoff is exercised.
    let h = vec![
        vec![2.0, 0.5, 0.0],
        vec![0.5, 2.0, 0.0],
        vec![0.0, 0.0, 1.0e-13],
    ];

    let pa = vec![
        vec![1.0, 0.2, -0.3],
        vec![0.4, 1.1, 0.25],
        vec![-0.2, 0.35, 0.9],
        vec![0.6, -0.15, 0.45],
    ];
    let pb = vec![
        vec![0.9, -0.25, 0.4],
        vec![0.3, 1.05, -0.2],
        vec![-0.15, 0.4, 0.95],
        vec![0.5, 0.2, 0.35],
    ];

    let request = serde_json_line(&ab, &s, m_svd, n_svd, &h, &pa, &pb);

    let mut child = incumbent()
        .command()
        .args(["-u", "-c", PYTHON])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn live scipy.linalg child");
    let mut stdin = child.stdin.take().expect("python stdin");
    writeln!(stdin, "{request}").expect("send request");
    stdin.flush().expect("flush request");
    drop(stdin);
    let mut reader = BufReader::new(child.stdout.take().expect("python stdout"));
    let mut reply: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut meta = String::new();
    let mut line = String::new();
    while reader.read_line(&mut line).expect("read scipy reply") > 0 {
        let trimmed = line.trim().to_owned();
        line.clear();
        if trimmed == "END" {
            break;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(key) = parts.next() else { continue };
        if key == "META" {
            meta = parts.collect::<Vec<_>>().join(" ");
            continue;
        }
        let count: usize = parts
            .next()
            .unwrap_or_else(|| panic!("no length for {key}"))
            .parse()
            .unwrap_or_else(|_| panic!("bad length for {key}"));
        let values: Vec<f64> = parts
            .map(|t| {
                t.parse()
                    .unwrap_or_else(|_| panic!("bad number {t:?} under {key}"))
            })
            .collect();
        assert_eq!(
            values.len(),
            count,
            "{key}: declared {count} values, parsed {}",
            values.len()
        );
        reply.insert(key.to_owned(), values);
    }
    let _ = child.wait();
    assert!(
        !reply.is_empty(),
        "scipy arm produced no values — a silently empty reply would make every case \
         look like a shape mismatch rather than naming the real failure"
    );
    println!("scipy_meta={meta}");
    let get = |key: &str| -> Vec<f64> {
        reply
            .get(key)
            .unwrap_or_else(|| panic!("scipy reply is missing {key}"))
            .clone()
    };

    let mut failures: Vec<String> = Vec::new();

    let ours = cholesky_banded(&ab, true).expect("fsci cholesky_banded");
    report(
        "cholesky_banded/lower",
        &flat(&ours),
        &get("cholesky_banded_lower"),
        1e-12,
        &mut failures,
    );

    let ours = eigvals_banded(&ab, true, DecompOptions::default()).expect("fsci eigvals_banded");
    report(
        "eigvals_banded/lower",
        &ours,
        &get("eigvals_banded_lower"),
        1e-10,
        &mut failures,
    );

    let ours = diagsvd(&s, m_svd, n_svd).expect("fsci diagsvd");
    report("diagsvd", &flat(&ours), &get("diagsvd"), 0.0, &mut failures);

    let ours = pinvh(&h, None, None).expect("fsci pinvh");
    report(
        "pinvh/small-eigenvalue",
        &flat(&ours),
        &get("pinvh"),
        1e-9,
        &mut failures,
    );

    let (r, scale) = orthogonal_procrustes(&pa, &pb).expect("fsci orthogonal_procrustes");
    report(
        "orthogonal_procrustes/R",
        &flat(&r),
        &get("procrustes_r"),
        1e-10,
        &mut failures,
    );
    report(
        "orthogonal_procrustes/scale",
        &[scale],
        &get("procrustes_scale"),
        1e-10,
        &mut failures,
    );

    // SELF-CHECK (must-hit). "ALL AGREE" is indistinguishable from "the comparison is
    // inert" without an arm that MUST fail. Perturb SciPy's own cholesky_banded answer by
    // one part in 10^6 and require the same comparator to call it a divergence. Earlier in
    // this probe's life the reply parser returned empty vectors and every case reported a
    // shape mismatch; a silently-empty reply is the realistic failure here, and this arm
    // plus the length assertion on the reply are what make it loud.
    let mut control_failures: Vec<String> = Vec::new();
    let reference = get("cholesky_banded_lower");
    let mut perturbed = reference.clone();
    assert!(!perturbed.is_empty(), "control needs a non-empty reference");
    perturbed[0] += 1.0e-6 * perturbed[0].abs().max(1.0);
    report(
        "SELFCHECK/perturbed-must-diverge",
        &perturbed,
        &reference,
        1e-12,
        &mut control_failures,
    );
    assert_eq!(
        control_failures.len(),
        1,
        "the comparator did NOT flag a deliberately perturbed value, so every AGREES \
         verdict above is meaningless"
    );
    println!("selfcheck=OK comparator flags a 1e-6 relative perturbation");

    if failures.is_empty() {
        println!("ALL AGREE — these five can now carry a conformance reference (ivxx6)");
    } else {
        println!("DIVERGENCES ({}):", failures.len());
        for f in &failures {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}

/// Hand-rolled request encoder: `fsci-linalg` does not depend on serde and a probe must not
/// add a production dependency to print JSON (dependency smuggling).
fn serde_json_line(
    ab: &[Vec<f64>],
    s: &[f64],
    m: usize,
    n: usize,
    h: &[Vec<f64>],
    pa: &[Vec<f64>],
    pb: &[Vec<f64>],
) -> String {
    let mat = |rows: &[Vec<f64>]| -> String {
        let inner: Vec<String> = rows
            .iter()
            .map(|r| {
                let cells: Vec<String> = r.iter().map(|v| format!("{v:?}")).collect();
                format!("[{}]", cells.join(","))
            })
            .collect();
        format!("[{}]", inner.join(","))
    };
    let vec1 = |v: &[f64]| -> String {
        let cells: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
        format!("[{}]", cells.join(","))
    };
    format!(
        "{{\"ab\":{},\"s\":{},\"m\":{},\"n\":{},\"h\":{},\"pa\":{},\"pb\":{}}}",
        mat(ab),
        vec1(s),
        m,
        n,
        mat(h),
        mat(pa),
        mat(pb)
    )
}
