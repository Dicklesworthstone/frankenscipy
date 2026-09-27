//! `eigsh`/`eigs` versus a live SciPy incumbent, in one process run (frankenscipy-f5kx5).
//!
//! 1ksfv.10 replaced eigsh's thick-restart Lanczos and eigs's single Arnoldi pass with
//! Krylov–Schur, so no earlier eigensolver ratio carries over. Each case sends the exact CSR
//! arrays and one start vector to a SciPy child once. Both sides then solve the same problem:
//! same `k`, `which`, `sigma`, `v0` and `tol = 0`, eigenvalues only on SciPy's side (ARPACK's
//! `return_eigenvectors=False`). fsci always forms its eigenvectors, which is charged to fsci.
//! Timing is position-balanced A-B-B-A / B-A-A-B rounds with A/A nulls. Before any timing, the
//! agreement of the eigenvalues is checked in the child, so a speed number never stands over a
//! numerical difference.
//!
//! Run: `perf_eigsh_vs_scipy [rounds]`; `FSCI_EIG_CASES=name,name` narrows the case list, and
//! `taskset -c N` gives the single-core rows.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::time::Instant;

use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use fsci_sparse::{CooMatrix, CsrMatrix, EigsOptions, EigsWhich, FormatConvertible, Shape2D};
use fsci_sparse::{eigs, eigsh};
use sha2::{Digest, Sha256};

const PYTHON: &str = r#"
import os, sys, time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh

n = int(os.environ['FSCI_EIG_N']); nnz = int(os.environ['FSCI_EIG_NNZ'])
k = int(os.environ['FSCI_EIG_K']); which = os.environ['FSCI_EIG_WHICH']
sigma = os.environ.get('FSCI_EIG_SIGMA', '')
sigma = None if sigma == '' else float(sigma)
solver = eigsh if os.environ['FSCI_EIG_SOLVER'] == 'eigsh' else eigs
stdin = sys.stdin.buffer

def read(count, dtype):
    raw = stdin.read(count * 8)
    if len(raw) != count * 8:
        raise RuntimeError('short fixture')
    return np.frombuffer(raw, dtype=dtype).copy()

indptr = read(n + 1, '<u8').astype(np.int64)
indices = read(nnz, '<u8').astype(np.int64)
data = read(nnz, '<f8')
v0 = read(n, '<f8')
A = csr_matrix((data, indices, indptr), shape=(n, n))

def run():
    return solver(A, k=k, which=which, sigma=sigma, v0=v0, tol=0, return_eigenvectors=False)

ref = np.asarray(run(), dtype=complex)
ref = ref[np.lexsort((ref.imag, ref.real))]
print(f'READY scipy={scipy.__version__} numpy={np.__version__} n={n} nnz={nnz} k={k} '
      f'which={which} sigma={sigma} tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1" and np.__version__ == "2.4.3"}', flush=True)

while True:
    line = stdin.readline()
    if not line:
        break
    cmd = line.decode('ascii').split()
    if not cmd:
        continue
    if cmd[0] == 'TIME':
        reps = int(cmd[1])
        t0 = time.perf_counter()
        for _ in range(reps):
            run()
        print(f'TIME {time.perf_counter() - t0:.17e}', flush=True)
    elif cmd[0] == 'CHECK':
        ours = read(2 * k, '<f8')
        ours = ours[:k] + 1j * ours[k:]
        ours = ours[np.lexsort((ours.imag, ours.real))]
        scale = max(np.max(np.abs(ref)), np.finfo(float).tiny)
        print(f'CHECK max_rel={np.max(np.abs(ours - ref)) / scale:.3e}', flush=True)
    elif cmd[0] == 'quit':
        break
    else:
        raise RuntimeError(f'bad command {cmd}')
"#;

/// Emit a result line on both stdout and stderr (rch relays stderr reliably).
macro_rules! emit {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        eprintln!("{line}");
    }};
}

fn incumbent() -> &'static ScipyIncumbent {
    static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
    INCUMBENT.get_or_init(|| {
        let resolved = ScipyIncumbent::resolve_with(&[], &["scipy.sparse.linalg"])
            .expect("resolve the pinned SciPy incumbent");
        println!("{}", resolved.provenance_line());
        resolved
    })
}

struct Scipy {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    ready: String,
}

impl Scipy {
    fn start(case: &Case, a: &CsrMatrix, v0: &[f64]) -> Self {
        let n = a.shape().rows;
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_EIG_N", n.to_string())
            .env("FSCI_EIG_NNZ", a.nnz().to_string())
            .env("FSCI_EIG_K", case.k.to_string())
            .env("FSCI_EIG_WHICH", case.which)
            .env(
                "FSCI_EIG_SIGMA",
                case.sigma.map_or_else(String::new, |s| format!("{s:e}")),
            )
            .env(
                "FSCI_EIG_SOLVER",
                if case.symmetric { "eigsh" } else { "eigs" },
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity(8 * (2 * n + 1 + 2 * a.nnz()));
        for &p in a.indptr() {
            bytes.extend_from_slice(&(p as u64).to_le_bytes());
        }
        for &j in a.indices() {
            bytes.extend_from_slice(&(j as u64).to_le_bytes());
        }
        for &value in a.data().iter().chain(v0) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        stdin.write_all(&bytes).expect("send fixture");
        stdin.flush().expect("flush fixture");
        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("read readiness");
        assert!(ready.starts_with("READY "), "not ready: {ready:?}");
        Self {
            child,
            stdin,
            stdout,
            ready: ready.trim().to_owned(),
        }
    }

    fn reply(&mut self) -> String {
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read reply");
        line.trim().to_owned()
    }

    /// Milliseconds per solve over `reps` solves.
    fn time(&mut self, reps: usize) -> f64 {
        writeln!(self.stdin, "TIME {reps}").expect("request timing");
        self.stdin.flush().expect("flush");
        let line = self.reply();
        let seconds: f64 = line
            .strip_prefix("TIME ")
            .expect("TIME reply")
            .parse()
            .expect("numeric timing");
        seconds * 1.0e3 / reps as f64
    }

    fn check(&mut self, re: &[f64], im: &[f64]) -> String {
        writeln!(self.stdin, "CHECK").expect("request check");
        let mut bytes = Vec::with_capacity(16 * re.len());
        for value in re.iter().chain(im) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.stdin.write_all(&bytes).expect("send result");
        self.stdin.flush().expect("flush");
        self.reply()
    }
}

impl Drop for Scipy {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

struct Case {
    name: &'static str,
    symmetric: bool,
    k: usize,
    which: &'static str,
    sigma: Option<f64>,
    matrix: fn() -> CsrMatrix,
}

fn from_triplets(n: usize, rows: Vec<usize>, cols: Vec<usize>, data: Vec<f64>) -> CsrMatrix {
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("valid triplets")
        .to_csr()
        .expect("csr")
}

/// Tridiagonal (−1, 2, −1): eigenvalues 2 − 2cos(jπ/(n + 1)), a clustered low end.
fn laplacian_1d(n: usize) -> CsrMatrix {
    let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(2.0);
        if i + 1 < n {
            rows.extend([i, i + 1]);
            cols.extend([i + 1, i]);
            data.extend([-1.0, -1.0]);
        }
    }
    from_triplets(n, rows, cols, data)
}

/// perf_eigsh's matrix: twelve planted, well-separated top eigenvalues (100, 88, ...) over a
/// bulk in [0, 1], with small symmetric off-diagonal bands at offsets 1, 4, 17, 53.
fn planted_banded(n: usize) -> CsrMatrix {
    let mut state = 0x1234_u64 ^ n as u64;
    let mut unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(if i < 12 {
            12.0f64.mul_add(-(i as f64), 100.0)
        } else {
            unit()
        });
        for off in [1_usize, 4, 17, 53] {
            if i + off < n {
                let w = unit().mul_add(2.0, -1.0) * 0.1;
                rows.extend([i, i + off]);
                cols.extend([i + off, i]);
                data.extend([w, w]);
            }
        }
    }
    from_triplets(n, rows, cols, data)
}

/// Five-point Laplacian on an m×m grid.
fn laplacian_2d(m: usize) -> CsrMatrix {
    let n = m * m;
    let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..m {
        for c in 0..m {
            let i = r * m + c;
            rows.push(i);
            cols.push(i);
            data.push(4.0);
            let mut link = |j: usize| {
                rows.push(i);
                cols.push(j);
                data.push(-1.0);
            };
            if r > 0 {
                link(i - m);
            }
            if r + 1 < m {
                link(i + m);
            }
            if c > 0 {
                link(i - 1);
            }
            if c + 1 < m {
                link(i + 1);
            }
        }
    }
    from_triplets(n, rows, cols, data)
}

/// Nonsymmetric 2-D convection–diffusion (upwind), m×m grid, with MILD convection.
///
/// Its eigenvalues are real and known, 4.4 − 2√(w·e)·cos(jπ/(m+1)) − 2cos(kπ/(m+1)), but the
/// eigenvector condition grows like (w/e)^(m/2). With w = 1.4, e = 0.6 (the first choice) that
/// is ~1e18 at m = 100. SciPy's own eigs(σ = 0) then returned complex pairs up to 0.01 away from
/// the real truth, and "agreement with SciPy" measured nothing. At w = 1.1, e = 0.9 the growth is
/// ~(1.22)^50 ≈ 2e4, and both solvers are held to a well-posed problem.
fn convection_diffusion_2d(m: usize) -> CsrMatrix {
    let n = m * m;
    let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..m {
        for c in 0..m {
            let i = r * m + c;
            rows.push(i);
            cols.push(i);
            data.push(4.4);
            let mut link = |j: usize, w: f64| {
                rows.push(i);
                cols.push(j);
                data.push(w);
            };
            if r > 0 {
                link(i - m, -1.0);
            }
            if r + 1 < m {
                link(i + m, -1.0);
            }
            if c > 0 {
                link(i - 1, -1.1);
            }
            if c + 1 < m {
                link(i + 1, -0.9);
            }
        }
    }
    from_triplets(n, rows, cols, data)
}

const CASES: &[Case] = &[
    // Largest-magnitude on the 1-D Laplacian is not a case: its top eigenvalues sit ~1e-8
    // apart at n = 20000 and both solvers crawl. The planted matrix is perf_eigsh's.
    Case {
        name: "planted_20000_LM_k6",
        symmetric: true,
        k: 6,
        which: "LM",
        sigma: None,
        matrix: || planted_banded(20_000),
    },
    Case {
        name: "lap1d_20000_sigma0.5_k6",
        symmetric: true,
        k: 6,
        which: "LM",
        sigma: Some(0.5),
        matrix: || laplacian_1d(20_000),
    },
    Case {
        name: "lap2d_100_LM_k1",
        symmetric: true,
        k: 1,
        which: "LM",
        sigma: None,
        matrix: || laplacian_2d(100),
    },
    Case {
        name: "lap2d_100_LM_k20",
        symmetric: true,
        k: 20,
        which: "LM",
        sigma: None,
        matrix: || laplacian_2d(100),
    },
    Case {
        name: "lap2d_60_SA_k6",
        symmetric: true,
        k: 6,
        which: "SA",
        sigma: None,
        matrix: || laplacian_2d(60),
    },
    Case {
        name: "convdiff2d_60_LR_k6",
        symmetric: false,
        k: 6,
        which: "LR",
        sigma: None,
        matrix: || convection_diffusion_2d(60),
    },
    Case {
        name: "convdiff2d_100_sigma0_k6",
        symmetric: false,
        k: 6,
        which: "LM",
        sigma: Some(0.0),
        matrix: || convection_diffusion_2d(100),
    },
];

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn elf_sha256() -> String {
    let exe = std::env::current_exe().expect("current exe");
    let bytes = std::fs::read(exe).expect("read own executable");
    format!("{:x}", Sha256::digest(bytes))
}

fn host_line() -> String {
    let first = |path: &str| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| s.lines().next().map(|l| l.trim().to_owned()))
            .unwrap_or_else(|| "unknown".into())
    };
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    format!(
        "host={} logical_threads={threads} governor={} loadavg=[{}]",
        first("/proc/sys/kernel/hostname"),
        first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        first("/proc/loadavg"),
    )
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let selected = std::env::var("FSCI_EIG_CASES").ok();
    emit!("elf_sha256={}", elf_sha256());
    emit!("provenance_before {}", host_line());
    for case in CASES {
        if let Some(list) = &selected
            && !list.split(',').any(|name| name.trim() == case.name)
        {
            continue;
        }
        let a = (case.matrix)();
        let n = a.shape().rows;
        let v0: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (0.7 * i as f64).sin()).collect();
        let which: EigsWhich = case.which.parse().expect("SciPy which spelling");
        let options = EigsOptions {
            which,
            sigma: case.sigma,
            v0: Some(&v0),
            ..EigsOptions::default()
        };
        let solve = || {
            if case.symmetric {
                eigsh(&a, case.k, options)
            } else {
                eigs(&a, case.k, options)
            }
            .expect("fsci eigensolve on the case matrix")
        };
        let mut scipy = Scipy::start(case, &a, &v0);
        println!("{}", scipy.ready);
        let first = solve();
        let check = scipy.check(&first.eigenvalues, &first.eigenvalues_im);

        let single = {
            let started = Instant::now();
            black_box(solve());
            started.elapsed().as_secs_f64() * 1.0e3
        };
        let reps = ((200.0 / single.max(1.0e-3)).ceil() as usize).clamp(1, 200);
        let time_ours = || {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(solve());
            }
            started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
        };
        let _ = scipy.time(1);
        let (mut fsci, mut sp, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for round in 0..rounds {
            let (a1, s1, s2, a2) = if round % 2 == 0 {
                let a1 = time_ours();
                let s1 = scipy.time(reps);
                let s2 = scipy.time(reps);
                (a1, s1, s2, time_ours())
            } else {
                let s1 = scipy.time(reps);
                let a1 = time_ours();
                let a2 = time_ours();
                (a1, s1, scipy.time(reps), a2)
            };
            fsci.push(a1.min(a2));
            sp.push(s1.min(s2));
            null_f.push(a1.max(a2) / a1.min(a2));
            null_s.push(s1.max(s2) / s1.min(s2));
        }
        let (f_ms, s_ms) = (median(fsci), median(sp));
        emit!(
            "case={} n={n} nnz={} k={} which={} sigma={:?} fsci={f_ms:.3}ms scipy={s_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} iterations={} nmatvec={} {check}",
            case.name,
            a.nnz(),
            case.k,
            case.which,
            case.sigma,
            s_ms / f_ms,
            median(null_f),
            median(null_s),
            first.iterations,
            first.nmatvec,
        );
    }
    emit!("provenance_after {}", host_line());
}
