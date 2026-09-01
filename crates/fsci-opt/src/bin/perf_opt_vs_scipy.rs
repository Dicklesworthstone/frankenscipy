//! Persistent live-SciPy whole-job benchmark for `fsci-opt`.
//!
//! WHY THIS FILE EXISTS. `fsci-opt` carries 25 tracked perf binaries and not one of them
//! puts a live SciPy arm next to ours. This is the eighth live-SciPy harness and the LAST
//! crate to get one, after signal, spatial, cluster, interpolate, stats, special and
//! ndimage. Writing these has found a new worst cell three times — `ks_2samp` at 55x,
//! `gammaln` at 2.07x, `median_filter` at 21.4x — so an unmeasured crate has been where the
//! large gaps live.
//!
//! CHOOSING OPS HERE NEEDS MORE CARE THAN IN THE OTHER CRATES. Most of `scipy.optimize`
//! calls a PYTHON CALLBACK once per iteration — `minimize`, `least_squares`, `brentq`,
//! `curve_fit` all do. Beating those measures Python's per-call overhead against a Rust
//! closure, not our algorithm against theirs, and would produce a large flattering number
//! that means nothing. (The same trap as benchmarking FITPACK on shuffled queries.) So the
//! three cases here take DATA IN AND DATA OUT with no callback:
//!
//!   * `linear_sum_assignment` — SciPy's is a COMPILED builtin (C++ Jonker-Volgenant).
//!     `inspect.getsource` cannot retrieve it. This is the fully fair comparison.
//!   * `linprog` — a Python wrapper whose actual solve is HiGHS in C++. Fair once the
//!     problem is big enough for the solve to dominate the wrapper, which the size here is.
//!   * `nnls` — pure Python/numpy in SciPy (88 lines), so its inner loop pays Python
//!     per-iteration cost. Reported, but flagged: a win here is partly Python overhead and
//!     should NOT be read as an algorithmic result.
//!
//! AGREEMENT IS CHECKED by shipping our result back to the Python child. For the assignment
//! problem the comparison is on the ACHIEVED COST, not the permutation: a cost matrix with
//! ties has several optimal assignments, and two correct solvers may return different ones.
//! The cost is what the problem asks for; the permutation is an implementation detail.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_opt::{linear_sum_assignment, linprog, nnls};
use fsci_runtime::scipy_incumbent::ScipyIncumbent;

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.optimize"];

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

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment, linprog, nnls

op = os.environ['FSCI_OPT_OP']
n = int(os.environ['FSCI_OPT_N'])
m = int(os.environ['FSCI_OPT_M'])

raw = sys.stdin.buffer.read(n * m * 8)
if len(raw) != n * m * 8: raise RuntimeError('short fixture')
mat = np.frombuffer(raw, dtype='<f8').reshape(n, m).copy()
vraw = sys.stdin.buffer.read(max(n, m) * 8)
if len(vraw) != max(n, m) * 8: raise RuntimeError('short vector')
vec = np.frombuffer(vraw, dtype='<f8').copy()

if op == 'assignment':
    def run():
        r, c = linear_sum_assignment(mat)
        # Compare ACHIEVED COST, not the permutation: ties admit several optima.
        return np.array([mat[r, c].sum()], dtype='<f8')
elif op == 'nnls':
    def run():
        x, rnorm = nnls(mat, vec[:n])
        return np.ascontiguousarray(np.append(x, rnorm), dtype='<f8')
else:
    # min c^T x  s.t.  A_ub x <= b_ub,  x >= 0.  Feasible and bounded by construction:
    # every A entry is positive and every b is positive, so x = 0 is feasible and the
    # optimum is finite.
    c = vec[:m]
    b_ub = np.abs(vec[:n]) * m * 0.5 + 1.0
    def run():
        r = linprog(c, A_ub=mat, b_ub=b_ub, bounds=[(0, None)] * m, method='highs')
        return np.array([r.fun if r.success else np.nan], dtype='<f8')

ref = np.ascontiguousarray(run(), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} m={m} '
      f'fixture_sha256={hashlib.sha256(raw).hexdigest()} '
      f'tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1" and np.__version__ == "2.4.3"} out_len={ref.size}', flush=True)

for line in sys.stdin.buffer:
    cmd = line.decode('ascii').strip().split()
    if not cmd: continue
    if cmd[0] == 'TIME':
        reps, minimum = int(cmd[1]), int(cmd[2])
        best = float('inf')
        for _ in range(minimum):
            t0 = time.perf_counter()
            for _ in range(reps): out = run()
            best = min(best, time.perf_counter() - t0)
        print(f'TIME {best:.17e}', flush=True)
    elif cmd[0] == 'CHECK':
        raw_ours = sys.stdin.buffer.read(ref.size * 8)
        if len(raw_ours) != ref.size * 8: raise RuntimeError('short Rust result')
        ours = np.frombuffer(raw_ours, dtype='<f8')
        diff = np.abs(ours - ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e} '
              f'compared={ref.size}', flush=True)
    elif cmd[0] == 'quit':
        break
    else:
        raise RuntimeError(f'bad command {cmd}')
"#;

struct Scipy {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    ready: String,
}

impl Scipy {
    fn start(op: &str, n: usize, m: usize, matrix: &[f64], vector: &[f64]) -> Self {
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_OPT_OP", op)
            .env("FSCI_OPT_N", n.to_string())
            .env("FSCI_OPT_M", m.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.optimize child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity((matrix.len() + vector.len()) * 8);
        for value in matrix.iter().chain(vector.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        stdin.write_all(&bytes).expect("send fixture");
        stdin.flush().expect("flush fixture");

        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("read readiness");
        assert!(ready.starts_with("READY scipy="), "not ready: {ready:?}");
        Self {
            child,
            stdin,
            stdout,
            ready: ready.trim().to_owned(),
        }
    }

    fn time(&mut self, reps: usize, minimum: usize) -> f64 {
        writeln!(self.stdin, "TIME {reps} {minimum}").expect("request timing");
        self.stdin.flush().expect("flush timing");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read timing");
        let value: f64 = line
            .trim()
            .strip_prefix("TIME ")
            .expect("TIME reply")
            .parse()
            .expect("numeric timing");
        value * 1.0e3 / reps as f64
    }

    fn check(&mut self, ours: &[f64]) -> String {
        writeln!(self.stdin, "CHECK").expect("request check");
        let mut bytes = Vec::with_capacity(ours.len() * 8);
        for value in ours {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.stdin.write_all(&bytes).expect("send result");
        self.stdin.flush().expect("flush result");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read check");
        line.trim().to_owned()
    }
}

impl Drop for Scipy {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

/// SHA-256 of the running executable, so a row names the binary that produced it and a
/// stale build cannot masquerade as a fresh one. Shelled out rather than hashed in
/// process: taking a production dependency so a benchmark can print a digest would be
/// dependency smuggling.
fn elf_sha256() -> String {
    let exe = std::env::current_exe().expect("current exe");
    let output = Command::new("sha256sum")
        .arg(exe)
        .output()
        .expect("run sha256sum");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("digest")
        .to_owned()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let env_usize = |key: &str, fallback: usize| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback)
    };
    let n = env_usize("FSCI_OPT_N", 256);
    let m = env_usize("FSCI_OPT_M", 256);
    let rounds = env_usize("FSCI_OPT_ROUNDS", 5);
    let selected =
        std::env::var("FSCI_OPT_OPS").unwrap_or_else(|_| "assignment,nnls,linprog".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_OPT_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    // `FSCI_OPT_NNLS_BLOCKED=0` restores the per-row rank-1 Gram sweep.
    match std::env::var("FSCI_OPT_NNLS_BLOCKED").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_opt::NNLS_GRAM_BLOCKED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_opt::NNLS_GRAM_BLOCKED.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    match std::env::var("FSCI_OPT_NNLS_DIRECT_QR").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_opt::NNLS_DIRECT_QR.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_opt::NNLS_DIRECT_QR.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }

    println!("elf_sha256={}", elf_sha256());
    println!(
        "n={n} m={m} nnls_gram_blocked={} nnls_direct_qr={}",
        fsci_opt::NNLS_GRAM_BLOCKED.load(std::sync::atomic::Ordering::Relaxed),
        fsci_opt::NNLS_DIRECT_QR.load(std::sync::atomic::Ordering::Relaxed),
    );

    // Deterministic and STRICTLY POSITIVE, which keeps the linprog instance feasible and
    // bounded (x = 0 is feasible, the optimum is finite) without needing a repair step that
    // would differ between the two arms.
    let unit = |i: usize, salt: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(salt * 40_503) % 1_000_003;
        k as f64 / 1_000_003.0
    };
    let matrix_flat: Vec<f64> = (0..n * m).map(|i| 0.05 + unit(i, 7)).collect();
    let matrix: Vec<Vec<f64>> = matrix_flat.chunks_exact(m).map(<[f64]>::to_vec).collect();
    let vector: Vec<f64> = (0..n.max(m)).map(|i| 0.05 + unit(i, 29)).collect();

    for op in ["assignment", "nnls", "linprog"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        let mut scipy = Scipy::start(op, n, m, &matrix_flat, &vector);
        println!("{}", scipy.ready);

        let b_ub: Vec<f64> = vector[..n]
            .iter()
            .map(|v| v.abs() * m as f64 * 0.5 + 1.0)
            .collect();
        let bounds: Vec<(Option<f64>, Option<f64>)> = vec![(Some(0.0), None); m];
        let empty_rows: Vec<Vec<f64>> = Vec::new();

        let ours = || -> Vec<f64> {
            match op {
                "assignment" => {
                    let (rows, cols) = linear_sum_assignment(&matrix).expect("fsci assignment");
                    let cost: f64 = rows
                        .iter()
                        .zip(cols.iter())
                        .map(|(&r, &c)| matrix[r][c])
                        .sum();
                    vec![cost]
                }
                "nnls" => {
                    let (x, rnorm) = nnls(&matrix, &vector[..n]).expect("fsci nnls");
                    let mut out = x;
                    out.push(rnorm);
                    out
                }
                _ => {
                    let result = linprog(
                        &vector[..m],
                        &matrix,
                        &b_ub,
                        &empty_rows,
                        &[],
                        &bounds,
                        None,
                    )
                    .expect("fsci linprog");
                    vec![if result.success { result.fun } else { f64::NAN }]
                }
            }
        };

        black_box(ours());
        let _ = scipy.time(1, 1);

        // Calibrate off the FASTEST of several calls: one call carries cold caches and any
        // scheduler hiccup that lands on it, which over-estimates cost and under-sizes
        // `reps`, leaving samples too short and the A/A nulls loose. Both arms share the
        // repetition count, so this sharpens the nulls without moving the ratio.
        const MIN_SAMPLE_MS: f64 = 20.0;
        let mut single = f64::INFINITY;
        for _ in 0..3 {
            let started = Instant::now();
            black_box(ours());
            single = single.min(started.elapsed().as_secs_f64() * 1.0e3);
        }
        let reps = fixed_reps
            .unwrap_or_else(|| (MIN_SAMPLE_MS / single.max(1.0e-6)).ceil() as usize)
            .clamp(1, 4096);
        println!("op={op} calibration single={single:.4}ms reps={reps}");

        let time_ours = || -> f64 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(ours());
            }
            started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
        };

        // A-B-B-A per round, FLIPPED to B-A-A-B on odd rounds. Interleaving alone cancels
        // drift but not POSITION: in a fixed A-B-B-A one arm's samples are always the
        // outermost, separated by both of the other's, so its A/A null spans a longer
        // interval and reads worse for reasons that have nothing to do with the code.
        let (mut fsci, mut sp, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for round in 0..rounds {
            let (a1, s1, s2, a2) = if round % 2 == 0 {
                let a1 = time_ours();
                let s1 = scipy.time(reps, 1);
                let s2 = scipy.time(reps, 1);
                let a2 = time_ours();
                (a1, s1, s2, a2)
            } else {
                let s1 = scipy.time(reps, 1);
                let a1 = time_ours();
                let a2 = time_ours();
                let s2 = scipy.time(reps, 1);
                (a1, s1, s2, a2)
            };
            fsci.push(a1.min(a2));
            sp.push(s1.min(s2));
            null_f.push(a1.max(a2) / a1.min(a2));
            null_s.push(s1.max(s2) / s1.min(s2));
        }

        let (fsci_ms, scipy_ms) = (median(fsci), median(sp));
        let check = scipy.check(&ours());
        println!(
            "case=n{n}m{m} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );

        if op == "nnls" {
            // The SciPy comparison above runs with the shipping QR arm. Pair it with the
            // incumbent normal-equations arm IN THIS PROCESS so a source/ELF/fixture mismatch
            // cannot turn a historical self-number into a result.
            let time_arm = |direct_qr: bool| {
                fsci_opt::NNLS_DIRECT_QR.store(direct_qr, std::sync::atomic::Ordering::Relaxed);
                time_ours()
            };
            let (mut qr, mut gram, mut null_qr, mut null_gram) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for round in 0..rounds {
                let (q1, g1, g2, q2) = if round % 2 == 0 {
                    let q1 = time_arm(true);
                    let g1 = time_arm(false);
                    let g2 = time_arm(false);
                    let q2 = time_arm(true);
                    (q1, g1, g2, q2)
                } else {
                    let g1 = time_arm(false);
                    let q1 = time_arm(true);
                    let q2 = time_arm(true);
                    let g2 = time_arm(false);
                    (q1, g1, g2, q2)
                };
                qr.push(q1.min(q2));
                gram.push(g1.min(g2));
                null_qr.push(q1.max(q2) / q1.min(q2));
                null_gram.push(g1.max(g2) / g1.min(g2));
            }
            let q0 = fsci_opt::NNLS_DIRECT_QR_HITS.load(std::sync::atomic::Ordering::Relaxed);
            fsci_opt::NNLS_DIRECT_QR.store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = ours();
            let q1 = fsci_opt::NNLS_DIRECT_QR_HITS.load(std::sync::atomic::Ordering::Relaxed);
            fsci_opt::NNLS_DIRECT_QR.store(false, std::sync::atomic::Ordering::Relaxed);
            let _ = ours();
            let q2 = fsci_opt::NNLS_DIRECT_QR_HITS.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(q1, q0 + 1, "direct QR toggle did not hit exactly once");
            assert_eq!(
                q2, q1,
                "normal-equations arm incremented the QR hit counter"
            );
            fsci_opt::NNLS_DIRECT_QR.store(true, std::sync::atomic::Ordering::Relaxed);
            let qr_ms = median(qr);
            let gram_ms = median(gram);
            println!(
                "armab=nnls_direct_qr qr={qr_ms:.3}ms gram={gram_ms:.3}ms \
                 gram/qr={:.3}x null_qr={:.3} null_gram={:.3} hits_on={} hits_off={}",
                gram_ms / qr_ms,
                median(null_qr),
                median(null_gram),
                q1 - q0,
                q2 - q1,
            );
        }
    }
}
