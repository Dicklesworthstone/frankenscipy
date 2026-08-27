//! Persistent live-SciPy whole-job benchmark for `fsci-spatial`.
//!
//! WHY THIS FILE EXISTS. `fsci-spatial` carries 37 perf binaries and none of them puts a live
//! SciPy arm next to ours; `fsci-cluster` and `fsci-interpolate` are in the same position.
//! This is the second harness written to close that gap, after `perf_signal_vs_scipy`.
//!
//! The two cases are chosen because SciPy is at its STRONGEST there, so a loss is the likely
//! outcome and therefore the informative one: `scipy.spatial.distance.pdist` is a hand-written
//! C loop, and `cKDTree.query` is C++ with its own node layout. Comparing against SciPy's
//! numpy-level code would mostly measure Python overhead.
//!
//! Gates on nothing a running measurement cannot supply for itself — twelve harnesses in this
//! tree abort on a booking claim that is unsatisfiable (bead fr78g), and copying that would
//! make this one unrunnable too. Agreement is CHECKED by shipping our result back to the
//! Python child rather than compared through two separately-computed digests.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_spatial::{DistanceMetric, KDTree, pdist};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

op = os.environ['FSCI_SPATIAL_OP']
n = int(os.environ['FSCI_SPATIAL_N'])
dim = int(os.environ['FSCI_SPATIAL_DIM'])
nq = int(os.environ['FSCI_SPATIAL_NQ'])

raw = sys.stdin.buffer.read(n * dim * 8)
if len(raw) != n * dim * 8: raise RuntimeError('short fixture')
pts = np.frombuffer(raw, dtype='<f8').reshape(n, dim).copy()
qraw = sys.stdin.buffer.read(nq * dim * 8)
if len(qraw) != nq * dim * 8: raise RuntimeError('short queries')
qry = np.frombuffer(qraw, dtype='<f8').reshape(nq, dim).copy()

tree = cKDTree(pts) if op == 'kdtree' else None

def run():
    if op == 'pdist':
        return pdist(pts, metric='euclidean')
    d, _i = tree.query(qry, k=1)
    return d

ref = np.ascontiguousarray(run(), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} dim={dim} nq={nq} '
      f'fixture_sha256={hashlib.sha256(raw).hexdigest()} '
      f'tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1"} out_len={ref.size}', flush=True)

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
        m = ref.size
        raw_ours = sys.stdin.buffer.read(m * 8)
        if len(raw_ours) != m * 8: raise RuntimeError('short Rust result')
        ours = np.frombuffer(raw_ours, dtype='<f8')
        diff = np.abs(ours - ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e}', flush=True)
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
    fn start(op: &str, points: &[Vec<f64>], queries: &[Vec<f64>]) -> Self {
        let dim = points[0].len();
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_SPATIAL_OP", op)
            .env("FSCI_SPATIAL_N", points.len().to_string())
            .env("FSCI_SPATIAL_DIM", dim.to_string())
            .env("FSCI_SPATIAL_NQ", queries.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.spatial child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::new();
        for row in points.iter().chain(queries.iter()) {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
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

/// SHA-256 of this running executable, via `sha256sum` — `fsci-spatial` has no `sha2`, and
/// adding one so a benchmark can print a digest would be a production dependency taken on for
/// a measurement.
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
    let n: usize = std::env::var("FSCI_SPATIAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let dim: usize = std::env::var("FSCI_SPATIAL_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let nq: usize = std::env::var("FSCI_SPATIAL_NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let rounds: usize = std::env::var("FSCI_SPATIAL_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("elf_sha256={}", elf_sha256());

    // Deterministic and well spread, so the tree is balanced and no metric degenerates.
    let coord = |i: usize, d: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(d * 40_503) % 100_003;
        k as f64 / 100_003.0 - 0.5
    };
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dim).map(|d| coord(i, d)).collect())
        .collect();
    let queries: Vec<Vec<f64>> = (0..nq)
        .map(|i| (0..dim).map(|d| coord(i + n, d)).collect())
        .collect();

    for op in ["pdist", "kdtree"] {
        let mut scipy = Scipy::start(op, &points, &queries);
        println!("{}", scipy.ready);

        let tree = (op == "kdtree").then(|| KDTree::new(&points).expect("build fsci KDTree"));

        let ours = || -> Vec<f64> {
            if op == "pdist" {
                pdist(&points, DistanceMetric::Euclidean).expect("fsci pdist")
            } else {
                let tree = tree.as_ref().expect("tree");
                queries
                    .iter()
                    .map(|q| tree.query(q).expect("fsci query").1)
                    .collect()
            }
        };

        black_box(ours());
        let _ = scipy.time(1, 1);

        let time_ours = || -> f64 {
            let started = Instant::now();
            let value = ours();
            let elapsed = started.elapsed().as_secs_f64() * 1.0e3;
            black_box(value);
            elapsed
        };

        // A-B-B-A per round so each arm carries its own A/A null across the same window.
        let (mut fsci, mut sp, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rounds {
            let a1 = time_ours();
            let s1 = scipy.time(1, 1);
            let s2 = scipy.time(1, 1);
            let a2 = time_ours();
            fsci.push(a1.min(a2));
            sp.push(s1.min(s2));
            null_f.push(a1.max(a2) / a1.min(a2));
            null_s.push(s1.max(s2) / s1.min(s2));
        }

        let (fsci_ms, scipy_ms) = (median(fsci), median(sp));
        let check = scipy.check(&ours());
        println!(
            "case=n{n}d{dim} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
