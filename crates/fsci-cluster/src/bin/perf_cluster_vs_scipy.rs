//! Persistent live-SciPy whole-job benchmark for `fsci-cluster`.
//!
//! WHY THIS FILE EXISTS. `fsci-cluster` carries 41 perf binaries and none of them puts a live
//! SciPy arm next to ours. This is the third harness written to close that gap, after
//! `perf_signal_vs_scipy` and `perf_spatial_vs_scipy`; `fsci-interpolate` is the last crate
//! still without one.
//!
//! Both cases are ones SciPy compiles: `scipy.cluster.hierarchy.linkage` and
//! `scipy.cluster.vq.vq` are Cython, so a loss is the likely and therefore informative
//! outcome. Gates on nothing a running measurement cannot supply for itself — twelve
//! harnesses in this tree abort on a booking claim that is unsatisfiable (bead fr78g).
//!
//! Agreement is CHECKED by shipping our result back to the Python child. For `linkage` the
//! comparison is on the MERGE DISTANCES only, in SciPy's own row order, because the label of
//! a merged cluster is an implementation detail that two correct implementations may number
//! differently; the distances are not.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_cluster::{LinkageMethod, linkage, vq};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import vq

op = os.environ['FSCI_CLUSTER_OP']
n = int(os.environ['FSCI_CLUSTER_N'])
dim = int(os.environ['FSCI_CLUSTER_DIM'])
k = int(os.environ['FSCI_CLUSTER_K'])

raw = sys.stdin.buffer.read(n * dim * 8)
if len(raw) != n * dim * 8: raise RuntimeError('short fixture')
pts = np.frombuffer(raw, dtype='<f8').reshape(n, dim).copy()
craw = sys.stdin.buffer.read(k * dim * 8)
if len(craw) != k * dim * 8: raise RuntimeError('short centroids')
cent = np.frombuffer(craw, dtype='<f8').reshape(k, dim).copy()

def run():
    if op == 'linkage':
        return np.ascontiguousarray(linkage(pts, method='single')[:, 2], dtype='<f8')
    _codes, dist = vq(pts, cent)
    return np.ascontiguousarray(dist, dtype='<f8')

ref = run()
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} dim={dim} k={k} '
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
    fn start(op: &str, points: &[Vec<f64>], centroids: &[Vec<f64>]) -> Self {
        let dim = points[0].len();
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_CLUSTER_OP", op)
            .env("FSCI_CLUSTER_N", points.len().to_string())
            .env("FSCI_CLUSTER_DIM", dim.to_string())
            .env("FSCI_CLUSTER_K", centroids.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.cluster child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::new();
        for row in points.iter().chain(centroids.iter()) {
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

/// SHA-256 of this running executable via `sha256sum` — `fsci-cluster` has no `sha2`, and
/// taking a production dependency so a benchmark can print a digest is dependency smuggling.
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
    let n: usize = std::env::var("FSCI_CLUSTER_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let dim: usize = std::env::var("FSCI_CLUSTER_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let k: usize = std::env::var("FSCI_CLUSTER_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let rounds: usize = std::env::var("FSCI_CLUSTER_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("elf_sha256={}", elf_sha256());

    // `FSCI_CLUSTER_VQ_SOA=0` restores the `sq_dist` scan. Only-override-when-asked.
    match std::env::var("FSCI_CLUSTER_VQ_SOA").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_cluster::VQ_SOA_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_cluster::VQ_SOA_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    println!(
        "vq_soa_scan={}",
        fsci_cluster::VQ_SOA_SCAN.load(std::sync::atomic::Ordering::Relaxed)
    );

    let coord = |i: usize, d: usize| -> f64 {
        let key = (i * 2_654_435_761usize).wrapping_add(d * 40_503) % 100_003;
        key as f64 / 100_003.0 - 0.5
    };
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dim).map(|d| coord(i, d)).collect())
        .collect();
    let centroids: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..dim).map(|d| coord(i * 37 + 11, d)).collect())
        .collect();

    for op in ["vq", "linkage"] {
        let mut scipy = Scipy::start(op, &points, &centroids);
        println!("{}", scipy.ready);

        let ours = || -> Vec<f64> {
            if op == "vq" {
                vq(&points, &centroids).expect("fsci vq").1
            } else {
                linkage(&points, LinkageMethod::Single)
                    .expect("fsci linkage")
                    .iter()
                    .map(|row| row[2])
                    .collect()
            }
        };

        black_box(ours());
        let _ = scipy.time(1, 1);

        // CALIBRATE REPETITIONS SO A SAMPLE IS LONG ENOUGH TO TIME.
        //
        // `vq` at this size runs in ~0.2 ms and `linkage` in ~13 ms. Timing ONE call per
        // sample put the A/A nulls at 1.03-1.11, i.e. the window noise was larger than the 2%
        // bar every other harness in this campaign holds to, which makes a ratio an ordering
        // signal rather than a measurement. Repeat each sample until it spans at least
        // `MIN_SAMPLE_MS`; both arms use the SAME repetition count, so neither is advantaged.
        const MIN_SAMPLE_MS: f64 = 20.0;
        let single = {
            let started = Instant::now();
            black_box(ours());
            started.elapsed().as_secs_f64() * 1.0e3
        };
        let reps = ((MIN_SAMPLE_MS / single.max(1.0e-6)).ceil() as usize).clamp(1, 4096);
        println!("op={op} calibration single={single:.4}ms reps={reps}");

        let time_ours = || -> f64 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(ours());
            }
            started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
        };

        let (mut fsci, mut sp, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rounds {
            let a1 = time_ours();
            let s1 = scipy.time(reps, 1);
            let s2 = scipy.time(reps, 1);
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
