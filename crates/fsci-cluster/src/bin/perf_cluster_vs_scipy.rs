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

use fsci_cluster::{LinkageMethod, kmeans2, linkage, vq};
use fsci_runtime::scipy_incumbent::ScipyIncumbent;

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.cluster.hierarchy"];

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
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans2, vq

op = os.environ['FSCI_CLUSTER_OP']
n = int(os.environ['FSCI_CLUSTER_N'])
dim = int(os.environ['FSCI_CLUSTER_DIM'])
k = int(os.environ['FSCI_CLUSTER_K'])
iterations = int(os.environ['FSCI_CLUSTER_ITER'])

raw = sys.stdin.buffer.read(n * dim * 8)
if len(raw) != n * dim * 8: raise RuntimeError('short fixture')
pts = np.frombuffer(raw, dtype='<f8').reshape(n, dim).copy()
craw = sys.stdin.buffer.read(k * dim * 8)
if len(craw) != k * dim * 8: raise RuntimeError('short centroids')
cent = np.frombuffer(craw, dtype='<f8').reshape(k, dim).copy()

def run():
    if op == 'linkage':
        return np.ascontiguousarray(linkage(pts, method='single')[:, 2], dtype='<f8')
    if op == 'kmeans2':
        _centroids, labels = kmeans2(pts, cent, iter=iterations, minit='matrix')
        return np.ascontiguousarray(labels, dtype='<f8')
    _codes, dist = vq(pts, cent)
    return np.ascontiguousarray(dist, dtype='<f8')

ref = run()
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} dim={dim} k={k} '
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
    fn start(op: &str, points: &[Vec<f64>], centroids: &[Vec<f64>], iterations: usize) -> Self {
        let dim = points[0].len();
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_CLUSTER_OP", op)
            .env("FSCI_CLUSTER_N", points.len().to_string())
            .env("FSCI_CLUSTER_DIM", dim.to_string())
            .env("FSCI_CLUSTER_K", centroids.len().to_string())
            .env("FSCI_CLUSTER_ITER", iterations.to_string())
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
    let iterations: usize = std::env::var("FSCI_CLUSTER_ITER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

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

    // `FSCI_CLUSTER_MST_FUSED=0` restores the two-pass Prim's inner loop. Only-override-
    // when-asked: storing the parsed value unconditionally would overwrite a newly flipped
    // library default with `false` whenever the variable is unset.
    match std::env::var("FSCI_CLUSTER_MST_FUSED").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_cluster::MST_FUSED_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_cluster::MST_FUSED_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    println!(
        "mst_fused_scan={}",
        fsci_cluster::MST_FUSED_SCAN.load(std::sync::atomic::Ordering::Relaxed)
    );

    // `FSCI_CLUSTER_DM_BLOCKED=0` restores the one-pair-at-a-time `sq_dist` triangle fill;
    // `FSCI_CLUSTER_DM_FORCE_BLOCK=1` restores the pre-fix single-thread block fill that
    // evaluated every pair twice. Both only-override-when-asked.
    match std::env::var("FSCI_CLUSTER_DM_BLOCKED").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_cluster::LINKAGE_DM_BLOCKED_FOLD.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_cluster::LINKAGE_DM_BLOCKED_FOLD
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    match std::env::var("FSCI_CLUSTER_DM_FORCE_BLOCK").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_cluster::LINKAGE_DM_FORCE_BLOCK_FILL
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_cluster::LINKAGE_DM_FORCE_BLOCK_FILL
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    println!(
        "dm_blocked_fold={} dm_force_block_fill={} available_parallelism={}",
        fsci_cluster::LINKAGE_DM_BLOCKED_FOLD.load(std::sync::atomic::Ordering::Relaxed),
        fsci_cluster::LINKAGE_DM_FORCE_BLOCK_FILL.load(std::sync::atomic::Ordering::Relaxed),
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
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

    // `FSCI_CLUSTER_OPS=vq` restricts the run to one op. Needed to attribute a hardware
    // counter to a single op: with both ops in the run, a `perf stat` total mixes them and
    // a change in either moves the number.
    let selected =
        std::env::var("FSCI_CLUSTER_OPS").unwrap_or_else(|_| "vq,linkage,kmeans2".to_owned());
    // `FSCI_CLUSTER_FIXED_REPS=N` pins the repetition count instead of calibrating it, so two
    // runs being compared on instructions retired do the SAME amount of work. Calibration
    // picks reps from measured time, so a faster arm would otherwise run FEWER reps and
    // retire fewer instructions for that reason alone.
    let fixed_reps: Option<usize> = std::env::var("FSCI_CLUSTER_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    for op in ["vq", "linkage", "kmeans2"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        let mut scipy = Scipy::start(op, &points, &centroids, iterations);
        println!("{}", scipy.ready);

        let ours = || -> Vec<f64> {
            if op == "vq" {
                vq(&points, &centroids).expect("fsci vq").1
            } else if op == "kmeans2" {
                kmeans2(&points, &centroids, iterations)
                    .expect("fsci kmeans2")
                    .1
                    .into_iter()
                    .map(|label| label as f64)
                    .collect()
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
        // Calibrate off the FASTEST of several calls, not one. A single call carries cold
        // caches and any scheduler hiccup that lands on it, so it over-estimates the cost
        // and under-estimates `reps`. That was visible once `linkage` got faster: a 4.8 ms
        // op measured 11.9-20.5 ms on its first call, which set `reps` to 1-2 and left
        // samples at ~10 ms against a 20 ms target, pushing the A/A nulls to 1.10-1.16.
        // The repetition count is shared by both arms either way, so this sharpens the
        // nulls rather than moving the ratio.
        const MIN_SAMPLE_MS: f64 = 20.0;
        const CALIBRATION_CALLS: usize = 3;
        let mut single = f64::INFINITY;
        for _ in 0..CALIBRATION_CALLS {
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
        if op == "linkage" {
            // Stage split of OUR arm, over one freshly-zeroed call, so the two stages are
            // attributed against the same invocation the ratio above was measured on.
            for slot in &fsci_cluster::LINKAGE_STAGE_NANOS {
                slot.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            black_box(ours());
            let build =
                fsci_cluster::LINKAGE_STAGE_NANOS[0].load(std::sync::atomic::Ordering::Relaxed);
            let agglomerate =
                fsci_cluster::LINKAGE_STAGE_NANOS[1].load(std::sync::atomic::Ordering::Relaxed);
            let total = (build + agglomerate).max(1);
            println!(
                "op=linkage stages dm_build={:.3}ms ({:.1}%) agglomerate={:.3}ms ({:.1}%)",
                build as f64 / 1.0e6,
                100.0 * build as f64 / total as f64,
                agglomerate as f64 / 1.0e6,
                100.0 * agglomerate as f64 / total as f64,
            );
        }
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
