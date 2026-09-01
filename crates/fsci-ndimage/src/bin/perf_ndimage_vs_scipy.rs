//! Persistent live-SciPy whole-job benchmark for `fsci-ndimage`.
//!
//! WHY THIS FILE EXISTS. `fsci-ndimage` carries 64 tracked perf binaries and not one of
//! them puts a live SciPy arm next to ours, so every number the crate reports is a
//! self-timing. This is the seventh live-SciPy harness, after signal, spatial, cluster,
//! interpolate, stats and special. `fsci-opt` (25 tracked bins) is the last crate with no
//! live incumbent.
//!
//! Writing these has found a new worst cell in the suite twice running — `ks_2samp` at 55x
//! and `gammaln` at 2.07x — so an unmeasured crate keeps being where the large gaps are.
//!
//! Every case is compiled C in SciPy (`scipy.ndimage` is a C extension), so a loss is the
//! likely and therefore informative outcome. The four span different kernel shapes on
//! purpose: `gaussian_filter` and `uniform_filter` are SEPARABLE per-axis passes,
//! `median_filter` is a rank filter with no separable form, and `distance_transform_edt`
//! is a global two-pass algorithm rather than a neighbourhood one. A harness that only
//! measured separable convolutions would say nothing about the other two shapes.
//!
//! AGREEMENT IS CHECKED by shipping our result back to the Python child and comparing
//! elementwise against SciPy's own output, so a speed number can never be reported over a
//! numerical difference. Boundary handling is where these implementations most often
//! disagree, so the fixture is deliberately small enough that the border is a meaningful
//! fraction of it.

use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.ndimage"];

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

use fsci_ndimage::{
    BoundaryMode, NdArray, distance_transform_edt, gaussian_filter, label, median_filter,
    uniform_filter,
};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.ndimage import (distance_transform_edt, gaussian_filter, label, median_filter,
                           uniform_filter)

op = os.environ['FSCI_ND_OP']
n = int(os.environ['FSCI_ND_N'])
sigma = float(os.environ['FSCI_ND_SIGMA'])
size = int(os.environ['FSCI_ND_SIZE'])

raw = sys.stdin.buffer.read(n * n * 8)
if len(raw) != n * n * 8: raise RuntimeError('short fixture')
img = np.frombuffer(raw, dtype='<f8').reshape(n, n).copy()

if op == 'gaussian':
    def run(): return gaussian_filter(img, sigma=sigma, mode='reflect')
elif op == 'uniform':
    def run(): return uniform_filter(img, size=size, mode='reflect')
elif op == 'median':
    def run(): return median_filter(img, size=size, mode='reflect')
elif op == 'label':
    def run(): return label(img)[0]
else:
    # EDT takes a binary image; threshold so roughly half the pixels are background.
    binary = (img > 0.5).astype(np.float64)
    def run(): return distance_transform_edt(binary)

ref = np.ascontiguousarray(run(), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} '
      f'sigma={sigma} size={size} '
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
        ours = np.frombuffer(raw_ours, dtype='<f8').reshape(ref.shape)
        diff = np.abs(ours - ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        # Border pixels are where boundary handling differs; report them separately so an
        # interior-only agreement cannot be mistaken for whole-array agreement.
        b = max(1, int(os.environ.get('FSCI_ND_BORDER', '8')))
        interior = diff[b:-b, b:-b]
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e} '
              f'max_abs_interior={np.max(interior):.17e} '
              f'border_worse={bool(np.max(diff) > np.max(interior))}', flush=True)
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
    fn start(op: &str, n: usize, sigma: f64, size: usize, image: &[f64]) -> Self {
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_ND_OP", op)
            .env("FSCI_ND_N", n.to_string())
            .env("FSCI_ND_SIGMA", sigma.to_string())
            .env("FSCI_ND_SIZE", size.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.ndimage child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity(image.len() * 8);
        for value in image {
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
    let n = env_usize("FSCI_ND_N", 512);
    let size = env_usize("FSCI_ND_SIZE", 5);
    let rounds = env_usize("FSCI_ND_ROUNDS", 5);
    let sigma: f64 = std::env::var("FSCI_ND_SIGMA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    let selected =
        std::env::var("FSCI_ND_OPS").unwrap_or_else(|_| "gaussian,uniform,median,edt".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_ND_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    // `FSCI_ND_MEDIAN_DIRECT=0` restores the sliding rank tree. Only-override-when-asked,
    // so an unset variable cannot overwrite a flipped library default.
    match std::env::var("FSCI_ND_MEDIAN_DIRECT").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_ndimage::NDIMAGE_MEDIAN_DIRECT_SELECT
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_ndimage::NDIMAGE_MEDIAN_DIRECT_SELECT
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }

    println!("elf_sha256={}", elf_sha256());
    println!(
        "median_direct={}",
        fsci_ndimage::NDIMAGE_MEDIAN_DIRECT_SELECT.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "n={n} sigma={sigma} size={size} available_parallelism={}",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    );

    // Deterministic, and NOT smooth: a smooth image would let a rank filter's running
    // structure stay valid across many pixels and would flatter whichever implementation
    // exploits that. This mixes a low-frequency component with per-pixel hash noise.
    let image: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (r, c) = (idx / n, idx % n);
            let k = (idx * 2_654_435_761usize).wrapping_add(40_503) % 1_000_003;
            let noise = k as f64 / 1_000_003.0;
            0.5 * ((r as f64 * 0.05).sin() + (c as f64 * 0.037).cos()) * 0.5 + 0.5 + 0.35 * noise
        })
        .collect();
    let binary: Vec<f64> = image
        .iter()
        .map(|&v| if v > 0.85 { 1.0 } else { 0.0 })
        .collect();

    for op in ["gaussian", "uniform", "median", "edt", "label"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        // Built ONCE, outside the timed region: SciPy's arm is handed an array it already
        // holds, so cloning the image into a fresh NdArray per call would charge us a
        // memcpy the incumbent never pays — the defect that cost 23% of a measured number
        // in the fsci-special harness.
        let source = if op == "edt" || op == "label" {
            &binary
        } else {
            &image
        };
        let mut scipy = Scipy::start(op, n, sigma, size, source);
        println!("{}", scipy.ready);
        let array = NdArray::new(source.clone(), vec![n, n]).expect("build NdArray");

        let ours = || -> Vec<f64> {
            let out = match op {
                "gaussian" => gaussian_filter(&array, sigma, BoundaryMode::Reflect, 0.0),
                "uniform" => uniform_filter(&array, size, BoundaryMode::Reflect, 0.0),
                "median" => median_filter(&array, size, BoundaryMode::Reflect, 0.0),
                "label" => label(&array).map(|(labels, _)| labels),
                _ => distance_transform_edt(&array, None),
            };
            out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")).data
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
            "case=n{n} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
