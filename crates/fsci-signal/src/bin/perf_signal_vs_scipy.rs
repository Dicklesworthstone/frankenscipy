//! Persistent live-SciPy whole-job benchmark for `fsci-signal`.
//!
//! WHY THIS FILE EXISTS. `fsci-signal` carries 70 perf binaries and NOT ONE of them puts a
//! live SciPy arm next to ours — the same is true of `fsci-spatial`, `fsci-cluster` and
//! `fsci-interpolate`, 189 perf binaries between the four. Every vs-incumbent number this
//! campaign has is from linalg, sparse, fft, ndimage or integrate, so an entire quadrant of
//! the op space has never been compared against the thing it reimplements.
//!
//! Modelled on `perf_separable_filter`, which is the one live-SciPy harness in the tree that
//! is NOT behind the booking claim. Twelve others are — `curve_fit`, `minimize`, `newton`,
//! `root`, `gmres_job`, `quad`, `dblquad`, `tplquad`, `sparse_nnz`, `cdf`,
//! `truncweibull` and `normality` all abort on `TRJ_BOOKING_CLAIM_MESSAGE_ID` or
//! `FSCI_CLAIM_MESSAGE_ID`, which this campaign established is unsatisfiable
//! (`acquire_build_slot` disabled server-side, bead fr78g). This harness deliberately gates
//! on nothing a running measurement cannot supply for itself.
//!
//! Both arms are timed in ONE invocation with per-arm A/A nulls, and the Rust result is sent
//! back to the Python child so agreement is CHECKED rather than assumed — the FFT harness's
//! two arms compute their digests with different algorithms and therefore verify nothing
//! about each other, which is a trap worth not repeating.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_signal::{ConvolveMode, convolve, lfilter};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy import signal

op = os.environ['FSCI_SIGNAL_OP']
n = int(os.environ['FSCI_SIGNAL_N'])
raw = sys.stdin.buffer.read(n * 8)
if len(raw) != n * 8: raise RuntimeError('short fixture')
x = np.frombuffer(raw, dtype='<f8').copy()

# A 4th-order Butterworth low-pass, and a 65-tap kernel for the convolution.
b, a = signal.butter(4, 0.25)
kernel = np.exp(-0.5 * ((np.arange(65) - 32.0) / 8.0) ** 2)
kernel = kernel / kernel.sum()

def run():
    if op == 'lfilter':
        return signal.lfilter(b, a, x)
    return np.convolve(x, kernel, mode='full')

ref = run()
print('COEFF ' + ','.join(f'{v:.17e}' for v in b) + ' ' + ','.join(f'{v:.17e}' for v in a)
      + ' ' + ','.join(f'{v:.17e}' for v in kernel), flush=True)
print(f'READY scipy={scipy.__version__} numpy={np.__version__} n={n} op={op} '
      f'fixture_sha256={hashlib.sha256(raw).hexdigest()} '
      f'tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1"} out_len={len(ref)}', flush=True)

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
        m = len(ref)
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
    b: Vec<f64>,
    a: Vec<f64>,
    kernel: Vec<f64>,
}

fn parse_list(text: &str) -> Vec<f64> {
    text.split(',')
        .map(|v| v.parse().expect("coefficient is a float"))
        .collect()
}

impl Scipy {
    fn start(op: &str, x: &[f64]) -> Self {
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_SIGNAL_OP", op)
            .env("FSCI_SIGNAL_N", x.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.signal child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity(x.len() * 8);
        for value in x {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        stdin.write_all(&bytes).expect("send fixture");
        stdin.flush().expect("flush fixture");

        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut coeff = String::new();
        stdout.read_line(&mut coeff).expect("read coefficients");
        let fields: Vec<&str> = coeff.trim().split(' ').collect();
        assert_eq!(fields[0], "COEFF", "expected COEFF, got {coeff:?}");
        let (b, a, kernel) = (
            parse_list(fields[1]),
            parse_list(fields[2]),
            parse_list(fields[3]),
        );
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("read readiness");
        assert!(ready.starts_with("READY scipy="), "not ready: {ready:?}");
        Self {
            child,
            stdin,
            stdout,
            ready: ready.trim().to_owned(),
            b,
            a,
            kernel,
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

/// SHA-256 of this running executable, via `sha256sum`.
///
/// Shelled out rather than hashed in-process ON PURPOSE: `fsci-signal` has no `sha2`
/// dependency, and adding one so a perf binary can print a digest would be pulling a
/// production dependency in for a benchmark. `perf_fft_vs_scipy` resolves it the same way.
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
    let n: usize = std::env::var("FSCI_SIGNAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);
    let rounds: usize = std::env::var("FSCI_SIGNAL_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("elf_sha256={}", elf_sha256());

    // `FSCI_SIGNAL_LOW_ORDER_MAX=<n>` sets the unrolled-lfilter gate so both shapes live in
    // ONE binary; 3 is the pre-widening value. Only-override-when-asked, so an unset
    // variable keeps the shipping default.
    if let Ok(value) = std::env::var("FSCI_SIGNAL_LOW_ORDER_MAX")
        && let Ok(value) = value.parse::<usize>()
    {
        fsci_signal::LFILTER_LOW_ORDER_MAX_NFILT.store(value, std::sync::atomic::Ordering::Relaxed);
    }
    println!(
        "lfilter_low_order_max_nfilt={}",
        fsci_signal::LFILTER_LOW_ORDER_MAX_NFILT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Deterministic, bounded, and not pathological for an IIR filter.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 7.0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 61.0 * t).cos()
        })
        .collect();

    for op in ["lfilter", "convolve"] {
        let mut scipy = Scipy::start(op, &x);
        println!("{}", scipy.ready);

        let ours = |scipy: &Scipy| -> Vec<f64> {
            if op == "lfilter" {
                lfilter(&scipy.b, &scipy.a, &x, None).expect("fsci lfilter")
            } else {
                convolve(&x, &scipy.kernel, ConvolveMode::Full).expect("fsci convolve")
            }
        };

        // Warm both arms before either is timed.
        black_box(ours(&scipy));
        let _ = scipy.time(1, 1);

        let time_ours = |scipy: &Scipy| -> f64 {
            let started = Instant::now();
            let value = ours(scipy);
            let elapsed = started.elapsed().as_secs_f64() * 1.0e3;
            black_box(value);
            elapsed
        };

        // A-B-B-A per round, so each arm carries its own A/A null across the same window.
        let (mut fsci, mut scipy_ms, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rounds {
            let a1 = time_ours(&scipy);
            let s1 = scipy.time(1, 1);
            let s2 = scipy.time(1, 1);
            let a2 = time_ours(&scipy);
            fsci.push(a1.min(a2));
            scipy_ms.push(s1.min(s2));
            null_f.push(a1.max(a2) / a1.min(a2));
            null_s.push(s1.max(s2) / s1.min(s2));
        }

        let (fsci_ms, scipy_ms) = (median(fsci), median(scipy_ms));
        let check = scipy.check(&ours(&scipy));
        println!(
            "case=n{n} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
