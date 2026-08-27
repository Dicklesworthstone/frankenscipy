//! Persistent live-SciPy whole-job benchmark for `fsci-special`.
//!
//! WHY THIS FILE EXISTS. `fsci-special` carries 120 perf binaries and not one of them puts
//! a live SciPy arm next to ours, so every number the crate reports is a self-timing. This
//! is the sixth live-SciPy harness, after signal, spatial, cluster, interpolate and stats.
//! `fsci-ndimage` (64 bins) and `fsci-opt` (25) are the two crates still uncovered.
//!
//! Writing the fifth of these found the worst cell in the whole suite (`ks_2samp`, 55x
//! slower), so an unmeasured crate is where the large gaps live, not the well-trodden ones.
//!
//! Every case is a `scipy.special` UFUNC — compiled Cephes C called once over a whole
//! array, with essentially no per-element Python cost. That makes a loss the likely and
//! therefore informative outcome, and it makes the comparison honest: SciPy's arm is one
//! vectorized call, so ours is the crate's BATCH entry point rather than a scalar loop.
//! (Timing `*_scalar` in a loop against a ufunc would compare two different jobs — the
//! mistake that manufactured a phantom loss in the interpolate harness.)
//!
//! AGREEMENT IS CHECKED by shipping our result back to the Python child and comparing
//! elementwise against SciPy's own output, so a speed number can never be reported over a
//! numerical difference.
//!
//! PROFILING THIS BINARY REQUIRES `perf stat --no-inherit`. It spawns a live SciPy child,
//! and `perf stat` follows children by default, so a plain `perf stat` on this harness
//! counts BOTH arms and attributes the total to us. The error is large and it does not look
//! like an error — measured here on `gammaln`, the same run reads:
//!
//! ```text
//! perf stat              10,343,020,541 instructions   IPC 2.67
//! perf stat --no-inherit  6,043,533,900 instructions   IPC 2.97
//! ```
//!
//! 1.71x too many instructions, and an IPC that is wrong in the direction that makes our
//! arm look WORSE at scheduling than it is. The giveaway is `fp_ret_sse_avx_ops.mac_flops`:
//! this crate is built with `fp-contract=off` and emits no FMA at all, so any non-zero
//! MAC-FLOP count on this binary is the SciPy child being counted. The same applies to
//! every `perf_*_vs_scipy` harness in this workspace — they all spawn a child.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_runtime::RuntimeMode;
use fsci_special::{SpecialTensor, digamma, erf, gammaln, zeta};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.special import digamma, erf, gammaln, zeta

op = os.environ['FSCI_SPECIAL_OP']
n = int(os.environ['FSCI_SPECIAL_N'])

raw = sys.stdin.buffer.read(n * 8)
if len(raw) != n * 8: raise RuntimeError('short fixture')
x = np.frombuffer(raw, dtype='<f8').copy()

fn = {'gammaln': gammaln, 'digamma': digamma, 'erf': erf, 'zeta': zeta}[op]
def run(): return np.ascontiguousarray(fn(x), dtype='<f8')

ref = run()
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} '
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
        raw_ours = sys.stdin.buffer.read(ref.size * 8)
        if len(raw_ours) != ref.size * 8: raise RuntimeError('short Rust result')
        ours = np.frombuffer(raw_ours, dtype='<f8')
        finite = np.isfinite(ref) & np.isfinite(ours)
        diff = np.abs(ours[finite] - ref[finite])
        rel = diff / np.maximum(np.abs(ref[finite]), np.finfo(np.float64).tiny)
        mismatched = int(np.sum(np.isfinite(ref) != np.isfinite(ours)))
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e} '
              f'compared={int(finite.sum())} nonfinite_mismatch={mismatched}', flush=True)
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
    fn start(op: &str, x: &[f64]) -> Self {
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_SPECIAL_OP", op)
            .env("FSCI_SPECIAL_N", x.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.special child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity(x.len() * 8);
        for value in x {
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

fn real_vec(tensor: SpecialTensor, op: &str) -> Vec<f64> {
    match tensor {
        SpecialTensor::RealVec(values) => values,
        other => panic!("{op} returned a non-real-vector tensor: {other:?}"),
    }
}

fn main() {
    let env_usize = |key: &str, fallback: usize| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback)
    };
    let n = env_usize("FSCI_SPECIAL_N", 200_000);
    let rounds = env_usize("FSCI_SPECIAL_ROUNDS", 5);
    let selected =
        std::env::var("FSCI_SPECIAL_OPS").unwrap_or_else(|_| "gammaln,digamma,erf,zeta".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_SPECIAL_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    // `FSCI_SPECIAL_GAMMALN_CROSSOVER=100` restores the pre-2026-08-27 Lanczos/asymptotic
    // crossover, so the change can be A/B'd inside ONE binary rather than against a
    // differently-built one.
    if let Some(x) = std::env::var("FSCI_SPECIAL_GAMMALN_CROSSOVER")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        fsci_special::GAMMALN_ASYMPTOTIC_MIN_OVERRIDE
            .store(x.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    // `FSCI_SPECIAL_ZETA_EARLY=0` restores the unconditional eight-term direct prefix, so
    // the early exit can be A/B'd inside ONE binary. Without this the two "arms" are the
    // same arm and the comparison is inert — which is exactly how it first read.
    match std::env::var("FSCI_SPECIAL_ZETA_EARLY").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_special::ZETA_DIRECT_EARLY_EXIT.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_special::ZETA_DIRECT_EARLY_EXIT.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }

    // `FSCI_SPECIAL_ZETA_BITS=1` runs the zeta early-exit bit-identity check and exits.
    // This lives in a BIN rather than only in `#[cfg(test)]` because the test profile of
    // this crate does not currently build: the workspace `Cargo.toml` asks for
    // `rand = "0.10.1"` while `Cargo.lock` still pins 0.8.7/0.9.5, so cargo re-resolves to
    // rand 0.10.2 and fails on a `getrandom` it cannot find offline. The `#[test]` version
    // is kept and is the durable check; this is what can actually be run today.
    if matches!(
        std::env::var("FSCI_SPECIAL_ZETA_BITS").ok().as_deref(),
        Some("1")
    ) {
        let mut points: Vec<f64> = (0..=4000)
            .map(|i| 1.0 + 1.0e-6 + 59.0 * (i as f64 / 4000.0))
            .collect();
        points.extend([
            1.000_001, 1.05, 1.5, 2.0, 7.0, 12.0, 15.0, 16.0, 20.0, 30.0, 60.0, 200.0,
        ]);
        let eval = |on: bool| -> Vec<f64> {
            fsci_special::ZETA_DIRECT_EARLY_EXIT.store(on, std::sync::atomic::Ordering::Relaxed);
            let t = SpecialTensor::RealVec(points.clone());
            match zeta(&t, RuntimeMode::Hardened).expect("fsci zeta") {
                SpecialTensor::RealVec(v) => v,
                other => panic!("zeta returned {other:?}"),
            }
        };
        let full = eval(false);
        let exited = eval(true);
        fsci_special::ZETA_DIRECT_EARLY_EXIT.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut differing = 0usize;
        for (i, (a, b)) in full.iter().zip(exited.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                if differing < 5 {
                    println!("  s={} full={a:e} exited={b:e}", points[i]);
                }
                differing += 1;
            }
        }
        println!(
            "zeta_bits: points={} differing={differing} VERDICT={}",
            points.len(),
            if differing == 0 {
                "BIT-IDENTICAL"
            } else {
                "DIVERGES"
            }
        );
        std::process::exit(i32::from(differing != 0));
    }

    println!("elf_sha256={}", elf_sha256());
    println!("n={n} gammaln_crossover={}", {
        let bits = fsci_special::GAMMALN_ASYMPTOTIC_MIN_OVERRIDE
            .load(std::sync::atomic::Ordering::Relaxed);
        if bits == 0 {
            fsci_special::GAMMALN_ASYMPTOTIC_MIN_X_DEFAULT
        } else {
            f64::from_bits(bits)
        }
    });

    let unit = |i: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(40_503) % 1_000_003;
        k as f64 / 1_000_003.0
    };

    for op in ["gammaln", "digamma", "erf", "zeta"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        // Per-op domains, chosen so the input SPANS the regimes each implementation
        // switches between rather than sitting inside one of them. A fixture confined to a
        // single branch benchmarks that branch and says nothing about the function.
        //   gammaln/digamma: 0.01 .. 60      (reflection, series and asymptotic all live)
        //   erf:            -6   .. 6        (series core, continued fraction, saturated tail)
        //   zeta:            1.5 .. 30       (above the pole, into where the sum truncates fast)
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let u = unit(i);
                match op {
                    "gammaln" | "digamma" => 0.01 + u * 59.99,
                    "erf" => -6.0 + u * 12.0,
                    _ => 1.5 + u * 28.5,
                }
            })
            .collect();

        let mut scipy = Scipy::start(op, &x);
        println!("{}", scipy.ready);

        // Built ONCE, outside the timed region. SciPy's arm is `fn(x)` on an array it
        // already holds — it does not copy its input — so cloning 200k f64 into a fresh
        // tensor on every call would charge us 1.6 MB of memcpy per iteration that the
        // incumbent never pays, and would be measuring the wrapper rather than the kernel.
        let tensor = SpecialTensor::RealVec(x.clone());

        let ours = || -> Vec<f64> {
            let out = match op {
                "gammaln" => gammaln(&tensor, RuntimeMode::Hardened),
                "digamma" => digamma(&tensor, RuntimeMode::Hardened),
                "erf" => erf(&tensor, RuntimeMode::Hardened),
                _ => zeta(&tensor, RuntimeMode::Hardened),
            };
            real_vec(out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")), op)
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
