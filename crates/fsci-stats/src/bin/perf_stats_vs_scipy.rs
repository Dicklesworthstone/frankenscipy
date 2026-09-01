//! Persistent live-SciPy whole-job benchmark for `fsci-stats`.
//!
//! WHY THIS FILE EXISTS. `fsci-stats` carries 212 perf binaries — the largest such surface
//! in the workspace — and not one of them puts a live SciPy arm next to ours, so every
//! number the crate reports is a self-timing. This is the fifth live-SciPy harness, after
//! signal, spatial, cluster and interpolate. `fsci-special` (120 bins), `fsci-ndimage` (64)
//! and `fsci-opt` (25) are still uncovered.
//!
//! The four cases are chosen because SciPy does them in COMPILED code, so a loss is the
//! likely and therefore informative outcome: `rankdata` is a numpy argsort, `spearmanr`
//! ranks then correlates, `kendalltau` is a Cython merge-sort pair count, and `ks_2samp`
//! is numpy `searchsorted` over two sorted samples.
//!
//! AGREEMENT IS CHECKED by shipping our result back to the Python child and comparing
//! elementwise against SciPy's own output. For the scalar tests that is `[statistic,
//! pvalue]`; for `rankdata` it is the whole rank vector, which is what actually catches a
//! tie-handling divergence — a statistic alone can agree while the ranks behind it do not.
//!
//! The fixture deliberately contains TIES (values quantized onto a coarse grid). Ranking
//! and pair-counting both have a separate, slower path for ties, and a tie-free fixture
//! would benchmark the easy branch of all four ops while claiming to cover them.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use fsci_stats::{kendalltau, ks_2samp, rankdata, spearmanr};

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.stats"];

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
from scipy.stats import kendalltau, ks_2samp, rankdata, spearmanr

op = os.environ['FSCI_STATS_OP']
n = int(os.environ['FSCI_STATS_N'])

raw = sys.stdin.buffer.read(n * 8)
if len(raw) != n * 8: raise RuntimeError('short fixture x')
x = np.frombuffer(raw, dtype='<f8').copy()
raw_y = sys.stdin.buffer.read(n * 8)
if len(raw_y) != n * 8: raise RuntimeError('short fixture y')
y = np.frombuffer(raw_y, dtype='<f8').copy()

if op == 'rankdata':
    def run(): return np.ascontiguousarray(rankdata(x, method='average'), dtype='<f8')
elif op == 'spearmanr':
    def run():
        r = spearmanr(x, y)
        return np.array([r.statistic, r.pvalue], dtype='<f8')
elif op == 'kendalltau':
    def run():
        r = kendalltau(x, y)
        return np.array([r.statistic, r.pvalue], dtype='<f8')
else:
    def run():
        r = ks_2samp(x, y)
        return np.array([r.statistic, r.pvalue], dtype='<f8')

ref = np.ascontiguousarray(run(), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} '
      f'fixture_sha256={hashlib.sha256(raw + raw_y).hexdigest()} '
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
    fn start(op: &str, x: &[f64], y: &[f64]) -> Self {
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_STATS_OP", op)
            .env("FSCI_STATS_N", x.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.stats child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity((x.len() + y.len()) * 8);
        for value in x.iter().chain(y.iter()) {
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
    let n = env_usize("FSCI_STATS_N", 200_000);
    let rounds = env_usize("FSCI_STATS_ROUNDS", 5);
    // Ties per distinct value, on average. 1 means an (almost) tie-free fixture; larger
    // values push both implementations onto their tie-handling paths.
    let tie_factor = env_usize("FSCI_STATS_TIE_FACTOR", 8).max(1);
    let selected = std::env::var("FSCI_STATS_OPS")
        .unwrap_or_else(|_| "rankdata,spearmanr,kendalltau,ks_2samp".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_STATS_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    // `FSCI_STATS_KS_BANDED=0` restores the full-rectangle exact KS sweep.
    // Only-override-when-asked, so an unset variable cannot overwrite a flipped default.
    match std::env::var("FSCI_STATS_KS_BANDED").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_stats::KS_2SAMP_BANDED_EXACT.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_stats::KS_2SAMP_BANDED_EXACT.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }

    // `FSCI_STATS_KS_SERIES=0` restores the path-counting sweep for equal sample sizes.
    match std::env::var("FSCI_STATS_KS_SERIES").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_stats::KS_2SAMP_SQUARE_SERIES.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_stats::KS_2SAMP_SQUARE_SERIES.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }

    println!("elf_sha256={}", elf_sha256());
    println!(
        "n={n} tie_factor={tie_factor} ks_banded={} ks_series={}",
        fsci_stats::KS_2SAMP_BANDED_EXACT.load(std::sync::atomic::Ordering::Relaxed),
        fsci_stats::KS_2SAMP_SQUARE_SERIES.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Deterministic, correlated but not identical, and QUANTIZED so ties are common.
    let raw = |i: usize, salt: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(salt * 40_503) % 1_000_003;
        k as f64 / 1_000_003.0
    };
    let levels = (n / tie_factor).max(2) as f64;
    let quantize = |v: f64| -> f64 { (v * levels).floor() / levels };
    let x: Vec<f64> = (0..n).map(|i| quantize(raw(i, 7))).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| quantize(0.6 * raw(i, 7) + 0.4 * raw(i, 29)))
        .collect();

    for op in ["rankdata", "spearmanr", "kendalltau", "ks_2samp"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        let mut scipy = Scipy::start(op, &x, &y);
        println!("{}", scipy.ready);

        let ours = || -> Vec<f64> {
            match op {
                "rankdata" => rankdata(&x, Some("average")).expect("fsci rankdata"),
                "spearmanr" => {
                    let r = spearmanr(&x, &y);
                    vec![r.statistic, r.pvalue]
                }
                "kendalltau" => {
                    let r = kendalltau(&x, &y);
                    vec![r.statistic, r.pvalue]
                }
                _ => {
                    let r = ks_2samp(&x, &y);
                    vec![r.statistic, r.pvalue]
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
            "case=n{n}t{tie_factor} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
