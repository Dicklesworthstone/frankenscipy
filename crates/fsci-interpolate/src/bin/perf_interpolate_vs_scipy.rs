//! Persistent live-SciPy whole-job benchmark for `fsci-interpolate`.
//!
//! WHY THIS FILE EXISTS. `fsci-interpolate` carries 41 perf binaries and not one of them
//! puts a live SciPy arm next to ours, so every number the crate reports is a self-timing.
//! This is the fourth and last harness written to close that gap, after
//! `perf_signal_vs_scipy`, `perf_spatial_vs_scipy` and `perf_cluster_vs_scipy`; with this
//! one, every crate in the workspace has a live incumbent.
//!
//! All three cases are ones SciPy COMPILES, so a loss is the likely outcome and therefore
//! the informative one: `splev` is FITPACK Fortran, `RegularGridInterpolator` and
//! `CubicSpline.__call__` (via `PPoly`) are Cython. Gates on nothing a running measurement
//! cannot supply for itself — twelve harnesses in this tree abort on a booking claim that
//! is unsatisfiable (bead fr78g).
//!
//! AGREEMENT IS CHECKED, not assumed: our result is shipped back to the Python child and
//! compared against SciPy's own output elementwise. Two digests computed by different
//! algorithms would report a clean speed number over a numerical difference.
//!
//! `splev` needs SciPy to hand US something first. Comparing two splines fitted
//! independently would measure `splrep`'s knot placement as much as evaluation, so the
//! child runs `splrep` once and ships the resulting knots and coefficients BACK over a
//! binary reverse channel. Both arms then evaluate THE SAME spline and the comparison is
//! about evaluation alone.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_interpolate::{
    CubicSplineStandalone, RegularGridInterpolator, RegularGridMethod, SplineBc, splev,
};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.interpolate import CubicSpline, RegularGridInterpolator, splev, splrep

op = os.environ['FSCI_INTERP_OP']
n = int(os.environ['FSCI_INTERP_N'])
m = int(os.environ['FSCI_INTERP_M'])
g = int(os.environ['FSCI_INTERP_G'])

def take(count):
    raw = sys.stdin.buffer.read(count * 8)
    if len(raw) != count * 8: raise RuntimeError('short fixture')
    return np.frombuffer(raw, dtype='<f8').copy()

nt = nc = 0
if op in ('splev', 'cubic'):
    x, y, q = take(n), take(n), take(m)
    fixture = x.tobytes() + y.tobytes() + q.tobytes()
    if op == 'splev':
        tck = splrep(x, y, s=0, k=3)
        t, c, k = tck[0], tck[1], tck[2]
        nt, nc = t.size, c.size
        def run(): return splev(q, (t, c, k))
    else:
        cs = CubicSpline(x, y, bc_type='not-a-knot')
        def run(): return cs(q)
else:
    axes = [take(g) for _ in range(3)]
    values = take(g * g * g).reshape(g, g, g)
    queries = take(m * 3).reshape(m, 3)
    fixture = b''.join(a.tobytes() for a in axes) + values.tobytes() + queries.tobytes()
    rgi = RegularGridInterpolator(tuple(axes), values, method='linear', bounds_error=False)
    def run(): return rgi(queries)

ref = np.ascontiguousarray(run(), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} m={m} g={g} '
      f'fixture_sha256={hashlib.sha256(fixture).hexdigest()} '
      f'tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1"} out_len={ref.size} nt={nt} nc={nc}', flush=True)

# Reverse channel: hand the fitted spline back so both arms evaluate the SAME one.
if op == 'splev':
    sys.stdout.buffer.write(np.ascontiguousarray(t, dtype='<f8').tobytes())
    sys.stdout.buffer.write(np.ascontiguousarray(c, dtype='<f8').tobytes())
    sys.stdout.buffer.flush()

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
    /// Knots and coefficients SciPy fitted, empty for ops that do not use them.
    tck: (Vec<f64>, Vec<f64>),
}

impl Scipy {
    fn start(op: &str, n: usize, m: usize, g: usize, blobs: &[&[f64]]) -> Self {
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_INTERP_OP", op)
            .env("FSCI_INTERP_N", n.to_string())
            .env("FSCI_INTERP_M", m.to_string())
            .env("FSCI_INTERP_G", g.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.interpolate child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::new();
        for blob in blobs {
            for value in *blob {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        stdin.write_all(&bytes).expect("send fixture");
        stdin.flush().expect("flush fixture");

        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("read readiness");
        assert!(ready.starts_with("READY scipy="), "not ready: {ready:?}");

        let field = |key: &str| -> usize {
            ready
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix(key))
                .unwrap_or_else(|| panic!("missing {key} in {ready:?}"))
                .parse()
                .expect("numeric field")
        };
        let (nt, nc) = (field("nt="), field("nc="));
        let mut read_f64s = |count: usize| -> Vec<f64> {
            let mut raw = vec![0u8; count * 8];
            stdout.read_exact(&mut raw).expect("read reverse channel");
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect()
        };
        let tck = (read_f64s(nt), read_f64s(nc));

        Self {
            child,
            stdin,
            stdout,
            ready: ready.trim().to_owned(),
            tck,
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
/// process: `sha2` is only a DEV dependency here, and promoting it to a real one so a
/// benchmark can print a digest would be dependency smuggling.
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
    let n = env_usize("FSCI_INTERP_N", 2000);
    let m = env_usize("FSCI_INTERP_M", 100_000);
    let g = env_usize("FSCI_INTERP_G", 48);
    let rounds = env_usize("FSCI_INTERP_ROUNDS", 5);
    let selected =
        std::env::var("FSCI_INTERP_OPS").unwrap_or_else(|_| "splev,cubic,rgi".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_INTERP_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());

    println!("elf_sha256={}", elf_sha256());

    // Deterministic, strictly increasing, and NOT evenly spaced — an even grid would let
    // both sides take a uniform-stride fast path and would hide the interval search that
    // dominates irregular data.
    let jitter = |i: usize, salt: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(salt * 40_503) % 100_003;
        k as f64 / 100_003.0
    };
    let mut x = Vec::with_capacity(n);
    let mut cursor = 0.0_f64;
    for i in 0..n {
        cursor += 0.5 + jitter(i, 7);
        x.push(cursor);
    }
    let y: Vec<f64> = (0..n)
        .map(|i| (x[i] * 0.05).sin() + 0.3 * (x[i] * 0.011).cos())
        .collect();
    // Queries strictly inside the data range, so no arm spends its time in extrapolation.
    let (lo, hi) = (x[0], x[n - 1]);
    let mut q: Vec<f64> = (0..m)
        .map(|i| lo + (hi - lo) * (0.001 + 0.998 * jitter(i, 13)))
        .collect();
    // `FSCI_INTERP_SORT_QUERIES=1` presents the batch in ascending order. This is NOT a
    // cosmetic knob: FITPACK's `splev` walks to the next interval INCREMENTALLY from the
    // previous one, so sorted input costs it O(1) per point and shuffled input degrades it
    // toward a scan over the knot vector. Reporting only one ordering would be picking the
    // fixture that flatters one implementation, so both are measured and both are reported.
    let sort_queries = matches!(
        std::env::var("FSCI_INTERP_SORT_QUERIES").ok().as_deref(),
        Some("1") | Some("true")
    );
    if sort_queries {
        q.sort_by(f64::total_cmp);
    }
    println!("sorted_queries={sort_queries}");
    let q = q;

    // Irregular 3-D grid for RegularGridInterpolator.
    let axis = |salt: usize| -> Vec<f64> {
        let mut axis = Vec::with_capacity(g);
        let mut at = 0.0_f64;
        for i in 0..g {
            at += 0.5 + jitter(i, salt);
            axis.push(at);
        }
        axis
    };
    let axes = [axis(3), axis(5), axis(11)];
    let values: Vec<f64> = (0..g * g * g)
        .map(|idx| {
            let (a, b, c) = (idx / (g * g), (idx / g) % g, idx % g);
            (axes[0][a] * 0.07).sin() + (axes[1][b] * 0.05).cos() + axes[2][c] * 0.001
        })
        .collect();
    let axis_span: [(f64, f64); 3] = [
        (axes[0][0], axes[0][g - 1]),
        (axes[1][0], axes[1][g - 1]),
        (axes[2][0], axes[2][g - 1]),
    ];
    let queries_flat: Vec<f64> = (0..m)
        .flat_map(|i| {
            (0..3).map(move |d| {
                let (alo, ahi) = axis_span[d];
                alo + (ahi - alo) * (0.001 + 0.998 * jitter(i * 3 + d, 17 + d))
            })
        })
        .collect();
    let queries: Vec<Vec<f64>> = queries_flat.chunks_exact(3).map(<[f64]>::to_vec).collect();

    for op in ["splev", "cubic", "rgi"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        let blobs: Vec<&[f64]> = if op == "rgi" {
            vec![&axes[0], &axes[1], &axes[2], &values, &queries_flat]
        } else {
            vec![&x, &y, &q]
        };
        let mut scipy = Scipy::start(op, n, m, g, &blobs);
        println!("{}", scipy.ready);

        // Built ONCE, outside the timed region, for both `cubic` and `rgi`: SciPy's arm
        // also constructs its interpolator once and times only the call, so timing our
        // construction here would compare two different jobs.
        let spline = (op == "cubic")
            .then(|| CubicSplineStandalone::new(&x, &y, SplineBc::NotAKnot).expect("cubic spline"));
        let grid = (op == "rgi").then(|| {
            RegularGridInterpolator::new(
                axes.to_vec(),
                values.clone(),
                RegularGridMethod::Linear,
                false,
                None,
            )
            .expect("regular grid")
        });
        let tck = (op == "splev").then(|| (scipy.tck.0.clone(), scipy.tck.1.clone(), 3usize));

        let ours = || -> Vec<f64> {
            match op {
                "splev" => splev(&q, tck.as_ref().expect("tck")).expect("fsci splev"),
                // `eval_many`, NOT a loop over the scalar `eval`. SciPy's arm is `cs(q)`, a
                // single batch call that carries an interval hint from one query to the
                // next, so timing a scalar loop here would compare two different jobs and
                // charge us a per-query interval search the public batch API does not do.
                // Measured wrong first: the scalar loop read 0.788x at n=20000 where the
                // batch call is well over parity.
                "cubic" => spline.as_ref().expect("spline").eval_many(&q),
                _ => grid
                    .as_ref()
                    .expect("grid")
                    .eval_many(&queries)
                    .expect("fsci rgi"),
            }
        };

        black_box(ours());
        let _ = scipy.time(1, 1);

        // Calibrate off the FASTEST of several calls, not one: a single call carries cold
        // caches and any scheduler hiccup that lands on it, which over-estimates cost and
        // under-sizes `reps`, leaving samples too short and the A/A nulls loose. Both arms
        // share the repetition count, so this sharpens the nulls without moving the ratio.
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
            "case=n{n}m{m}g{g} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
