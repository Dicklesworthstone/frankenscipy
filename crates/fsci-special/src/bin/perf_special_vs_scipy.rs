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

/// Emit a result line on BOTH stdout and stderr.
///
/// `rch` relays the remote job's stderr, and a run whose numbers went only to stdout can come
/// back with the numbers missing — the measurement then has to be repeated for no reason
/// other than where it was written. Duplicating only the RESULT lines keeps that cheap:
/// per-round chatter stays on stdout alone.
macro_rules! emit {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        eprintln!("{line}");
    }};
}

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialTensor, beta, betaln, dawsn, digamma, erf, erfc, erfcinv, erfinv, expit, exprel, gamma,
    gammainc, gammaincc, gammaln, hyp0f1, i0, i1, iv, ive, j0, j1, jn, jv, jve, k0, k1, kn, kv,
    kve, rgamma, spence, y0, y1, yn, yv, yve, zeta,
};

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy import special as sp

op = os.environ['FSCI_SPECIAL_OP']
n = int(os.environ['FSCI_SPECIAL_N'])
nargs = int(os.environ.get('FSCI_SPECIAL_NARGS', '1'))

raw = sys.stdin.buffer.read(n * 8 * nargs)
if len(raw) != n * 8 * nargs: raise RuntimeError('short fixture')
flat = np.frombuffer(raw, dtype='<f8').copy()
args = [flat[i * n:(i + 1) * n] for i in range(nargs)]

fn = getattr(sp, op)   # resolved by name: adding a case needs no change here
def run(): return np.ascontiguousarray(fn(*args), dtype='<f8')

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
        Self::start_n(op, &[x])
    }

    /// Start the child for an op of any arity. Arguments are sent as one flat little-endian
    /// stream, argument-major, and the child slices them back apart — so adding a
    /// two-argument case needs no protocol change beyond `FSCI_SPECIAL_NARGS`.
    fn start_n(op: &str, args: &[&[f64]]) -> Self {
        let n = args[0].len();
        assert!(
            args.iter().all(|a| a.len() == n),
            "all argument arrays must have the same length"
        );
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON])
            .env("FSCI_SPECIAL_OP", op)
            .env("FSCI_SPECIAL_N", n.to_string())
            .env("FSCI_SPECIAL_NARGS", args.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.special child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::with_capacity(n * 8 * args.len());
        for arg in args {
            for value in *arg {
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

/// Every measured case: the `scipy.special` ufunc name (ours must match it exactly, since
/// the Python child resolves it with `getattr`) and the input domain.
///
/// DOMAINS SPAN REGIMES ON PURPOSE. A fixture sitting inside one branch benchmarks that
/// branch and says nothing about the function — the mistake that made the first `gammaln`
/// reading unrepresentative. Each range below crosses the switch points of the
/// implementation it exercises: series against asymptotic, reflection against direct,
/// small-argument rational against continued fraction.
///
/// The list started at four ops. That was the problem: `zeta` turned out to be nearly twice
/// SciPy's instruction count while three cheaper functions sat next to it looking fine, and
/// this crate exports 578 public functions. Unmeasured surface is where the large cells have
/// been every time.
const CASES: &[(&str, f64, f64)] = &[
    // Gamma family: reflection, Lanczos and asymptotic kernels all live in these ranges.
    ("gammaln", 0.01, 60.0),
    ("digamma", 0.01, 60.0),
    ("gamma", 0.01, 30.0),
    ("rgamma", -10.0, 10.0),
    ("zeta", 1.5, 30.0),
    // Error function family: series core, continued fraction, and the saturated tail.
    ("erf", -6.0, 6.0),
    ("erfc", -6.0, 6.0),
    ("erfinv", -0.999, 0.999),
    ("erfcinv", 0.001, 1.999),
    ("dawsn", -10.0, 10.0),
    // Bessel: oscillatory small-argument series through to the asymptotic expansion.
    ("j0", -30.0, 30.0),
    ("j1", -30.0, 30.0),
    ("y0", 0.01, 30.0),
    ("y1", 0.01, 30.0),
    ("i0", -15.0, 15.0),
    ("i1", -15.0, 15.0),
    ("k0", 0.01, 20.0),
    ("k1", 0.01, 20.0),
    // Miscellaneous single-argument ufuncs.
    ("spence", 0.0, 10.0),
    ("expit", -20.0, 20.0),
    ("exprel", -10.0, 10.0),
];

/// Two-argument cases: the `scipy.special` name and a domain for each argument.
///
/// The one-argument survey was widened from 4 ops to 21 in 7f0e93d26 and immediately found
/// `i0`/`i1` at 12x, the worst cells in the crate. TWO-argument ufuncs were never measured
/// at all — 70 of SciPy's 232 are binary, 20 of them have a matching two-tensor entry point
/// here — so this is the same unmeasured surface one arity up.
///
/// `hankel1`/`hankel2` and their scaled forms are excluded: they return complex, and this
/// harness compares real arrays. Orders for `jn`/`kn`/`yn` are integer-valued because that
/// is what those entry points mean by their first argument.
const CASES2: &[(&str, f64, f64, f64, f64)] = &[
    ("beta", 0.5, 8.0, 0.5, 8.0),
    ("betaln", 0.5, 40.0, 0.5, 40.0),
    ("gammainc", 0.5, 12.0, 0.1, 25.0),
    ("gammaincc", 0.5, 12.0, 0.1, 25.0),
    ("hyp0f1", 0.5, 10.0, -10.0, 10.0),
    ("jv", 0.0, 10.0, 0.5, 30.0),
    ("yv", 0.0, 10.0, 0.5, 30.0),
    ("iv", 0.0, 5.0, 0.1, 12.0),
    ("kv", 0.0, 5.0, 0.1, 12.0),
    ("jve", 0.0, 10.0, 0.5, 30.0),
    ("yve", 0.0, 10.0, 0.5, 30.0),
    ("ive", 0.0, 5.0, 0.1, 12.0),
    ("kve", 0.0, 5.0, 0.1, 12.0),
];

/// Integer-order siblings, kept apart because their first argument is quantised.
const CASES2_INTEGER_ORDER: &[(&str, f64, f64, f64, f64)] = &[
    ("jn", 0.0, 10.0, 0.5, 20.0),
    ("yn", 0.0, 10.0, 0.5, 20.0),
    ("kn", 0.0, 10.0, 0.1, 12.0),
];

/// Dispatch to our two-argument entry point for `op`.
fn call_ours2(op: &str, a: &SpecialTensor, b: &SpecialTensor) -> fsci_special::SpecialResult {
    let mode = RuntimeMode::Hardened;
    match op {
        "beta" => beta(a, b, mode),
        "betaln" => betaln(a, b, mode),
        "gammainc" => gammainc(a, b, mode),
        "gammaincc" => gammaincc(a, b, mode),
        "hyp0f1" => hyp0f1(a, b, mode),
        "jv" => jv(a, b, mode),
        "yv" => yv(a, b, mode),
        "iv" => iv(a, b, mode),
        "kv" => kv(a, b, mode),
        "jve" => jve(a, b, mode),
        "yve" => yve(a, b, mode),
        "ive" => ive(a, b, mode),
        "kve" => kve(a, b, mode),
        "jn" => jn(a, b, mode),
        "yn" => yn(a, b, mode),
        "kn" => kn(a, b, mode),
        other => panic!("no fsci two-argument entry point wired for {other}"),
    }
}

/// Dispatch to our entry point for `op`. Kept next to [`CASES`] so a new case is two lines
/// in one file rather than an edit in four places.
fn call_ours(op: &str, tensor: &SpecialTensor) -> fsci_special::SpecialResult {
    let mode = RuntimeMode::Hardened;
    match op {
        "gammaln" => gammaln(tensor, mode),
        "digamma" => digamma(tensor, mode),
        "gamma" => gamma(tensor, mode),
        "rgamma" => rgamma(tensor, mode),
        "zeta" => zeta(tensor, mode),
        "erf" => erf(tensor, mode),
        "erfc" => erfc(tensor, mode),
        "erfinv" => erfinv(tensor, mode),
        "erfcinv" => erfcinv(tensor, mode),
        "dawsn" => dawsn(tensor, mode),
        "j0" => j0(tensor, mode),
        "j1" => j1(tensor, mode),
        "y0" => y0(tensor, mode),
        "y1" => y1(tensor, mode),
        "i0" => i0(tensor, mode),
        "i1" => i1(tensor, mode),
        "k0" => k0(tensor, mode),
        "k1" => k1(tensor, mode),
        "spence" => spence(tensor, mode),
        "expit" => expit(tensor, mode),
        "exprel" => exprel(tensor, mode),
        other => panic!("no fsci entry point wired for {other}"),
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

/// Host provenance, printed by the binary that is actually being timed.
///
/// WHY IT IS SELF-REPORTED. Builds are remote, so the machine that runs the benchmark is
/// not the machine the agent is typing on, and a row that names the local box names the
/// wrong host. Every field here is read on the worker, in the timed process, by the same
/// executable whose SHA-256 is printed next to it — so "which host, which binary" is one
/// answer rather than two claims that have to be trusted to match.
///
/// A ledger row is required to carry this, and until now it was assembled by hand from the
/// wrong machine. Fields that cannot be read return `unknown` rather than a plausible
/// default: a fabricated governor is worse than an absent one.
fn host_provenance() -> String {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let first_line =
        |path: &str| read(path).map(|s| s.lines().next().unwrap_or("").trim().to_owned());

    let hostname = first_line("/proc/sys/kernel/hostname").unwrap_or_else(|| "unknown".into());
    let logical = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "unknown".into());

    let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
    let model = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().replace(' ', "_"))
        .unwrap_or_else(|| "unknown".into());
    // Physical cores = distinct (physical id, core id) pairs. Counting `processor` lines
    // instead would report SMT siblings as cores and silently double the number.
    let mut cores: Vec<(&str, &str)> = Vec::new();
    let (mut pkg, mut core) = ("", "");
    for line in cpuinfo.lines() {
        if let Some((k, v)) = line.split_once(':') {
            match k.trim() {
                "physical id" => pkg = v.trim(),
                "core id" => {
                    core = v.trim();
                    if !cores.contains(&(pkg, core)) {
                        cores.push((pkg, core));
                    }
                }
                _ => {}
            }
        }
    }
    let physical = if cores.is_empty() {
        "unknown".to_owned()
    } else {
        cores.len().to_string()
    };

    let ram_kb = read("/proc/meminfo")
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1).map(str::to_owned))
        })
        .unwrap_or_else(|| "unknown".into());
    let numa = std::fs::read_dir("/sys/devices/system/node")
        .map(|d| {
            d.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                .count()
                .to_string()
        })
        .unwrap_or_else(|_| "unknown".into());
    let governor = first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|| "unknown".into());
    let loadavg = first_line("/proc/loadavg").unwrap_or_else(|| "unknown".into());

    format!(
        "host={hostname} cpu_model={model} physical_cores={physical} logical_threads={logical} \
         ram_kb={ram_kb} numa_nodes={numa} governor={governor} loadavg=[{loadavg}]"
    )
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
    let selected = std::env::var("FSCI_SPECIAL_OPS").unwrap_or_else(|_| {
        CASES
            .iter()
            .map(|&(op, _, _)| op)
            .collect::<Vec<_>>()
            .join(",")
    });
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

    // `FSCI_SPECIAL_GAMMALN_HOIST=0` restores the per-element read of the Lanczos/asymptotic
    // crossover, so the batch-level hoist can be A/B'd inside ONE binary.
    let hoist = !matches!(
        std::env::var("FSCI_SPECIAL_GAMMALN_HOIST").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMALN_HOIST_THRESHOLD.store(hoist, std::sync::atomic::Ordering::Relaxed);
    println!("gammaln_hoist_threshold={hoist}");

    // `FSCI_SPECIAL_PREALLOC=0` restores `collect::<Result<Vec<_>, _>>()` in the shared
    // gamma-family array mapper, so the preallocated fill can be A/B'd inside ONE binary.
    let prealloc = !matches!(
        std::env::var("FSCI_SPECIAL_PREALLOC").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMA_FAMILY_PREALLOC_FILL.store(prealloc, std::sync::atomic::Ordering::Relaxed);
    println!("gamma_family_prealloc_fill={prealloc}");

    // `FSCI_SPECIAL_GAMMALN_LGAM=0` restores the Lanczos band, so Cephes' one-log `lgam`
    // can be A/B'd inside ONE invocation on the worker.
    let gammaln_lgam = !matches!(
        std::env::var("FSCI_SPECIAL_GAMMALN_LGAM").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMALN_CEPHES_LGAM.store(gammaln_lgam, std::sync::atomic::Ordering::Relaxed);
    println!("gammaln_cephes_lgam={gammaln_lgam}");

    // `FSCI_SPECIAL_IV_TERMRATIO=0` restores the log-carried `I_v` series, so the term-ratio
    // recurrence can be A/B'd -- for ACCURACY as much as cost -- inside ONE binary.
    let iv_ratio = !matches!(
        std::env::var("FSCI_SPECIAL_IV_TERMRATIO").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::IV_SERIES_TERM_RATIO.store(iv_ratio, std::sync::atomic::Ordering::Relaxed);
    println!("iv_series_term_ratio={iv_ratio}");

    // `FSCI_SPECIAL_BETA_DIRECT=0` restores `exp(betaln(a,b))` for `beta`, so the
    // direct-Gamma product can be A/B'd inside ONE binary.
    let beta_direct = !matches!(
        std::env::var("FSCI_SPECIAL_BETA_DIRECT").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::BETA_CEPHES_DIRECT.store(beta_direct, std::sync::atomic::Ordering::Relaxed);
    println!("beta_cephes_direct={beta_direct}");

    // `FSCI_SPECIAL_GAMMA_REFLECTFREE=0` restores the reflection formula (and its `sin`) for
    // negative arguments, so the recurrence route can be A/B'd -- for ACCURACY as much as
    // cost -- inside ONE binary.
    let reflect_free = !matches!(
        std::env::var("FSCI_SPECIAL_GAMMA_REFLECTFREE")
            .ok()
            .as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMA_CEPHES_REFLECTION_FREE
        .store(reflect_free, std::sync::atomic::Ordering::Relaxed);
    println!("gamma_cephes_reflection_free={reflect_free}");

    // `FSCI_SPECIAL_GAMMA_INFALLIBLE=0` restores the `Result`-per-element batch path for
    // `gamma`, so the infallible batch can be A/B'd inside ONE binary.
    let gamma_infallible = !matches!(
        std::env::var("FSCI_SPECIAL_GAMMA_INFALLIBLE")
            .ok()
            .as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMA_INFALLIBLE_BATCH
        .store(gamma_infallible, std::sync::atomic::Ordering::Relaxed);
    println!("gamma_infallible_batch={gamma_infallible}");

    // `FSCI_SPECIAL_RGAMMA_CHEB=0` restores `1.0 / gamma_scalar(x)` for |x| <= 4, so the
    // Chebyshev series can be A/B'd -- for ACCURACY as much as cost -- in ONE binary.
    let rgamma_cheb = !matches!(
        std::env::var("FSCI_SPECIAL_RGAMMA_CHEB").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::RGAMMA_CEPHES_CHEB.store(rgamma_cheb, std::sync::atomic::Ordering::Relaxed);
    println!("rgamma_cephes_cheb={rgamma_cheb}");

    // `FSCI_SPECIAL_GAMMA_RATIONAL=0` restores the Lanczos kernel for `Γ(x)` on
    // 0.5 <= x <= 33, so the Cephes rational can be A/B'd -- for ACCURACY as much as cost --
    // inside ONE binary against the same live SciPy arm. `rgamma` inherits it.
    let gamma_rational = !matches!(
        std::env::var("FSCI_SPECIAL_GAMMA_RATIONAL").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::GAMMA_CEPHES_RATIONAL.store(gamma_rational, std::sync::atomic::Ordering::Relaxed);
    println!("gamma_cephes_rational={gamma_rational}");

    // `FSCI_SPECIAL_K01_CEPHES=0` restores the general order-v `kv_scalar` path for
    // `k0`/`k1`, so the Chebyshev kernels can be A/B'd -- for ACCURACY as much as cost --
    // inside ONE binary against the same live SciPy arm.
    let k01_cephes = !matches!(
        std::env::var("FSCI_SPECIAL_K01_CEPHES").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::BESSEL_K01_CEPHES.store(k01_cephes, std::sync::atomic::Ordering::Relaxed);
    println!("bessel_k01_cephes={k01_cephes}");

    // `FSCI_SPECIAL_Y01_LARGE=0` restores the harmonic series and general order-v
    // asymptotic for `y0`/`y1` above x = 5, so the Cephes large-argument form can be A/B'd
    // -- for ACCURACY as much as cost -- inside ONE binary against the same live SciPy arm.
    let y01_large = !matches!(
        std::env::var("FSCI_SPECIAL_Y01_LARGE").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::BESSEL_Y01_CEPHES_LARGE.store(y01_large, std::sync::atomic::Ordering::Relaxed);
    println!("bessel_y01_cephes_large={y01_large}");

    // `FSCI_SPECIAL_I01_CEPHES=0` restores the general order-v `iv_scalar` path for
    // `i0`/`i1`, so the Chebyshev kernels can be A/B'd -- for ACCURACY as much as cost --
    // inside ONE binary against the same live SciPy arm.
    let i01_cephes = !matches!(
        std::env::var("FSCI_SPECIAL_I01_CEPHES").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::BESSEL_I01_CEPHES.store(i01_cephes, std::sync::atomic::Ordering::Relaxed);
    println!("bessel_i01_cephes={i01_cephes}");

    // `FSCI_SPECIAL_ZETA_RATIONAL=0` restores our eight-`exp` Euler-Maclaurin prefix in
    // place of SciPy's rational approximations, so the two can be A/B'd -- for ACCURACY as
    // much as cost -- inside ONE binary against the same live SciPy arm.
    let zeta_rational = !matches!(
        std::env::var("FSCI_SPECIAL_ZETA_RATIONAL").ok().as_deref(),
        Some("0") | Some("false")
    );
    fsci_special::ZETA_CEPHES_RATIONAL.store(zeta_rational, std::sync::atomic::Ordering::Relaxed);
    println!("zeta_cephes_rational={zeta_rational}");

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

    emit!("elf_sha256={}", elf_sha256());
    emit!("provenance_before {}", host_provenance());
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

    // ── two-argument sweep ───────────────────────────────────────────────────────────
    //
    // Same protocol, same agreement check, same probe mode; only the arity differs. Run
    // with `FSCI_SPECIAL_OPS=<name>` like the one-argument cases.
    for (cases, integer_order) in [(CASES2, false), (CASES2_INTEGER_ORDER, true)] {
        for &(op, alo, ahi, blo, bhi) in cases {
            if !selected.split(',').any(|name| name.trim() == op) {
                continue;
            }
            let a: Vec<f64> = (0..n)
                .map(|i| {
                    let v = alo + unit(i) * (ahi - alo);
                    if integer_order { v.floor() } else { v }
                })
                .collect();
            // A second, decorrelated stream for the other argument — reusing `unit(i)` for
            // both would put every sample on the diagonal and exercise one line of a
            // two-dimensional domain.
            let b: Vec<f64> = (0..n)
                .map(|i| blo + unit(i * 7 + 13) * (bhi - blo))
                .collect();
            println!("n={n} op={op} domain_a=[{alo}, {ahi}] domain_b=[{blo}, {bhi}]");

            let mut scipy = Scipy::start_n(op, &[&a, &b]);
            println!("{}", scipy.ready);

            let ta = SpecialTensor::RealVec(a.clone());
            let tb = SpecialTensor::RealVec(b.clone());
            let ours = || -> Vec<f64> {
                let out = call_ours2(op, &ta, &tb);
                real_vec(out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")), op)
            };
            black_box(ours());

            if let Ok(k) = std::env::var("FSCI_SPECIAL_PROBE") {
                let k: usize = k.parse().expect("FSCI_SPECIAL_PROBE must be an integer");
                let started = Instant::now();
                for _ in 0..k {
                    black_box(ours());
                }
                let ms = started.elapsed().as_secs_f64() * 1.0e3;
                println!(
                    "PROBE op={op} calls={k} n={n} elements={} ms={ms:.3}",
                    k * n
                );
                continue;
            }

            let _ = scipy.time(1, 1);
            const MIN_SAMPLE_MS2: f64 = 20.0;
            let mut single = f64::INFINITY;
            for _ in 0..3 {
                let started = Instant::now();
                black_box(ours());
                single = single.min(started.elapsed().as_secs_f64() * 1.0e3);
            }
            let reps = fixed_reps
                .unwrap_or_else(|| (MIN_SAMPLE_MS2 / single.max(1.0e-6)).ceil() as usize)
                .clamp(1, 4096);
            println!("op={op} calibration single={single:.4}ms reps={reps}");

            let time_ours = || -> f64 {
                let started = Instant::now();
                for _ in 0..reps {
                    black_box(ours());
                }
                started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
            };
            let f1 = time_ours();
            let s1 = scipy.time(reps, 1);
            let s2 = scipy.time(reps, 1);
            let f2 = time_ours();
            let fsci = f1.min(f2);
            let sci = s1.min(s2);
            let check = scipy.check(&ours());
            emit!(
                "case=n{n} op={op} fsci={fsci:.3}ms scipy={sci:.3}ms scipy/fsci={:.3}x \
                 null_fsci={:.3} null_scipy={:.3} {check}",
                sci / fsci,
                f1.max(f2) / f1.min(f2),
                s1.max(s2) / s1.min(s2),
            );
        }
    }

    for &(op, default_lo, default_hi) in CASES {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        // `FSCI_SPECIAL_XMIN`/`FSCI_SPECIAL_XMAX` narrow the domain to ONE band, which is
        // how the per-branch instruction counts are taken. The default spans everything;
        // a band is a diagnostic, never the headline, because a fixture confined to one
        // branch benchmarks that branch and says nothing about the function.
        let (lo, hi) = (default_lo, default_hi);
        let lo = std::env::var("FSCI_SPECIAL_XMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(lo);
        let hi = std::env::var("FSCI_SPECIAL_XMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(hi);
        println!("n={n} op={op} domain=[{lo}, {hi}]");
        let x: Vec<f64> = (0..n).map(|i| lo + unit(i) * (hi - lo)).collect();

        let mut scipy = Scipy::start(op, &x);
        println!("{}", scipy.ready);

        // Built ONCE, outside the timed region. SciPy's arm is `fn(x)` on an array it
        // already holds — it does not copy its input — so cloning 200k f64 into a fresh
        // tensor on every call would charge us 1.6 MB of memcpy per iteration that the
        // incumbent never pays, and would be measuring the wrapper rather than the kernel.
        let tensor = SpecialTensor::RealVec(x.clone());

        let ours = || -> Vec<f64> {
            let out = call_ours(op, &tensor);
            real_vec(out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")), op)
        };

        black_box(ours());

        // `FSCI_SPECIAL_PROBE=<k>` runs EXACTLY k calls of our arm and exits before SciPy
        // is ever timed. It exists so `perf stat --no-inherit -e instructions` divided by
        // `k * n` is an instructions-per-element figure with NOTHING else in it.
        //
        // Reading that number off the full A/B run does not work and the failure is quiet:
        // the harness executes a fixed but unstated number of timing ROUNDS, so a
        // per-element figure computed from the whole run is inflated by that multiplier and
        // looks like ~1750 instructions for a kernel that is one log and one divide. The
        // BAND-TO-BAND RATIO survives that error, which is exactly why it is easy to keep
        // and quote a number that is wrong by an unknown constant factor.
        if let Ok(k) = std::env::var("FSCI_SPECIAL_PROBE") {
            let k: usize = k.parse().expect("FSCI_SPECIAL_PROBE must be an integer");
            let started = Instant::now();
            for _ in 0..k {
                black_box(ours());
            }
            let ms = started.elapsed().as_secs_f64() * 1.0e3;
            println!(
                "PROBE op={op} calls={k} n={n} elements={} ms={ms:.3} hoist_hits={} \
                 prealloc_hits={}",
                k * n,
                fsci_special::GAMMALN_HOIST_THRESHOLD_HITS
                    .load(std::sync::atomic::Ordering::Relaxed),
                fsci_special::GAMMA_FAMILY_PREALLOC_FILL_HITS
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            continue;
        }

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
        emit!(
            "case=n{n} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );

        // ── The two arms of a lever, interleaved INSIDE ONE PROCESS ──────────────────
        //
        // WHY THIS EXISTS, AND IT IS NOT A CONVENIENCE. Builds are remote, so the arms used
        // to be selected by an environment variable and compared across two `rch exec`
        // invocations. That silently did not work: `.rch.env` sets
        // `RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,FSCI_REQUIRE_SCIPY_ORACLE`, so NO
        // `FSCI_SPECIAL_*` variable reaches the worker at all. Both "arms" ran the same
        // code and reported 0.459x and 0.393x — a 17% spread between two runs that were
        // bit-identical, which is the cross-invocation noise floor and is far larger than
        // any lever worth arguing about.
        //
        // So the arms are switched here, in-process, and interleaved round by round with
        // the same position-balancing as the SciPy comparison above. `arm_null` is the A/A
        // control: the SAME arm timed twice within a round. A lever ratio is only readable
        // if it stands clear of the nulls.
        if let Some(sweep) = arm_sweep(op) {
            let (name, set, shipping_on) = sweep;
            let time_arm = |on: bool| -> f64 {
                set(on);
                let started = Instant::now();
                for _ in 0..reps {
                    black_box(ours());
                }
                started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
            };
            let (mut on_ms, mut off_ms, mut null_on, mut null_off) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            // Deliberately MORE rounds than the SciPy comparison gets. That one is bounded
            // by the cost of driving a Python child through a pipe; this one is two Rust
            // closures, so precision is cheap here and it is needed: the first run of this
            // sweep returned 1.122x against A/A nulls of 1.194 and 1.126, an effect smaller
            // than its own control, which is not a measurement of anything.
            let arm_rounds = rounds * 5;
            for round in 0..arm_rounds {
                let (t1, f1, f2, t2) = if round % 2 == 0 {
                    let t1 = time_arm(true);
                    let f1 = time_arm(false);
                    let f2 = time_arm(false);
                    let t2 = time_arm(true);
                    (t1, f1, f2, t2)
                } else {
                    let f1 = time_arm(false);
                    let t1 = time_arm(true);
                    let t2 = time_arm(true);
                    let f2 = time_arm(false);
                    (t1, f1, f2, t2)
                };
                on_ms.push(t1.min(t2));
                off_ms.push(f1.min(f2));
                null_on.push(t1.max(t2) / t1.min(t2));
                null_off.push(f1.max(f2) / f1.min(f2));
            }
            // Accuracy of each arm against the live SciPy values, from the same process.
            //
            // The hit counter is sampled around each arm as a two-sided control: the `on`
            // arm MUST increment it and the `off` arm MUST NOT. "Enabled" is not "took
            // effect", and a lever wired to nothing prints a perfectly clean 1.000x.
            let hits = arm_hits(op).expect("a lever with a sweep must expose a hit counter");
            set(true);
            let h0 = hits();
            let check_on = scipy.check(&ours());
            let on_hits = hits() - h0;
            set(false);
            let h1 = hits();
            let check_off = scipy.check(&ours());
            let off_hits = hits() - h1;
            emit!(
                "armab op={op} lever={name} control on_hits={on_hits} off_hits={off_hits} \
                 (on_hits must be >0; off_hits 0 UNLESS the shipped default already equals \
                 the on arm at this size, in which case the sweep is an A/A null)"
            );
            // Aggregate at the REPLICATE level: one ratio per round, then the median of
            // those, so the pairing that the interleave bought is not thrown away by
            // dividing two independently-computed medians. The range is printed because a
            // median that sits inside the spread of its own A/A nulls is not a result.
            //
            // `per_round` is built BEFORE the medians because `median` takes its vector by
            // value; the other order does not compile, and a merge put it back once.
            let mut per_round: Vec<f64> = on_ms
                .iter()
                .zip(off_ms.iter())
                .map(|(on, off)| off / on)
                .collect();
            per_round.sort_by(f64::total_cmp);
            let (on_med, off_med) = (median(on_ms), median(off_ms));
            emit!(
                "armab op={op} lever={name} on={on_med:.3}ms off={off_med:.3}ms \
                 rounds={arm_rounds}"
            );
            emit!(
                "armab op={op} lever={name} off/on={:.3}x paired_median={:.3}x \
                 paired_min={:.3}x paired_max={:.3}x null_on={:.3} null_off={:.3}",
                off_med / on_med,
                median(per_round.clone()),
                per_round[0],
                per_round[per_round.len() - 1],
                median(null_on),
                median(null_off),
            );
            // Each arm against the INCUMBENT, not just against each other. `off/on` says
            // which of our two schedules is quicker and says nothing about whether either
            // beats SciPy — and it is the second question the cell is actually judged on.
            //
            // CROSS-WINDOW, deliberately labelled: `scipy_ms` is the median from the paired
            // SciPy comparison above, taken in a different window from the arm sweep. It is
            // the same figure the `case=` line already reports, so these two ratios are
            // directly comparable to that line and NOT to the paired `off/on` beside them.
            emit!(
                "armab op={op} lever={name} crosswindow scipy/on={:.3}x scipy/off={:.3}x \
                 (case line reports scipy/off)",
                scipy_ms / on_med,
                scipy_ms / off_med,
            );
            emit!("armab op={op} lever={name} arm=on  {check_on}");
            emit!("armab op={op} lever={name} arm=off {check_off}");
            // RESTORE THE SHIPPING VALUE, which is not always `on`.
            //
            // This used to be `set(true)`, on the assumption that a lever's `on` arm is what
            // ships. For a DIAGNOSTIC lever such as `force_serial` it is not, and the
            // assumption leaked: after y0's sweep the override stayed pinned and every later
            // Bessel op ran serial, which is exactly what the y1 hit counter caught by
            // reporting on_hits=0. State that outlives its op silently rewrites every
            // measurement after it.
            set(shipping_on);
        }
    }

    // Printed AFTER the timed work as well as before it. A single reading taken at startup
    // says the host was quiet when the process began and nothing about whether it stayed
    // that way; the pair brackets the measurement, so a load spike that arrived mid-run is
    // visible in the row rather than hidden inside a median.
    gamma_gate_size_sweep();
    // y1 is the worst cell and its deficit survives every structural explanation tried so
    // far; x = 5 is where its kernel changes shape, so that is where to look next.
    band_sweep("y1", (0.01, 30.0), 5.0, 9);

    emit!("provenance_after {}", host_provenance());
}

/// Where does fanning out start to pay for `gamma`? Measured, per size, in this process.
///
/// WHY THIS EXISTS. `gamma` has preferred the SERIAL schedule in every run so far and across
/// three different host load levels, so its fan-out threshold is wrong at the harness's
/// fixture size. Knowing that is not enough to fix it: the gate is a SIZE threshold, and
/// moving it needs the size at which the two schedules cross, not a single point plus a
/// guess. One point would only justify "not at 200000", which is not a threshold.
///
/// This is an INTERNAL comparison — our two schedules against each other — so it is
/// maintenance, not a win, and it is reported as such. What makes it actionable is that the
/// live-incumbent ratio at the fixture size is measured separately by the arm sweep above.
///
/// Both schedules are bit-identical: `map_real_infallible` gives each index its own output
/// slot and applies the same kernel either way.
/// Split an op's domain at a KERNEL BRANCH BOUNDARY and compare each side against the
/// incumbent separately.
///
/// WHY. Every cell in this harness averages an op over its whole fixture, so a kernel with two
/// branches reports one number that belongs to neither of them. `y1` splits at x = 5: below it
/// Cephes evaluates a rational AND calls `j1` AND takes a logarithm; above it evaluates a
/// different rational with a sin/cos phase. Those are not the same amount of work, they are
/// not present in the same proportion, and a single ratio cannot say which one is behind.
///
/// This is a DIAGNOSTIC, not a headline: a fixture confined to one branch benchmarks that
/// branch and says nothing about the op as a whole. Both bands are always reported together,
/// with the proportion each occupies in the real fixture, so neither can be quoted alone as
/// "y1's ratio".
fn band_sweep(op: &str, full: (f64, f64), split: f64, rounds: usize) {
    let n: usize = 200_000;
    let unit = |i: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(40_503) % 1_000_003;
        k as f64 / 1_000_003.0
    };
    let share = (split - full.0) / (full.1 - full.0);
    emit!(
        "bandsweep op={op} split_at={split} lower_share_of_full_fixture={share:.3} \
         note=diagnostic-per-branch-not-a-headline-ratio"
    );

    for (lo, hi, label) in [(full.0, split, "lower"), (split, full.1, "upper")] {
        let x: Vec<f64> = (0..n).map(|i| lo + unit(i) * (hi - lo)).collect();
        let mut scipy = Scipy::start(op, &x);
        let tensor = SpecialTensor::RealVec(x);
        let ours = || -> Vec<f64> {
            let out = call_ours(op, &tensor);
            real_vec(out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")), op)
        };
        black_box(ours());
        let _ = scipy.time(1, 1);

        // Same reps for both arms, sized off the faster of a few calls, as in the main loop.
        let mut single = f64::INFINITY;
        for _ in 0..3 {
            let started = Instant::now();
            black_box(ours());
            single = single.min(started.elapsed().as_secs_f64() * 1.0e3);
        }
        let reps = (20.0_f64 / single.max(1.0e-6)).ceil() as usize;
        let reps = reps.clamp(1, 4096);

        let time_ours = || -> f64 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(ours());
            }
            started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
        };

        let (mut fs, mut sp, mut nf, mut ns) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
            fs.push(a1.min(a2));
            sp.push(s1.min(s2));
            nf.push(a1.max(a2) / a1.min(a2));
            ns.push(s1.max(s2) / s1.min(s2));
        }
        // Aggregate at the replicate level, and withhold a ratio that does not clear the
        // looser of the two nulls rather than printing one that decides nothing.
        let mut per_round: Vec<f64> = sp.iter().zip(fs.iter()).map(|(s, f)| s / f).collect();
        per_round.sort_by(f64::total_cmp);
        let ratio = median(per_round.clone());
        let worst_null = median(nf.clone()).max(median(ns.clone()));
        let resolved = ratio > worst_null || ratio < 1.0 / worst_null;
        let check = scipy.check(&ours());
        if resolved {
            emit!(
                "bandsweep op={op} band={label} domain=[{lo}, {hi}] scipy/fsci={ratio:.3}x \
                 min={:.3}x max={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
                per_round[0],
                per_round[per_round.len() - 1],
                median(nf),
                median(ns),
            );
        } else {
            emit!(
                "bandsweep op={op} band={label} domain=[{lo}, {hi}] scipy/fsci=WITHHELD-UNRESOLVED \
                 null_fsci={:.3} null_scipy={:.3} {check}",
                median(nf),
                median(ns),
            );
        }
    }
}

fn gamma_gate_size_sweep() {
    use std::sync::atomic::Ordering::Relaxed;

    // Three ops, and the third is a CONTROL rather than a target.
    //
    // A sweep that answered "SERIAL" for every op would be indistinguishable from a sweep
    // whose threaded arm is broken — forcing a threshold of 1 could, for instance, be
    // producing chunks so small that fanning out is doomed by construction. `digamma`
    // preferred the THREADED schedule in both runs that measured it, so it is included
    // precisely to see the instrument say THREADED. If it does not, the other two rows are
    // evidence about the harness, not about the gate.
    for (op, lo, hi, set_serial) in [
        (
            "gamma",
            0.01_f64,
            30.0_f64,
            (|serial: bool| {
                fsci_special::GAMMA_FAMILY_PAR_MIN_OVERRIDE.store(
                    if serial { usize::MAX } else { 1 },
                    std::sync::atomic::Ordering::Relaxed,
                );
            }) as fn(bool),
        ),
        ("digamma", 0.01, 60.0, |serial: bool| {
            fsci_special::GAMMA_FAMILY_PAR_MIN_OVERRIDE.store(
                if serial { usize::MAX } else { 1 },
                std::sync::atomic::Ordering::Relaxed,
            );
        }),
        ("y1", 0.01, 30.0, |serial: bool| {
            fsci_special::BESSEL_PAR_MIN_OVERRIDE.store(
                if serial { usize::MAX } else { 1 },
                std::sync::atomic::Ordering::Relaxed,
            );
        }),
    ] {
        gate_size_sweep_one(op, lo, hi, set_serial);
    }
    fsci_special::GAMMA_FAMILY_PAR_MIN_OVERRIDE.store(0, Relaxed);
    fsci_special::BESSEL_PAR_MIN_OVERRIDE.store(0, Relaxed);
}

/// One op's schedule-crossover sweep. See `gamma_gate_size_sweep` for why this exists.
fn gate_size_sweep_one(op: &str, lo: f64, hi: f64, set_serial: fn(bool)) {
    use std::sync::atomic::Ordering::Relaxed;

    // Sizes bracket the shipped threshold (1 << 17) on both sides, so the sweep can show it
    // being wrong in one direction and right in the other rather than only confirming a
    // prior. Below the threshold BOTH arms are serial, which is a built-in null: any
    // difference reported there is measurement noise and calibrates the rest of the column.
    const SIZES: &[usize] = &[1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21];
    const ROUNDS: usize = 9;

    emit!("gatesweep op={op} note=internal-A/B-not-an-incumbent-comparison");
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for &n in SIZES {
        // Same generator and same domain as the `gamma` case above, so the branch mix inside
        // `gamma_core` is the one the headline cell exercises rather than a different fixture
        // that happens to be easier or harder.
        let unit = |i: usize| -> f64 {
            let k = (i * 2_654_435_761usize).wrapping_add(40_503) % 1_000_003;
            k as f64 / 1_000_003.0
        };
        let x: Vec<f64> = (0..n).map(|i| lo + unit(i) * (hi - lo)).collect();
        let tensor = SpecialTensor::RealVec(x);
        let run = || {
            let out = call_ours(op, &tensor);
            real_vec(out.unwrap_or_else(|e| panic!("fsci {op} failed: {e}")), op)
        };

        // The setter uses `1`, not `0`, for the threaded arm: `0` means "use the shipped
        // constant", which at the small sizes here would silently make the threaded arm
        // serial too and turn the whole row into a null that looks like a result.
        set_serial(false);
        black_box(run());

        let time = |serial: bool| -> f64 {
            set_serial(serial);
            let started = Instant::now();
            black_box(run());
            started.elapsed().as_secs_f64() * 1.0e3
        };

        let (mut ser, mut par, mut null_s, mut null_p) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            let (s1, p1, p2, s2) = if round % 2 == 0 {
                let s1 = time(true);
                let p1 = time(false);
                let p2 = time(false);
                let s2 = time(true);
                (s1, p1, p2, s2)
            } else {
                let p1 = time(false);
                let s1 = time(true);
                let s2 = time(true);
                let p2 = time(false);
                (s1, p1, p2, s2)
            };
            ser.push(s1.min(s2));
            par.push(p1.min(p2));
            null_s.push(s1.max(s2) / s1.min(s2));
            null_p.push(p1.max(p2) / p1.min(p2));
        }
        let mut per_round: Vec<f64> = ser.iter().zip(par.iter()).map(|(s, p)| p / s).collect();
        per_round.sort_by(f64::total_cmp);
        let verdict = {
            let ratio = median(per_round.clone());
            let worst_null = median(null_s.clone()).max(median(null_p.clone()));
            // Withhold rather than publish an unresolvable ratio: if the effect does not
            // clear the looser of the two A/A nulls in BOTH directions it decides nothing,
            // and a number printed next to the word "prefers" will be read as a decision.
            if ratio > worst_null {
                "SERIAL"
            } else if ratio < 1.0 / worst_null {
                "THREADED"
            } else {
                "UNRESOLVED"
            }
        };
        let serial_ms = median(ser);
        rows.push((n, serial_ms));
        emit!(
            "gatesweep op={op} n={n} serial={serial_ms:.3}ms threaded={:.3}ms par/ser={:.3}x \
             min={:.3}x max={:.3}x null_serial={:.3} null_threaded={:.3} prefers={verdict}",
            median(par),
            median(per_round.clone()),
            per_round[0],
            per_round[per_round.len() - 1],
            median(null_s),
            median(null_p),
        );
    }

    // SELF-FLAG ROWS THAT WERE DISTURBED MID-MEASUREMENT.
    //
    // The serial arm is a plain O(n) loop, so its cost PER ELEMENT must be constant across
    // sizes. It is the one quantity in this sweep whose correct value is known a priori, and
    // a row that departs from it was interfered with while it ran — by another tenant on the
    // box, or by this harness's own neighbours — no matter how tidy its own A/A null looked.
    //
    // This matters because such a row does not fail loudly: it reports UNRESOLVED, which
    // reads like a genuine "no effect here" and quietly weakens a real conclusion. Two gamma
    // rows in this very run did exactly that. Flagged rather than dropped, because deciding
    // for the reader which measurements to discard is not the harness's job.
    let mut costs: Vec<f64> = rows.iter().map(|&(n, ms)| ms / n as f64).collect();
    costs.sort_by(f64::total_cmp);
    let typical = costs[costs.len() / 2];
    for (n, ms) in rows {
        let rel = (ms / n as f64) / typical;
        let flag = if !(0.67..=1.5).contains(&rel) {
            "DISTURBED-discard-this-row"
        } else {
            "ok"
        };
        emit!("gatesweep op={op} n={n} serial_cost_rel={rel:.2} row={flag}");
    }
    set_serial(false);
    fsci_special::GAMMA_FAMILY_PAR_MIN_OVERRIDE.store(0, Relaxed);
    fsci_special::BESSEL_PAR_MIN_OVERRIDE.store(0, Relaxed);
}

/// The in-process A/B lever for an op, if it has one: a display name and a setter.
///
/// Returning `None` for an op with no lever is what keeps the sweep from printing a
/// meaningless self-comparison — an `armab` line for an op whose two "arms" are the same
/// code would read exactly like a measured null and mean nothing.
fn arm_sweep(op: &str) -> Option<(&'static str, fn(bool), bool)> {
    match op {
        "gammaln" => Some((
            "cephes_lgam",
            |on| {
                fsci_special::GAMMALN_CEPHES_LGAM.store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        "erfcinv" => Some((
            "ndtri_not_acklam",
            |on| {
                fsci_special::ERFCINV_NDTRI.store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        "erfinv" => Some((
            "infallible_batch",
            |on| {
                fsci_special::ERFINV_INFALLIBLE_BATCH
                    .store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        // TWO DIFFERENT QUESTIONS, one on each sibling, so a single run answers both and
        // neither can be reported selectively. y1 asks whether the threaded path's
        // per-worker Vec plus concatenation is worth removing; y0 asks whether threading
        // pays here at all. They share `map_real_input`, so each answer informs the other.
        // Was `fused_par_write`, which measured null and is settled; y1's live question is
        // now whether the per-element atomics in its core are what stops it scaling.
        "y1" => Some((
            "hoist_flag_out_of_kernel",
            |on| {
                fsci_special::BESSEL_Y01_HOIST_FLAG.store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        // shipping = FALSE: `on` here is the diagnostic serial arm, not what ships.
        "y0" => Some((
            "force_serial",
            |on| {
                fsci_special::BESSEL_PAR_MIN_OVERRIDE.store(
                    if on { usize::MAX } else { 0 },
                    std::sync::atomic::Ordering::Relaxed,
                );
            },
            false,
        )),
        // `on` = stay SERIAL for this batch, `off` = the shipped threshold (fan out at
        // 131072). All four of gamma/rgamma/digamma/gammaln share the gate, so one lever
        // drives each of them.
        //
        // Applied to THREE ops rather than one on purpose. A single op showing that its own
        // fan-out is what puts it behind the incumbent is a fact about that op; the same
        // result across ops that share the SCHEDULE but not the kernel is what separates
        // "our gamma kernel is slow" from "our thread policy is wrong at this size on this
        // host" — and those two have entirely different fixes. (`gammaln`'s sweep slot is
        // taken by `cephes_lgam`; one lever per op.)
        // shipping = FALSE: `on` is the diagnostic serial arm. Restoring `true` here is
        // what pinned the gate for every op that ran after gamma.
        // rgamma's live question is now the same per-element-atomics one that was worth
        // several-fold on y0/y1, not the schedule (which came back contradictory twice and
        // is deliberately still open).
        // One lever, several ops: the defect is a class, and ops sharing a mapper should show
        // effects that track worker count. i0/i1/k0/k1 fan out through `map_real_input`
        // (16 workers at this size); rgamma through the gamma family's mapper (6).
        "i0" | "i1" | "k0" | "k1" => Some((
            "hoist_element_flags",
            |on| {
                fsci_special::SPECIAL_HOIST_ELEMENT_FLAGS
                    .store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        "rgamma" => Some((
            "hoist_flag_out_of_kernel",
            |on| {
                fsci_special::RGAMMA_HOIST_FLAG.store(on, std::sync::atomic::Ordering::Relaxed);
            },
            true,
        )),
        "gamma" | "digamma" => Some((
            "force_serial",
            |on| {
                fsci_special::GAMMA_FAMILY_PAR_MIN_OVERRIDE.store(
                    if on { usize::MAX } else { 0 },
                    std::sync::atomic::Ordering::Relaxed,
                );
            },
            false,
        )),
        _ => None,
    }
}

/// The hit counter for an op's lever, so the sweep can prove the arm actually fired.
///
/// A lever wired to nothing prints a perfectly clean 1.000x and a perfectly clean accuracy
/// comparison; only a counter that moves on one arm and not the other separates "the switch
/// did nothing" from "the switch did nothing measurable".
fn arm_hits(op: &str) -> Option<fn() -> usize> {
    match op {
        "gammaln" => Some(|| {
            fsci_special::GAMMALN_CEPHES_LGAM_HITS.load(std::sync::atomic::Ordering::Relaxed)
        }),
        "erfcinv" => {
            Some(|| fsci_special::ERFCINV_NDTRI_HITS.load(std::sync::atomic::Ordering::Relaxed))
        }
        "erfinv" => Some(|| {
            fsci_special::ERFINV_INFALLIBLE_BATCH_HITS.load(std::sync::atomic::Ordering::Relaxed)
        }),
        "y1" => Some(|| {
            fsci_special::BESSEL_Y01_HOIST_FLAG_HITS.load(std::sync::atomic::Ordering::Relaxed)
        }),
        "y0" => {
            Some(|| fsci_special::BESSEL_SERIAL_HITS.load(std::sync::atomic::Ordering::Relaxed))
        }
        "i0" | "i1" | "k0" | "k1" => Some(|| {
            fsci_special::SPECIAL_HOIST_ELEMENT_FLAGS_HITS
                .load(std::sync::atomic::Ordering::Relaxed)
        }),
        "rgamma" => {
            Some(|| fsci_special::RGAMMA_HOIST_FLAG_HITS.load(std::sync::atomic::Ordering::Relaxed))
        }
        "gamma" | "digamma" => Some(|| {
            fsci_special::GAMMA_FAMILY_SERIAL_HITS.load(std::sync::atomic::Ordering::Relaxed)
        }),
        _ => None,
    }
}
