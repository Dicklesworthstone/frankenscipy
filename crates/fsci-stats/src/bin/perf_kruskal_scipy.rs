//! `frankenscipy-921i0` — LIVE head-to-head for `kruskal` against `scipy.stats.kruskal`.
//!
//! One invocation, both arms: this process times fsci's `kruskal`, then shells out to the
//! interpreter to time `scipy.stats.kruskal` on the SAME integers built by the SAME expression,
//! in the same window. A remembered or published scipy number is not evidence here.
//!
//! The two arms print their H and p so agreement is checked before any ratio is believed — a
//! head-to-head between two functions computing different things is not a ratio.
//!
//! Requires `FSCI_REQUIRE_SCIPY=1` to certify: without it a missing scipy would make this
//! print an fsci-only line and look like a pass (the green-by-skipping hazard).
//!
//! Usage: perf_kruskal_scipy [reps] [n ...]
use fsci_stats::kruskal;
use std::hint::black_box;
use std::time::Instant;

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            let f: Vec<&str> = s.split_whitespace().take(3).collect();
            (f.len() == 3).then(|| f.join("/"))
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

fn mhz_mean() -> f64 {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|t| {
            let v: Vec<f64> = t
                .lines()
                .filter(|l| l.starts_with("cpu MHz"))
                .filter_map(|l| l.split(':').nth(1)?.trim().parse::<f64>().ok())
                .collect();
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        })
        .unwrap_or(f64::NAN)
}

/// SHA-256 of the ELF actually executing, plus the path it resolved to.
///
/// `/proc/self/exe` MUST be resolved in THIS process before handing anything to `sha256sum`:
/// passing the literal path makes the child read ITS OWN `/proc/self/exe`, so every row reports
/// the checksum of `/usr/bin/sha256sum`. That is not a hypothetical — it is what this harness
/// did on its first run, and it was caught only because two DIFFERENT binaries printed the same
/// sha (2a9b9ccf..., which is `sha256sum` itself). The resolved path is returned and printed
/// alongside the digest so the row names the file that was actually hashed.
fn self_elf_sha256() -> (String, String) {
    let exe = match std::fs::read_link("/proc/self/exe") {
        Ok(p) => p,
        Err(e) => return ("unavailable".to_string(), format!("<unresolved: {e}>")),
    };
    let digest = std::process::Command::new("sha256sum")
        .arg(&exe)
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8(o.stdout)
                .ok()?
                .split_whitespace()
                .next()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unavailable".to_string());
    (digest, exe.display().to_string())
}

fn fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let modulus = (n / 4).max(2) as u64;
    let mut g = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        let v = ((i as u64).wrapping_mul(7919) % modulus) as f64;
        g[i % 3].push(v);
    }
    let [a, b, c] = g;
    (a, b, c)
}

/// The python arm. `modulus` and the `i % 3` split are written out here rather than passed as
/// data so a reader can check by eye that both arms build the same integers.
const PY: &str = r#"
import sys, time, numpy as np
from scipy.stats import kruskal
n = int(sys.argv[1]); reps = int(sys.argv[2])
modulus = max(n // 4, 2)
i = np.arange(n, dtype=np.uint64)
v = ((i * np.uint64(7919)) % np.uint64(modulus)).astype(np.float64)
groups = [np.ascontiguousarray(v[k::3]) for k in range(3)]  # match fsci's contiguous Vecs
kruskal(*groups)                      # warm
ts = []
for _ in range(reps):
    t = time.perf_counter()
    r = kruskal(*groups)
    ts.append((time.perf_counter() - t) * 1e3)
ts.sort()
print("%.6f %.17g %.17g" % (ts[len(ts)//2], r.statistic, r.pvalue))
"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(15);
    let sizes: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![20_000, 200_000, 2_000_000]
    };
    let require = std::env::var("FSCI_REQUIRE_SCIPY").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".to_string());

    println!("# harness=perf_kruskal_scipy bead=frankenscipy-921i0 incumbent=scipy.stats.kruskal");
    let (elf_sha, elf_path) = self_elf_sha256();
    println!("# elf_sha256={elf_sha} elf_path={elf_path}");
    println!(
        "# threads_observed={} require_scipy={require}",
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(0)
    );

    let mut failed = false;
    for &n in &sizes {
        let (a, b, c) = fixture(n);
        let groups: Vec<&[f64]> = vec![&a, &b, &c];

        // Interleave the two arms so neither owns a window of its own: scipy, fsci, scipy.
        let load_a = loadavg();
        let mhz_a = mhz_mean();
        let py1 = run_python(&python, n, reps);

        let _ = black_box(kruskal(&groups));
        let mut fsci_ms = Vec::with_capacity(reps);
        let mut r = kruskal(&groups);
        for _ in 0..reps {
            let t = Instant::now();
            r = black_box(kruskal(black_box(&groups)));
            fsci_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let fsci_med = med(&fsci_ms);
        let mhz_f = mhz_mean();
        let load_f = loadavg();

        let py2 = run_python(&python, n, reps);
        let load_b = loadavg();

        let (sp1, sp2) = match (py1, py2) {
            (Some(p1), Some(p2)) => (p1, p2),
            _ => {
                println!(
                    "n={n} SCIPY-ARM-MISSING fsci {fsci_med:.3}ms H={} p={}",
                    r.statistic, r.pvalue
                );
                if require {
                    failed = true;
                }
                continue;
            }
        };

        // Incumbent A/A null: the two scipy samples bracket the fsci arm, so a window drift
        // over the triple shows up here rather than in the ratio.
        let scipy_med = (sp1.0 + sp2.0) / 2.0;
        let scipy_null = sp1.0 / sp2.0;

        let h_gap = ((r.statistic - sp1.1) / sp1.1.abs().max(1.0)).abs();
        let p_gap = ((r.pvalue - sp1.2) / sp1.2.abs().max(1e-300)).abs();
        let agree = h_gap < 1e-12 && p_gap < 1e-9;

        println!(
            "n={n} fsci {fsci_med:.3}ms scipy {scipy_med:.3}ms | RATIO(scipy/fsci) {:.3}x \
             | scipy A/A null {scipy_null:.4} | agree={agree} H fsci={} scipy={} \
             (relgap {h_gap:.3e}) p fsci={} scipy={} (relgap {p_gap:.3e})",
            scipy_med / fsci_med,
            r.statistic,
            sp1.1,
            r.pvalue,
            sp1.2
        );
        println!(
            "    load scipy1={load_a} fsci={load_f} scipy2={load_b} | MHz mean scipy1={mhz_a:.0} \
             fsci={mhz_f:.0}"
        );
        if !agree {
            failed = true;
        }
    }

    if failed {
        eprintln!("FAIL: an arm was missing (with FSCI_REQUIRE_SCIPY=1) or the two arms disagreed");
        std::process::exit(1);
    }
}

fn run_python(python: &str, n: usize, reps: usize) -> Option<(f64, f64, f64)> {
    let out = std::process::Command::new(python)
        .arg("-c")
        .arg(PY)
        .arg(n.to_string())
        .arg(reps.to_string())
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "# scipy arm failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let f: Vec<f64> = text
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    (f.len() == 3).then(|| (f[0], f[1], f[2]))
}
