//! `frankenscipy-921i0` GATE — paired-interleaved A/B for the `kruskal` tie-correction lever.
//!
//! ARM `resort` (ORIG): clone the combined observations, sort them a SECOND time, re-walk the
//! `==` tie groups to rebuild `Σ(t³ − t)`.
//! ARM `pass` (HEAD):   take `Σ(t³ − t)` from the ranking pass that already sorted and already
//! walked those groups.
//!
//! Both arms live in THIS binary, selected by `KRUSKAL_FORCE_TIE_RESORT`, and are alternated
//! ABBA inside one window — window drift on this host is larger than the lever, so an absolute
//! before/after across two windows would measure the window (verify-paired-interleaved).
//!
//! WHAT THIS DOES NOT MEASURE. The tie-sum ACCUMULATION added to the ranking pass runs in BOTH
//! arms, so the reported ratio is a LOWER BOUND on the true pre-lever/post-lever ratio, and it
//! says nothing about whether that accumulation slowed down `rankdata`'s other callers.
//!
//! Gate (a) SHARE is answered by the size sweep: the removed work is a sort, O(n log n), against
//! an O(n) remainder, so `share = 1 − 1/ratio` must GROW with n. If it does not, the model is
//! wrong and the lever should be reported as a loss regardless of the ratio.
//!
//! Usage: perf_kruskal_tie_resort [reps] [n ...]
use fsci_stats::{KRUSKAL_FORCE_TIE_RESORT, kruskal};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            let f: Vec<&str> = s.split_whitespace().take(3).collect();
            (f.len() == 3).then(|| f.join("/"))
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

/// Cross-core clock spread at this instant, not one figure for the run: this host runs the
/// powersave governor and different cores sit at different frequencies simultaneously, so a
/// single number would hide the variable that matters.
fn mhz() -> (f64, f64, f64) {
    let text = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(t) => t,
        Err(_) => return (f64::NAN, f64::NAN, f64::NAN),
    };
    let v: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("cpu MHz"))
        .filter_map(|l| l.split(':').nth(1)?.trim().parse::<f64>().ok())
        .collect();
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (
        v.iter().copied().fold(f64::MAX, f64::min),
        mean,
        v.iter().copied().fold(f64::MIN, f64::max),
    )
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

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// Three groups over `n` observations, values from a small alphabet so tie groups are real.
/// The python arm of the head-to-head builds the SAME integers with the SAME expression, so
/// neither arm can be handed different data.
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(21);
    let sizes: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![20_000, 200_000, 2_000_000]
    };

    println!("# harness=perf_kruskal_tie_resort bead=frankenscipy-921i0");
    let (elf_sha, elf_path) = self_elf_sha256();
    println!("# elf_sha256={elf_sha} elf_path={elf_path}");
    println!(
        "# host={} threads_observed={} reps={reps}",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".into()),
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(0),
    );

    let mut trend: Vec<(usize, f64)> = Vec::new();

    for &n in &sizes {
        let (a, b, c) = fixture(n);
        let groups: Vec<&[f64]> = vec![&a, &b, &c];

        // BYTE-IDENTITY, checked on the real fixture before any timing is believed.
        KRUSKAL_FORCE_TIE_RESORT.store(true, Ordering::Relaxed);
        let r_orig = kruskal(&groups);
        KRUSKAL_FORCE_TIE_RESORT.store(false, Ordering::Relaxed);
        let r_pass = kruskal(&groups);
        let bitmism = usize::from(r_orig.statistic.to_bits() != r_pass.statistic.to_bits())
            + usize::from(r_orig.pvalue.to_bits() != r_pass.pvalue.to_bits());

        // MUST-HIT control, reported not assumed: the tie mass `Σ(t³−t)/(N³−N)` and the number
        // of tie GROUPS. Zero means the fixture has no ties, the correction is vacuous, and the
        // row would be describing a case the lever's byte-identity argument never has to
        // survive. Reported as a fraction in scientific notation rather than as `C = 1 − it`,
        // because at these sizes a real correction is O(1e-8) and `C` prints as `1.000000`,
        // which is exactly how a tie-free fixture would look.
        let (tie_frac, tie_groups) = {
            let mut all: Vec<f64> = Vec::with_capacity(n);
            for g in &groups {
                all.extend_from_slice(g);
            }
            all.sort_unstable_by(f64::total_cmp);
            let mut tie_sum = 0.0_f64;
            let mut groups = 0usize;
            let mut i = 0usize;
            while i < all.len() {
                let mut j = i + 1;
                while j < all.len() && all[j] == all[i] {
                    j += 1;
                }
                let t = (j - i) as f64;
                if t > 1.0 {
                    tie_sum += t * t * t - t;
                    groups += 1;
                }
                i = j;
            }
            let nf = all.len() as f64;
            (tie_sum / (nf * nf * nf - nf), groups)
        };

        let time_one = |resort: bool| -> f64 {
            KRUSKAL_FORCE_TIE_RESORT.store(resort, Ordering::Relaxed);
            let t = Instant::now();
            let r = black_box(kruskal(black_box(&groups)));
            let dt = t.elapsed().as_secs_f64() * 1e3;
            black_box(r);
            KRUSKAL_FORCE_TIE_RESORT.store(false, Ordering::Relaxed);
            dt
        };

        // Warm the allocator and the page cache for both arms before the first timed pair.
        let _ = time_one(true);
        let _ = time_one(false);

        let load_before = loadavg();
        let mhz_before = mhz();

        let (mut cand, mut null) = (Vec::new(), Vec::new());
        let (mut orig_ms, mut pass_ms) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            // ABBA: A B B A. The two A samples bracket the two B samples, so a monotone
            // drift over the quartet cancels in the A/B ratio and shows up in the A/A null.
            let a1 = time_one(true);
            let b1 = time_one(false);
            let b2 = time_one(false);
            let a2 = time_one(true);
            cand.push(((a1 + a2) / 2.0) / ((b1 + b2) / 2.0));
            null.push(a1 / a2);
            orig_ms.push((a1 + a2) / 2.0);
            pass_ms.push((b1 + b2) / 2.0);
        }
        let load_after = loadavg();
        let mhz_after = mhz();

        let cand_med = med(&cand);
        let null_med = med(&null);
        let null_lo = null.iter().copied().fold(f64::MAX, f64::min);
        let null_hi = null.iter().copied().fold(f64::MIN, f64::max);
        let decided = cand_med > null_hi || cand_med < null_lo;
        let share = 1.0 - 1.0 / cand_med;
        trend.push((n, share));

        println!(
            "n={n} {} resort {:.3}ms pass {:.3}ms | CAND(resort/pass) median {cand_med:.4}x \
             | share {:.2}% | NULL(A/A) median {null_med:.4} range [{null_lo:.4},{null_hi:.4}] \
             | bitmism={bitmism} H={} p={} tie_mass={tie_frac:.3e} tie_groups={tie_groups} \
             ranking_path={}",
            if decided { "DECIDED " } else { "IN-FLOOR" },
            med(&orig_ms),
            med(&pass_ms),
            share * 100.0,
            r_pass.statistic,
            r_pass.pvalue,
            // Which sort the RANKING pass used. Above 1<<14 it is the radix argsort, so the
            // removed comparison sort is being weighed against O(n) work; below it, against
            // another comparison sort. The trend claim is meaningless without knowing which
            // regime each point is in.
            if n >= (1 << 14) {
                "radix"
            } else {
                "comparison"
            },
        );
        println!(
            "    load before={load_before} after={load_after} | MHz before min/mean/max \
             {:.0}/{:.0}/{:.0} after {:.0}/{:.0}/{:.0}",
            mhz_before.0, mhz_before.1, mhz_before.2, mhz_after.0, mhz_after.1, mhz_after.2
        );
    }

    if trend.len() >= 2 {
        // GATE(a) is a claim about the SIZE TREND, reported as observed rather than as
        // pass/fail: the bead predicted the share would GROW with n, on the model "an
        // O(n log n) sort removed from an O(n) remainder". Print the points and let the row
        // say which way it actually went.
        let grows = trend.windows(2).all(|w| w[1].1 >= w[0].1);
        println!(
            "# GATE(a) SHARE trend {} : {}",
            if grows {
                "GROWS-WITH-N"
            } else {
                "FLAT-OR-FALLING"
            },
            trend
                .iter()
                .map(|(n, s)| format!("n={n}:{:.2}%", s * 100.0))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
