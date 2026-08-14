//! Does this binary contain the source that is on disk RIGHT NOW? (frankenscipy-eibro)
//!
//! THE DEFECT THIS EXISTS FOR. 2026-08-08 (TopazOsprey) an identical
//! `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p fsci-stats ...` invocation reported
//! `1 passed` over a golden constant that had been edited on disk to be wrong by
//! ~2000x its declared tolerance; a later run on a different worker failed exactly as
//! predicted. Every bead in this campaign closes on remote test output, so a remote
//! green that can be served from a stale binary is not evidence that the change under
//! test was compiled at all.
//!
//! WHY THE OBVIOUS CHECK DOES NOT WORK. Grepping the run log for `Compiling <crate>`
//! is unsound in both directions: under `cargo test -q` / `cargo run -q` the line is
//! suppressed, so 10 provably-fresh runs logged it zero times (measured 2026-08-08,
//! recorded in AGENTS.md). Its absence proves nothing.
//!
//! WHAT THIS PROBE DOES INSTEAD. It carries two hashes of the same two source files
//! and compares them:
//!
//!   * COMPILE-TIME — `include_str!` embeds the text of `../lib.rs` and of this file
//!     into the binary at build time. rustc cannot produce this binary without
//!     reading those exact bytes, so the embedded text IS the source the binary was
//!     built from. No build script, no dependency on cargo's own freshness logic.
//!   * RUN TIME — the same two files are re-read from `CARGO_MANIFEST_DIR` when the
//!     probe runs, i.e. from the checkout as it exists on the machine executing it.
//!
//! A mismatch is a stale binary: the compiler never saw the bytes now on disk. This
//! catches the cross-worker case the earlier 12-run negative could not (all 12 landed
//! on one worker, so no worker ever held a stale per-worker target cache).
//!
//! IT ALSO SEPARATES THE TWO FAILURES. When run under `rch`, the run-time read
//! happens on the WORKER, so:
//!   `compile_time != run_time`            → stale build cache on that worker
//!   `run_time != your local source hash`  → source sync never delivered your edit
//! Pass `--expect-marker <value>` to assert the third, end-to-end question — did MY
//! edit reach the binary — and the probe exits 1 when it did not.
//!
//! TWO-ARM CONTROL (mandatory before quoting this probe — a check that cannot fail
//! proves nothing). Run it once as-is: it must PASS. Then edit `SELF_MARKER` below,
//! DO NOT rebuild, and run the binary again: `self` must report STALE. Observing only
//! the passing arm shows blindness, not freshness. `--selftest` runs the equivalent
//! must-hit/must-miss pair on the hashing itself, in-process.
//!
//! Run:
//!   cargo run --release -p fsci-linalg --bin probe_build_freshness \
//!       --features freshness-probe -- [--expect-marker <value>] [--selftest]

#[cfg(feature = "freshness-probe")]
mod probe {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};

    /// Flip this to run the must-miss arm of the two-arm control. Its value is echoed
    /// so a caller can assert end-to-end that its own edit reached the binary.
    const SELF_MARKER: &str = "MARKER-A";

    /// The library source, as rustc read it while building this binary.
    const COMPILED_LIB_SOURCE: &str = include_str!("../lib.rs");
    /// This probe's own source, as rustc read it while building this binary.
    const COMPILED_SELF_SOURCE: &str = include_str!("probe_build_freshness.rs");

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Outcome of comparing one file's compiled-in text against the same file on disk.
    struct Comparison {
        name: &'static str,
        path: PathBuf,
        compiled_sha256: String,
        compiled_len: usize,
        runtime: Option<(String, usize)>,
    }

    impl Comparison {
        fn new(name: &'static str, path: PathBuf, compiled: &str) -> Self {
            let runtime = std::fs::read(&path)
                .ok()
                .map(|bytes| (sha256(&bytes), bytes.len()));
            Self {
                name,
                path,
                compiled_sha256: sha256(compiled.as_bytes()),
                compiled_len: compiled.len(),
                runtime,
            }
        }

        /// `Some(true)` fresh, `Some(false)` stale, `None` when the source file is not
        /// reachable at run time (a bare binary copied off its checkout) — which is
        /// UNDECIDED, never a pass.
        fn fresh(&self) -> Option<bool> {
            self.runtime
                .as_ref()
                .map(|(hash, _)| *hash == self.compiled_sha256)
        }

        fn report(&self) {
            let verdict = match self.fresh() {
                Some(true) => "FRESH",
                Some(false) => "STALE",
                None => "UNDECIDED(source unreadable at run time)",
            };
            let (runtime_sha, runtime_len) = self
                .runtime
                .as_ref()
                .map_or_else(|| ("-".to_string(), 0), |(hash, len)| (hash.clone(), *len));
            println!(
                "{:<5} {verdict} compiled_sha256={} compiled_bytes={} runtime_sha256={runtime_sha} \
                 runtime_bytes={runtime_len} path={}",
                self.name,
                self.compiled_sha256,
                self.compiled_len,
                self.path.display()
            );
        }
    }

    /// Must-hit / must-miss pair on the comparison itself, so a run can show the check
    /// is capable of reporting STALE at all before any FRESH verdict is believed.
    fn selftest() -> bool {
        let hit = sha256(b"identical") == sha256(b"identical");
        let miss = sha256(b"identical") != sha256(b"identicaL");
        println!("selftest: must_hit={hit} must_miss={miss}");
        hit && miss
    }

    pub fn run() {
        let exe = std::env::current_exe().expect("current_exe");
        let elf_sha256 = std::fs::read(&exe).map_or_else(|_| "-".to_string(), |b| sha256(&b));
        println!("probe_build_freshness marker={SELF_MARKER}");
        println!("elf_sha256={elf_sha256} elf_path={}", exe.display());
        println!(
            "host={} manifest_dir={}",
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .map_or_else(|_| "unavailable".to_string(), |t| t.trim().to_string()),
            env!("CARGO_MANIFEST_DIR")
        );

        let mut args = std::env::args().skip(1);
        let mut expect_marker: Option<String> = None;
        let mut want_selftest = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--expect-marker" => expect_marker = args.next(),
                "--selftest" => want_selftest = true,
                other => {
                    eprintln!("unknown argument {other:?}");
                    std::process::exit(2);
                }
            }
        }

        if want_selftest && !selftest() {
            eprintln!("ABORT: hashing selftest failed — no freshness verdict is admissible");
            std::process::exit(3);
        }

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let comparisons = [
            Comparison::new("lib", manifest.join("src/lib.rs"), COMPILED_LIB_SOURCE),
            Comparison::new(
                "self",
                manifest.join("src/bin/probe_build_freshness.rs"),
                COMPILED_SELF_SOURCE,
            ),
        ];
        for comparison in &comparisons {
            comparison.report();
        }

        let stale: Vec<&str> = comparisons
            .iter()
            .filter(|c| c.fresh() == Some(false))
            .map(|c| c.name)
            .collect();
        let undecided: Vec<&str> = comparisons
            .iter()
            .filter(|c| c.fresh().is_none())
            .map(|c| c.name)
            .collect();

        let marker_ok = expect_marker
            .as_ref()
            .map(|expected| expected == SELF_MARKER);
        if let Some(expected) = expect_marker.as_ref() {
            println!(
                "marker_check: expected={expected} observed={SELF_MARKER} => {}",
                if marker_ok == Some(true) {
                    "MATCH"
                } else {
                    "MISMATCH (this binary predates your edit)"
                }
            );
        }

        let verdict_ok = stale.is_empty() && undecided.is_empty() && marker_ok != Some(false);
        println!(
            "VERDICT: {} stale={:?} undecided={:?}",
            if verdict_ok { "FRESH" } else { "NOT FRESH" },
            stale,
            undecided
        );
        if !verdict_ok {
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "freshness-probe")]
fn main() {
    probe::run();
}

#[cfg(not(feature = "freshness-probe"))]
fn main() {
    eprintln!("probe_build_freshness requires --features freshness-probe");
    std::process::exit(2);
}
