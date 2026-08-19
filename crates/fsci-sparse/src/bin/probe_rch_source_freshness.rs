//! frankenscipy-ozg54: does `rch exec --` ever compile a STALE snapshot of the working tree?
//!
//! ozg54 recorded a clippy run on `vmi1293453` that reported pre-edit line numbers after 65
//! lines had been inserted above them, then produced shifted, correct locations on the very
//! next identical invocation with no local edit in between. The bead lists the cause as NOT
//! DIAGNOSED: rsync raciness against a checkout several agents write continuously, an
//! incremental-sync fingerprint missing a same-size write, and a snapshot cache were all
//! left open.
//!
//! ## Why this probe rather than repeating the clippy observation
//!
//! What made the original detectable was luck of a specific kind: the edit shifted LINE
//! NUMBERS, and diagnostics that cite locations are self-checking against a stale snapshot
//! in a way a pass/fail count is not. An edit changing only values would have been
//! invisible. So rather than depend on that luck, this file makes the staleness question
//! directly observable and independent of what the compiler chooses to report:
//!
//!   * `MARKER` is rewritten by the driver before every build, so each iteration has a
//!     content the previous build could not have seen.
//!   * `include_bytes!` of this very file is compiled in, so the binary carries the SHA-256
//!     of the source the compiler actually read.
//!
//! Comparing the compiled-in SHA against the local file's SHA answers the question with no
//! interpretation: equal means the remote build saw the working tree, unequal means it
//! compiled something else, and the printed marker says WHICH earlier revision it was.
//!
//! ## This is a COUNT, not a timing
//!
//! The output is a match/mismatch per iteration. It reads identically on an idle and a
//! saturated host, so it needs no build slot and load cannot invalidate it — which matters
//! because `acquire_build_slot` is currently disabled fleet-wide (frankenscipy-fr78g) and no
//! timed row is certifiable at all.

use sha2::{Digest, Sha256};

/// Rewritten by the driver before each remote build. The value is deliberately unique per
/// iteration so that a stale build is identifiable rather than merely suspicious: the
/// binary reports which revision it was compiled from, not just that something is off.
const MARKER: &str = "ozg54-baseline";

/// The bytes the COMPILER read for this file, captured at compile time. If `rch` served a
/// stale snapshot, this is the stale content and its digest will not match the file now on
/// local disk.
const SELF_SOURCE: &[u8] = include_bytes!("probe_rch_source_freshness.rs");

fn main() {
    println!("marker={MARKER}");
    println!("compiled_source_sha256={:x}", Sha256::digest(SELF_SOURCE));
    println!("compiled_source_len={}", SELF_SOURCE.len());
    // Host identity from inside the measuring process: `rch` dispatches per invocation and
    // does move between them, so a worker named in a build log is not evidence about which
    // worker ran this.
    println!(
        "worker_host={}",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .unwrap_or_default()
            .trim()
    );
}
