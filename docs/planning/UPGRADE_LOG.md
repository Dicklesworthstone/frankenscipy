# Dependency Upgrade Log

**Date:** 2026-04-22  
**Project:** frankenscipy  
**Language:** Rust

## Summary

- Updated: 4 manifest entries plus 1 dev-dependency pin
- Verified current: `asupersync`, `ftui`, `blake3`, `serde`, `serde_json`, `thiserror`, `proptest`, `criterion`
- Blocked validation: workspace currently does not compile due unrelated live `fsci-opt` breakage

## crates.io Verification

Verified against the crates.io API:

- `asupersync` → `0.3.1`
- `ftui` → `0.3.1`
- `blake3` → `1.8.4`
- `serde` → `1.0.228`
- `serde_json` → `1.0.149`
- `thiserror` → `2.0.18`
- `proptest` → `1.11.0`
- `criterion` → `0.8.2`
- `rand` → `0.10.1`
- `toml` → `1.1.2+spec-1.1.0`
- `nalgebra` → `0.34.2`
- `tempfile` → `3.27.0`

## Manifest Updates

### Root workspace

- `rand`: `0.9.1` → `0.10.1`
- `toml`: `1.1` → `1.1.2`

### Crate manifests

- `crates/fsci-arrayapi/Cargo.toml`
  - `nalgebra`: `0.33.2` → `0.34.2`
- `crates/fsci-linalg/Cargo.toml`
  - `nalgebra`: `0.33.2` → `0.34.2`
- `crates/fsci-sparse/Cargo.toml`
  - `nalgebra`: `0.33.2` → `0.34.2`
- `crates/fsci-integrate/Cargo.toml`
  - `nalgebra`: `0.33.2` → `0.34.2`
- `crates/fsci-conformance/Cargo.toml`
  - `tempfile`: `3` → `3.27.0`

### Fuzz crate (workspace-excluded)

- `fuzz/Cargo.toml`
  - `libfuzzer-sys`: `0.4.10` → `0.4.12` (2026-04-22, cc-scipy) — no breaking changes; patch-level bumps of libFuzzer upstream and `rand`/`flate2` transitive pins.

## asupersync Consistency

- Workspace dependency is pinned to `asupersync = 0.3.1`
- `fsci-runtime` and `fsci-conformance` consume `asupersync` via `{ workspace = true }`
- `Cargo.lock` currently resolves `asupersync` to `0.3.1`

## Validation Status

Requested commands will be run after manifest edits:

- `rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_frankenscipy_cod cargo update`
- `rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_frankenscipy_cod cargo check --workspace`
- `rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_frankenscipy_cod cargo test --workspace`

Current known blocker before/after upgrade:

- `crates/fsci-opt/src/minimize.rs`
- `crates/fsci-opt/src/types.rs`

Those files currently fail workspace compilation independent of these dependency bumps.

## 2026-04-22 Re-verification (cc-scipy)

Re-ran crates.io lookups for all 12 workspace deps plus fuzz-crate deps (`arbitrary`, `libfuzzer-sys`). All are at latest stable. Only delta was `libfuzzer-sys 0.4.10 → 0.4.12` in `fuzz/Cargo.toml` (see above). The `fsci-opt` compile breakage in `minimize.rs` persists and is tracked as a separate workstream — it was not introduced by any dep bump.
