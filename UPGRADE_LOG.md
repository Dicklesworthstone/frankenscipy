# Dependency Upgrade Log

**Date:** 2026-09-11  
**Project:** frankenscipy  
**Language:** Rust  
**Manifest:** Cargo.toml  

---

## Summary

| Metric | Count |
|--------|-------|
| **Total dependencies** | 12 |
| **Updated** | 6 |
| **Skipped** | 6 |
| **Failed (rolled back)** | 0 |
| **Requires attention** | 0 |

---

## Successfully Updated

### blake3: 1.8.4 → 1.8.7
- **Breaking changes:** None
- **Notable changes:** Maintenance release, AVX-512/NEON optimizations, bug fixes.
- **Tests:** ✓ Passed

### serde: 1.0.228 → 1.0.229
- **Breaking changes:** None
- **Notable changes:** Derive macro optimizations, compiler compatibility updates.
- **Tests:** ✓ Passed

### serde_json: 1.0.149 → 1.0.151
- **Breaking changes:** None
- **Notable changes:** Float parsing edge cases and formatter performance improvements.
- **Tests:** ✓ Passed

### thiserror: 2.0.18 → 2.0.20
- **Breaking changes:** None
- **Notable changes:** Macro diagnostic precision enhancements.
- **Tests:** ✓ Passed

### rand: 0.10.1 → 0.10.2
- **Breaking changes:** None
- **Notable changes:** Distribution sampling optimizations, documentation refinements.
- **Tests:** ✓ Passed

### toml: 1.1.2 → 1.1.6
- **Breaking changes:** None
- **Notable changes:** Spec 1.1.0 conformance alignments and decoder performance.
- **Tests:** ✓ Passed

---

## Skipped

### asupersync: 0.3.9
- **Reason:** Pinned version for RaptorQ systematic encoding contracts in fsci-conformance; synchronous numerical core prohibits async runtime drift.

### ftui: 0.3.1
- **Reason:** Pinned optional dashboard TUI facade.

### proptest: 1.11.0
- **Reason:** Already on latest stable version.

### criterion: 0.8.2
- **Reason:** Already on latest stable version.

### npyz: 0.9.1
- **Reason:** Already on latest stable version.

### zip: 8.6.0
- **Reason:** Latest published is 9.0.0-pre3 (pre-release); skipped per stability rules.

---

## Post-Upgrade Checklist

- [x] All compiler checks passing (`cargo check --workspace --all-targets`)
- [x] No deprecation warnings introduced
- [x] Tokio ban maintained (`cargo tree -i tokio` empty)
- [x] Documentation updated in UPGRADE_LOG.md

---

# Dependency Pass — 2026-11-26

**Scope:** semver-compatible refresh of `Cargo.lock` and `fuzz/Cargo.lock`; no manifest
requirement changed.

## Summary

| Metric | Count |
|--------|-------|
| **Lock entries refreshed** | 20 (`Cargo.lock`), 14 (`fuzz/Cargo.lock`) |
| **Major upgrades taken** | 0 |
| **Held** | `sha2` 0.10.9 (0.11.2 available) |
| **Failed (rolled back)** | 0 |
| **Advisories (`cargo audit`)** | 0 before, 0 after (334 and 259 crates) |

## Refreshed (patch and minor releases)

`ahash` 0.8.14, `anyhow` 1.0.105, `clap` 4.6.10, `clap_builder` 4.6.10,
`constant_time_eq` 0.6.2, `crc32fast` 1.5.6, `litrs` 1.1.0, `lru` 0.19.2,
`pest`/`pest_derive`/`pest_generator`/`pest_meta` 2.9.7, `sha2` 0.11.2 (the copy
`asupersync` uses), `toml` 1.2.3, `toml_edit` 0.25.21, `toml_parser` 1.1.7,
`zerocopy`/`zerocopy-derive` 0.8.66, `zip` 9.8.2, `zlib-rs` 0.6.9. The fuzz lock takes
the subset it contains.

- **Hot paths:** none of these is in the normal dependency graph of fsci-linalg,
  fsci-fft, fsci-opt, fsci-special or fsci-stats (`cargo tree -e normal -p <crate>`).
  In fsci-sparse they sit only under the optional `npz` archive codec (`zip`, `zlib-rs`,
  `crc32fast`, `constant_time_eq`, and `pest` through `npyz`). The rest serve
  fsci-conformance, benches and build tooling. So there was no kernel to A/B.
- **Gates (pinned nightly, x86_64 Linux):** `cargo clippy --locked --workspace
  --all-targets -- -D warnings` is clean. Against live SciPy 1.17.1, `diff_sparse`
  (39/39, including `.npz` wire compatibility), `diff_sparse_lobpcg` and
  `diff_stats_multivariate` pass. `cargo audit` reports nothing for either lock.

## Held

### sha2: 0.10.9 → 0.11.2
- **Reason:** `digest` 0.11's output type no longer implements `LowerHex`, so 27 files
  that format digests with `{:x}` need an explicit encoder. All of them are optional
  perf/identity binaries, benches or test helpers; no library code path uses `sha2`.
  Tracked in [#9](https://github.com/Dicklesworthstone/frankenscipy/issues/9).
- `generic-array` stays at 0.14.7 because `crypto-common` 0.1.7, in the same 0.10
  `digest` chain, requires exactly `=0.14.7`.
