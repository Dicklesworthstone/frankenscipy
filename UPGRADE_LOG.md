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
