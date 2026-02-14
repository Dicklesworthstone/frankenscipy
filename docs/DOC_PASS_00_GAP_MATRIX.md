# DOC-PASS-00: Baseline Gap Matrix + Quantitative Expansion Targets

Generated: 2026-02-14
Mandate: TOTAL SciPy feature parity (1500+ public functions across 20 domains)

## 1. Document Baseline Inventory

| Document | Lines | Words | Sections | Domains Covered | Domains Required |
|---|---|---|---|---|---|
| EXHAUSTIVE_LEGACY_ANALYSIS.md | 279 | 1390 | 17 | 7 | 20 |
| EXISTING_SCIPY_STRUCTURE.md | 62 | 308 | 8 | 7 | 20 |
| COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md | 437 | 2189 | 25 | 8 | 20 |
| FEATURE_PARITY.md | 42 | 327 | 4 | 8 | 20 |
| PROPOSED_ARCHITECTURE.md | 55 | 292 | 6 | 9 (crates) | 21 (crates) |
| PLAN_TO_PORT_SCIPY_TO_RUST.md | 58 | 242 | 5 | generic | 20 |
| PHASE2C_EXTRACTION_PACKET.md | 116 | 639 | 8 | 8 | 20 |
| MASTER_EXECUTION_TODO.md | 177 | 988 | varies | 8 | 20 |
| **TOTAL** | **1226** | **6375** | — | — | — |

## 2. Per-Section Gap Matrix: EXHAUSTIVE_LEGACY_ANALYSIS.md

| Section | Current Words | Target Words | Expansion Factor | Completeness % | Gap Severity |
|---|---|---|---|---|---|
| §0 Mission and Completion Criteria | 40 | 200 | 5x | 20% | high |
| §1 Source-of-Truth Crosswalk | 55 | 300 | 5x | 18% | high |
| §2 Quantitative Legacy Inventory | 80 | 400 | 5x | 40% | medium |
| §3 Subsystem Extraction Matrix | 120 | 2000 | 17x | 6% | **critical** |
| §4 Alien-Artifact Invariant Ledger | 60 | 600 | 10x | 10% | **critical** |
| §5 Native/C/Fortran/Cython Boundary Register | 80 | 800 | 10x | 10% | **critical** |
| §6 Compatibility and Security Doctrine | 80 | 400 | 5x | 20% | high |
| §7 Conformance Program | 60 | 600 | 10x | 10% | **critical** |
| §8 Extreme Optimization Program | 70 | 400 | 6x | 18% | high |
| §9 RaptorQ-Everywhere Artifact Contract | 40 | 200 | 5x | 20% | medium |
| §10 Phase-2C Execution Backlog | 80 | 1000 | 13x | 8% | **critical** |
| §11 Residual Gaps and Risks | 30 | 300 | 10x | 10% | high |
| §12 Deep-Pass Hotspot Inventory | 100 | 500 | 5x | 20% | medium |
| §13 Phase-2C Extraction Payload Contract | 90 | 400 | 4x | 23% | medium |
| §14 Strict/Hardened Compatibility Drift Budgets | 55 | 300 | 5x | 18% | medium |
| §15 Extreme-Software-Optimization Execution Law | 60 | 300 | 5x | 20% | medium |
| §16 RaptorQ Evidence Topology and Recovery Drills | 50 | 300 | 6x | 17% | medium |
| §17 Phase-2C Exit Checklist | 40 | 200 | 5x | 20% | medium |
| **TOTAL** | **1290** | **~9200** | **~7x** | **14%** | — |

### Gap Root Causes for EXHAUSTIVE_LEGACY_ANALYSIS.md

1. **Missing 13 domains entirely**: stats, signal, spatial, interpolate, ndimage, io, cluster, constants, misc, odr, datasets, differentiate are absent from §3 extraction matrix, §4 invariant ledger, §5 boundary register, §7 conformance program, and §10 execution backlog.
2. **V1-scoped language**: References to "scoped" and "V1" throughout limit perceived coverage.
3. **No per-function granularity**: The extraction matrix lists one row per domain when it needs one row per submodule (e.g., linalg alone has 10 submodules).
4. **No algorithm-level detail**: Missing specific algorithm descriptions, convergence properties, and numerical stability characteristics.
5. **No cross-domain dependency analysis**: No mapping of how domains interact (e.g., optimize uses linalg, signal uses fft).

## 3. Per-Section Gap Matrix: EXISTING_SCIPY_STRUCTURE.md

| Section | Current Words | Target Words | Expansion Factor | Completeness % | Gap Severity |
|---|---|---|---|---|---|
| §1 Legacy Oracle | 10 | 50 | 5x | 20% | low |
| §2 Subsystem Map | 60 | 1200 | 20x | 5% | **critical** |
| §3 Semantic Hotspots | 60 | 800 | 13x | 8% | **critical** |
| §4 Compatibility-Critical Behaviors | 35 | 600 | 17x | 6% | **critical** |
| §5 Security and Stability Risk Areas | 40 | 400 | 10x | 10% | high |
| §6 V1 Extraction Boundary | 30 | 0 | DELETE | — | **critical** |
| §7 High-Value Conformance Fixture Families | 40 | 400 | 10x | 10% | high |
| §8 Extraction Notes for Rust Spec | 25 | 200 | 8x | 13% | medium |
| **TOTAL** | **300** | **~3650** | **~12x** | **8%** | — |

### Gap Root Causes for EXISTING_SCIPY_STRUCTURE.md

1. **§6 must be deleted or rewritten**: "V1 Extraction Boundary" with "Exclude for V1: stats/spatial/ndimage/io etc." directly contradicts the total-parity mandate.
2. **§2 covers only 7 subsystems**: Needs all 20 SciPy domains with submodule breakdown.
3. **No file count per subsystem**: Only mentions legacy_scipy_code root without measured per-module counts.
4. **No API surface quantification**: Missing function/class counts per domain.

## 4. Per-Document Gap Matrix: Other Documents

### COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md

| Issue | Severity |
|---|---|
| Title contains "V1" implying scoped subset | **critical** |
| Scope section lists only 8 domains | **critical** |
| Conformance matrix covers only 7 routine families | high |
| Performance budgets reference only 4 packet workloads | high |
| Architecture section lists only 9 crates | high |
| Missing domains: stats, signal, spatial, interpolate, ndimage, io, cluster, constants, differentiate | **critical** |
| **Expansion Factor**: 10x-15x | — |

### FEATURE_PARITY.md

| Issue | Severity |
|---|---|
| Only 8 packets listed (P2C-001 through P2C-008) | **critical** |
| Missing 12 packets (P2C-009 through P2C-020) | **critical** |
| No per-function parity tracking | high |
| No completion percentage metrics | medium |
| **Expansion Factor**: 15x-20x | — |

### PROPOSED_ARCHITECTURE.md

| Issue | Severity |
|---|---|
| Crate map lists only 9 crates | **critical** |
| Missing 12 crates: fsci-stats, fsci-signal, fsci-spatial, fsci-interpolate, fsci-ndimage, fsci-io, fsci-cluster, fsci-constants, fsci-misc, fsci-odr, fsci-datasets, fsci-differentiate | **critical** |
| No inter-crate dependency diagram | high |
| No CASP integration architecture per domain | high |
| **Expansion Factor**: 10x-15x | — |

### PLAN_TO_PORT_SCIPY_TO_RUST.md

| Issue | Severity |
|---|---|
| Very thin (58 lines, 242 words) | high |
| No domain-specific porting strategy | **critical** |
| No risk assessment per domain | high |
| **Expansion Factor**: 15x-20x | — |

### PHASE2C_EXTRACTION_PACKET.md

| Issue | Severity |
|---|---|
| Only 8 packets defined | **critical** |
| Missing 12 packets for full parity | **critical** |
| No packet dependency/interaction matrix | high |
| **Expansion Factor**: 10x-15x | — |

## 5. Top-5 Highest-Gap Sections (Priority Ranking)

Ranked by (number of missing domains) * (downstream implementation dependency):

| Rank | Section | Document | Gap Score | Reason |
|---|---|---|---|---|
| **1** | §3 Subsystem Extraction Matrix | EXHAUSTIVE_LEGACY_ANALYSIS.md | 95/100 | Missing 13 of 20 domains. Every packet's anchor bead (-A) depends on this matrix. Currently 7 rows, needs 40+ rows (one per submodule). |
| **2** | §2 Subsystem Map | EXISTING_SCIPY_STRUCTURE.md | 93/100 | Missing 13 of 20 domains. Single-line descriptions need multi-paragraph depth with file counts, API surface, and semantic annotations. |
| **3** | Feature Family Matrix | FEATURE_PARITY.md | 90/100 | Missing 12 of 20 packets. Every conformance gate depends on this tracking document. |
| **4** | Crate Map | PROPOSED_ARCHITECTURE.md | 88/100 | Missing 12 of 21 crates. Architecture document is the reference for all implementation decisions. |
| **5** | §4 Alien-Artifact Invariant Ledger | EXHAUSTIVE_LEGACY_ANALYSIS.md | 85/100 | Only 5 invariants defined. Need invariants for all 20 domains (60+ total). Every -E test bead depends on these invariants. |

## 6. Required New Crates (Full Parity)

Current crates (9): fsci-integrate, fsci-linalg, fsci-opt, fsci-sparse, fsci-fft, fsci-special, fsci-arrayapi, fsci-runtime, fsci-conformance

Required additions (12):

| Crate | SciPy Domain | Estimated API Surface | Risk Level |
|---|---|---|---|
| fsci-stats | scipy.stats | 200+ functions, 100+ distributions | critical |
| fsci-signal | scipy.signal | 80+ functions | high |
| fsci-spatial | scipy.spatial | 40+ functions/classes | high |
| fsci-interpolate | scipy.interpolate | 30+ functions/classes | high |
| fsci-ndimage | scipy.ndimage | 50+ functions | medium |
| fsci-io | scipy.io | 15+ functions | medium |
| fsci-cluster | scipy.cluster | 25+ functions | medium |
| fsci-constants | scipy.constants | 50+ constants, 5+ functions | low |
| fsci-misc | scipy.misc | 4 functions (deprecated) | low |
| fsci-odr | scipy.odr | 10+ classes/functions (deprecated) | medium |
| fsci-datasets | scipy.datasets | 5 functions | low |
| fsci-differentiate | scipy.differentiate | 3 functions | medium |

## 7. Required New Packets (Full Parity)

Current packets (8): FSCI-P2C-001 through FSCI-P2C-008

Required additions (12):

| Packet ID | Domain | Risk Tier | Estimated Functions |
|---|---|---|---|
| FSCI-P2C-009 | stats | Critical | 200+ |
| FSCI-P2C-010 | signal | High | 80+ |
| FSCI-P2C-011 | spatial | High | 40+ |
| FSCI-P2C-012 | interpolate | High | 30+ |
| FSCI-P2C-013 | ndimage | Medium | 50+ |
| FSCI-P2C-014 | io | Medium | 15+ |
| FSCI-P2C-015 | cluster | Medium | 25+ |
| FSCI-P2C-016 | constants | Low | 55+ |
| FSCI-P2C-017 | misc | Low | 4 |
| FSCI-P2C-018 | odr | Medium | 10+ |
| FSCI-P2C-019 | datasets | Low | 5 |
| FSCI-P2C-020 | differentiate | Medium | 3 |

## 8. Cross-Domain Dependency Map

Dependencies that affect implementation ordering:

```
fsci-constants ← (used by) stats, special, integrate
fsci-special   ← (used by) stats, signal, integrate, optimize
fsci-linalg    ← (used by) stats, signal, spatial, interpolate, optimize, sparse
fsci-fft       ← (used by) signal, ndimage
fsci-sparse    ← (used by) spatial, signal, optimize, interpolate
fsci-integrate ← (used by) stats, special
fsci-opt       ← (used by) stats, signal, interpolate
fsci-spatial   ← (used by) interpolate (Delaunay for scattered data)
fsci-interpolate ← (used by) signal, ndimage
fsci-stats     ← (standalone with dependencies above)
fsci-signal    ← (standalone with dependencies above)
fsci-ndimage   ← (standalone with dependencies above)
fsci-io        ← (standalone, no numeric dependencies)
fsci-cluster   ← (uses spatial.distance, linalg)
```

## 9. Expansion Priority Matrix

| Priority | Action | Documents Affected | Impact |
|---|---|---|---|
| P0 | Add 13 missing domains to EXISTING_SCIPY_STRUCTURE.md §2 | 1 doc | Unblocks all domain-specific analysis |
| P0 | Delete/rewrite §6 "V1 Extraction Boundary" | 1 doc | Removes scope contradiction |
| P0 | Add 13 missing rows to EXHAUSTIVE_LEGACY_ANALYSIS.md §3 | 1 doc | Unblocks all packet anchor beads |
| P0 | Add 12 missing crates to PROPOSED_ARCHITECTURE.md | 1 doc | Architecture completeness |
| P0 | Add 12 missing packets to FEATURE_PARITY.md | 1 doc | Tracking completeness |
| P1 | Rename COMPREHENSIVE_SPEC title (remove "V1") | 1 doc | Signal full-parity intent |
| P1 | Expand §4 invariant ledger to all 20 domains | 1 doc | Test anchor completeness |
| P1 | Expand §5 boundary register to all native code | 1 doc | Security coverage |
| P1 | Add 12 new packets to PHASE2C_EXTRACTION_PACKET.md | 1 doc | Packet execution framework |
| P2 | Per-function parity tracking in FEATURE_PARITY.md | 1 doc | Fine-grained progress |
| P2 | Cross-domain interaction documentation | 1 doc | Dependency awareness |
| P2 | Algorithm-level detail in extraction matrix | 1 doc | Implementation guidance |

## 10. Summary Metrics

- **Current total documentation**: 1,226 lines / 6,375 words
- **Target total documentation**: ~15,000-20,000 lines / ~80,000 words
- **Overall expansion factor**: 10x-13x
- **Domain coverage**: 7/20 (35%) → 20/20 (100%)
- **Crate coverage**: 9/21 (43%) → 21/21 (100%)
- **Packet coverage**: 8/20 (40%) → 20/20 (100%)
- **Top bottleneck**: EXISTING_SCIPY_STRUCTURE.md §6 (explicit exclusion contradicting mandate)
