# SPEC_CROSSWALK_FRANKENSQLITE_TO_FRANKENSCIPY

This document defines how FrankenSciPy inherits proven methodology from:

- `reference/frankensqlite/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

The rule is inheritance-first, divergence-explicit.

## 1. Section-Level Crosswalk

| FrankenSQLite pattern family | FrankenSciPy adaptation |
|---|---|
| Multi-gate release topology (lint, tests, conformance, perf, durability) | identical gate layering for numerical routines and solver portfolios |
| Evidence-ledger-first decision systems | runtime policy controllers emit explicit posterior + expected-loss records |
| RaptorQ pervasive artifact durability | conformance/benchmark/risk artifacts emit sidecars + decode-proof metadata |
| strict vs hardened compatibility doctrine | strict SciPy parity vs hardened defensive operations with audited drift |
| profile-first optimization with isomorphism proofs | solver and kernel optimization requires parity + invariant replay proofs |

## 2. Mandatory Divergence Notes

Where FrankenSciPy diverges from FrankenSQLite, each change must explain:

1. Why the domain (scientific numerics) requires adaptation.
2. Why the original SQLite-centric pattern is insufficient unchanged.
3. Which guardrails remain equivalent.

## 3. Adopted Artifact Schemas

FrankenSciPy standardizes the following artifact classes:

1. `parity_report.json`
2. `parity_report.raptorq.json`
3. `parity_report.decode_proof.json`
4. benchmark delta report
5. compatibility drift report
6. runtime decision evidence ledger snapshot

## 4. Immediate Execution Mapping (P2C)

| Packet | Immediate inherited pattern |
|---|---|
| `FSCI-P2C-001` (`validate_tol` and IVP contracts) | spec-first differential fixture + sidecar durability |
| `FSCI-P2C-002` (`linalg`) | conformance-first workspace with one-lever optimization loop |
| `FSCI-P2C-003` (`optimize`) | decision-theoretic controller + evidence ledger |
| `FSCI-P2C-004` (`sparse`) | invariant-first tests + adversarial drift gates |
| `FSCI-P2C-005..007` (`fft/special/array-api`) | strict/hardened route auditing + compatibility matrix |
| `FSCI-P2C-008` (`conformance`) | release-grade artifact pipeline and dashboard surface |

## 5. Current State

Implemented in this cycle:

1. Imported full exemplar spec file into `reference/frankensqlite`.
2. Added explicit inheritance contract sections to `COMPREHENSIVE_SPEC_FOR_FRANKENSCIPY_V1.md`.
3. Implemented first packet-grade conformance + sidecar path in `fsci-conformance`.
