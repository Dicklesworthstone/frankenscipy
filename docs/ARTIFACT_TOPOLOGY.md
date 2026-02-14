# Artifact Topology (Locked)

This document defines the canonical directory structure for all FrankenSciPy project artifacts. Changes to this topology require explicit governance approval (see AGENTS.md).

## Schema Files

```
docs/schemas/
  behavior_ledger.schema.json    # -A bead output schema
  contract_table.schema.json     # -B bead output schema
  threat_matrix.schema.json      # -C bead output schema
```

## Foundation Artifacts

```
docs/
  essence_extraction_ledger.json  # bd-3jh.1: project-wide behavior ledger
  threat_matrix.json              # bd-3jh.2: project-wide threat matrix
  ARTIFACT_TOPOLOGY.md            # this file (topology lock)
```

## Per-Packet Artifacts (Phase-2C)

Each packet `FSCI-P2C-{001..008}` produces artifacts in two locations:

### Conformance Fixtures and Reports

Runtime test fixtures and generated parity evidence:

```
crates/fsci-conformance/fixtures/
  FSCI-P2C-{NNN}_{family}.json             # input fixture corpus
  smoke_case.json                           # minimal smoke test
  artifacts/FSCI-P2C-{NNN}/
    parity_report.json                      # conformance parity report
    parity_report.raptorq.json              # RaptorQ repair sidecar
    parity_report.decode_proof.json         # decode proof artifact
    oracle_capture.json                     # Python oracle output (optional)
    oracle_capture.error.txt                # oracle fallback log (when scipy absent)
```

### Packet Documentation Artifacts

Per-packet extraction and governance documents:

```
artifacts/phase2c/FSCI-P2C-{NNN}/
  legacy_anchor_map.md                      # -A bead: legacy path + behavior extraction
  behavior_ledger.json                      # -A bead: structured ledger (validates against schema)
  contract_table.json                       # -B bead: contract table (validates against schema)
  contract_table.md                         # -B bead: human-readable contract summary
  threat_model.json                         # -C bead: packet threat matrix (validates against schema)
  risk_note.md                              # -C bead: boundary risks and mitigations
  implementation_plan.md                    # -D bead: Rust implementation plan
  fixture_manifest.json                     # fixture IDs and oracle mapping
  parity_gate.yaml                          # strict + hardened pass criteria
```

## RaptorQ Artifact Envelope

All long-lived evidence artifacts (parity reports, benchmark baselines, migration manifests, reproducibility ledgers) must be emitted with RaptorQ sidecars following the envelope schema from COMPREHENSIVE_SPEC section 19:

```
{artifact_name}.json                        # primary artifact
{artifact_name}.raptorq.json                # RaptorQ repair symbol manifest
{artifact_name}.decode_proof.json           # decode proof log
```

Required sidecar fields:
- `schema_version`: integer (currently 1)
- `source_hash`: blake3 hash of the primary artifact
- `symbol_size`: encoding symbol size in bytes
- `source_symbols`: number of source symbols
- `repair_symbols`: number of repair symbols
- `repair_symbol_hashes`: blake3 hashes of each repair symbol

## Benchmark Artifacts

```
target/criterion/                            # criterion benchmark reports (gitignored)
benches/                                     # benchmark source files per crate
```

## Validation

Schema validation is enforced by:
```
cargo test -p fsci-conformance --test schema_validation
```

This test loads all three schemas, verifies structural validity, and validates sample documents against each schema.

## Governance

This topology is locked. Modifications require:
1. An explicit governance proposal in a bead or issue.
2. Review and approval by the project owner.
3. Update to this document and the schema validation test.
4. Zero-regression confirmation on all existing artifacts.
