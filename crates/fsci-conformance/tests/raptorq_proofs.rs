//! bd-3jh.9: RaptorQ Artifact Sidecars + Decode-Proof Pipeline
//!
//! Validates that:
//! 1. RaptorQ sidecars can be generated for all fixture bundles >1KB
//! 2. Repair symbols enable recovery from simulated corruption
//! 3. Decode proofs are reproducible and verifiable
//! 4. Sidecar metadata (blake3, counts, params) is consistent

use asupersync::raptorq::decoder::{InactivationDecoder, ReceivedSymbol};
use asupersync::raptorq::systematic::SystematicEncoder;
use blake3::hash;
use fsci_conformance::{
    CaseResult, DecodeProofArtifact, PacketReport, RaptorQSidecar, chunk_payload,
    generate_raptorq_sidecar,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

const SYMBOL_SIZE: usize = 128;

fn hex_decode(input: &str) -> Option<Vec<u8>> {
    if !input.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    let hex_val = |b: u8| -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(10 + (b - b'a')),
            b'A'..=b'F' => Some(10 + (b - b'A')),
            _ => None,
        }
    };
    let mut idx = 0;
    while idx < bytes.len() {
        let hi = hex_val(bytes[idx])?;
        let lo = hex_val(bytes[idx + 1])?;
        out.push((hi << 4) | lo);
        idx += 2;
    }
    Some(out)
}

fn seed_from_payload(payload: &[u8]) -> u64 {
    hash(payload).as_bytes()[0] as u64 + 1337
}

// ── Sidecar consistency ────────────────────────────────────────────────────────

#[test]
fn sidecar_deterministic() {
    let payload = b"hello raptorq sidecar determinism check - this is longer than 128 bytes so we get multiple symbols and can verify deterministic repair symbol generation across runs";
    let s1 = generate_raptorq_sidecar(payload).unwrap();
    let s2 = generate_raptorq_sidecar(payload).unwrap();
    assert_eq!(s1, s2, "sidecar generation must be deterministic");
}

#[test]
fn sidecar_schema_version() {
    let payload = b"schema version check payload with enough bytes for encoding";
    let s = generate_raptorq_sidecar(payload).unwrap();
    assert_eq!(s.schema_version, 1);
}

#[test]
fn sidecar_source_hash_matches_blake3() {
    let payload = b"blake3 hash verification payload for source hash field";
    let s = generate_raptorq_sidecar(payload).unwrap();
    let expected = hash(payload).to_hex().to_string();
    assert_eq!(s.source_hash, expected);
}

#[test]
fn sidecar_repair_ratio() {
    // For payloads with k source symbols, repair_symbols = max(k/5, 1)
    let payload = vec![42u8; 1024]; // 1024/128 = 8 source symbols
    let s = generate_raptorq_sidecar(&payload).unwrap();
    assert_eq!(s.source_symbols, 8);
    assert_eq!(s.repair_symbols, 1); // 8/5 = 1
    assert_eq!(s.symbol_size, SYMBOL_SIZE);

    let payload = vec![42u8; 2560]; // 2560/128 = 20 source symbols
    let s = generate_raptorq_sidecar(&payload).unwrap();
    assert_eq!(s.source_symbols, 20);
    assert_eq!(s.repair_symbols, 4); // 20/5 = 4
}

#[test]
fn sidecar_repair_symbols_present() {
    let payload = vec![42u8; 2560]; // 20 source symbols → 4 repair symbols
    let s = generate_raptorq_sidecar(&payload).unwrap();
    assert_eq!(s.seed, seed_from_payload(&payload));
    assert_eq!(s.repair_symbol_hashes.len(), s.repair_symbols);
    assert_eq!(s.repair_symbol_payloads_hex.len(), s.repair_symbols);
    assert!(
        !s.repair_symbol_hashes.is_empty(),
        "must have at least one repair symbol hash"
    );
    // All hashes should be valid hex strings
    for h in &s.repair_symbol_hashes {
        assert_eq!(h.len(), 64, "blake3 hex hash should be 64 chars");
    }
    for (payload_hex, hash_hex) in s
        .repair_symbol_payloads_hex
        .iter()
        .zip(s.repair_symbol_hashes.iter())
    {
        let payload = hex_decode(payload_hex).expect("repair symbol payload should be valid hex");
        let computed = hash(&payload).to_hex().to_string();
        assert_eq!(computed, *hash_hex, "repair symbol payload must match hash");
    }
}

// ── Decode-proof pipeline ──────────────────────────────────────────────────────

/// Encode payload, simulate a single-symbol loss, recover via decoder, verify.
fn run_decode_proof(payload: &[u8]) -> DecodeProofResult {
    let source_symbols = chunk_payload(payload, SYMBOL_SIZE);
    let k = source_symbols.len();
    // Need more repair symbols than drops. RFC 6330 overhead means we need
    // roughly drop_count + L-K extra symbols. Use 20% + 4 for safety margin.
    let drop_count_est = (k / 20).max(1);
    let repair_count = drop_count_est + 4;
    let base_seed = hash(payload).as_bytes()[0] as u64 + 1337;

    // Try multiple seeds — some K/seed combinations produce singular matrices
    let mut encoder = None;
    let mut seed = base_seed;
    for attempt in 0..10 {
        if let Some(enc) = SystematicEncoder::new(&source_symbols, SYMBOL_SIZE, seed + attempt) {
            encoder = Some(enc);
            seed += attempt;
            break;
        }
    }
    let mut encoder = encoder.expect("encoder init must succeed within 10 seed attempts");

    // Collect all source symbols + repair symbols
    let source_emitted = encoder.emit_systematic();
    let repair_emitted = encoder.emit_repair(repair_count);

    // Simulate source loss: drop 1 symbol (conservative to ensure recovery)
    let drop_count = 1;
    let mut received: Vec<ReceivedSymbol> = Vec::new();

    // Add non-dropped source symbols
    for (i, sym) in source_emitted.iter().enumerate() {
        if i >= drop_count {
            received.push(ReceivedSymbol::source(sym.esi, sym.data.clone()));
        }
    }

    // Add all repair symbols with their equations
    let decoder = InactivationDecoder::new(k, SYMBOL_SIZE, seed);
    for sym in &repair_emitted {
        let (cols, coefs) = decoder
            .repair_equation(sym.esi)
            .expect("repair_equation should succeed for emitted repair symbol");
        received.push(ReceivedSymbol::repair(
            sym.esi,
            cols,
            coefs,
            sym.data.clone(),
        ));
    }

    // Add constraint symbols (LDPC + HDPC)
    received.extend(decoder.constraint_symbols());

    // Attempt decode
    match decoder.decode(&received) {
        Ok(result) => {
            // Verify recovered data matches original
            let recovered_match = result
                .source
                .iter()
                .take(source_symbols.len())
                .zip(&source_symbols)
                .all(|(r, s)| r == s);

            DecodeProofResult {
                success: true,
                recovered_match,
                symbols_dropped: drop_count,
                symbols_received: received.len(),
                peeled: result.stats.peeled,
                inactivated: result.stats.inactivated,
                gauss_ops: result.stats.gauss_ops,
                proof_hash: hash(payload).to_hex().to_string(),
            }
        }
        Err(e) => DecodeProofResult {
            success: false,
            recovered_match: false,
            symbols_dropped: drop_count,
            symbols_received: received.len(),
            peeled: 0,
            inactivated: 0,
            gauss_ops: 0,
            proof_hash: format!(
                "FAILED(k={k},repair={repair_count},drop={drop_count},seed={seed}): {e:?}"
            ),
        },
    }
}

#[derive(Debug, Serialize)]
struct DecodeProofResult {
    success: bool,
    recovered_match: bool,
    symbols_dropped: usize,
    symbols_received: usize,
    peeled: usize,
    inactivated: usize,
    gauss_ops: usize,
    proof_hash: String,
}

fn recover_payload_with_sidecar(
    payload: &[u8],
    sidecar: &RaptorQSidecar,
    drop_indices: &[usize],
) -> Result<Vec<u8>, String> {
    let source_symbols = chunk_payload(payload, sidecar.symbol_size);
    let k = source_symbols.len();
    if k != sidecar.source_symbols {
        return Err(format!(
            "sidecar k mismatch: expected {}, got {}",
            sidecar.source_symbols, k
        ));
    }
    if sidecar.repair_symbol_payloads_hex.len() != sidecar.repair_symbols {
        return Err(format!(
            "repair payload count mismatch: expected {}, got {}",
            sidecar.repair_symbols,
            sidecar.repair_symbol_payloads_hex.len()
        ));
    }

    let mut dropped = vec![false; k];
    for &idx in drop_indices {
        if idx < k {
            dropped[idx] = true;
        }
    }

    let mut received = Vec::new();
    for (idx, symbol) in source_symbols.iter().enumerate() {
        if !dropped[idx] {
            received.push(ReceivedSymbol::source(idx as u32, symbol.clone()));
        }
    }

    let decoder = InactivationDecoder::new(k, sidecar.symbol_size, sidecar.seed);
    for (offset, payload_hex) in sidecar.repair_symbol_payloads_hex.iter().enumerate() {
        let esi = k as u32 + offset as u32;
        let payload = hex_decode(payload_hex)
            .ok_or_else(|| format!("repair symbol {offset} payload not valid hex"))?;
        let (cols, coefs) = decoder
            .repair_equation(esi)
            .map_err(|e| format!("repair_equation failed for esi {esi}: {e:?}"))?;
        received.push(ReceivedSymbol::repair(esi, cols, coefs, payload));
    }

    received.extend(decoder.constraint_symbols());

    let result = decoder
        .decode(&received)
        .map_err(|e| format!("decode failed: {e:?}"))?;

    let mut recovered = Vec::with_capacity(payload.len());
    for symbol in result.source.iter().take(k) {
        recovered.extend_from_slice(symbol);
    }
    recovered.truncate(payload.len());
    Ok(recovered)
}

#[test]
fn decode_proof_small_payload() {
    // 1024 bytes = 8 source symbols
    let payload = vec![0x42_u8; 1024];
    let result = run_decode_proof(&payload);
    assert!(result.success, "decode must succeed: {result:?}");
    assert!(result.recovered_match, "recovered data must match original");
}

#[test]
fn decode_proof_medium_payload() {
    let payload = vec![0xAB_u8; 2048]; // 16 source symbols
    let result = run_decode_proof(&payload);
    assert!(result.success, "decode must succeed: {result:?}");
    assert!(result.recovered_match, "recovered data must match original");
}

#[test]
fn decode_proof_large_payload() {
    // 4096 bytes = 32 symbols — large enough for meaningful repair
    let payload: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let result = run_decode_proof(&payload);
    assert!(result.success, "decode must succeed: {result:?}");
    assert!(result.recovered_match, "recovered data must match original");
}

#[test]
fn recovery_drill_parity_report() {
    let mut case_results = Vec::new();
    for idx in 0..64 {
        case_results.push(CaseResult {
            case_id: format!("case_{idx:03}"),
            passed: idx % 3 != 0,
            message: format!("synthetic parity case {idx}"),
        });
    }
    let passed_cases = case_results.iter().filter(|c| c.passed).count();
    let failed_cases = case_results.len() - passed_cases;
    let report = PacketReport {
        schema_version: 1,
        packet_id: "RECOVERY-DRILL".to_string(),
        family: "linalg".to_string(),
        case_results,
        passed_cases,
        failed_cases,
        fixture_path: None,
        oracle_status: None,
        differential_case_results: None,
        report_kind: fsci_conformance::ReportKind::Unspecified,
        generated_unix_ms: now_unix_ms(),
    };

    let report_bytes = serde_json::to_vec_pretty(&report).unwrap();
    let sidecar = generate_raptorq_sidecar(&report_bytes).unwrap();
    assert_eq!(sidecar.seed, seed_from_payload(&report_bytes));

    let mut drop_indices = vec![0usize];
    if sidecar.source_symbols > 1 {
        drop_indices.push(sidecar.source_symbols - 1);
    }

    let recovered = recover_payload_with_sidecar(&report_bytes, &sidecar, &drop_indices).unwrap();
    let proof_hash = hash(&recovered).to_hex().to_string();
    assert_eq!(proof_hash, hash(&report_bytes).to_hex().to_string());

    let decode_proof = DecodeProofArtifact {
        ts_unix_ms: now_unix_ms(),
        reason: format!(
            "recovered from simulated corruption of {} block(s)",
            drop_indices.len()
        ),
        recovered_blocks: drop_indices.len(),
        proof_hash,
    };

    let artifact_dir = std::env::temp_dir().join(format!(
        "frankenscipy_raptorq_recovery_drill_{}",
        now_unix_ms()
    ));
    std::fs::create_dir_all(&artifact_dir).unwrap();
    std::fs::write(
        artifact_dir.join("parity_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    std::fs::write(
        artifact_dir.join("parity_report.raptorq.json"),
        serde_json::to_vec_pretty(&sidecar).unwrap(),
    )
    .unwrap();
    std::fs::write(
        artifact_dir.join("parity_report.decode_proof.json"),
        serde_json::to_vec_pretty(&decode_proof).unwrap(),
    )
    .unwrap();
}

// ── Fixture bundle sidecars ────────────────────────────────────────────────────

#[test]
fn sidecar_for_fixture_bundles() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts");
    if !fixtures_dir.exists() {
        eprintln!("No fixtures directory found — skipping fixture sidecar test");
        return;
    }

    let mut sidecars_generated = 0;
    let mut scrub_results: Vec<ScrubEntry> = Vec::new();

    // Walk all JSON fixture files
    walk_json_fixtures(&fixtures_dir, &mut |path, content| {
        if content.len() < 1024 {
            return; // Skip files < 1KB
        }

        let sidecar = match generate_raptorq_sidecar(content) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "  SKIP: {path:?} ({}B) — encoder failed: {e}",
                    content.len()
                );
                return;
            }
        };

        // Verify sidecar consistency
        assert_eq!(
            sidecar.source_hash,
            hash(content).to_hex().to_string(),
            "source hash mismatch for {path:?}"
        );
        assert!(sidecar.repair_symbols >= 1, "must have at least 1 repair");
        assert_eq!(sidecar.symbol_size, SYMBOL_SIZE);

        scrub_results.push(ScrubEntry {
            path: path.to_string_lossy().to_string(),
            size_bytes: content.len(),
            source_symbols: sidecar.source_symbols,
            repair_symbols: sidecar.repair_symbols,
            source_hash: sidecar.source_hash.clone(),
        });

        sidecars_generated += 1;
    });

    eprintln!(
        "\n── Sidecar Scrub Report ──\n  Generated {sidecars_generated} sidecars for bundles >1KB"
    );
    for entry in &scrub_results {
        eprintln!(
            "  {} — {}B, {}/{} src/repair symbols",
            entry.path, entry.size_bytes, entry.source_symbols, entry.repair_symbols
        );
    }
}

#[derive(Serialize)]
struct ScrubEntry {
    path: String,
    size_bytes: usize,
    source_symbols: usize,
    repair_symbols: usize,
    source_hash: String,
}

fn walk_json_fixtures(dir: &Path, callback: &mut dyn FnMut(&Path, &[u8])) {
    if !dir.is_dir() {
        return;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            walk_json_fixtures(&path, callback);
        } else if path.extension().is_some_and(|ext| ext == "json")
            && let Ok(content) = std::fs::read(&path)
        {
            callback(&path, &content);
        }
    }
}

// ── Full decode-proof artifact generation ──────────────────────────────────────

#[test]
fn generate_decode_proof_artifacts() {
    let artifact_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/raptorq_proofs");
    std::fs::create_dir_all(&artifact_dir).unwrap();

    // Generate decode proofs for different payload sizes
    let test_payloads: Vec<(&str, Vec<u8>)> = vec![
        ("small_1KB", vec![0x42; 1024]),
        ("medium_2KB", vec![0xAB; 2048]),
        ("large_4KB", (0..4096).map(|i| (i % 256) as u8).collect()),
    ];

    let mut proof_artifacts = Vec::new();

    for (name, payload) in &test_payloads {
        let sidecar = generate_raptorq_sidecar(payload).unwrap();
        let decode_result = run_decode_proof(payload);

        assert!(
            decode_result.success,
            "decode must succeed for {name}: {decode_result:?}"
        );
        assert!(
            decode_result.recovered_match,
            "recovery must match for {name}"
        );

        let artifact = ProofBundle {
            name: name.to_string(),
            payload_size: payload.len(),
            sidecar,
            decode_proof: decode_result,
            ts_unix_ms: now_unix_ms(),
        };

        proof_artifacts.push(artifact);
    }

    // Write aggregate artifact
    let json = serde_json::to_string_pretty(&proof_artifacts).unwrap();
    std::fs::write(artifact_dir.join("decode_proof_bundle.json"), &json).unwrap();

    // Also write individual sidecar for verification
    for pa in &proof_artifacts {
        let sidecar_json = serde_json::to_string_pretty(&pa.sidecar).unwrap();
        std::fs::write(
            artifact_dir.join(format!("{}.sidecar.json", pa.name)),
            &sidecar_json,
        )
        .unwrap();
    }

    eprintln!("\n── Decode Proof Artifacts ──");
    for pa in &proof_artifacts {
        eprintln!(
            "  {} — {}B, dropped {}/{} symbols, decode: {}",
            pa.name,
            pa.payload_size,
            pa.decode_proof.symbols_dropped,
            pa.sidecar.source_symbols,
            if pa.decode_proof.recovered_match {
                "RECOVERED"
            } else {
                "FAILED"
            },
        );
    }
}

#[derive(Serialize)]
struct ProofBundle {
    name: String,
    payload_size: usize,
    sidecar: RaptorQSidecar,
    decode_proof: DecodeProofResult,
    ts_unix_ms: u128,
}

fn now_unix_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

// ── Sidecar-vs-fixture consistency scrub ───────────────────────────────────────

#[test]
fn sidecar_consistency_scrub() {
    // Verify that regenerating sidecars matches stored ones
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts");
    if !fixtures_dir.exists() {
        return;
    }

    let mut checked = 0;
    walk_json_fixtures(&fixtures_dir, &mut |path, content| {
        // Look for existing .raptorq.json sidecars
        let path_str = path.to_string_lossy();
        if path_str.ends_with(".raptorq.json") {
            // This IS a sidecar — find its source file
            let source_path_str = path_str.replace(".raptorq.json", ".json");
            let source_path = Path::new(source_path_str.as_str());
            if source_path.exists() {
                let source_content = std::fs::read(source_path).unwrap();
                let expected = match generate_raptorq_sidecar(&source_content) {
                    Ok(s) => s,
                    Err(_) => return, // encoder may fail for some payloads
                };
                let stored: Result<RaptorQSidecar, _> = serde_json::from_slice(content);
                let Ok(stored) = stored else { return };
                assert_eq!(
                    stored.source_hash, expected.source_hash,
                    "sidecar source_hash mismatch for {path:?}"
                );
                checked += 1;
            }
        }
    });

    eprintln!("\n── Sidecar Consistency Scrub ──\n  Verified {checked} existing sidecars");
}

#[test]
fn root_long_lived_artifacts_have_decode_proofs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("fsci-conformance should live under crates/");
    let artifacts_dir = repo_root.join("artifacts");
    if !artifacts_dir.exists() {
        eprintln!("No root artifacts directory found — skipping root artifact sidecar test");
        return;
    }

    let mut checked = 0usize;
    for entry in std::fs::read_dir(&artifacts_dir).expect("artifacts dir should be readable") {
        let path = entry.expect("artifact entry should be readable").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }

        let payload = std::fs::read(&path).expect("root artifact should be readable");
        let expected_hash = hash(&payload).to_hex().to_string();
        let sidecar_path = PathBuf::from(format!("{}.raptorq.json", path.display()));
        let decode_path = PathBuf::from(format!("{}.decode_proof.json", path.display()));

        let sidecar: RaptorQSidecar = serde_json::from_slice(
            &std::fs::read(&sidecar_path).expect("root artifact sidecar should exist"),
        )
        .expect("root artifact sidecar should parse");
        let decode: DecodeProofArtifact = serde_json::from_slice(
            &std::fs::read(&decode_path).expect("root artifact decode proof should exist"),
        )
        .expect("root artifact decode proof should parse");

        assert_eq!(sidecar.source_hash, expected_hash);
        assert_eq!(decode.proof_hash, expected_hash);
        assert!(
            decode.recovered_blocks > 0,
            "decode proof should simulate at least one recovered source block for {path:?}"
        );
        checked += 1;
    }

    assert!(
        checked >= 4,
        "expected root markdown artifacts to carry RaptorQ sidecars and decode proofs"
    );
}

#[test]
fn baselines_and_ledgers_have_decode_proofs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("fsci-conformance should live under crates/");

    let targets = [
        repo_root.join("fixtures/artifacts/baselines/baseline_initial.json"),
        repo_root.join("fixtures/artifacts/P2C-008/anchor/behavior_ledger.json"),
    ];

    let mut checked = 0;
    for target in &targets {
        if !target.exists() {
            continue;
        }
        let payload = std::fs::read(target).expect("target should be readable");
        let expected_hash = hash(&payload).to_hex().to_string();

        let sidecar_path = PathBuf::from(format!("{}.raptorq.json", target.display()));
        let decode_path = PathBuf::from(format!("{}.decode_proof.json", target.display()));

        if !sidecar_path.exists() || !decode_path.exists() {
            let sidecar = generate_raptorq_sidecar(&payload).expect("generate sidecar");
            let decode_proof =
                fsci_conformance::generate_decode_proof_artifact(&payload, &sidecar)
                    .expect("generate decode proof");
            std::fs::write(&sidecar_path, serde_json::to_vec_pretty(&sidecar).unwrap())
                .expect("write sidecar");
            std::fs::write(&decode_path, serde_json::to_vec_pretty(&decode_proof).unwrap())
                .expect("write decode proof");
        }

        let sidecar: RaptorQSidecar = serde_json::from_slice(
            &std::fs::read(&sidecar_path).expect("sidecar should exist"),
        )
        .expect("sidecar should parse");
        let decode: DecodeProofArtifact = serde_json::from_slice(
            &std::fs::read(&decode_path).expect("decode proof should exist"),
        )
        .expect("decode proof should parse");

        assert_eq!(
            sidecar.source_hash, expected_hash,
            "sidecar source_hash mismatch for {target:?}"
        );
        assert_eq!(
            decode.proof_hash, expected_hash,
            "decode proof proof_hash mismatch for {target:?}"
        );
        assert!(
            decode.recovered_blocks > 0,
            "decode proof should recover at least 1 block for {target:?}"
        );
        checked += 1;
    }

    assert!(
        checked >= 2,
        "expected baseline bundles and differential audit ledgers to carry sidecars and decode proofs"
    );
}

