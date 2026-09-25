#![forbid(unsafe_code)]

//! Bounded FIFO evidence ledger for policy decision audit trail.

use blake3::hash;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::mode::RuntimeMode;
use crate::policy::{PolicyAction, RiskState, decision_loss_matrix};
use crate::signals::DecisionSignals;

/// Spec §6 decision-theory record with every model input and output surfaced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlienArtifactDecision {
    pub state_space: [RiskState; 3],
    pub state: RiskState,
    pub evidence: DecisionSignals,
    pub logits: [f64; 3],
    pub loss_matrix: [[f64; 3]; 3],
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub action: PolicyAction,
    pub confidence: f64,
    pub calibration_fallback_trigger: bool,
    pub reason: String,
}

/// Complete record of a single policy decision for audit/forensic analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionEvidenceEntry {
    pub mode: RuntimeMode,
    pub signals: DecisionSignals,
    pub logits: [f64; 3],
    pub posterior: [f64; 3],
    pub expected_losses: [f64; 3],
    pub action: PolicyAction,
    pub top_state: RiskState,
    pub reason: String,
}

impl DecisionEvidenceEntry {
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.posterior[self.top_state.index()]
    }

    #[must_use]
    pub fn calibration_fallback_trigger(&self) -> bool {
        !self.signals.is_finite()
            || (self.action == PolicyAction::FailClosed
                && self.top_state == RiskState::IncompatibleMetadata)
    }

    #[must_use]
    pub fn alien_artifact_decision(&self) -> AlienArtifactDecision {
        AlienArtifactDecision {
            state_space: RiskState::ALL,
            state: self.top_state,
            evidence: self.signals,
            logits: self.logits,
            loss_matrix: decision_loss_matrix(self.mode),
            posterior: self.posterior,
            expected_losses: self.expected_losses,
            action: self.action,
            confidence: self.confidence(),
            calibration_fallback_trigger: self.calibration_fallback_trigger(),
            reason: self.reason.clone(),
        }
    }
}

/// Bounded FIFO evidence buffer recording all policy decisions.
///
/// Capacity is enforced via `capacity.max(1)` — minimum 1 entry.
/// When full, the oldest entry (front of `VecDeque`) is evicted before
/// a new entry is appended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvidenceLedger {
    capacity: usize,
    entries: VecDeque<DecisionEvidenceEntry>,
}

impl PolicyEvidenceLedger {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
        }
    }

    /// Append an entry, evicting the oldest if at capacity.
    pub fn record(&mut self, entry: DecisionEvidenceEntry) {
        if self.entries.len() == self.capacity {
            let _ = self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most recently recorded entry.
    #[must_use]
    pub fn latest(&self) -> Option<&DecisionEvidenceEntry> {
        self.entries.back()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &DecisionEvidenceEntry> {
        self.entries.iter()
    }

    #[must_use]
    pub fn alien_artifact_decisions(&self) -> Vec<AlienArtifactDecision> {
        self.entries
            .iter()
            .map(DecisionEvidenceEntry::alien_artifact_decision)
            .collect()
    }

    #[must_use]
    pub fn latest_alien_artifact_decision(&self) -> Option<AlienArtifactDecision> {
        self.latest()
            .map(DecisionEvidenceEntry::alien_artifact_decision)
    }

    /// Serialize Spec §6 decision records as JSONL for audit artifacts.
    #[must_use]
    pub fn to_alien_artifact_jsonl(&self) -> String {
        let mut output = Vec::with_capacity(self.entries.len().saturating_mul(512));
        for entry in &self.entries {
            let entry_start = output.len();
            if entry_start != 0 {
                output.push(b'\n');
            }
            if serde_json::to_writer(&mut output, &entry.alien_artifact_decision()).is_err() {
                output.truncate(entry_start);
            }
        }
        String::from_utf8(output).expect("serde_json always emits UTF-8")
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Audit actions recorded by the runtime for forensic analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuditAction {
    ModeDecision {
        mode: RuntimeMode,
    },
    BoundedRecovery {
        recovery_action: String,
    },
    FailClosed {
        reason: String,
    },
    AlienArtifactDecision {
        decision: Box<AlienArtifactDecision>,
    },
    // br-egba-1: `PolicyOverride { override_action: String }` was
    // defined here but never constructed by any crate in the workspace
    // (grep confirms zero call sites outside the enum definition).
    // Removed as dead code. If policy-override semantics are added
    // back, re-introduce the variant alongside at least one emission
    // site so it remains non-dead.
}

/// Single audit event entry with input fingerprint and outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub input_fingerprint: String,
    pub action: AuditAction,
    pub outcome: String,
}

impl AuditEvent {
    #[must_use]
    pub fn new(
        timestamp_ms: u64,
        input_fingerprint: impl Into<String>,
        action: AuditAction,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms,
            input_fingerprint: input_fingerprint.into(),
            action,
            outcome: outcome.into(),
        }
    }
}

/// Append-only ledger for audit events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditLedger {
    entries: Vec<AuditEvent>,
}

impl AuditLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record an audit event (append-only).
    pub fn record(&mut self, event: AuditEvent) {
        self.entries.push(event);
    }

    /// Record a fully surfaced Spec §6 decision-theory event.
    pub fn record_alien_artifact_decision(
        &mut self,
        timestamp_ms: u64,
        input_fingerprint: impl Into<String>,
        decision: AlienArtifactDecision,
        outcome: impl Into<String>,
    ) {
        self.record(AuditEvent::new(
            timestamp_ms,
            input_fingerprint,
            AuditAction::AlienArtifactDecision {
                decision: Box::new(decision),
            },
            outcome,
        ));
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[AuditEvent] {
        &self.entries
    }

    /// Serialize ledger as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize ledger from JSON.
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(payload)
    }

    /// Compute a blake3 hex fingerprint for raw input bytes.
    #[must_use]
    pub fn fingerprint_bytes(bytes: &[u8]) -> String {
        hash(bytes).to_hex().to_string()
    }

    /// Construct a shared, thread-safe ledger handle.
    #[must_use]
    pub fn shared() -> SharedAuditLedger {
        Arc::new(Mutex::new(Self::new()))
    }
}

impl Default for AuditLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// The canonical `AuditEvent::input_fingerprint`: BLAKE3 over a self-delimiting encoding of
/// the routine's name and EVERY input, shape and option, returned as `"blake3:<64 hex>"`. Two
/// calls share a fingerprint only if they feed the same records, value for value and bit for
/// bit (`-0.0` and `0.0` differ, as do NaN payloads), in the same order — so a ledger can be
/// filtered by the fingerprint of one request (frankenscipy-3cu8u.1).
///
/// Encoding, one record per call in call order; every integer is a little-endian `u64`:
///
/// | call | bytes |
/// |------|-------|
/// | `routine(name)` | `b'R'`, byte length, UTF-8 |
/// | `shape(dims)` | `b'S'`, count, each dimension |
/// | `f64s(values)` | `b'F'`, count, each `f64::to_bits` |
/// | `complex(values)` | `b'C'`, count, each pair's real then imaginary `to_bits` |
/// | `rows(rows)` | `b'M'`, row count, then per row: its length and each `to_bits` |
/// | `u64(v)` / `usize(v)` | `b'U'`, `v` |
/// | `i64(v)` | `b'I'`, `v` as two's-complement `u64` |
/// | `f64(v)` | `b'D'`, `v.to_bits()` |
/// | `bool(v)` | `b'B'`, one byte, 0 or 1 |
/// | `str(s)` | `b'T'`, byte length, UTF-8 |
/// | `bytes(b)` | `b'Y'`, length, the bytes |
///
/// Any caller can reproduce a fingerprint by feeding the same bytes to BLAKE3; the test
/// `fingerprinter_matches_its_documented_encoding` rebuilds it that way.
#[derive(Debug, Clone)]
pub struct Fingerprinter {
    hasher: blake3::Hasher,
}

impl Fingerprinter {
    /// A fingerprint for `routine`, which is its first record.
    #[must_use]
    pub fn new(routine: &str) -> Self {
        let mut fingerprinter = Self {
            hasher: blake3::Hasher::new(),
        };
        fingerprinter.tagged_bytes(b'R', routine.as_bytes());
        fingerprinter
    }

    fn word(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    /// `values` as consecutive little-endian words: the same bytes as one [`Self::word`] each,
    /// packed so BLAKE3 sees one `update` per 16 KiB instead of one per word (an update spanning
    /// many 1 KiB BLAKE3 chunks is what lets it hash them across SIMD lanes). Measured on hz2
    /// for a 1024×1024 matrix (frankenscipy-3cu8u.1): word-at-a-time 17–21 ms, 11–16% of
    /// `solve_with_audit`; 4 KiB updates 7.7–8.2 ms.
    fn words(&mut self, values: impl IntoIterator<Item = u64>) {
        let mut chunk = [0_u8; 16 * 1024];
        let mut filled = 0;
        for value in values {
            chunk[filled..filled + 8].copy_from_slice(&value.to_le_bytes());
            filled += 8;
            if filled == chunk.len() {
                self.hasher.update(&chunk);
                filled = 0;
            }
        }
        self.hasher.update(&chunk[..filled]);
    }

    fn tagged_bytes(&mut self, tag: u8, bytes: &[u8]) {
        self.hasher.update(&[tag]);
        self.word(bytes.len() as u64);
        self.hasher.update(bytes);
    }

    pub fn shape(&mut self, dims: &[usize]) -> &mut Self {
        self.hasher.update(b"S");
        self.word(dims.len() as u64);
        self.words(dims.iter().map(|&dim| dim as u64));
        self
    }

    pub fn f64s(&mut self, values: &[f64]) -> &mut Self {
        self.hasher.update(b"F");
        self.word(values.len() as u64);
        self.words(values.iter().map(|value| value.to_bits()));
        self
    }

    pub fn complex(&mut self, values: &[(f64, f64)]) -> &mut Self {
        self.hasher.update(b"C");
        self.word(values.len() as u64);
        self.words(
            values
                .iter()
                .flat_map(|&(re, im)| [re.to_bits(), im.to_bits()]),
        );
        self
    }

    /// A matrix as rows; each row's length is part of the record, so ragged inputs differ.
    pub fn rows(&mut self, rows: &[Vec<f64>]) -> &mut Self {
        self.hasher.update(b"M");
        self.word(rows.len() as u64);
        self.words(rows.iter().flat_map(|row| {
            std::iter::once(row.len() as u64).chain(row.iter().map(|value| value.to_bits()))
        }));
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.hasher.update(b"U");
        self.word(value);
        self
    }

    pub fn usize(&mut self, value: usize) -> &mut Self {
        self.u64(value as u64)
    }

    pub fn i64(&mut self, value: i64) -> &mut Self {
        self.hasher.update(b"I");
        self.word(value as u64);
        self
    }

    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.hasher.update(b"D");
        self.word(value.to_bits());
        self
    }

    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.hasher.update(&[b'B', u8::from(value)]);
        self
    }

    pub fn str(&mut self, value: &str) -> &mut Self {
        self.tagged_bytes(b'T', value.as_bytes());
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.tagged_bytes(b'Y', value);
        self
    }

    /// `"blake3:<64 hex>"`.
    #[must_use]
    pub fn finish(&self) -> String {
        format!("blake3:{}", self.hasher.finalize().to_hex())
    }
}

/// Thread-safe audit ledger handle shared by synchronous crate APIs.
pub type SharedAuditLedger = Arc<Mutex<AuditLedger>>;

/// Canonical synchronous audit ledger handle.
pub type SyncSharedAuditLedger = SharedAuditLedger;

#[cfg(test)]
mod tests {
    use super::*;

    /// The documented encoding, rebuilt by hand: what an external caller would do.
    #[test]
    fn fingerprinter_matches_its_documented_encoding() {
        let rows = vec![vec![1.0, -0.0], vec![f64::NAN]];
        let mut fingerprinter = Fingerprinter::new("solve");
        fingerprinter
            .rows(&rows)
            .f64s(&[2.5])
            .shape(&[2, 2])
            .complex(&[(1.0, -1.0)])
            .usize(7)
            .i64(-3)
            .f64(0.1)
            .bool(true)
            .str("pos")
            .bytes(b"\x00\x01");

        let word = |bytes: &mut Vec<u8>, value: u64| bytes.extend_from_slice(&value.to_le_bytes());
        let mut expected = Vec::new();
        expected.push(b'R');
        word(&mut expected, 5);
        expected.extend_from_slice(b"solve");
        expected.push(b'M');
        word(&mut expected, 2);
        for row in &rows {
            word(&mut expected, row.len() as u64);
            for value in row {
                word(&mut expected, value.to_bits());
            }
        }
        expected.push(b'F');
        word(&mut expected, 1);
        word(&mut expected, 2.5_f64.to_bits());
        expected.push(b'S');
        word(&mut expected, 2);
        word(&mut expected, 2);
        word(&mut expected, 2);
        expected.push(b'C');
        word(&mut expected, 1);
        word(&mut expected, 1.0_f64.to_bits());
        word(&mut expected, (-1.0_f64).to_bits());
        expected.push(b'U');
        word(&mut expected, 7);
        expected.push(b'I');
        word(&mut expected, (-3_i64) as u64);
        expected.push(b'D');
        word(&mut expected, 0.1_f64.to_bits());
        expected.extend_from_slice(&[b'B', 1]);
        expected.push(b'T');
        word(&mut expected, 3);
        expected.extend_from_slice(b"pos");
        expected.push(b'Y');
        word(&mut expected, 2);
        expected.extend_from_slice(b"\x00\x01");

        assert_eq!(
            fingerprinter.finish(),
            format!("blake3:{}", hash(&expected).to_hex())
        );
    }

    /// Records longer than one 16 KiB (2048-word) chunk, ragged rows straddling chunk
    /// boundaries, and a record ending exactly on a boundary hash to the same bytes as the
    /// documented word-at-a-time encoding.
    #[test]
    fn fingerprinter_chunking_is_invisible_in_the_encoding() {
        let long: Vec<f64> = (0..5000).map(|i| f64::from(i) * 0.25 - 7.0).collect();
        // 2048 words: exactly one full chunk, then an empty tail.
        let exact: Vec<f64> = (0..2048).map(f64::from).collect();
        // 9 ragged rows, 3528 words with the lengths: crosses one boundary mid-row.
        let rows: Vec<Vec<f64>> = (0..9)
            .map(|r| (0..(r * 97 + 3)).map(|c| f64::from(r * 1000 + c)).collect())
            .collect();
        let pairs: Vec<(f64, f64)> = (0..1500).map(|i| (f64::from(i), -f64::from(i))).collect();
        let dims: Vec<usize> = (0..600).collect();
        let mut fingerprinter = Fingerprinter::new("chunks");
        fingerprinter
            .f64s(&long)
            .f64s(&exact)
            .rows(&rows)
            .complex(&pairs)
            .shape(&dims);

        let word = |bytes: &mut Vec<u8>, value: u64| bytes.extend_from_slice(&value.to_le_bytes());
        let mut expected = vec![b'R'];
        word(&mut expected, 6);
        expected.extend_from_slice(b"chunks");
        for values in [&long, &exact] {
            expected.push(b'F');
            word(&mut expected, values.len() as u64);
            for value in values.iter() {
                word(&mut expected, value.to_bits());
            }
        }
        expected.push(b'M');
        word(&mut expected, rows.len() as u64);
        for row in &rows {
            word(&mut expected, row.len() as u64);
            for value in row {
                word(&mut expected, value.to_bits());
            }
        }
        expected.push(b'C');
        word(&mut expected, pairs.len() as u64);
        for &(re, im) in &pairs {
            word(&mut expected, re.to_bits());
            word(&mut expected, im.to_bits());
        }
        expected.push(b'S');
        word(&mut expected, dims.len() as u64);
        for &dim in &dims {
            word(&mut expected, dim as u64);
        }
        assert_eq!(
            fingerprinter.finish(),
            format!("blake3:{}", hash(&expected).to_hex())
        );
    }

    /// The old fingerprints hashed a prefix (linalg: the first KiB of the matrix) or only a
    /// length; these inputs collided under them and must not now.
    #[test]
    fn fingerprinter_distinguishes_inputs_the_old_digests_merged() {
        let fingerprint = |build: &dyn Fn(&mut Fingerprinter)| {
            let mut fingerprinter = Fingerprinter::new("routine");
            build(&mut fingerprinter);
            fingerprinter.finish()
        };
        // Identical in the first KiB (128 values), different in the last element.
        let big: Vec<Vec<f64>> = (0..20).map(|i| vec![f64::from(i); 20]).collect();
        let mut tail = big.clone();
        tail[19][19] += 1.0;
        assert_ne!(
            fingerprint(&|f| {
                f.rows(&big);
            }),
            fingerprint(&|f| {
                f.rows(&tail);
            })
        );
        // Same length, different values (the old length-only digests).
        assert_ne!(
            fingerprint(&|f| {
                f.f64s(&[1.0, 2.0]);
            }),
            fingerprint(&|f| {
                f.f64s(&[1.0, 3.0]);
            })
        );
        // Signed zero and NaN payloads are bits, not values.
        assert_ne!(
            fingerprint(&|f| {
                f.f64s(&[0.0]);
            }),
            fingerprint(&|f| {
                f.f64s(&[-0.0]);
            })
        );
        assert_ne!(
            fingerprint(&|f| {
                f.f64s(&[f64::NAN]);
            }),
            fingerprint(&|f| {
                f.f64s(&[f64::from_bits(f64::NAN.to_bits() ^ 1)]);
            })
        );
        // Ragged rows with the same flattened values, and a transposed shape.
        assert_ne!(
            fingerprint(&|f| {
                f.rows(&[vec![1.0, 2.0], vec![3.0]]);
            }),
            fingerprint(&|f| {
                f.rows(&[vec![1.0], vec![2.0, 3.0]]);
            })
        );
        assert_ne!(
            fingerprint(&|f| {
                f.shape(&[2, 3]);
            }),
            fingerprint(&|f| {
                f.shape(&[3, 2]);
            })
        );
        // A record boundary cannot be moved: "ab" + "c" is not "a" + "bc".
        assert_ne!(
            fingerprint(&|f| {
                f.str("ab").str("c");
            }),
            fingerprint(&|f| {
                f.str("a").str("bc");
            })
        );
        // Routines are separated, and the same records give the same fingerprint.
        let mut solve = Fingerprinter::new("solve");
        let mut inv = Fingerprinter::new("inv");
        solve.f64s(&[1.0]);
        inv.f64s(&[1.0]);
        assert_ne!(solve.finish(), inv.finish());
        assert_eq!(
            fingerprint(&|f| {
                f.rows(&big).bool(false);
            }),
            fingerprint(&|f| {
                f.rows(&big).bool(false);
            })
        );
        assert!(solve.finish().starts_with("blake3:") && solve.finish().len() == 7 + 64);
    }

    #[test]
    fn audit_ledger_roundtrip_preserves_entries() {
        let mut ledger = AuditLedger::new();
        ledger.record(AuditEvent::new(
            1,
            AuditLedger::fingerprint_bytes(b"mode"),
            AuditAction::ModeDecision {
                mode: RuntimeMode::Strict,
            },
            "accepted",
        ));
        ledger.record(AuditEvent::new(
            2,
            AuditLedger::fingerprint_bytes(b"recover"),
            AuditAction::BoundedRecovery {
                recovery_action: "trim_nan".to_string(),
            },
            "recovered",
        ));
        ledger.record(AuditEvent::new(
            3,
            AuditLedger::fingerprint_bytes(b"reject"),
            AuditAction::FailClosed {
                reason: "non_finite_input".to_string(),
            },
            "rejected",
        ));

        let json = ledger.to_json().expect("serialize failed");
        let decoded = AuditLedger::from_json(&json).expect("deserialize failed");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded, ledger);
    }

    #[test]
    fn audit_fail_closed_includes_reason() {
        let event = AuditEvent::new(
            7,
            AuditLedger::fingerprint_bytes(b"fail"),
            AuditAction::FailClosed {
                reason: "invalid_metadata".to_string(),
            },
            "rejected",
        );
        match &event.action {
            AuditAction::FailClosed { reason } => {
                assert_eq!(reason, "invalid_metadata");
            }
            other => {
                unreachable!("expected fail closed action, got {other:?}");
            }
        }
    }

    #[test]
    fn audit_bounded_recovery_includes_action_and_outcome() {
        let event = AuditEvent::new(
            9,
            AuditLedger::fingerprint_bytes(b"recover"),
            AuditAction::BoundedRecovery {
                recovery_action: "drop_outliers".to_string(),
            },
            "recovered",
        );
        match &event.action {
            AuditAction::BoundedRecovery { recovery_action } => {
                assert_eq!(recovery_action, "drop_outliers");
            }
            other => {
                unreachable!("expected bounded recovery action, got {other:?}");
            }
        }
        assert_eq!(event.outcome, "recovered");
    }

    #[test]
    fn alien_artifact_decision_surfaces_spec_fields() {
        let entry = DecisionEvidenceEntry {
            mode: RuntimeMode::Strict,
            signals: DecisionSignals::new(8.0, 0.4, 0.2),
            logits: [1.0, 0.5, -1.0],
            posterior: [0.6, 0.3, 0.1],
            expected_losses: [39.5, 14.0, 31.6],
            action: PolicyAction::FullValidate,
            top_state: RiskState::Compatible,
            reason: String::from("test"),
        };

        let decision = entry.alien_artifact_decision();

        assert_eq!(decision.state_space, RiskState::ALL);
        assert_eq!(decision.state, RiskState::Compatible);
        assert_eq!(decision.evidence, entry.signals);
        assert_eq!(
            decision.loss_matrix,
            decision_loss_matrix(RuntimeMode::Strict)
        );
        assert_eq!(decision.posterior, entry.posterior);
        assert_eq!(decision.action, PolicyAction::FullValidate);
        assert_eq!(decision.confidence, 0.6);
        assert!(!decision.calibration_fallback_trigger);
    }

    #[test]
    fn audit_ledger_records_structured_alien_artifact_decision() {
        let entry = DecisionEvidenceEntry {
            mode: RuntimeMode::Hardened,
            signals: DecisionSignals::new(14.0, 0.2, 0.1),
            logits: [-1.0, 1.0, 0.0],
            posterior: [0.1, 0.7, 0.2],
            expected_losses: [71.0, 15.2, 26.7],
            action: PolicyAction::FullValidate,
            top_state: RiskState::IllConditioned,
            reason: String::from("structured audit decision"),
        };
        let decision = entry.alien_artifact_decision();
        let mut ledger = AuditLedger::new();

        ledger.record_alien_artifact_decision(
            11,
            AuditLedger::fingerprint_bytes(b"structured"),
            decision,
            "validated",
        );

        let json = ledger.to_json().expect("serialize structured audit");
        let parsed = serde_json::from_str::<serde_json::Value>(&json).expect("parse ledger JSON");
        let action = &parsed["entries"][0]["action"];
        assert_eq!(action["kind"], "alien_artifact_decision");
        assert_eq!(action["decision"]["state"], "IllConditioned");
        assert!(action["decision"].get("evidence").is_some());
        assert!(action["decision"].get("loss_matrix").is_some());
        assert_eq!(action["decision"]["confidence"], 0.7);
    }

    #[test]
    fn policy_evidence_ledger_emits_alien_artifact_jsonl() {
        let mut ledger = PolicyEvidenceLedger::new(2);
        ledger.record(DecisionEvidenceEntry {
            mode: RuntimeMode::Hardened,
            signals: DecisionSignals::new(f64::NAN, 0.0, 0.0),
            logits: [-1.0e30, -1.0e30, 0.0],
            posterior: [0.0, 0.0, 1.0],
            expected_losses: [180.0, 60.0, 1.0],
            action: PolicyAction::FailClosed,
            top_state: RiskState::IncompatibleMetadata,
            reason: String::from("non_finite_signals=true"),
        });
        ledger.record(DecisionEvidenceEntry {
            mode: RuntimeMode::Strict,
            signals: DecisionSignals::new(8.0, 0.25, 0.1),
            logits: [0.5, 1.0, -1.0],
            posterior: [0.25, 0.7, 0.05],
            expected_losses: [35.0, 12.0, 40.0],
            action: PolicyAction::FullValidate,
            top_state: RiskState::IllConditioned,
            reason: String::from("escaped=\"line\"\nnext"),
        });

        let latest = ledger
            .latest_alien_artifact_decision()
            .expect("latest decision");
        assert_eq!(latest.confidence, 0.7);
        assert!(!latest.calibration_fallback_trigger);

        let former = ledger
            .iter()
            .filter_map(|entry| serde_json::to_string(&entry.alien_artifact_decision()).ok())
            .collect::<Vec<_>>()
            .join("\n");
        let jsonl = ledger.to_alien_artifact_jsonl();
        assert_eq!(jsonl, former);
        let parsed = serde_json::from_str::<serde_json::Value>(
            jsonl.lines().next().expect("missing JSONL decision"),
        )
        .expect("valid JSONL decision");
        assert!(parsed.get("state_space").is_some());
        assert!(parsed.get("evidence").is_some());
        assert!(parsed.get("loss_matrix").is_some());
        assert_eq!(parsed["confidence"], 1.0);
        assert_eq!(parsed["calibration_fallback_trigger"], true);
    }
}
