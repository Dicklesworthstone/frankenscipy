#![forbid(unsafe_code)]

//! Bounded FIFO evidence ledger for policy decision audit trail.

use blake3::hash;
use serde::{Deserialize, Serialize};
use std::cell::{Cell, OnceCell};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Display;
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
    /// A CASP portfolio's solver choice (frankenscipy-7tb8d.11): which portfolio, the runtime
    /// mode the call ran in, the action taken, the posterior over the portfolio's states, the
    /// expected loss of every action (in the portfolio's action order) and of the chosen one,
    /// whether it is a fallback from the action first selected, and the evidence that drove
    /// the posterior, by name.
    CaspDecision {
        portfolio: String,
        mode: RuntimeMode,
        action: String,
        posterior: Vec<f64>,
        expected_losses: Vec<f64>,
        chosen_expected_loss: f64,
        fallback: bool,
        evidence: BTreeMap<String, EvidenceValue>,
    },
    // br-egba-1: `PolicyOverride { override_action: String }` was
    // defined here but never constructed by any crate in the workspace
    // (grep confirms zero call sites outside the enum definition).
    // Removed as dead code. If policy-override semantics are added
    // back, re-introduce the variant alongside at least one emission
    // site so it remains non-dead.
}

/// One piece of the evidence behind a CASP decision: a measurement (an rcond estimate) or a
/// label (a structure class). JSON carries it as a plain number or string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EvidenceValue {
    Number(f64),
    Label(String),
}

impl From<f64> for EvidenceValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for EvidenceValue {
    fn from(value: &str) -> Self {
        Self::Label(value.to_string())
    }
}

impl From<String> for EvidenceValue {
    fn from(value: String) -> Self {
        Self::Label(value)
    }
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

/// Append-only ledger for audit events. Unbounded by default; with a capacity
/// ([`AuditLedger::with_capacity`]) it keeps the newest `capacity` events, evicting the oldest
/// first and counting what it evicted (frankenscipy-7tb8d.11). Both show in its JSON only when
/// set, so an unbounded ledger serializes as `{"entries": [...]}`.
#[derive(Debug, Clone)]
pub struct AuditLedger {
    /// The events are `buffer[start..]`. Evicted events are dropped from the front in
    /// batches, so recording stays amortized O(1) and `entries()` stays one slice.
    buffer: Vec<AuditEvent>,
    start: usize,
    capacity: Option<usize>,
    evicted: u64,
}

/// [`AuditLedger`]'s JSON form.
#[derive(Serialize, Deserialize)]
struct AuditLedgerJson<'a> {
    entries: std::borrow::Cow<'a, [AuditEvent]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capacity: Option<usize>,
    #[serde(default, skip_serializing_if = "is_zero")]
    evicted: u64,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` passes a reference
const fn is_zero(value: &u64) -> bool {
    *value == 0
}

impl Serialize for AuditLedger {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        AuditLedgerJson {
            entries: std::borrow::Cow::Borrowed(self.entries()),
            capacity: self.capacity,
            evicted: self.evicted,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AuditLedger {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let json = AuditLedgerJson::deserialize(deserializer)?;
        let mut ledger = Self {
            buffer: json.entries.into_owned(),
            start: 0,
            capacity: json.capacity,
            evicted: json.evicted,
        };
        if let Some(capacity) = ledger.capacity {
            let excess = ledger.buffer.len().saturating_sub(capacity);
            ledger.buffer.drain(..excess);
            ledger.evicted += excess as u64;
        }
        Ok(ledger)
    }
}

impl PartialEq for AuditLedger {
    fn eq(&self, other: &Self) -> bool {
        self.entries() == other.entries()
            && self.capacity == other.capacity
            && self.evicted == other.evicted
    }
}

impl AuditLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            start: 0,
            capacity: None,
            evicted: 0,
        }
    }

    /// A ledger that keeps the newest `capacity` events.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity),
            ..Self::new()
        }
    }

    /// Record an audit event, evicting the oldest one when a bounded ledger is full.
    pub fn record(&mut self, event: AuditEvent) {
        self.buffer.push(event);
        let Some(capacity) = self.capacity else {
            return;
        };
        if self.buffer.len() - self.start > capacity {
            self.start += 1;
            self.evicted += 1;
            // Drop the evicted prefix once it is as long as a full ledger.
            if self.start >= capacity.max(1) {
                self.buffer.drain(..self.start);
                self.start = 0;
            }
        }
    }

    /// The capacity of a bounded ledger; `None` when unbounded.
    #[must_use]
    pub const fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// How many events a bounded ledger has evicted.
    #[must_use]
    pub const fn evicted(&self) -> u64 {
        self.evicted
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
        self.buffer.len() - self.start
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The events held, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[AuditEvent] {
        &self.buffer[self.start..]
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

    /// A shared ledger that keeps the newest `capacity` events.
    #[must_use]
    pub fn shared_with_capacity(capacity: usize) -> SharedAuditLedger {
        Arc::new(Mutex::new(Self::with_capacity(capacity)))
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

/// Lock a shared ledger, recovering from a poisoned mutex so events still record after another
/// thread panicked while holding it.
fn lock_ledger(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// One audited public call: the ledger its events go to, the recipe for its fingerprint, and
/// whether it has failed closed (frankenscipy-3cu8u.2).
///
/// Every audited API keeps one contract through it: a call that returns `Err` records exactly
/// one [`AuditAction::FailClosed`], after any other event of the call, under the call's
/// fingerprint, in every runtime mode; a call that succeeds records none. A check that knows the
/// specific cause calls [`reject`](Self::reject) just before returning its error, and
/// [`finish`](Self::finish), at the call's exit, records one for any `Err` that no check
/// rejected, with a reason derived from the error. Only a call's first `reject` records, so no
/// error is recorded twice.
///
/// Reasons are machine-matchable codes (`non_finite_input`, `singular_matrix`), not prose; the
/// event's outcome carries the error's message.
///
/// The fingerprint recipe (a [`Fingerprinter`] digest of the routine and every input,
/// frankenscipy-3cu8u.1) runs at most once per call and only when an event is recorded, so a
/// call that records nothing does not hash its inputs.
pub struct AuditScope<'a> {
    ledger: &'a SyncSharedAuditLedger,
    fingerprint_of: &'a dyn Fn() -> String,
    fingerprint: OnceCell<String>,
    rejected: Cell<bool>,
}

impl<'a> AuditScope<'a> {
    #[must_use]
    pub fn new(ledger: &'a SyncSharedAuditLedger, fingerprint_of: &'a dyn Fn() -> String) -> Self {
        Self {
            ledger,
            fingerprint_of,
            fingerprint: OnceCell::new(),
            rejected: Cell::new(false),
        }
    }

    /// The call's fingerprint, computed on first use.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        self.fingerprint.get_or_init(self.fingerprint_of)
    }

    /// The ledger this call records to, for events built outside the scope.
    #[must_use]
    pub const fn ledger(&self) -> &'a SyncSharedAuditLedger {
        self.ledger
    }

    /// Record an event of this call.
    pub fn record(&self, action: AuditAction, outcome: &str) {
        let event = AuditEvent::new(
            crate::casp_now_unix_ms(),
            self.fingerprint(),
            action,
            outcome,
        );
        lock_ledger(self.ledger).record(event);
    }

    /// Record a bounded recovery of this call.
    pub fn recover(&self, recovery_action: &str, outcome: &str) {
        self.record(
            AuditAction::BoundedRecovery {
                recovery_action: recovery_action.to_string(),
            },
            outcome,
        );
    }

    /// Fail this call closed with `reason`; the caller then returns its error. Only the call's
    /// first rejection records.
    pub fn reject(&self, reason: &str, outcome: &str) {
        if self.rejected.replace(true) {
            return;
        }
        self.record(
            AuditAction::FailClosed {
                reason: reason.to_string(),
            },
            outcome,
        );
    }

    /// Whether this call has failed closed.
    #[must_use]
    pub fn has_rejected(&self) -> bool {
        self.rejected.get()
    }

    /// The call's exit: `result`, unchanged, after failing the call closed with
    /// `reason_of(error)` when it is an `Err` that no check rejected.
    pub fn finish<T, E: Display, R: AsRef<str>>(
        &self,
        result: Result<T, E>,
        reason_of: impl FnOnce(&E) -> R,
    ) -> Result<T, E> {
        match &result {
            Err(error) => {
                if !self.has_rejected() {
                    self.reject(reason_of(error).as_ref(), &format!("rejected: {error}"));
                }
            }
            Ok(_) => debug_assert!(
                !self.has_rejected(),
                "an audited call failed closed and then returned Ok"
            ),
        }
        result
    }
}

/// [`AuditScope::reject`] for a call that may not be audited.
pub fn audit_reject(audit: Option<&AuditScope<'_>>, reason: &str, outcome: &str) {
    if let Some(audit) = audit {
        audit.reject(reason, outcome);
    }
}

/// [`AuditScope::recover`] for a call that may not be audited.
pub fn audit_recover(audit: Option<&AuditScope<'_>>, recovery_action: &str, outcome: &str) {
    if let Some(audit) = audit {
        audit.recover(recovery_action, outcome);
    }
}

/// [`AuditScope::finish`] for a call that may not be audited.
pub fn audit_finish<T, E: Display, R: AsRef<str>>(
    audit: Option<&AuditScope<'_>>,
    result: Result<T, E>,
    reason_of: impl FnOnce(&E) -> R,
) -> Result<T, E> {
    match audit {
        Some(audit) => audit.finish(result, reason_of),
        None => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail_closed_reasons(ledger: &SyncSharedAuditLedger) -> Vec<String> {
        lock_ledger(ledger)
            .entries()
            .iter()
            .filter_map(|event| match &event.action {
                AuditAction::FailClosed { reason } => Some(reason.clone()),
                _ => None,
            })
            .collect()
    }

    /// frankenscipy-3cu8u.2: whichever way a call fails, the ledger gains one `FailClosed`, the
    /// call's last event; a call that succeeds gains none, and its fingerprint is never
    /// computed.
    #[test]
    fn audit_scope_fails_closed_exactly_once_per_error() {
        let hashed = Cell::new(0);
        let recipe = || {
            hashed.set(hashed.get() + 1);
            "blake3:call".to_string()
        };

        // A check rejects with its own reason; the exit must not record the error again.
        let ledger = AuditLedger::shared();
        let scope = AuditScope::new(&ledger, &recipe);
        scope.recover("clamp", "clamped");
        scope.reject("non_finite_input", "rejected");
        scope.reject("second_check", "rejected");
        let result: Result<(), &str> = scope.finish(Err("NaN"), |_| "from_error");
        assert!(result.is_err());
        assert_eq!(fail_closed_reasons(&ledger), ["non_finite_input"]);
        let entries = lock_ledger(&ledger).entries().to_vec();
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[1].action, AuditAction::FailClosed { .. }));
        assert!(entries.iter().all(|e| e.input_fingerprint == "blake3:call"));
        assert_eq!(hashed.get(), 1, "the recipe runs once per call");

        // No check knew the cause: the exit records one, from the error.
        let ledger = AuditLedger::shared();
        let scope = AuditScope::new(&ledger, &recipe);
        let result: Result<(), &str> = scope.finish(Err("singular"), |e| format!("{e}_matrix"));
        assert_eq!(result, Err("singular"));
        assert_eq!(fail_closed_reasons(&ledger), ["singular_matrix"]);
        assert_eq!(
            lock_ledger(&ledger).entries()[0].outcome,
            "rejected: singular"
        );

        // Success records nothing and hashes nothing; an unaudited call records nowhere.
        let ledger = AuditLedger::shared();
        hashed.set(0);
        let scope = AuditScope::new(&ledger, &recipe);
        assert_eq!(scope.finish(Ok::<_, &str>(3), |_| "unused"), Ok(3));
        assert!(lock_ledger(&ledger).is_empty());
        assert_eq!(hashed.get(), 0);
        assert_eq!(
            audit_finish(None, Err::<(), _>("x"), |_| "unused"),
            Err("x")
        );
        audit_reject(None, "unused", "unused");
        assert_eq!(hashed.get(), 0);
    }

    fn numbered_event(n: u64) -> AuditEvent {
        AuditEvent::new(
            n,
            format!("blake3:{n}"),
            AuditAction::FailClosed {
                reason: "non_finite_input".to_string(),
            },
            "rejected",
        )
    }

    /// frankenscipy-7tb8d.11: a bounded ledger keeps its newest `capacity` events, FIFO, and
    /// counts what it evicted, in memory and through JSON; an unbounded one never evicts and
    /// its JSON carries neither field.
    #[test]
    fn bounded_audit_ledger_evicts_fifo_and_counts_evictions() {
        let capacity = 5;
        let mut ledger = AuditLedger::with_capacity(capacity);
        for n in 0..(capacity as u64 + 10) {
            ledger.record(numbered_event(n));
        }
        assert_eq!(ledger.len(), capacity);
        assert_eq!(ledger.evicted(), 10);
        let kept: Vec<u64> = ledger.entries().iter().map(|e| e.timestamp_ms).collect();
        assert_eq!(kept, [10, 11, 12, 13, 14]);

        let json = ledger.to_json().expect("serialize");
        assert!(json.contains("\"capacity\":5"), "{json}");
        assert!(json.contains("\"evicted\":10"), "{json}");
        assert_eq!(AuditLedger::from_json(&json).expect("deserialize"), ledger);

        let mut unbounded = AuditLedger::new();
        for n in 0..(capacity as u64 + 10) {
            unbounded.record(numbered_event(n));
        }
        assert_eq!(unbounded.len(), capacity + 10);
        assert_eq!((unbounded.capacity(), unbounded.evicted()), (None, 0));
        let json = unbounded.to_json().expect("serialize");
        assert!(
            json.starts_with("{\"entries\":[") && !json.contains("\"evicted\""),
            "{json}"
        );

        // A bounded ledger read from JSON that holds more than its capacity evicts on load.
        let overfull = r#"{"entries":[
            {"timestamp_ms":1,"input_fingerprint":"a","action":{"kind":"fail_closed","reason":"r"},"outcome":"o"},
            {"timestamp_ms":2,"input_fingerprint":"b","action":{"kind":"fail_closed","reason":"r"},"outcome":"o"},
            {"timestamp_ms":3,"input_fingerprint":"c","action":{"kind":"fail_closed","reason":"r"},"outcome":"o"}
        ],"capacity":2,"evicted":4}"#;
        let loaded = AuditLedger::from_json(overfull).expect("deserialize");
        assert_eq!((loaded.len(), loaded.evicted()), (2, 5));
        assert_eq!(loaded.entries()[0].timestamp_ms, 2);
    }

    /// frankenscipy-7tb8d.11: the CASP decision event's JSON, byte for byte (a new golden, not
    /// a change to an existing one), and its round trip.
    #[test]
    fn casp_decision_event_json_is_stable_and_round_trips() {
        let event = AuditEvent::new(
            1_700_000_000_000,
            "blake3:0f",
            AuditAction::CaspDecision {
                portfolio: "solver".to_string(),
                mode: RuntimeMode::Hardened,
                action: "PivotedQR".to_string(),
                posterior: vec![0.125, 0.25, 0.5, 0.125],
                expected_losses: vec![40.0, 8.0],
                chosen_expected_loss: 8.0,
                fallback: true,
                evidence: BTreeMap::from([
                    ("structural_evidence".to_string(), "General".into()),
                    ("rcond_estimate".to_string(), 1e-9.into()),
                ]),
            },
            "solved",
        );
        let json = serde_json::to_string(&event).expect("serialize");
        assert_eq!(
            json,
            "{\"timestamp_ms\":1700000000000,\"input_fingerprint\":\"blake3:0f\",\"action\":\
             {\"kind\":\"casp_decision\",\"portfolio\":\"solver\",\"mode\":\"Hardened\",\
             \"action\":\"PivotedQR\",\"posterior\":[0.125,0.25,0.5,0.125],\
             \"expected_losses\":[40.0,8.0],\"chosen_expected_loss\":8.0,\"fallback\":true,\
             \"evidence\":{\"rcond_estimate\":1e-9,\"structural_evidence\":\"General\"}},\
             \"outcome\":\"solved\"}"
        );
        let decoded: AuditEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, event);
    }

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
