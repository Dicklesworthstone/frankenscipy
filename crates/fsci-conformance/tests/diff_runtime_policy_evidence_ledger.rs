#![forbid(unsafe_code)]
//! Cover fsci_runtime::PolicyEvidenceLedger public API from outside.
//!
//! Resolves [frankenscipy-e3qv2]. Exercises capacity clamping, FIFO
//! eviction, latest()/iter()/alien_artifact_decisions()/
//! latest_alien_artifact_decision()/to_alien_artifact_jsonl() plus the
//! DecisionEvidenceEntry helpers confidence() and
//! calibration_fallback_trigger(). Runs as a single test function
//! so each invariant is observed against its own constructed ledger.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{DecisionSignals, PolicyController, RuntimeMode};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ledger diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[test]
fn diff_runtime_policy_evidence_ledger() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === 1. capacity clamps to min 1 ===
    {
        let mut ctrl_zero = PolicyController::new(RuntimeMode::Strict, 0);
        let cap = ctrl_zero.ledger().capacity();
        check(
            "capacity_zero_clamps_to_one",
            cap == 1,
            format!("got {cap}"),
        );
        // record one entry — should fit
        let _ = ctrl_zero.decide(DecisionSignals::new(8.0, 0.0, 0.1));
        check(
            "after_one_decide_len_1",
            ctrl_zero.ledger().len() == 1,
            format!("len={}", ctrl_zero.ledger().len()),
        );
        // record a second — first should evict
        let _ = ctrl_zero.decide(DecisionSignals::new(7.0, 0.5, 0.2));
        check(
            "after_two_decides_len_still_1",
            ctrl_zero.ledger().len() == 1,
            format!("len={}", ctrl_zero.ledger().len()),
        );
    }

    // === 2. Empty ledger semantics ===
    {
        let ctrl = PolicyController::new(RuntimeMode::Strict, 4);
        let led = ctrl.ledger();
        check(
            "empty_is_empty",
            led.is_empty(),
            format!("len={}", led.len()),
        );
        check(
            "empty_latest_none",
            led.latest().is_none(),
            "expected None".into(),
        );
        check(
            "empty_latest_alien_none",
            led.latest_alien_artifact_decision().is_none(),
            "expected None".into(),
        );
        check(
            "empty_decisions_empty",
            led.alien_artifact_decisions().is_empty(),
            String::new(),
        );
        check(
            "empty_jsonl_empty_string",
            led.to_alien_artifact_jsonl().is_empty(),
            led.to_alien_artifact_jsonl(),
        );
    }

    // === 3. Capacity 3 FIFO eviction order ===
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 3);
        // Distinguishable signals so we can identify which entries survived.
        for (idx, x) in [1.0_f64, 2.0, 3.0, 4.0, 5.0].into_iter().enumerate() {
            let sigs = DecisionSignals::new(x, 0.0, 0.1);
            let _ = ctrl.decide(sigs);
            let len = ctrl.ledger().len().min(idx + 1);
            assert!(len <= 3, "len should be ≤ capacity");
        }
        let led = ctrl.ledger();
        check(
            "fifo_len_clamped_to_3",
            led.len() == 3,
            format!("len={}", led.len()),
        );

        // After 5 records with capacity 3: entries 3, 4, 5 should remain.
        let xs: Vec<f64> = led
            .iter()
            .map(|e| e.signals.condition_number_log10)
            .collect();
        check(
            "fifo_kept_last_three",
            xs == vec![3.0, 4.0, 5.0],
            format!("xs={xs:?}"),
        );
        check(
            "latest_is_last_record",
            led.latest().map(|e| e.signals.condition_number_log10) == Some(5.0),
            String::new(),
        );
    }

    // === 4. alien_artifact_decisions length matches len() ===
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 8);
        for x in [0.5_f64, 1.0, 1.5, 2.0] {
            let _ = ctrl.decide(DecisionSignals::new(x, 0.0, 0.1));
        }
        let led = ctrl.ledger();
        let n = led.alien_artifact_decisions().len();
        check(
            "alien_decisions_match_len",
            n == led.len() && n == 4,
            format!("n={n} len={}", led.len()),
        );

        // latest_alien matches the latest entry's decision
        let lat = led.latest_alien_artifact_decision().expect("latest");
        let same = led
            .latest()
            .map(|e| e.alien_artifact_decision())
            .expect("latest entry");
        check("latest_alien_matches_entry", lat == same, String::new());
    }

    // === 5. to_alien_artifact_jsonl yields N lines for N entries ===
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 5);
        for x in [1.0_f64, 2.0, 3.0] {
            let _ = ctrl.decide(DecisionSignals::new(x, 0.0, 0.1));
        }
        let jsonl = ctrl.ledger().to_alien_artifact_jsonl();
        let line_count = jsonl.lines().filter(|l| !l.is_empty()).count();
        check(
            "jsonl_three_lines",
            line_count == 3,
            format!("lines={line_count} text={jsonl:?}"),
        );

        // Each line should be parseable JSON
        let all_parse_ok = jsonl
            .lines()
            .filter(|l| !l.is_empty())
            .all(|l| serde_json::from_str::<serde_json::Value>(l).is_ok());
        check("jsonl_each_line_valid_json", all_parse_ok, String::new());
    }

    // === 6. DecisionEvidenceEntry::confidence == posterior[top_state.index()] ===
    // === 7. calibration_fallback_trigger: true iff non-finite signals OR ===
    //        (FailClosed + IncompatibleMetadata) ===
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 4);
        // Strict mode with high condition_number/violations should
        // route to FailClosed + IncompatibleMetadata.
        let _ = ctrl.decide(DecisionSignals::new(1.0e12, 100.0, 0.99));
        let entry = ctrl.ledger().latest().expect("latest");
        let posterior_at_top = entry.posterior[entry.top_state.index()];
        check(
            "confidence_eq_posterior_at_top_state",
            (entry.confidence() - posterior_at_top).abs() < 1.0e-15,
            format!("conf={} post={}", entry.confidence(), posterior_at_top),
        );

        // Non-finite signals: build via NaN — but the ledger only records what
        // decide() generates. Use a strict mode probe with high evidence —
        // it should trigger FailClosed + IncompatibleMetadata which makes
        // calibration_fallback_trigger() true.
        let trig = entry.calibration_fallback_trigger();
        check(
            "calibration_trigger_true_on_fail_closed_incompatible",
            trig,
            format!(
                "trig={trig} action={:?} top={:?}",
                entry.action, entry.top_state
            ),
        );
    }

    // === 8. Capacity larger than entries returns len < capacity ===
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 100);
        for x in [1.0_f64, 2.0] {
            let _ = ctrl.decide(DecisionSignals::new(x, 0.0, 0.1));
        }
        let led = ctrl.ledger();
        check(
            "len_under_capacity",
            led.len() == 2 && led.capacity() == 100,
            format!("len={} cap={}", led.len(), led.capacity()),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_runtime_policy_evidence_ledger".into(),
        category: "fsci_runtime::PolicyEvidenceLedger public-API coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("ledger mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "PolicyEvidenceLedger coverage failed: {} cases",
        diffs.len(),
    );
}
