use criterion::{Criterion, criterion_group, criterion_main};
use fsci_runtime::{
    ConformalCalibrator, DecisionEvidenceEntry, DecisionSignals, MatrixConditionState,
    PolicyAction, PolicyController, PolicyEvidenceLedger, PortfolioEvidence, RiskState,
    RuntimeMode, SolverAction, SolverEvidenceEntry, SolverPortfolio,
};
use std::hint::black_box;

fn bench_policy_decide(c: &mut Criterion) {
    let signals = DecisionSignals::new(5.0, 0.3, 0.2);
    c.bench_function("policy_decide_strict", |b| {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        b.iter(|| ctrl.decide(signals));
    });
    c.bench_function("policy_decide_hardened", |b| {
        let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 64);
        b.iter(|| ctrl.decide(signals));
    });
}

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-12,
        MatrixConditionState::NearSingular => 1e-18,
    }
}

fn bench_solver_select_action(c: &mut Criterion) {
    let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    for state in &MatrixConditionState::ALL {
        let name = format!("solver_select_{state:?}");
        c.bench_function(&name, |b| {
            b.iter(|| portfolio.select_action(state_to_rcond(state), None));
        });
    }
}

fn bench_calibrator_observe(c: &mut Criterion) {
    c.bench_function("calibrator_observe_200", |b| {
        b.iter(|| {
            let mut cal = ConformalCalibrator::new(0.05, 200);
            for i in 0..200 {
                cal.observe(i as f64 * 1e-10);
            }
        });
    });
}

fn solver_evidence(sequence: usize) -> SolverEvidenceEntry {
    SolverEvidenceEntry {
        component: "runtime_bench",
        matrix_shape: (sequence, 32),
        rcond_estimate: 1e-6,
        chosen_action: SolverAction::PivotedQR,
        posterior: vec![0.0, 1.0, 0.0, 0.0],
        expected_losses: vec![5.0, 1.0, 10.0, 0.0, 0.0],
        chosen_expected_loss: 1.0,
        fallback_active: false,
        backward_error: None,
    }
}

/// The naive serializer the streaming one is measured against: each entry's JSON with the
/// portfolio's name and mode spliced in front, collected and joined.
fn serialize_jsonl_collect_join(entries: &[SolverEvidenceEntry]) -> String {
    entries
        .iter()
        .filter_map(|entry| serde_json::to_string(entry).ok())
        .map(|json| {
            format!(
                "{{\"portfolio\":\"solver\",\"mode\":\"Strict\",{}",
                &json[1..]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn bench_solver_evidence_jsonl(c: &mut Criterion) {
    const ENTRIES: usize = 1_024;
    let entries = (0..ENTRIES).map(solver_evidence).collect::<Vec<_>>();
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, ENTRIES);
    for entry in entries.iter().cloned() {
        portfolio.record_evidence(entry);
    }
    assert_eq!(
        portfolio.serialize_jsonl(),
        serialize_jsonl_collect_join(&entries)
    );

    let mut group = c.benchmark_group("solver_evidence_jsonl");
    group.bench_function("stream", |b| {
        b.iter(|| black_box(portfolio.serialize_jsonl()));
    });
    group.bench_function("collect_join_baseline", |b| {
        b.iter(|| black_box(serialize_jsonl_collect_join(black_box(&entries))));
    });
    group.finish();
}

fn policy_evidence(sequence: usize) -> DecisionEvidenceEntry {
    DecisionEvidenceEntry {
        mode: RuntimeMode::Strict,
        signals: DecisionSignals::new(8.0, 0.25, 0.1),
        logits: [0.5, 1.0, -1.0],
        posterior: [0.25, 0.7, 0.05],
        expected_losses: [35.0, 12.0, 40.0],
        action: PolicyAction::FullValidate,
        top_state: RiskState::IllConditioned,
        reason: format!("runtime_bench_sequence={sequence}"),
    }
}

fn policy_jsonl_collect_join(entries: &[DecisionEvidenceEntry]) -> String {
    entries
        .iter()
        .filter_map(|entry| serde_json::to_string(&entry.alien_artifact_decision()).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

fn bench_policy_evidence_jsonl(c: &mut Criterion) {
    const ENTRIES: usize = 1_024;
    let entries = (0..ENTRIES).map(policy_evidence).collect::<Vec<_>>();
    let mut ledger = PolicyEvidenceLedger::new(ENTRIES);
    for entry in entries.iter().cloned() {
        ledger.record(entry);
    }
    assert_eq!(
        ledger.to_alien_artifact_jsonl(),
        policy_jsonl_collect_join(&entries)
    );

    let mut group = c.benchmark_group("policy_evidence_jsonl");
    group.bench_function("stream", |b| {
        b.iter(|| black_box(ledger.to_alien_artifact_jsonl()));
    });
    group.bench_function("collect_join_baseline", |b| {
        b.iter(|| black_box(policy_jsonl_collect_join(black_box(&entries))));
    });
    group.finish();
}

fn bench_solver_evidence_rollover(c: &mut Criterion) {
    const CAPACITY: usize = 1_024;
    let mut group = c.benchmark_group("solver_evidence_rollover");

    group.bench_function("deque", |b| {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, CAPACITY);
        for sequence in 0..CAPACITY {
            portfolio.record_evidence(solver_evidence(sequence));
        }
        let mut sequence = CAPACITY;
        b.iter(|| {
            portfolio.record_evidence(solver_evidence(sequence));
            sequence = sequence.wrapping_add(1);
            black_box(portfolio.evidence_len());
        });
    });

    group.bench_function("vec_remove_zero_baseline", |b| {
        let mut evidence = (0..CAPACITY).map(solver_evidence).collect::<Vec<_>>();
        let mut sequence = CAPACITY;
        b.iter(|| {
            let _ = evidence.remove(0);
            evidence.push(solver_evidence(sequence));
            sequence = sequence.wrapping_add(1);
            black_box(evidence.len());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_policy_decide,
    bench_solver_select_action,
    bench_calibrator_observe,
    bench_solver_evidence_jsonl,
    bench_policy_evidence_jsonl,
    bench_solver_evidence_rollover
);
criterion_main!(benches);
