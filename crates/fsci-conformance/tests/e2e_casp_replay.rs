#![forbid(unsafe_code)]
//! frankenscipy-7tb8d.12: CASP decisions replay from their certificates.
//!
//! 1,000 systems are solved through ONE persistent portfolio, so its outcome counts, and with
//! them its posteriors, change from solve to solve. Before each solve the portfolio is
//! snapshotted, and the certificate the solve returns must replay against that snapshot bit
//! for bit: the portfolio's choice, the posterior, every expected loss and the chosen one's.
//! The replay must REFUSE the portfolio as it stands after the solve (its state has advanced)
//! and a snapshot with one outcome count changed, rather than return another decision. A
//! certificate with one posterior bit flipped must not replay against the right snapshot.

use fsci_linalg::{SolveOptions, replay_decision, solve_with_casp};
use fsci_runtime::{AttemptOutcome, RuntimeMode, SolverPortfolio};

/// xorshift64, for a deterministic corpus.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A random `n × n` system whose columns are scaled across `decades` decades, so its rcond
/// spans the portfolio's decades from well to badly conditioned.
fn system(rng: &mut Rng) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = 2 + (rng.next() % 5) as usize;
    let decades = (rng.next() % 13) as f64;
    let a = (0..n)
        .map(|_| {
            (0..n)
                .map(|j| rng.unit() * 10f64.powf(-decades * j as f64 / (n - 1) as f64))
                .collect()
        })
        .collect();
    let b = (0..n).map(|_| rng.unit()).collect();
    (a, b)
}

#[test]
fn casp_decisions_replay_from_their_certificates() {
    const CASES: usize = 1_000;
    let mut rng = Rng(0x7B8D_0012_CA5E_0001);
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let (mut compared, mut unsolved) = (0, 0);
    let (mut advanced_refused, mut mutated_refused, mut tampered_caught) = (0, 0, 0);
    let mut failures = Vec::new();
    let mut digests = std::collections::BTreeSet::new();
    while compared < CASES {
        let (a, b) = system(&mut rng);
        let snapshot = portfolio.clone();
        let Ok(result) = solve_with_casp(&a, &b, SolveOptions::default(), &mut portfolio) else {
            unsolved += 1;
            continue;
        };
        let certificate = result
            .certificate
            .expect("a portfolio solve carries a certificate");
        compared += 1;
        digests.insert(certificate.decision.state_digest.clone());

        let replay = replay_decision(&certificate, &snapshot);
        if !replay.replayed {
            failures.push(format!(
                "case {compared}: {} (certificate {certificate:?}, snapshot digest {})",
                replay.reason,
                snapshot.state_digest()
            ));
        }

        // Negative: the portfolio after the solve has recorded its outcome.
        let advanced = replay_decision(&certificate, &portfolio);
        if !advanced.replayed && advanced.reason.contains("digest") {
            advanced_refused += 1;
        }

        // Negative: one outcome count changed in the snapshot.
        let mut mutated = snapshot.clone();
        mutated.record_outcome(
            certificate.rcond_estimate,
            certificate.action,
            AttemptOutcome::Ok,
        );
        let refused = replay_decision(&certificate, &mutated);
        if !refused.replayed && refused.reason.contains("digest") {
            mutated_refused += 1;
        }

        // Must-hit for the comparison itself: one bit of the recorded posterior flipped, under
        // the right snapshot, is a mismatch (so a zero and a negative zero differ too).
        let mut tampered = certificate.clone();
        tampered.posterior[0] = f64::from_bits(tampered.posterior[0].to_bits() ^ 1);
        let caught = replay_decision(&tampered, &snapshot);
        if !caught.replayed && caught.reason.contains("posterior") {
            tampered_caught += 1;
        }
    }
    eprintln!(
        "casp replay: {compared} certificates, {unsolved} systems unsolved, {} distinct portfolio \
         states; advanced state refused {advanced_refused}, mutated count refused \
         {mutated_refused}, flipped posterior bit caught {tampered_caught}",
        digests.len()
    );
    for failure in &failures {
        eprintln!("  FAIL {failure}");
    }
    assert_eq!(
        compared, CASES,
        "every certificate in the corpus was replayed"
    );
    assert!(failures.is_empty(), "{} replays failed", failures.len());
    assert_eq!(
        advanced_refused, CASES,
        "a snapshot after the solve must be refused"
    );
    assert_eq!(
        mutated_refused, CASES,
        "a mutated outcome count must be refused"
    );
    assert_eq!(
        tampered_caught, CASES,
        "a flipped posterior bit must be caught"
    );
    assert!(
        digests.len() > CASES / 2,
        "the portfolio state must move between solves, or the snapshot proves nothing: {} states",
        digests.len()
    );
}
