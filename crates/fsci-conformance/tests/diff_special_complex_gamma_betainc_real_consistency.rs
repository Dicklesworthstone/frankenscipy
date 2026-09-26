#![forbid(unsafe_code)]
//! Meta-consistency parity for fsci_special complex scalar variants:
//! complex_gammainc, complex_gammaincc, complex_betainc.
//!
//! Resolves [frankenscipy-p3jfg]. When the imaginary part is 0, each
//! complex variant must reduce to the already-tested real variant of
//! the same function. Pair the complex implementation against fsci's
//! own real implementation as the reference. Imaginary part of the
//! result must also be 0 (within 1e-9).
//!
//! Tolerance: 1e-9 abs (series/continued-fraction precision when the
//! complex codepath is forced even on real inputs).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::beta::{betainc_scalar, complex_betainc_scalar};
use fsci_special::gamma::{
    complex_gammainc_scalar, complex_gammaincc_scalar, gammainc_scalar, gammaincc_scalar,
};
use fsci_special::types::Complex64;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create complex_consistency diff dir");
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct _Unused;

#[test]
fn diff_special_complex_gamma_betainc_real_consistency() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The real variant is the reference side, as `[value, 0.0]`; the complex variant's
    // `[re, im]` is compared against it, so a NaN in either part is recorded by the ledger.
    let mut ledger = CompareLedger::new(
        "diff_special_complex_gamma_betainc_real_consistency",
        &["gammainc", "gammaincc", "betainc"],
    );

    // gammainc(a, x) — a > 0, x ≥ 0
    let gi_probes: &[(f64, f64)] = &[
        (0.5, 0.3),
        (0.5, 1.0),
        (0.5, 5.0),
        (1.0, 0.5),
        (1.0, 2.0),
        (2.0, 1.0),
        (2.0, 3.5),
        (5.0, 1.0),
        (5.0, 7.0),
        (10.0, 5.0),
    ];
    for &(a, x) in gi_probes {
        let real = gammainc_scalar(a, x, RuntimeMode::Strict)
            .ok()
            .map(|r| vec![r, 0.0]);
        let cmpx = complex_gammainc_scalar(a, Complex64::new(x, 0.0), RuntimeMode::Strict)
            .ok()
            .map(|c| vec![c.re, c.im]);
        let gi_id = format!("gammainc_a{a}_x{x}");
        if let Some((reference, got)) =
            ledger.slices("gammainc", &gi_id, real.as_deref(), cmpx.as_deref())
        {
            let abs_d = (got[0] - reference[0]).abs().max(got[1].abs());
            max_overall = max_overall.max(abs_d);
            ledger.compared("gammainc", &gi_id, abs_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: gi_id,
                op: "gammainc".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        let real_q = gammaincc_scalar(a, x, RuntimeMode::Strict)
            .ok()
            .map(|r| vec![r, 0.0]);
        let cmpx_q = complex_gammaincc_scalar(a, Complex64::new(x, 0.0), RuntimeMode::Strict)
            .ok()
            .map(|c| vec![c.re, c.im]);
        let gq_id = format!("gammaincc_a{a}_x{x}");
        if let Some((reference, got)) =
            ledger.slices("gammaincc", &gq_id, real_q.as_deref(), cmpx_q.as_deref())
        {
            let abs_d = (got[0] - reference[0]).abs().max(got[1].abs());
            max_overall = max_overall.max(abs_d);
            ledger.compared("gammaincc", &gq_id, abs_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: gq_id,
                op: "gammaincc".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    // betainc(a, b, x) — a > 0, b > 0, x ∈ [0, 1]
    let bi_probes: &[(f64, f64, f64)] = &[
        (1.0, 1.0, 0.5),
        (2.0, 3.0, 0.25),
        (2.0, 3.0, 0.75),
        (0.5, 0.5, 0.5),
        (3.0, 5.0, 0.4),
        (5.0, 2.0, 0.6),
    ];
    for &(a, b, x) in bi_probes {
        let real = betainc_scalar(a, b, x, RuntimeMode::Strict)
            .ok()
            .map(|r| vec![r, 0.0]);
        let cmpx = complex_betainc_scalar(
            Complex64::new(a, 0.0),
            Complex64::new(b, 0.0),
            Complex64::new(x, 0.0),
        );
        let cmpx = [cmpx.re, cmpx.im];
        let bi_id = format!("betainc_a{a}_b{b}_x{x}");
        let Some((reference, got)) =
            ledger.slices("betainc", &bi_id, real.as_deref(), Some(cmpx.as_slice()))
        else {
            continue;
        };
        let abs_d = (got[0] - reference[0]).abs().max(got[1].abs());
        max_overall = max_overall.max(abs_d);
        ledger.compared("betainc", &bi_id, abs_d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: bi_id,
            op: "betainc".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_complex_gamma_betainc_real_consistency".into(),
        category:
            "fsci_special::{complex_gammainc, complex_gammaincc, complex_betainc} vs real variants on real inputs"
                .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "complex/real-consistency conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // gammainc and gammaincc share gi_probes; betainc has the fewer probes.
    ledger.finish(gi_probes.len().min(bi_probes.len()));
}
