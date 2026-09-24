#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.zipfian`.
//!
//! Resolves [frankenscipy-75nmq]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_zipfian_golden.json` and
//! pins fsci-stats Zipfian pmf/cdf against it. Runs offline —
//! catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Zipfian is closed-form via finite zeta sum. Tolerance: 1e-12 abs.

use std::path::PathBuf;

use fsci_stats::{DiscreteDistribution, Zipfian};
use serde::Deserialize;

const POINT_TOL: f64 = 1.0e-12;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    a: f64,
    n: u32,
    k: u64,
    pmf: f64,
    cdf: f64,
}

#[derive(Debug, Deserialize)]
struct GoldenTable {
    packet_id: String,
    family: String,
    oracle: String,
    points: Vec<PointCase>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/FSCI-P2C-007_zipfian_golden.json")
}

#[test]
fn golden_stats_zipfian() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read zipfian golden fixture {path:?}: {e}"));
    let table: GoldenTable = serde_json::from_str(&raw).expect("parse zipfian golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.zipfian");
    assert_eq!(table.oracle, "scipy.stats.zipfian");

    let mut max_pmf = 0.0_f64;
    let mut max_cdf = 0.0_f64;

    for case in &table.points {
        let dist = Zipfian::new(case.a, case.n);
        let pmf_diff = (dist.pmf(case.k as i64) - case.pmf).abs();
        let cdf_diff = (dist.cdf(case.k as i64) - case.cdf).abs();
        max_pmf = max_pmf.max(pmf_diff);
        max_cdf = max_cdf.max(cdf_diff);
        assert!(
            pmf_diff < POINT_TOL,
            "{}: pmf(a={}, n={}, k={}) drift = {pmf_diff} > {POINT_TOL}",
            case.case_id,
            case.a,
            case.n,
            case.k
        );
        assert!(
            cdf_diff < POINT_TOL,
            "{}: cdf(a={}, n={}, k={}) drift = {cdf_diff} > {POINT_TOL}",
            case.case_id,
            case.a,
            case.n,
            case.k
        );
    }

    eprintln!("zipfian golden: pmf_max={max_pmf:e} cdf_max={max_cdf:e}");
}
