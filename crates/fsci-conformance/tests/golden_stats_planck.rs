#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.planck`.
//!
//! Resolves [frankenscipy-l0r9e]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_planck_golden.json` and
//! pins fsci-stats Planck pmf/cdf against it. Runs offline —
//! catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Planck is fully closed-form; tolerance 1e-12 abs.

use std::path::PathBuf;

use fsci_stats::{DiscreteDistribution, Planck};
use serde::Deserialize;

const POINT_TOL: f64 = 1.0e-12;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    #[serde(rename = "lambda")]
    lambda_: f64,
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/FSCI-P2C-007_planck_golden.json")
}

#[test]
fn golden_stats_planck() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read planck golden fixture {path:?}: {e}"));
    let table: GoldenTable = serde_json::from_str(&raw).expect("parse planck golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.planck");
    assert_eq!(table.oracle, "scipy.stats.planck");

    let mut max_pmf = 0.0_f64;
    let mut max_cdf = 0.0_f64;

    for case in &table.points {
        let dist = Planck::new(case.lambda_);
        let pmf_diff = (dist.pmf(case.k as i64) - case.pmf).abs();
        let cdf_diff = (dist.cdf(case.k as i64) - case.cdf).abs();
        max_pmf = max_pmf.max(pmf_diff);
        max_cdf = max_cdf.max(cdf_diff);
        assert!(
            pmf_diff < POINT_TOL,
            "{}: pmf(λ={}, k={}) drift = {pmf_diff} > {POINT_TOL}",
            case.case_id,
            case.lambda_,
            case.k
        );
        assert!(
            cdf_diff < POINT_TOL,
            "{}: cdf(λ={}, k={}) drift = {cdf_diff} > {POINT_TOL}",
            case.case_id,
            case.lambda_,
            case.k
        );
    }

    eprintln!("planck golden: pmf_max={max_pmf:e} cdf_max={max_cdf:e}");
}
