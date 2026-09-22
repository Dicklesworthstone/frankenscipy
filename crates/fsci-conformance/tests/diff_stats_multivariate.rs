#![forbid(unsafe_code)]
//! Live SciPy differential coverage for multivariate distributions:
//! - `MultivariateNormal`: pdf, logpdf, entropy, mahalanobis, cov, from_covariance
//! - `MultivariateT`: pdf, logpdf, cov, mahalanobis, from_covariance
//! - `Wishart`: pdf, logpdf, entropy, mean, from_covariance
//! - `InvWishart`: pdf, logpdf, mean, from_covariance
//!
//! Cross-checks FrankenSciPy against live SciPy reference oracle.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{Covariance, InvWishart, MultivariateNormal, MultivariateT, Wishart};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiffRecord {
    case_id: String,
    family: String,
    rust_val: f64,
    scipy_val: f64,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    records: Vec<DiffRecord>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    let _ = fs::create_dir_all(output_dir());
    let path = output_dir().join(format!("{}.json", log.test_id));
    if let Ok(json) = serde_json::to_string_pretty(log) {
        let _ = fs::write(path, json);
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

#[derive(Debug, Clone, Serialize)]
struct MvnCase {
    case_id: String,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MvtCase {
    case_id: String,
    loc: Vec<f64>,
    shape: Vec<Vec<f64>>,
    df: f64,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct WishartCase {
    case_id: String,
    df: f64,
    scale: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct InvWishartCase {
    case_id: String,
    df: f64,
    scale: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    mvn_cases: Vec<MvnCase>,
    mvt_cases: Vec<MvtCase>,
    wishart_cases: Vec<WishartCase>,
    invwishart_cases: Vec<InvWishartCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct MvnOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    mahalanobis_sq: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct MvtOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mahalanobis_sq: f64,
    cov_00: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct WishartOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    entropy: f64,
    mean_00: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct InvWishartOracleResponse {
    case_id: String,
    pdf: f64,
    logpdf: f64,
    mean_00: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResponse {
    mvn: Vec<MvnOracleResponse>,
    mvt: Vec<MvtOracleResponse>,
    wishart: Vec<WishartOracleResponse>,
    invwishart: Vec<InvWishartOracleResponse>,
}

fn run_python_oracle(query: &OracleQuery) -> Option<OracleResponse> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import stats

query = json.load(sys.stdin)
out = {"mvn": [], "mvt": [], "wishart": [], "invwishart": []}

for c in query["mvn_cases"]:
    cid = c["case_id"]
    mean = np.array(c["mean"], dtype=np.float64)
    cov = np.array(c["cov"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    maha_sq = float(diff.T @ inv_cov @ diff)
    out["mvn"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "mahalanobis_sq": maha_sq,
    })

for c in query["mvt_cases"]:
    cid = c["case_id"]
    loc = np.array(c["loc"], dtype=np.float64)
    shape = np.array(c["shape"], dtype=np.float64)
    df = float(c["df"])
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.multivariate_t(loc=loc, shape=shape, df=df)
    diff = x - loc
    inv_shape = np.linalg.inv(shape)
    maha_sq = float(diff.T @ inv_shape @ diff)
    cov_00 = float(shape[0, 0] * df / (df - 2.0)) if df > 2.0 else None
    out["mvt"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "mahalanobis_sq": maha_sq,
        "cov_00": cov_00,
    })

for c in query["wishart_cases"]:
    cid = c["case_id"]
    df = float(c["df"])
    scale = np.array(c["scale"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    rv = stats.wishart(df=df, scale=scale)
    m = rv.mean()
    out["wishart"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "entropy": float(rv.entropy()),
        "mean_00": float(m[0, 0]),
    })

for c in query["invwishart_cases"]:
    cid = c["case_id"]
    df = float(c["df"])
    scale = np.array(c["scale"], dtype=np.float64)
    x = np.array(c["x"], dtype=np.float64)
    p = scale.shape[0]
    rv = stats.invwishart(df=df, scale=scale)
    mean_00 = float(scale[0, 0] / (df - p - 1)) if df > p + 1 else None
    out["invwishart"].append({
        "case_id": cid,
        "pdf": float(rv.pdf(x)),
        "logpdf": float(rv.logpdf(x)),
        "mean_00": mean_00,
    })

json.dump(out, sys.stdout)
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let json_bytes = serde_json::to_vec(query).ok()?;
        stdin.write_all(&json_bytes).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn check_pair(
    case_id: &str,
    family: &str,
    rust_val: f64,
    scipy_val: f64,
    atol: f64,
    rtol: f64,
    records: &mut Vec<DiffRecord>,
) {
    let abs_diff = (rust_val - scipy_val).abs();
    let rel_diff = if scipy_val.abs() > 1e-15 {
        abs_diff / scipy_val.abs()
    } else {
        abs_diff
    };
    let pass = abs_diff <= atol || rel_diff <= rtol;
    records.push(DiffRecord {
        case_id: case_id.to_string(),
        family: family.to_string(),
        rust_val,
        scipy_val,
        abs_diff,
        rel_diff,
        pass,
    });
    assert!(
        pass,
        "[{family}] {case_id}: rust={rust_val}, scipy={scipy_val}, abs_diff={abs_diff}, rel_diff={rel_diff} (atol={atol}, rtol={rtol})"
    );
}

#[test]
fn diff_multivariate_stats_scipy_oracle() {
    let t0 = Instant::now();

    // 1. Build test cases across dimensions
    let mvn_cases = vec![
        MvnCase {
            case_id: "mvn_2d_diag".into(),
            mean: vec![0.5, -1.0],
            cov: vec![vec![2.0, 0.0], vec![0.0, 1.5]],
            x: vec![1.0, -0.5],
        },
        MvnCase {
            case_id: "mvn_2d_rotated".into(),
            mean: vec![1.0, 2.0],
            cov: vec![vec![2.0, 0.6], vec![0.6, 1.0]],
            x: vec![1.5, 2.5],
        },
        MvnCase {
            case_id: "mvn_3d_corr".into(),
            mean: vec![0.0, 1.0, 2.0],
            cov: vec![
                vec![3.0, 0.5, 0.2],
                vec![0.5, 2.0, 0.4],
                vec![0.2, 0.4, 1.5],
            ],
            x: vec![0.2, 1.2, 1.8],
        },
    ];

    let mvt_cases = vec![
        MvtCase {
            case_id: "mvt_2d_df3".into(),
            loc: vec![0.0, 0.0],
            shape: vec![vec![1.5, 0.3], vec![0.3, 2.0]],
            df: 3.0,
            x: vec![0.5, -0.5],
        },
        MvtCase {
            case_id: "mvt_2d_df10".into(),
            loc: vec![1.0, -1.0],
            shape: vec![vec![2.0, 0.5], vec![0.5, 1.2]],
            df: 10.0,
            x: vec![1.2, -0.8],
        },
        MvtCase {
            case_id: "mvt_3d_df5".into(),
            loc: vec![0.5, 0.5, 0.5],
            shape: vec![
                vec![2.0, 0.2, 0.1],
                vec![0.2, 1.5, 0.3],
                vec![0.1, 0.3, 1.8],
            ],
            df: 5.0,
            x: vec![1.0, 0.0, 0.8],
        },
        MvtCase {
            case_id: "mvt_2d_df1_5_no_cov".into(),
            loc: vec![0.0, 0.0],
            shape: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            df: 1.5,
            x: vec![0.1, 0.2],
        },
    ];

    let wishart_cases = vec![
        WishartCase {
            case_id: "wishart_2d_df5".into(),
            df: 5.0,
            scale: vec![vec![1.5, 0.4], vec![0.4, 2.0]],
            x: vec![vec![3.0, 0.8], vec![0.8, 4.0]],
        },
        WishartCase {
            case_id: "wishart_3d_df8".into(),
            df: 8.0,
            scale: vec![
                vec![2.0, 0.3, 0.1],
                vec![0.3, 1.5, 0.2],
                vec![0.1, 0.2, 1.8],
            ],
            x: vec![
                vec![5.0, 0.7, 0.3],
                vec![0.7, 4.0, 0.5],
                vec![0.3, 0.5, 4.5],
            ],
        },
    ];

    let invwishart_cases = vec![
        InvWishartCase {
            case_id: "invwishart_2d_df6".into(),
            df: 6.0,
            scale: vec![vec![1.5, 0.4], vec![0.4, 2.0]],
            x: vec![vec![0.8, 0.1], vec![0.1, 1.0]],
        },
        InvWishartCase {
            case_id: "invwishart_3d_df9".into(),
            df: 9.0,
            scale: vec![
                vec![2.0, 0.3, 0.1],
                vec![0.3, 1.5, 0.2],
                vec![0.1, 0.2, 1.8],
            ],
            x: vec![
                vec![0.5, 0.05, 0.02],
                vec![0.05, 0.4, 0.03],
                vec![0.02, 0.03, 0.6],
            ],
        },
    ];

    let query = OracleQuery {
        mvn_cases: mvn_cases.clone(),
        mvt_cases: mvt_cases.clone(),
        wishart_cases: wishart_cases.clone(),
        invwishart_cases: invwishart_cases.clone(),
    };

    let oracle_opt = run_python_oracle(&query);
    if oracle_opt.is_none() {
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            panic!("SciPy oracle required via {REQUIRE_SCIPY_ENV} but unavailable or failed");
        }
        eprintln!("Skipping live differential tests: SciPy oracle not available");
        return;
    }
    let oracle = oracle_opt.unwrap();

    let mut records = Vec::new();

    // Test MultivariateNormal
    for (case, resp) in mvn_cases.iter().zip(oracle.mvn.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MultivariateNormal::new(&case.mean, &case.cov).expect("mvn new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("mvn pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("mvn logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MultivariateNormal",
            rust_pdf,
            resp.pdf,
            1e-11,
            1e-10,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MultivariateNormal",
            rust_logpdf,
            resp.logpdf,
            1e-11,
            1e-10,
            &mut records,
        );

        // entropy
        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "MultivariateNormal",
            rust_entropy,
            resp.entropy,
            1e-11,
            1e-10,
            &mut records,
        );

        // mahalanobis squared
        let rust_maha_sq = dist.mahalanobis_squared(&case.x).expect("mvn maha_sq");
        check_pair(
            &format!("{}_maha_sq", case.case_id),
            "MultivariateNormal",
            rust_maha_sq,
            resp.mahalanobis_sq,
            1e-11,
            1e-10,
            &mut records,
        );
        let rust_maha = dist.mahalanobis(&case.x).expect("mvn maha");
        check_pair(
            &format!("{}_maha", case.case_id),
            "MultivariateNormal",
            rust_maha,
            resp.mahalanobis_sq.sqrt(),
            1e-11,
            1e-10,
            &mut records,
        );

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.cov).expect("cov psd");
        let dist_from_cov =
            MultivariateNormal::from_covariance(&case.mean, &cov_rep).expect("mvn from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.mean(), dist_from_cov.mean());
        assert_eq!(dist.cov(), dist_from_cov.cov());
        assert!((dist.entropy() - dist_from_cov.entropy()).abs() < 1e-14);
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs() < 1e-14
        );
    }

    // Test MultivariateT
    for (case, resp) in mvt_cases.iter().zip(oracle.mvt.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = MultivariateT::new(&case.loc, &case.shape, case.df).expect("mvt new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("mvt pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("mvt logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "MultivariateT",
            rust_pdf,
            resp.pdf,
            1e-11,
            1e-10,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "MultivariateT",
            rust_logpdf,
            resp.logpdf,
            1e-11,
            1e-10,
            &mut records,
        );

        // mahalanobis squared
        let rust_maha_sq = dist.mahalanobis_squared(&case.x).expect("mvt maha_sq");
        check_pair(
            &format!("{}_maha_sq", case.case_id),
            "MultivariateT",
            rust_maha_sq,
            resp.mahalanobis_sq,
            1e-11,
            1e-10,
            &mut records,
        );

        // cov check
        if let Some(scipy_c00) = resp.cov_00 {
            let rust_cov = dist.cov().expect("mvt cov should exist for df > 2");
            check_pair(
                &format!("{}_cov_00", case.case_id),
                "MultivariateT",
                rust_cov[0][0],
                scipy_c00,
                1e-11,
                1e-10,
                &mut records,
            );
        } else {
            assert!(dist.cov().is_none(), "mvt cov must be None for df <= 2");
        }

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.shape).expect("cov psd");
        let dist_from_cov =
            MultivariateT::from_covariance(&case.loc, &cov_rep, case.df).expect("mvt from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.loc(), dist_from_cov.loc());
        assert_eq!(dist.shape(), dist_from_cov.shape());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs() < 1e-14
        );
    }

    // Test Wishart
    for (case, resp) in wishart_cases.iter().zip(oracle.wishart.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = Wishart::new(case.df, &case.scale).expect("wishart new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("wishart pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("wishart logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "Wishart",
            rust_pdf,
            resp.pdf,
            1e-10,
            1e-9,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "Wishart",
            rust_logpdf,
            resp.logpdf,
            1e-10,
            1e-9,
            &mut records,
        );

        // entropy
        let rust_entropy = dist.entropy();
        check_pair(
            &format!("{}_entropy", case.case_id),
            "Wishart",
            rust_entropy,
            resp.entropy,
            1e-10,
            1e-9,
            &mut records,
        );

        // mean
        let rust_mean = dist.mean();
        check_pair(
            &format!("{}_mean_00", case.case_id),
            "Wishart",
            rust_mean[0][0],
            resp.mean_00,
            1e-11,
            1e-10,
            &mut records,
        );

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.scale).expect("cov psd");
        let dist_from_cov = Wishart::from_covariance(case.df, &cov_rep).expect("wishart from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert_eq!(dist.scale(), dist_from_cov.scale());
        assert!((dist.entropy() - dist_from_cov.entropy()).abs() < 1e-14);
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs() < 1e-14
        );
    }

    // Test InvWishart
    for (case, resp) in invwishart_cases.iter().zip(oracle.invwishart.iter()) {
        assert_eq!(case.case_id, resp.case_id);
        let dist = InvWishart::new(case.df, &case.scale).expect("invwishart new");

        // pdf & logpdf
        let rust_pdf = dist.pdf(&case.x).expect("invwishart pdf");
        let rust_logpdf = dist.logpdf(&case.x).expect("invwishart logpdf");
        check_pair(
            &format!("{}_pdf", case.case_id),
            "InvWishart",
            rust_pdf,
            resp.pdf,
            1e-10,
            1e-9,
            &mut records,
        );
        check_pair(
            &format!("{}_logpdf", case.case_id),
            "InvWishart",
            rust_logpdf,
            resp.logpdf,
            1e-10,
            1e-9,
            &mut records,
        );

        // mean
        if let Some(scipy_m00) = resp.mean_00 {
            let rust_mean = dist.mean();
            check_pair(
                &format!("{}_mean_00", case.case_id),
                "InvWishart",
                rust_mean[0][0],
                scipy_m00,
                1e-11,
                1e-10,
                &mut records,
            );
        }

        // Test from_covariance parity
        let cov_rep = Covariance::from_psd(&case.scale).expect("cov psd");
        let dist_from_cov =
            InvWishart::from_covariance(case.df, &cov_rep).expect("invwishart from_cov");
        assert_eq!(dist.dim(), dist_from_cov.dim());
        assert_eq!(dist.df(), dist_from_cov.df());
        assert_eq!(dist.scale(), dist_from_cov.scale());
        assert!(
            (dist.logpdf(&case.x).unwrap() - dist_from_cov.logpdf(&case.x).unwrap()).abs() < 1e-14
        );
    }

    let duration_ns = t0.elapsed().as_nanos();
    let max_abs_diff = records.iter().map(|r| r.abs_diff).fold(0.0_f64, f64::max);
    let max_rel_diff = records.iter().map(|r| r.rel_diff).fold(0.0_f64, f64::max);
    let all_pass = records.iter().all(|r| r.pass);

    let log = DiffLog {
        test_id: "diff_stats_multivariate".into(),
        category: "stats.multivariate_distributions".into(),
        case_count: records.len(),
        max_abs_diff,
        max_rel_diff,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns,
        records,
    };
    emit_log(&log);
    assert!(
        all_pass,
        "all multivariate differential test cases must pass"
    );
}
