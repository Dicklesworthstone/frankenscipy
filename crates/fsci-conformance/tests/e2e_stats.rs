#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-009 (Statistical distributions).
//!
//! Acceptance criteria:
//!   Happy-path:     1-3  (distribution evaluation, CDF/PPF roundtrip, parameter estimation)
//!   Error recovery: 4-6  (invalid parameters, boundary quantiles, NaN inputs)
//!   Adversarial:    7-10 (extreme parameters, tail probabilities, identity checks)
//!
//! Each scenario emits topology-compliant artifacts to
//! `fixtures/artifacts/FSCI-P2C-012/e2e/runs/{run_id}/{scenario_id}/`
//! containing `events.jsonl` and `summary.json`.

use fsci_conformance::PacketFamily;
use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    Argus,
    BetaDist,
    Burr12,
    ChiSquared,
    ContinuousDistribution,
    CrystalBall,
    ExponWeibull,
    Exponential,
    FrechetR,
    GammaDist,
    GenHalfLogistic,
    GenLogistic,
    Gompertz,
    Gumbel,
    InvWeibull,
    LogLaplace,
    Logistic,
    Lognormal,
    Maxwell,
    Mielke,
    Normal,
    Pareto,
    Rayleigh,
    StudentT,
    TukeyLambda,
    Uniform,
    Weibull,
    chi2_contingency,
    // Hypothesis tests
    chisquare,
    combine_pvalues,
    describe,
    expected_freq_uniform,
    f_oneway,
    false_discovery_control,
    fisher_exact,
    fit,
    gzscore,
    jarque_bera,
    kendalltau,
    kruskal,
    ks_1samp,
    ks_2samp,
    kurtosis,
    linregress,
    mannwhitneyu,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_abs_deviation,
    median_test,
    mode,
    mood,
    multipletests_bonferroni,
    multipletests_fdr_bh,
    multipletests_holm,
    multiscale_graphcorr,
    normaltest,
    pearsonr,
    poisson_means_test,
    power_divergence,
    probplot_quantiles,
    r2_score,
    rankdata,
    root_mean_squared_error,
    scoreatpercentile,
    shapiro,
    skew,
    skewtest,
    spearmanr,
    tmean,
    tsem,
    tstd,
    ttest_1samp,
    ttest_ind,
    ttest_ind_from_stats,
    ttest_rel,
    tvar,
    wilcoxon,
    winsorize,
    zmap,
};
use serde::{Deserialize, Serialize};

// ---- Topology-compliant log types ----

#[derive(Debug, Clone, Serialize)]
struct EventEntry {
    scenario_id: String,
    step_name: String,
    timestamp_ms: u128,
    duration_ms: u128,
    outcome: String,
    message: String,
    environment: EnvironmentInfo,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RunSummary {
    packet_id: String,
    scenario_id: String,
    run_id: String,
    passed: bool,
    failed_step: Option<String>,
    replay_command: String,
    generated_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

type ScalarFn = Box<dyn Fn(f64) -> f64>;
type RoundtripDistribution = (&'static str, ScalarFn, ScalarFn);
type CdfDistribution = (&'static str, ScalarFn);

fn packet_id() -> &'static str {
    PacketFamily::Stats.packet_id()
}

// ---- Helpers ----

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn run_id() -> String {
    format!("run-{}", now_unix_ms())
}

fn e2e_runs_dir(run_id: &str, scenario_id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(packet_id())
        .join("e2e/runs")
        .join(run_id)
        .join(scenario_id)
}

fn rustc_version_string() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| stdout.trim().to_string())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| String::from("unknown"))
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: rustc_version_string(),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_stats -- {scenario_id} --nocapture"
    )
}

fn write_topology_artifacts(
    scenario_id: &str,
    steps: &[ForensicStep],
    all_pass: bool,
) -> Result<(), String> {
    let rid = run_id();
    let dir = e2e_runs_dir(&rid, scenario_id);
    fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create run dir {}: {e}", dir.display()))?;

    // Write events.jsonl — one JSON line per step
    let events_path = dir.join("events.jsonl");
    let file = fs::File::create(&events_path)
        .map_err(|e| format!("failed to create {}: {e}", events_path.display()))?;
    let mut writer = BufWriter::new(file);
    let env = make_env();
    for step in steps {
        let entry = EventEntry {
            scenario_id: scenario_id.to_string(),
            step_name: step.step_name.clone(),
            timestamp_ms: now_unix_ms(),
            duration_ms: step.duration_ns / 1_000_000,
            outcome: step.outcome.clone(),
            message: format!("{}: {}", step.action, step.output_summary),
            environment: env.clone(),
            artifact_refs: vec![],
        };
        serde_json::to_writer(&mut writer, &entry)
            .map_err(|e| format!("failed to serialize event for {scenario_id}: {e}"))?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("failed to write newline to {}: {e}", events_path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("failed to flush {}: {e}", events_path.display()))?;

    // Write summary.json
    let first_fail = steps
        .iter()
        .find(|s| s.outcome == "FAIL")
        .map(|s| s.step_name.clone());
    let summary = RunSummary {
        packet_id: packet_id().to_string(),
        scenario_id: scenario_id.to_string(),
        run_id: rid,
        passed: all_pass,
        failed_step: first_fail,
        replay_command: replay_cmd(scenario_id),
        generated_unix_ms: now_unix_ms(),
    };
    let summary_path = dir.join("summary.json");
    let json = serde_json::to_vec_pretty(&summary)
        .map_err(|e| format!("failed to serialize summary for {scenario_id}: {e}"))?;
    fs::write(&summary_path, &json)
        .map_err(|e| format!("failed to write {}: {e}", summary_path.display()))?;
    Ok(())
}

const TOL: f64 = 1e-8;
const TRIMMED_STATS_SCIPY_SCRIPT: &str = r#"
import json
from scipy import stats

CASES = [
    {
        "case_id": "left_closed_right_open",
        "data": [1.0, 2.0, 2.0, 3.5, 4.0, 7.0, 9.0],
        "limits": (2.0, 7.0),
        "inclusive": (True, False),
        "ddof": 1,
    },
    {
        "case_id": "left_open_right_closed",
        "data": [1.0, 2.0, 2.0, 3.5, 4.0, 7.0, 9.0],
        "limits": (2.0, 7.0),
        "inclusive": (False, True),
        "ddof": 1,
    },
    {
        "case_id": "population_ddof_zero",
        "data": [-3.0, -1.0, 0.0, 2.0, 5.0, 8.0],
        "limits": (-1.0, 8.0),
        "inclusive": (True, True),
        "ddof": 0,
    },
]

outputs = []
for case in CASES:
    data = case["data"]
    limits = case["limits"]
    inclusive = case["inclusive"]
    ddof = case["ddof"]
    outputs.append({
        "case_id": case["case_id"],
        "tmean": float(stats.tmean(data, limits=limits, inclusive=inclusive)),
        "tvar": float(stats.tvar(data, limits=limits, inclusive=inclusive, ddof=ddof)),
        "tstd": float(stats.tstd(data, limits=limits, inclusive=inclusive, ddof=ddof)),
        "tsem": float(stats.tsem(data, limits=limits, inclusive=inclusive, ddof=ddof)),
    })

print(json.dumps(outputs, allow_nan=False))
"#;

const MGC_SCIPY_SCRIPT: &str = r#"
import json
import numpy as np
from scipy.stats import multiscale_graphcorr

x_linear = np.arange(8.0)[:, None]
y_linear = (2.0 * np.arange(8.0) + 1.0)[:, None]
linear = multiscale_graphcorr(x_linear, y_linear, reps=0, random_state=0)

print(json.dumps({
    "linear": {
        "statistic": float(linear.statistic),
        "pvalue": float(linear.pvalue),
        "opt_scale": [int(v) for v in linear.mgc_dict["opt_scale"]],
        "map_shape": [int(v) for v in linear.mgc_dict["mgc_map"].shape],
    },
}, allow_nan=False))
"#;

#[derive(Debug, Clone, Copy)]
struct TrimmedStatsCase {
    case_id: &'static str,
    data: &'static [f64],
    limits: (f64, f64),
    inclusive: (bool, bool),
    ddof: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct TrimmedStatsOracle {
    case_id: String,
    tmean: f64,
    tvar: f64,
    tstd: f64,
    tsem: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct MgcOracleOutput {
    linear: MgcOracleCase,
}

#[derive(Debug, Clone, Deserialize)]
struct MgcOracleCase {
    statistic: f64,
    pvalue: f64,
    opt_scale: [usize; 2],
    map_shape: [usize; 2],
}

fn trimmed_stats_cases() -> Vec<TrimmedStatsCase> {
    vec![
        TrimmedStatsCase {
            case_id: "left_closed_right_open",
            data: &[1.0, 2.0, 2.0, 3.5, 4.0, 7.0, 9.0],
            limits: (2.0, 7.0),
            inclusive: (true, false),
            ddof: 1,
        },
        TrimmedStatsCase {
            case_id: "left_open_right_closed",
            data: &[1.0, 2.0, 2.0, 3.5, 4.0, 7.0, 9.0],
            limits: (2.0, 7.0),
            inclusive: (false, true),
            ddof: 1,
        },
        TrimmedStatsCase {
            case_id: "population_ddof_zero",
            data: &[-3.0, -1.0, 0.0, 2.0, 5.0, 8.0],
            limits: (-1.0, 8.0),
            inclusive: (true, true),
            ddof: 0,
        },
    ]
}

fn make_step(
    step_id: usize,
    name: &str,
    action: &str,
    input: &str,
    output: &str,
    dur: u128,
    outcome: &str,
) -> ForensicStep {
    ForensicStep {
        step_id,
        step_name: name.to_string(),
        action: action.to_string(),
        input_summary: input.to_string(),
        output_summary: output.to_string(),
        duration_ns: dur,
        mode: "strict".to_string(),
        outcome: outcome.to_string(),
    }
}

fn assert_artifacts_written(scenario_id: &str, steps: &[ForensicStep], all_pass: bool) {
    let artifact_write = write_topology_artifacts(scenario_id, steps, all_pass);
    assert!(
        artifact_write.is_ok(),
        "artifact write failed for {scenario_id}: {}",
        artifact_write
            .as_ref()
            .err()
            .map_or("unknown error", String::as_str)
    );
}

// ======================================================================
// HAPPY-PATH SCENARIOS (1-3)
// ======================================================================

/// Scenario 1: Normal distribution evaluation chain.
/// Verifies PDF at mean, CDF symmetry, and PPF roundtrip.
#[test]
fn e2e_001_normal_distribution_chain() {
    let scenario_id = "e2e_stats_001_normal";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    // Step 1: PDF at mean = 1/sqrt(2*pi)
    let t = Instant::now();
    let pdf_at_0 = n.pdf(0.0);
    let expected_pdf = 1.0 / (2.0 * PI).sqrt();
    let pass = (pdf_at_0 - expected_pdf).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "pdf_at_mean",
        "Normal::pdf",
        "x=0.0",
        &format!("{pdf_at_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 2: CDF(0) = 0.5 (symmetry)
    let t = Instant::now();
    let cdf_at_0 = n.cdf(0.0);
    let pass = (cdf_at_0 - 0.5).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "cdf_symmetry",
        "Normal::cdf",
        "x=0.0",
        &format!("{cdf_at_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 3: PPF(0.5) = 0.0
    let t = Instant::now();
    let ppf_half = n.ppf(0.5);
    let pass = ppf_half.abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ppf_median",
        "Normal::ppf",
        "q=0.5",
        &format!("{ppf_half}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 4: CDF/PPF roundtrip for q = 0.975
    let t = Instant::now();
    let q = 0.975;
    let x = n.ppf(q);
    let roundtrip = n.cdf(x);
    let pass = (roundtrip - q).abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "cdf_ppf_roundtrip",
        "CDF(PPF(0.975))",
        &format!("q={q}"),
        &format!("roundtrip={roundtrip}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Step 5: SF(x) + CDF(x) = 1
    let t = Instant::now();
    let x = 1.96;
    let sum = n.cdf(x) + n.sf(x);
    let pass = (sum - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "sf_cdf_complement",
        "CDF(1.96) + SF(1.96)",
        &format!("x={x}"),
        &format!("sum={sum}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 2: Multi-distribution CDF/PPF roundtrip.
/// Verifies the inverse CDF contract for several distributions.
#[test]
fn e2e_002_multi_distribution_roundtrip() {
    let scenario_id = "e2e_stats_002_roundtrip";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let distributions: Vec<RoundtripDistribution> = vec![
        {
            let dist = Normal::standard();
            (
                "Normal(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = StudentT::new(5.0);
            (
                "StudentT(5)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = ChiSquared::new(3.0);
            (
                "ChiSq(3)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Exponential::new(2.0);
            (
                "Exponential(2)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Uniform::new(0.0, 1.0);
            (
                "Uniform(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = BetaDist::new(2.0, 5.0);
            (
                "Beta(2,5)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = GammaDist::new(3.0, 2.0);
            (
                "Gamma(3,2)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Weibull::new(1.5, 1.0);
            (
                "Weibull(1.5,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Lognormal::new(1.0, 1.0);
            (
                "Lognormal(1,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Gumbel::new(0.0, 1.0);
            (
                "Gumbel(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
        {
            let dist = Logistic::new(0.0, 1.0);
            (
                "Logistic(0,1)",
                Box::new(move |q| dist.ppf(q)),
                Box::new(move |x| dist.cdf(x)),
            )
        },
    ];

    let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

    for (step_id, (name, ppf, cdf)) in distributions.iter().enumerate() {
        let t = Instant::now();
        let mut dist_pass = true;
        for &q in &quantiles {
            let x = ppf(q);
            if !x.is_finite() {
                continue;
            }
            let roundtrip = cdf(x);
            if (roundtrip - q).abs() > 1e-4 {
                dist_pass = false;
            }
        }
        if !dist_pass {
            all_pass = false;
        }
        steps.push(make_step(
            step_id + 1,
            name,
            "CDF(PPF(q)) roundtrip",
            &format!("quantiles={quantiles:?}"),
            &format!("all_close={dist_pass}"),
            t.elapsed().as_nanos(),
            if dist_pass { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

#[test]
fn e2e_002b_gamma_beta_scipy_golden() {
    let scenario_id = "e2e_stats_002b_gamma_beta_scipy_golden";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let gamma = GammaDist::new(3.0, 2.0);
    let t = Instant::now();
    let gamma_actual = [gamma.pdf(4.5), gamma.cdf(4.5), gamma.ppf(0.7)];
    let gamma_expected = [
        0.133_395_893_586_109_54,
        0.390_660_733_001_721_94,
        7.231_135_331_731_982,
    ];
    let gamma_diff = gamma_actual
        .iter()
        .zip(gamma_expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let gamma_pass = gamma_diff <= TOL;
    all_pass &= gamma_pass;
    steps.push(make_step(
        1,
        "gamma_pdf_cdf_ppf",
        "GammaDist(3,2) pdf/cdf/ppf",
        "x=4.5, q=0.7",
        &format!("max_diff={gamma_diff:.3e}"),
        t.elapsed().as_nanos(),
        if gamma_pass { "pass" } else { "FAIL" },
    ));

    let beta = BetaDist::new(2.0, 5.0);
    let t = Instant::now();
    let beta_actual = [beta.pdf(0.4), beta.cdf(0.4), beta.ppf(0.7)];
    let beta_expected = [
        1.555_200_000_000_000_6,
        0.766_720_000_000_000_1,
        0.360_357_690_380_020_13,
    ];
    let beta_diff = beta_actual
        .iter()
        .zip(beta_expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let beta_pass = beta_diff <= TOL;
    all_pass &= beta_pass;
    steps.push(make_step(
        2,
        "beta_pdf_cdf_ppf",
        "BetaDist(2,5) pdf/cdf/ppf",
        "x=0.4, q=0.7",
        &format!("max_diff={beta_diff:.3e}"),
        t.elapsed().as_nanos(),
        if beta_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 3: Distribution moments verification.
/// mean and var should match known analytical values.
#[test]
fn e2e_003_moments_verification() {
    let scenario_id = "e2e_stats_003_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Normal(3, 2): mean=3, var=4
    let t = Instant::now();
    let n = Normal::new(3.0, 2.0);
    let pass = (n.mean() - 3.0).abs() < TOL && (n.var() - 4.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "normal_moments",
        "Normal(3,2)",
        "loc=3, scale=2",
        &format!("mean={}, var={}", n.mean(), n.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Exponential(2): mean=0.5, var=0.25
    let t = Instant::now();
    let e = Exponential::new(2.0);
    let pass = (e.mean() - 0.5).abs() < TOL && (e.var() - 0.25).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "exp_moments",
        "Exponential(2)",
        "lambda=2",
        &format!("mean={}, var={}", e.mean(), e.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Uniform(0,1): mean=0.5, var=1/12
    let t = Instant::now();
    let u = Uniform::new(0.0, 1.0);
    let pass = (u.mean() - 0.5).abs() < TOL && (u.var() - 1.0 / 12.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "uniform_moments",
        "Uniform(0,1)",
        "loc=0, scale=1",
        &format!("mean={}, var={}", u.mean(), u.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // StudentT(5): mean=0, var=5/3
    let t = Instant::now();
    let st = StudentT::new(5.0);
    let pass = st.mean().abs() < TOL && (st.var() - 5.0 / 3.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "student_t_moments",
        "StudentT(5)",
        "df=5",
        &format!("mean={}, var={}", st.mean(), st.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // ChiSquared(4): mean=4, var=8
    let t = Instant::now();
    let c = ChiSquared::new(4.0);
    let pass = (c.mean() - 4.0).abs() < TOL && (c.var() - 8.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "chi2_moments",
        "ChiSquared(4)",
        "df=4",
        &format!("mean={}, var={}", c.mean(), c.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

// ======================================================================
// ERROR RECOVERY SCENARIOS (4-6)
// ======================================================================

/// Scenario 4: Boundary quantile handling.
/// PPF(0) = -inf, PPF(1) = +inf for unbounded distributions.
#[test]
fn e2e_004_boundary_quantiles() {
    let scenario_id = "e2e_stats_004_boundaries";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    let t = Instant::now();
    let ppf_0 = n.ppf(0.0);
    let pass = ppf_0 == f64::NEG_INFINITY;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ppf_zero",
        "Normal::ppf(0)",
        "q=0",
        &format!("{ppf_0}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let ppf_1 = n.ppf(1.0);
    let pass = ppf_1 == f64::INFINITY;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ppf_one",
        "Normal::ppf(1)",
        "q=1",
        &format!("{ppf_1}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // CDF(-inf) = 0, CDF(+inf) = 1
    let t = Instant::now();
    let cdf_neg_inf = n.cdf(f64::NEG_INFINITY);
    let cdf_pos_inf = n.cdf(f64::INFINITY);
    let pass = cdf_neg_inf.abs() < TOL && (cdf_pos_inf - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "cdf_infinities",
        "Normal::cdf",
        "x=-inf, x=+inf",
        &format!("cdf(-inf)={cdf_neg_inf}, cdf(+inf)={cdf_pos_inf}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // PDF should be non-negative everywhere
    let t = Instant::now();
    let pass = [
        f64::NEG_INFINITY,
        -100.0,
        -1.0,
        0.0,
        1.0,
        100.0,
        f64::INFINITY,
    ]
    .iter()
    .all(|&x| n.pdf(x) >= 0.0);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "pdf_nonnegative",
        "Normal::pdf",
        "various x",
        "all >= 0",
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 5: CDF monotonicity verification across distributions.
#[test]
fn e2e_005_cdf_monotonicity() {
    let scenario_id = "e2e_stats_005_monotonicity";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let distributions: Vec<CdfDistribution> = vec![
        {
            let dist = Normal::standard();
            ("Normal(0,1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Exponential::new(1.0);
            ("Exponential(1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = ChiSquared::new(2.0);
            ("ChiSq(2)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Pareto::new(2.0, 1.0);
            ("Pareto(2,1)", Box::new(move |x| dist.cdf(x)))
        },
        {
            let dist = Rayleigh::new(1.0);
            ("Rayleigh(1)", Box::new(move |x| dist.cdf(x)))
        },
    ];

    for (step_id, (name, cdf)) in distributions.iter().enumerate() {
        let t = Instant::now();
        let xs: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.2).collect();
        let cdfs: Vec<f64> = xs.iter().map(|&x| cdf(x)).collect();
        let monotonic = cdfs.windows(2).all(|w| w[1] >= w[0] - 1e-15);
        if !monotonic {
            all_pass = false;
        }
        steps.push(make_step(
            step_id + 1,
            name,
            "CDF monotonicity check",
            "x in [-10, 10] step 0.2",
            &format!("monotonic={monotonic}"),
            t.elapsed().as_nanos(),
            if monotonic { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 6: PDF integrates to 1 (trapezoidal approximation).
#[test]
fn e2e_006_pdf_normalization() {
    let scenario_id = "e2e_stats_006_normalization";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Normal: integrate PDF from -10 to 10
    let t = Instant::now();
    let n = Normal::standard();
    let integral = trapezoidal_integrate(|x| n.pdf(x), -10.0, 10.0, 10_000);
    let pass = (integral - 1.0).abs() < 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "normal_pdf_integral",
        "trapezoidal(Normal::pdf, -10, 10)",
        "n=10000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Exponential: integrate PDF from 0 to 50
    let t = Instant::now();
    let e = Exponential::new(1.0);
    let integral = trapezoidal_integrate(|x| e.pdf(x), 0.0, 50.0, 10_000);
    let pass = (integral - 1.0).abs() < 1e-4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "exp_pdf_integral",
        "trapezoidal(Exp::pdf, 0, 50)",
        "n=10000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Uniform: integrate PDF from -1 to 2
    // Uniform PDF is discontinuous at 0 and 1, so trapezoidal rule needs more
    // points or a relaxed tolerance to handle the step boundaries.
    let t = Instant::now();
    let u = Uniform::new(0.0, 1.0);
    let integral = trapezoidal_integrate(|x| u.pdf(x), -1.0, 2.0, 100_000);
    let pass = (integral - 1.0).abs() < 1e-4;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "uniform_pdf_integral",
        "trapezoidal(Uniform::pdf, -1, 2)",
        "n=1000",
        &format!("integral={integral}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

fn trapezoidal_integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

// ======================================================================
// ADVERSARIAL SCENARIOS (7-10)
// ======================================================================

/// Scenario 7: Extreme tail probabilities.
/// CDF at far tails should be near 0 or 1.
#[test]
fn e2e_007_extreme_tails() {
    let scenario_id = "e2e_stats_007_tails";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    // CDF(-8) should be very close to 0
    let t = Instant::now();
    let cdf_far_left = n.cdf(-8.0);
    let pass = (0.0..1e-10).contains(&cdf_far_left);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "cdf_far_left",
        "Normal::cdf(-8)",
        "x=-8",
        &format!("{cdf_far_left:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // CDF(8) should be very close to 1
    let t = Instant::now();
    let cdf_far_right = n.cdf(8.0);
    let pass = (cdf_far_right - 1.0).abs() < 1e-10;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "cdf_far_right",
        "Normal::cdf(8)",
        "x=8",
        &format!("{cdf_far_right}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // SF should complement CDF even in tails
    let t = Instant::now();
    let sf_8 = n.sf(8.0);
    let pass = sf_8 > 0.0 && sf_8 < 1e-10;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "sf_far_right",
        "Normal::sf(8)",
        "x=8",
        &format!("{sf_8:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 8: Rapid sequential evaluations — stress test.
/// Evaluate PDF/CDF thousands of times to catch state corruption.
#[test]
fn e2e_008_rapid_sequential() {
    let scenario_id = "e2e_stats_008_rapid";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let n = Normal::standard();

    let t = Instant::now();
    let mut anomalies = 0usize;
    for i in 0..10_000 {
        let x = -5.0 + 10.0 * (i as f64 / 10_000.0);
        let cdf = n.cdf(x);
        let pdf = n.pdf(x);
        if !cdf.is_finite() || !pdf.is_finite() || !(0.0..=1.0).contains(&cdf) || pdf < 0.0 {
            anomalies += 1;
        }
    }
    let pass = anomalies == 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rapid_normal_eval",
        "10000 PDF+CDF evaluations",
        "x in [-5, 5]",
        &format!("anomalies={anomalies}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let e = Exponential::new(1.0);
    let mut anomalies = 0usize;
    for i in 0..10_000 {
        let x = 0.001 + 20.0 * (i as f64 / 10_000.0);
        let cdf = e.cdf(x);
        let pdf = e.pdf(x);
        if !cdf.is_finite() || !pdf.is_finite() || !(0.0..=1.0).contains(&cdf) || pdf < 0.0 {
            anomalies += 1;
        }
    }
    let pass = anomalies == 0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "rapid_exp_eval",
        "10000 PDF+CDF evaluations",
        "x in [0.001, 20]",
        &format!("anomalies={anomalies}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 9: Location-scale invariance.
/// Normal(mu, sigma).cdf(x) should equal Normal(0,1).cdf((x-mu)/sigma).
#[test]
fn e2e_009_location_scale_invariance() {
    let scenario_id = "e2e_stats_009_invariance";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let standard = Normal::standard();
    let shifted = Normal::new(5.0, 3.0);

    let t = Instant::now();
    let test_points = [-2.0, 0.0, 1.0, 5.0, 8.0, 11.0];
    let mut max_err = 0.0_f64;
    for &x in &test_points {
        let cdf_shifted = shifted.cdf(x);
        let z = (x - 5.0) / 3.0;
        let cdf_standard = standard.cdf(z);
        let err = (cdf_shifted - cdf_standard).abs();
        max_err = max_err.max(err);
    }
    let pass = max_err < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "loc_scale_cdf",
        "Normal(5,3).cdf(x) vs Normal(0,1).cdf((x-5)/3)",
        &format!("test_points={test_points:?}"),
        &format!("max_err={max_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 10: Maxwell distribution mode verification.
/// Mode of Maxwell(sigma) = sigma * sqrt(2).
#[test]
fn e2e_010_maxwell_mode() {
    let scenario_id = "e2e_stats_010_maxwell_mode";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let sigma = 2.0;
    let m = Maxwell::new(sigma);
    let expected_mode = sigma * 2.0_f64.sqrt();

    // Find mode numerically by scanning PDF
    let t = Instant::now();
    let mut max_pdf = 0.0_f64;
    let mut mode_x = 0.0;
    for i in 0..10_000 {
        let x = 0.001 + 10.0 * (i as f64 / 10_000.0);
        let p = m.pdf(x);
        if p > max_pdf {
            max_pdf = p;
            mode_x = x;
        }
    }
    let pass = (mode_x - expected_mode).abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "maxwell_mode",
        "scan Maxwell PDF for mode",
        &format!("sigma={sigma}"),
        &format!("mode={mode_x:.4}, expected={expected_mode:.4}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 11: Closed-form moments for newly remediated distributions.
#[test]
fn e2e_011_closed_form_moments() {
    let scenario_id = "e2e_stats_011_closed_form_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let burr = Burr12::new(2.0, 3.0);
    let pass = (burr.mean() - 0.589_048_622_548_086_1).abs() < 1e-12
        && (burr.var() - 0.153_021_720_274_202_2).abs() < 1e-12
        && Burr12::new(1.0, 2.0).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "burr12_closed_form_moments",
        "Burr12::mean/var",
        "c=2, d=3 and boundary c*d=2",
        &format!("mean={}, var={}", burr.mean(), burr.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let inv_weibull = InvWeibull::new(3.0);
    let pass = (inv_weibull.mean() - 1.354_117_939_426_400_2).abs() < 1e-12
        && (inv_weibull.var() - 0.845_303_140_831_346_9).abs() < 1e-10
        && InvWeibull::new(2.0).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "invweibull_closed_form_moments",
        "InvWeibull::mean/var",
        "c=3 and boundary c=2",
        &format!("mean={}, var={}", inv_weibull.mean(), inv_weibull.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let tukey = TukeyLambda::new(0.14);
    let pass = tukey.mean().abs() < TOL
        && (tukey.var() - 2.110_297_022_214_417).abs() < 1e-12
        && (TukeyLambda::new(0.0).var() - PI * PI / 3.0).abs() < 1e-12
        && TukeyLambda::new(-0.5).var().is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "tukey_lambda_closed_form_moments",
        "TukeyLambda::mean/var",
        "lam=0.14, logistic special case, undefined boundary",
        &format!("mean={}, var={}", tukey.mean(), tukey.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let gen_half = GenHalfLogistic::new(2.0);
    // Re-anchored to scipy.stats.genhalflogistic(c=2): mean = 2·ln 2 − 1,
    // var follows from numerical integration on x∈[0,1/c]. The earlier
    // 0.3068528 / 0.2337006 anchors came from the pre-fix (ψ(1/c+1)+γ)/c
    // formula (frankenscipy-viybt).
    let pass = (gen_half.mean() - 0.386_294_361_119_890_6).abs() < 1e-9
        && (gen_half.var() - 0.017_443_333_233_642_28).abs() < 1e-9
        && gen_half.pdf(0.1).is_finite()
        && (0.0..=1.0).contains(&gen_half.cdf(0.1));
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "gen_half_logistic_closed_form_moments",
        "GenHalfLogistic::mean/var",
        "c=2 with pdf/cdf sanity",
        &format!("mean={}, var={}", gen_half.mean(), gen_half.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 12: Special-function-backed moments for GenLogistic and Gompertz.
#[test]
fn e2e_012_special_function_moments() {
    let scenario_id = "e2e_stats_012_special_function_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let gen_logistic = GenLogistic::new(5.0);
    let pass = (gen_logistic.mean() - 2.083_333_333_333_333).abs() < 1e-9
        && (gen_logistic.var() - 1.866_257_022_585_341_7).abs() < 1e-9
        && (GenLogistic::new(1.0).mean() - Logistic::new(0.0, 1.0).mean()).abs() < TOL
        && (GenLogistic::new(1.0).var() - Logistic::new(0.0, 1.0).var()).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "gen_logistic_special_function_moments",
        "GenLogistic::mean/var",
        "c=5 and logistic identity at c=1",
        &format!("mean={}, var={}", gen_logistic.mean(), gen_logistic.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let gompertz = Gompertz::new(3.0);
    let pass = (gompertz.mean() - 0.262_083_740_252_362_75).abs() < 1e-10
        && (gompertz.var() - 0.046_960_406_809_075_47).abs() < 1e-10
        && gompertz.pdf(0.1).is_finite()
        && (0.0..=1.0).contains(&gompertz.cdf(0.1));
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "gompertz_special_function_moments",
        "Gompertz::mean/var",
        "c=3 with pdf/cdf sanity",
        &format!("mean={}, var={}", gompertz.mean(), gompertz.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let log_laplace = LogLaplace::new(3.0);
    let pass = (log_laplace.mean() - 1.125).abs() < TOL
        && (log_laplace.var() - 0.534_375).abs() < TOL
        && LogLaplace::new(1.5).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "log_laplace_moments",
        "LogLaplace::mean/var",
        "c=3 and divergent second moment at c=1.5",
        &format!("mean={}, var={}", log_laplace.mean(), log_laplace.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let mielke = Mielke::new(2.0, 2.5);
    let pass = (mielke.mean() - 1.174_450_160_620_581_7).abs() < 1e-12
        && (mielke.var() - 2.144_017_302_080_036_4).abs() < 1e-10
        && Mielke::new(3.5, 1.2).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mielke_moments",
        "Mielke::mean/var",
        "k=2, s=2.5 and divergent variance at s=1.2",
        &format!("mean={}, var={}", mielke.mean(), mielke.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let frechet_r = FrechetR::new(3.0);
    let pass = (frechet_r.mean() + 0.892_979_511_569_248_9).abs() < 1e-12
        && (frechet_r.var() - 0.105_332_884_868_479_14).abs() < 1e-12
        && frechet_r.cdf(-0.5).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "frechet_r_moments",
        "FrechetR::mean/var",
        "c=3 with support sanity",
        &format!("mean={}, var={}", frechet_r.mean(), frechet_r.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 13: Piecewise and numerical-integral moments for remaining stats distributions.
#[test]
fn e2e_013_piecewise_integral_moments() {
    let scenario_id = "e2e_stats_013_piecewise_integral_moments";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let crystal_ball = CrystalBall::new(2.0, 4.0);
    let pass = (crystal_ball.mean() + 0.053_285_264_688_121_33).abs() < 1e-10
        && (crystal_ball.var() - 1.281_348_758_903_764).abs() < 1e-10
        && CrystalBall::new(1.0, 2.0).mean().is_infinite()
        && CrystalBall::new(2.0, 3.0).var().is_infinite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "crystal_ball_piecewise_moments",
        "CrystalBall::mean/var",
        "finite m=4 case plus divergent guards",
        &format!("mean={}, var={}", crystal_ball.mean(), crystal_ball.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let argus = Argus::new(2.0);
    let pass = (argus.mean() - 0.705_658_515_503_037_3).abs() < 1e-8
        && (argus.var() - 0.044_467_693_721_274_79).abs() < 1e-8
        && argus.pdf(0.5).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "argus_integral_moments",
        "Argus::mean/var",
        "chi=2 bounded-support integral",
        &format!("mean={}, var={}", argus.mean(), argus.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let expon_weibull = ExponWeibull::new(2.0, 0.5);
    let pass = (expon_weibull.mean() - 3.499_999_999_874_901_4).abs() < 1e-6
        && (expon_weibull.var() - 34.250_000_000_307_45).abs() < 1e-5
        && expon_weibull.cdf(1.0).is_finite();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "expon_weibull_integral_moments",
        "ExponWeibull::mean/var",
        "a=2, c=0.5 quantile-integral moments",
        &format!("mean={}, var={}", expon_weibull.mean(), expon_weibull.var()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 14: TukeyLambda analytical pdf and Newton-backed cdf inversion.
#[test]
fn e2e_014_tukey_lambda_analytical() {
    let scenario_id = "e2e_stats_014_tukey_lambda_analytical";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let logistic = TukeyLambda::new(0.0);
    let tukey = TukeyLambda::new(0.5);
    let pdf_logistic_zero = logistic.pdf(0.0);
    let pdf_pos = tukey.pdf(1.0);
    let pdf_neg = tukey.pdf(-1.0);
    let pass = (pdf_logistic_zero - 0.25).abs() < 1e-12
        && (pdf_pos - pdf_neg).abs() < 1e-12
        && pdf_pos > 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "tukey_lambda_analytical_pdf",
        "TukeyLambda::pdf",
        "lam=0.0 at x=0 and lam=0.5 at +/-1",
        &format!(
            "logistic pdf(0)={pdf_logistic_zero}, pdf(1)={}, pdf(-1)={}",
            pdf_pos, pdf_neg
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    let logistic = TukeyLambda::new(0.0);
    let pass = quantiles
        .iter()
        .all(|&q| (logistic.cdf(logistic.ppf(q)) - q).abs() < 1e-10);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "tukey_lambda_newton_roundtrip",
        "TukeyLambda::cdf(ppf(q))",
        "lam=0.0, q in {0.01,0.1,0.25,0.5,0.75,0.9,0.99}",
        "all roundtrips within 1e-10",
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let bounded = TukeyLambda::new(2.0);
    let pass = bounded.cdf(-0.5) == 0.0
        && bounded.cdf(0.5) == 1.0
        && bounded.pdf(-1.0) == 0.0
        && bounded.pdf(1.0) == 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "tukey_lambda_support_bounds",
        "TukeyLambda support guard",
        "lam=2.0 at and beyond support edges",
        &format!(
            "cdf(-0.5)={}, cdf(0.5)={}, pdf(-1)={}, pdf(1)={}",
            bounded.cdf(-0.5),
            bounded.cdf(0.5),
            bounded.pdf(-1.0),
            bounded.pdf(1.0)
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

// ======================================================================
// HYPOTHESIS TESTING SCENARIOS (15-23)
// ======================================================================

/// Scenario 15: T-tests (ttest_1samp, ttest_ind, ttest_rel).
/// Verifies t-statistics and p-values for standard test cases.
#[test]
fn e2e_015_t_tests() {
    let scenario_id = "e2e_stats_015_ttests";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // One-sample t-test: sample from N(1, 1) tested against μ=0
    let t = Instant::now();
    let sample: Vec<f64> = vec![1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 0.95];
    let result = ttest_1samp(&sample, 0.0);
    // With mean ≈ 1.085 and n=10, t-statistic should be positive and significant
    let pass = result.statistic > 0.0 && result.pvalue < 0.05 && result.df == 9.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ttest_1samp",
        "ttest_1samp(sample, 0)",
        "10 samples from N(1,1), μ0=0",
        &format!(
            "t={:.4}, p={:.4}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Independent two-sample t-test: comparing two different populations
    let t = Instant::now();
    let group_a: Vec<f64> = vec![5.1, 5.3, 5.0, 4.9, 5.2, 5.4, 5.1, 5.0];
    let group_b: Vec<f64> = vec![4.2, 4.0, 4.3, 3.9, 4.1, 4.4, 4.0, 4.2];
    let result = ttest_ind(&group_a, &group_b);
    // Group A has higher mean, so statistic should be positive
    let pass = result.statistic > 0.0 && result.pvalue < 0.001 && result.df > 0.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ttest_ind",
        "ttest_ind(A, B)",
        "8 samples each, distinct means",
        &format!(
            "t={:.4}, p={:.6}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Paired t-test: before/after measurements
    let t = Instant::now();
    let before: Vec<f64> = vec![10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 12.0, 11.0];
    let after: Vec<f64> = vec![12.0, 14.0, 13.0, 15.0, 11.0, 15.0, 13.0, 12.0];
    let result = ttest_rel(&before, &after, None).expect("ttest_rel");
    // After values are higher, so difference (before - after) is negative, t < 0
    let pass = result.statistic < 0.0 && result.pvalue < 0.01 && result.df == 7.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ttest_rel",
        "ttest_rel(before, after)",
        "8 paired measurements",
        &format!(
            "t={:.4}, p={:.4}, df={}",
            result.statistic, result.pvalue, result.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Null hypothesis should not be rejected when data is from same distribution
    let t = Instant::now();
    let null_sample: Vec<f64> = vec![0.1, -0.2, 0.15, -0.05, 0.08, -0.1, 0.12, -0.08];
    let result = ttest_1samp(&null_sample, 0.0);
    // p-value should be high (not significant)
    let pass = result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "ttest_null_hypothesis",
        "ttest_1samp(centered, 0)",
        "8 samples centered around 0",
        &format!("t={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 16: ANOVA and variance tests (f_oneway, chisquare).
/// Verifies F-statistics and chi-square statistics.
#[test]
fn e2e_016_anova_chisquare() {
    let scenario_id = "e2e_stats_016_anova";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // One-way ANOVA: three groups with different means
    let t = Instant::now();
    let group1: [f64; 6] = [1.0, 1.2, 0.9, 1.1, 1.0, 0.8];
    let group2: [f64; 6] = [2.0, 2.1, 1.9, 2.2, 2.0, 1.8];
    let group3: [f64; 6] = [3.0, 3.2, 2.9, 3.1, 3.0, 2.8];
    let result = f_oneway(&[&group1, &group2, &group3]);
    // Groups are clearly different, F should be large and p small
    let pass = result.statistic > 50.0 && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "f_oneway",
        "f_oneway(g1, g2, g3)",
        "3 groups with means ~1, ~2, ~3",
        &format!("F={:.2}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square goodness of fit: observed vs expected uniform
    let t = Instant::now();
    let observed: [f64; 4] = [50.0, 45.0, 55.0, 50.0];
    let (chi2, pvalue) = chisquare(&observed, None);
    // Close to uniform, so chi2 should be small and p large
    let pass = chi2 < 5.0 && pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "chisquare_uniform",
        "chisquare(obs, None)",
        "observed=[50,45,55,50]",
        &format!("chi2={:.4}, p={:.4}", chi2, pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square with custom expected frequencies
    let t = Instant::now();
    let observed: [f64; 3] = [100.0, 50.0, 50.0];
    let expected: [f64; 3] = [100.0, 50.0, 50.0];
    let (chi2, pvalue) = chisquare(&observed, Some(&expected));
    // Perfect match, chi2 = 0
    let pass = chi2.abs() < TOL && pvalue > 0.999;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "chisquare_perfect",
        "chisquare(obs, exp)",
        "obs=exp=[100,50,50]",
        &format!("chi2={:.6}, p={:.4}", chi2, pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // ANOVA with no difference between groups
    let t = Instant::now();
    let g1: [f64; 5] = [5.0, 5.1, 4.9, 5.2, 4.8];
    let g2: [f64; 5] = [5.1, 4.9, 5.0, 5.2, 4.8];
    let g3: [f64; 5] = [4.9, 5.0, 5.1, 4.8, 5.2];
    let result = f_oneway(&[&g1, &g2, &g3]);
    // No real difference, p should be high
    let pass = result.pvalue > 0.5;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "f_oneway_null",
        "f_oneway(similar groups)",
        "3 groups with ~same mean",
        &format!("F={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 17: Non-parametric tests (mannwhitneyu, wilcoxon, kruskal).
/// Tests rank-based methods for comparing distributions.
#[test]
fn e2e_017_nonparametric_tests() {
    let scenario_id = "e2e_stats_017_nonparametric";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Mann-Whitney U: comparing two groups with shift
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let result = mannwhitneyu(&x, &y);
    // Clear separation, U should be 0 (all x < all y) and p very small
    let pass = result.statistic < 5.0 && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "mannwhitneyu_distinct",
        "mannwhitneyu(x, y)",
        "x=[1..8], y=[9..16]",
        &format!("U={:.2}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Wilcoxon signed-rank: paired comparison (needs n >= 10)
    let t = Instant::now();
    let before: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let after: Vec<f64> = vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0];
    let result = wilcoxon(&before, &after);
    // All after > before, so significant (with enough pairs)
    let pass = result.pvalue < 0.2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "wilcoxon_paired",
        "wilcoxon(before, after)",
        "10 pairs, all show increase",
        &format!("statistic={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kruskal-Wallis: non-parametric ANOVA
    let t = Instant::now();
    let g1: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let g2: [f64; 5] = [6.0, 7.0, 8.0, 9.0, 10.0];
    let g3: [f64; 5] = [11.0, 12.0, 13.0, 14.0, 15.0];
    let result = kruskal(&[&g1, &g2, &g3]);
    // Clear separation, H should be large
    let pass = result.statistic > 10.0 && result.pvalue < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "kruskal_distinct",
        "kruskal(g1, g2, g3)",
        "3 non-overlapping groups",
        &format!("H={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Mann-Whitney with overlapping groups
    let t = Instant::now();
    let a: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let b: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = mannwhitneyu(&a, &b);
    // Interleaved, no significant difference expected
    let pass = result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mannwhitneyu_overlap",
        "mannwhitneyu(odd, even)",
        "interleaved sequences",
        &format!("U={:.2}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 18: Correlation tests (pearsonr, spearmanr, kendalltau).
/// Verifies correlation coefficients for known relationships.
#[test]
fn e2e_018_correlation_tests() {
    let scenario_id = "e2e_stats_018_correlation";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Perfect positive linear correlation
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let result = pearsonr(&x, &y);
    let pass = (result.statistic - 1.0).abs() < TOL && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "pearsonr_perfect",
        "pearsonr(x, 2x)",
        "perfect linear",
        &format!("r={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Spearman rank correlation: monotonic but non-linear
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]; // y = x^2
    let result = spearmanr(&x, &y);
    // Perfect monotonic relationship
    let pass = (result.statistic - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "spearmanr_monotonic",
        "spearmanr(x, x^2)",
        "quadratic but monotonic",
        &format!("rho={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kendall tau: concordance measure
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // perfectly negative
    let result = kendalltau(&x, &y);
    let pass = (result.statistic + 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "kendalltau_negative",
        "kendalltau(asc, desc)",
        "perfectly negative",
        &format!("tau={:.6}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // No correlation
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 7.0, 4.0]; // scrambled
    let result = pearsonr(&x, &y);
    // Low correlation expected
    let pass = result.statistic.abs() < 0.5 && result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "pearsonr_uncorrelated",
        "pearsonr(x, scrambled)",
        "no linear relationship",
        &format!("r={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 19: Linear regression (linregress).
/// Verifies slope, intercept, and regression statistics.
#[test]
fn e2e_019_linear_regression() {
    let scenario_id = "e2e_stats_019_regression";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Perfect linear fit: y = 2x + 3
    let t = Instant::now();
    let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    let result = linregress(&x, &y);
    let pass = (result.slope - 2.0).abs() < TOL
        && (result.intercept - 3.0).abs() < TOL
        && (result.rvalue - 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "linregress_perfect",
        "linregress(x, 2x+3)",
        "perfect linear",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}",
            result.slope, result.intercept, result.rvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Regression with noise
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = vec![2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9];
    let result = linregress(&x, &y);
    // Slope should be close to 2, intercept close to 0
    let pass = (result.slope - 2.0).abs() < 0.1
        && result.intercept.abs() < 0.5
        && result.rvalue > 0.99
        && result.pvalue < 0.001;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "linregress_noisy",
        "linregress with noise",
        "y ≈ 2x with small noise",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}, p={:.6}",
            result.slope, result.intercept, result.rvalue, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Negative slope
    let t = Instant::now();
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let result = linregress(&x, &y);
    let pass = (result.slope + 2.0).abs() < TOL
        && (result.intercept - 12.0).abs() < TOL
        && (result.rvalue + 1.0).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "linregress_negative",
        "linregress(x, -2x+12)",
        "negative slope",
        &format!(
            "slope={:.4}, intercept={:.4}, r={:.4}",
            result.slope, result.intercept, result.rvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Stderr should be small for good fit
    let t = Instant::now();
    let pass = result.stderr < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "linregress_stderr",
        "check stderr",
        "stderr for perfect fit",
        &format!("stderr={:.6}", result.stderr),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 20: Descriptive statistics (describe, skew, kurtosis).
/// Verifies summary statistics computation.
#[test]
fn e2e_020_descriptive_stats() {
    let scenario_id = "e2e_stats_020_descriptive";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Basic describe
    let t = Instant::now();
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = describe(&data);
    let expected_mean = 5.5;
    let expected_var = 9.166_666_666_666_666; // sample variance with ddof=1
    let pass = result.nobs == 10
        && (result.mean - expected_mean).abs() < TOL
        && (result.variance - expected_var).abs() < 0.001
        && result.minmax == (1.0, 10.0);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "describe_basic",
        "describe([1..10])",
        "n=10, uniform spacing",
        &format!(
            "n={}, mean={:.4}, var={:.4}, minmax={:?}",
            result.nobs, result.mean, result.variance, result.minmax
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Skewness: symmetric distribution should have skew ≈ 0
    let t = Instant::now();
    let symmetric: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let sk = skew(&symmetric);
    let pass = sk.abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "skew_symmetric",
        "skew(symmetric)",
        "palindromic data",
        &format!("skew={:.6}", sk),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Skewness: right-skewed distribution
    let t = Instant::now();
    let right_skewed: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 10.0];
    let sk = skew(&right_skewed);
    let pass = sk > 1.0; // positive skew
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "skew_right",
        "skew(right-skewed)",
        "long right tail",
        &format!("skew={:.4}", sk),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Kurtosis: normal distribution should have excess kurtosis ≈ 0
    // Heavy-tailed should have positive kurtosis
    let t = Instant::now();
    let heavy_tailed: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let kurt = kurtosis(&heavy_tailed);
    let pass = kurt > 3.0; // heavy tails -> high kurtosis
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "kurtosis_heavy",
        "kurtosis(heavy-tailed)",
        "one extreme outlier",
        &format!("kurtosis={:.4}", kurt),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 21: Normality tests (shapiro, normaltest, jarque_bera).
/// Verifies detection of normal vs non-normal distributions.
#[test]
fn e2e_021_normality_tests() {
    let scenario_id = "e2e_stats_021_normality";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Shapiro-Wilk: uniform data should fail normality
    let t = Instant::now();
    let uniform_data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let result = shapiro(&uniform_data);
    // Uniform should reject normality (low p-value or W < 1)
    let pass = result.statistic < 0.98 || result.pvalue < 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "shapiro_uniform",
        "shapiro(uniform)",
        "50 evenly spaced points",
        &format!("W={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // D'Agostino-Pearson normaltest: bimodal should fail
    let t = Instant::now();
    let mut bimodal: Vec<f64> = vec![0.0; 25];
    bimodal.extend([10.0; 25]);
    let result = normaltest(&bimodal);
    // Bimodal should reject normality
    let pass = result.pvalue < 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "normaltest_bimodal",
        "normaltest(bimodal)",
        "25 at 0, 25 at 10",
        &format!("statistic={:.4}, p={:.6}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Jarque-Bera: check skewness and kurtosis
    let t = Instant::now();
    let skewed: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 100.0];
    let result = jarque_bera(&skewed);
    // Highly skewed should reject normality
    let pass = result.statistic > 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "jarque_bera_skewed",
        "jarque_bera(skewed)",
        "right-skewed data",
        &format!("JB={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Shapiro should give high p-value for nearly-normal data
    let t = Instant::now();
    // Approximate normal samples (actually generated, but pretend they're from N(0,1))
    let normal_ish: Vec<f64> = vec![-1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 1.5];
    let result = shapiro(&normal_ish);
    // Should not reject normality
    let pass = result.pvalue > 0.05;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "shapiro_normal",
        "shapiro(normal-like)",
        "roughly bell-shaped",
        &format!("W={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 22: Kolmogorov-Smirnov tests (ks_1samp, ks_2samp).
/// Verifies goodness-of-fit testing.
#[test]
fn e2e_022_ks_tests() {
    let scenario_id = "e2e_stats_022_ks";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // KS one-sample: test against uniform CDF
    let t = Instant::now();
    let uniform_samples: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
    let result = ks_1samp(&uniform_samples, |x| x.clamp(0.0, 1.0)); // uniform CDF
    // Should match well (small D, high p)
    let pass = result.statistic < 0.1 && result.pvalue > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ks_1samp_uniform",
        "ks_1samp(uniform, F_uniform)",
        "100 uniform samples",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS one-sample: non-uniform data against uniform CDF
    let t = Instant::now();
    let skewed_samples: Vec<f64> = (0..50).map(|i| (i as f64 / 50.0).powi(2)).collect();
    let result = ks_1samp(&skewed_samples, |x| x.clamp(0.0, 1.0));
    // Should reject (large D, small p)
    let pass = result.statistic > 0.1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ks_1samp_reject",
        "ks_1samp(x^2, F_uniform)",
        "squared samples vs uniform",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS two-sample: same distribution
    let t = Instant::now();
    let sample1: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let sample2: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();
    let result = ks_2samp(&sample1, &sample2);
    // Similar distributions, small D
    let pass = result.statistic < 0.2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ks_2samp_similar",
        "ks_2samp(shifted)",
        "same shape, slight offset",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // KS two-sample: different distributions
    let t = Instant::now();
    let normal_like: Vec<f64> = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let uniform_like: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = ks_2samp(&normal_like, &uniform_like);
    // Different distributions, should have larger D
    let pass = result.statistic > 0.3;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "ks_2samp_different",
        "ks_2samp(normal, uniform)",
        "different distributions",
        &format!("D={:.4}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 23: Contingency table tests (fisher_exact, chi2_contingency).
/// Verifies tests for independence in categorical data.
#[test]
fn e2e_023_contingency_tests() {
    let scenario_id = "e2e_stats_023_contingency";
    let mut steps = Vec::new();
    let mut all_pass = true;

    // Fisher's exact test: 2x2 table with strong association
    let t = Instant::now();
    let table = [[10.0, 2.0], [1.0, 12.0]];
    let result = fisher_exact(&table);
    // Strong association, p should be small
    let pass = result.pvalue < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "fisher_exact_assoc",
        "fisher_exact([[10,2],[1,12]])",
        "strong association",
        &format!(
            "odds_ratio={:.4}, p={:.6}",
            result.odds_ratio, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Fisher's exact test: independent (null)
    let t = Instant::now();
    let table = [[5.0, 5.0], [5.0, 5.0]];
    let result = fisher_exact(&table);
    // No association, p should be high
    let pass = result.pvalue > 0.5 && (result.odds_ratio - 1.0).abs() < 0.01;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "fisher_exact_null",
        "fisher_exact([[5,5],[5,5]])",
        "no association",
        &format!(
            "odds_ratio={:.4}, p={:.4}",
            result.odds_ratio, result.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square test of independence: larger table
    let t = Instant::now();
    let observed = vec![vec![20.0, 30.0, 50.0], vec![30.0, 40.0, 30.0]];
    let result = chi2_contingency(&observed, true);
    // Some deviation from independence expected
    let pass = result.statistic > 0.0 && result.dof == 2;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "chi2_contingency",
        "chi2_contingency(2x3, correction=True)",
        "2x3 contingency table",
        &format!(
            "chi2={:.4}, p={:.4}, dof={}",
            result.statistic, result.pvalue, result.dof
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    // Chi-square: perfectly proportional (independence)
    let t = Instant::now();
    let observed = vec![vec![10.0, 20.0, 30.0], vec![10.0, 20.0, 30.0]];
    let result = chi2_contingency(&observed, true);
    // Perfect independence, chi2 ≈ 0
    let pass = result.statistic < 0.01 && result.pvalue > 0.99;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "chi2_independent",
        "chi2_contingency(proportional, correction=True)",
        "perfectly proportional rows",
        &format!("chi2={:.6}, p={:.4}", result.statistic, result.pvalue),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 24: Filliben order-statistic medians for probplot.
/// Verifies SciPy-compatible endpoint medians, interior Filliben spacing,
/// and normal symmetry for odd sample sizes.
#[test]
fn e2e_024_probplot_filliben_quantiles() {
    let scenario_id = "e2e_stats_024_probplot_filliben";
    let mut steps = Vec::new();
    let mut all_pass = true;
    let normal = Normal::standard();

    let t = Instant::now();
    let singleton = probplot_quantiles(1);
    let pass = singleton.len() == 1 && singleton[0].abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "probplot_singleton",
        "probplot_quantiles(1)",
        "sample size n=1",
        &format!("quantiles={singleton:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let quantiles = probplot_quantiles(4);
    let probabilities: Vec<f64> = quantiles.iter().map(|&value| normal.cdf(value)).collect();
    let expected = [0.159_103_58, 0.385_452_46, 0.614_547_54, 0.840_896_42];
    let max_err = probabilities
        .iter()
        .zip(&expected)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0_f64, f64::max);
    let pass = max_err < 2e-8;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "probplot_filliben_probabilities",
        "Normal::cdf(probplot_quantiles(4))",
        "n=4 Filliben reference probabilities",
        &format!("probabilities={probabilities:?}, max_err={max_err:.2e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let quantiles = probplot_quantiles(5);
    let endpoint_err = (quantiles[0] + quantiles[4]).abs();
    let interior_err = (quantiles[1] + quantiles[3]).abs();
    let center_err = quantiles[2].abs();
    let pass = endpoint_err < 1e-12 && interior_err < 1e-12 && center_err < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "probplot_normal_symmetry",
        "probplot_quantiles(5)",
        "odd n should preserve normal symmetry",
        &format!(
            "quantiles={quantiles:?}, endpoint_err={endpoint_err:.2e}, interior_err={interior_err:.2e}, center_err={center_err:.2e}"
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 25: Robust helper edge semantics.
/// Verifies SciPy-compatible saturation and sentinel behavior for `winsorize`,
/// `zmap`, and `gzscore`.
#[test]
fn e2e_025_robust_helper_edge_semantics() {
    let scenario_id = "e2e_stats_025_robust_helper_edges";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let saturated = winsorize(&[1.0, 2.0, 3.0], (0.8, 0.8));
    let pass = saturated == vec![3.0, 3.0, 3.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "winsorize_overlapping_limits",
        "winsorize([1,2,3], (0.8, 0.8))",
        "overlapping clip windows",
        &format!("result={saturated:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let fail_closed = winsorize(&[1.0, 2.0, 3.0], (0.0, 1.1));
    let pass = fail_closed == vec![1.0, 1.0, 1.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "winsorize_out_of_range_limits",
        "winsorize([1,2,3], (0.0, 1.1))",
        "invalid upper limit should fail closed",
        &format!("result={fail_closed:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let constant_compare = zmap(&[1.0, 2.0], &[1.0, 1.0]);
    let same_object = zmap(&[1.0, 1.0], &[1.0, 1.0]);
    let pass = constant_compare[0].is_nan()
        && constant_compare[1].is_infinite()
        && constant_compare[1].is_sign_positive()
        && same_object.iter().all(|&value| value.is_nan());
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "zmap_constant_reference_shape",
        "zmap(scores, compare)",
        "constant compare vector and identical constant vectors",
        &format!("constant_compare={constant_compare:?}, same_object={same_object:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let zero = gzscore(&[0.0, 1.0, 2.0]);
    let negative = gzscore(&[-1.0, 1.0, 2.0]);
    let pass =
        zero.iter().all(|&value| value.is_nan()) && negative.iter().all(|&value| value.is_nan());
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "gzscore_invalid_inputs",
        "gzscore(a)",
        "zero and negative values should propagate NaN shape",
        &format!("zero={zero:?}, negative={negative:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 26: Descriptive helper contracts.
/// Verifies SciPy-shaped semantics for scaled MAD, mode tie-breaking, and
/// uniform expected frequencies.
#[test]
fn e2e_026_descriptive_helper_contracts() {
    let scenario_id = "e2e_stats_026_descriptive_helper_contracts";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let data = [1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
    let mad = median_abs_deviation(&data, 1.4826);
    // Updated for [frankenscipy-xxfuo]: scipy.stats.median_abs_deviation
    // DIVIDES the raw MAD by `scale`, not multiplies. Raw MAD here is
    // 1.0 (median is 2.0; |x-2| sorted = [0,0,0,1,2,4,7]; median = 1.0);
    // scaled = 1.0 / 1.4826 ≈ 0.67450 (normal-consistent std estimator).
    let expected = 1.0 / 1.4826;
    let pass = (mad - expected).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "median_abs_deviation_scaled_sample",
        "median_abs_deviation([1,1,2,2,4,6,9], 1.4826)",
        "sample with unit unscaled MAD and normal-consistency scale",
        &format!("mad={mad:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let constant = median_abs_deviation(&[5.0, 5.0, 5.0, 5.0], 1.0);
    let pass = constant.abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "median_abs_deviation_constant_sample",
        "median_abs_deviation([5,5,5,5], 1.0)",
        "constant sample should have zero dispersion",
        &format!("mad={constant:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let tie_mode = mode(&[3.0, 1.0, 2.0, 3.0, 2.0]);
    let pass = (tie_mode - 2.0).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "mode_prefers_smallest_tied_value",
        "mode([3,1,2,3,2])",
        "equal-frequency tie should resolve to the smallest value",
        &format!("mode={tie_mode:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let expected = expected_freq_uniform(&[2.0, 4.0, 6.0, 8.0]);
    let pass = expected == vec![5.0, 5.0, 5.0, 5.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "expected_freq_uniform_preserves_total",
        "expected_freq_uniform([2,4,6,8])",
        "uniform expected bins should evenly divide the observed total",
        &format!("expected={expected:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 27: Regression metric helper contracts.
/// Verifies deterministic 1D metric outputs against the documented sklearn-like
/// helper formulas in `fsci_stats`.
#[test]
fn e2e_027_regression_metric_helper_contracts() {
    let scenario_id = "e2e_stats_027_regression_metric_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let perfect_true = [1.0, 2.0, 3.0];
    let perfect_pred = [1.0, 2.0, 3.0];
    let r2 = r2_score(&perfect_true, &perfect_pred);
    let mae = mean_absolute_error(&perfect_true, &perfect_pred);
    let mse = mean_squared_error(&perfect_true, &perfect_pred);
    let rmse = root_mean_squared_error(&perfect_true, &perfect_pred);
    let pass =
        (r2 - 1.0).abs() < 1e-12 && mae.abs() < 1e-12 && mse.abs() < 1e-12 && rmse.abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "regression_metrics_perfect_fit",
        "metrics([1,2,3], [1,2,3])",
        "perfect predictions should yield ideal scores",
        &format!("r2={r2:.12}, mae={mae:.12}, mse={mse:.12}, rmse={rmse:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let sample_true = [3.0, -0.5, 2.0, 7.0];
    let sample_pred = [2.5, 0.0, 2.0, 8.0];
    let r2 = r2_score(&sample_true, &sample_pred);
    let mae = mean_absolute_error(&sample_true, &sample_pred);
    let mse = mean_squared_error(&sample_true, &sample_pred);
    let rmse = root_mean_squared_error(&sample_true, &sample_pred);
    let pass = (r2 - 0.948_608_137_044_967_9).abs() < TOL
        && (mae - 0.5).abs() < 1e-12
        && (mse - 0.375).abs() < 1e-12
        && (rmse - 0.612_372_435_695_794_5).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "regression_metrics_reference_example",
        "metrics([3,-0.5,2,7], [2.5,0,2,8])",
        "reference 1D regression example with non-zero residuals",
        &format!("r2={r2:.12}, mae={mae:.12}, mse={mse:.12}, rmse={rmse:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let constant_true = [1.0, 1.0, 1.0];
    let constant_pred = [1.0, 2.0, 1.0];
    let r2 = r2_score(&constant_true, &constant_pred);
    let pass = r2.abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "r2_constant_target_sentinel",
        "r2_score([1,1,1], [1,2,1])",
        "non-perfect constant targets should return the 0.0 sentinel",
        &format!("r2={r2:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let mape = mean_absolute_percentage_error(&[0.0, 10.0], &[5.0, 8.0]);
    let pass = (mape - 0.1).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mape_zero_target_fail_closed",
        "mean_absolute_percentage_error([0,10], [5,8])",
        "zero targets should contribute 0.0 instead of dividing by zero",
        &format!("mape={mape:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 28: Multiple-testing helper contracts.
/// Verifies SciPy-shaped false-discovery-control outputs and wrapper alignment
/// for the Bonferroni, Holm, and Benjamini-Hochberg helpers.
#[test]
fn e2e_028_multiple_testing_helper_contracts() {
    let scenario_id = "e2e_stats_028_multiple_testing_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let pvalues = [0.01, 0.04, 0.03, 0.005];

    let t = Instant::now();
    let bh = false_discovery_control(&pvalues, None).expect("default bh");
    let expected_bh = [0.02, 0.04, 0.04, 0.02];
    let pass = bh
        .iter()
        .zip(expected_bh.iter())
        .all(|(&actual, &expected)| (actual - expected).abs() < TOL);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "false_discovery_control_bh_reference_vector",
        "false_discovery_control([0.01,0.04,0.03,0.005], None)",
        "default method should match SciPy's Benjamini-Hochberg adjusted p-values",
        &format!("corrected={bh:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let by = false_discovery_control(&pvalues, Some("by")).expect("by");
    let expected_by = [
        0.041_666_666_666_666_664,
        0.083_333_333_333_333_33,
        0.083_333_333_333_333_31,
        0.041_666_666_666_666_664,
    ];
    let pass = by.iter().zip(expected_by.iter()).zip(bh.iter()).all(
        |((&actual, &expected), &bh_value)| {
            (actual - expected).abs() < TOL && actual + TOL >= bh_value
        },
    );
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "false_discovery_control_by_reference_vector",
        "false_discovery_control([0.01,0.04,0.03,0.005], Some(\"by\"))",
        "BY should match SciPy's adjusted vector and remain at least as conservative as BH",
        &format!("corrected={by:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let bonferroni = multipletests_bonferroni(&pvalues, 0.05);
    let expected_corrected = [0.04, 0.16, 0.12, 0.02];
    let expected_reject = [true, false, false, true];
    let pass = bonferroni
        .pvalues_corrected
        .iter()
        .zip(expected_corrected.iter())
        .all(|(&actual, &expected)| (actual - expected).abs() < TOL)
        && bonferroni.reject.as_slice() == expected_reject.as_slice();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "multipletests_bonferroni_reference_vector",
        "multipletests_bonferroni([0.01,0.04,0.03,0.005], 0.05)",
        "Bonferroni should scale each p-value by the test count and derive the matching reject mask",
        &format!(
            "corrected={:?}, reject={:?}",
            bonferroni.pvalues_corrected, bonferroni.reject
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let holm = multipletests_holm(&pvalues, 0.05);
    let expected_corrected = [0.03, 0.06, 0.06, 0.02];
    let expected_reject = [true, false, false, true];
    let pass = holm
        .pvalues_corrected
        .iter()
        .zip(expected_corrected.iter())
        .all(|(&actual, &expected)| (actual - expected).abs() < TOL)
        && holm.reject.as_slice() == expected_reject.as_slice();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "multipletests_holm_reference_vector",
        "multipletests_holm([0.01,0.04,0.03,0.005], 0.05)",
        "Holm should enforce monotone step-down corrected p-values with the matching reject mask",
        &format!(
            "corrected={:?}, reject={:?}",
            holm.pvalues_corrected, holm.reject
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let fdr_bh = multipletests_fdr_bh(&pvalues, 0.05);
    let invalid = false_discovery_control(&[0.01, 0.02], Some("unknown"));
    let expected_corrected = [0.02, 0.04, 0.04, 0.02];
    let expected_reject = [true, true, true, true];
    let pass = fdr_bh
        .pvalues_corrected
        .iter()
        .zip(expected_corrected.iter())
        .all(|(&actual, &expected)| (actual - expected).abs() < TOL)
        && fdr_bh.reject.as_slice() == expected_reject.as_slice()
        && invalid.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "multipletests_fdr_bh_alignment_and_fail_closed_invalid_method",
        "multipletests_fdr_bh([0.01,0.04,0.03,0.005], 0.05)",
        "FDR-BH helper should align with the BH-adjusted vector and invalid methods should fail closed",
        &format!(
            "corrected={:?}, reject={:?}, invalid_method_error={}",
            fdr_bh.pvalues_corrected,
            fdr_bh.reject,
            invalid.is_err()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 29: Combined p-value helper contracts.
/// Verifies SciPy-shaped `combine_pvalues` outputs for Fisher, Pearson,
/// Tippett, weighted Stouffer, and Mudholkar-George methods, plus fail-closed
/// invalid-method handling.
#[test]
fn e2e_029_combine_pvalues_helper_contracts() {
    let scenario_id = "e2e_stats_029_combine_pvalues_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let fisher = combine_pvalues(&[0.01, 0.03, 0.2], None, None).expect("default fisher");
    let pass = (fisher.statistic - 19.442_331_991_484_345).abs() < TOL
        && (fisher.pvalue - 0.003_478_302_009_247_749_2).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "combine_pvalues_default_fisher_reference",
        "combine_pvalues([0.01,0.03,0.2], None, None)",
        "default method should match SciPy's Fisher statistic and p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            fisher.statistic, fisher.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let pearson = combine_pvalues(&[0.01, 0.03, 0.2], Some("pearson"), None).expect("pearson");
    let pass = (pearson.statistic + 0.527_306_189_304_839_5).abs() < TOL
        && (pearson.pvalue - 0.002_509_830_550_904_323_7).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "combine_pvalues_pearson_reference",
        "combine_pvalues([0.01,0.03,0.2], Some(\"pearson\"), None)",
        "Pearson mode should match SciPy's statistic sign and lower-tail p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            pearson.statistic, pearson.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let tippett = combine_pvalues(&[0.4, 0.03, 0.8, 0.2], Some("tippett"), None).expect("tippett");
    let pass =
        (tippett.statistic - 0.03).abs() < 1e-12 && (tippett.pvalue - 0.114_707_19).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "combine_pvalues_tippett_reference",
        "combine_pvalues([0.4,0.03,0.8,0.2], Some(\"tippett\"), None)",
        "Tippett mode should use the smallest p-value and SciPy's combined tail probability",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            tippett.statistic, tippett.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let stouffer = combine_pvalues(&[0.01, 0.05, 0.2], Some("stouffer"), Some(&[2.0, 1.0, 0.5]))
        .expect("stouffer");
    let pass = (stouffer.statistic - 2.932_132_686_521_549_6).abs() < TOL
        && (stouffer.pvalue - 0.001_683_214_419_543_920_8).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "combine_pvalues_weighted_stouffer_reference",
        "combine_pvalues([0.01,0.05,0.2], Some(\"stouffer\"), Some([2.0,1.0,0.5]))",
        "weighted Stouffer mode should match SciPy's z-score aggregation and p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            stouffer.statistic, stouffer.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let mudholkar_george = combine_pvalues(&[0.01, 0.03, 0.2], Some("mudholkar_george"), None)
        .expect("mudholkar_george");
    // Updated for [frankenscipy-v51r8]: fsci's mudholkar_george now
    // matches scipy's Student-t form T(5n+4).sf(stat·sqrt(3/n)/π·sqrt(nu/(nu-2))).
    // Previous pinned value 0.04695... was the old Logistic-tail
    // approximation; scipy reference is 0.00245...
    let pass = (mudholkar_george.statistic - 9.457_512_901_089_753).abs() < TOL
        && (mudholkar_george.pvalue - 0.002_450_914_441_120_641).abs() < 1e-7;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "combine_pvalues_mudholkar_george_reference",
        "combine_pvalues([0.01,0.03,0.2], Some(\"mudholkar_george\"), None)",
        "Mudholkar-George mode should match SciPy-shaped logistic-normalized logit aggregation",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            mudholkar_george.statistic, mudholkar_george.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let invalid = combine_pvalues(&[0.01, 0.02], Some("unknown"), None);
    let pass = invalid.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        6,
        "combine_pvalues_invalid_method_fail_closed",
        "combine_pvalues([0.01,0.02], Some(\"unknown\"), None)",
        "unsupported methods should fail closed rather than silently selecting a fallback",
        &format!("invalid_method_error={}", invalid.is_err()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 30: Poisson means test helper contracts.
/// Verifies SciPy-shaped `poisson_means_test` outputs for two-sided,
/// one-sided, and nonzero-diff calls, plus the degenerate unit-pvalue path
/// and fail-closed invalid-alternative handling.
#[test]
fn e2e_030_poisson_means_test_helper_contracts() {
    let scenario_id = "e2e_stats_030_poisson_means_test_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let two_sided =
        poisson_means_test(0, 100.0, 3, 100.0, 0.0, None).expect("two-sided poisson means");
    let pass = (two_sided.statistic + 1.732_050_807_568_877_2).abs() < TOL
        && (two_sided.pvalue - 0.088_379_009_454_835_18).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "poisson_means_test_two_sided_reference",
        "poisson_means_test(0, 100.0, 3, 100.0, 0.0, None)",
        "two-sided poisson means test should match SciPy's reference statistic and p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            two_sided.statistic, two_sided.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let less =
        poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some("less")).expect("less poisson means");
    let pass = (less.statistic + 1.732_050_807_568_877_2).abs() < TOL
        && (less.pvalue - 0.044_189_504_727_417_59).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "poisson_means_test_less_reference",
        "poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some(\"less\"))",
        "the less alternative should preserve SciPy's test statistic and halve the two-sided tail",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            less.statistic, less.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let greater = poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some("greater"))
        .expect("greater poisson means");
    let pass = (greater.statistic + 1.732_050_807_568_877_2).abs() < TOL
        && (greater.pvalue - 0.934_031_619_723_871_6).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "poisson_means_test_greater_reference",
        "poisson_means_test(0, 100.0, 3, 100.0, 0.0, Some(\"greater\"))",
        "the greater alternative should match SciPy's upper-tail p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            greater.statistic, greater.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let nonzero_diff =
        poisson_means_test(10, 100.0, 5, 80.0, 0.02, None).expect("nonzero-diff poisson means");
    let pass = (nonzero_diff.statistic - 0.414_644_214_431_364_8).abs() < TOL
        && (nonzero_diff.pvalue - 0.683_267_124_060_036_6).abs() < 5e-7;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "poisson_means_test_nonzero_diff_reference",
        "poisson_means_test(10, 100.0, 5, 80.0, 0.02, None)",
        "nonzero null differences should match SciPy's E-test statistic and p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            nonzero_diff.statistic, nonzero_diff.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let degenerate =
        poisson_means_test(1, 1.0, 0, 1.0, 10.0, None).expect("degenerate poisson means");
    let pass = degenerate.statistic == 0.0 && degenerate.pvalue == 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "poisson_means_test_lambda_hat2_nonpositive_unit_pvalue",
        "poisson_means_test(1, 1.0, 0, 1.0, 10.0, None)",
        "when SciPy's intermediate lambda_hat2 is nonpositive the helper should return the unit p-value sentinel",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            degenerate.statistic, degenerate.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let invalid = poisson_means_test(1, 1.0, 0, 1.0, 0.0, Some("sideways"));
    let pass = invalid.is_err();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        6,
        "poisson_means_test_invalid_alternative_fail_closed",
        "poisson_means_test(1, 1.0, 0, 1.0, 0.0, Some(\"sideways\"))",
        "unsupported alternatives should fail closed rather than silently changing the hypothesis test",
        &format!("invalid_alternative_error={}", invalid.is_err()),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 31: Power divergence helper contracts.
/// Verifies SciPy-shaped `power_divergence` outputs for Pearson chi-squared,
/// the G-test, custom expected frequencies, and fail-closed unequal-total
/// handling.
#[test]
fn e2e_031_power_divergence_helper_contracts() {
    let scenario_id = "e2e_stats_031_power_divergence_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let (uniform_stat, uniform_pvalue) = power_divergence(&[25.0, 25.0, 25.0, 25.0], None, 1.0);
    let pass = uniform_stat == 0.0 && uniform_pvalue == 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "power_divergence_pearson_uniform_reference",
        "power_divergence([25.0,25.0,25.0,25.0], None, 1.0)",
        "uniform observed frequencies should match SciPy's zero Pearson statistic and unit p-value",
        &format!("statistic={uniform_stat:.12}, pvalue={uniform_pvalue:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let (pearson_stat, pearson_pvalue) = power_divergence(&[50.0, 10.0, 10.0, 10.0], None, 1.0);
    let pass = (pearson_stat - 60.0).abs() < TOL
        && (pearson_pvalue - 5.878_230_727_906_921e-13).abs() < 1e-18;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "power_divergence_pearson_skewed_reference",
        "power_divergence([50.0,10.0,10.0,10.0], None, 1.0)",
        "Pearson mode should match SciPy's chi-squared statistic and tail probability for a skewed multinomial sample",
        &format!("statistic={pearson_stat:.12}, pvalue={pearson_pvalue:.18e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let (gtest_stat, gtest_pvalue) = power_divergence(&[50.0, 10.0, 10.0, 10.0], None, 0.0);
    let pass = (gtest_stat - 50.040_242_353_818_8).abs() < TOL
        && (gtest_pvalue - 7.833_065_358_568_862e-11).abs() < 1e-16;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "power_divergence_gtest_reference",
        "power_divergence([50.0,10.0,10.0,10.0], None, 0.0)",
        "lambda=0 should reproduce SciPy's log-likelihood-ratio statistic and p-value",
        &format!("statistic={gtest_stat:.12}, pvalue={gtest_pvalue:.18e}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let (custom_stat, custom_pvalue) =
        power_divergence(&[10.0, 20.0, 30.0], Some(&[20.0, 20.0, 20.0]), 1.0);
    let pass =
        (custom_stat - 10.0).abs() < TOL && (custom_pvalue - 0.006_737_946_999_085_468).abs() < TOL;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "power_divergence_custom_expected_reference",
        "power_divergence([10.0,20.0,30.0], Some([20.0,20.0,20.0]), 1.0)",
        "custom expected frequencies should match SciPy's Pearson statistic and p-value",
        &format!("statistic={custom_stat:.12}, pvalue={custom_pvalue:.12}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let (nan_stat, nan_pvalue) = power_divergence(&[10.0, 20.0], Some(&[20.0, 20.0]), 1.0);
    let pass = nan_stat.is_nan() && nan_pvalue.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "power_divergence_unequal_totals_fail_closed",
        "power_divergence([10.0,20.0], Some([20.0,20.0]), 1.0)",
        "mismatched observed and expected totals should fail closed with NaN sentinels instead of producing a misleading statistic",
        &format!(
            "statistic_is_nan={}, pvalue_is_nan={}",
            nan_stat.is_nan(),
            nan_pvalue.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 32: Median test helper contracts.
/// Verifies SciPy-shaped `median_test` outputs for same-median, shifted-median,
/// and three-group inputs, plus fail-closed invalid-input handling.
#[test]
fn e2e_032_median_test_helper_contracts() {
    let scenario_id = "e2e_stats_032_median_test_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let same_left = [1.0, 2.0, 3.0, 4.0, 5.0];
    let same_right = [1.5, 2.5, 3.5, 4.5, 5.5];
    let same = median_test(&[&same_left, &same_right]);
    let pass = same.statistic == 0.0 && same.pvalue == 1.0 && same.df == 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "median_test_same_median_reference",
        "median_test([[1,2,3,4,5], [1.5,2.5,3.5,4.5,5.5]])",
        "groups with the same median should match SciPy's zero statistic, unit p-value, and one degree of freedom",
        &format!(
            "statistic={:.12}, pvalue={:.12}, df={:.1}",
            same.statistic, same.pvalue, same.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let shifted_left: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let shifted_right: Vec<f64> = (100..120).map(|i| i as f64).collect();
    let shifted = median_test(&[&shifted_left, &shifted_right]);
    let pass = (shifted.statistic - 36.1).abs() < 1e-12
        && (shifted.pvalue - 1.874_468_450_406_542_3e-9).abs() < 1e-18
        && shifted.df == 1.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "median_test_shifted_medians_reference",
        "median_test([0..19], [100..119])",
        "widely separated medians should match SciPy's chi-squared statistic and tiny p-value",
        &format!(
            "statistic={:.12}, pvalue={:.18e}, df={:.1}",
            shifted.statistic, shifted.pvalue, shifted.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let tri_a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let tri_b = [6.0, 7.0, 8.0, 9.0, 10.0];
    let tri_c = [11.0, 12.0, 13.0, 14.0, 15.0];
    let tri = median_test(&[&tri_a, &tri_b, &tri_c]);
    let pass = (tri.statistic - 10.178_571_428_571_43).abs() < 1e-12
        && (tri.pvalue - 0.006_162_420_044_100_12).abs() < 1e-15
        && tri.df == 2.0;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "median_test_three_group_reference",
        "median_test([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])",
        "three-group inputs should match SciPy's contingency-based statistic, p-value, and degrees of freedom",
        &format!(
            "statistic={:.12}, pvalue={:.12}, df={:.1}",
            tri.statistic, tri.pvalue, tri.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let invalid_single = median_test(&[&[1.0, 2.0, 3.0]]);
    let pass = invalid_single.statistic.is_nan()
        && invalid_single.pvalue.is_nan()
        && invalid_single.df.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "median_test_single_group_fail_closed",
        "median_test([[1,2,3]])",
        "invalid single-group input should fail closed with NaN sentinels instead of a misleading test result",
        &format!(
            "statistic_is_nan={}, pvalue_is_nan={}, df_is_nan={}",
            invalid_single.statistic.is_nan(),
            invalid_single.pvalue.is_nan(),
            invalid_single.df.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 33: Mood test helper contracts.
/// Verifies SciPy-shaped `mood` outputs for equal-scale, unequal-scale, and
/// small-sample inputs, plus fail-closed NaN-input handling.
#[test]
fn e2e_033_mood_helper_contracts() {
    let scenario_id = "e2e_stats_033_mood_helpers";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let equal_left: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let equal_right: Vec<f64> = (0..30).map(|i| i as f64 + 100.0).collect();
    let equal = mood(&equal_left, &equal_right);
    let pass = equal.statistic == 0.0 && equal.pvalue == 1.0 && equal.df.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "mood_equal_scale_reference",
        "mood([0..29], [100..129])",
        "samples with equal scale should match SciPy's zero z-score and unit p-value",
        &format!(
            "statistic={:.12}, pvalue={:.12}, df_is_nan={}",
            equal.statistic,
            equal.pvalue,
            equal.df.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let wide: Vec<f64> = (0..50).map(|i| i as f64 * 10.0).collect();
    let narrow: Vec<f64> = (0..50).map(|i| 250.0 + i as f64 * 0.1).collect();
    let unequal = mood(&wide, &narrow);
    let pass = (unequal.statistic - 8.325_634_389_708_824).abs() < 1e-12
        && (unequal.pvalue - 8.387_831_487_417_383e-17).abs() < 1e-25
        && unequal.df.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "mood_unequal_scale_reference",
        "mood([0,10,20,...,490], [250.0,250.1,...,254.9])",
        "samples with sharply different scales should match SciPy's z-score and two-sided p-value",
        &format!(
            "statistic={:.12}, pvalue={:.18e}, df_is_nan={}",
            unequal.statistic,
            unequal.pvalue,
            unequal.df.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let small = mood(&[1.0, 2.0], &[3.0, 4.0]);
    let pass = small.statistic == 0.0 && small.pvalue == 1.0 && small.df.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "mood_small_sample_reference",
        "mood([1.0,2.0], [3.0,4.0])",
        "the minimal non-degenerate 2x2 case should match SciPy's finite small-sample result instead of collapsing to NaN",
        &format!(
            "statistic={:.12}, pvalue={:.12}, df_is_nan={}",
            small.statistic,
            small.pvalue,
            small.df.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let nan_input = mood(&[1.0, f64::NAN, 3.0], &[4.0, 5.0, 6.0]);
    let pass = nan_input.statistic.is_nan() && nan_input.pvalue.is_nan() && nan_input.df.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mood_nan_input_fail_closed",
        "mood([1.0, NaN, 3.0], [4.0, 5.0, 6.0])",
        "NaN-contaminated input should fail closed with NaN sentinels rather than emitting a misleading scale statistic",
        &format!(
            "statistic_is_nan={}, pvalue_is_nan={}, df_is_nan={}",
            nan_input.statistic.is_nan(),
            nan_input.pvalue.is_nan(),
            nan_input.df.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 34: Skewtest nan_policy and alternative parity.
/// Verifies SciPy-shaped `skewtest` outputs for the default two-sided path,
/// one-sided alternatives, and the `propagate` / `omit` / `raise` NaN policies.
#[test]
fn e2e_034_skewtest_nan_policy_parity() {
    let scenario_id = "e2e_stats_034_skewtest_nan_policy";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let symmetric = skewtest(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], None, None)
        .expect("symmetric skewtest");
    let pass = (symmetric.statistic - 1.010_804_860_917_778_7).abs() < 1e-12
        && (symmetric.pvalue - 0.312_109_836_142_189_7).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "skewtest_two_sided_reference",
        "skewtest([1,2,3,4,5,6,7,8], nan_policy='propagate', alternative='two-sided')",
        "the default path should match SciPy's nonzero D'Agostino z-score for a length-8 symmetric sample",
        &format!(
            "statistic={:.12}, pvalue={:.12}",
            symmetric.statistic, symmetric.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let skewed = skewtest(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8_000.0],
        None,
        Some("greater"),
    )
    .expect("greater-tail skewtest");
    let pass = (skewed.statistic - 3.571_773_510_360_407).abs() < 1e-12
        && (skewed.pvalue - 1.772_859_952_911_566_6e-4).abs() < 1e-15;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "skewtest_greater_reference",
        "skewtest([1,2,3,4,5,6,7,8000], alternative='greater')",
        "a heavily right-skewed sample should match SciPy's one-sided p-value",
        &format!(
            "statistic={:.12}, pvalue={:.18e}",
            skewed.statistic, skewed.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let propagate = skewtest(
        &[1.0, 2.0, 3.0, 4.0, f64::NAN, 5.0, 6.0, 7.0, 8_000.0],
        Some("propagate"),
        None,
    )
    .expect("propagate");
    let pass = propagate.statistic.is_nan() && propagate.pvalue.is_nan();
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "skewtest_nan_policy_propagate",
        "skewtest([... NaN ...], nan_policy='propagate')",
        "the default NaN policy should fail closed with NaN statistic and p-value",
        &format!(
            "statistic_is_nan={}, pvalue_is_nan={}",
            propagate.statistic.is_nan(),
            propagate.pvalue.is_nan()
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let omit = skewtest(
        &[1.0, 2.0, 3.0, 4.0, f64::NAN, 5.0, 6.0, 7.0, 8_000.0],
        Some("omit"),
        Some("greater"),
    )
    .expect("omit");
    let pass = (omit.statistic - 3.571_773_510_360_407).abs() < 1e-12
        && (omit.pvalue - 1.772_859_952_911_566_6e-4).abs() < 1e-15;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "skewtest_nan_policy_omit",
        "skewtest([... NaN ...], nan_policy='omit', alternative='greater')",
        "omitting NaNs should recover the SciPy reference statistic and one-sided p-value",
        &format!(
            "statistic={:.12}, pvalue={:.18e}",
            omit.statistic, omit.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let raise_error = skewtest(
        &[1.0, 2.0, 3.0, 4.0, f64::NAN, 5.0, 6.0, 7.0, 8_000.0],
        Some("raise"),
        None,
    )
    .expect_err("raise should reject NaNs");
    let pass = raise_error
        == fsci_stats::StatsError::InvalidArgument("The input contains nan values".to_string());
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "skewtest_nan_policy_raise",
        "skewtest([... NaN ...], nan_policy='raise')",
        "the strict NaN policy should reject the sample instead of silently coercing it",
        &raise_error.to_string(),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 35: Rankdata tie-method parity.
/// Verifies SciPy-shaped tie handling for the implemented public `rankdata`
/// methods, plus NaN propagation and invalid-method rejection.
#[test]
fn e2e_035_rankdata_method_parity() {
    let scenario_id = "e2e_stats_035_rankdata_methods";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let average = rankdata(&[1.0, 2.0, 2.0, 4.0], Some("average")).expect("average rankdata");
    let pass = average == vec![1.0, 2.5, 2.5, 4.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "rankdata_average_reference",
        "rankdata([1,2,2,4], method='average')",
        "tied observations should receive the mean of their 1-based rank interval",
        &format!("ranks={average:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let ordinal = rankdata(&[1.0, 2.0, 2.0, 4.0], Some("ordinal")).expect("ordinal rankdata");
    let pass = ordinal == vec![1.0, 2.0, 3.0, 4.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "rankdata_ordinal_reference",
        "rankdata([1,2,2,4], method='ordinal')",
        "tied observations should receive distinct ranks according to stable input order",
        &format!("ranks={ordinal:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let nan_ranks = rankdata(&[1.0, f64::NAN, 2.0], Some("ordinal")).expect("nan rankdata");
    let pass = nan_ranks.iter().all(|rank| rank.is_nan());
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "rankdata_nan_propagates",
        "rankdata([1, NaN, 2], method='ordinal')",
        "the current SciPy-default NaN handling should fail closed with all-NaN ranks",
        &format!("all_nan={}", nan_ranks.iter().all(|rank| rank.is_nan())),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let dense = rankdata(&[1.0, 2.0, 2.0, 4.0], Some("dense")).expect("dense rankdata");
    let pass = dense == vec![1.0, 2.0, 2.0, 3.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "rankdata_dense_reference",
        "rankdata([1,2,2,4], method='dense')",
        "tied observations should share a rank and the next distinct value should advance by one",
        &format!("ranks={dense:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let min = rankdata(&[1.0, 2.0, 2.0, 4.0], Some("min")).expect("min rankdata");
    let pass = min == vec![1.0, 2.0, 2.0, 4.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "rankdata_min_reference",
        "rankdata([1,2,2,4], method='min')",
        "tied observations should receive the minimum rank in their 1-based interval",
        &format!("ranks={min:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let max = rankdata(&[1.0, 2.0, 2.0, 4.0], Some("max")).expect("max rankdata");
    let pass = max == vec![1.0, 3.0, 3.0, 4.0];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        6,
        "rankdata_max_reference",
        "rankdata([1,2,2,4], method='max')",
        "tied observations should receive the maximum rank in their 1-based interval",
        &format!("ranks={max:?}"),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let err = rankdata(&[1.0, 2.0], Some("competition")).expect_err("invalid method");
    let pass = err
        == fsci_stats::StatsError::InvalidArgument(
            "method must be one of {'average', 'min', 'max', 'dense', 'ordinal'}".to_string(),
        );
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        7,
        "rankdata_invalid_method_rejected",
        "rankdata([1,2], method='competition')",
        "unsupported tie methods should still be rejected explicitly instead of silently coercing to a supported mode",
        &err.to_string(),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 36: ttest_ind_from_stats equal_var vs Welch parity.
#[test]
fn e2e_036_ttest_ind_from_stats_equal_var_parity() {
    let scenario_id = "e2e_stats_036_ttest_ind_from_stats";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let equal_var =
        ttest_ind_from_stats(15.0, 87.5_f64.sqrt(), 13, 12.0, 39.0_f64.sqrt(), 11, true);
    let pass = (equal_var.statistic - 0.9051358093310269).abs() < 1e-12
        && (equal_var.pvalue - 0.3751996797581489).abs() < 1e-12
        && (equal_var.df - 22.0).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ttest_ind_from_stats_equal_var",
        "ttest_ind_from_stats(..., equal_var=True)",
        "summary-stat pooled-variance branch should match SciPy's equal_var=True example",
        &format!(
            "t={:.16}, p={:.16}, df={:.16}",
            equal_var.statistic, equal_var.pvalue, equal_var.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let welch = ttest_ind_from_stats(15.0, 87.5_f64.sqrt(), 13, 12.0, 39.0_f64.sqrt(), 11, false);
    let pass = (welch.statistic - 0.9358461935556048).abs() < 1e-12
        && (welch.pvalue - 0.35999818693244234).abs() < 1e-12
        && (welch.df - 20.98461123342992).abs() < 1e-12;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ttest_ind_from_stats_welch",
        "ttest_ind_from_stats(..., equal_var=False)",
        "summary-stat Welch branch should match SciPy's unequal-variance result",
        &format!(
            "t={:.16}, p={:.16}, df={:.16}",
            welch.statistic, welch.pvalue, welch.df
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let binary_equal = ttest_ind_from_stats(
        0.2,
        0.161073_f64.sqrt(),
        150,
        0.225,
        0.175251_f64.sqrt(),
        200,
        true,
    );
    let binary_welch = ttest_ind_from_stats(
        0.2,
        0.161073_f64.sqrt(),
        150,
        0.225,
        0.175251_f64.sqrt(),
        200,
        false,
    );
    let pass = (binary_equal.statistic + 0.5627187905196761).abs() < 1e-12
        && (binary_equal.pvalue - 0.5739887114209542).abs() < 1e-12
        && (binary_welch.statistic + 0.5661276301071694).abs() < 1e-12
        && (binary_welch.pvalue - 0.5716942537704799).abs() < 1e-12
        && (binary_equal.statistic - binary_welch.statistic).abs() > 1e-6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ttest_ind_from_stats_binary_reference",
        "ttest_ind_from_stats(binary summaries, equal_var={True,False})",
        "pooled and Welch summary-stat branches should both match SciPy and remain observably distinct",
        &format!(
            "equal_t={:.16}, equal_p={:.16}, welch_t={:.16}, welch_p={:.16}",
            binary_equal.statistic,
            binary_equal.pvalue,
            binary_welch.statistic,
            binary_welch.pvalue
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 37: chi2_contingency honors correction=False for 2x2 tables.
#[test]
fn e2e_037_chi2_contingency_correction_false_parity() {
    let scenario_id = "e2e_stats_037_chi2_contingency_correction";
    let mut steps = Vec::new();
    let mut all_pass = true;
    let observed = vec![vec![12.0, 5.0], vec![29.0, 2.0]];

    let t = Instant::now();
    let corrected = chi2_contingency(&observed, true);
    let pass = (corrected.statistic - 2.9860164364723074).abs() < 1e-12
        && (corrected.pvalue - 0.08398654171499235).abs() < 1e-12
        && corrected.dof == 1;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "chi2_contingency_correction_true_reference",
        "chi2_contingency([[12,5],[29,2]], correction=True)",
        "the default Yates-corrected 2x2 result should match SciPy",
        &format!(
            "chi2={:.16}, p={:.16}, dof={}",
            corrected.statistic, corrected.pvalue, corrected.dof
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let uncorrected = chi2_contingency(&observed, false);
    let pass = (uncorrected.statistic - 4.646430720203109).abs() < 1e-12
        && (uncorrected.pvalue - 0.03111818732925684).abs() < 1e-12
        && uncorrected.dof == 1
        && uncorrected.statistic > corrected.statistic;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "chi2_contingency_correction_false_reference",
        "chi2_contingency([[12,5],[29,2]], correction=False)",
        "disabling Yates correction should match SciPy's uncorrected 2x2 branch",
        &format!(
            "chi2={:.16}, p={:.16}, dof={}",
            uncorrected.statistic, uncorrected.pvalue, uncorrected.dof
        ),
        t.elapsed().as_nanos(),
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 38: Paired t-test alternative parity.
/// Verifies SciPy-style one-sided tails and invalid-alternative rejection.
#[test]
fn e2e_038_ttest_rel_alternative_parity() {
    let scenario_id = "e2e_stats_038_ttest_rel_alternative";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let before = vec![10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 12.0, 11.0];
    let after = vec![12.0, 14.0, 13.0, 15.0, 11.0, 15.0, 13.0, 12.0];

    let t = Instant::now();
    let less = ttest_rel(&before, &after, Some("less")).expect("ttest_rel less");
    let less_pass = (less.statistic - (-8.880_690_663_831_652)).abs() < 1.0e-12
        && (less.pvalue - 2.326_064_069_551_914_4e-5).abs() < 1.0e-16
        && less.df == 7.0;
    if !less_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ttest_rel_less",
        "ttest_rel(before, after, alternative='less')",
        "paired sample with negative mean difference",
        &format!(
            "t={:.12}, p={:.17}, df={}",
            less.statistic, less.pvalue, less.df
        ),
        t.elapsed().as_nanos(),
        if less_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let greater = ttest_rel(&before, &after, Some("greater")).expect("ttest_rel greater");
    let greater_pass = (greater.statistic - (-8.880_690_663_831_652)).abs() < 1.0e-12
        && (greater.pvalue - 0.999_976_739_359_304_5).abs() < 1.0e-15
        && greater.df == 7.0;
    if !greater_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ttest_rel_greater",
        "ttest_rel(before, after, alternative='greater')",
        "paired sample with negative mean difference",
        &format!(
            "t={:.12}, p={:.15}, df={}",
            greater.statistic, greater.pvalue, greater.df
        ),
        t.elapsed().as_nanos(),
        if greater_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let invalid = ttest_rel(&before, &after, Some("sideways"));
    let invalid_pass = invalid.is_err();
    if !invalid_pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ttest_rel_invalid_alternative",
        "ttest_rel(before, after, alternative='sideways')",
        "paired sample with unsupported alternative",
        &format!("invalid_alternative_error={}", invalid.is_err()),
        t.elapsed().as_nanos(),
        if invalid_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 39: scoreatpercentile array-percentile parity.
#[test]
fn e2e_039_scoreatpercentile_array_percentiles() {
    let scenario_id = "e2e_stats_039_scoreatpercentile_array_percentiles";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let quartiles = scoreatpercentile(&[1.0, 2.0, 3.0, 4.0], &[25.0, 50.0, 75.0], None, None)
        .expect("scoreatpercentile quartiles");
    let quartiles_pass = quartiles.len() == 3
        && (quartiles[0] - 1.75).abs() < 1.0e-12
        && (quartiles[1] - 2.5).abs() < 1.0e-12
        && (quartiles[2] - 3.25).abs() < 1.0e-12;
    if !quartiles_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "scoreatpercentile_array_quartiles",
        "scoreatpercentile([1,2,3,4], per=[25,50,75])",
        "SciPy-style array percentile interpolation",
        &format!("quartiles={quartiles:?}"),
        t.elapsed().as_nanos(),
        if quartiles_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let limited = scoreatpercentile(
        &[1.0, 2.0, 3.0, 4.0, 100.0],
        &[50.0],
        Some((0.0, 4.0)),
        None,
    )
    .expect("scoreatpercentile limit");
    let limited_pass = limited == vec![2.5];
    if !limited_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "scoreatpercentile_limit",
        "scoreatpercentile([1,2,3,4,100], per=[50], limit=(0,4))",
        "limit should filter outlier before interpolation",
        &format!("limited={limited:?}"),
        t.elapsed().as_nanos(),
        if limited_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let lower =
        scoreatpercentile(&[1.0, 2.0, 3.0, 4.0], &[25.0], None, Some("lower")).expect("lower");
    let higher =
        scoreatpercentile(&[1.0, 2.0, 3.0, 4.0], &[25.0], None, Some("higher")).expect("higher");
    let interp_pass = lower == vec![1.0] && higher == vec![2.0];
    if !interp_pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "scoreatpercentile_interpolation_methods",
        "scoreatpercentile(..., interpolation_method in {'lower','higher'})",
        "lower/higher should round the fractional index the SciPy way",
        &format!("lower={lower:?}, higher={higher:?}"),
        t.elapsed().as_nanos(),
        if interp_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let empty = scoreatpercentile(&[], &[25.0, 50.0], None, None).expect("empty input");
    let empty_pass = empty.len() == 2 && empty.iter().all(|value| value.is_nan());
    if !empty_pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "scoreatpercentile_empty_shape",
        "scoreatpercentile([], per=[25,50])",
        "empty input should return NaNs matching the percentile shape",
        &format!("empty={empty:?}"),
        t.elapsed().as_nanos(),
        if empty_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 40: ks_2samp exact p-value parity for small samples.
#[test]
fn e2e_040_ks_2samp_exact_pvalue_parity() {
    let scenario_id = "e2e_stats_040_ks_2samp_exact_pvalue_parity";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let equal = ks_2samp(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 5.0, 6.0]);
    let equal_pass = (equal.statistic - 0.5).abs() < 1.0e-12
        && (equal.pvalue - 0.771_428_571_428_571_6).abs() < 1.0e-15;
    if !equal_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ks_2samp_exact_equal_size",
        "ks_2samp([1,2,3,4], [1,2,5,6])",
        "small equal-size samples should use the exact lattice-path tail",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            equal.statistic, equal.pvalue
        ),
        t.elapsed().as_nanos(),
        if equal_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let unequal = ks_2samp(&[0.0, 1.0, 2.0], &[0.0, 2.0, 4.0, 6.0]);
    let unequal_pass = (unequal.statistic - 0.5).abs() < 1.0e-12
        && (unequal.pvalue - 0.657_142_857_142_857_1).abs() < 1.0e-15;
    if !unequal_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ks_2samp_exact_unequal_size",
        "ks_2samp([0,1,2], [0,2,4,6])",
        "unequal sample sizes should still follow the exact SciPy path-counting tail",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            unequal.statistic, unequal.pvalue
        ),
        t.elapsed().as_nanos(),
        if unequal_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let identical = ks_2samp(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    let identical_pass = identical.statistic == 0.0 && identical.pvalue == 1.0;
    if !identical_pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ks_2samp_exact_identity",
        "ks_2samp([1,2,3], [1,2,3])",
        "identical empirical CDFs should keep the exact p-value at one",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            identical.statistic, identical.pvalue
        ),
        t.elapsed().as_nanos(),
        if identical_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 40b: combined SciPy goldens for independent t-test and two-sample KS.
#[test]
fn e2e_040b_ttest_ind_ks_2samp_scipy_golden() {
    let scenario_id = "e2e_stats_040b_ttest_ind_ks_2samp_scipy_golden";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let ttest = ttest_ind(&[1.2, 2.4, 2.8, 3.1, 4.0, 4.2], &[0.9, 1.7, 2.1, 2.9, 3.3]);
    let ttest_pass = (ttest.statistic - 1.223_421_537_810_768_7).abs() < 1.0e-12
        && (ttest.pvalue - 0.252_234_998_836_302_5).abs() < 1.0e-12
        && ttest.df == 9.0;
    if !ttest_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ttest_ind_equal_var_scipy_golden",
        "ttest_ind([1.2,2.4,2.8,3.1,4.0,4.2], [0.9,1.7,2.1,2.9,3.3])",
        "independent equal-variance t-test should match SciPy statistic, p-value, and df",
        &format!(
            "statistic={:.16}, pvalue={:.16}, df={:.1}",
            ttest.statistic, ttest.pvalue, ttest.df
        ),
        t.elapsed().as_nanos(),
        if ttest_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let ks = ks_2samp(&[0.1, 0.4, 0.9, 1.3, 1.8], &[0.2, 0.3, 1.1, 1.4, 2.2, 2.8]);
    let ks_pass = (ks.statistic - 0.333_333_333_333_333_3).abs() < 1.0e-15
        && (ks.pvalue - 0.818_181_818_181_818_2).abs() < 1.0e-15;
    if !ks_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ks_2samp_exact_scipy_golden",
        "ks_2samp([0.1,0.4,0.9,1.3,1.8], [0.2,0.3,1.1,1.4,2.2,2.8])",
        "two-sample KS exact branch should match SciPy statistic and p-value",
        &format!("statistic={:.16}, pvalue={:.16}", ks.statistic, ks.pvalue),
        t.elapsed().as_nanos(),
        if ks_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 41: ks_1samp asymptotic p-value parity.
#[test]
fn e2e_041_ks_1samp_asymptotic_pvalue_parity() {
    let scenario_id = "e2e_stats_041_ks_1samp_asymptotic_pvalue_parity";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let uniform_samples: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
    let uniform = ks_1samp(&uniform_samples, |x| x.clamp(0.0, 1.0));
    let uniform_pass = (uniform.statistic - 0.005_000_000_000_000_004_4).abs() < 1.0e-15
        && (uniform.pvalue - 1.0).abs() < 1.0e-15;
    if !uniform_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "ks_1samp_asymp_uniform",
        "ks_1samp(uniform_grid, F_uniform)",
        "near-perfect uniform samples should match SciPy's asymptotic D and clipped p-value",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            uniform.statistic, uniform.pvalue
        ),
        t.elapsed().as_nanos(),
        if uniform_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let skewed_samples: Vec<f64> = (0..50).map(|i| (i as f64 / 50.0).powi(2)).collect();
    let skewed = ks_1samp(&skewed_samples, |x| x.clamp(0.0, 1.0));
    let skewed_pass = (skewed.statistic - 0.27).abs() < 1.0e-15
        && (skewed.pvalue - 0.001_364_656_105_079_237).abs() < 1.0e-15;
    if !skewed_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "ks_1samp_asymp_skewed",
        "ks_1samp(x^2 grid, F_uniform)",
        "skewed samples should reproduce SciPy's asymptotic rejection tail",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            skewed.statistic, skewed.pvalue
        ),
        t.elapsed().as_nanos(),
        if skewed_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let normal = Normal::standard();
    let normalish = [-1.2, -0.7, -0.2, 0.0, 0.1, 0.4, 0.9, 1.3];
    let normalish_result = ks_1samp(&normalish, |x| ContinuousDistribution::cdf(&normal, x));
    let normalish_pass = (normalish_result.statistic - 0.170_740_290_560_896_96).abs() < 1.0e-15
        && (normalish_result.pvalue - 0.973_828_230_896_588_7).abs() < 1.0e-15;
    if !normalish_pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "ks_1samp_asymp_normal",
        "ks_1samp(normalish, F_normal)",
        "moderately normal-looking samples should match SciPy's asymptotic non-rejection p-value",
        &format!(
            "statistic={:.16}, pvalue={:.16}",
            normalish_result.statistic, normalish_result.pvalue
        ),
        t.elapsed().as_nanos(),
        if normalish_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 42: continuous-distribution fit parity for implemented fit helpers.
#[test]
fn e2e_042_continuous_distribution_fit_parity() {
    let scenario_id = "e2e_stats_042_continuous_distribution_fit";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let normal: Normal = fit(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let normal_pass = (normal.loc - 3.0).abs() < 1.0e-12
        && (normal.scale - std::f64::consts::SQRT_2).abs() < 1.0e-12;
    if !normal_pass {
        all_pass = false;
    }
    steps.push(make_step(
        1,
        "fit_normal_reference",
        "fit::<Normal>([1,2,3,4,5])",
        "normal MLE should match SciPy's norm.fit loc and scale on finite data",
        &format!("loc={:.16}, scale={:.16}", normal.loc, normal.scale),
        t.elapsed().as_nanos(),
        if normal_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let uniform: Uniform = fit(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let uniform_pass = (uniform.loc - 1.0).abs() < 1.0e-12 && (uniform.scale - 4.0).abs() < 1.0e-12;
    if !uniform_pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "fit_uniform_reference",
        "fit::<Uniform>([1,2,3,4,5])",
        "uniform MLE should recover SciPy's loc=min(data) and scale=max-min",
        &format!("loc={:.16}, scale={:.16}", uniform.loc, uniform.scale),
        t.elapsed().as_nanos(),
        if uniform_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let exponential: Exponential = fit(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let exponential_pass = (exponential.lambda - 0.5).abs() < 1.0e-12;
    if !exponential_pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "fit_exponential_fixed_origin_reference",
        "fit::<Exponential>([0,1,2,3,4])",
        "fixed-origin exponential MLE should match SciPy's expon.fit behavior when loc stays at zero",
        &format!("lambda={:.16}", exponential.lambda),
        t.elapsed().as_nanos(),
        if exponential_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let pareto: Pareto = fit(&[1.0, 2.0, 4.0, 8.0]);
    let pareto_pass =
        (pareto.scale - 1.0).abs() < 1.0e-12 && (pareto.b - (4.0 / 64.0_f64.ln())).abs() < 1.0e-12;
    if !pareto_pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "fit_pareto_reference",
        "fit::<Pareto>([1,2,4,8])",
        "Pareto MLE should recover scale=min(data) and shape=n/sum(log(x/scale))",
        &format!("shape={:.16}, scale={:.16}", pareto.b, pareto.scale),
        t.elapsed().as_nanos(),
        if pareto_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let maxwell: Maxwell = fit(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let maxwell_expected_scale = (11.0_f64 / 3.0).sqrt();
    let maxwell_pass = (maxwell.scale - maxwell_expected_scale).abs() < 1.0e-12;
    if !maxwell_pass {
        all_pass = false;
    }
    steps.push(make_step(
        5,
        "fit_maxwell_reference",
        "fit::<Maxwell>([1,2,3,4,5])",
        "Maxwell MLE should recover sqrt(sum(x^2)/(3n)) for non-negative samples",
        &format!("scale={:.16}", maxwell.scale),
        t.elapsed().as_nanos(),
        if maxwell_pass { "pass" } else { "FAIL" },
    ));

    let t = Instant::now();
    let constant_normal: Normal = fit(&[7.0, 7.0, 7.0]);
    let constant_uniform: Uniform = fit(&[7.0, 7.0, 7.0]);
    let constant_exponential: Exponential = fit(&[0.0, 0.0, 0.0]);
    let degenerate_pass = constant_normal.loc == 7.0
        && constant_normal.scale == 0.0
        && constant_uniform.loc == 7.0
        && constant_uniform.scale == 0.0
        && constant_exponential.lambda.is_infinite()
        && constant_exponential.lambda.is_sign_positive();
    if !degenerate_pass {
        all_pass = false;
    }
    steps.push(make_step(
        6,
        "fit_degenerate_support",
        "fit::<{Normal,Uniform,Exponential}>(degenerate samples)",
        "degenerate data should preserve zero-width support instead of panicking",
        &format!(
            "normal=({:.16},{:.16}), uniform=({:.16},{:.16}), exponential_lambda={}",
            constant_normal.loc,
            constant_normal.scale,
            constant_uniform.loc,
            constant_uniform.scale,
            constant_exponential.lambda
        ),
        t.elapsed().as_nanos(),
        if degenerate_pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 43: trimmed descriptive-statistic parity against live SciPy.
#[test]
fn e2e_043_trimmed_statistics_live_scipy_parity() {
    let scipy_check = Command::new("python3")
        .arg("-c")
        .arg("import scipy; import numpy")
        .status();
    if !matches!(scipy_check, Ok(status) if status.success()) {
        eprintln!("SciPy/NumPy not available; skipping trimmed stats oracle match");
        return;
    }

    let scenario_id = "e2e_stats_043_trimmed_statistics_live_scipy";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let output = Command::new("python3")
        .arg("-c")
        .arg(TRIMMED_STATS_SCIPY_SCRIPT)
        .output()
        .expect("run trimmed-stats SciPy oracle");
    let oracle_duration = t.elapsed().as_nanos();
    assert!(
        output.status.success(),
        "trimmed-stats SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let expected: Vec<TrimmedStatsOracle> =
        serde_json::from_slice(&output.stdout).expect("parse trimmed-stats SciPy oracle output");
    let cases = trimmed_stats_cases();
    assert_eq!(
        expected.len(),
        cases.len(),
        "trimmed-stats oracle case count mismatch"
    );
    steps.push(make_step(
        1,
        "run_scipy_trimmed_stats_oracle",
        "python3 -c TRIMMED_STATS_SCIPY_SCRIPT",
        &format!("cases={}", cases.len()),
        &format!("oracle_cases={}", expected.len()),
        oracle_duration,
        "pass",
    ));

    for (idx, (case, oracle)) in cases.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            case.case_id, oracle.case_id,
            "trimmed-stats oracle case id mismatch"
        );

        let t = Instant::now();
        let actual_tmean = tmean(case.data, case.limits, case.inclusive);
        let actual_tvar = tvar(case.data, case.limits, case.inclusive, case.ddof);
        let actual_tstd = tstd(case.data, case.limits, case.inclusive, case.ddof);
        let actual_tsem = tsem(case.data, case.limits, case.inclusive, case.ddof);
        let tolerance = 1.0e-12;
        let diffs = [
            (actual_tmean - oracle.tmean).abs(),
            (actual_tvar - oracle.tvar).abs(),
            (actual_tstd - oracle.tstd).abs(),
            (actual_tsem - oracle.tsem).abs(),
        ];
        let pass = diffs.iter().all(|diff| *diff <= tolerance);
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            idx + 2,
            &format!("compare_trimmed_stats_{}", case.case_id),
            "tmean/tvar/tstd/tsem vs scipy.stats",
            &format!(
                "limits=({:.6},{:.6}), inclusive=({},{}) ddof={}",
                case.limits.0, case.limits.1, case.inclusive.0, case.inclusive.1, case.ddof
            ),
            &format!(
                "actual=({:.16},{:.16},{:.16},{:.16}) expected=({:.16},{:.16},{:.16},{:.16}) max_diff={:.3e}",
                actual_tmean,
                actual_tvar,
                actual_tstd,
                actual_tsem,
                oracle.tmean,
                oracle.tvar,
                oracle.tstd,
                oracle.tsem,
                diffs.iter().copied().fold(0.0, f64::max)
            ),
            t.elapsed().as_nanos(),
            if pass { "pass" } else { "FAIL" },
        ));
    }

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}

/// Scenario 44: multiscale_graphcorr conformance against live SciPy.
/// Tests the full MGC algorithm (commit 35eb603) matches scipy.stats.multiscale_graphcorr.
#[test]
fn e2e_044_multiscale_graphcorr_scipy_parity() {
    let scipy_check = Command::new("python3")
        .arg("-c")
        .arg("from scipy.stats import multiscale_graphcorr")
        .status();
    if !matches!(scipy_check, Ok(status) if status.success()) {
        eprintln!("SciPy not available; skipping MGC oracle match");
        return;
    }

    let scenario_id = "e2e_stats_044_multiscale_graphcorr_scipy";
    let mut steps = Vec::new();
    let mut all_pass = true;

    let t = Instant::now();
    let output = Command::new("python3")
        .arg("-c")
        .arg(MGC_SCIPY_SCRIPT)
        .output()
        .expect("run MGC SciPy oracle");
    let oracle_duration = t.elapsed().as_nanos();
    assert!(
        output.status.success(),
        "MGC SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: MgcOracleOutput =
        serde_json::from_slice(&output.stdout).expect("parse MGC SciPy oracle output");
    steps.push(make_step(
        1,
        "run_scipy_mgc_oracle",
        "python3 -c MGC_SCIPY_SCRIPT",
        "linear reps=0 case",
        &format!(
            "stat={:.10}, pvalue={:.10}, opt_scale=({},{}), map_shape={}x{}",
            oracle.linear.statistic,
            oracle.linear.pvalue,
            oracle.linear.opt_scale[0],
            oracle.linear.opt_scale[1],
            oracle.linear.map_shape[0],
            oracle.linear.map_shape[1]
        ),
        oracle_duration,
        "pass",
    ));

    // Test case 1: Perfect linear relationship (8 points)
    let x_linear: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
    let y_linear: Vec<Vec<f64>> = (0..8).map(|i| vec![2.0 * i as f64 + 1.0]).collect();

    let t = Instant::now();
    let result = multiscale_graphcorr(&x_linear, &y_linear, 0, Some(0)).expect("MGC linear case");
    let duration = t.elapsed().as_nanos();

    let pass = (result.statistic - oracle.linear.statistic).abs() < 1.0e-12
        && (result.pvalue - oracle.linear.pvalue).abs() < 1.0e-12
        && result.opt_scale == (oracle.linear.opt_scale[0], oracle.linear.opt_scale[1])
        && result.mgc_map.len() == oracle.linear.map_shape[0]
        && result.mgc_map.first().map_or(0, |r| r.len()) == oracle.linear.map_shape[1];
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        2,
        "mgc_linear_8pt",
        "multiscale_graphcorr on perfect linear vs scipy",
        "x=[0..8], y=2x+1",
        &format!(
            "actual=(stat={:.10}, p={:.10}, opt=({},{}), shape={}x{}) expected=(stat={:.10}, p={:.10}, opt=({},{}), shape={}x{})",
            result.statistic,
            result.pvalue,
            result.opt_scale.0,
            result.opt_scale.1,
            result.mgc_map.len(),
            result.mgc_map.first().map_or(0, |r| r.len()),
            oracle.linear.statistic,
            oracle.linear.pvalue,
            oracle.linear.opt_scale[0],
            oracle.linear.opt_scale[1],
            oracle.linear.map_shape[0],
            oracle.linear.map_shape[1]
        ),
        duration,
        if pass { "pass" } else { "FAIL" },
    ));

    // Test case 2: 2D grid points
    let x_grid: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
    ];
    let y_grid: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let t = Instant::now();
    let result = multiscale_graphcorr(&x_grid, &y_grid, 0, Some(0)).expect("MGC grid case");
    let duration = t.elapsed().as_nanos();

    // SciPy returns stat≈0.57, opt_scale=[5,4] for grid case
    let pass = result.statistic > 0.4 && result.statistic < 0.8 && result.mgc_map.len() == 6;
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        3,
        "mgc_grid_2d",
        "multiscale_graphcorr on 2D grid",
        "x=grid(3x2), y=x+y coords",
        &format!(
            "stat={:.10}, opt_scale=({},{})",
            result.statistic, result.opt_scale.0, result.opt_scale.1
        ),
        duration,
        if pass { "pass" } else { "FAIL" },
    ));

    // Test case 3: Permutation p-value bounds
    let x_perm: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
    let y_perm: Vec<Vec<f64>> = vec![
        vec![4.0],
        vec![1.0],
        vec![7.0],
        vec![0.0],
        vec![6.0],
        vec![2.0],
        vec![5.0],
        vec![3.0],
    ];

    let t = Instant::now();
    let result =
        multiscale_graphcorr(&x_perm, &y_perm, 50, Some(1234)).expect("MGC permutation case");
    let duration = t.elapsed().as_nanos();

    // P-value should be between 0 and 1
    let pass = (0.0..=1.0).contains(&result.pvalue);
    if !pass {
        all_pass = false;
    }
    steps.push(make_step(
        4,
        "mgc_permutation_pvalue",
        "multiscale_graphcorr permutation test",
        "reps=50, shuffled y",
        &format!(
            "stat={:.10}, pvalue={:.10}",
            result.statistic, result.pvalue
        ),
        duration,
        if pass { "pass" } else { "FAIL" },
    ));

    assert_artifacts_written(scenario_id, &steps, all_pass);
    assert!(all_pass, "scenario {scenario_id} had failures");
}
