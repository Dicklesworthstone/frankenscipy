//! Distribution modes: analytical boundaries, range safety, and live SciPy densities.
//!
//! Run with `FSCI_REQUIRE_SCIPY_ORACLE=1` to require SciPy 1.17.1 / NumPy 2.4.3.
//! `FSCI_PYTHON` selects the interpreter. The numerical oracle maximizes SciPy's
//! density independently; it does not use the Rust mode as an optimizer start.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_stats::{
    ContinuousDistribution, CrystalBall, GenLogistic, GenNorm, Gompertz, HalfGenNorm,
    InverseGaussian, LaplaceAsymmetric, LocationScale, LogGamma, Moyal, Pearson3, TruncExpon,
    TruncNormal, TruncPareto, TruncWeibullMin,
};

struct ModeCase {
    name: &'static str,
    shapes: Vec<f64>,
    lower: f64,
    upper: f64,
    mode: f64,
    log_density: f64,
    grid_maximum: f64,
}

fn case<D: ContinuousDistribution>(
    name: &'static str,
    shapes: Vec<f64>,
    distribution: D,
    lower: f64,
    upper: f64,
) -> ModeCase {
    let mode = distribution.mode();
    let log_density = distribution.logpdf(mode);
    let grid_maximum = (0..=128)
        .map(|i| distribution.logpdf(lower + (upper - lower) * f64::from(i) / 128.0))
        .fold(f64::NEG_INFINITY, f64::max);
    ModeCase {
        name,
        shapes,
        lower,
        upper,
        mode,
        log_density,
        grid_maximum,
    }
}

fn closed_form_cases() -> Vec<ModeCase> {
    let mut cases = Vec::new();
    for mu in [0.1, 1.0, 10.0] {
        cases.push(case(
            "invgauss",
            vec![mu],
            InverseGaussian::new(mu),
            0.0,
            3.0,
        ));
    }
    for skew in [-3.0, -2.0, -0.5, 0.0, 0.5, 2.0, 3.0] {
        let lower = if skew > 0.0 { -2.0 / skew } else { -5.0 };
        let upper = if skew < 0.0 { -2.0 / skew } else { 5.0 };
        cases.push(case(
            "pearson3",
            vec![skew],
            Pearson3::new(skew),
            lower,
            upper,
        ));
    }
    cases.push(case("moyal", vec![], Moyal, -5.0, 5.0));
    for c in [0.1, 1.0, 10.0] {
        cases.push(case("gompertz", vec![c], Gompertz::new(c), 0.0, 8.0));
        cases.push(case("genlogistic", vec![c], GenLogistic::new(c), -8.0, 8.0));
        cases.push(case("loggamma", vec![c], LogGamma::new(c), -8.0, 8.0));
    }
    for beta in [0.5, 2.0, 8.0] {
        cases.push(case("gennorm", vec![beta], GenNorm::new(beta), -3.0, 3.0));
        cases.push(case(
            "halfgennorm",
            vec![beta],
            HalfGenNorm::new(beta),
            0.0,
            3.0,
        ));
    }
    for (a, b) in [(-3.0, 2.0), (2.0, 5.0), (-5.0, -2.0)] {
        cases.push(case("truncnorm", vec![a, b], TruncNormal::new(a, b), a, b));
    }
    for (c, a, b) in [
        (0.5, 0.0, 3.0),
        (1.0, 0.0, 3.0),
        (2.0, 0.0, 2.0),
        (0.5, 0.2, 3.0),
        (2.0, 0.1, 3.0),
        (2.0, 2.0, 3.0),
        (3.0, 0.1, 0.2),
    ] {
        cases.push(case(
            "truncweibull_min",
            vec![c, a, b],
            TruncWeibullMin::new(c, a, b),
            a,
            b,
        ));
    }
    for b in [2.0, -0.5, -2.0] {
        cases.push(case(
            "truncpareto",
            vec![b, 3.0],
            TruncPareto::new(b, 3.0),
            1.0,
            3.0,
        ));
    }
    for b in [0.25, 1.0, 5.0] {
        cases.push(case("truncexpon", vec![b], TruncExpon::new(b), 0.0, b));
    }
    for (beta, m) in [(0.5, 1.5), (2.0, 3.0), (6.0, 9.0)] {
        cases.push(case(
            "crystalball",
            vec![beta, m],
            CrystalBall::new(beta, m),
            -8.0,
            8.0,
        ));
    }
    for kappa in [0.1, 1.0, 10.0] {
        cases.push(case(
            "laplace_asymmetric",
            vec![kappa],
            LaplaceAsymmetric::new(kappa),
            -3.0,
            3.0,
        ));
    }
    assert_eq!(cases.len(), 48);
    cases
}

#[test]
fn closed_form_modes_maximize_public_log_densities() {
    for c in closed_form_cases() {
        assert!(
            c.mode.is_finite(),
            "{} {:?}: missing mode",
            c.name,
            c.shapes
        );
        assert!((c.lower..=c.upper).contains(&c.mode));
        assert!(c.log_density.is_finite() || c.log_density == f64::INFINITY);
        assert!(
            c.log_density >= c.grid_maximum - 2.0e-12,
            "{} {:?}: mode {} has log density {}, grid found {}",
            c.name,
            c.shapes,
            c.mode,
            c.log_density,
            c.grid_maximum
        );
    }
}

#[test]
fn inverse_gaussian_mode_is_range_safe_and_solves_the_stationarity_equation() {
    assert!((InverseGaussian::new(1.0).mode() - 0.302_775_637_731_994_6).abs() < 1.0e-16);
    for mu in [
        f64::from_bits(1),
        1.0e-300,
        1.0e-100,
        0.1,
        1.0,
        10.0,
        1.0e100,
        f64::MAX,
    ] {
        let mode = InverseGaussian::new(mu).mode();
        assert!(mode > 0.0 && mode.is_finite());
        assert!(mode <= mu && mode <= 1.0 / 3.0);
        let stationarity = (mode / mu).powi(2) + 3.0 * mode - 1.0;
        assert!(
            stationarity.abs() <= 8.0 * f64::EPSILON,
            "mu={mu}: {stationarity}"
        );
    }
}

#[test]
fn truncated_and_pearson_modes_preserve_endpoint_semantics() {
    assert_eq!(TruncNormal::new(38.0, 39.0).mode(), 38.0);
    assert_eq!(TruncNormal::new(-39.0, -38.0).mode(), -38.0);
    assert_eq!(
        TruncNormal::new(f64::NEG_INFINITY, f64::INFINITY).mode(),
        0.0
    );
    assert_eq!(TruncWeibullMin::new(0.5, 0.0, 3.0).mode(), 0.0);
    assert_eq!(TruncWeibullMin::new(2.0, 0.0, 0.25).mode(), 0.25);
    assert!(
        (TruncWeibullMin::new(2.0, 0.0, 2.0).mode() - std::f64::consts::FRAC_1_SQRT_2).abs()
            < 1.0e-15
    );
    assert_eq!(TruncPareto::new(-2.0, 3.0).mode(), 3.0);
    assert_eq!(TruncPareto::new(-0.5, 3.0).mode(), 1.0);
    assert!(TruncPareto::new(-1.0, 3.0).mode().is_nan());
    for skew in [-3.0, 3.0] {
        let distribution = Pearson3::new(skew);
        assert_eq!(distribution.mode(), -2.0 / skew);
        assert_eq!(distribution.pdf(distribution.mode()), f64::INFINITY);
        assert_eq!(distribution.logpdf(distribution.mode()), f64::INFINITY);
    }
    for skew in [-2.0, 2.0] {
        let distribution = Pearson3::new(skew);
        assert_eq!(distribution.pdf(distribution.mode()), 1.0);
        assert_eq!(distribution.logpdf(distribution.mode()), 0.0);
        let outside = distribution.mode() - skew.signum() * 1.0e-6;
        assert_eq!(distribution.logpdf(outside), f64::NEG_INFINITY);
    }
    assert_eq!(Pearson3::new(1.0e-8).mode(), 0.0);
}

#[test]
fn modes_transform_with_location_scale_and_reject_invalid_public_shapes() {
    let base = InverseGaussian::new(1.0);
    let expected = -3.0 + 5.0 * base.mode();
    assert_eq!(
        LocationScale::new(base, -3.0, 5.0).unwrap().mode(),
        expected
    );
    assert_eq!(
        LocationScale::new(TruncNormal::new(2.0, 4.0), -3.0, 5.0)
            .unwrap()
            .mode(),
        7.0
    );
    assert!(InverseGaussian { mu: f64::NAN }.mode().is_nan());
    assert!(
        Pearson3 {
            skew: f64::INFINITY
        }
        .mode()
        .is_nan()
    );
    assert!(Gompertz { c: -1.0 }.mode().is_nan());
    assert!(GenLogistic { c: 0.0 }.mode().is_nan());
    assert!(
        GenNorm {
            beta: f64::INFINITY
        }
        .mode()
        .is_nan()
    );
    assert!(HalfGenNorm { beta: 0.0 }.mode().is_nan());
    assert!(LogGamma { c: f64::NAN }.mode().is_nan());
    assert!(TruncNormal { a: 2.0, b: 1.0 }.mode().is_nan());
    assert!(
        TruncWeibullMin {
            c: 0.0,
            a: 0.0,
            b: 1.0
        }
        .mode()
        .is_nan()
    );
    assert!(TruncPareto { b: 0.0, c: 2.0 }.mode().is_nan());
    assert!(TruncExpon { b: -1.0 }.mode().is_nan());
    assert!(
        CrystalBall {
            beta_param: 1.0,
            m: 1.0
        }
        .mode()
        .is_nan()
    );
    assert!(LaplaceAsymmetric { kappa: -1.0 }.mode().is_nan());
}

const ORACLE: &str = r#"
import os
import sys
import numpy as np
import scipy
from scipy import optimize, stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
compared = 0
families = set()
for line in sys.stdin:
    fields = line.split()
    name = fields[0]
    lower, upper, mode, rust_logpdf = map(float, fields[1:5])
    shapes = tuple(map(float, fields[5:]))
    distribution = getattr(stats, name)
    logpdf = lambda x: float(distribution.logpdf(x, *shapes))
    result = optimize.minimize_scalar(lambda x: -logpdf(x), bounds=(lower, upper),
                                     method='bounded', options={'xatol': 1e-12})
    assert result.success, (name, shapes, result)
    # A fixed zero probe resolves symmetric flat peaks (e.g. gennorm beta=8),
    # where floating-point logpdf cannot locate the maximum from objective
    # differences alone. This candidate is independent of the Rust result.
    candidates = [lower, upper]
    if lower <= 0 <= upper:
        candidates.append(0.0)
    candidates.append(result.x)
    oracle_mode = max(candidates, key=logpdf)
    expected_logpdf = logpdf(mode)
    assert np.isfinite(mode), (name, shapes, mode)
    assert abs(mode - oracle_mode) <= 3e-7 * max(1.0, abs(oracle_mode)), \
        (name, shapes, mode, oracle_mode)
    if expected_logpdf == np.inf:
        assert rust_logpdf == np.inf, (name, shapes, rust_logpdf)
    else:
        assert np.isfinite(expected_logpdf) and np.isfinite(rust_logpdf)
        assert abs(rust_logpdf - expected_logpdf) <= 2e-11 * max(1.0, abs(expected_logpdf)), \
            (name, shapes, rust_logpdf, expected_logpdf)
        assert expected_logpdf >= logpdf(oracle_mode) - 2e-11, (name, shapes)
    print(name, shapes, 'Rust mode', mode, 'SciPy maximum', oracle_mode)
    compared += 1
    families.add(name)
assert compared == 48 and len(families) == 14, (compared, families)
print('compared cases=48; families=14')
"#;

#[test]
fn closed_form_modes_match_live_scipy_density_maxima() {
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let available = Command::new(&python)
        .args(["-c", "import scipy, numpy"])
        .output();
    if !matches!(available, Ok(ref result) if result.status.success()) {
        assert!(
            !required,
            "required SciPy oracle is unavailable: {available:?}"
        );
        eprintln!("SKIP: SciPy oracle unavailable; set FSCI_REQUIRE_SCIPY_ORACLE=1 to require it");
        return;
    }
    let mut input = String::new();
    for c in closed_form_cases() {
        use std::fmt::Write as _;
        write!(
            &mut input,
            "{} {:.17e} {:.17e} {:.17e} {:.17e}",
            c.name, c.lower, c.upper, c.mode, c.log_density
        )
        .unwrap();
        for shape in c.shapes {
            write!(&mut input, " {shape:.17e}").unwrap();
        }
        input.push('\n');
    }
    let mut child = Command::new(&python)
        .args(["-c", ORACLE])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start SciPy mode oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "SciPy mode oracle failed\n{stdout}\n{stderr}"
    );
    assert!(stdout.contains("compared cases=48; families=14"));
    eprintln!("{stdout}");
}
