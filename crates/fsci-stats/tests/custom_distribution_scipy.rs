//! Differential test of the public custom continuous distribution interface.
//! Set FSCI_REQUIRE_SCIPY_ORACLE=1 to require SciPy 1.17.1 and NumPy 2.4.3.
//! FSCI_PYTHON selects the interpreter.

use std::{io::ErrorKind, process::Command};

use fsci_stats::{ContinuousDistribution, ContinuousDistributionSpec, make_distribution};

const ORACLE: &str = r#"
import os
import sys
try:
    import numpy as np
    import scipy
    from scipy import stats
except ModuleNotFoundError as e:
    if e.name not in ('numpy', 'scipy'):
        raise
    sys.exit(77)

if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
print('versions', scipy.__version__, np.__version__)

class LogUniform:
    __make_distribution_version__ = '1.16.0'
    parameters = {'a': (0, np.inf), 'b': ('a', np.inf)}
    support = ('a', 'b')
    def pdf(self, x, *, a, b):
        return 1 / (x * np.log(b/a))

family = stats.make_distribution(LogUniform())
for a, b in [(1., 3.), (2., 17.)]:
    d = family(a=a, b=b)
    print('moments', d.mean(), d.variance())
    for q in [0.01, 0.2, 0.49, 0.5, 0.8, 0.99]:
        x = (1-q)*a + q*b
        print('loguniform', d.cdf(x), d.ccdf(x), d.icdf(q), d.iccdf(q))

d = stats.expon()
print('moments', d.mean(), d.var())
for q in [1e-40, 1e-15, 0.01, 0.5, 0.8]:
    print('exponential', d.ppf(q), d.isf(q))
"#;

struct LogUniform(f64, f64);

impl ContinuousDistributionSpec for LogUniform {
    fn support(&self) -> (f64, f64) {
        (self.0, self.1)
    }

    fn pdf(&self, x: f64) -> f64 {
        1.0 / (x * (self.1 / self.0).ln())
    }
}

struct Exponential;

impl ContinuousDistributionSpec for Exponential {
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    fn pdf(&self, x: f64) -> f64 {
        (-x).exp()
    }
}

#[test]
fn pdf_only_custom_distributions_match_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = match Command::new(&python).args(["-c", ORACLE]).output() {
        Err(err) if !required && err.kind() == ErrorKind::NotFound => {
            eprintln!("skipping custom distribution oracle: interpreter {python} unavailable");
            return;
        }
        other => other.expect("launch SciPy custom distribution oracle"),
    };
    if !required && output.status.code() == Some(77) {
        eprintln!("skipping custom distribution oracle: missing NumPy or SciPy");
        return;
    }
    assert!(
        output.status.success(),
        "SciPy oracle failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let versions = lines.next().expect("version banner");
    let version_fields: Vec<_> = versions.split_whitespace().collect();
    assert_eq!(version_fields.len(), 3);
    assert_eq!(version_fields[0], "versions");
    eprintln!("{versions}");

    let mut compared = 0;
    let mut compare = |kind: &str, actual: &[f64]| {
        let mut row = lines.next().expect("missing oracle row").split_whitespace();
        assert_eq!(row.next(), Some(kind), "oracle row order");
        for &value in actual {
            let reference: f64 = row.next().expect("missing oracle field").parse().unwrap();
            assert!(
                reference.is_finite()
                    && value.is_finite()
                    && (value - reference).abs() <= reference.abs() * 1e-10,
                "{kind}: {value} differs from SciPy {reference}"
            );
            compared += 1;
        }
        assert!(row.next().is_none(), "unexpected extra field in {kind}");
    };
    for (a, b) in [(1.0, 3.0), (2.0, 17.0)] {
        let d = make_distribution(LogUniform(a, b)).unwrap();
        compare("moments", &[d.mean(), d.var()]);
        for q in [0.01, 0.2, 0.49, 0.5, 0.8, 0.99] {
            let x = (1.0 - q) * a + q * b;
            compare("loguniform", &[d.cdf(x), d.sf(x), d.ppf(q), d.isf(q)]);
        }
    }
    let d = make_distribution(Exponential).unwrap();
    compare("moments", &[d.mean(), d.var()]);
    for q in [1e-40, 1e-15, 0.01, 0.5, 0.8] {
        compare("exponential", &[d.ppf(q), d.isf(q)]);
    }
    assert!(lines.next().is_none(), "unexpected extra oracle row");
    assert_eq!(compared, 64);
}
