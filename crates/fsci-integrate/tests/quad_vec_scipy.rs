#![forbid(unsafe_code)]

//! Live vector quadrature comparisons on discontinuous and improper integrals.

use fsci_integrate::{QuadVecNorm, QuadVecOptions, QuadVecRule, quad_vec_with_options};
use std::process::Command;

#[test]
fn vector_breakpoints_rules_and_norms_match_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import scipy.integrate, numpy"])
        .output()
        .is_ok_and(|o| o.status.success())
    {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP quad_vec: SciPy unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import json, os, numpy as np, scipy
from scipy.integrate import quad_vec
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
def integrand(x):
    y = np.exp(-abs(x)) * (1 if x < 0.314 else -2)
    return np.array([y, x*y])
rows = []
for a,b in [(0.,1.), (np.inf,-2.), (-np.inf,np.inf)]:
    for norm in ['2','max']:
        for rule in ['gk15','gk21']:
            result, error, info = quad_vec(integrand, a,b, points=[0.,0.314],
                epsabs=1e-10, epsrel=1e-10, norm=norm, quadrature=rule, full_output=True)
            assert info.success
            rows.append(result.tolist())
print(json.dumps(rows))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<[f64; 2]> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rows.len(), 12);
    let mut index = 0;
    for (a, b) in [
        (0.0, 1.0),
        (f64::INFINITY, -2.0),
        (f64::NEG_INFINITY, f64::INFINITY),
    ] {
        for norm in [QuadVecNorm::Euclidean, QuadVecNorm::Max] {
            for rule in [QuadVecRule::Gk15, QuadVecRule::Gk21] {
                let result = quad_vec_with_options(
                    |x| {
                        let y = (-x.abs()).exp() * if x < 0.314 { 1.0 } else { -2.0 };
                        vec![y, x * y]
                    },
                    a,
                    b,
                    QuadVecOptions {
                        epsabs: 1e-10,
                        epsrel: 1e-10,
                        norm,
                        rule,
                        points: vec![0.0, 0.314],
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(result.converged, "{a}..{b} {norm:?} {rule:?}: {result:?}");
                assert_eq!(result.integral.len(), 2, "case {index}: vector shape");
                for (&value, reference) in result.integral.iter().zip(rows[index]) {
                    assert!(
                        (value - reference).abs() < 1e-10,
                        "{a}..{b} {norm:?} {rule:?}: {value} vs {reference}"
                    );
                }
                index += 1;
            }
        }
    }
    eprintln!("quad_vec compared {index} vector integrals with live SciPy");
}
