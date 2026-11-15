#![forbid(unsafe_code)]

//! Public nested-domain quadrature against a live SciPy oracle.

use fsci_integrate::{NquadRange, QuadOptions, QuadResult, nquad_with_ranges};
use std::cell::Cell;
use std::process::Command;

#[test]
fn callable_ranges_and_dimension_options_match_live_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import scipy.integrate; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP nquad: SciPy unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import json, os, warnings
import numpy as np
import scipy
from scipy.integrate import nquad
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
cases = [
    (lambda x,y: x + 10*y, [lambda y: [0,2*y], [0,3]], {}),
    (lambda x,y,z: 1., [lambda y,z: [0,1-y-z], lambda z: [0,1-z], [0,1]], {}),
    (lambda x,y: x + 10*y, [lambda y: [2*y,0], [0,3]], {}),
    (lambda x,y: np.exp(-x-y), [lambda y: [0,y], [0,np.inf]],
     {'epsabs':1e-10, 'epsrel':1e-10, 'limit':100}),
    (lambda x,y: abs(x-1/3)**(-.9), [[0,1],[0,1]], [{'limit':3}, {}]),
    (lambda x,y: x*y, [[0,2],[1,3]], {}),
]
rows = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for fun,ranges,opts in cases:
        count = [0]
        def counted(*args):
            count[0] += 1
            return fun(*args)
        value,error,info = nquad(counted,ranges,opts=opts,full_output=True)
        assert info['neval'] == count[0]
        rows.append({'integral':value,'error':error,'neval':info['neval']})
print(json.dumps({'scipy':scipy.__version__,'numpy':np.__version__,'rows':rows}))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy nquad failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(
        rows.len(),
        6,
        "the live oracle must return every requested case"
    );

    let triangle = |outer: &[f64]| (0.0, 2.0 * outer[0]);
    let reverse_triangle = |outer: &[f64]| (2.0 * outer[0], 0.0);
    let simplex_inner = |outer: &[f64]| (0.0, 1.0 - outer[0] - outer[1]);
    let simplex_middle = |outer: &[f64]| (0.0, 1.0 - outer[0]);
    let infinite_inner = |outer: &[f64]| (0.0, outer[0]);
    let calls = Cell::new(0usize);
    let evaluate = |id: usize| -> QuadResult {
        let func = |args: &[f64]| {
            calls.set(calls.get() + 1);
            match id {
                0 | 2 => args[0] + 10.0 * args[1],
                1 => 1.0,
                3 => (-args[0] - args[1]).exp(),
                4 => (args[0] - 1.0 / 3.0).abs().powf(-0.9),
                5 => args[0] * args[1],
                _ => unreachable!(),
            }
        };
        let ranges = match id {
            0 => vec![NquadRange::Dynamic(&triangle), NquadRange::Fixed(0.0, 3.0)],
            1 => vec![
                NquadRange::Dynamic(&simplex_inner),
                NquadRange::Dynamic(&simplex_middle),
                NquadRange::Fixed(0.0, 1.0),
            ],
            2 => vec![
                NquadRange::Dynamic(&reverse_triangle),
                NquadRange::Fixed(0.0, 3.0),
            ],
            3 => vec![
                NquadRange::Dynamic(&infinite_inner),
                NquadRange::Fixed(0.0, f64::INFINITY),
            ],
            4 => vec![NquadRange::Fixed(0.0, 1.0); 2],
            5 => vec![NquadRange::Fixed(0.0, 2.0), NquadRange::Fixed(1.0, 3.0)],
            _ => unreachable!(),
        };
        let options = match id {
            3 => vec![QuadOptions {
                epsabs: 1e-10,
                epsrel: 1e-10,
                limit: 100,
            }],
            4 => vec![
                QuadOptions {
                    limit: 3,
                    ..QuadOptions::default()
                },
                QuadOptions::default(),
            ],
            _ => vec![],
        };
        nquad_with_ranges(func, &ranges, &options).expect("public nquad solve")
    };
    let mut compared = 0;
    for (id, row) in rows.iter().enumerate() {
        calls.set(0);
        let actual = evaluate(id);
        let expected = row["integral"].as_f64().unwrap();
        let expected_error = row["error"].as_f64().unwrap();
        assert!(actual.integral.is_finite() && actual.error.is_finite());
        assert!(
            (actual.integral - expected).abs() <= 1e-10 * (1.0 + expected.abs()),
            "case {id}: Rust {} versus SciPy {expected}",
            actual.integral
        );
        assert!(
            (actual.error - expected_error).abs() <= 1e-10 * (1.0 + expected_error),
            "case {id}: Rust error {} versus SciPy {expected_error}",
            actual.error
        );
        assert_eq!(
            actual.neval,
            calls.get(),
            "case {id}: count only integrand calls"
        );
        assert!(actual.neval > 0);
        // Smooth finite intervals and the deliberately capped inner solve
        // have the same deterministic subdivision pattern in both engines.
        if id != 3 {
            assert_eq!(actual.neval as u64, row["neval"].as_u64().unwrap());
        }
        assert_eq!(actual.converged, id != 4);
        compared += 1;
    }
    assert_eq!(compared, 6);
    eprintln!(
        "nquad compared {compared}/6 cases; SciPy {}, NumPy {}",
        oracle["scipy"], oracle["numpy"]
    );
}
