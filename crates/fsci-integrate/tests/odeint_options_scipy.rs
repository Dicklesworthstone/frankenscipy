//! Compare the configured ODEPACK driver against SciPy on stiff, mixed-scale states.

use fsci_integrate::{OdeintOptions, ToleranceValue, odeint_with_options};
use std::process::Command;

#[test]
fn configured_odeint_matches_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import scipy.integrate, numpy"])
        .output()
        .is_ok_and(|output| output.status.success())
    {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP odeint options: SciPy unavailable; compared cases = 0");
        return;
    }
    let script = r#"
import json, os, warnings, numpy as np, scipy
from scipy.integrate import odeint, ODEintWarning
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
warnings.simplefilter('error', ODEintWarning)
rates = np.array([-1., -2000., -.5])
rows, info = odeint(lambda y, t: rates*y, [1., 1e-14, 1e8],
    [0., .0001, .002, .1, 1.], rtol=[0., 1e-8, 1e-10], atol=[1e-10, 1e-26, 0.],
    ml=0, mu=0, Dfun=lambda y, t: rates.reshape(1, 3), full_output=True)
assert info['mused'][-1] == 2 and info['nje'][-1] > 0
print(json.dumps(rows.tolist()))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<[f64; 3]> = serde_json::from_slice(&output.stdout).unwrap();
    let times = [0.0, 0.0001, 0.002, 0.1, 1.0];
    let options = OdeintOptions {
        rtol: ToleranceValue::Vector(vec![0.0, 1e-8, 1e-10]),
        atol: ToleranceValue::Vector(vec![1e-10, 1e-26, 0.0]),
        jac: Some(|_, _| vec![vec![-1.0, -2000.0, -0.5]]),
        bandwidth: Some((0, 0)),
        ..OdeintOptions::default()
    };
    let result = odeint_with_options(
        &mut |y, _| vec![-y[0], -2000.0 * y[1], -0.5 * y[2]],
        &[1.0, 1e-14, 1e8],
        &times,
        &options,
    )
    .unwrap();
    assert_eq!(expected.len(), times.len());
    assert_eq!(result.len(), times.len());
    for (row, scipy) in result.iter().zip(expected) {
        assert_eq!(row.len(), 3);
        for (i, floor) in [2e-9, 4e-24, 0.1].into_iter().enumerate() {
            let error = (row[i] - scipy[i]).abs();
            assert!(
                error <= floor + 1e-8 * scipy[i].abs(),
                "{i}: {row:?}, {scipy:?}"
            );
        }
    }
}
