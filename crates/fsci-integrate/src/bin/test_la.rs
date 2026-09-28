use fsci_integrate::{SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp};
use fsci_runtime::RuntimeMode;

fn main() {
    let opts = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &[0.0],
        method: SolverKind::Bdf,
        t_eval: None,
        dense_output: false,
        events: None,
        rtol: 0.0,
        atol: ToleranceValue::Scalar(0.0),
        first_step: None,
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        jac: None,
    };
    let res = solve_ivp(&mut |_t: f64, _y: &[f64]| vec![0.0], &opts);
    println!("{:?}", res.map(|r| r.y.last().unwrap().clone()));
}
