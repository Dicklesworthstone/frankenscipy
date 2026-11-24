#![forbid(unsafe_code)]

//! MILP objectives, feasible incumbents and global bounds against live SciPy.
//! The required lane must run SciPy 1.17.1 / NumPy 2.4.3 and compare every case.

use fsci_opt::{Integrality, MilpOptions, MilpProblem, MilpResult, milp};
use serde::Deserialize;
use std::process::Command;

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    c: Vec<f64>,
    integrality: Vec<u8>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    bounds: Vec<(Option<f64>, Option<f64>)>,
    status: u8,
    fun: Option<f64>,
}

#[derive(Deserialize)]
struct Oracle {
    scipy: String,
    numpy: String,
    cases: Vec<Case>,
}

fn assert_feasible(case: &Case, result: &MilpResult) {
    assert_eq!(result.x.len(), case.c.len(), "{}: {result:?}", case.name);
    for (i, &value) in result.x.iter().enumerate() {
        let (lo, hi) = case.bounds[i];
        let (lo, hi) = (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY));
        let semi_zero = case.integrality[i] >= 2 && value.abs() <= 1e-8;
        assert!(value.is_finite());
        assert!(
            semi_zero || (value >= lo - 1e-7 && value <= hi + 1e-7),
            "{}: {result:?}",
            case.name
        );
        if matches!(case.integrality[i], 1 | 3) {
            assert!(
                (value - value.round()).abs() < 1e-8,
                "{}: {result:?}",
                case.name
            );
        }
    }
    for (row, &rhs) in case.a_ub.iter().zip(&case.b_ub) {
        let lhs: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
        assert!(
            lhs <= rhs + 1e-7 * (1.0 + rhs.abs()),
            "{}: {lhs} > {rhs}; {result:?}",
            case.name
        );
    }
    for (row, &rhs) in case.a_eq.iter().zip(&case.b_eq) {
        let lhs: f64 = row.iter().zip(&result.x).map(|(a, x)| a * x).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-7 * (1.0 + rhs.abs()),
            "{}: {lhs} != {rhs}; {result:?}",
            case.name
        );
    }
    let objective: f64 = case.c.iter().zip(&result.x).map(|(c, x)| c * x).sum();
    assert!((objective - result.fun).abs() <= 1e-10 * (1.0 + objective.abs()));
}

#[test]
fn milp_global_bounds_and_gap_controls_match_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON")
        .or_else(|_| std::env::var("SCIPY_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args([
            "-c",
            "import scipy,numpy; assert scipy.__version__=='1.17.1' and numpy.__version__=='2.4.3'",
        ])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            !required,
            "required SciPy 1.17.1 / NumPy 2.4.3 oracle unavailable: {python}"
        );
        eprintln!("skipping optional MILP oracle: pinned SciPy/NumPy unavailable");
        return;
    }
    let script = r#"
import json
import numpy as np
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint

rng = np.random.default_rng(20261123)
cases = []
def add(name, c, integrality, bounds, a_ub=(), b_ub=(), a_eq=(), b_eq=()):
    c = list(map(float, c))
    a_ub = [list(map(float, row)) for row in a_ub]
    a_eq = [list(map(float, row)) for row in a_eq]
    b_ub, b_eq = list(map(float, b_ub)), list(map(float, b_eq))
    lower = [-np.inf if lo is None else lo for lo, hi in bounds]
    upper = [np.inf if hi is None else hi for lo, hi in bounds]
    constraints = []
    if a_ub:
        constraints.append(LinearConstraint(a_ub, -np.inf, b_ub))
    if a_eq:
        constraints.append(LinearConstraint(a_eq, b_eq, b_eq))
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=constraints, options={'mip_rel_gap': 0.0})
    if result.status == 4:
        # Retry ambiguous presolve outcomes so HiGHS distinguishes feasible
        # unbounded models from infeasible ones; keep exact status assertions.
        result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=constraints, options={'mip_rel_gap': 0.0, 'presolve': False})
    cases.append(dict(name=name, c=c, integrality=list(map(int,integrality)),
                      a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq,
                      bounds=bounds, status=int(result.status),
                      fun=None if result.fun is None else float(result.fun)))

for k in range(24):
    n = 8
    weights = rng.integers(1, 12, (2, n))
    capacity = np.floor(weights.sum(axis=1) * [0.42, 0.57]) + 0.25
    costs = -(rng.integers(1, 30, n) + rng.random(n))
    add(f'knapsack-{k}', costs, [1]*n, [(0.0,1.0)]*n, weights, capacity)
for k in range(8):
    demand = 4.25 + 0.5*k
    costs = [1.0 + k/10.0, 1.7, 2.2]
    add(f'mixed-production-{k}', costs, [1,1,0], [(0.0,5.0)]*3,
        [[1.0,0.0,0.0]], [3.25], [[1.0,1.0,1.0]], [demand])
for code in [2,3]:
    for required in [0.0,1.0,3.5]:
        add(f'semi-{code}-{required}', [1.0], [code], [(2.2,5.5)],
            [[-1.0]], [-required])
add('unbounded-lp-infeasible-integer', [0.0,-1.0], [1,0], [(0.0,1.0),(0.0,None)],
    a_eq=[[1.0,0.0]], b_eq=[0.5])
add('unbounded-integer-feasible', [0.0,-1.0], [1,0], [(0.0,1.0),(0.0,None)],
    a_eq=[[1.0,0.0]], b_eq=[1.0])
add('empty-integer-domain', [1.0], [1], [(0.2,0.8)])
add('zero-optimum', [-1.0,1.0], [1,0], [(0.0,1.0),(0.0,2.0)], [[2.0,-1.0]], [1.0])
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, cases=cases), allow_nan=False))
"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .output()
        .expect("launch pinned SciPy");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).expect("SciPy MILP response");
    assert_eq!(oracle.scipy, "1.17.1");
    assert_eq!(oracle.numpy, "2.4.3");
    assert_eq!(
        oracle.cases.len(),
        42,
        "every generated case must be returned"
    );
    let mut compared = 0usize;
    let mut bound_checks = 0usize;
    for case in &oracle.cases {
        let integrality: Vec<_> = case
            .integrality
            .iter()
            .map(|kind| match kind {
                0 => Integrality::Continuous,
                1 => Integrality::Integer,
                2 => Integrality::SemiContinuous,
                3 => Integrality::SemiInteger,
                _ => panic!("unknown integrality"),
            })
            .collect();
        let problem = MilpProblem {
            c: &case.c,
            integrality: &integrality,
            a_ub: &case.a_ub,
            b_ub: &case.b_ub,
            a_eq: &case.a_eq,
            b_eq: &case.b_eq,
            bounds: &case.bounds,
        };
        let result = milp(
            problem,
            MilpOptions {
                max_nodes: Some(20_000),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, case.status, "{}: {result:?}", case.name);
        if let Some(optimum) = case.fun {
            assert!(result.success, "{}: {result:?}", case.name);
            assert_feasible(case, &result);
            assert!(
                (result.fun - optimum).abs() <= 1e-8 * (1.0 + optimum.abs()),
                "{}: {} != {optimum}; {result:?}",
                case.name,
                result.fun
            );
            assert_eq!(result.mip_dual_bound, result.fun);
            assert_eq!(result.mip_gap, 0.0);
            for max_nodes in [1, 2, 4, 8] {
                let limited = milp(
                    problem,
                    MilpOptions {
                        max_nodes: Some(max_nodes),
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(
                    limited.mip_dual_bound <= optimum + 1e-8 * (1.0 + optimum.abs()),
                    "{}: invalid lower bound above true optimum {optimum}: {limited:?}",
                    case.name
                );
                assert!(
                    matches!(limited.status, 0 | 1),
                    "{}: {limited:?}",
                    case.name
                );
                if limited.fun.is_finite() {
                    assert_feasible(case, &limited);
                    assert!(
                        limited.fun >= optimum - 1e-8 * (1.0 + optimum.abs()),
                        "{}: {limited:?}",
                        case.name
                    );
                } else {
                    assert!(limited.x.is_empty());
                    assert!(limited.mip_gap.is_infinite());
                }
                bound_checks += 1;
            }
            let approximate = milp(
                problem,
                MilpOptions {
                    mip_rel_gap: 0.15,
                    max_nodes: Some(20_000),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(approximate.success, "{}: {approximate:?}", case.name);
            assert_feasible(case, &approximate);
            assert!(approximate.mip_dual_bound <= optimum + 1e-8 * (1.0 + optimum.abs()));
            assert!(
                approximate.mip_gap <= 0.15 + 1e-12,
                "{}: {approximate:?}",
                case.name
            );
            assert!(
                approximate.fun - optimum <= 0.15 * approximate.fun.abs() + 1e-8,
                "{}: true objective gap exceeds requested tolerance: {approximate:?}",
                case.name
            );
        } else {
            assert!(!result.success);
            assert!(result.x.is_empty());
            assert!(result.fun.is_nan());
        }
        eprintln!(
            "{}: SciPy status={} fun={:?}; fsci status={} fun={} nodes={} lower={} gap={}",
            case.name,
            case.status,
            case.fun,
            result.status,
            result.fun,
            result.mip_node_count,
            result.mip_dual_bound,
            result.mip_gap
        );
        compared += 1;
    }
    assert_eq!(compared, 42);
    assert_eq!(bound_checks, 156);
    eprintln!(
        "MILP: {compared} live SciPy comparisons and {bound_checks} interrupted-search bound checks; SciPy {} / NumPy {}",
        oracle.scipy, oracle.numpy
    );
}
