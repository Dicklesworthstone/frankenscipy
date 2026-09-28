#![forbid(unsafe_code)]
//! Live SciPy diff for LSODA (frankenscipy-1ksfv.9): `solve_ivp(method="LSODA")`, with and
//! without `jac`, its `t_eval` samples, and `odeint`, against SciPy 1.17.1, whose LSODA is
//! ODEPACK's as translated to C in `scipy/integrate/src/lsoda.c`, the file `fsci_integrate::lsoda`
//! ports.
//!
//! Problems (right-hand sides written with the same operations in the same order on both sides,
//! none of them a fixture of the unit tests in `fsci-integrate`): Robertson's kinetics (n = 3,
//! vector atol, short and long spans, and with first_step/max_step), van der Pol at mu = 100
//! (two tolerances), HIRES (n = 8), the Arenstorf orbit (nonstiff, yet SciPy switches to BDF and
//! back once), a pendulum integrated backward in time (Adams only), and the Oregonator (eleven
//! method switches). Each is integrated by both sides and compared on:
//! - the step sequence: the number of output times, nfev, njev and nlu, and the method-switch
//!   history (the steps at which ODEPACK's MUSED, SciPy's `iwork[18]`, changed, with the new
//!   method), all exactly;
//! - the t grid: every step time within T_GRID_REL_TOL of SciPy's, relative to the span;
//! - y at every step within Y_SCALE_TOL units of SciPy's own error scale `rtol |y| + atol`.
//!
//! On the pinned host every step row and every `odeint` row is bit-identical to SciPy (the log
//! prints the largest ulp distance per row); the tolerances only leave room for last-bit LAPACK
//! differences on other hosts. The counts are compared for equality because SciPy's are not
//! kernel-dependent on these rows: this test run with SciPy's OpenBLAS on its default Haswell
//! kernel (FMA) and under `OPENBLAS_CORETYPE=Nehalem`, `Sandybridge` and `Prescott` (reported as
//! Katmai) gives the same counts and switch histories and the same 15 bit-identical rows.
//!
//! `t_eval` samples are SciPy's `LsodaDenseOutput`, `np.dot(yh, ((t - t_n)/h) ** p)` over the
//! Nordsieck history. numpy's `dot` is a BLAS `dgemv` whose summation order is its own; fsci sums
//! left to right. Measured on the pinned host, SciPy's `dot` and a left-to-right sum over the SAME
//! history differ by up to 7.16e-7 units on the Arenstorf row, exactly the gap fsci shows there
//! with the Haswell kernel (2.08e-7 with the other three kernels above), so these samples are
//! held to DENSE_SCALE_TOL units: far above summation noise, far below what
//! a wrong interpolant (wrong h, missing the `(h/hu)^q` rescale of a column ODEPACK has not yet
//! rescaled) moves a sample, which is of the order of the local error, i.e. of one unit.
//!
//! `odeint` rows compare every output row and nfev (counted by the right-hand side) at SciPy's
//! defaults (rtol = atol = 1.49012e-8). The compared count per arm is asserted.

use std::cell::Cell;
use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_integrate::{
    JacFn, LSODA, LsodaMethod, LsodaSolverConfig, OdeSolverState, SolveIvpOptions, SolverKind,
    ToleranceValue, odeint, solve_ivp,
};
use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// `solve_ivp` step by step (finite-difference Jacobian, jt = 2), with a user Jacobian (jt = 1),
/// `t_eval` samples from the dense output, and `odeint`.
const ARMS: [&str; 4] = ["solve_ivp", "solve_ivp_jac", "t_eval", "odeint"];
/// Steps and `odeint` rows: |fsci - SciPy| <= Y_SCALE_TOL * (rtol |y_scipy| + atol), per
/// component and output time (0 on the pinned host).
const Y_SCALE_TOL: f64 = 1e-6;
/// `t_eval` samples, the same measure: numpy's `dot` summation order is worth 7.16e-7 units.
const DENSE_SCALE_TOL: f64 = 1e-4;
/// |t_fsci - t_scipy| <= T_GRID_REL_TOL * |t_end - t0| at every step.
const T_GRID_REL_TOL: f64 = 1e-12;

const MU: f64 = 0.012_277_471;
/// The Arenstorf period and initial velocity as the nearest doubles.
const T_ARENSTORF: f64 = 17.065_216_560_157_964;
const ARENSTORF_Y0: [f64; 4] = [0.994, 0.0, 0.0, -2.001_585_106_379_082_4];
/// odeint's default rtol and atol, the error scale its rows are measured in (an input, not a
/// comparison tolerance).
const ODEINT_RTOL_ATOL: f64 = 1.49012e-8;

#[derive(Debug, Clone, Serialize)]
struct Case {
    name: String,
    problem: String,
    arm: String,
    t0: f64,
    t1: f64,
    y0: Vec<f64>,
    rtol: f64,
    /// One value, or one per component.
    atol: Vec<f64>,
    first_step: Option<f64>,
    max_step: Option<f64>,
    jac: bool,
    /// `t_eval` for the t_eval arm, the output times for odeint.
    times: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    status: i32,
    t: Vec<f64>,
    /// y at every output time.
    y: Vec<Vec<f64>>,
    nfev: usize,
    njev: usize,
    nlu: usize,
    /// (nst, MUSED) at each change of MUSED; empty for the sampled arms.
    switches: Vec<(usize, u8)>,
}

fn robertson(y: &[f64]) -> Vec<f64> {
    vec![
        -0.04 * y[0] + 1.0e4 * y[1] * y[2],
        0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
        3.0e7 * y[1] * y[1],
    ]
}

fn robertson_jac(_t: f64, y: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
        vec![0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
        vec![0.0, 6.0e7 * y[1], 0.0],
    ]
}

fn vdp100_jac(_t: f64, y: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 1.0],
        vec![-200.0 * y[0] * y[1] - 1.0, 100.0 * (1.0 - y[0] * y[0])],
    ]
}

fn oregonator_jac(_t: f64, y: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![
            77.27 * (1.0 - 2.0 * 8.375e-6 * y[0] - y[1]),
            77.27 * (1.0 - y[0]),
            0.0,
        ],
        vec![-y[1] / 77.27, -(1.0 + y[0]) / 77.27, 1.0 / 77.27],
        vec![0.161, 0.0, -0.161],
    ]
}

fn rhs(problem: &str, _t: f64, y: &[f64]) -> Vec<f64> {
    match problem {
        "robertson" => robertson(y),
        "vdp100" => vec![y[1], 100.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]],
        "hires" => vec![
            -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007,
            1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
            8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
            -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
            280.0 * y[5] * y[7] - 1.81 * y[6],
            -280.0 * y[5] * y[7] + 1.81 * y[6],
        ],
        "pendulum" => vec![y[1], -y[0].sin()],
        "oregonator" => vec![
            77.27 * (y[1] + y[0] * (1.0 - 8.375e-6 * y[0] - y[1])),
            (y[2] - (1.0 + y[0]) * y[1]) / 77.27,
            0.161 * (y[0] - y[2]),
        ],
        _ => {
            let mup = 1.0 - MU;
            let (x, yy, vx, vy) = (y[0], y[1], y[2], y[3]);
            let (a, b) = (x + MU, x - mup);
            let r1 = (a * a + yy * yy).sqrt();
            let r2 = (b * b + yy * yy).sqrt();
            let (d1, d2) = (r1 * r1 * r1, r2 * r2 * r2);
            vec![
                vx,
                vy,
                x + 2.0 * vy - mup * a / d1 - MU * b / d2,
                yy - 2.0 * vx - mup * yy / d1 - MU * yy / d2,
            ]
        }
    }
}

fn jac_of(problem: &str) -> JacFn {
    match problem {
        "robertson" => robertson_jac,
        "vdp100" => vdp100_jac,
        _ => oregonator_jac,
    }
}

fn case(
    arm: &str,
    problem: &str,
    label: &str,
    span: (f64, f64),
    y0: &[f64],
    rtol: f64,
    atol: &[f64],
    steps: (Option<f64>, Option<f64>),
    times: Option<Vec<f64>>,
) -> Case {
    Case {
        name: format!("{arm}_{problem}_{label}"),
        problem: problem.to_owned(),
        arm: arm.to_owned(),
        t0: span.0,
        t1: span.1,
        y0: y0.to_vec(),
        rtol,
        atol: atol.to_vec(),
        first_step: steps.0,
        max_step: steps.1,
        jac: arm == "solve_ivp_jac",
        times,
    }
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * (i as f64) / ((n - 1) as f64))
        .collect()
}

fn cases() -> Vec<Case> {
    let rob0 = [1.0, 0.0, 0.0];
    let rob_atol = [1e-8, 1e-14, 1e-6];
    let hires0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057];
    let none = (None, None);
    let mut out = vec![
        case(
            "solve_ivp",
            "robertson",
            "t40",
            (0.0, 40.0),
            &rob0,
            1e-4,
            &rob_atol,
            none,
            None,
        ),
        case(
            "solve_ivp",
            "robertson",
            "t4e5",
            (0.0, 4.0e5),
            &rob0,
            1e-6,
            &[1e-10, 1e-14, 1e-8],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "robertson",
            "first_max_step",
            (0.0, 40.0),
            &rob0,
            1e-4,
            &rob_atol,
            (Some(1e-5), Some(2.0)),
            None,
        ),
        case(
            "solve_ivp",
            "vdp100",
            "default_tol",
            (0.0, 300.0),
            &[2.0, 0.0],
            1e-3,
            &[1e-6],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "vdp100",
            "rtol1e-7",
            (0.0, 300.0),
            &[2.0, 0.0],
            1e-7,
            &[1e-9],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "hires",
            "rtol1e-6",
            (0.0, 321.8122),
            &hires0,
            1e-6,
            &[1e-10],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "arenstorf",
            "rtol1e-9",
            (0.0, T_ARENSTORF),
            &ARENSTORF_Y0,
            1e-9,
            &[1e-12],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "pendulum",
            "backward",
            (8.0, 0.0),
            &[1.2, 0.0],
            1e-8,
            &[1e-10],
            none,
            None,
        ),
        case(
            "solve_ivp",
            "oregonator",
            "rtol1e-6",
            (0.0, 360.0),
            &[1.0, 2.0, 3.0],
            1e-6,
            &[1e-8],
            none,
            None,
        ),
        case(
            "solve_ivp_jac",
            "robertson",
            "t40",
            (0.0, 40.0),
            &rob0,
            1e-4,
            &rob_atol,
            none,
            None,
        ),
        case(
            "solve_ivp_jac",
            "vdp100",
            "default_tol",
            (0.0, 300.0),
            &[2.0, 0.0],
            1e-3,
            &[1e-6],
            none,
            None,
        ),
        case(
            "solve_ivp_jac",
            "oregonator",
            "rtol1e-6",
            (0.0, 360.0),
            &[1.0, 2.0, 3.0],
            1e-6,
            &[1e-8],
            none,
            None,
        ),
        case(
            "t_eval",
            "vdp100",
            "61pts",
            (0.0, 300.0),
            &[2.0, 0.0],
            1e-3,
            &[1e-6],
            none,
            Some(linspace(0.0, 300.0, 61)),
        ),
        case(
            "t_eval",
            "arenstorf",
            "41pts",
            (0.0, T_ARENSTORF),
            &ARENSTORF_Y0,
            1e-9,
            &[1e-12],
            none,
            Some(linspace(0.0, T_ARENSTORF, 41)),
        ),
        case(
            "t_eval",
            "robertson",
            "log",
            (0.0, 4.0e4),
            &rob0,
            1e-6,
            &[1e-10, 1e-14, 1e-8],
            none,
            Some(vec![0.0, 1e-4, 1e-2, 0.4, 4.0, 40.0, 400.0, 4000.0, 4.0e4]),
        ),
    ];
    for (problem, y0, times) in [
        (
            "robertson",
            rob0.to_vec(),
            vec![0.0, 0.4, 4.0, 40.0, 400.0, 4000.0, 40000.0],
        ),
        ("vdp100", vec![2.0, 0.0], linspace(0.0, 100.0, 11)),
        (
            "arenstorf",
            ARENSTORF_Y0.to_vec(),
            linspace(0.0, T_ARENSTORF, 9),
        ),
    ] {
        out.push(case(
            "odeint",
            problem,
            "defaults",
            (times[0], times[times.len() - 1]),
            &y0,
            ODEINT_RTOL_ATOL,
            &[ODEINT_RTOL_ATOL],
            none,
            Some(times),
        ));
    }
    out
}

fn scipy_answers(cases: &[Case]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys, math
import numpy as np
from scipy.integrate import solve_ivp, odeint, LSODA

MU = 0.012277471

def robertson(t, y):
    return [-0.04 * y[0] + 1.0e4 * y[1] * y[2],
            0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
            3.0e7 * y[1] * y[1]]

def robertson_jac(t, y):
    return [[-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
            [0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
            [0.0, 6.0e7 * y[1], 0.0]]

def vdp100(t, y):
    return [y[1], 100.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]

def vdp100_jac(t, y):
    return [[0.0, 1.0], [-200.0 * y[0] * y[1] - 1.0, 100.0 * (1.0 - y[0] * y[0])]]

def hires(t, y):
    return [-1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007,
            1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
            8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
            -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
            280.0 * y[5] * y[7] - 1.81 * y[6],
            -280.0 * y[5] * y[7] + 1.81 * y[6]]

def pendulum(t, y):
    return [y[1], -math.sin(y[0])]

def oregonator(t, y):
    return [77.27 * (y[1] + y[0] * (1.0 - 8.375e-6 * y[0] - y[1])),
            (y[2] - (1.0 + y[0]) * y[1]) / 77.27,
            0.161 * (y[0] - y[2])]

def oregonator_jac(t, y):
    return [[77.27 * (1.0 - 2.0 * 8.375e-6 * y[0] - y[1]), 77.27 * (1.0 - y[0]), 0.0],
            [-y[1] / 77.27, -(1.0 + y[0]) / 77.27, 1.0 / 77.27],
            [0.161, 0.0, -0.161]]

def arenstorf(t, y):
    mup = 1.0 - MU
    x, yy, vx, vy = y[0], y[1], y[2], y[3]
    a, b = x + MU, x - mup
    r1 = math.sqrt(a * a + yy * yy)
    r2 = math.sqrt(b * b + yy * yy)
    d1, d2 = r1 * r1 * r1, r2 * r2 * r2
    return [vx, vy, x + 2.0 * vy - mup * a / d1 - MU * b / d2,
            yy - 2.0 * vx - mup * yy / d1 - MU * yy / d2]

RHS = {"robertson": robertson, "vdp100": vdp100, "hires": hires, "pendulum": pendulum,
       "oregonator": oregonator, "arenstorf": arenstorf}
JAC = {"robertson": robertson_jac, "vdp100": vdp100_jac, "oregonator": oregonator_jac}

def solver_kwargs(c):
    kw = {"rtol": c["rtol"], "atol": c["atol"][0] if len(c["atol"]) == 1 else c["atol"]}
    if c["first_step"] is not None:
        kw["first_step"] = c["first_step"]
    if c["max_step"] is not None:
        kw["max_step"] = c["max_step"]
    if c["jac"]:
        kw["jac"] = JAC[c["problem"]]
    return kw

out = []
for c in json.load(sys.stdin):
    fun = RHS[c["problem"]]
    if c["arm"] == "odeint":
        calls = [0]
        def counted(y, t):
            calls[0] += 1
            return fun(t, y)
        y, info = odeint(counted, c["y0"], c["times"], full_output=True)
        ok = info["message"] == "Integration successful."
        out.append({"status": 0 if ok else -1, "t": list(c["times"]),
                    "y": [[float(v) for v in row] for row in y], "nfev": calls[0],
                    "njev": int(info["nje"][-1]), "nlu": int(info["nje"][-1]), "switches": []})
        continue
    kw = solver_kwargs(c)
    r = solve_ivp(fun, (c["t0"], c["t1"]), c["y0"], method="LSODA", t_eval=c["times"], **kw)
    switches = []
    if c["times"] is None:
        s = LSODA(fun, c["t0"], c["y0"], c["t1"], **kw)
        last = None
        while s.status == "running":
            s.step()
            iw = s._lsoda_solver._integrator.iwork
            if int(iw[18]) != last:
                last = int(iw[18])
                switches.append([int(iw[10]), last])
    out.append({"status": int(r.status), "t": [float(v) for v in r.t],
                "y": [[float(v) for v in col] for col in r.y.T], "nfev": int(r.nfev),
                "njev": int(r.njev), "nlu": int(r.nlu), "switches": switches})
print(json.dumps(out))
"#;
    let query = serde_json::to_string(cases).expect("serialize cases");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the LSODA oracle: {e}"
            );
            eprintln!("skipping LSODA oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for the LSODA oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "LSODA oracle failed: {stderr}"
        );
        eprintln!("skipping LSODA oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse LSODA oracle JSON"))
}

/// What fsci produced for one case, in the oracle's shape.
struct Observed {
    status: i32,
    t: Vec<f64>,
    y: Vec<Vec<f64>>,
    nfev: usize,
    njev: usize,
    nlu: usize,
    switches: Vec<(usize, u8)>,
}

fn atol_of(case: &Case) -> ToleranceValue {
    if case.atol.len() == 1 {
        ToleranceValue::Scalar(case.atol[0])
    } else {
        ToleranceValue::Vector(case.atol.clone())
    }
}

fn run_fsci(case: &Case) -> Result<Observed, String> {
    let problem = case.problem.clone();
    let calls = Cell::new(0_usize);
    let mut fun = |t: f64, y: &[f64]| {
        calls.set(calls.get() + 1);
        rhs(&problem, t, y)
    };
    if case.arm == "odeint" {
        let times = case.times.as_deref().expect("odeint times");
        let rows = odeint(&mut |y: &[f64], t: f64| fun(t, y), &case.y0, times)
            .map_err(|e| format!("{e:?}"))?;
        return Ok(Observed {
            status: 0,
            t: times.to_vec(),
            y: rows,
            nfev: calls.get(),
            njev: 0,
            nlu: 0,
            switches: Vec::new(),
        });
    }
    let jac = case.jac.then(|| jac_of(&case.problem));
    let options = SolveIvpOptions {
        t_span: (case.t0, case.t1),
        y0: &case.y0,
        method: SolverKind::Lsoda,
        t_eval: case.times.as_deref(),
        rtol: case.rtol,
        atol: atol_of(case),
        first_step: case.first_step,
        max_step: case.max_step.unwrap_or(f64::INFINITY),
        jac,
        ..SolveIvpOptions::default()
    };
    let r = solve_ivp(&mut fun, &options).map_err(|e| format!("{e:?}"))?;
    let mut switches = Vec::new();
    if case.times.is_none() {
        let mut solver = LSODA::new(LsodaSolverConfig {
            t0: case.t0,
            y0: &case.y0,
            t_bound: case.t1,
            rtol: case.rtol,
            atol: atol_of(case),
            max_step: case.max_step.unwrap_or(f64::INFINITY),
            first_step: case.first_step,
            jac,
            mode: RuntimeMode::Strict,
        })
        .map_err(|e| format!("{e:?}"))?;
        let mut last = None;
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).map_err(|e| format!("{e:?}"))?;
            let used = match solver.method_used() {
                Some(LsodaMethod::Adams) => 1,
                Some(LsodaMethod::Bdf) => 2,
                None => 0,
            };
            if last != Some(used) {
                last = Some(used);
                switches.push((solver.n_steps(), used));
            }
        }
    }
    Ok(Observed {
        status: r.status,
        t: r.t,
        y: r.y,
        nfev: r.nfev,
        njev: r.njev,
        nlu: r.nlu,
        switches,
    })
}

/// Distance in units in the last place (0 = bit-identical).
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn diff_integrate_lsoda() {
    let cases = cases();
    let Some(answers) = scipy_answers(&cases) else {
        return;
    };
    assert_eq!(
        answers.len(),
        cases.len(),
        "the oracle must answer every case"
    );
    let mut ledger = CompareLedger::new("diff_integrate_lsoda", &ARMS);
    let mut failures = Vec::new();
    let mut bit_identical_rows = 0;
    for (case, answer) in cases.iter().zip(&answers) {
        let arm = case.arm.as_str();
        let failures_before = failures.len();
        let observed = run_fsci(case);
        if let Err(e) = &observed {
            failures.push(format!("{}: fsci Err({e})", case.name));
        }
        let Some((answer, got)) = ledger.both(arm, &case.name, Some(answer), observed.ok()) else {
            continue;
        };
        if got.status != 0 || answer.status != 0 {
            failures.push(format!(
                "{}: status fsci {} SciPy {}",
                case.name, got.status, answer.status
            ));
        }
        // Step sequence: exact.
        let counts = (got.t.len(), got.nfev, got.njev, got.nlu);
        let scipy_counts = (answer.t.len(), answer.nfev, answer.njev, answer.nlu);
        let counts_match = if arm == "odeint" {
            (counts.0, counts.1) == (scipy_counts.0, scipy_counts.1)
        } else {
            counts == scipy_counts
        };
        if !counts_match {
            failures.push(format!(
                "{}: (n_t, nfev, njev, nlu) fsci {counts:?} SciPy {scipy_counts:?}",
                case.name
            ));
        }
        if got.switches != answer.switches {
            failures.push(format!(
                "{}: switch history fsci {:?} SciPy {:?}",
                case.name, got.switches, answer.switches
            ));
        }
        // Output times, then every component at every output time; a length mismatch or a
        // non-finite value is recorded by the ledger instead of vanishing in the max folds.
        let scipy_flat: Vec<f64> = answer
            .t
            .iter()
            .chain(answer.y.iter().flatten())
            .copied()
            .collect();
        let fsci_flat: Vec<f64> = got
            .t
            .iter()
            .chain(got.y.iter().flatten())
            .copied()
            .collect();
        if ledger
            .slices(
                arm,
                &case.name,
                Some(scipy_flat.as_slice()),
                Some(fsci_flat.as_slice()),
            )
            .is_none()
        {
            failures.push(format!(
                "{}: t/y shapes or non-finite values differ",
                case.name
            ));
            continue;
        }
        let span = (case.t1 - case.t0).abs();
        let t_gap = got
            .t
            .iter()
            .zip(&answer.t)
            .map(|(f, s)| (f - s).abs() / span)
            .fold(0.0_f64, f64::max);
        let mut y_gap = 0.0_f64;
        let mut max_ulps = 0_u64;
        for (row_f, row_s) in got.y.iter().zip(&answer.y) {
            for (i, (f, s)) in row_f.iter().zip(row_s).enumerate() {
                let atol = if case.atol.len() == 1 {
                    case.atol[0]
                } else {
                    case.atol[i]
                };
                y_gap = y_gap.max((f - s).abs() / (case.rtol * s.abs() + atol));
                max_ulps = max_ulps.max(ulps(*f, *s));
            }
        }
        for (f, s) in got.t.iter().zip(&answer.t) {
            max_ulps = max_ulps.max(ulps(*f, *s));
        }
        if max_ulps == 0 {
            bit_identical_rows += 1;
        }
        println!(
            "{}: n_t {} nfev {} njev {} nlu {} (SciPy {:?}) | switches {} | t gap {t_gap:.3e} | y gap \
             {y_gap:.3e} units | max {max_ulps} ulp",
            case.name,
            got.t.len(),
            got.nfev,
            got.njev,
            got.nlu,
            scipy_counts,
            got.switches.len()
        );
        if t_gap.is_nan() || t_gap > T_GRID_REL_TOL {
            failures.push(format!("{}: t grid gap {t_gap:e}", case.name));
        }
        let y_tol = if arm == "t_eval" {
            DENSE_SCALE_TOL
        } else {
            Y_SCALE_TOL
        };
        if y_gap.is_nan() || y_gap > y_tol {
            failures.push(format!("{}: y gap {y_gap:e} units", case.name));
        }
        ledger.compared(arm, &case.name, failures.len() == failures_before);
    }
    println!(
        "{} LSODA rows compared, {bit_identical_rows} bit-identical to SciPy in every t and y",
        cases.len()
    );
    assert!(
        failures.is_empty(),
        "LSODA disagrees with SciPy: {failures:#?}"
    );
    let min_per_arm = ARMS
        .iter()
        .map(|a| cases.iter().filter(|c| c.arm == *a).count())
        .min()
        .expect("ARMS is non-empty");
    let counts = ledger.finish(min_per_arm);
    println!("compared per arm: {counts:?}");
}
