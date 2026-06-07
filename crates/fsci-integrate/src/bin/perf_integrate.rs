//! Profiling-only harness for integrate hot paths.
//!
//! NOT a product binary. It exists so RCH, Criterion follow-ups, and sha256
//! checks can attach to deterministic `solve_ivp` scenarios.
//!
//! Usage:
//!   `perf_integrate lorenz <repeats>`
//!   `perf_integrate exponential <repeats>`
//!   `perf_integrate golden [path]`

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_integrate::{SolveIvpOptions, SolveIvpResult, SolverKind, ToleranceValue, solve_ivp};
use fsci_runtime::RuntimeMode;

fn exponential_decay(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-y[0]]
}

fn lorenz(_t: f64, y: &[f64]) -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2],
    ]
}

fn exponential_options<'a>(y0: &'a [f64; 1]) -> SolveIvpOptions<'a> {
    SolveIvpOptions {
        t_span: (0.0, 10.0),
        y0,
        method: SolverKind::Rk45,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

fn lorenz_options<'a>(y0: &'a [f64; 3]) -> SolveIvpOptions<'a> {
    SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0,
        method: SolverKind::Rk45,
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

fn solve_exponential() -> SolveIvpResult {
    let y0 = [1.0];
    let opts = exponential_options(&y0);
    let mut rhs = exponential_decay;
    solve_ivp(&mut rhs, &opts).expect("solve exponential")
}

fn solve_lorenz() -> SolveIvpResult {
    let y0 = [1.0, 1.0, 1.0];
    let opts = lorenz_options(&y0);
    let mut rhs = lorenz;
    solve_ivp(&mut rhs, &opts).expect("solve lorenz")
}

fn write_result(output: &mut String, label: &str, result: &SolveIvpResult) {
    writeln!(
        output,
        "case={label} status={} success={} nfev={} njev={} nlu={} sol={} t_events={} y_events={} message={:?}",
        result.status,
        u8::from(result.success),
        result.nfev,
        result.njev,
        result.nlu,
        u8::from(result.sol.is_some()),
        u8::from(result.t_events.is_some()),
        u8::from(result.y_events.is_some()),
        result.message,
    )
    .expect("write result header");

    output.push_str("t=");
    for value in &result.t {
        write!(output, "{:016x},", value.to_bits()).expect("write t bits");
    }
    output.push('\n');

    output.push_str("y=");
    for row in &result.y {
        output.push('[');
        for value in row {
            write!(output, "{:016x},", value.to_bits()).expect("write y bits");
        }
        output.push(']');
    }
    output.push('\n');
}

fn golden_text() -> String {
    let mut output = String::new();
    write_result(
        &mut output,
        "exponential-rk45-no-output",
        &solve_exponential(),
    );
    write_result(&mut output, "lorenz-rk45-no-output", &solve_lorenz());
    output
}

fn write_or_print_golden(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        print!("{output}");
    }
}

fn timed_mode(mode: &str, repeats: usize) {
    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    let mut nfev = 0usize;
    for _ in 0..repeats {
        let result = match mode {
            "lorenz" => solve_lorenz(),
            "exponential" => solve_exponential(),
            _ => unreachable!("validated mode"),
        };
        checksum += result
            .y
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .sum::<f64>()
            + result.t.iter().copied().sum::<f64>();
        nfev += result.nfev;
        black_box(&result);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_us = elapsed.as_secs_f64() * 1e6 / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_us\":{per_call_us:.6},\"nfev\":{nfev},\"checksum\":{checksum:.12e}}}",
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("lorenz");

    if mode == "golden" {
        write_or_print_golden(golden_text(), args.get(2).map(String::as_str));
        return;
    }

    if mode != "lorenz" && mode != "exponential" {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }

    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    timed_mode(mode, repeats);
}
