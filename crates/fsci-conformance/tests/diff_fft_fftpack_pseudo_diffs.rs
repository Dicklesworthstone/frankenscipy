#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the legacy `scipy.fftpack` pseudo-differential operators
//! (`fsci_fft::fftpack`): `diff` (orders −2..5, with and without `period`), `tilbert`,
//! `itilbert`, `hilbert`, `ihilbert`, `cs_diff`, `sc_diff`, `ss_diff`, `cc_diff` and `shift`,
//! over odd and even lengths 1–64 (the Nyquist handling differs between them).
//!
//! Both sides run a real FFT, multiply by the same kernel and invert, so the agreement is at
//! rounding level: max |fsci − SciPy| ≤ 1e-13 · max(1, max |SciPy|).

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_fft::fftpack;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const TOL: f64 = 1.0e-13;

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    op: String,
    x: Vec<f64>,
    order: i32,
    a: f64,
    b: f64,
    period: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    y: Option<Vec<f64>>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn cases() -> Vec<Query> {
    let mut rng = Lcg(77);
    let mut out = Vec::new();
    for &n in &[1usize, 2, 3, 4, 5, 8, 9, 16, 31, 64] {
        let x: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let mut push = |op: &str, order: i32, a: f64, b: f64, period: Option<f64>| {
            out.push(Query {
                id: format!("{op}_n{n}_o{order}_a{a}_b{b}_p{}", period.unwrap_or(0.0)),
                op: op.to_string(),
                x: x.clone(),
                order,
                a,
                b,
                period,
            });
        };
        for order in [-2, -1, 1, 2, 3, 4, 5] {
            push("diff", order, 0.0, 0.0, None);
            push("diff", order, 0.0, 0.0, Some(3.5));
        }
        for period in [None, Some(10.0)] {
            push("tilbert", 0, 0.4, 0.0, period);
            push("itilbert", 0, 0.4, 0.0, period);
            push("cs_diff", 0, 0.3, 0.7, period);
            push("sc_diff", 0, 0.3, 0.7, period);
            push("ss_diff", 0, 0.3, 0.7, period);
            push("cc_diff", 0, 0.3, 0.7, period);
            push("shift", 0, 0.55, 0.0, period);
        }
        push("hilbert", 0, 0.0, 0.0, None);
        push("ihilbert", 0, 0.0, 0.0, None);
    }
    out
}

const ORACLE: &str = r#"
import json, sys
import numpy as np
from scipy import fftpack as fp
out = []
for q in json.load(sys.stdin):
    x = np.array(q["x"], dtype=float)
    op, a, b, p = q["op"], q["a"], q["b"], q["period"]
    try:
        if op == "diff": y = fp.diff(x, q["order"], p)
        elif op == "tilbert": y = fp.tilbert(x, a, p)
        elif op == "itilbert": y = fp.itilbert(x, a, p)
        elif op == "hilbert": y = fp.hilbert(x)
        elif op == "ihilbert": y = fp.ihilbert(x)
        elif op == "cs_diff": y = fp.cs_diff(x, a, b, p)
        elif op == "sc_diff": y = fp.sc_diff(x, a, b, p)
        elif op == "ss_diff": y = fp.ss_diff(x, a, b, p)
        elif op == "cc_diff": y = fp.cc_diff(x, a, b, p)
        elif op == "shift": y = fp.shift(x, a, p)
        out.append({"id": q["id"], "y": [float(v) for v in y]})
    except Exception:
        out.append({"id": q["id"], "y": None})
print(json.dumps(out))
"#;

fn fsci(q: &Query) -> Result<Vec<f64>, fsci_fft::FftError> {
    let (x, a, b, p) = (&q.x, q.a, q.b, q.period);
    match q.op.as_str() {
        "diff" => fftpack::diff(x, q.order, p),
        "tilbert" => fftpack::tilbert(x, a, p),
        "itilbert" => fftpack::itilbert(x, a, p),
        "hilbert" => fftpack::hilbert(x),
        "ihilbert" => fftpack::ihilbert(x),
        "cs_diff" => fftpack::cs_diff(x, a, b, p),
        "sc_diff" => fftpack::sc_diff(x, a, b, p),
        "ss_diff" => fftpack::ss_diff(x, a, b, p),
        "cc_diff" => fftpack::cc_diff(x, a, b, p),
        "shift" => fftpack::shift(x, a, p),
        other => panic!("unknown op {other}"),
    }
}

#[test]
fn diff_fft_fftpack_pseudo_diffs() {
    let queries = cases();
    let payload = serde_json::to_string(&queries).expect("serialize");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "spawn oracle: {e}"
            );
            eprintln!("skipping fftpack oracle: {e}");
            return;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write query");
    let output = child.wait_with_output().expect("oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping fftpack oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let ops = [
        "diff", "tilbert", "itilbert", "hilbert", "ihilbert", "cs_diff", "sc_diff", "ss_diff",
        "cc_diff", "shift",
    ];
    let mut ledger = CompareLedger::new("diff_fft_fftpack_pseudo_diffs", &ops);
    let mut worst = 0.0_f64;
    let mut failures = Vec::new();
    for (q, arm) in queries.iter().zip(&arms) {
        assert_eq!(q.id, arm.id);
        let ours = fsci(q).ok();
        let Some((scipy, ours)) = ledger.slices(&q.op, &q.id, arm.y.as_deref(), ours.as_deref())
        else {
            failures.push(format!("{}: missing or non-finite value", q.id));
            continue;
        };
        let scale = scipy.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        let err = ours
            .iter()
            .zip(scipy)
            .map(|(a, b)| (a - b).abs() / scale)
            .fold(0.0, f64::max);
        worst = worst.max(err);
        let pass = err <= TOL;
        if !pass {
            failures.push(format!("{}: rel err {err:e}", q.id));
        }
        ledger.compared(&q.op, &q.id, pass);
    }
    for f in failures.iter().take(10) {
        println!("FAIL {f}");
    }
    println!(
        "{} cases compared, worst rel err {worst:.3e}, {} failed",
        queries.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "fftpack pseudo-differential operators differ from SciPy"
    );
    ledger.finish(1);
}
