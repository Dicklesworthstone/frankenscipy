#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.linalg.interpolative.estimate_rank(A, eps, rng)`
//! against `fsci_linalg::interpolative::estimate_rank`.
//!
//! Both sides sketch `A` with a RANDOM subsampled Fourier transform, and fsci does not reproduce
//! numpy's random stream, so the comparison is only meaningful where the answer does not depend
//! on the stream. The ORACLE decides that, not this file: SciPy runs every matrix with 40
//! seeds, and a matrix whose 40 SciPy estimates agree is "seed-stable"; there every one of six
//! fsci seeds must give exactly SciPy's estimate. The corpus is built so that this is most of
//! it (a rank-`r` part plus a tail far below `eps`, over shapes on both sides of SciPy's
//! `r + 12 < min(n, n₂)` full-rank cut-off, and `eps` on both sides of the tail).
//!
//! Matrices with a tail AT the threshold are included on purpose: there the estimate is a
//! random variable on both sides. Six SciPy seeds were not enough to see that (one such matrix
//! gave SciPy's full-rank 150 six times while 300 seeds put 25% of the mass on 118–122, where
//! fsci's other seeds landed). For those, every fsci estimate must be one SciPy also produces in
//! kind: the full-rank answer only if SciPy gave it, otherwise a value within one of the range
//! of SciPy's non-full-rank estimates. The test fails if fewer than 85% of the matrices are
//! seed-stable, so the exact comparison cannot quietly shrink.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_linalg::interpolative::estimate_rank;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const SCIPY_SEEDS: u64 = 40;
const FSCI_SEEDS: u64 = 6;

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    a: Vec<Vec<f64>>,
    eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    ranks: Option<Vec<usize>>,
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

/// `Σᵢ sᵢ·uᵢ·vᵢᵀ` with random `uᵢ`, `vᵢ`: `r` terms of size 1 and `tail_terms` of size `tail`.
fn matrix(m: usize, n: usize, r: usize, tail: f64, tail_terms: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    let mut a = vec![vec![0.0; n]; m];
    for t in 0..r + tail_terms {
        let s = if t < r { 1.0 } else { tail };
        let u: Vec<f64> = (0..m).map(|_| rng.next()).collect();
        let v: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        for (row, ui) in a.iter_mut().zip(&u) {
            for (x, vj) in row.iter_mut().zip(&v) {
                *x += s * ui * vj;
            }
        }
    }
    a
}

fn corpus() -> Vec<Query> {
    let mut out = Vec::new();
    let shapes = [
        (3, 4),
        (5, 5),
        (9, 6),
        (17, 40),
        (33, 33),
        (40, 30),
        (64, 64),
        (65, 50),
        (100, 80),
        (129, 60),
        (150, 150),
        (200, 30),
        (30, 200),
        (257, 120),
    ];
    let mut seed = 1000;
    for &(m, n) in &shapes {
        let full = m.min(n);
        for r in [
            0usize,
            1,
            3,
            full / 4,
            full / 2,
            full.saturating_sub(1),
            full,
        ] {
            for (tag, tail, eps) in [
                ("clean", 0.0, 1e-10),
                ("tail_far", 1e-13, 1e-8),
                ("tail_far2", 1e-9, 1e-4),
                ("tail_edge", 1e-10, 1e-10),
            ] {
                seed += 1;
                let tail_terms = if tail == 0.0 { 0 } else { full - r.min(full) };
                out.push(Query {
                    id: format!("m{m}_n{n}_r{r}_{tag}"),
                    a: matrix(m, n, r.min(full), tail, tail_terms, seed),
                    eps,
                });
            }
        }
    }
    out
}

const ORACLE: &str = r#"
import json, sys
import numpy as np
from scipy.linalg import interpolative as sli
out = []
for q in json.load(sys.stdin):
    try:
        a = np.array(q["a"], dtype=float)
        ranks = [int(sli.estimate_rank(a, q["eps"], rng=s)) for s in range(SCIPY_SEEDS)]
        out.append({"id": q["id"], "ranks": ranks})
    except Exception:
        out.append({"id": q["id"], "ranks": None})
print(json.dumps(out))
"#;

#[test]
fn diff_linalg_estimate_rank() {
    let queries = corpus();
    let payload = serde_json::to_string(&queries).expect("serialize");
    let oracle = ORACLE.replace("SCIPY_SEEDS", &SCIPY_SEEDS.to_string());
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&oracle)
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
            eprintln!("skipping estimate_rank oracle: {e}");
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
        eprintln!("skipping estimate_rank oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut ledger = CompareLedger::new("diff_linalg_estimate_rank", &["estimate_rank"]);
    let (mut stable, mut spread) = (0usize, 0usize);
    let mut failures = Vec::new();
    for (q, arm) in queries.iter().zip(&arms) {
        assert_eq!(q.id, arm.id);
        let Some(scipy) = &arm.ranks else {
            ledger.oracle_missing("estimate_rank", &q.id, "SciPy raised");
            failures.push(format!("{}: SciPy raised", q.id));
            continue;
        };
        let ours: Result<Vec<usize>, _> = (0..FSCI_SEEDS)
            .map(|s| estimate_rank(&q.a, q.eps, s))
            .collect();
        let ours = match ours {
            Ok(v) => v,
            Err(e) => {
                ledger.rust_failed("estimate_rank", &q.id, &format!("{e:?}"));
                failures.push(format!("{}: fsci error {e:?}", q.id));
                continue;
            }
        };
        let lo = *scipy.iter().min().expect("SciPy seeds");
        let hi = *scipy.iter().max().expect("SciPy seeds");
        let pass = if lo == hi {
            stable += 1;
            ours.iter().all(|&r| r == lo)
        } else {
            spread += 1;
            let full = q.a.len().min(q.a.first().map_or(0, Vec::len));
            let partial: Vec<usize> = scipy.iter().copied().filter(|&r| r != full).collect();
            ours.iter().all(|&r| {
                if r == full {
                    scipy.contains(&full)
                } else {
                    let plo = partial.iter().min().copied().unwrap_or(usize::MAX);
                    let phi = partial.iter().max().copied().unwrap_or(0);
                    r + 1 >= plo && r <= phi + 1
                }
            })
        };
        if !pass {
            failures.push(format!("{}: fsci {ours:?} vs SciPy {scipy:?}", q.id));
        }
        ledger.compared("estimate_rank", &q.id, pass);
    }
    for f in failures.iter().take(15) {
        println!("FAIL {f}");
    }
    println!(
        "{} matrices: {stable} seed-stable over {SCIPY_SEEDS} SciPy seeds (exact match required), \
         {spread} seed-dependent (fsci within SciPy's support); {} failed",
        queries.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "estimate_rank differs from SciPy");
    assert!(
        stable * 100 >= queries.len() * 85,
        "only {stable} of {} matrices are seed-stable in SciPy; the exact comparison shrank",
        queries.len()
    );
    ledger.finish(queries.len());
}
