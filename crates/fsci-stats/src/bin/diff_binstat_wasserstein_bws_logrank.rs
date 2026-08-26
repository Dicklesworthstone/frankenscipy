//! Differential oracle probe for `scipy.stats` entry points with NO differential coverage
//! (frankenscipy-ivxx6): `binned_statistic_2d`, `wasserstein_distance_nd`, `bws_test`, `logrank`.
//!
//! Each was confirmed at zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus.
//!
//! Randomised entry points from the same backlog -- `monte_carlo_test`, `sobol_indices`,
//! `dunnett` -- are deliberately excluded: comparing two draws is not comparing two answers.
//!
//! FIXTURE NOTES.
//!   * `binned_statistic_2d` uses a grid coarse enough that some bins are EMPTY, because the empty
//!     bin is where the two libraries could disagree (NaN vs 0) and a dense fixture would never
//!     reach it. Every statistic is swept.
//!   * `bws_test` and `logrank` are run under EVERY alternative. A two-sided-only probe cannot see
//!     a tail being taken on the wrong side.
//!   * `wasserstein_distance_nd` is probed weighted and unweighted; the weighted arm is where a
//!     normalisation convention would show.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_binstat_wasserstein_bws_logrank.py`.
use fsci_stats::{binned_statistic_2d, bws_test, logrank, wasserstein_distance_nd};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v
        .iter()
        .map(|x| {
            if x.is_nan() {
                "nan".to_string()
            } else {
                format!("{x:.17e}")
            }
        })
        .collect();
    println!("{name}|{}", s.join(";"));
}

fn main() {
    // Scattered 2-D samples; the 4x4 grid leaves several bins empty on purpose.
    let x: Vec<f64> = vec![
        0.1, 0.4, 0.6, 0.9, 0.2, 0.75, 0.35, 0.95, 0.05, 0.55, 0.8, 0.45,
    ];
    let y: Vec<f64> = vec![
        0.2, 0.1, 0.7, 0.3, 0.85, 0.55, 0.4, 0.95, 0.6, 0.15, 0.9, 0.5,
    ];
    let vals: Vec<f64> = vec![
        1.0, 2.5, -1.0, 4.0, 0.5, 3.5, -2.0, 6.0, 1.5, 2.0, -0.5, 3.0,
    ];

    for stat in ["mean", "sum", "count", "median", "min", "max", "std"] {
        let (s, xe, ye) = binned_statistic_2d(&x, &y, &vals, 4, stat);
        let flat: Vec<f64> = s.iter().flat_map(|r| r.iter().copied()).collect();
        dump(&format!("binstat2d_{stat}"), &flat);
        if stat == "mean" {
            dump("binstat2d_xedges", &xe);
            dump("binstat2d_yedges", &ye);
        }
    }

    // ---- wasserstein_distance_nd -------------------------------------------------------------
    let u: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.5],
        vec![0.5, 1.5],
        vec![2.0, 1.0],
    ];
    let v: Vec<Vec<f64>> = vec![vec![0.5, 0.25], vec![1.5, 1.0], vec![2.5, 0.5]];
    dump(
        "wass_nd_plain",
        &[wasserstein_distance_nd(&u, &v, None, None)],
    );
    let uw = [1.0, 2.0, 0.5, 1.5];
    let vw = [2.0, 1.0, 3.0];
    dump(
        "wass_nd_weighted",
        &[wasserstein_distance_nd(&u, &v, Some(&uw), Some(&vw))],
    );

    // ---- bws_test, every alternative ---------------------------------------------------------
    let a = [1.2, 2.4, 0.7, 3.1, 1.9, 2.8, 0.4];
    let b = [2.9, 3.6, 1.8, 4.2, 3.3, 2.2];
    let mut bws = Vec::new();
    for alt in ["two-sided", "less", "greater"] {
        match bws_test(&a, &b, alt, None) {
            Ok(r) => {
                bws.push(r.statistic);
                bws.push(r.pvalue);
            }
            Err(_) => {
                bws.push(f64::NAN);
                bws.push(f64::NAN);
            }
        }
    }
    dump("bws", &bws);

    // ---- logrank, every alternative ----------------------------------------------------------
    let s1 = [6.0, 7.0, 10.0, 15.0, 19.0, 25.0, 30.0];
    let s2 = [4.0, 8.0, 11.0, 13.0, 16.0, 21.0];
    let mut lr = Vec::new();
    for alt in ["two-sided", "less", "greater"] {
        let r = logrank(&s1, &s2, alt);
        lr.push(r.statistic);
        lr.push(r.pvalue);
    }
    dump("logrank", &lr);
}
