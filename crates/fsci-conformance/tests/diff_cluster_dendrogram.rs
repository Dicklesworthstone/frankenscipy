#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)`
//! against `fsci_cluster::dendrogram`: `icoord`, `dcoord`, `ivl`, `leaves`, `color_list` and
//! `leaves_color_list`, over linkages from four methods and sizes 2–60, with every truncation
//! mode, both sort keys in both directions, colour thresholds below/inside/above the tree,
//! custom labels, a custom palette and `show_leaf_counts=False`.
//!
//! Both sides lay out the SAME linkage matrix (fsci's `linkage`, sent to SciPy), and every
//! coordinate is sums and midpoints of multiples of 5 or a copied merge height, so all outputs
//! are compared for exact equality.

use std::io::Write;
use std::process::Stdio;

use fsci_cluster::{
    Dendrogram, DendrogramOptions, DendrogramSort, DendrogramTruncate, LinkageMethod, dendrogram,
    linkage,
};
use fsci_conformance::CompareLedger;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    z: Vec<[f64; 4]>,
    truncate_mode: Option<String>,
    p: usize,
    color_threshold: Option<f64>,
    count_sort: Option<String>,
    distance_sort: Option<String>,
    show_leaf_counts: bool,
    labels: Option<Vec<String>>,
    palette: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    ok: bool,
    icoord: Vec<[f64; 4]>,
    dcoord: Vec<[f64; 4]>,
    ivl: Vec<String>,
    leaves: Vec<usize>,
    color_list: Vec<String>,
    leaves_color_list: Vec<Option<String>>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn points(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    // Three loose blobs so the colour threshold splits the tree into subtrees.
    (0..n)
        .map(|i| {
            let c = (i % 3) as f64 * 4.0;
            vec![c + rng.next(), c * 0.5 + rng.next(), rng.next()]
        })
        .collect()
}

struct Case {
    query: Query,
    truncate: DendrogramTruncate,
    sort: DendrogramSort,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let methods = [
        ("single", LinkageMethod::Single),
        ("complete", LinkageMethod::Complete),
        ("average", LinkageMethod::Average),
        ("ward", LinkageMethod::Ward),
    ];
    let variants: Vec<(&str, DendrogramTruncate, DendrogramSort)> = vec![
        ("plain", DendrogramTruncate::None, DendrogramSort::None),
        ("lastp4", DendrogramTruncate::LastP(4), DendrogramSort::None),
        ("lastp8", DendrogramTruncate::LastP(8), DendrogramSort::None),
        ("lastp0", DendrogramTruncate::LastP(0), DendrogramSort::None),
        ("level2", DendrogramTruncate::Level(2), DendrogramSort::None),
        ("level3", DendrogramTruncate::Level(3), DendrogramSort::None),
        (
            "count_asc",
            DendrogramTruncate::None,
            DendrogramSort::CountAscending,
        ),
        (
            "count_desc",
            DendrogramTruncate::None,
            DendrogramSort::CountDescending,
        ),
        (
            "dist_asc",
            DendrogramTruncate::None,
            DendrogramSort::DistanceAscending,
        ),
        (
            "dist_desc",
            DendrogramTruncate::None,
            DendrogramSort::DistanceDescending,
        ),
        (
            "lastp5_count_desc",
            DendrogramTruncate::LastP(5),
            DendrogramSort::CountDescending,
        ),
    ];
    for (mi, (mname, method)) in methods.iter().enumerate() {
        for (si, &n) in [2usize, 3, 5, 10, 25, 60].iter().enumerate() {
            let data = points(n, 900 + 10 * mi as u64 + si as u64);
            let z = linkage(&data, *method).expect("linkage");
            for (vname, truncate, sort) in &variants {
                let (truncate_mode, p) = match truncate {
                    DendrogramTruncate::None => (None, 30),
                    DendrogramTruncate::LastP(p) => (Some("lastp".to_string()), *p),
                    DendrogramTruncate::Level(p) => (Some("level".to_string()), *p),
                };
                let (count_sort, distance_sort) = match sort {
                    DendrogramSort::None => (None, None),
                    DendrogramSort::CountAscending => (Some("ascending".to_string()), None),
                    DendrogramSort::CountDescending => (Some("descending".to_string()), None),
                    DendrogramSort::DistanceAscending => (None, Some("ascending".to_string())),
                    DendrogramSort::DistanceDescending => (None, Some("descending".to_string())),
                };
                out.push(Case {
                    query: Query {
                        id: format!("{mname}_n{n}_{vname}"),
                        z: z.clone(),
                        truncate_mode,
                        p,
                        color_threshold: None,
                        count_sort,
                        distance_sort,
                        show_leaf_counts: true,
                        labels: None,
                        palette: None,
                    },
                    truncate: *truncate,
                    sort: *sort,
                });
            }
            // Colour thresholds, labels, palette and leaf-count labels on the plain layout.
            let max_h = z.iter().map(|r| r[2]).fold(0.0, f64::max);
            let extras: Vec<(&str, Option<f64>, bool, bool, bool)> = vec![
                ("ct0", Some(0.0), false, false, true),
                ("ct_mid", Some(max_h * 0.3), false, false, true),
                ("ct_high", Some(max_h * 2.0), false, false, true),
                ("labels", None, true, false, true),
                ("palette", Some(max_h * 0.5), false, true, true),
                ("lastp3_nocounts", None, false, false, false),
            ];
            for (ename, ct, use_labels, use_palette, counts) in extras {
                let lastp = ename == "lastp3_nocounts";
                out.push(Case {
                    query: Query {
                        id: format!("{mname}_n{n}_{ename}"),
                        z: z.clone(),
                        truncate_mode: lastp.then(|| "lastp".to_string()),
                        p: if lastp { 3 } else { 30 },
                        color_threshold: ct,
                        count_sort: None,
                        distance_sort: None,
                        show_leaf_counts: counts,
                        labels: use_labels.then(|| (0..n).map(|i| format!("obs{i}")).collect()),
                        palette: use_palette
                            .then(|| vec!["r".to_string(), "g".to_string(), "b".to_string()]),
                    },
                    truncate: if lastp {
                        DendrogramTruncate::LastP(3)
                    } else {
                        DendrogramTruncate::None
                    },
                    sort: DendrogramSort::None,
                });
            }
        }
    }
    out
}

const ORACLE: &str = r#"
import json, sys
import numpy as np
from scipy.cluster import hierarchy as h
out = []
for q in json.load(sys.stdin):
    arm = {"id": q["id"], "ok": False, "icoord": [], "dcoord": [], "ivl": [], "leaves": [],
           "color_list": [], "leaves_color_list": []}
    try:
        h.set_link_color_palette(q["palette"])
        r = h.dendrogram(np.array(q["z"], dtype=float), p=q["p"], truncate_mode=q["truncate_mode"],
                         color_threshold=q["color_threshold"],
                         count_sort=q["count_sort"] or False,
                         distance_sort=q["distance_sort"] or False,
                         show_leaf_counts=q["show_leaf_counts"], labels=q["labels"], no_plot=True)
        arm.update(ok=True, icoord=[[float(v) for v in x] for x in r["icoord"]],
                   dcoord=[[float(v) for v in x] for x in r["dcoord"]],
                   ivl=[str(s) for s in r["ivl"]], leaves=[int(v) for v in r["leaves"]],
                   color_list=list(r["color_list"]), leaves_color_list=list(r["leaves_color_list"]))
    except Exception as e:
        arm["ivl"] = [f"{type(e).__name__}: {e}"]
    finally:
        h.set_link_color_palette(None)
    out.append(arm)
print(json.dumps(out))
"#;

fn fsci_layout(case: &Case) -> Result<Dendrogram, String> {
    let q = &case.query;
    let palette: Vec<&str> = q
        .palette
        .as_ref()
        .map(|p| p.iter().map(String::as_str).collect())
        .unwrap_or_else(|| fsci_cluster::DENDROGRAM_DEFAULT_PALETTE.to_vec());
    let options = DendrogramOptions {
        truncate: case.truncate,
        color_threshold: q.color_threshold,
        labels: q.labels.as_deref(),
        sort: case.sort,
        show_leaf_counts: q.show_leaf_counts,
        palette: &palette,
        ..DendrogramOptions::default()
    };
    dendrogram(&q.z, options).map_err(|e| e.to_string())
}

#[test]
fn diff_cluster_dendrogram() {
    let cases = cases();
    let queries: Vec<&Query> = cases.iter().map(|c| &c.query).collect();
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
            eprintln!("skipping dendrogram oracle: {e}");
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
        eprintln!("skipping dendrogram oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut ledger = CompareLedger::new("diff_cluster_dendrogram", &["dendrogram"]);
    let mut failures = Vec::new();
    for (case, arm) in cases.iter().zip(&arms) {
        assert_eq!(case.query.id, arm.id);
        if !arm.ok {
            ledger.oracle_missing("dendrogram", &arm.id, &arm.ivl.join(""));
            failures.push(format!("{}: SciPy failed {:?}", arm.id, arm.ivl));
            continue;
        }
        match fsci_layout(case) {
            Err(e) => {
                ledger.rust_failed("dendrogram", &arm.id, &e);
                failures.push(format!("{}: fsci error {e}", arm.id));
            }
            Ok(d) => {
                let mut diffs = Vec::new();
                if d.icoord != arm.icoord {
                    diffs.push("icoord");
                }
                if d.dcoord != arm.dcoord {
                    diffs.push("dcoord");
                }
                if d.ivl != arm.ivl {
                    diffs.push("ivl");
                }
                if d.leaves != arm.leaves {
                    diffs.push("leaves");
                }
                if d.color_list != arm.color_list {
                    diffs.push("color_list");
                }
                if d.leaves_color_list != arm.leaves_color_list {
                    diffs.push("leaves_color_list");
                }
                if !diffs.is_empty() {
                    failures.push(format!("{}: {} differ", arm.id, diffs.join(", ")));
                }
                ledger.compared("dendrogram", &arm.id, diffs.is_empty());
            }
        }
    }
    for f in failures.iter().take(20) {
        println!("FAIL {f}");
    }
    println!(
        "{} layouts compared, {} differ",
        cases.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "dendrogram differs from SciPy");
    ledger.finish(cases.len());
}
