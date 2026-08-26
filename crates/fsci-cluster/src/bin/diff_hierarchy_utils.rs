//! Differential oracle probe for `scipy.cluster.hierarchy` entry points with NO differential
//! coverage (frankenscipy-ivxx6): `cut_tree`, `leaders`, `maxdists`, `maxinconsts`,
//! `is_isomorphic`, `from_mlab_linkage`.
//!
//! Each was confirmed at zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus.
//!
//! `linkage` ITSELF IS EMITTED AS A CONTROL. Every function here consumes a linkage matrix, so if
//! `Z` already disagreed with SciPy then every downstream group would disagree too and the probe
//! would be blaming the wrong function. The control must agree before any other line is read.
//!
//! CLUSTER LABELS ARE PERMUTATION-ARBITRARY, so `cut_tree` is emitted twice: the raw labels, and
//! a canonicalised form (labels renumbered in order of first appearance). Two implementations can
//! produce the same PARTITION under different label numbering, and only the canonical form
//! distinguishes a real disagreement from a naming difference.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_hierarchy_utils.py`.
use fsci_cluster::{
    LinkageMethod, cut_tree, from_mlab_linkage, inconsistent, is_isomorphic, leaders, linkage,
    maxdists, maxinconsts,
};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

fn canonical(labels: &[usize]) -> Vec<f64> {
    let mut map = std::collections::HashMap::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|&l| {
            let id = *map.entry(l).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            });
            id as f64
        })
        .collect()
}

fn main() {
    // Twelve points in 2-D, deliberately in three loose groups with one outlier so the dendrogram
    // has structure at several heights rather than one obvious split.
    let data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.3, 0.2],
        vec![0.1, 0.4],
        vec![3.0, 3.1],
        vec![3.4, 2.8],
        vec![3.1, 3.5],
        vec![6.2, 0.4],
        vec![6.0, 0.1],
        vec![6.5, 0.6],
        vec![1.7, 5.9],
        vec![2.0, 6.2],
        vec![9.4, 9.1],
    ];

    let z = linkage(&data, LinkageMethod::Average).expect("linkage");

    // CONTROL: the linkage matrix itself.
    let flat: Vec<f64> = z.iter().flat_map(|r| r.iter().copied()).collect();
    dump("linkage_average", &flat);

    dump("maxdists", &maxdists(&z));

    let r = inconsistent(&z, 2);
    dump(
        "inconsistent_d2",
        &r.iter()
            .flat_map(|q| q.iter().copied())
            .collect::<Vec<f64>>(),
    );
    if let Ok(mi) = maxinconsts(&z, &r) {
        dump("maxinconsts_d2", &mi);
    }

    for k in [2usize, 3, 4, 6] {
        if let Ok(labels) = cut_tree(&z, Some(k), None) {
            dump(
                &format!("cuttree_k{k}_raw"),
                &labels.iter().map(|&l| l as f64).collect::<Vec<f64>>(),
            );
            dump(&format!("cuttree_k{k}_canon"), &canonical(&labels));
        }
    }

    // `leaders` needs a flat clustering; reuse the k=3 cut.
    if let Ok(labels) = cut_tree(&z, Some(3), None) {
        // SciPy's leaders expects 1-based cluster ids from fcluster; shift to match.
        let t: Vec<usize> = labels.iter().map(|&l| l + 1).collect();
        if let Ok((l, m)) = leaders(&z, &t) {
            dump(
                "leaders_L",
                &l.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            );
            dump(
                "leaders_M",
                &m.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            );
        }
    }

    // is_isomorphic: BOTH answers. A predicate probed only where it is true proves nothing.
    let a = [0usize, 0, 1, 1, 2, 2];
    let relabelled = [5usize, 5, 9, 9, 1, 1];
    let different = [0usize, 1, 1, 1, 2, 2];
    dump(
        "isomorphic",
        &[
            f64::from(is_isomorphic(&a, &relabelled)),
            f64::from(is_isomorphic(&a, &different)),
            f64::from(is_isomorphic(&a, &a)),
        ],
    );

    // from_mlab_linkage: MATLAB-style 1-based 3-column form -> SciPy 4-column form.
    let mlab: Vec<[f64; 3]> = vec![[1.0, 2.0, 0.5], [3.0, 4.0, 0.8], [5.0, 6.0, 1.4]];
    let converted = from_mlab_linkage(&mlab);
    dump(
        "from_mlab",
        &converted
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect::<Vec<f64>>(),
    );
}
