//! Differential oracle probe for `scipy.spatial` entry points with NO differential coverage
//! (frankenscipy-ivxx6): `minkowski_distance`, `minkowski_distance_p`, `tsearch`,
//! `SphericalVoronoi`.
//!
//! Each was confirmed at zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus.
//!
//! TWO OF THESE HAVE IMPLEMENTATION-DEPENDENT OUTPUT, and comparing it directly would produce a
//! false divergence:
//!   * `tsearch` returns a SIMPLEX INDEX. Two libraries may build different (both valid) Delaunay
//!     triangulations of the same points, so the indices need not match even when both are right.
//!     What IS canonical is whether a query lies inside the hull at all, so the probe emits the
//!     inside/outside classification, and separately checks that each returned simplex really
//!     contains its query point.
//!   * `SphericalVoronoi` vertex and region ORDER is not canonical either. The probe emits
//!     order-independent invariants: the vertex count, the lexicographically sorted vertices, and
//!     the sorted region sizes.
//! `minkowski_distance*` has no such freedom and is compared elementwise.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_minkowski_tsearch_svoronoi.py`.
use fsci_spatial::{Delaunay, SphericalVoronoi, minkowski_distance, minkowski_distance_p, tsearch};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

fn main() {
    // ---- minkowski_distance / _p -------------------------------------------------------------
    let xa: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 2.0, -1.0],
        vec![-3.0, 0.5, 2.0],
        vec![4.0, -4.0, 1.5],
    ];
    let xb: Vec<Vec<f64>> = vec![
        vec![1.0, -1.0, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![2.5, 1.5, -2.0],
        vec![-1.0, 3.0, 0.0],
    ];
    // p < 1 is included on purpose: it is not a metric there, and an implementation that assumes
    // the triangle inequality can quietly special-case it.
    for p in [0.5_f64, 1.0, 1.5, 2.0, 3.0] {
        // Fixed one-decimal formatting so the group tags match the comparator: Rust prints
        // 1.0 as "1" where Python prints "1.0", which silently splits every group in two.
        let tag = format!("{p:.1}").replace('.', "_");
        if let Ok(d) = minkowski_distance(&xa, &xb, p) {
            dump(&format!("mink_p{tag}"), &d);
        }
        if let Ok(d) = minkowski_distance_p(&xa, &xb, p) {
            dump(&format!("minkp_p{tag}"), &d);
        }
    }
    if let Ok(d) = minkowski_distance(&xa, &xb, f64::INFINITY) {
        dump("mink_pinf", &d);
    }

    // ---- tsearch -----------------------------------------------------------------------------
    let pts: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.35, 0.45),
        (0.7, 0.25),
        (0.25, 0.75),
    ];
    let tri = Delaunay::new(&pts).expect("delaunay");
    let queries: Vec<(f64, f64)> = vec![
        (0.5, 0.5),
        (0.2, 0.2),
        (0.85, 0.6),
        (0.1, 0.9),
        (0.5, 0.05),
        (-0.5, 0.5),
        (1.5, 0.5),
        (0.5, -0.4),
    ];
    let found = tsearch(&tri, &queries);
    // Canonical: inside the hull or not.
    let inside: Vec<f64> = found.iter().map(|&s| f64::from(s >= 0)).collect();
    dump("tsearch_inside", &inside);
    // And the returned simplex must actually contain the query. This is checked here rather than
    // against SciPy because it is a property, not a convention.
    let mut containment_ok = 1.0_f64;
    for (q, &s) in queries.iter().zip(found.iter()) {
        if s < 0 {
            continue;
        }
        let (i0, i1, i2) = tri.simplices[s as usize];
        let (ax, ay) = pts[i0];
        let (bx, by) = pts[i1];
        let (cx, cy) = pts[i2];
        let det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
        let l1 = ((bx - q.0) * (cy - q.1) - (cx - q.0) * (by - q.1)) / det;
        let l2 = ((cx - q.0) * (ay - q.1) - (ax - q.0) * (cy - q.1)) / det;
        let l3 = 1.0 - l1 - l2;
        if l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9 {
            containment_ok = 0.0;
        }
    }
    dump("tsearch_containment", &[containment_ok]);

    // ---- SphericalVoronoi --------------------------------------------------------------------
    // Eight points on the unit sphere, not a regular polyhedron, so the diagram has varied
    // region sizes rather than one symmetric answer.
    let sphere_pts: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [
            0.577_350_269_189_625_8,
            0.577_350_269_189_625_8,
            0.577_350_269_189_625_8,
        ],
        [
            -0.577_350_269_189_625_8,
            -0.577_350_269_189_625_8,
            0.577_350_269_189_625_8,
        ],
    ];
    if let Ok(sv) = SphericalVoronoi::new(&sphere_pts, [0.0, 0.0, 0.0], 1.0) {
        dump("svor_nvertices", &[sv.vertices.len() as f64]);
        let mut sorted = sv.vertices.clone();
        sorted.sort_by(|a, b| {
            a[0].total_cmp(&b[0])
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });
        dump(
            "svor_vertices_sorted",
            &sorted
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect::<Vec<f64>>(),
        );
        let mut sizes: Vec<f64> = sv.regions.iter().map(|r| r.len() as f64).collect();
        sizes.sort_by(f64::total_cmp);
        dump("svor_region_sizes_sorted", &sizes);
        // Property, not a convention: every Voronoi vertex lies on the sphere.
        let worst = sv
            .vertices
            .iter()
            .map(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - 1.0).abs())
            .fold(0.0_f64, f64::max);
        dump("svor_on_sphere_maxdev", &[worst]);
    }
}
