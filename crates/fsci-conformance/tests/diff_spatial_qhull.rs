#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the N-D Qhull-equivalent geometry in `fsci_spatial`:
//! `ConvexHull`, `Delaunay` (with `find_simplex`, `transform`, `lift_points`, `plane_distance`),
//! `Voronoi` and `HalfspaceIntersection` (frankenscipy-1ksfv.13).
//!
//! Qhull's facet, simplex, vertex and ridge ORDER is an implementation artifact, so nothing here
//! compares indices positionally. Every structure is compared through a canonical key:
//!   * hull facets and Delaunay simplices as sorted point-index tuples. Inputs are random points
//!     in general position (no cospherical subsets), where the triangulation is unique, so the
//!     sets must be EQUAL, not merely close;
//!   * equations facet-to-facet through that key, neighbours as the map
//!     `(facet, opposite vertex) -> neighbouring facet`;
//!   * barycentric transforms through the map `vertex -> coordinate` of a probe point;
//!   * `find_simplex` through the located simplex's key, only for queries at least
//!     `QUERY_MARGIN` away from every simplex boundary (SciPy decides which queries qualify from
//!     its own barycentric coordinates or hull distances, and reports how many);
//!   * Voronoi vertices through a nearest-neighbour bijection, then ridges and regions through
//!     that bijection: as sets, and as cyclic sequences (up to rotation and reflection) where
//!     SciPy's are cyclic (2-D regions, 3-D ridges);
//!   * halfspace dual facets as sorted tuples (merged, non-simplicial ones included).
//!
//! Area and volume are compared at a tight relative tolerance, 2-D hull vertices as a
//! counterclockwise cycle up to rotation.
//!
//! Inputs are generated at runtime from a portable LCG: 2-D to 5-D, several sizes per dimension.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_spatial::{ConvexHull, Delaunay, HalfspaceIntersection, Voronoi};
use serde::Serialize;
use serde_json::Value;

const PACKET_ID: &str = "FSCI-P2C-010";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Area, volume, dual area and dual volume, relative.
const MEASURE_REL_TOL: f64 = 1.0e-11;
/// Unit normals and offsets of hull facets, lifted Delaunay facets and dual facets, absolute.
const EQUATION_ABS_TOL: f64 = 1.0e-11;
/// `paraboloid_scale` / `paraboloid_shift`, relative.
const PARABOLOID_REL_TOL: f64 = 1.0e-14;
/// Barycentric coordinates of a probe point, relative to `max(1, |c|)`.
const BARYCENTRIC_REL_TOL: f64 = 1.0e-9;
/// Lifted query coordinates and plane distances, absolute.
const LIFT_ABS_TOL: f64 = 1.0e-12;
/// Voronoi vertex coordinates, relative to `max(1, |v|)`.
const VORONOI_VERTEX_REL_TOL: f64 = 1.0e-9;
/// Halfspace intersection vertices, absolute.
const INTERSECTION_ABS_TOL: f64 = 1.0e-10;
/// Distance a `find_simplex` query must keep from every simplex boundary to be compared. Not a
/// tolerance: it selects the queries whose answer is unique.
const QUERY_MARGIN: f64 = 1.0e-6;

fn lcg_points(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    points: Vec<Vec<f64>>,
    queries: Vec<Vec<f64>>,
    probe: Vec<f64>,
    margin: f64,
}

#[derive(Debug, Clone, Serialize)]
struct HalfspaceCase {
    case_id: String,
    halfspaces: Vec<Vec<f64>>,
    interior_point: Vec<f64>,
}

#[derive(Debug, Default, Serialize)]
struct Query {
    hull: Vec<PointCase>,
    delaunay: Vec<PointCase>,
    voronoi: Vec<PointCase>,
    halfspace: Vec<HalfspaceCase>,
}

#[derive(Debug, Serialize)]
struct DiffLog {
    test_id: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    notes: Vec<String>,
    timestamp_ms: u128,
}

fn point_case(case_id: String, points: Vec<Vec<f64>>, seed: u64) -> PointCase {
    let d = points[0].len();
    // 100 queries: a 2-D batch that large takes fsci's grid locator; 3-D and up take the walk.
    // Over the 12 Delaunay cases that is 1200, the bead's 1000 with room for boundary drops.
    let queries = lcg_points(100, d, seed ^ 0x5bd1_e995)
        .into_iter()
        .map(|q| q.into_iter().map(|x| 1.2 * x - 0.1).collect())
        .collect();
    let probe = lcg_points(1, d, seed ^ 0x2545_f491).remove(0);
    PointCase {
        case_id,
        points,
        queries,
        probe,
        margin: QUERY_MARGIN,
    }
}

fn random_cases(sizes: &[(usize, &[usize])], family: &str) -> Vec<PointCase> {
    let mut out = Vec::new();
    for &(d, ns) in sizes {
        for (k, &n) in ns.iter().enumerate() {
            let seed = 1000 * d as u64 + 17 * k as u64 + family.len() as u64;
            out.push(point_case(
                format!("{family}_{d}d_n{n}"),
                lcg_points(n, d, seed),
                seed,
            ));
        }
    }
    out
}

fn with_duplicates(case_id: &str, d: usize, n: usize, seed: u64) -> PointCase {
    let mut points = lcg_points(n, d, seed);
    for twin in [0usize, 5, 7] {
        let copy = points[twin].clone();
        points.push(copy);
    }
    point_case(case_id.to_string(), points, seed)
}

fn halfspace_cases() -> Vec<HalfspaceCase> {
    let mut out = Vec::new();
    for &(d, m) in &[
        (2usize, 8usize),
        (2, 20),
        (3, 12),
        (3, 40),
        (4, 20),
        (4, 50),
        (5, 25),
        (5, 40),
    ] {
        // Random tangent halfspaces a . x <= c around a shifted interior point: the dual points
        // a / c are not cospherical, so some halfspaces are redundant (interior dual points).
        let normals = lcg_points(m, d, 77 + 13 * (d * 100 + m) as u64);
        let shift = lcg_points(1, d, 5 + d as u64).remove(0);
        let interior: Vec<f64> = shift.iter().map(|x| 0.2 * x - 0.1).collect();
        let halfspaces = normals
            .into_iter()
            .enumerate()
            .map(|(i, raw)| {
                let a: Vec<f64> = raw.iter().map(|x| 2.0 * x - 1.0).collect();
                let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let a: Vec<f64> = a.iter().map(|x| x / norm).collect();
                let c = 0.6 + 0.4 * ((i * 7919) % 101) as f64 / 101.0;
                let mut row = a.clone();
                // a . x - c <= 0 is strictly satisfied at the interior point (|interior| < 0.2).
                row.push(-c);
                row
            })
            .collect();
        out.push(HalfspaceCase {
            case_id: format!("halfspace_{d}d_m{m}"),
            halfspaces,
            interior_point: interior,
        });
    }
    // A square pyramid: its apex dual facet is not a simplex and must come back merged.
    out.push(HalfspaceCase {
        case_id: "halfspace_3d_pyramid_merged".into(),
        halfspaces: vec![
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 0.0, 0.5, -1.0],
            vec![-1.0, 0.0, 0.5, -1.0],
            vec![0.0, 1.0, 0.5, -1.0],
            vec![0.0, -1.0, 0.5, -1.0],
        ],
        interior_point: vec![0.0, 0.0, 0.5],
    });
    // An unbounded region: one intersection row is non-finite in SciPy too.
    out.push(HalfspaceCase {
        case_id: "halfspace_3d_unbounded".into(),
        halfspaces: vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 1.0, 0.0, -1.0],
        ],
        interior_point: vec![0.2, 0.2, 0.2],
    });
    out
}

fn degenerate_cases() -> Vec<PointCase> {
    let mut out = Vec::new();
    let collinear: Vec<Vec<f64>> = (0..10)
        .map(|k| vec![k as f64 / 8.0, 2.0 * k as f64 / 8.0])
        .collect();
    out.push(point_case("collinear_2d".into(), collinear, 1));
    let coplanar: Vec<Vec<f64>> = lcg_points(20, 2, 3)
        .into_iter()
        .map(|p| vec![p[0], p[1], 0.0])
        .collect();
    out.push(point_case("coplanar_3d".into(), coplanar, 2));
    let third = 1.0 / 3.0;
    out.push(point_case(
        "coplanar_3d_rounded".into(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![third, third, third],
        ],
        3,
    ));
    let flat4: Vec<Vec<f64>> = lcg_points(30, 3, 4)
        .into_iter()
        .map(|p| vec![p[0], p[1], p[2], p[0] - p[1]])
        .collect();
    out.push(point_case("hyperplane_4d".into(), flat4, 4));
    out
}

const ORACLE: &str = r#"
import json, math, sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, HalfspaceIntersection, QhullError

def fv(v):
    v = float(v)
    if math.isfinite(v):
        return v
    return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")

def rows(a):
    return [[fv(x) for x in row] for row in np.asarray(a, dtype=float)]

def bary(t, s, x):
    T = t.transform[s]
    c = T[:-1] @ (x - T[-1])
    return np.append(c, 1.0 - c.sum())

q = json.load(sys.stdin)
out = {"hull": [], "delaunay": [], "voronoi": [], "halfspace": [], "degenerate": []}
for case in q.get("hull", []):
    r = {"case_id": case["case_id"]}
    try:
        h = ConvexHull(np.asarray(case["points"], dtype=float))
        r.update(vertices=h.vertices.tolist(), simplices=h.simplices.tolist(),
                 neighbors=h.neighbors.tolist(), equations=rows(h.equations),
                 area=fv(h.area), volume=fv(h.volume), ncoplanar=int(len(h.coplanar)))
    except QhullError as e:
        r["error"] = str(e).splitlines()[0]
    out["hull"].append(r)
for case in q.get("delaunay", []):
    r = {"case_id": case["case_id"]}
    pts = np.asarray(case["points"], dtype=float)
    try:
        t = Delaunay(pts)
        xq = np.asarray(case["queries"], dtype=float)
        found = t.find_simplex(xq)
        hull_eq = ConvexHull(pts).equations
        keep = []
        for i, s in enumerate(found.tolist()):
            if s >= 0:
                margin = bary(t, s, xq[i]).min()
            else:
                margin = (hull_eq[:, :-1] @ xq[i] + hull_eq[:, -1]).max()
            keep.append(bool(margin > case["margin"]))
        probe = np.asarray(case["probe"], dtype=float)
        r.update(simplices=t.simplices.tolist(), neighbors=t.neighbors.tolist(),
                 equations=rows(t.equations), convex_hull=t.convex_hull.tolist(),
                 coplanar=t.coplanar.tolist(), vertex_to_simplex=t.vertex_to_simplex.tolist(),
                 scale=fv(t.paraboloid_scale), shift=fv(t.paraboloid_shift),
                 found=found.tolist(), keep=keep,
                 bary=[[fv(c) for c in bary(t, s, probe)] for s in range(t.nsimplex)],
                 lift=rows(t.lift_points(xq)), plane=rows(t.plane_distance(xq)))
    except QhullError as e:
        r["error"] = str(e).splitlines()[0]
    out["delaunay"].append(r)
for case in q.get("voronoi", []):
    r = {"case_id": case["case_id"]}
    try:
        v = Voronoi(np.asarray(case["points"], dtype=float))
        r.update(vertices=rows(v.vertices), ridge_points=v.ridge_points.tolist(),
                 ridge_vertices=[list(map(int, rv)) for rv in v.ridge_vertices],
                 regions=[list(map(int, rg)) for rg in v.regions],
                 point_region=v.point_region.tolist())
    except QhullError as e:
        r["error"] = str(e).splitlines()[0]
    out["voronoi"].append(r)
for case in q.get("halfspace", []):
    r = {"case_id": case["case_id"]}
    try:
        h = HalfspaceIntersection(np.asarray(case["halfspaces"], dtype=float),
                                  np.asarray(case["interior_point"], dtype=float))
        try:
            dual_vertices = h.dual_vertices.tolist()
            from_facets = False
        except ValueError:
            # SciPy 1.17.1's property does np.unique(np.array(dual_facets)), which raises when a
            # merged dual facet makes the list ragged; take the documented value directly.
            dual_vertices = np.unique(np.concatenate([np.asarray(f) for f in h.dual_facets])).tolist()
            from_facets = True
        r.update(intersections=rows(h.intersections),
                 dual_facets=[list(map(int, f)) for f in h.dual_facets],
                 dual_equations=rows(h.dual_equations), dual_area=fv(h.dual_area),
                 dual_volume=fv(h.dual_volume), dual_vertices=dual_vertices,
                 dual_vertices_from_facets=from_facets)
    except QhullError as e:
        r["error"] = str(e).splitlines()[0]
    out["halfspace"].append(r)
print(json.dumps(out, allow_nan=False))
"#;

fn scipy_oracle_or_skip(query: &Query) -> Option<Value> {
    let query_json = serde_json::to_string(query).expect("serialize qhull query");
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
                "failed to spawn the qhull oracle: {e}"
            );
            eprintln!("skipping qhull oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open qhull oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "qhull oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping qhull oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for qhull oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "qhull oracle failed: {stderr}"
        );
        eprintln!("skipping qhull oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse qhull oracle JSON"))
}

fn emit_log(test_id: &str, case_count: usize, ledger: &CompareLedger, notes: Vec<String>) {
    for note in &notes {
        eprintln!("{test_id}: {note}");
    }
    for (arm, counts) in ledger.counts() {
        eprintln!(
            "{test_id}: arm {arm} compared={} failed={} rust_failed={} oracle_missing={}",
            counts.compared_cases,
            counts.compared_failed,
            counts.rust_failed,
            counts.oracle_missing
        );
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/diff"));
    fs::create_dir_all(&dir).expect("create qhull diff output dir");
    let log = DiffLog {
        test_id: test_id.to_string(),
        case_count,
        compared: ledger.counts().clone(),
        notes,
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
    };
    let json = serde_json::to_string_pretty(&log).expect("serialize qhull diff log");
    fs::write(dir.join(format!("{test_id}.json")), json).expect("write qhull diff log");
}

// ── Oracle value helpers ─────────────────────────────────────────────────────────────────────

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        _ => f64::NAN,
    }
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .map_or_else(Vec::new, |a| a.iter().map(num).collect())
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .map_or_else(Vec::new, |a| a.iter().map(nums).collect())
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_i64).collect())
}

fn int_rows(v: &Value) -> Vec<Vec<i64>> {
    v.as_array()
        .map_or_else(Vec::new, |a| a.iter().map(ints).collect())
}

fn usize_rows(v: &Value) -> Vec<Vec<usize>> {
    int_rows(v)
        .into_iter()
        .map(|row| row.into_iter().map(|x| x as usize).collect())
        .collect()
}

fn by_case(v: &Value, family: &str) -> HashMap<String, Value> {
    v[family]
        .as_array()
        .expect("oracle family")
        .iter()
        .map(|r| (r["case_id"].as_str().unwrap_or("").to_string(), r.clone()))
        .collect()
}

// ── Canonical forms ──────────────────────────────────────────────────────────────────────────

/// Exact duplicates are one site: which copy Qhull keeps as the vertex, and which it reports as
/// coplanar, follows its processing order (SciPy 1.17.1 keeps the LATER copy of some pairs).
/// Every index maps to the lowest index among its bitwise-identical copies, which is the identity
/// when there are no duplicates, so general-position cases are compared exactly as indices.
fn twin_classes(points: &[Vec<f64>]) -> Vec<usize> {
    let mut first: HashMap<Vec<u64>, usize> = HashMap::new();
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            *first
                .entry(p.iter().map(|x| x.to_bits()).collect())
                .or_insert(i)
        })
        .collect()
}

fn identity_classes(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn key(set: &[usize], classes: &[usize]) -> Vec<usize> {
    let mut k: Vec<usize> = set.iter().map(|&i| classes[i]).collect();
    k.sort_unstable();
    k
}

fn keys(sets: &[Vec<usize>], classes: &[usize]) -> BTreeSet<Vec<usize>> {
    sets.iter().map(|s| key(s, classes)).collect()
}

/// `(facet key, opposite vertex) -> neighbouring facet key` (`None` for `-1`).
fn neighbor_map(
    simplices: &[Vec<usize>],
    neighbors: &[Vec<i64>],
    classes: &[usize],
) -> BTreeMap<(Vec<usize>, usize), Option<Vec<usize>>> {
    let mut out = BTreeMap::new();
    for (s, simplex) in simplices.iter().enumerate() {
        for (k, &v) in simplex.iter().enumerate() {
            let nb = neighbors[s][k];
            let value = usize::try_from(nb)
                .ok()
                .map(|n| key(&simplices[n], classes));
            out.insert((key(simplex, classes), classes[v]), value);
        }
    }
    out
}

fn rel_close(got: f64, want: f64, rtol: f64) -> bool {
    (got - want).abs() <= rtol * want.abs().max(1.0)
}

fn same_value(got: f64, want: f64, atol: f64) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() {
        return got == want;
    }
    (got - want).abs() <= atol
}

fn rows_close(got: &[f64], want: &[f64], atol: f64) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(&g, &w)| same_value(g, w, atol))
}

/// Measured worst difference per arm, reported in the diff log and on stderr so the margin under
/// each tolerance is visible. A difference in class (finite against non-finite, NaN against a
/// number) is recorded as NaN, which a later finite value cannot overwrite.
#[derive(Default)]
struct Worst(BTreeMap<&'static str, f64>);

impl Worst {
    fn see(&mut self, arm: &'static str, d: f64) {
        let entry = self.0.entry(arm).or_insert(0.0);
        if !entry.is_nan() && (d.is_nan() || d > *entry) {
            *entry = d;
        }
    }

    fn rel(&mut self, arm: &'static str, got: f64, want: f64) {
        self.see(arm, diff_of(got, want) / want.abs().max(1.0));
    }

    fn abs_rows(&mut self, arm: &'static str, got: &[f64], want: &[f64]) {
        if got.len() != want.len() {
            self.see(arm, f64::NAN);
        }
        for (&g, &w) in got.iter().zip(want) {
            self.see(arm, diff_of(g, w));
        }
    }

    fn notes(&self) -> Vec<String> {
        self.0
            .iter()
            .map(|(arm, d)| format!("measured worst {arm}: {d:.3e}"))
            .collect()
    }
}

/// `|got - want|` for finite values, 0 for matching non-finite values, NaN for a class mismatch.
fn diff_of(got: f64, want: f64) -> f64 {
    if got.is_finite() && want.is_finite() {
        (got - want).abs()
    } else if (got.is_nan() && want.is_nan()) || got == want {
        0.0
    } else {
        f64::NAN
    }
}

fn is_rotation<T: PartialEq>(got: &[T], want: &[T]) -> bool {
    let n = want.len();
    got.len() == n && (n == 0 || (0..n).any(|s| (0..n).all(|i| got[(i + s) % n] == want[i])))
}

fn cyclic_equivalent<T: PartialEq + Clone>(got: &[T], want: &[T]) -> bool {
    let mut reversed = got.to_vec();
    reversed.reverse();
    is_rotation(got, want) || is_rotation(&reversed, want)
}

/// Nearest-neighbour bijection fsci vertex -> SciPy vertex; `None` when any fsci vertex has no
/// SciPy vertex within tolerance, or two claim the same one.
fn match_vertices(fsci: &[Vec<f64>], scipy: &[Vec<f64>], worst: &mut Worst) -> Option<Vec<usize>> {
    if fsci.len() != scipy.len() {
        return None;
    }
    let mut taken = vec![false; scipy.len()];
    let mut map = Vec::with_capacity(fsci.len());
    for v in fsci {
        let (best, dist) = scipy
            .iter()
            .enumerate()
            .map(|(j, w)| {
                let d = v.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
                (j, d.sqrt())
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        let scale = scipy[best]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        worst.see("voronoi vertex distance (rel)", dist / scale);
        if !(dist <= VORONOI_VERTEX_REL_TOL * scale) || taken[best] {
            return None;
        }
        taken[best] = true;
        map.push(best);
    }
    Some(map)
}

fn mapped(list: &[isize], map: &[usize]) -> Vec<i64> {
    list.iter()
        .map(|&v| usize::try_from(v).map_or(-1, |i| map[i] as i64))
        .collect()
}

fn as_set(list: &[i64]) -> BTreeSet<i64> {
    list.iter().copied().collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────────────────────

#[test]
fn diff_spatial_qhull_convex_hull() {
    let cases = random_cases(
        &[
            (2, &[10, 60, 400]),
            (3, &[12, 60, 300]),
            (4, &[20, 60, 150]),
            (5, &[25, 50, 100]),
        ],
        "hull",
    );
    let query = Query {
        hull: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let scipy = by_case(&oracle, "hull");
    let arms = ["vertices", "facets", "measures", "equations", "neighbors"];
    let mut ledger = CompareLedger::new("diff_spatial_qhull_convex_hull", &arms);
    let mut worst = Worst::default();
    for case in &cases {
        let s = &scipy[&case.case_id];
        let fsci = ConvexHull::new(&case.points).ok();
        let id = case.case_id.as_str();
        let s_ok = s.get("error").is_none().then_some(s);
        let classes = twin_classes(&case.points);

        if let Some((s, h)) = ledger.both("vertices", id, s_ok, fsci.as_ref()) {
            let want: Vec<usize> = ints(&s["vertices"]).iter().map(|&x| x as usize).collect();
            let pass = if h.ndim == 2 {
                is_rotation(&h.vertices, &want)
            } else {
                h.vertices == want
            };
            ledger.compared("vertices", id, pass);
        }
        let simplices = s_ok.map(|s| usize_rows(&s["simplices"]));
        if let Some((want, h)) = ledger.both("facets", id, simplices.clone(), fsci.as_ref()) {
            ledger.compared(
                "facets",
                id,
                keys(&want, &classes) == keys(&h.simplices, &classes),
            );
        }
        if let Some((s, h)) = ledger.both("measures", id, s_ok, fsci.as_ref()) {
            worst.rel("measures (rel)", h.area, num(&s["area"]));
            worst.rel("measures (rel)", h.volume, num(&s["volume"]));
            let pass = rel_close(h.area, num(&s["area"]), MEASURE_REL_TOL)
                && rel_close(h.volume, num(&s["volume"]), MEASURE_REL_TOL);
            ledger.compared("measures", id, pass);
        }
        if let Some((s, h)) = ledger.both("equations", id, s_ok, fsci.as_ref()) {
            let want: HashMap<Vec<usize>, Vec<f64>> = usize_rows(&s["simplices"])
                .iter()
                .zip(matrix(&s["equations"]))
                .map(|(f, eq)| (key(f, &classes), eq))
                .collect();
            let pass = h.simplices.iter().zip(&h.equations).all(|(f, eq)| {
                want.get(&key(f, &classes)).is_some_and(|w| {
                    worst.abs_rows("equations (abs)", eq, w);
                    rows_close(eq, w, EQUATION_ABS_TOL)
                })
            });
            ledger.compared("equations", id, pass);
        }
        if let Some((s, h)) = ledger.both("neighbors", id, s_ok, fsci.as_ref()) {
            let want = neighbor_map(
                &usize_rows(&s["simplices"]),
                &int_rows(&s["neighbors"]),
                &classes,
            );
            let got_nb: Vec<Vec<i64>> = h
                .neighbors
                .iter()
                .map(|row| row.iter().map(|&x| x as i64).collect())
                .collect();
            ledger.compared(
                "neighbors",
                id,
                neighbor_map(&h.simplices, &got_nb, &classes) == want,
            );
        }
    }
    let notes = worst.notes();
    emit_log(
        "diff_spatial_qhull_convex_hull",
        cases.len(),
        &ledger,
        notes,
    );
    ledger.finish(cases.len());
}

#[test]
fn diff_spatial_qhull_delaunay() {
    let mut cases = random_cases(
        &[
            (2, &[10, 60, 300]),
            (3, &[12, 50, 150]),
            (4, &[15, 40]),
            (5, &[12, 25]),
        ],
        "delaunay",
    );
    cases.push(with_duplicates("delaunay_2d_duplicates", 2, 30, 91));
    cases.push(with_duplicates("delaunay_3d_duplicates", 3, 30, 92));
    let query = Query {
        delaunay: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let scipy = by_case(&oracle, "delaunay");
    let arms = [
        "simplices",
        "volume",
        "equations",
        "paraboloid",
        "neighbors",
        "convex_hull",
        "coplanar",
        "find_simplex",
        "transform",
        "lift_plane",
    ];
    let mut ledger = CompareLedger::new("diff_spatial_qhull_delaunay", &arms);
    let mut notes = Vec::new();
    let mut worst = Worst::default();
    for case in &cases {
        let s = &scipy[&case.case_id];
        let fsci = Delaunay::new(&case.points).ok();
        let id = case.case_id.as_str();
        let s_ok = s.get("error").is_none().then_some(s);
        let d = case.points[0].len();
        let classes = twin_classes(&case.points);
        let ck = |f: &[usize]| key(f, &classes);

        let want_simplices = s_ok.map(|s| usize_rows(&s["simplices"]));
        if let Some((want, t)) = ledger.both("simplices", id, want_simplices.clone(), fsci.as_ref())
        {
            ledger.compared(
                "simplices",
                id,
                keys(&want, &classes) == keys(&t.simplices, &classes),
            );
        }
        if let Some((want, t)) = ledger.both("volume", id, want_simplices.clone(), fsci.as_ref()) {
            // Both triangulations tile the hull: their volumes agree with each other and with
            // the hull's (which also sees the duplicates) when every simplex is counted once.
            let volume = |simplices: &[Vec<usize>]| {
                simplices
                    .iter()
                    .map(|s| simplex_volume(&case.points, s))
                    .sum::<f64>()
            };
            let hull = ConvexHull::new(&case.points)
                .map(|h| h.volume)
                .unwrap_or(f64::NAN);
            let (got, scipy_total) = (volume(&t.simplices), volume(&want));
            worst.rel("volume (rel)", got, scipy_total);
            worst.rel("volume vs hull (rel)", got, hull);
            ledger.compared(
                "volume",
                id,
                rel_close(got, scipy_total, MEASURE_REL_TOL)
                    && rel_close(got, hull, MEASURE_REL_TOL),
            );
        }
        if let Some((s, t)) = ledger.both("equations", id, s_ok, fsci.as_ref()) {
            let want: HashMap<Vec<usize>, Vec<f64>> = usize_rows(&s["simplices"])
                .iter()
                .zip(matrix(&s["equations"]))
                .map(|(f, eq)| (ck(f), eq))
                .collect();
            let pass = t.simplices.iter().zip(&t.equations).all(|(f, eq)| {
                want.get(&ck(f)).is_some_and(|w| {
                    worst.abs_rows("equations (abs)", eq, w);
                    rows_close(eq, w, EQUATION_ABS_TOL)
                })
            });
            ledger.compared("equations", id, pass);
        }
        if let Some((s, t)) = ledger.both("paraboloid", id, s_ok, fsci.as_ref()) {
            worst.rel(
                "paraboloid_scale (rel)",
                t.paraboloid_scale,
                num(&s["scale"]),
            );
            worst.see(
                "paraboloid_shift (abs)",
                diff_of(t.paraboloid_shift, num(&s["shift"])),
            );
            let pass = rel_close(t.paraboloid_scale, num(&s["scale"]), PARABOLOID_REL_TOL)
                && (t.paraboloid_shift - num(&s["shift"])).abs()
                    <= PARABOLOID_REL_TOL * t.paraboloid_scale.max(1.0);
            ledger.compared("paraboloid", id, pass);
        }
        if let Some((s, t)) = ledger.both("neighbors", id, s_ok, fsci.as_ref()) {
            let want = neighbor_map(
                &usize_rows(&s["simplices"]),
                &int_rows(&s["neighbors"]),
                &classes,
            );
            let got_nb: Vec<Vec<i64>> = t
                .neighbors
                .iter()
                .map(|row| row.iter().map(|&x| x as i64).collect())
                .collect();
            ledger.compared(
                "neighbors",
                id,
                neighbor_map(&t.simplices, &got_nb, &classes) == want,
            );
        }
        if let Some((s, t)) = ledger.both("convex_hull", id, s_ok, fsci.as_ref()) {
            let want = keys(&usize_rows(&s["convex_hull"]), &classes);
            ledger.compared("convex_hull", id, keys(&t.convex_hull, &classes) == want);
        }
        if let Some((s, t)) = ledger.both("coplanar", id, s_ok, fsci.as_ref()) {
            // (point, nearest vertex) by twin class, as a multiset; the simplex column is
            // Qhull's choice among the simplices around the twin. Like SciPy, vertex_to_simplex
            // of a coplanar point holds its nearest VERTEX (SciPy's column 2).
            let scipy_rows = usize_rows(&s["coplanar"]);
            let mut want: Vec<(usize, usize)> = scipy_rows
                .iter()
                .map(|r| (classes[r[0]], classes[r[2]]))
                .collect();
            let mut got: Vec<(usize, usize)> = t
                .coplanar
                .iter()
                .map(|r| (classes[r[0]], classes[r[2]]))
                .collect();
            want.sort_unstable();
            got.sort_unstable();
            let vts = ints(&s["vertex_to_simplex"]);
            let quirk = scipy_rows.iter().all(|r| vts[r[0]] == r[2] as i64)
                && t.coplanar
                    .iter()
                    .all(|r| t.vertex_to_simplex[r[0]] == r[2] as isize);
            notes.push(format!("{id}: {} coplanar points", want.len()));
            ledger.compared("coplanar", id, got == want && quirk);
        }
        if let Some((s, t)) = ledger.both("find_simplex", id, s_ok, fsci.as_ref()) {
            let found = t
                .find_simplex(&case.queries, false, None)
                .unwrap_or_default();
            let want_found = ints(&s["found"]);
            let keep: Vec<bool> = s["keep"].as_array().map_or_else(Vec::new, |a| {
                a.iter().map(|v| v.as_bool() == Some(true)).collect()
            });
            let simplices = usize_rows(&s["simplices"]);
            let mut kept = 0usize;
            let pass = found.len() == want_found.len()
                && (0..found.len()).all(|i| {
                    if !keep[i] {
                        return true;
                    }
                    kept += 1;
                    match (usize::try_from(found[i]), usize::try_from(want_found[i])) {
                        (Ok(g), Ok(w)) => ck(&t.simplices[g]) == ck(&simplices[w]),
                        (Err(_), Err(_)) => true,
                        _ => false,
                    }
                });
            notes.push(format!(
                "{id}: {kept} of {} queries clear of boundaries",
                found.len()
            ));
            ledger.compared("find_simplex", id, pass && kept > 0);
        }
        if let Some((s, t)) = ledger.both("transform", id, s_ok, fsci.as_ref()) {
            let simplices = usize_rows(&s["simplices"]);
            let want: HashMap<Vec<usize>, BTreeMap<usize, f64>> = simplices
                .iter()
                .zip(matrix(&s["bary"]))
                .map(|(f, c)| (ck(f), f.iter().map(|&v| classes[v]).zip(c).collect()))
                .collect();
            let pass = t.simplices.iter().zip(&t.transform).all(|(f, tr)| {
                let got = barycentric(tr, &case.probe, d);
                want.get(&ck(f)).is_some_and(|w| {
                    f.iter().zip(&got).all(|(v, &c)| {
                        let wc = w[&classes[*v]];
                        worst.rel("barycentric (rel)", c, wc);
                        (c.is_nan() && wc.is_nan()) || rel_close(c, wc, BARYCENTRIC_REL_TOL)
                    })
                })
            });
            ledger.compared("transform", id, pass);
        }
        if let Some((s, t)) = ledger.both("lift_plane", id, s_ok, fsci.as_ref()) {
            let lifted = t.lift_points(&case.queries).unwrap_or_default();
            let want_lift = matrix(&s["lift"]);
            let lift_ok = lifted.len() == want_lift.len()
                && lifted.iter().zip(&want_lift).all(|(g, w)| {
                    worst.abs_rows("lift_points (abs)", g, w);
                    rows_close(g, w, LIFT_ABS_TOL)
                });
            let plane = t.plane_distance(&case.queries).unwrap_or_default();
            let want_plane = matrix(&s["plane"]);
            let order: HashMap<Vec<usize>, usize> = usize_rows(&s["simplices"])
                .iter()
                .enumerate()
                .map(|(i, f)| (ck(f), i))
                .collect();
            let plane_ok = plane.len() == want_plane.len()
                && plane.iter().zip(&want_plane).all(|(row, want_row)| {
                    t.simplices.iter().zip(row).all(|(f, &dist)| {
                        order.get(&ck(f)).is_some_and(|&j| {
                            worst.see("plane_distance (abs)", diff_of(dist, want_row[j]));
                            same_value(dist, want_row[j], LIFT_ABS_TOL)
                        })
                    })
                });
            ledger.compared("lift_plane", id, lift_ok && plane_ok);
        }
    }
    notes.extend(worst.notes());
    emit_log("diff_spatial_qhull_delaunay", cases.len(), &ledger, notes);
    ledger.finish(cases.len());
}

fn barycentric(transform: &[Vec<f64>], x: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0; d + 1];
    c[d] = 1.0;
    for i in 0..d {
        for j in 0..d {
            c[i] += transform[i][j] * (x[j] - transform[d][j]);
        }
        c[d] -= c[i];
    }
    c
}

/// Unsigned volume of a simplex by Gaussian elimination on its edge vectors.
fn simplex_volume(points: &[Vec<f64>], simplex: &[usize]) -> f64 {
    let d = points[0].len();
    let base = &points[simplex[0]];
    let mut m: Vec<Vec<f64>> = simplex[1..]
        .iter()
        .map(|&i| points[i].iter().zip(base).map(|(a, b)| a - b).collect())
        .collect();
    let mut det = 1.0_f64;
    for col in 0..d {
        let pivot = (col..d)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .unwrap_or(col);
        if m[pivot][col] == 0.0 {
            return 0.0;
        }
        m.swap(pivot, col);
        det *= m[col][col];
        for row in col + 1..d {
            let f = m[row][col] / m[col][col];
            for k in col..d {
                let delta = f * m[col][k];
                m[row][k] -= delta;
            }
        }
    }
    let factorial: f64 = (1..=d).map(|k| k as f64).product();
    det.abs() / factorial
}

#[test]
fn diff_spatial_qhull_voronoi() {
    let mut cases = random_cases(
        &[
            (2, &[10, 60, 200]),
            (3, &[12, 40, 100]),
            (4, &[12, 30]),
            (5, &[10, 20]),
        ],
        "voronoi",
    );
    cases.push(with_duplicates("voronoi_2d_duplicates", 2, 30, 93));
    let query = Query {
        voronoi: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let scipy = by_case(&oracle, "voronoi");
    let arms = ["vertices", "ridge_points", "ridge_vertices", "regions"];
    let mut ledger = CompareLedger::new("diff_spatial_qhull_voronoi", &arms);
    let mut notes = Vec::new();
    let mut worst = Worst::default();
    for case in &cases {
        let s = &scipy[&case.case_id];
        let fsci = Voronoi::new(&case.points).ok();
        let id = case.case_id.as_str();
        let s_ok = s.get("error").is_none().then_some(s);
        let d = case.points[0].len();
        let classes = twin_classes(&case.points);

        let bijection = s_ok
            .zip(fsci.as_ref())
            .and_then(|(s, v)| match_vertices(&v.vertices, &matrix(&s["vertices"]), &mut worst));
        if let Some((s, v)) = ledger.both("vertices", id, s_ok, fsci.as_ref()) {
            notes.push(format!(
                "{id}: {} vertices (SciPy {})",
                v.vertices.len(),
                matrix(&s["vertices"]).len()
            ));
            ledger.compared("vertices", id, bijection.is_some());
        }
        if let Some((s, v)) = ledger.both("ridge_points", id, s_ok, fsci.as_ref()) {
            let want: BTreeSet<Vec<usize>> = keys(&usize_rows(&s["ridge_points"]), &classes);
            let got: BTreeSet<Vec<usize>> =
                v.ridge_points.iter().map(|p| key(p, &classes)).collect();
            ledger.compared(
                "ridge_points",
                id,
                got == want && v.ridge_points.len() == usize_rows(&s["ridge_points"]).len(),
            );
        }
        if let Some(((s, v), map)) = ledger
            .both("ridge_vertices", id, s_ok, fsci.as_ref())
            .map(|pair| (pair, bijection.clone()))
        {
            let want: HashMap<Vec<usize>, Vec<i64>> = usize_rows(&s["ridge_points"])
                .iter()
                .zip(int_rows(&s["ridge_vertices"]))
                .map(|(p, r)| (key(p, &classes), r))
                .collect();
            let pass = map.is_some_and(|map| {
                v.ridge_points.iter().zip(&v.ridge_vertices).all(|(p, r)| {
                    let got = mapped(r, &map);
                    want.get(&key(p, &classes)).is_some_and(|w| {
                        if d == 3 {
                            cyclic_equivalent(&got, w)
                        } else {
                            as_set(&got) == as_set(w) && got.len() == w.len()
                        }
                    })
                })
            });
            ledger.compared("ridge_vertices", id, pass);
        }
        if let Some(((s, v), map)) = ledger
            .both("regions", id, s_ok, fsci.as_ref())
            .map(|pair| (pair, bijection.clone()))
        {
            let regions = int_rows(&s["regions"]);
            let point_region = ints(&s["point_region"]);
            let pass = map.is_some_and(|map| {
                (0..case.points.len()).all(|i| {
                    let got = mapped(&v.regions[v.point_region[i]], &map);
                    let want = &regions[point_region[i] as usize];
                    if d == 2 {
                        cyclic_equivalent(&got, want)
                    } else {
                        as_set(&got) == as_set(want) && got.len() == want.len()
                    }
                })
            }) && v.regions.len() == regions.len();
            ledger.compared("regions", id, pass);
        }
    }
    notes.extend(worst.notes());
    emit_log("diff_spatial_qhull_voronoi", cases.len(), &ledger, notes);
    ledger.finish(cases.len());
}

#[test]
fn diff_spatial_qhull_halfspace() {
    let cases = halfspace_cases();
    let query = Query {
        halfspace: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let scipy = by_case(&oracle, "halfspace");
    let arms = [
        "intersections",
        "dual_facets",
        "dual_measures",
        "dual_equations",
        "dual_vertices",
    ];
    let mut ledger = CompareLedger::new("diff_spatial_qhull_halfspace", &arms);
    let mut notes = Vec::new();
    let mut worst = Worst::default();
    for case in &cases {
        let s = &scipy[&case.case_id];
        let fsci = HalfspaceIntersection::new(&case.halfspaces, &case.interior_point).ok();
        let id = case.case_id.as_str();
        let s_ok = s.get("error").is_none().then_some(s);
        let classes = identity_classes(case.halfspaces.len());
        let ck = |f: &[usize]| key(f, &classes);

        if let Some((s, h)) = ledger.both("intersections", id, s_ok, fsci.as_ref()) {
            // One intersection per dual facet: match each through its facet key.
            let want: HashMap<Vec<usize>, Vec<f64>> = usize_rows(&s["dual_facets"])
                .iter()
                .zip(matrix(&s["intersections"]))
                .map(|(f, x)| (ck(f), x))
                .collect();
            let pass = h.intersections.len() == want.len()
                && h.dual_facets.iter().zip(&h.intersections).all(|(f, x)| {
                    want.get(&ck(f)).is_some_and(|w| {
                        worst.abs_rows("intersections (abs)", x, w);
                        rows_close(x, w, INTERSECTION_ABS_TOL)
                    })
                });
            let nonfinite = h
                .intersections
                .iter()
                .filter(|r| r.iter().any(|x| !x.is_finite()))
                .count();
            notes.push(format!(
                "{id}: {} intersections ({nonfinite} non-finite), {} dual facets",
                h.intersections.len(),
                h.dual_facets.len()
            ));
            ledger.compared("intersections", id, pass);
        }
        if let Some((s, h)) = ledger.both("dual_facets", id, s_ok, fsci.as_ref()) {
            let want = keys(&usize_rows(&s["dual_facets"]), &classes);
            ledger.compared("dual_facets", id, keys(&h.dual_facets, &classes) == want);
        }
        if let Some((s, h)) = ledger.both("dual_measures", id, s_ok, fsci.as_ref()) {
            worst.rel("dual measures (rel)", h.dual_area, num(&s["dual_area"]));
            worst.rel("dual measures (rel)", h.dual_volume, num(&s["dual_volume"]));
            let pass = rel_close(h.dual_area, num(&s["dual_area"]), MEASURE_REL_TOL)
                && rel_close(h.dual_volume, num(&s["dual_volume"]), MEASURE_REL_TOL);
            ledger.compared("dual_measures", id, pass);
        }
        if let Some((s, h)) = ledger.both("dual_equations", id, s_ok, fsci.as_ref()) {
            let want: HashMap<Vec<usize>, Vec<f64>> = usize_rows(&s["dual_facets"])
                .iter()
                .zip(matrix(&s["dual_equations"]))
                .map(|(f, eq)| (ck(f), eq))
                .collect();
            let pass = h.dual_facets.iter().zip(&h.dual_equations).all(|(f, eq)| {
                want.get(&ck(f)).is_some_and(|w| {
                    worst.abs_rows("dual equations (abs)", eq, w);
                    rows_close(eq, w, EQUATION_ABS_TOL)
                })
            });
            ledger.compared("dual_equations", id, pass);
        }
        if let Some((s, h)) = ledger.both("dual_vertices", id, s_ok, fsci.as_ref()) {
            let want: Vec<usize> = ints(&s["dual_vertices"])
                .iter()
                .map(|&x| x as usize)
                .collect();
            if s["dual_vertices_from_facets"].as_bool() == Some(true) {
                notes.push(format!(
                    "{id}: SciPy's dual_vertices property raises on merged dual facets; compared \
                     against np.unique of its dual_facets"
                ));
            }
            // Ascending in every dimension, SciPy's 2-D included.
            ledger.compared("dual_vertices", id, h.dual_vertices == want);
        }
    }
    notes.extend(worst.notes());
    emit_log("diff_spatial_qhull_halfspace", cases.len(), &ledger, notes);
    ledger.finish(cases.len());
}

/// Input in a lower-dimensional flat (exactly, or up to the rounding of 1/3) is a QhullError in
/// SciPy for both ConvexHull and Delaunay; fsci must refuse it too. The last case is 4-D data on
/// a hyperplane.
#[test]
fn diff_spatial_qhull_degenerate() {
    let cases = degenerate_cases();
    let query = Query {
        hull: cases.clone(),
        delaunay: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let hull = by_case(&oracle, "hull");
    let tri = by_case(&oracle, "delaunay");
    let mut ledger = CompareLedger::new("diff_spatial_qhull_degenerate", &["hull", "delaunay"]);
    let mut notes = Vec::new();
    for case in &cases {
        let id = case.case_id.as_str();
        for (arm, scipy, refused) in [
            ("hull", &hull[id], ConvexHull::new(&case.points).is_err()),
            ("delaunay", &tri[id], Delaunay::new(&case.points).is_err()),
        ] {
            match scipy.get("error").and_then(Value::as_str) {
                Some(message) => {
                    notes.push(format!("{id}/{arm}: SciPy raised `{message}`"));
                    ledger.expected_raise(arm, id, refused);
                }
                None => ledger.compared(arm, id, !refused),
            }
        }
    }
    emit_log("diff_spatial_qhull_degenerate", cases.len(), &ledger, notes);
    ledger.finish(cases.len());
}

/// Cospherical and lattice input: cube and hypercube corners, a cube with its centre, a 3-D
/// lattice and a 2-D grid. The Delaunay triangulation is not unique here, so simplex sets are
/// not compared.
fn cospherical_cases() -> Vec<PointCase> {
    let corners = |d: usize| -> Vec<Vec<f64>> {
        (0..1usize << d)
            .map(|mask| (0..d).map(|k| ((mask >> k) & 1) as f64).collect())
            .collect()
    };
    let lattice = |d: usize, n: usize| -> Vec<Vec<f64>> {
        (0..n.pow(d as u32))
            .map(|index| {
                let mut rest = index;
                (0..d)
                    .map(|_| {
                        let c = (rest % n) as f64 / (n - 1) as f64;
                        rest /= n;
                        c
                    })
                    .collect()
            })
            .collect()
    };
    let mut cube_and_centre = corners(3);
    cube_and_centre.push(vec![0.5, 0.5, 0.5]);
    vec![
        point_case("cube_corners_3d".into(), corners(3), 11),
        point_case("cube_corners_centre_3d".into(), cube_and_centre, 12),
        point_case("lattice_3d_3x3x3".into(), lattice(3, 3), 13),
        point_case("grid_2d_5x5".into(), lattice(2, 5), 14),
        point_case("hypercube_corners_4d".into(), corners(4), 15),
    ]
}

/// What must hold on cospherical input is SciPy's behaviour class.
/// - The hull has SciPy's vertex set, area and volume.
/// - The triangulation is valid: every fsci simplex is non-degenerate, and both sides' simplex
///   volumes sum to the hull volume.
#[test]
fn diff_spatial_qhull_cospherical() {
    let cases = cospherical_cases();
    let query = Query {
        hull: cases.clone(),
        delaunay: cases.clone(),
        ..Query::default()
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let hull = by_case(&oracle, "hull");
    let tri = by_case(&oracle, "delaunay");
    let mut ledger =
        CompareLedger::new("diff_spatial_qhull_cospherical", &["hull", "triangulation"]);
    let mut notes = Vec::new();
    let mut worst = Worst::default();
    for case in &cases {
        let id = case.case_id.as_str();
        let s_hull = hull[id].get("error").is_none().then_some(&hull[id]);
        let fsci_hull = ConvexHull::new(&case.points).ok();
        if let Some((s, h)) = ledger.both("hull", id, s_hull, fsci_hull.as_ref()) {
            let mut want: Vec<usize> = ints(&s["vertices"]).iter().map(|&v| v as usize).collect();
            want.sort_unstable();
            let mut got = h.vertices.clone();
            got.sort_unstable();
            worst.rel("hull measures (rel)", h.volume, num(&s["volume"]));
            worst.rel("hull measures (rel)", h.area, num(&s["area"]));
            let pass = got == want
                && rel_close(h.volume, num(&s["volume"]), MEASURE_REL_TOL)
                && rel_close(h.area, num(&s["area"]), MEASURE_REL_TOL);
            ledger.compared("hull", id, pass);
        }
        let s_tri = tri[id].get("error").is_none().then_some(&tri[id]);
        let fsci_tri = Delaunay::new(&case.points).ok();
        if let Some((s, t)) = ledger.both("triangulation", id, s_tri, fsci_tri.as_ref()) {
            let volume = num(&hull[id]["volume"]);
            let theirs = usize_rows(&s["simplices"]);
            let ours: Vec<f64> = t
                .simplices
                .iter()
                .map(|simplex| simplex_volume(&case.points, simplex))
                .collect();
            let total: f64 = ours.iter().sum();
            let scipy_total: f64 = theirs
                .iter()
                .map(|simplex| simplex_volume(&case.points, simplex))
                .sum();
            let thinnest = ours.iter().copied().fold(f64::INFINITY, f64::min);
            worst.rel("triangulation volume sum (rel)", total, volume);
            notes.push(format!(
                "{id}: fsci {} simplices, SciPy {}; volume sums {total} and {scipy_total} of hull \
                 {volume}; thinnest fsci simplex {thinnest:e}",
                t.simplices.len(),
                theirs.len()
            ));
            let pass = rel_close(total, volume, MEASURE_REL_TOL)
                && rel_close(scipy_total, volume, MEASURE_REL_TOL)
                && thinnest > 1e-12 * volume;
            ledger.compared("triangulation", id, pass);
        }
    }
    notes.extend(worst.notes());
    emit_log(
        "diff_spatial_qhull_cospherical",
        cases.len(),
        &ledger,
        notes,
    );
    ledger.finish(cases.len());
}
