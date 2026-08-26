//! Is the NaN-for-interior-points defect a TOLERANCE problem or an INCOMPLETE TRIANGULATION?
//!
//! `LinearNDInterpolator::eval` returns NaN when `Delaunay::find_simplex` finds no containing
//! triangle, and that search uses a `-1e-10` barycentric tolerance. The failing queries sit
//! roughly 1e-4 to 2e-3 INSIDE the convex hull, which is far too deep for a 1e-10 tolerance to
//! explain -- so the suspicion is that the triangulation does not cover the whole hull.
//!
//! Euler's formula settles it: a Delaunay triangulation of N points in general position with `h`
//! of them on the convex hull has EXACTLY `2N - 2 - h` triangles. This prints that expectation
//! against the actual count, so the answer is arithmetic rather than opinion.
use fsci_spatial::{ConvexHull, Delaunay};
use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/tmp/fsci_griddata_rate_800_2000.bin".to_string());
    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .expect("fixture")
        .read_to_end(&mut buf)
        .expect("read");
    let rd = |o: usize| f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    let np = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let nq = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let mut pts = Vec::with_capacity(np);
    for _ in 0..np {
        pts.push((rd(off), rd(off + 8)));
        off += 16;
    }
    off += np * 8; // values
    let mut xi = Vec::with_capacity(nq);
    for _ in 0..nq {
        xi.push((rd(off), rd(off + 8)));
        off += 16;
    }

    let tri = Delaunay::new(&pts).expect("delaunay");
    let hull = ConvexHull::new(&pts).expect("hull");
    let h = hull.vertices.len();
    let expected = 2 * np - 2 - h;
    println!(
        "npoints={np} hull_vertices={h} triangles_expected(2N-2-h)={expected} triangles_actual={} deficit={}",
        tri.simplices.len(),
        expected as i64 - tri.simplices.len() as i64
    );

    // Total area of the triangulation against the hull area: a covering triangulation matches.
    let area = |(ax, ay): (f64, f64), (bx, by): (f64, f64), (cx, cy): (f64, f64)| {
        ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs() * 0.5
    };
    let tri_area: f64 = tri
        .simplices
        .iter()
        .map(|&(i, j, k)| area(pts[i], pts[j], pts[k]))
        .sum();
    let hv: Vec<(f64, f64)> = hull.vertices.iter().map(|&i| pts[i]).collect();
    let mut hull_area = 0.0;
    for i in 1..hv.len() - 1 {
        hull_area += area(hv[0], hv[i], hv[i + 1]);
    }
    println!(
        "hull_area={hull_area:.12} triangulated_area={tri_area:.12} uncovered={:.3e} ({:.4}%)",
        hull_area - tri_area,
        100.0 * (hull_area - tri_area) / hull_area
    );

    let missed = xi
        .iter()
        .filter(|&&q| tri.find_simplex(q).is_none())
        .count();
    println!("queries={nq} queries_with_no_containing_triangle={missed}");
}
