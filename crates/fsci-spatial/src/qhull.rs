//! N-dimensional Quickhull with exact orientation predicates.
//!
//! This is the geometric kernel behind [`crate::ConvexHull`], [`crate::Delaunay`],
//! [`crate::Voronoi`] and [`crate::HalfspaceIntersection`]. It builds the simplicial convex hull
//! of a point set in dimensions `2 <= D <= 8` with the Quickhull algorithm of Barber, Dobkin &
//! Huhdanpaa (ACM TOMS 22(4), 1996): start from a full-dimensional simplex, repeatedly take the
//! furthest outside point of a facet, delete the facets that point can see, and cone the horizon
//! to the point.
//!
//! ROBUSTNESS. Every combinatorial decision (is a point strictly beyond a facet?) is made by the
//! sign of an orientation determinant, and that sign is EXACT: a floating-point evaluation with a
//! rigorous forward error bound answers when the bound certifies the sign, and otherwise the
//! determinant is re-evaluated exactly with Shewchuk-style floating-point expansions
//! (`two_sum` / `two_product` error-free transformations). With exact signs the visible region
//! is always a connected set of facets whose horizon closes, so the output is always a valid
//! simplicial complex; floating-point Quickhull without this guarantee is what forces Qhull to
//! merge facets. A point exactly on a facet's hyperplane is not "beyond" it, so points in the
//! boundary of the hull (coplanar) and inside it are never hull vertices.
//!
//! Inputs are rescaled by a power of two before any predicate runs (exact, sign-preserving), so
//! the filter and the expansion arithmetic see coordinates of magnitude about 1.
//!
//! What this kernel does NOT reproduce is Qhull's facet order: Qhull's is an artifact of its
//! point order, merge heuristics and facet lists. Callers compare facets as sets.

use std::collections::HashMap;

const EPS: f64 = f64::EPSILON;
/// gh#2: determinant tables grow as `2^D`; bound the kernel before any predicate allocation.
pub(crate) const MAX_DIM: usize = 8;

pub(crate) fn validate_dimension(dim: usize) -> Result<(), KernelError> {
    if dim > MAX_DIM {
        return Err(KernelError::DimensionTooLarge {
            have: dim,
            max: MAX_DIM,
        });
    }
    Ok(())
}

/// Below this the filter's error bound could itself be lost to underflow, so the exact path
/// decides instead.
const TINY: f64 = 1.0e-250;

// ══════════════════════════════════════════════════════════════════════
// Exact arithmetic (Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust
// Geometric Predicates", 1997). An expansion is a sum of non-overlapping doubles stored in
// increasing magnitude; the empty expansion is zero.
// ══════════════════════════════════════════════════════════════════════

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    let a_virtual = x - b_virtual;
    let b_round = b - b_virtual;
    let a_round = a - a_virtual;
    (x, a_round + b_round)
}

/// Requires `|a| >= |b|` (or `a == 0`).
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    (x, b - b_virtual)
}

/// `a * b = x + y` exactly; the fused multiply-add computes the rounding error of the product
/// without any intermediate rounding.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    (x, a.mul_add(b, -x))
}

/// `h = e + b`, zero components eliminated.
fn grow_expansion(e: &[f64], b: f64, h: &mut Vec<f64>) {
    h.clear();
    let mut q = b;
    for &component in e {
        let (sum, err) = two_sum(q, component);
        q = sum;
        if err != 0.0 {
            h.push(err);
        }
    }
    if q != 0.0 {
        h.push(q);
    }
}

/// `h = e * b`, zero components eliminated.
fn scale_expansion(e: &[f64], b: f64, h: &mut Vec<f64>) {
    h.clear();
    if e.is_empty() || b == 0.0 {
        return;
    }
    let (mut q, err) = two_product(e[0], b);
    if err != 0.0 {
        h.push(err);
    }
    for &component in &e[1..] {
        let (product_hi, product_lo) = two_product(component, b);
        let (sum, err) = two_sum(q, product_lo);
        if err != 0.0 {
            h.push(err);
        }
        let (next, err) = fast_two_sum(product_hi, sum);
        q = next;
        if err != 0.0 {
            h.push(err);
        }
    }
    if q != 0.0 {
        h.push(q);
    }
}

/// `acc += f`, one component at a time.
fn add_expansion(acc: &mut Vec<f64>, f: &[f64], scratch: &mut Vec<f64>) {
    for &component in f {
        grow_expansion(acc, component, scratch);
        std::mem::swap(acc, scratch);
    }
}

/// Shewchuk's `compress`: the same value in as few components as possible.
fn compress(e: &[f64]) -> Vec<f64> {
    let m = e.len();
    if m == 0 {
        return Vec::new();
    }
    let mut h = vec![0.0; m];
    let mut bottom = m - 1;
    let mut q = e[bottom];
    for index in (0..m - 1).rev() {
        let (sum, err) = fast_two_sum(q, e[index]);
        if err != 0.0 {
            h[bottom] = sum;
            bottom -= 1;
            q = err;
        } else {
            q = sum;
        }
    }
    let mut top = 0;
    for index in bottom + 1..m {
        let (sum, err) = fast_two_sum(h[index], q);
        if err != 0.0 {
            h[top] = err;
            top += 1;
        }
        q = sum;
    }
    h[top] = q;
    h.truncate(top + 1);
    h.retain(|&component| component != 0.0);
    h
}

fn expansion_sign(e: &[f64]) -> i8 {
    match e.last() {
        Some(&top) if top > 0.0 => 1,
        Some(&top) if top < 0.0 => -1,
        _ => 0,
    }
}

/// Exact sign of `det [[p_0, 1], [p_1, 1], ..., [p_D, 1]]` for `D + 1` points of dimension `D`.
///
/// Laplace expansion by columns from the right: the minors over the last `k` columns and every
/// `k`-subset of rows are built from the `k - 1` level, so the whole determinant costs
/// `O(2^(D+1) * D)` expansion operations. Only reached when the floating-point filter cannot
/// certify the sign, i.e. for (near-)degenerate configurations.
fn exact_homogeneous_sign(rows: &[&[f64]]) -> i8 {
    let m = rows.len();
    let dim = m - 1;
    let full = 1usize << m;
    let mut minors: Vec<Vec<f64>> = vec![Vec::new(); full];
    for i in 0..m {
        minors[1 << i] = vec![1.0];
    }
    let mut term = Vec::new();
    let mut scratch = Vec::new();
    for level in 2..=m {
        let col = dim + 1 - level;
        for mask in 1..full {
            if mask.count_ones() as usize != level {
                continue;
            }
            let mut acc: Vec<f64> = Vec::new();
            let mut bits = mask;
            let mut position = 0usize;
            while bits != 0 {
                let row = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let entry = rows[row][col];
                let sub = mask & !(1 << row);
                if entry != 0.0 && !minors[sub].is_empty() {
                    let signed = if position.is_multiple_of(2) {
                        entry
                    } else {
                        -entry
                    };
                    scale_expansion(&minors[sub], signed, &mut term);
                    add_expansion(&mut acc, &term, &mut scratch);
                }
                position += 1;
            }
            minors[mask] = compress(&acc);
        }
    }
    expansion_sign(&minors[full - 1])
}

/// Laplace expansion of the `r x c` row-major matrix `a` (`r <= c`) along its rows, bottom-up.
/// On return `det[mask]` and `perm[mask]` hold the determinant of `a` restricted to all `r` rows
/// and the columns in `mask`, and the permanent of `|a|` on the same entries, for every `mask`
/// with exactly `r` bits set. The permanent bounds the floating-point error of the determinant.
fn row_minors(a: &[f64], r: usize, c: usize, det: &mut Vec<f64>, perm: &mut Vec<f64>) {
    let full = 1usize << c;
    det.clear();
    det.resize(full, 0.0);
    perm.clear();
    perm.resize(full, 0.0);
    det[0] = 1.0;
    perm[0] = 1.0;
    for level in 1..=r {
        let row = &a[(r - level) * c..(r - level + 1) * c];
        for mask in 1..full {
            if mask.count_ones() as usize != level {
                continue;
            }
            let mut det_sum = 0.0;
            let mut perm_sum = 0.0;
            let mut bits = mask;
            let mut position = 0usize;
            while bits != 0 {
                let col = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let sub = mask & !(1 << col);
                let entry = row[col];
                let term = entry * det[sub];
                if position.is_multiple_of(2) {
                    det_sum += term;
                } else {
                    det_sum -= term;
                }
                perm_sum += entry.abs() * perm[sub];
                position += 1;
            }
            det[mask] = det_sum;
            perm[mask] = perm_sum;
        }
    }
}

/// Error bound multiplier for a filtered `D x D` determinant of differences: every monomial of
/// the Laplace expansion carries at most `D` rounded differences, `D - 1` products and
/// `D(D - 1)/2` rounded partial sums, plus one product and `D` sums for the final dot product.
/// `(D^2 + 4D + 8) * eps` (i.e. twice that count in units of the unit roundoff) bounds the
/// accumulated relative error of every monomial, hence of the whole sum against the permanent.
fn error_coefficient(dim: usize) -> f64 {
    (dim * dim + 4 * dim + 8) as f64 * EPS
}

fn pow2_scale(max_abs: f64) -> f64 {
    if !(max_abs > 0.0 && max_abs.is_finite()) {
        return 1.0;
    }
    let exponent = (max_abs.log2().floor() as i32 + 1).clamp(-1000, 1000);
    2f64.powi(-exponent)
}

// ══════════════════════════════════════════════════════════════════════
// Point sets and predicates
// ══════════════════════════════════════════════════════════════════════

/// A point set rescaled by a power of two (so every predicate sign is unchanged) together with
/// the exact orientation predicate over it.
#[derive(Debug, Clone)]
pub(crate) struct Points {
    pub(crate) dim: usize,
    coords: Vec<f64>,
}

impl Points {
    /// `flat` holds points of dimension `dim`, row-major.
    pub(crate) fn new(flat: &[f64], dim: usize) -> Self {
        let max_abs = flat.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let scale = pow2_scale(max_abs);
        Self {
            dim,
            coords: flat.iter().map(|&x| x * scale).collect(),
        }
    }

    #[inline]
    pub(crate) fn row(&self, index: usize) -> &[f64] {
        &self.coords[index * self.dim..(index + 1) * self.dim]
    }

    /// Exact sign of `det [[p_0, 1], ..., [p_D, 1]]` for the `D + 1` points `idx`. In 2-D it is
    /// positive for a counterclockwise triangle.
    pub(crate) fn orient(&self, idx: &[usize]) -> i8 {
        let d = self.dim;
        debug_assert_eq!(idx.len(), d + 1);
        let p0 = self.row(idx[0]);
        let mut e = vec![0.0; d * d];
        for i in 0..d {
            let pi = self.row(idx[i + 1]);
            for j in 0..d {
                e[i * d + j] = pi[j] - p0[j];
            }
        }
        let mut det = Vec::new();
        let mut perm = Vec::new();
        row_minors(&e, d, d, &mut det, &mut perm);
        let full = (1usize << d) - 1;
        let (value, bound) = (det[full], error_coefficient(d) * perm[full]);
        // det H = (-1)^D det(E) for the difference matrix E.
        let sign_e = if perm[full] > TINY && value > bound {
            1
        } else if perm[full] > TINY && value < -bound {
            -1
        } else {
            let rows: Vec<&[f64]> = idx.iter().map(|&i| self.row(i)).collect();
            return exact_homogeneous_sign(&rows);
        };
        if d.is_multiple_of(2) { sign_e } else { -sign_e }
    }

    /// Floating-point hyperplane of the facet through the `D` points `verts`, with the data the
    /// side filter needs: `normal . (p - v0)` has the sign of `orient(verts ++ [p])`.
    fn plane(&self, verts: &[usize]) -> Plane {
        let d = self.dim;
        let v0 = self.row(verts[0]);
        let mut e = vec![0.0; (d - 1) * d];
        for i in 1..d {
            let vi = self.row(verts[i]);
            for j in 0..d {
                e[(i - 1) * d + j] = vi[j] - v0[j];
            }
        }
        let mut det = Vec::new();
        let mut perm = Vec::new();
        row_minors(&e, d - 1, d, &mut det, &mut perm);
        let full = (1usize << d) - 1;
        let mut normal = vec![0.0; d];
        let mut bound = vec![0.0; d];
        for j in 0..d {
            let mask = full ^ (1 << j);
            // Cofactor expansion of det(E) along its last row, times (-1)^D for det H.
            normal[j] = if j.is_multiple_of(2) {
                -det[mask]
            } else {
                det[mask]
            };
            bound[j] = perm[mask];
        }
        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        Plane {
            normal,
            perm: bound,
            norm,
        }
    }

    /// Side of point `q` relative to the facet `verts` with plane `plane`: `1` strictly beyond
    /// (outside), `-1` strictly beneath, `0` on the hyperplane. Also returns the floating-point
    /// signed value (unnormalized), used only to rank candidate points.
    fn side(&self, verts: &[usize], plane: &Plane, q: usize) -> (i8, f64) {
        let v0 = self.row(verts[0]);
        let p = self.row(q);
        let mut value = 0.0;
        let mut bound = 0.0;
        for j in 0..self.dim {
            let x = p[j] - v0[j];
            value += plane.normal[j] * x;
            bound += plane.perm[j] * x.abs();
        }
        let err = error_coefficient(self.dim) * bound;
        if bound > TINY {
            if value > err {
                return (1, value);
            }
            if value < -err {
                return (-1, value);
            }
        }
        let mut rows: Vec<&[f64]> = verts.iter().map(|&v| self.row(v)).collect();
        rows.push(p);
        (exact_homogeneous_sign(&rows), value)
    }

    /// Qhull's `qh_distround`: the roundoff of a distance computation for these coordinates.
    pub(crate) fn distround(&self, count: usize) -> f64 {
        let d = self.dim as f64;
        let mut max_abs = 0.0_f64;
        let mut max_sum_abs = 0.0_f64;
        for i in 0..count {
            let row = self.row(i);
            let sum_abs: f64 = row.iter().map(|x| x.abs()).sum();
            max_sum_abs = max_sum_abs.max(sum_abs);
            max_abs = row.iter().fold(max_abs, |m, &x| m.max(x.abs()));
        }
        let max_dist_sum = (d.sqrt() * max_abs).min(max_sum_abs);
        EPS * (d * max_dist_sum * 1.01 + max_abs)
    }
}

#[derive(Debug, Clone)]
struct Plane {
    normal: Vec<f64>,
    perm: Vec<f64>,
    norm: f64,
}

// ══════════════════════════════════════════════════════════════════════
// Quickhull
// ══════════════════════════════════════════════════════════════════════

/// Why the kernel refused; the public types map these onto SciPy's `QhullError` texts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelError {
    /// Exceeds the exact-predicate allocation budget.
    DimensionTooLarge { have: usize, max: usize },
    /// Fewer points than `D + 1`.
    TooFewPoints { have: usize, need: usize },
    /// Every point lies, to Qhull's roundoff, in a lower-dimensional flat.
    Flat,
    /// The horizon failed to close. Unreachable with exact predicates; reported, never panicked.
    Topology,
}

/// A simplicial hull facet. `vertices` is ordered so that a point strictly outside the facet
/// has positive orientation (`Points::orient(vertices ++ [p]) > 0`); `neighbors[k]` is the facet
/// sharing every vertex except `vertices[k]`.
#[derive(Debug, Clone)]
pub(crate) struct HullFacet {
    pub(crate) vertices: Vec<usize>,
    pub(crate) neighbors: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Hull {
    pub(crate) facets: Vec<HullFacet>,
}

struct WorkFacet {
    verts: Vec<usize>,
    nbrs: Vec<usize>,
    plane: Plane,
    outside: Vec<usize>,
    furthest: usize,
    furthest_value: f64,
    alive: bool,
    stamp: usize,
}

impl WorkFacet {
    fn new(pts: &Points, verts: Vec<usize>, nbrs: Vec<usize>) -> Self {
        let plane = pts.plane(&verts);
        Self {
            verts,
            nbrs,
            plane,
            outside: Vec::new(),
            furthest: usize::MAX,
            furthest_value: f64::NEG_INFINITY,
            alive: true,
            stamp: 0,
        }
    }

    fn assign(&mut self, point: usize, value: f64) {
        self.outside.push(point);
        if value > self.furthest_value || self.furthest == usize::MAX {
            self.furthest_value = value;
            self.furthest = point;
        }
    }
}

/// `D + 1` affinely independent points chosen greedily (each maximizing its distance from the
/// affine hull of the previous ones), rejected as flat the way Qhull rejects its initial simplex:
/// when the centroid lies within `qh_distround` of one of the simplex's facets.
pub(crate) fn initial_simplex(pts: &Points, count: usize) -> Result<Vec<usize>, KernelError> {
    let d = pts.dim;
    validate_dimension(d)?;
    if count < d + 1 {
        return Err(KernelError::TooFewPoints {
            have: count,
            need: d + 1,
        });
    }
    let mut first = 0usize;
    for i in 1..count {
        if pts.row(i)[0] < pts.row(first)[0] {
            first = i;
        }
    }
    let origin = pts.row(first).to_vec();
    let mut chosen = vec![first];
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut residual = vec![0.0; d];
    for _ in 0..d {
        let mut best = usize::MAX;
        let mut best_norm = 0.0_f64;
        let mut best_residual = vec![0.0; d];
        for i in 0..count {
            if chosen.contains(&i) {
                continue;
            }
            let p = pts.row(i);
            for j in 0..d {
                residual[j] = p[j] - origin[j];
            }
            // Two Gram-Schmidt passes keep the residual orthogonal in floating point.
            for _ in 0..2 {
                for q in &basis {
                    let dot: f64 = q.iter().zip(&residual).map(|(a, b)| a * b).sum();
                    for j in 0..d {
                        residual[j] -= dot * q[j];
                    }
                }
            }
            let norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > best_norm {
                best_norm = norm;
                best = i;
                best_residual.copy_from_slice(&residual);
            }
        }
        if best == usize::MAX || !(best_norm > 0.0) || !best_norm.is_finite() {
            return Err(KernelError::Flat);
        }
        basis.push(best_residual.iter().map(|x| x / best_norm).collect());
        chosen.push(best);
    }

    if pts.orient(&chosen) == 0 {
        return Err(KernelError::Flat);
    }
    let centroid: Vec<f64> = (0..d)
        .map(|j| chosen.iter().map(|&i| pts.row(i)[j]).sum::<f64>() / (d + 1) as f64)
        .collect();
    let roundoff = pts.distround(count);
    for omit in 0..=d {
        let verts: Vec<usize> = chosen
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != omit)
            .map(|(_, &i)| i)
            .collect();
        let plane = pts.plane(&verts);
        let v0 = pts.row(verts[0]);
        let dist = plane
            .normal
            .iter()
            .zip(centroid.iter().zip(v0))
            .map(|(n, (c, v))| n * (c - v))
            .sum::<f64>()
            .abs()
            / plane.norm;
        if !(dist > roundoff) {
            return Err(KernelError::Flat);
        }
    }
    Ok(chosen)
}

/// Simplicial convex hull of the first `count` points of `pts`.
pub(crate) fn quickhull(pts: &Points, count: usize) -> Result<Hull, KernelError> {
    let d = pts.dim;
    let simplex = initial_simplex(pts, count)?;

    let mut facets: Vec<WorkFacet> = Vec::new();
    for omit in 0..=d {
        let mut verts: Vec<usize> = simplex
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != omit)
            .map(|(_, &i)| i)
            .collect();
        let mut probe = verts.clone();
        probe.push(simplex[omit]);
        if pts.orient(&probe) > 0 {
            verts.swap(0, 1);
        }
        let nbrs = verts
            .iter()
            .map(|v| simplex.iter().position(|s| s == v).unwrap_or(usize::MAX))
            .collect();
        facets.push(WorkFacet::new(pts, verts, nbrs));
    }

    let mut in_simplex = vec![false; count];
    for &s in &simplex {
        in_simplex[s] = true;
    }
    for q in 0..count {
        if in_simplex[q] {
            continue;
        }
        for facet in facets.iter_mut() {
            let (sign, value) = pts.side(&facet.verts, &facet.plane, q);
            if sign > 0 {
                facet.assign(q, value);
                break;
            }
        }
    }

    let mut pending: Vec<usize> = (0..facets.len())
        .filter(|&f| !facets[f].outside.is_empty())
        .collect();
    let mut round = 0usize;
    let mut visible: Vec<usize> = Vec::new();
    let mut horizon: Vec<(usize, usize, usize)> = Vec::new();
    let mut new_ids: Vec<usize> = Vec::new();
    let mut ridges: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
    while let Some(start) = pending.pop() {
        if !facets[start].alive || facets[start].outside.is_empty() {
            continue;
        }
        round += 1;
        let seen_visible = 2 * round;
        let seen_hidden = 2 * round + 1;
        let apex = facets[start].furthest;

        visible.clear();
        visible.push(start);
        facets[start].stamp = seen_visible;
        let mut cursor = 0;
        while cursor < visible.len() {
            let v = visible[cursor];
            cursor += 1;
            for k in 0..d {
                let nb = facets[v].nbrs[k];
                let stamp = facets[nb].stamp;
                if stamp == seen_visible || stamp == seen_hidden {
                    continue;
                }
                let (sign, _) = pts.side(&facets[nb].verts, &facets[nb].plane, apex);
                if sign > 0 {
                    facets[nb].stamp = seen_visible;
                    visible.push(nb);
                } else {
                    facets[nb].stamp = seen_hidden;
                }
            }
        }

        horizon.clear();
        for &v in &visible {
            for k in 0..d {
                let nb = facets[v].nbrs[k];
                if facets[nb].stamp != seen_visible {
                    horizon.push((v, k, nb));
                }
            }
        }

        // Cone the horizon to the apex. Replacing the vertex opposite a horizon ridge by the apex
        // keeps the orientation: the removed vertex stays strictly beneath the new facet.
        new_ids.clear();
        for &(v, k, nb) in &horizon {
            let mut verts = facets[v].verts.clone();
            verts[k] = apex;
            let mut nbrs = vec![usize::MAX; d];
            nbrs[k] = nb;
            let id = facets.len();
            match facets[nb].nbrs.iter().position(|&x| x == v) {
                Some(slot) => facets[nb].nbrs[slot] = id,
                None => return Err(KernelError::Topology),
            }
            facets.push(WorkFacet::new(pts, verts, nbrs));
            new_ids.push(id);
        }
        ridges.clear();
        for &g in &new_ids {
            for j in 0..d {
                if facets[g].nbrs[j] != usize::MAX {
                    continue;
                }
                let mut key: Vec<usize> = facets[g]
                    .verts
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != j)
                    .map(|(_, &x)| x)
                    .collect();
                key.sort_unstable();
                if let Some((other, other_slot)) = ridges.remove(&key) {
                    facets[g].nbrs[j] = other;
                    facets[other].nbrs[other_slot] = g;
                } else {
                    ridges.insert(key, (g, j));
                }
            }
        }
        if !ridges.is_empty() {
            return Err(KernelError::Topology);
        }

        // Repartition the outside sets of the deleted facets. A point beyond a deleted facet that
        // is still outside the new hull is beyond some new facet (Barber et al., sec. 3), so only
        // the new facets need testing.
        let mut orphans: Vec<usize> = Vec::new();
        for &v in &visible {
            facets[v].alive = false;
            orphans.append(&mut facets[v].outside);
        }
        for q in orphans {
            if q == apex {
                continue;
            }
            for &g in &new_ids {
                let (sign, value) = pts.side(&facets[g].verts, &facets[g].plane, q);
                if sign > 0 {
                    facets[g].assign(q, value);
                    break;
                }
            }
        }
        for &g in &new_ids {
            if !facets[g].outside.is_empty() {
                pending.push(g);
            }
        }
    }

    let mut remap = vec![usize::MAX; facets.len()];
    let mut next = 0usize;
    for (f, facet) in facets.iter().enumerate() {
        if facet.alive {
            remap[f] = next;
            next += 1;
        }
    }
    let mut out = Vec::with_capacity(next);
    for facet in facets.into_iter().filter(|f| f.alive) {
        let neighbors: Vec<usize> = facet.nbrs.iter().map(|&n| remap[n]).collect();
        if neighbors.contains(&usize::MAX) {
            return Err(KernelError::Topology);
        }
        out.push(HullFacet {
            vertices: facet.verts,
            neighbors,
        });
    }
    Ok(Hull { facets: out })
}

/// Group adjacent facets lying in one hyperplane, as Qhull's default merging does for output
/// that is not triangulated (`Voronoi`, `HalfspaceIntersection`): two neighbours merge when the
/// neighbour's opposite vertex is exactly on the facet's hyperplane, or within `tol` of it
/// (normalized distance in the rescaled coordinates of `pts`). Only facets with equal `class`
/// merge. Returns a group id per facet, numbered in order of first appearance.
pub(crate) fn coplanar_groups(pts: &Points, hull: &Hull, class: &[u8], tol: f64) -> Vec<usize> {
    let nf = hull.facets.len();
    let mut parent: Vec<usize> = (0..nf).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for f in 0..nf {
        let facet = &hull.facets[f];
        let mut plane: Option<Plane> = None;
        for &g in &facet.neighbors {
            if g <= f || class[g] != class[f] {
                continue;
            }
            let Some(slot) = hull.facets[g].neighbors.iter().position(|&x| x == f) else {
                continue;
            };
            let opposite = hull.facets[g].vertices[slot];
            let plane = plane.get_or_insert_with(|| pts.plane(&facet.vertices));
            let (sign, value) = pts.side(&facet.vertices, plane, opposite);
            let coplanar = sign == 0 || (plane.norm > 0.0 && value.abs() / plane.norm <= tol);
            if coplanar {
                let (a, b) = (find(&mut parent, f), find(&mut parent, g));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut label = vec![usize::MAX; nf];
    let mut out = vec![0usize; nf];
    let mut next = 0usize;
    for f in 0..nf {
        let root = find(&mut parent, f);
        if label[root] == usize::MAX {
            label[root] = next;
            next += 1;
        }
        out[f] = label[root];
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// Floating-point geometry for outputs (never used for decisions)
// ══════════════════════════════════════════════════════════════════════

/// Unit normal, offset and (D-1)-volume of the hyperplane through the `D` points `rows` (each of
/// dimension `D`). The normal points to the side where `orient(rows ++ [p]) > 0`, so for a hull
/// facet it points outward and `normal . x + offset <= 0` holds inside.
pub(crate) fn hyperplane(rows: &[&[f64]]) -> (Vec<f64>, f64, f64) {
    let d = rows.len();
    let v0 = rows[0];
    let mut e = vec![0.0; (d - 1) * d];
    for i in 1..d {
        for j in 0..d {
            e[(i - 1) * d + j] = rows[i][j] - v0[j];
        }
    }
    let mut det = Vec::new();
    let mut perm = Vec::new();
    row_minors(&e, d - 1, d, &mut det, &mut perm);
    let full = (1usize << d) - 1;
    let mut normal: Vec<f64> = (0..d)
        .map(|j| {
            let minor = det[full ^ (1 << j)];
            if j.is_multiple_of(2) { -minor } else { minor }
        })
        .collect();
    let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut factorial = 1.0;
    for k in 2..d {
        factorial *= k as f64;
    }
    let measure = norm / factorial;
    if norm > 0.0 {
        for x in &mut normal {
            *x /= norm;
        }
    }
    let offset = -normal.iter().zip(v0).map(|(n, v)| n * v).sum::<f64>();
    (normal, offset, measure)
}

/// Determinant by Gaussian elimination with partial pivoting.
fn gauss_determinant(mut m: Vec<Vec<f64>>) -> f64 {
    let n = m.len();
    let mut det = 1.0;
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            m.swap(pivot, col);
            det = -det;
        }
        det *= m[col][col];
        for row in col + 1..n {
            let factor = m[row][col] / m[col][col];
            if factor != 0.0 {
                for k in col..n {
                    let delta = factor * m[col][k];
                    m[row][k] -= delta;
                }
            }
        }
    }
    det
}

/// Qhull's `qh_determinant`: closed forms in 2-D and 3-D, elimination above.
fn qhull_determinant(m: &[Vec<f64>]) -> f64 {
    match m.len() {
        1 => m[0][0],
        2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
        3 => {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
                + m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
        }
        _ => gauss_determinant(m.to_vec()),
    }
}

/// Circumcenter of the simplex `rows` (`d + 1` points of dimension `d`) by Cramer's rule, the
/// way Qhull's `qh_voronoi_center` computes a Voronoi vertex. `None` for a degenerate simplex.
pub(crate) fn circumcenter(rows: &[&[f64]]) -> Option<Vec<f64>> {
    let d = rows.len() - 1;
    let p0 = rows[0];
    // m[k][j] = (p_{j+1} - p_0)[k]; the difference vectors are the columns.
    let m: Vec<Vec<f64>> = (0..d)
        .map(|k| (1..=d).map(|j| rows[j][k] - p0[k]).collect())
        .collect();
    let sums: Vec<f64> = (0..d)
        .map(|j| (0..d).map(|k| m[k][j] * m[k][j]).sum())
        .collect();
    let det = qhull_determinant(&m);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let factor = 0.5 / det;
    let mut center = vec![0.0; d];
    for i in 0..d {
        let mut replaced = m.clone();
        replaced[i].clone_from(&sums);
        center[i] = qhull_determinant(&replaced) * factor + p0[i];
    }
    center.iter().all(|x| x.is_finite()).then_some(center)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(rows: &[&[f64]]) -> Points {
        let dim = rows[0].len();
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Points::new(&flat, dim)
    }

    #[test]
    fn exact_sign_resolves_what_the_filter_cannot() {
        // Every point has x == y bit for bit, so the three stored doubles are exactly collinear
        // and the determinant is exactly zero.
        let a = [0.1, 0.1];
        let b = [0.2, 0.2];
        let c = [0.3, 0.3];
        let rows: [&[f64]; 3] = [&a, &b, &c];
        assert_eq!(exact_homogeneous_sign(&rows), 0);
        assert_eq!(pts(&rows).orient(&[0, 1, 2]), 0);

        // One ulp off the line: the determinant is ~1e-18, far below the filter's error bound,
        // so only the exact path can see its sign, and it must see it in both directions.
        let c_up = [0.3, f64::from_bits(0.3_f64.to_bits() + 1)];
        let rows: [&[f64]; 3] = [&a, &b, &c_up];
        assert_eq!(exact_homogeneous_sign(&rows), 1);
        assert_eq!(pts(&rows).orient(&[0, 1, 2]), 1);
        let c_down = [0.3, f64::from_bits(0.3_f64.to_bits() - 1)];
        let rows: [&[f64]; 3] = [&a, &b, &c_down];
        assert_eq!(exact_homogeneous_sign(&rows), -1);
        assert_eq!(pts(&rows).orient(&[0, 1, 2]), -1);
    }

    #[test]
    fn exact_sign_matches_float_on_well_conditioned_input_in_every_dimension() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for dim in 2..=6 {
            for _ in 0..20 {
                let rows_owned: Vec<Vec<f64>> = (0..=dim)
                    .map(|_| (0..dim).map(|_| next()).collect())
                    .collect();
                let rows: Vec<&[f64]> = rows_owned.iter().map(Vec::as_slice).collect();
                let m: Vec<Vec<f64>> = rows_owned
                    .iter()
                    .map(|r| {
                        let mut row = r.clone();
                        row.push(1.0);
                        row
                    })
                    .collect();
                let float = gauss_determinant(m);
                let exact = exact_homogeneous_sign(&rows);
                assert_eq!(exact, if float > 0.0 { 1 } else { -1 }, "dim {dim}");
                let flat: Vec<f64> = rows_owned.iter().flatten().copied().collect();
                let p = Points::new(&flat, dim);
                let idx: Vec<usize> = (0..=dim).collect();
                assert_eq!(p.orient(&idx), exact, "dim {dim}");
            }
        }
    }

    #[test]
    fn quickhull_cube_has_twelve_facets_and_every_facet_faces_out() {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    rows.push(vec![x, y, z]);
                }
            }
        }
        rows.push(vec![0.5, 0.5, 0.5]);
        rows.push(vec![0.5, 0.5, 0.0]); // on a face: must not become a vertex
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let p = Points::new(&flat, 3);
        let hull = quickhull(&p, rows.len()).expect("cube hull");
        assert_eq!(hull.facets.len(), 12);
        for (f, facet) in hull.facets.iter().enumerate() {
            assert!(!facet.vertices.contains(&8) && !facet.vertices.contains(&9));
            for q in 0..rows.len() {
                let mut idx = facet.vertices.clone();
                idx.push(q);
                assert!(p.orient(&idx) <= 0, "point {q} beyond facet {f}");
            }
            for (k, &nb) in facet.neighbors.iter().enumerate() {
                let shared: Vec<usize> = facet
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != k)
                    .map(|(_, &v)| v)
                    .collect();
                assert!(shared.iter().all(|v| hull.facets[nb].vertices.contains(v)));
                assert!(!hull.facets[nb].vertices.contains(&facet.vertices[k]));
            }
        }
    }

    #[test]
    fn flat_input_is_refused() {
        let flat = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let p = Points::new(&flat, 3);
        assert_eq!(quickhull(&p, 4).unwrap_err(), KernelError::Flat);
        // Coplanar only up to rounding (1/3 is not representable): Qhull still calls it flat.
        let third = 1.0 / 3.0;
        let flat = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, third, third, third,
        ];
        let p = Points::new(&flat, 3);
        assert_eq!(quickhull(&p, 4).unwrap_err(), KernelError::Flat);
        let p = Points::new(&[0.0, 0.0, 1.0, 1.0], 2);
        assert_eq!(
            quickhull(&p, 2).unwrap_err(),
            KernelError::TooFewPoints { have: 2, need: 3 }
        );
    }

    #[test]
    fn circumcenter_of_right_triangle_is_hypotenuse_midpoint() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 2.0];
        let center = circumcenter(&[&a, &b, &c]).expect("center");
        assert!((center[0] - 2.0).abs() < 1e-15 && (center[1] - 1.0).abs() < 1e-15);
        assert!(circumcenter(&[&a, &b, &[8.0, 0.0]]).is_none());
    }
}
