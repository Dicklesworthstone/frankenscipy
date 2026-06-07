#![forbid(unsafe_code)]

//! Clustering algorithms for FrankenSciPy.
//!
//! Matches `scipy.cluster` core operations:
//! - `scipy.cluster.vq` — K-means clustering and vector quantization
//! - `scipy.cluster.hierarchy` — Hierarchical/agglomerative clustering

/// Error type for clustering operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    InvalidArgument(String),
    EmptyData,
    ConvergenceFailed(String),
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::EmptyData => write!(f, "empty data"),
            Self::ConvergenceFailed(msg) => write!(f, "convergence failed: {msg}"),
        }
    }
}

impl std::error::Error for ClusterError {}

// ══════════════════════════════════════════════════════════════════════
// K-Means Clustering
// ══════════════════════════════════════════════════════════════════════

/// Result of K-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster centroids, shape (k, d).
    pub centroids: Vec<Vec<f64>>,
    /// Cluster labels for each data point.
    pub labels: Vec<usize>,
    /// Sum of squared distances to nearest centroid (inertia).
    pub inertia: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
}

/// K-means clustering using Lloyd's algorithm.
///
/// Matches `scipy.cluster.vq.kmeans2`.
///
/// # Arguments
/// * `data` — Input data points, each a Vec of features.
/// * `k` — Number of clusters.
/// * `max_iter` — Maximum iterations.
/// * `seed` — Random seed for initialization.
pub fn kmeans(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<KMeansResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    let d = validate_feature_dimensions(data, "kmeans")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "kmeans input must be finite".to_string(),
        ));
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "k={k} must be in [1, n={n}]"
        )));
    }

    // K-means++ initialization
    let mut centroids = kmeans_plusplus_init(data, k, seed);
    let mut labels = vec![0usize; n];
    let mut inertia = f64::INFINITY;

    for iter in 0..max_iter {
        // Assignment step: each point's nearest centroid is independent, so compute
        // (label, min_dist) in parallel. The inertia is then summed SEQUENTIALLY in
        // point order so its floating-point reduction — and therefore the convergence
        // check and iteration count — stay bit-identical to the serial version.
        let centroids_flat = flatten_centroids(&centroids, d);
        let assignments = assign_points(data, &centroids_flat, k, d);
        let mut new_inertia = 0.0;
        for (i, &(best_c, min_dist)) in assignments.iter().enumerate() {
            labels[i] = best_c;
            new_inertia += min_dist;
        }

        // Check convergence
        if (inertia - new_inertia).abs() < 1e-12 * inertia.abs().max(1.0) {
            return Ok(KMeansResult {
                centroids,
                labels,
                inertia: new_inertia,
                n_iter: iter + 1,
            });
        }
        inertia = new_inertia;

        // Update step
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![0.0; d]; k];
        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for val in &mut new_centroids[c][..d] {
                    *val /= counts[c] as f64;
                }
            } else {
                // Empty cluster: keep old centroid
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: max_iter,
    })
}

/// K-means++ initialization.
fn kmeans_plusplus_init(data: &[Vec<f64>], k: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut rng = seed;
    let next_rng = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random
    let idx = (next_rng(&mut rng) * n as f64) as usize % n;
    centroids.push(data[idx].clone());

    // `dists[i]` is the squared distance from point i to its NEAREST chosen
    // centroid. The previous code rebuilt it from scratch over every chosen
    // centroid on each of the k-1 picks — O(n·k²). Instead carry it across picks
    // and fold in only the newest centroid (O(n·k) total). `min` is independent
    // of fold order and `sq_dist_within`'s early-abandon never changes the min
    // value, so `dists` — and therefore every selection (same RNG sequence) — is
    // byte-identical to the rebuild.
    let mut dists = vec![f64::INFINITY; n];
    for (i, point) in data.iter().enumerate() {
        let d = sq_dist_within(point, &centroids[0], dists[i]);
        dists[i] = dists[i].min(d);
    }

    // Remaining centroids: probability proportional to D²
    for _ in 1..k {
        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            // All points are at existing centroids; pick randomly
            let idx = (next_rng(&mut rng) * n as f64) as usize % n;
            centroids.push(data[idx].clone());
            let new_c = data[idx].clone();
            for (i, point) in data.iter().enumerate() {
                let d = sq_dist_within(point, &new_c, dists[i]);
                dists[i] = dists[i].min(d);
            }
            continue;
        }

        let threshold = next_rng(&mut rng) * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
        // Fold the new centroid into the running nearest-centroid distances.
        let new_c = data[chosen].clone();
        for (i, point) in data.iter().enumerate() {
            let d = sq_dist_within(point, &new_c, dists[i]);
            dists[i] = dists[i].min(d);
        }
    }

    centroids
}

/// Mini-batch K-means for large datasets.
///
/// Matches `sklearn.cluster.MiniBatchKMeans` (used alongside scipy).
pub fn mini_batch_kmeans(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    batch_size: usize,
    seed: u64,
) -> Result<KMeansResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    let d = validate_feature_dimensions(data, "mini_batch_kmeans")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "mini_batch_kmeans input must be finite".to_string(),
        ));
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "k={k} must be in [1, n={n}]"
        )));
    }
    if batch_size == 0 {
        return Err(ClusterError::InvalidArgument(
            "batch_size must be at least 1".to_string(),
        ));
    }
    let batch = batch_size.min(n);

    let mut centroids = kmeans_plusplus_init(data, k, seed);
    let mut counts = vec![0usize; k];
    let mut rng = seed.wrapping_add(12345);

    for _ in 0..max_iter {
        // Sample mini-batch
        let mut batch_indices = Vec::with_capacity(batch);
        for _ in 0..batch {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            batch_indices.push((rng >> 33) as usize % n);
        }

        // Assignment
        let centroids_flat = flatten_centroids(&centroids, d);
        let mut batch_labels = Vec::with_capacity(batch);
        for &idx in &batch_indices {
            let (best_c, _) = nearest_centroid(&data[idx], &centroids_flat, k, d);
            batch_labels.push(best_c);
        }

        // Update with learning rate
        for (bi, &idx) in batch_indices.iter().enumerate() {
            let c = batch_labels[bi];
            counts[c] += 1;
            let lr = 1.0 / counts[c] as f64;
            for j in 0..d {
                centroids[c][j] += lr * (data[idx][j] - centroids[c][j]);
            }
        }
    }

    // Final assignment
    let mut labels = vec![0usize; n];
    let mut inertia = 0.0;
    let centroids_flat = flatten_centroids(&centroids, d);
    for (i, point) in data.iter().enumerate() {
        let (best_c, min_dist) = nearest_centroid(point, &centroids_flat, k, d);
        labels[i] = best_c;
        inertia += min_dist;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: max_iter,
    })
}

/// Vector quantization: assign each observation to the nearest centroid.
///
/// Matches `scipy.cluster.vq.vq`.
pub fn vq(
    data: &[Vec<f64>],
    centroids: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<f64>), ClusterError> {
    if data.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    if centroids.is_empty() {
        return Err(ClusterError::InvalidArgument(
            "vq requires at least one centroid".to_string(),
        ));
    }
    let d = validate_feature_dimensions(data, "vq")?;
    let cd = validate_feature_dimensions(centroids, "vq centroids")?;
    if d != cd {
        return Err(ClusterError::InvalidArgument(format!(
            "vq data dimension {} must match centroid dimension {}",
            d, cd
        )));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "vq data must be finite".to_string(),
        ));
    }
    if centroids.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "vq centroids must be finite".to_string(),
        ));
    }

    let mut labels = Vec::with_capacity(data.len());
    let mut dists = Vec::with_capacity(data.len());

    // Resolves [frankenscipy-dnunq]: compare in squared-distance space
    // so we only sqrt the winning min_sq once per data point instead
    // of once per (point, centroid) pair. Monotone-equivalent for
    // nearest-neighbor selection.
    let k = centroids.len();
    let centroids_flat = flatten_centroids(centroids, d);
    for point in data {
        let (best_c, min_sq) = nearest_centroid(point, &centroids_flat, k, d);
        labels.push(best_c);
        dists.push(min_sq.sqrt());
    }

    Ok((labels, dists))
}

/// Whiten observations by dividing by per-feature standard deviation.
///
/// Matches `scipy.cluster.vq.whiten`.
pub fn whiten(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ClusterError> {
    if data.is_empty() {
        return Ok(vec![]);
    }
    let d = validate_feature_dimensions(data, "whiten")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "whiten input must be finite".to_string(),
        ));
    }
    let n = data.len();

    // Compute per-feature std
    let mut means = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    let mut vars = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            vars[j] += (v - means[j]).powi(2);
        }
    }

    let stds: Vec<f64> = vars.iter().map(|&v| (v / n as f64).sqrt()).collect();

    Ok(data
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(stds.iter())
                .map(|(&v, &s)| if s > 0.0 { v / s } else { v })
                .collect()
        })
        .collect())
}

// ══════════════════════════════════════════════════════════════════════
// Hierarchical Clustering
// ══════════════════════════════════════════════════════════════════════

/// Linkage methods for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Ward,
    /// br-7kxr: weighted-average linkage. Lance-Williams parameters
    /// α_i = α_j = 0.5, β = γ = 0 → unweighted-pair update.
    Weighted,
    /// br-7kxr: centroid linkage. Distance between cluster centroids.
    /// Lance-Williams α_i = nᵢ/(nᵢ+nⱼ), α_j = nⱼ/(nᵢ+nⱼ),
    /// β = -nᵢ·nⱼ/(nᵢ+nⱼ)², γ = 0.
    Centroid,
    /// br-7kxr: median linkage. Lance-Williams α_i = α_j = 0.5,
    /// β = -0.25, γ = 0.
    Median,
}

// br-6m7l: indexed min-heap mirroring scipy's _structures.pxi Heap.
// Stores values keyed by 0..n. sift_down/sift_up use strict `>` so
// ties don't swap, preserving the order set by build_heap. Used by
// linkage_fast for Centroid/Median tie-break parity with scipy.
struct IndexedMinHeap {
    values: Vec<f64>,
    key_by_index: Vec<usize>,
    index_by_key: Vec<usize>,
    size: usize,
}

impl IndexedMinHeap {
    fn new(values: Vec<f64>) -> Self {
        let n = values.len();
        let mut h = IndexedMinHeap {
            values,
            key_by_index: (0..n).collect(),
            index_by_key: (0..n).collect(),
            size: n,
        };
        for i in (0..n / 2).rev() {
            h.sift_down(i);
        }
        h
    }

    fn get_min(&self) -> (usize, f64) {
        (self.key_by_index[0], self.values[0])
    }

    fn remove_min(&mut self) {
        if self.size == 0 {
            return;
        }
        self.swap(0, self.size - 1);
        self.size -= 1;
        if self.size > 0 {
            self.sift_down(0);
        }
    }

    fn change_value(&mut self, key: usize, value: f64) {
        let index = self.index_by_key[key];
        let old = self.values[index];
        self.values[index] = value;
        if value < old {
            self.sift_up(index);
        } else {
            self.sift_down(index);
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.values.swap(i, j);
        let ki = self.key_by_index[i];
        let kj = self.key_by_index[j];
        self.key_by_index.swap(i, j);
        self.index_by_key[ki] = j;
        self.index_by_key[kj] = i;
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.values[parent] > self.values[index] {
                self.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = 2 * index + 1;
            if left >= self.size {
                return;
            }
            let mut child = left;
            let right = left + 1;
            if right < self.size && self.values[right] < self.values[left] {
                child = right;
            }
            if self.values[index] > self.values[child] {
                self.swap(index, child);
                index = child;
            } else {
                return;
            }
        }
    }
}

// br-6m7l: scan upper-triangular row x for the lowest-index active
// neighbor with smallest distance. Mirrors scipy's find_min_dist.
fn find_min_dist_row(d: &[Vec<f64>], size: &[usize], x: usize, n: usize) -> (Option<usize>, f64) {
    let mut current_min = f64::INFINITY;
    let mut y_out: Option<usize> = None;
    for i in (x + 1)..n {
        if size[i] == 0 {
            continue;
        }
        let dist = d[x][i];
        if dist < current_min {
            current_min = dist;
            y_out = Some(i);
        }
    }
    (y_out, current_min)
}

// br-6m7l: scipy's fast_linkage algorithm ported to Rust. Used for
// Centroid and Median methods to match scipy's heap-based tie-break
// when multiple inter-cluster distances coincide. Lance-Williams
// updates are inlined per method.
fn linkage_fast(n: usize, initial_d: &[Vec<f64>], method: LinkageMethod) -> Vec<[f64; 4]> {
    let mut d = initial_d.to_vec();
    let mut size = vec![1usize; n];
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut neighbor = vec![0usize; n.saturating_sub(1)];
    let mut min_dist_arr = vec![f64::INFINITY; n.saturating_sub(1)];

    for x in 0..n.saturating_sub(1) {
        let (y_opt, dist) = find_min_dist_row(&d, &size, x, n);
        neighbor[x] = y_opt.unwrap_or(x);
        min_dist_arr[x] = dist;
    }
    let mut heap = IndexedMinHeap::new(min_dist_arr.clone());

    let mut z: Vec<[f64; 4]> = Vec::with_capacity(n.saturating_sub(1));
    for k in 0..n.saturating_sub(1) {
        // Pop from heap until cached neighbor matches current D.
        let mut x = 0usize;
        let mut y = 0usize;
        let mut dist = 0.0f64;
        for _ in 0..(n - k) {
            let (xk, dk) = heap.get_min();
            x = xk;
            dist = dk;
            y = neighbor[x];
            if dist == d[x][y] {
                break;
            }
            // Stale; recompute from scratch.
            let (y_new_opt, dist_new) = find_min_dist_row(&d, &size, x, n);
            if let Some(y_new) = y_new_opt {
                y = y_new;
                dist = dist_new;
                neighbor[x] = y_new;
                heap.change_value(x, dist_new);
            } else {
                heap.change_value(x, f64::INFINITY);
            }
        }
        heap.remove_min();

        let nx = size[x];
        let ny = size[y];
        let mut id_x = cluster_id[x];
        let mut id_y = cluster_id[y];
        if id_x > id_y {
            std::mem::swap(&mut id_x, &mut id_y);
        }
        z.push([id_x as f64, id_y as f64, dist, (nx + ny) as f64]);

        size[x] = 0;
        size[y] = nx + ny;
        cluster_id[y] = n + k;

        // Lance-Williams update of D for column y.
        for zi in 0..n {
            if size[zi] == 0 || zi == y {
                continue;
            }
            let d_zx = d[zi.min(x)][zi.max(x)];
            let d_zy = d[zi.min(y)][zi.max(y)];
            let new_d = match method {
                LinkageMethod::Centroid => {
                    let nif = nx as f64;
                    let njf = ny as f64;
                    let nt = nif + njf;
                    let alpha_i = nif / nt;
                    let alpha_j = njf / nt;
                    let beta = -(nif * njf) / (nt * nt);
                    (alpha_i * d_zx * d_zx + alpha_j * d_zy * d_zy + beta * dist * dist)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Median => (0.5 * d_zx * d_zx + 0.5 * d_zy * d_zy
                    - 0.25 * dist * dist)
                    .max(0.0)
                    .sqrt(),
                _ => unreachable!("linkage_fast only for Centroid/Median"),
            };
            // Store symmetrically.
            let lo = zi.min(y);
            let hi = zi.max(y);
            d[lo][hi] = new_d;
            d[hi][lo] = new_d;
        }

        // Reassign neighbor pointers that referenced the just-killed x.
        for zi in 0..x {
            if size[zi] > 0 && neighbor[zi] == x {
                neighbor[zi] = y;
            }
        }

        // Update lower bounds: any zi < y whose neighbor distance to y
        // tightened gets updated.
        for zi in 0..y {
            if size[zi] == 0 {
                continue;
            }
            let dz = d[zi][y];
            if dz < min_dist_arr[zi] {
                neighbor[zi] = y;
                min_dist_arr[zi] = dz;
                heap.change_value(zi, dz);
            }
        }

        // Rescan neighbor for y.
        if y < n - 1 {
            let (z_opt, dist_y) = find_min_dist_row(&d, &size, y, n);
            if let Some(z_new) = z_opt {
                neighbor[y] = z_new;
                min_dist_arr[y] = dist_y;
                heap.change_value(y, dist_y);
            } else {
                heap.change_value(y, f64::INFINITY);
            }
        }
    }
    z
}

/// Hierarchical clustering linkage matrix.
///
/// Each row [i, j, dist, count] represents a merge: clusters i and j
/// merged at distance dist, producing a cluster with count observations.
///
/// Matches `scipy.cluster.hierarchy.linkage`.
/// Nearest active successor (smallest `j > i`, both active, minimum distance)
/// of cluster `i`, scanning the same ascending-`j` strict-`<` order the naive
/// pairwise scan uses, so ties resolve to the same `j`.
fn agglo_nearest(inter_dist: &[Vec<f64>], active: &[bool], i: usize, total: usize) -> (usize, f64) {
    let mut best_j = i;
    let mut best_d = f64::INFINITY;
    for j in (i + 1)..total {
        if active[j] && inter_dist[i][j] < best_d {
            best_d = inter_dist[i][j];
            best_j = j;
        }
    }
    (best_j, best_d)
}

/// Agglomerative clustering core shared by `linkage` and `linkage_from_distances`.
///
/// Byte-identical to the naive O(n^3) "rescan every pair each step" loop, but
/// O(n^2) typical: a nearest-neighbour array keeps each active cluster's nearest
/// active successor, so the closest pair is found in O(active) instead of
/// O(active^2). Because the global minimum (and its smallest-`i`/smallest-`j`
/// tie-break) is identical each step, the merge sequence — and every
/// Lance-Williams distance, computed with the exact same operands — matches the
/// pairwise scan element-for-element.
fn agglomerate_nnarray(
    n: usize,
    mut inter_dist: Vec<Vec<f64>>,
    method: LinkageMethod,
) -> Vec<[f64; 4]> {
    let total = 2 * n - 1;
    let mut active = vec![false; total];
    active[..n].fill(true);
    let mut cluster_size = vec![1usize; total];
    let mut nn = vec![0usize; total];
    let mut d_nn = vec![f64::INFINITY; total];
    for i in 0..n {
        let (j, d) = agglo_nearest(&inter_dist, &active, i, total);
        nn[i] = j;
        d_nn[i] = d;
    }

    let mut result = Vec::with_capacity(n - 1);
    for step in 0..n - 1 {
        let new_id = n + step;

        // Closest active pair = smallest active i with minimal d_nn[i]; its
        // recorded neighbour nn[i] is the smallest-index minimiser j > i.
        let mut min_d = f64::INFINITY;
        let mut mi = 0;
        for i in 0..new_id {
            if active[i] && d_nn[i] < min_d {
                min_d = d_nn[i];
                mi = i;
            }
        }
        let mj = nn[mi];

        let new_size = cluster_size[mi] + cluster_size[mj];
        result.push([mi as f64, mj as f64, min_d, new_size as f64]);

        active[mi] = false;
        active[mj] = false;
        active[new_id] = true;
        cluster_size[new_id] = new_size;

        // Distances from the new cluster to every remaining active cluster.
        for k in 0..new_id {
            if !active[k] {
                continue;
            }
            let d_ki = inter_dist[k][mi];
            let d_kj = inter_dist[k][mj];
            let new_dist = match method {
                LinkageMethod::Single => d_ki.min(d_kj),
                LinkageMethod::Complete => d_ki.max(d_kj),
                LinkageMethod::Average => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    (ni * d_ki + nj * d_kj) / (ni + nj)
                }
                LinkageMethod::Ward => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nk = cluster_size[k] as f64;
                    let nt = ni + nj + nk;
                    (((nk + ni) * d_ki * d_ki + (nk + nj) * d_kj * d_kj - nk * min_d * min_d) / nt)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Weighted => 0.5 * (d_ki + d_kj),
                LinkageMethod::Centroid => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nt = ni + nj;
                    let alpha_i = ni / nt;
                    let alpha_j = nj / nt;
                    let beta = -(ni * nj) / (nt * nt);
                    (alpha_i * d_ki * d_ki + alpha_j * d_kj * d_kj + beta * min_d * min_d)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Median => (0.5 * d_ki * d_ki + 0.5 * d_kj * d_kj
                    - 0.25 * min_d * min_d)
                    .max(0.0)
                    .sqrt(),
            };
            inter_dist[k][new_id] = new_dist;
            inter_dist[new_id][k] = new_dist;
        }

        // The new cluster is the largest active id, so it has no successor yet.
        d_nn[new_id] = f64::INFINITY;
        nn[new_id] = new_id;

        // Refresh nearest neighbours: clusters that pointed at a merged cluster
        // recompute from scratch; the rest only need to test the new cluster as
        // a (strictly closer) candidate, matching the scan's smallest-j tie-break.
        for k in 0..new_id {
            if !active[k] {
                continue;
            }
            if nn[k] == mi || nn[k] == mj {
                let (j, d) = agglo_nearest(&inter_dist, &active, k, total);
                nn[k] = j;
                d_nn[k] = d;
            } else if inter_dist[k][new_id] < d_nn[k] {
                d_nn[k] = inter_dist[k][new_id];
                nn[k] = new_id;
            }
        }
    }

    result
}

pub fn linkage(data: &[Vec<f64>], method: LinkageMethod) -> Result<Vec<[f64; 4]>, ClusterError> {
    let n = data.len();
    if n < 2 {
        return Err(ClusterError::InvalidArgument(
            "need at least 2 observations".to_string(),
        ));
    }
    validate_feature_dimensions(data, "linkage")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "linkage input must be finite".to_string(),
        ));
    }

    // Compute pairwise distance matrix
    let mut dist_mat = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        dist_mat[i][i] = 0.0;
        for j in i + 1..n {
            let d = sq_dist(&data[i], &data[j]).sqrt();
            dist_mat[i][j] = d;
            dist_mat[j][i] = d;
        }
    }

    // br-6m7l: scipy uses fast_linkage (heap-based "Generic Clustering
    // Algorithm" from Mullner 2011) for Centroid/Median; fsci's simpler
    // O(n^3) scan diverges from scipy on tie-break of the column-0/1
    // cluster IDs. Route those two methods through the heap path so the
    // resulting linkage matrix matches scipy element-for-element.
    if matches!(method, LinkageMethod::Centroid | LinkageMethod::Median) {
        return Ok(linkage_fast(n, &dist_mat, method));
    }

    // Extend dist_mat to handle new clusters, then run the shared O(n^2)
    // nearest-neighbour-array agglomeration (byte-identical to the old scan).
    let total = 2 * n - 1;
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];
    for i in 0..n {
        inter_dist[i][..n].copy_from_slice(&dist_mat[i][..n]);
    }

    Ok(agglomerate_nnarray(n, inter_dist, method))
}

/// Cut a linkage tree to form flat clusters.
///
/// Matches `scipy.cluster.hierarchy.fcluster` with criterion='maxclust'.
pub fn fcluster(z: &[[f64; 4]], max_clusters: usize) -> Result<Vec<usize>, ClusterError> {
    if !is_valid_linkage(z) {
        return Err(ClusterError::InvalidArgument(
            "invalid linkage matrix".to_string(),
        ));
    }

    let n = z.len() + 1;
    if max_clusters >= n || max_clusters == 0 {
        return Ok((1..=n).collect());
    }

    // Union-find over the 2n-1 dendrogram nodes. Each set's label is the minimum
    // original-leaf index it contains — exactly what the previous code's
    // min-propagation produced — so the final renumbering is byte-identical, but
    // agglomeration is O(n·α(n)) instead of relabeling every node on each of the
    // up-to-n merges (the old O(n²) `for v in cluster_of` rescan).
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut min_leaf: Vec<usize> = (0..total)
        .map(|i| if i < n { i } else { usize::MAX })
        .collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    // Process merges in order, stopping when we have max_clusters
    let n_merges = n - max_clusters;
    for (step, row) in z.iter().enumerate().take(n_merges) {
        let new_id = n + step;
        let ci = find(&mut parent, row[0] as usize);
        let cj = find(&mut parent, row[1] as usize);
        // Root the merged set at the new node, carrying the minimum leaf index.
        let label = min_leaf[ci].min(min_leaf[cj]).min(min_leaf[new_id]);
        parent[ci] = new_id;
        parent[cj] = new_id;
        min_leaf[new_id] = label;
    }

    // Renumber labels to be contiguous 1..k
    let leaf_labels: Vec<usize> = (0..n)
        .map(|i| {
            let r = find(&mut parent, i);
            min_leaf[r]
        })
        .collect();
    let mut unique: Vec<usize> = leaf_labels.clone();
    unique.sort_unstable();
    unique.dedup();
    Ok(leaf_labels
        .iter()
        .map(|&l| unique.binary_search(&l).unwrap_or(0) + 1)
        .collect())
}

/// Validate a linkage matrix.
///
/// Checks that the linkage matrix has valid structure:
/// - 4 columns (idx1, idx2, distance, count)
/// - Valid cluster indices (0..n for original, n..2n-1 for merged)
/// - Distances are non-negative
/// - Counts are positive
///
/// Matches `scipy.cluster.hierarchy.is_valid_linkage`.
pub fn is_valid_linkage(z: &[[f64; 4]]) -> bool {
    if z.is_empty() {
        return true; // Empty linkage is valid (0 or 1 observations)
    }

    let n = z.len() + 1; // number of original observations
    let mut used_as_child = vec![false; 2 * n - 1];

    for (step, row) in z.iter().enumerate() {
        let Some(ci) = finite_usize_index(row[0]) else {
            return false;
        };
        let Some(cj) = finite_usize_index(row[1]) else {
            return false;
        };
        let dist = row[2];
        let Some(count) = finite_usize_index(row[3]) else {
            return false;
        };

        // Check cluster indices are valid
        let max_valid_idx = n + step; // can reference clusters 0..n+step
        if ci >= max_valid_idx || cj >= max_valid_idx {
            return false;
        }
        if ci == cj {
            return false; // can't merge cluster with itself
        }
        if used_as_child[ci] || used_as_child[cj] {
            return false; // each cluster can participate in only one later merge
        }
        used_as_child[ci] = true;
        used_as_child[cj] = true;

        // Distance must be non-negative and finite
        if !dist.is_finite() || dist < 0.0 {
            return false;
        }

        // Count must be positive
        if count == 0 {
            return false;
        }
    }

    true
}

fn finite_usize_index(value: f64) -> Option<usize> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64 {
        Some(value as usize)
    } else {
        None
    }
}

/// Check if linkage distances are monotonically non-decreasing.
///
/// Matches `scipy.cluster.hierarchy.is_monotonic`.
pub fn is_monotonic(z: &[[f64; 4]]) -> bool {
    if z.len() < 2 {
        return true;
    }
    for i in 1..z.len() {
        if z[i][2] < z[i - 1][2] {
            return false;
        }
    }
    true
}

/// Get the number of original observations from a linkage matrix.
///
/// Returns 0 if the linkage matrix is invalid.
/// Matches `scipy.cluster.hierarchy.num_obs_linkage`.
pub fn num_obs_linkage(z: &[[f64; 4]]) -> usize {
    if !is_valid_linkage(z) && !z.is_empty() {
        return 0;
    }
    z.len() + 1
}

/// Return the leaf ordering from a linkage matrix.
///
/// Performs a depth-first traversal of the dendrogram tree
/// and returns indices of the original observations in order.
/// Returns empty vec if linkage is invalid.
///
/// Matches `scipy.cluster.hierarchy.leaves_list`.
pub fn leaves_list(z: &[[f64; 4]]) -> Vec<usize> {
    let n = z.len() + 1;
    if n <= 1 {
        return (0..n).collect();
    }
    if !is_valid_linkage(z) {
        return vec![];
    }

    let mut result = Vec::with_capacity(n);
    let root = 2 * n - 2; // last cluster formed

    fn traverse(z: &[[f64; 4]], node: usize, n: usize, result: &mut Vec<usize>) {
        if node < n {
            result.push(node);
        } else {
            let step = node - n;
            if step < z.len() {
                traverse(z, z[step][0] as usize, n, result);
                traverse(z, z[step][1] as usize, n, result);
            }
        }
    }

    traverse(z, root, n, &mut result);
    result
}

/// Cluster data directly from observations.
///
/// Combines distance computation, linkage, and fcluster into one step.
/// Equivalent to calling `linkage` then `fcluster`.
///
/// Matches `scipy.cluster.hierarchy.fclusterdata`.
pub fn fclusterdata(
    data: &[Vec<f64>],
    max_clusters: usize,
    method: LinkageMethod,
) -> Result<Vec<usize>, ClusterError> {
    let z = linkage(data, method)?;
    fcluster(&z, max_clusters)
}

/// Compute cophenetic distances from a linkage matrix.
///
/// Returns the cophenetic distance matrix (condensed form).
/// Returns empty vec for empty linkage (0 or 1 observations).
/// Returns vec of NaN if linkage is invalid.
/// Matches `scipy.cluster.hierarchy.cophenet`.
pub fn cophenet(z: &[[f64; 4]]) -> Vec<f64> {
    if z.is_empty() {
        return vec![];
    }
    if !is_valid_linkage(z) {
        let n = z.len() + 1;
        return vec![f64::NAN; n * (n - 1) / 2];
    }
    let n = z.len() + 1;
    let mut membership = vec![vec![]; 2 * n - 1];
    for (i, mem) in membership.iter_mut().enumerate().take(n) {
        *mem = vec![i];
    }

    let mut condensed = vec![0.0; n * (n - 1) / 2];

    for (step, row) in z.iter().enumerate() {
        let ci = row[0] as usize;
        let cj = row[1] as usize;
        let dist = row[2];
        let new_id = n + step;

        // Set cophenetic distance for all pairs (one from ci, one from cj)
        for &a in &membership[ci] {
            for &b in &membership[cj] {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let idx = n * lo - lo * (lo + 1) / 2 + hi - lo - 1;
                condensed[idx] = dist;
            }
        }

        // Merge membership lists
        let mut merged = membership[ci].clone();
        merged.extend_from_slice(&membership[cj]);
        membership[new_id] = merged;
    }

    condensed
}

/// Compute the inconsistency statistic for each merge.
///
/// Matches `scipy.cluster.hierarchy.inconsistent`.
pub fn inconsistent(z: &[[f64; 4]], depth: usize) -> Vec<[f64; 4]> {
    if z.is_empty() {
        return vec![];
    }
    if depth == 0 {
        return z.iter().map(|row| [row[2], 0.0, 1.0, 0.0]).collect();
    }
    let n = z.len() + 1;
    let mut result = Vec::with_capacity(z.len());

    for (step, row) in z.iter().enumerate() {
        // Collect distances of all merges within `depth` levels below this merge
        let mut dists = Vec::new();
        collect_depths(z, n + step, n, depth, &mut dists);

        let count = dists.len() as f64;
        let mean = if count > 0.0 {
            dists.iter().sum::<f64>() / count
        } else {
            row[2]
        };
        let std = if count > 1.0 {
            let var = dists.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        let incon = if std > 0.0 {
            (row[2] - mean) / std
        } else {
            0.0
        };

        result.push([mean, std, count, incon]);
    }

    result
}

fn collect_depths(z: &[[f64; 4]], node: usize, n: usize, depth: usize, dists: &mut Vec<f64>) {
    if depth == 0 || node < n {
        return;
    }
    let step = node - n;
    if step < z.len() {
        dists.push(z[step][2]);
        collect_depths(z, z[step][0] as usize, n, depth - 1, dists);
        collect_depths(z, z[step][1] as usize, n, depth - 1, dists);
    }
}

// ══════════════════════════════════════════════════════════════════════
// DBSCAN
// ══════════════════════════════════════════════════════════════════════

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Cluster labels (-1 = noise).
    pub labels: Vec<i64>,
    /// Indices of core samples.
    pub core_sample_indices: Vec<usize>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// DBSCAN density-based clustering.
///
/// Matches `sklearn.cluster.DBSCAN`.
pub fn dbscan(
    data: &[Vec<f64>],
    eps: f64,
    min_samples: usize,
) -> Result<DbscanResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    let d = validate_feature_dimensions(data, "dbscan")?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(ClusterError::InvalidArgument(
            "eps must be finite and positive".to_string(),
        ));
    }
    if min_samples == 0 {
        return Err(ClusterError::InvalidArgument(
            "min_samples must be at least 1".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "dbscan input must be finite".to_string(),
        ));
    }

    let eps2 = eps * eps;
    let mut labels = vec![-1i64; n];
    let mut visited = vec![false; n];
    let mut core_samples = Vec::new();
    let mut cluster_id = 0i64;

    // The neighbor scan is O(n²) and re-walks every point, so pack the ragged
    // rows into one contiguous n×d buffer once (kills the per-point heap-pointer
    // chase + cache miss; same lever as the assignment path). Then bound each
    // squared distance by `eps2` with partial-distance early abandonment: a pair
    // is a neighbor iff its full distance is `≤ eps2`, and `sq_dist_within` bails
    // out the instant the running sum *exceeds* `eps2` while summing any genuine
    // neighbor (full distance `≤ eps2`) to completion — so the `≤ eps2`
    // membership test is bit-identical to the unbounded `sq_dist`. In density
    // clustering most pairs are non-neighbors and abandon after a few dimensions.
    let flat = flatten_points(data, d);
    let row = |idx: usize| -> &[f64] { &flat[idx * d..idx * d + d] };

    // Spatial-grid acceleration (low dimensions): bucket points into cells of
    // side `eps`. Any neighbour (full distance ≤ eps) differs by ≤ 1 cell per
    // axis, so only the 3^d cells around a point can hold one. This turns the
    // O(n²) all-pairs scan into roughly O(n) for bounded density while staying
    // BYTE-IDENTICAL to the scan: the candidate set is filtered with the same
    // `sq_dist_within ≤ eps2` test and sorted ascending, reproducing the exact
    // membership and the 0..n index order the linear filter produced. In high
    // dimensions (3^d blows up and offers no pruning) keep the linear scan.
    let use_grid = d <= 6 && n >= 256;
    let cell_of = |p: &[f64]| -> Vec<i64> { (0..d).map(|k| (p[k] / eps).floor() as i64).collect() };
    let grid: Option<std::collections::HashMap<Vec<i64>, Vec<usize>>> = use_grid.then(|| {
        let mut g: std::collections::HashMap<Vec<i64>, Vec<usize>> =
            std::collections::HashMap::with_capacity(n);
        for idx in 0..n {
            g.entry(cell_of(row(idx))).or_default().push(idx);
        }
        g
    });

    let neighbors = |idx: usize| -> Vec<usize> {
        let pi = row(idx);
        let Some(g) = &grid else {
            return (0..n)
                .filter(|&j| sq_dist_within(pi, row(j), eps2) <= eps2)
                .collect();
        };
        let base = cell_of(pi);
        let mut cell = base.clone();
        let mut out = Vec::new();
        for code in 0..3usize.pow(d as u32) {
            let mut c = code;
            for k in 0..d {
                cell[k] = base[k] + (c % 3) as i64 - 1;
                c /= 3;
            }
            if let Some(idxs) = g.get(&cell) {
                for &j in idxs {
                    if sq_dist_within(pi, row(j), eps2) <= eps2 {
                        out.push(j);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let nbrs = neighbors(i);
        if nbrs.len() < min_samples {
            // Noise (may be reclassified later)
            continue;
        }

        // Core point: start new cluster
        core_samples.push(i);
        labels[i] = cluster_id;

        let mut queue = std::collections::VecDeque::from(nbrs);
        while let Some(j) = queue.pop_front() {
            if labels[j] == -1 {
                labels[j] = cluster_id; // was noise, now border
            }
            if visited[j] {
                continue;
            }
            visited[j] = true;

            let j_nbrs = neighbors(j);
            if j_nbrs.len() >= min_samples {
                core_samples.push(j);
                for &nb in &j_nbrs {
                    if labels[nb] == -1 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    core_samples.sort();
    core_samples.dedup();

    Ok(DbscanResult {
        labels,
        core_sample_indices: core_samples,
        n_clusters: cluster_id as usize,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn validate_feature_dimensions(data: &[Vec<f64>], context: &str) -> Result<usize, ClusterError> {
    let first = data.first().ok_or(ClusterError::EmptyData)?;
    let d = first.len();
    if d == 0 {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} input must have at least one feature"
        )));
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != d {
            return Err(ClusterError::InvalidArgument(format!(
                "{context} input rows must have consistent length; row 0 has {d} but row {i} has {}",
                row.len()
            )));
        }
    }
    Ok(d)
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Leading dimensions probed to seed a tight incumbent bound before the full
/// nearest-centroid scan. A handful of coordinates is enough to identify a
/// likely-nearest centroid in well-separated data, after which partial-distance
/// abandonment rejects the rest cheaply. Purely a performance knob — it never
/// changes which centroid is ultimately selected.
const PREFILTER_DIMS: usize = 8;

/// Squared Euclidean distance with partial-distance early abandonment.
///
/// Accumulates `Σ (aᵢ − bᵢ)²` in ascending index order and bails out the moment
/// the running sum *exceeds* `bound`, returning that partial sum (`> bound`).
/// Every term is a square and therefore non-negative, so the running sum is
/// monotone: once it passes `bound` the full distance can only be larger. A pair
/// whose full distance equals `bound` is summed to completion and returned
/// exactly, so callers can break ties deterministically. The winning centroid is
/// never abandoned (its partial sums stay `≤` the incumbent), so the stored
/// minimum is always a fully-summed distance — labels and inertia are
/// bit-identical to the unbounded path. See
/// `tests/artifacts/perf/2026-06-03-cluster-vq-assign/` for the sha256 proof.
#[inline]
fn sq_dist_within(a: &[f64], b: &[f64], bound: f64) -> f64 {
    let mut acc = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff = ai - bi;
        acc += diff * diff;
        if acc > bound {
            return acc;
        }
    }
    acc
}

/// Pack a ragged observation list into one contiguous `n × d` row-major buffer.
/// Same rationale as [`flatten_centroids`]: an `O(n²)` neighbor scan re-walks
/// every row, so streaming a contiguous buffer beats chasing `n` heap pointers.
fn flatten_points(data: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(data.len() * d);
    for point in data {
        flat.extend_from_slice(&point[..d]);
    }
    flat
}

/// Pack a ragged centroid list into one contiguous `k × d` row-major buffer.
///
/// The assignment loops compare every observation against every centroid, so the
/// centroid set is re-walked `n` times. As `Vec<Vec<f64>>` the `k` rows are
/// scattered across the heap and each comparison pays a pointer load plus a cache
/// miss; flattening them once per assignment pass (`O(k·d)`, amortized over `n`
/// observations) lets the inner scan stream sequentially and prefetch cleanly.
fn flatten_centroids(centroids: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(centroids.len() * d);
    for centroid in centroids {
        flat.extend_from_slice(&centroid[..d]);
    }
    flat
}

/// Exact nearest-centroid search over a contiguous `k × d` centroid buffer:
/// returns `(index, squared_distance)` where `index` is the *lowest* index
/// attaining the minimum squared distance.
///
/// Identical in result to a plain `argmin` over [`sq_dist`] — same minimum value
/// (a full sequential sum) and same lowest-index tie-breaking — but cheap on two
/// axes the naive scan is not:
///
/// 1. Centroids live in one contiguous buffer (see [`flatten_centroids`]), so the
///    `c = 0..k` scan streams sequentially instead of chasing `k` heap pointers.
/// 2. A [`PREFILTER_DIMS`]-coordinate probe picks a likely-nearest centroid whose
///    *full* distance seeds a tight incumbent bound, so partial-distance
///    abandonment (strict-`>`, via [`sq_dist_within`]) rejects most centroids
///    after a few dimensions from the very first comparison. Ties are summed in
///    full and broken by lowest index (`sd == min_sq && c < best_c`), so the
///    result is independent of the seed and bit-identical to the naive argmin.
#[inline]
// Assign every point to its nearest centroid, returning the label and squared
// distance per point. For large n*k*d the points are split across threads; each pair
// comes from the same pure `nearest_centroid`, so the per-point result is
// bit-identical and order is preserved (the caller sums inertia sequentially).
fn assign_points(
    data: &[Vec<f64>],
    centroids_flat: &[f64],
    k: usize,
    d: usize,
) -> Vec<(usize, f64)> {
    let n = data.len();
    let work = (n as u64)
        .saturating_mul(k as u64)
        .saturating_mul(d.max(1) as u64);
    // High gate: nearest_centroid is only ~k·d cheap multiply-adds per point, so only
    // parallelise when the total assignment work clearly amortises thread spawn.
    let nthreads = if work < 1 << 21 || n < 64 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 32)
            .max(1)
    };
    if nthreads <= 1 {
        return data
            .iter()
            .map(|p| nearest_centroid(p, centroids_flat, k, d))
            .collect();
    }
    let chunk = n.div_ceil(nthreads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || {
                    data[i0..i1]
                        .iter()
                        .map(|p| nearest_centroid(p, centroids_flat, k, d))
                        .collect::<Vec<(usize, f64)>>()
                }))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("kmeans assign worker panicked"))
            .collect()
    })
}

fn nearest_centroid(point: &[f64], centroids_flat: &[f64], k: usize, d: usize) -> (usize, f64) {
    let probe = d.min(PREFILTER_DIMS);
    let mut seed = 0usize;
    let mut seed_partial = f64::INFINITY;
    for c in 0..k {
        let row = &centroids_flat[c * d..c * d + probe];
        let mut acc = 0.0;
        for (&pj, &cj) in point[..probe].iter().zip(row.iter()) {
            let diff = pj - cj;
            acc += diff * diff;
        }
        if acc < seed_partial {
            seed_partial = acc;
            seed = c;
        }
    }

    let mut best_c = seed;
    let mut min_sq = sq_dist(point, &centroids_flat[seed * d..seed * d + d]);
    for c in 0..k {
        let row = &centroids_flat[c * d..c * d + d];
        let sd = sq_dist_within(point, row, min_sq);
        if sd < min_sq || (sd == min_sq && c < best_c) {
            min_sq = sd;
            best_c = c;
        }
    }
    (best_c, min_sq)
}

fn dense_labels(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut mapping = std::collections::HashMap::new();
    let mut next = 0usize;
    let mut dense = Vec::with_capacity(labels.len());
    for &label in labels {
        let mapped = *mapping.entry(label).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        dense.push(mapped);
    }
    (dense, next)
}

fn validate_cluster_metric_data(
    data: &[Vec<f64>],
    labels: &[usize],
    context: &str,
) -> Result<(Vec<usize>, usize), ClusterError> {
    let n = data.len();
    if labels.len() != n {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} labels length {} must match sample count {n}",
            labels.len()
        )));
    }
    if n < 2 {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} requires at least 2 samples"
        )));
    }
    validate_feature_dimensions(data, context)?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} input must be finite"
        )));
    }
    let (dense, k) = dense_labels(labels);
    if k < 2 || k >= n {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} requires 2..n_samples-1 labels; got {k} labels for {n} samples"
        )));
    }
    Ok((dense, k))
}

fn validate_paired_labels(
    labels_true: &[usize],
    labels_pred: &[usize],
    context: &str,
) -> Result<usize, ClusterError> {
    let n = labels_true.len();
    if n == 0 {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} requires non-empty labels"
        )));
    }
    if n != labels_pred.len() {
        return Err(ClusterError::InvalidArgument(format!(
            "{context} label lengths differ: true={n}, pred={}",
            labels_pred.len()
        )));
    }
    Ok(n)
}

fn contingency_table(
    labels_true: &[usize],
    labels_pred: &[usize],
    context: &str,
) -> Result<Vec<Vec<usize>>, ClusterError> {
    let n = validate_paired_labels(labels_true, labels_pred, context)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let (true_dense, k1) = dense_labels(labels_true);
    let (pred_dense, k2) = dense_labels(labels_pred);
    let mut contingency = vec![vec![0usize; k2]; k1];
    for i in 0..n {
        contingency[true_dense[i]][pred_dense[i]] += 1;
    }
    Ok(contingency)
}

fn comb2_usize(x: usize) -> f64 {
    let xf = x as f64;
    xf * (xf - 1.0) / 2.0
}

/// Silhouette score: measure of cluster quality.
///
/// Returns mean silhouette coefficient over all samples.
/// Matches `sklearn.metrics.silhouette_score`.
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize]) -> Result<f64, ClusterError> {
    let n = data.len();
    let (labels, k) = validate_cluster_metric_data(data, labels, "silhouette_score")?;

    // Mean of the per-anchor silhouette coefficients. The anchor pass is parallel and
    // returns values in index order, and summing in index order matches the original
    // serial `total += s` accumulation, so the mean is bit-for-bit unchanged.
    let samples = silhouette_samples_bucket_pass(data, &labels, k);
    Ok(samples.iter().sum::<f64>() / n as f64)
}

/// Calinski-Harabasz index: ratio of between-cluster to within-cluster dispersion.
///
/// Higher is better. Matches `sklearn.metrics.calinski_harabasz_score`.
pub fn calinski_harabasz_score(data: &[Vec<f64>], labels: &[usize]) -> Result<f64, ClusterError> {
    let n = data.len();
    let (labels, k) = validate_cluster_metric_data(data, labels, "calinski_harabasz_score")?;
    let d = data[0].len();

    // Global centroid
    let mut global_centroid = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            global_centroid[j] += v;
        }
    }
    for v in &mut global_centroid {
        *v /= n as f64;
    }

    // Cluster centroids and counts
    let mut centroids = vec![vec![0.0; d]; k];
    let mut counts = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        counts[c] += 1;
        for (j, &v) in point.iter().enumerate() {
            centroids[c][j] += v;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for v in &mut centroids[c] {
                *v /= counts[c] as f64;
            }
        }
    }

    // Between-cluster dispersion
    let mut bg = 0.0;
    for c in 0..k {
        bg += counts[c] as f64 * sq_dist(&centroids[c], &global_centroid);
    }

    // Within-cluster dispersion
    let mut wg = 0.0;
    for (i, point) in data.iter().enumerate() {
        wg += sq_dist(point, &centroids[labels[i]]);
    }

    if wg == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok((bg / (k - 1) as f64) / (wg / (n - k) as f64))
}

/// Davies-Bouldin index: average similarity of clusters.
///
/// Lower is better. Matches `sklearn.metrics.davies_bouldin_score`.
pub fn davies_bouldin_score(data: &[Vec<f64>], labels: &[usize]) -> Result<f64, ClusterError> {
    let (labels, k) = validate_cluster_metric_data(data, labels, "davies_bouldin_score")?;
    let d = data[0].len();

    // Cluster centroids
    let mut centroids = vec![vec![0.0; d]; k];
    let mut counts = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        counts[c] += 1;
        for (j, &v) in point.iter().enumerate() {
            centroids[c][j] += v;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for v in &mut centroids[c] {
                *v /= counts[c] as f64;
            }
        }
    }

    // Average within-cluster distance for each cluster
    let mut s = vec![0.0; k];
    for (i, point) in data.iter().enumerate() {
        s[labels[i]] += sq_dist(point, &centroids[labels[i]]).sqrt();
    }
    for c in 0..k {
        if counts[c] > 0 {
            s[c] /= counts[c] as f64;
        }
    }

    // Davies-Bouldin index
    let mut db = 0.0;
    for i in 0..k {
        let mut max_r = 0.0f64;
        for j in 0..k {
            if i != j {
                let d_ij = sq_dist(&centroids[i], &centroids[j]).sqrt();
                if d_ij > 0.0 {
                    max_r = max_r.max((s[i] + s[j]) / d_ij);
                }
            }
        }
        db += max_r;
    }

    Ok(db / k as f64)
}

/// Adjusted Rand Index: similarity between two clusterings, adjusted for chance.
///
/// Matches `sklearn.metrics.adjusted_rand_score`.
pub fn adjusted_rand_score(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<f64, ClusterError> {
    let n = validate_paired_labels(labels_true, labels_pred, "adjusted_rand_score")?;
    if n == 1 {
        return Ok(1.0);
    }

    let contingency = contingency_table(labels_true, labels_pred, "adjusted_rand_score")?;
    let k2 = contingency.first().map_or(0, Vec::len);

    // Row and column sums
    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    let sum_comb_nij: f64 = contingency
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| comb2_usize(v))
        .sum();
    let sum_comb_a: f64 = row_sums.iter().map(|&v| comb2_usize(v)).sum();
    let sum_comb_b: f64 = col_sums.iter().map(|&v| comb2_usize(v)).sum();
    let comb_n = comb2_usize(n);

    let expected = sum_comb_a * sum_comb_b / comb_n;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;

    if (max_index - expected).abs() < 1e-15 {
        return Ok(if (sum_comb_nij - expected).abs() < 1e-15 {
            1.0
        } else {
            0.0
        });
    }

    Ok((sum_comb_nij - expected) / (max_index - expected))
}

/// Normalized Mutual Information between two clusterings.
///
/// Matches `sklearn.metrics.normalized_mutual_info_score`.
pub fn normalized_mutual_info(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<f64, ClusterError> {
    let n = validate_paired_labels(labels_true, labels_pred, "normalized_mutual_info")?;
    if n == 1 {
        return Ok(1.0);
    }
    let contingency = contingency_table(labels_true, labels_pred, "normalized_mutual_info")?;
    let k1 = contingency.len();
    let k2 = contingency.first().map_or(0, Vec::len);
    let nf = n as f64;

    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // Mutual information
    let mut mi = 0.0;
    for i in 0..k1 {
        for j in 0..k2 {
            if contingency[i][j] > 0 && row_sums[i] > 0 && col_sums[j] > 0 {
                let pij = contingency[i][j] as f64 / nf;
                let pi = row_sums[i] as f64 / nf;
                let pj = col_sums[j] as f64 / nf;
                mi += pij * (pij / (pi * pj)).ln();
            }
        }
    }

    // Entropies
    let h1: f64 = row_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let h2: f64 = col_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();

    let denom = ((h1 + h2) / 2.0).max(1e-15);
    Ok((mi / denom).clamp(0.0, 1.0))
}

/// Homogeneity score: each cluster contains only members of a single class.
///
/// Matches `sklearn.metrics.homogeneity_score`.
pub fn homogeneity_score(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<f64, ClusterError> {
    let n = validate_paired_labels(labels_true, labels_pred, "homogeneity_score")?;
    if n == 1 {
        return Ok(1.0);
    }
    let contingency = contingency_table(labels_true, labels_pred, "homogeneity_score")?;
    let k1 = contingency.len();
    let k2 = contingency.first().map_or(0, Vec::len);
    let nf = n as f64;

    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // H(C|K) = conditional entropy of true labels given predicted
    let mut hck = 0.0;
    for j in 0..k2 {
        if col_sums[j] == 0 {
            continue;
        }
        for row in contingency.iter().take(k1) {
            if row[j] > 0 {
                let p = row[j] as f64 / nf;
                let pj = col_sums[j] as f64 / nf;
                hck -= p * (p / pj).ln();
            }
        }
    }

    // H(C) = entropy of true labels
    let hc: f64 = row_sums
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| {
            let p = s as f64 / nf;
            -p * p.ln()
        })
        .sum();

    if hc < 1e-15 {
        return Ok(1.0);
    }
    Ok(1.0 - hck / hc)
}

/// Completeness score: all members of a class are assigned to the same cluster.
///
/// Matches `sklearn.metrics.completeness_score`.
pub fn completeness_score(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<f64, ClusterError> {
    // Completeness = homogeneity with args swapped
    homogeneity_score(labels_pred, labels_true)
}

/// V-measure: harmonic mean of homogeneity and completeness.
///
/// Matches `sklearn.metrics.v_measure_score`.
pub fn v_measure_score(labels_true: &[usize], labels_pred: &[usize]) -> Result<f64, ClusterError> {
    let h = homogeneity_score(labels_true, labels_pred)?;
    let c = completeness_score(labels_true, labels_pred)?;
    if h + c < 1e-15 {
        return Ok(0.0);
    }
    Ok(2.0 * h * c / (h + c))
}

/// Fowlkes-Mallows index: geometric mean of precision and recall.
///
/// Matches `sklearn.metrics.fowlkes_mallows_score`.
pub fn fowlkes_mallows_score(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<f64, ClusterError> {
    let n = validate_paired_labels(labels_true, labels_pred, "fowlkes_mallows_score")?;
    if n == 1 {
        return Ok(1.0);
    }

    // Pair counts derived from the true-vs-pred contingency table in O(n + k^2)
    // instead of enumerating all O(n^2) pairs. With C(m) = m(m-1)/2:
    //   tp        = sum over table cells of C(cell)   (pairs together in both)
    //   tp + fp   = sum over column sums of C(col)     (pairs sharing a predicted cluster)
    //   tp + fn_  = sum over row sums of C(row)        (pairs sharing a true cluster)
    // These are exact integer counts (sums stay < 2^53), so precision, recall, and the
    // index are byte-identical to the pair loop.
    let contingency = contingency_table(labels_true, labels_pred, "fowlkes_mallows_score")?;
    let k2 = contingency.first().map_or(0, Vec::len);
    let row_sums: Vec<usize> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..k2)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    let tp: f64 = contingency
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| comb2_usize(v))
        .sum();
    if tp == 0.0 {
        return Ok(0.0);
    }
    let tp_fp: f64 = col_sums.iter().map(|&v| comb2_usize(v)).sum();
    let tp_fn: f64 = row_sums.iter().map(|&v| comb2_usize(v)).sum();

    let precision = if tp_fp > 0.0 { tp / tp_fp } else { 0.0 };
    let recall = if tp_fn > 0.0 { tp / tp_fn } else { 0.0 };
    Ok((precision * recall).sqrt())
}

/// Elbow method helper: compute within-cluster sum of squares for k=1..max_k.
///
/// Returns a vector of inertia values useful for the elbow method.
/// Returns empty vec if data is empty, has non-finite values, or max_k is 0.
pub fn elbow_inertias(data: &[Vec<f64>], max_k: usize, seed: u64) -> Vec<f64> {
    if data.is_empty() || max_k == 0 {
        return vec![];
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return vec![];
    }
    (1..=max_k.min(data.len()))
        .map(|k| {
            kmeans(data, k, 50, seed.wrapping_add(k as u64))
                .map(|r| r.inertia)
                .unwrap_or(f64::INFINITY)
        })
        .collect()
}

/// Mean-shift clustering: find cluster centers via kernel density gradient ascent.
///
/// Matches `sklearn.cluster.MeanShift`.
pub fn mean_shift(
    data: &[Vec<f64>],
    bandwidth: f64,
    max_iter: usize,
) -> Result<(Vec<Vec<f64>>, Vec<usize>), ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    let d = validate_feature_dimensions(data, "mean_shift")?;
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(ClusterError::InvalidArgument(
            "bandwidth must be finite and positive".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "mean_shift input must be finite".to_string(),
        ));
    }
    let bw2 = bandwidth * bandwidth;

    // Start each point as its own center candidate
    let mut centers: Vec<Vec<f64>> = data.to_vec();

    for _ in 0..max_iter {
        let mut shifted = false;
        for center in &mut centers {
            let mut new_center = vec![0.0; d];
            let mut total_weight = 0.0;

            for point in data {
                let dist2: f64 = center
                    .iter()
                    .zip(point.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let weight = (-dist2 / (2.0 * bw2)).exp();
                total_weight += weight;
                for (j, &pj) in point.iter().enumerate() {
                    new_center[j] += weight * pj;
                }
            }

            if total_weight > 0.0 {
                for v in &mut new_center {
                    *v /= total_weight;
                }
            }

            let shift: f64 = center
                .iter()
                .zip(new_center.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if shift > 1e-6 {
                shifted = true;
            }
            *center = new_center;
        }

        if !shifted {
            break;
        }
    }

    // Merge nearby centers
    let merge_threshold = bandwidth / 2.0;
    let mut unique_centers: Vec<Vec<f64>> = Vec::new();
    let mut center_map = vec![0usize; n];

    for (i, center) in centers.iter().enumerate() {
        let mut found = false;
        for (j, uc) in unique_centers.iter().enumerate() {
            let dist: f64 = center
                .iter()
                .zip(uc.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < merge_threshold {
                center_map[i] = j;
                found = true;
                break;
            }
        }
        if !found {
            center_map[i] = unique_centers.len();
            unique_centers.push(center.clone());
        }
    }

    // Assign each original point to nearest unique center
    let mut labels = vec![0usize; n];
    for (i, point) in data.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut best = 0;
        for (j, uc) in unique_centers.iter().enumerate() {
            let dist: f64 = point
                .iter()
                .zip(uc.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            if dist < min_dist {
                min_dist = dist;
                best = j;
            }
        }
        labels[i] = best;
    }

    Ok((unique_centers, labels))
}

/// Agglomerative clustering with precomputed distance matrix.
///
/// Matches `scipy.cluster.hierarchy.linkage` with precomputed distances.
pub fn linkage_from_distances(
    condensed_dist: &[f64],
    n: usize,
    method: LinkageMethod,
) -> Result<Vec<[f64; 4]>, ClusterError> {
    if n < 2 {
        return Err(ClusterError::InvalidArgument(
            "need at least 2 observations".to_string(),
        ));
    }
    let expected_len = n * (n - 1) / 2;
    if condensed_dist.len() != expected_len {
        return Err(ClusterError::InvalidArgument(format!(
            "condensed_dist length {} does not match expected {} for n={}",
            condensed_dist.len(),
            expected_len,
            n
        )));
    }
    if condensed_dist.iter().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "linkage_from_distances input must be finite".to_string(),
        ));
    }

    // Build full distance matrix from condensed form
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut idx = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        dist[i][i] = 0.0;
        for j in i + 1..n {
            if idx < condensed_dist.len() {
                dist[i][j] = condensed_dist[idx];
                dist[j][i] = condensed_dist[idx];
                idx += 1;
            }
        }
    }

    // Run the shared O(n^2) nearest-neighbour-array agglomeration over the
    // distance matrix (byte-identical to the old pairwise-scan loop).
    let total = 2 * n - 1;
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];
    for i in 0..n {
        inter_dist[i][..n].copy_from_slice(&dist[i][..n]);
    }

    Ok(agglomerate_nnarray(n, inter_dist, method))
}

/// Maximal cliques in a proximity graph (for small graphs).
///
/// Given data and an epsilon threshold, find all maximal cliques.
pub fn proximity_cliques(data: &[Vec<f64>], eps: f64) -> Vec<Vec<usize>> {
    let n = data.len();
    if n == 0 || !eps.is_finite() || eps < 0.0 {
        return vec![];
    }
    let Ok(_d) = validate_feature_dimensions(data, "proximity_cliques") else {
        return vec![];
    };
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return vec![];
    }
    let eps2 = eps * eps;

    // Build adjacency
    let mut adj = vec![vec![]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            if d <= eps2 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // Bron-Kerbosch for maximal cliques
    let mut cliques = Vec::new();
    bron_kerbosch(
        &adj,
        &mut vec![],
        &mut (0..n).collect(),
        &mut vec![],
        &mut cliques,
    );
    cliques
}

fn bron_kerbosch(
    adj: &[Vec<usize>],
    r: &mut Vec<usize>,
    p: &mut Vec<usize>,
    x: &mut Vec<usize>,
    cliques: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if !r.is_empty() {
            cliques.push(r.clone());
        }
        return;
    }

    let p_copy = p.clone();
    for &v in &p_copy {
        r.push(v);
        let neighbors: std::collections::HashSet<usize> = adj[v].iter().cloned().collect();
        let mut new_p: Vec<usize> = p
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        let mut new_x: Vec<usize> = x
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        bron_kerbosch(adj, r, &mut new_p, &mut new_x, cliques);
        r.pop();
        p.retain(|&u| u != v);
        x.push(v);
    }
}

/// Compute silhouette coefficients for each sample.
///
/// Matches `sklearn.metrics.silhouette_samples`.
pub fn silhouette_samples(data: &[Vec<f64>], labels: &[usize]) -> Result<Vec<f64>, ClusterError> {
    let (labels, k) = validate_cluster_metric_data(data, labels, "silhouette_samples")?;
    // The per-anchor bucket pass is bit-identical to the former upper-triangle matrix
    // accumulation (same dist values and same per-cluster summation order) and is
    // parallel across anchors, so it supersedes the n×k matrix path for every size.
    Ok(silhouette_samples_bucket_pass(data, &labels, k))
}

/// Silhouette coefficient per anchor point. Each point's `s(i)` depends only on its
/// distances to every other point bucketed by cluster, so the anchors are independent
/// and the loop parallelizes byte-identically. The per-anchor bucket sum accumulates in
/// increasing `j` order — identical to the upper-triangle matrix accumulation
/// (dist(i,j) == dist(j,i) bit-for-bit, and each `cluster_sum[i][c]` there also fills in
/// increasing source order) — so this is bit-identical to both the matrix and the old
/// serial bucket paths.
fn silhouette_samples_bucket_pass(data: &[Vec<f64>], labels: &[usize], k: usize) -> Vec<f64> {
    let n = data.len();
    // Compute anchors [i0, i1) into `out`, reusing two length-k scratch buffers across
    // anchors (no per-anchor allocation).
    let run = |i0: usize, i1: usize, out: &mut Vec<f64>| {
        let mut cluster_sum = vec![0.0_f64; k];
        let mut cluster_count = vec![0usize; k];
        for i in i0..i1 {
            let li = labels[i];
            cluster_sum.iter_mut().for_each(|v| *v = 0.0);
            cluster_count.iter_mut().for_each(|v| *v = 0);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let lj = labels[j];
                cluster_sum[lj] += sq_dist(&data[i], &data[j]).sqrt();
                cluster_count[lj] += 1;
            }
            let a = if cluster_count[li] > 0 {
                cluster_sum[li] / cluster_count[li] as f64
            } else {
                0.0
            };
            let mut b = f64::INFINITY;
            for c in 0..k {
                if c == li || cluster_count[c] == 0 {
                    continue;
                }
                let mean_c = cluster_sum[c] / cluster_count[c] as f64;
                if mean_c < b {
                    b = mean_c;
                }
            }
            out.push(if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            });
        }
    };

    // Each anchor is an O(n·d) reduction; parallelize across anchors for large n²·d.
    let d = data.first().map_or(0, Vec::len);
    let work = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(d.max(1) as u64);
    let nthreads = if work < 1 << 21 || n < 64 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 32)
            .max(1)
    };
    if nthreads <= 1 {
        let mut out = Vec::with_capacity(n);
        run(0, n, &mut out);
        return out;
    }
    let chunk = n.div_ceil(nthreads);
    let run = &run;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || {
                    let mut out = Vec::with_capacity(i1 - i0);
                    run(i0, i1, &mut out);
                    out
                }))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("silhouette worker panicked"))
            .collect()
    })
}

/// Gap statistic: compare within-cluster dispersion to reference.
///
/// Returns gap values for k=1..max_k.
pub fn gap_statistic(data: &[Vec<f64>], max_k: usize, n_ref: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    if n == 0 || n_ref == 0 || max_k == 0 {
        return vec![];
    }
    let Ok(d) = validate_feature_dimensions(data, "gap_statistic") else {
        return vec![];
    };
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return vec![];
    }

    // Find data bounds
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for p in data {
        for (j, &v) in p.iter().enumerate() {
            mins[j] = mins[j].min(v);
            maxs[j] = maxs[j].max(v);
        }
    }

    let mut gaps = Vec::with_capacity(max_k);

    for k in 1..=max_k.min(n) {
        // Within-cluster dispersion for real data
        let real_result = kmeans(data, k, 50, seed);
        let log_wk = real_result
            .map(|r| r.inertia.max(1e-30).ln())
            .unwrap_or(0.0);

        // Average over reference datasets
        let mut ref_log_wks = Vec::with_capacity(n_ref);
        for r in 0..n_ref {
            let ref_seed = seed.wrapping_add(1000 * r as u64 + k as u64);
            let mut rng = ref_seed;

            // Generate uniform reference data
            let ref_data: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..d)
                        .map(|j| {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
                            mins[j] + u * (maxs[j] - mins[j])
                        })
                        .collect()
                })
                .collect();

            let ref_result = kmeans(&ref_data, k, 30, ref_seed);
            let ref_wk = ref_result.map(|r| r.inertia.max(1e-30).ln()).unwrap_or(0.0);
            ref_log_wks.push(ref_wk);
        }

        let mean_ref = ref_log_wks.iter().sum::<f64>() / n_ref as f64;
        gaps.push(mean_ref - log_wk);
    }

    gaps
}

/// K-medoids (PAM) clustering.
///
/// Similar to K-means but uses actual data points as centers.
pub fn kmedoids(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<KMeansResult, ClusterError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusterError::EmptyData);
    }
    let d = validate_feature_dimensions(data, "kmedoids")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "kmedoids input must be finite".to_string(),
        ));
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "k={k} must be in [1, n={n}]"
        )));
    }

    // Initialize medoids randomly
    let mut rng = seed;
    let mut medoid_indices: Vec<usize> = Vec::with_capacity(k);
    let mut used = vec![false; n];
    for _ in 0..k {
        loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng >> 33) as usize % n;
            if !used[idx] {
                used[idx] = true;
                medoid_indices.push(idx);
                break;
            }
        }
    }

    // The assignment scan re-walks every observation against every medoid each
    // iteration, and the medoid-update step gathers scattered cluster members for
    // its M×M distance matrix — both index `Vec<Vec<f64>>` rows (`data[i]`,
    // `data[med]`, `data[members[i]]`), paying a heap-pointer load + cache miss per
    // access. Pack the ragged rows into one contiguous `n × d` buffer once (the
    // observation set is loop-invariant) so every inner distance streams a
    // contiguous slice. Same lever as the vq/dbscan assignment paths; bit-identical
    // because `sq_dist` sums the same terms in the same index order regardless of
    // where the slice lives. See
    // `tests/artifacts/perf/2026-06-03-cluster-kmedoids-flatten/` for the sha256 proof.
    let flat = flatten_points(data, d);
    let row = |idx: usize| -> &[f64] { &flat[idx * d..idx * d + d] };

    let mut labels = vec![0usize; n];
    let mut actual_iter = 0;

    // Reused across iterations: contiguous k×d buffer of the current medoid rows
    // so the nearest-medoid scan streams + prefetches and the prefilter/abandonment
    // bound (see `nearest_centroid`) rejects most medoids after a few dimensions.
    let mut medoids_flat = vec![0.0_f64; k * d];

    for iter in 0..max_iter {
        actual_iter = iter + 1;

        // Pack the current medoid rows into the contiguous buffer.
        for (c, &med) in medoid_indices.iter().enumerate() {
            medoids_flat[c * d..c * d + d].copy_from_slice(row(med));
        }

        // Assign each observation to its nearest medoid. `nearest_centroid` is a
        // bit-identical replacement for the strict-`<` argmin: same lowest-index
        // tie-break, same fully-summed minimum (the winner is never abandoned).
        for i in 0..n {
            let (best_c, _) = nearest_centroid(&flat[i * d..i * d + d], &medoids_flat, k, d);
            labels[i] = best_c;
        }

        // Update medoids: for each cluster, find the point that
        // minimizes total within-cluster distance. Resolves
        // [frankenscipy-ebx8l]: precompute the M×M distance matrix
        // once per cluster (M(M-1)/2 sqrts) instead of recomputing
        // M² distances inside the candidate loop.
        let mut changed = false;
        for (c, medoid_index) in medoid_indices.iter_mut().enumerate().take(k) {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let m = members.len();

            // Gather this cluster's member rows into one contiguous m×d buffer so
            // the M(M-1)/2 distance evaluations stream sequentially instead of
            // chasing `members[i]` scattered indices back into `data`.
            let mut member_flat = Vec::with_capacity(m * d);
            for &mi in &members {
                member_flat.extend_from_slice(row(mi));
            }

            // Symmetric M×M intra-cluster distance matrix over the contiguous buffer.
            let mut dmat = vec![vec![0.0_f64; m]; m];
            for i in 0..m {
                let mi = &member_flat[i * d..i * d + d];
                for j in (i + 1)..m {
                    let dist = sq_dist(mi, &member_flat[j * d..j * d + d]).sqrt();
                    dmat[i][j] = dist;
                    dmat[j][i] = dist;
                }
            }

            // For each candidate row, sum the distances to all other
            // members. The minimizing row is the new medoid.
            let mut best_local = 0usize;
            let mut best_cost = dmat[0].iter().sum::<f64>();
            for (i, row) in dmat.iter().enumerate().skip(1) {
                let cost: f64 = row.iter().sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_local = i;
                }
            }
            let best_med = members[best_local];

            if best_med != *medoid_index {
                *medoid_index = best_med;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Compute final inertia
    let inertia: f64 = (0..n)
        .map(|i| sq_dist(row(i), row(medoid_indices[labels[i]])))
        .sum();

    let centroids: Vec<Vec<f64>> = medoid_indices.iter().map(|&i| data[i].clone()).collect();

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: actual_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let result = kmeans(&data, 2, 100, 42).unwrap();
        assert_eq!(result.labels.len(), 6);
        // First 3 should be in one cluster, last 3 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![0.9, 1.9]];
        let result = kmeans(&data, 1, 10, 42).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn kmeans_rejects_ragged_input() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        let err = kmeans(&data, 1, 10, 42).expect_err("ragged input");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn mini_batch_kmeans_rejects_ragged_input() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        let err = mini_batch_kmeans(&data, 1, 5, 1, 7).expect_err("ragged input");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn vq_metamorphic_centroids_self_assign_with_zero_distance() {
        // /testing-metamorphic: when data is the centroid set itself,
        // vq returns labels = [0, 1, 2, ...] and distances = [0, 0, ...].
        // Each centroid is its own nearest neighbor at zero distance.
        let centroids = vec![
            vec![0.0_f64, 0.0],
            vec![10.0, 5.0],
            vec![-3.0, 7.5],
            vec![1e6, 1e6],
        ];
        let (labels, dists) = vq(&centroids, &centroids).unwrap();
        for (i, &lbl) in labels.iter().enumerate() {
            assert_eq!(lbl, i, "centroid {i} should self-assign");
            assert!(
                dists[i].abs() < 1e-12,
                "centroid {i} self-distance = {}, expected 0",
                dists[i]
            );
        }
    }

    #[test]
    fn vq_metamorphic_translation_invariance() {
        // /testing-metamorphic: translating data and centroids by the
        // same vector preserves the (labels, distances) output exactly.
        // Catches any regression that introduces an absolute-position
        // dependence into the distance metric.
        let data = vec![vec![1.0_f64, 2.0], vec![5.0, 5.0], vec![-1.0, 0.5]];
        let centroids = vec![vec![0.0_f64, 0.0], vec![6.0, 6.0]];
        let (l0, d0) = vq(&data, &centroids).unwrap();
        let shift = [100.0_f64, -50.0];
        let shifted_data: Vec<Vec<f64>> = data
            .iter()
            .map(|p| vec![p[0] + shift[0], p[1] + shift[1]])
            .collect();
        let shifted_centroids: Vec<Vec<f64>> = centroids
            .iter()
            .map(|p| vec![p[0] + shift[0], p[1] + shift[1]])
            .collect();
        let (l1, d1) = vq(&shifted_data, &shifted_centroids).unwrap();
        assert_eq!(l0, l1, "labels must be translation-invariant");
        for (a, b) in d0.iter().zip(d1.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "distances must be translation-invariant: {a} vs {b}"
            );
        }
    }

    #[test]
    fn vq_assigns_nearest() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let data = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let (labels, dists) = vq(&data, &centroids).unwrap();
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
        assert!(dists[0] < 2.0);
    }

    #[test]
    fn vq_distances_are_true_euclidean() {
        // /profiling-software-performance regression for
        // frankenscipy-dnunq: vq used to call sqrt() inside the inner
        // k-loop; the new version compares squared distances and only
        // sqrt's the winning min. Verify the output dists are still the
        // *true Euclidean* distance to the assigned centroid, not the
        // squared distance.
        //
        //   centroid 0 = (0, 0), centroid 1 = (3, 4)
        //   point     = (3, 0)   → centroid 0 (sq 9 < 16) → dist √9 = 3.0
        //   point     = (3, 5)   → centroid 1 (sq 1 < 34) → dist √1 = 1.0
        //   point     = (0, 4)   → centroid 1 (sq 9 < 16) → dist √9 = 3.0
        let centroids = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let data = vec![vec![3.0, 0.0], vec![3.0, 5.0], vec![0.0, 4.0]];
        let (labels, dists) = vq(&data, &centroids).unwrap();
        assert_eq!(labels, vec![0, 1, 1]);
        let expected_dists = [3.0_f64, 1.0, 3.0];
        for (i, (&got, &exp)) in dists.iter().zip(expected_dists.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "vq dists[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn whiten_metamorphic_positive_scaling_invariant() {
        // /testing-metamorphic: whiten(α·x) ≡ whiten(x) for α > 0.
        // The std normalization cancels the scale factor: x/std(x) and
        // (αx)/(α·std(x)) are bit-equal up to f64 arithmetic.
        let data = vec![
            vec![1.0_f64, 10.0],
            vec![2.0, 30.0],
            vec![3.0, 60.0],
            vec![4.0, 50.0],
        ];
        let baseline = whiten(&data).unwrap();
        for &alpha in &[0.5_f64, 2.0, 7.3, 1000.0] {
            let scaled: Vec<Vec<f64>> = data
                .iter()
                .map(|row| row.iter().map(|&v| alpha * v).collect())
                .collect();
            let scaled_whitened = whiten(&scaled).unwrap();
            for (i, (b_row, s_row)) in baseline.iter().zip(scaled_whitened.iter()).enumerate() {
                for (j, (&b, &s)) in b_row.iter().zip(s_row.iter()).enumerate() {
                    assert!(
                        (b - s).abs() < 1e-12,
                        "α={alpha}: whiten[{i}][{j}] differs ({b} vs {s})"
                    );
                }
            }
        }
    }

    #[test]
    fn whiten_normalizes_std() {
        let data = vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]];
        let whitened = whiten(&data).unwrap();
        // After whitening, std of each column should be ~1
        let n = whitened.len() as f64;
        for col in 0..2 {
            let mean: f64 = whitened.iter().map(|r| r[col]).sum::<f64>() / n;
            let var: f64 = whitened
                .iter()
                .map(|r| (r[col] - mean).powi(2))
                .sum::<f64>()
                / n;
            assert!(
                (var.sqrt() - 1.0).abs() < 0.01,
                "column {col} std = {}",
                var.sqrt()
            );
        }
    }

    #[test]
    fn linkage_metamorphic_n_points_yield_n_minus_1_merges() {
        // /testing-metamorphic: scipy.cluster.hierarchy.linkage on n
        // observations always returns an (n−1)×4 linkage matrix —
        // each merge reduces the cluster count by 1. Pin across
        // multiple n and methods.
        for &n in &[2_usize, 3, 5, 8, 12] {
            // Build n distinct 1-D points.
            let data: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            for method in [
                LinkageMethod::Single,
                LinkageMethod::Complete,
                LinkageMethod::Average,
                LinkageMethod::Ward,
            ] {
                let z = linkage(&data, method).unwrap();
                assert_eq!(
                    z.len(),
                    n - 1,
                    "method={method:?}, n={n}: linkage rows = {}, expected n-1",
                    z.len()
                );
                // Every merge must produce a positive cluster size.
                for row in &z {
                    assert!(
                        row[3] >= 2.0,
                        "merge cluster count {} should be ≥ 2",
                        row[3]
                    );
                }
            }
        }
    }

    #[test]
    fn linkage_single() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert_eq!(z.len(), 2);
        // First merge: 0 and 1 (distance 1)
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn linkage_from_distances_skips_inactive_clusters_after_merge() {
        let condensed = vec![10.0, 4.0, 9.0, 8.0, 8.0, 1.0];
        let z = linkage_from_distances(&condensed, 4, LinkageMethod::Complete)
            .expect("complete linkage from condensed distances");
        assert_eq!(z.len(), 3);

        assert_eq!(z[0][0], 2.0);
        assert_eq!(z[0][1], 3.0);
        assert!((z[0][2] - 1.0).abs() < 1e-12);

        assert_eq!(z[1][0], 1.0);
        assert_eq!(z[1][1], 4.0);
        assert!((z[1][2] - 8.0).abs() < 1e-12);

        assert_eq!(z[2][0], 0.0);
        assert_eq!(z[2][1], 5.0);
        assert!((z[2][2] - 10.0).abs() < 1e-12);
    }

    /// Regression: verify that `linkage` never picks a merged (inactive) cluster
    /// as a merge candidate. With 5 collinear points at [0, 1, 3, 6, 10] using
    /// single linkage, each step must merge the closest *active* pair. If inactive
    /// clusters leaked into the nearest-pair search, merge distances would be wrong.
    ///
    /// Expected SciPy-equivalent merge sequence (single linkage):
    ///   step 0: (0,1) dist=1, size=2          -> cluster 5
    ///   step 1: (2,5) dist=2, size=3          -> cluster 6  (cluster 5 active, not 0 or 1)
    ///   step 2: (3,6) dist=3, size=4          -> cluster 7
    ///   step 3: (4,7) dist=4, size=5          -> cluster 8
    #[test]
    fn linkage_single_inactive_cluster_regression() {
        let data = vec![vec![0.0], vec![1.0], vec![3.0], vec![6.0], vec![10.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert_eq!(z.len(), 4);

        // Step 0: merge points 0 and 1 (distance 1)
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 1.0).abs() < 1e-12, "step 0 dist = {}", z[0][2]);
        assert_eq!(z[0][3], 2.0);

        // Step 1: merge point 2 with new cluster 5={0,1} (distance 2)
        assert_eq!(z[1][0], 2.0);
        assert_eq!(z[1][1], 5.0);
        assert!((z[1][2] - 2.0).abs() < 1e-12, "step 1 dist = {}", z[1][2]);
        assert_eq!(z[1][3], 3.0);

        // Step 2: merge point 3 with cluster 6={0,1,2} (distance 3)
        assert_eq!(z[2][0], 3.0);
        assert_eq!(z[2][1], 6.0);
        assert!((z[2][2] - 3.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);

        // Step 3: merge point 4 with cluster 7={0,1,2,3} (distance 4)
        assert_eq!(z[3][0], 4.0);
        assert_eq!(z[3][1], 7.0);
        assert!((z[3][2] - 4.0).abs() < 1e-12, "step 3 dist = {}", z[3][2]);
        assert_eq!(z[3][3], 5.0);
    }

    /// Regression: complete linkage with 5 points where merged-cluster distances
    /// differ significantly from original point distances. If inactive clusters
    /// pollute the search, the wrong pair gets picked.
    ///
    /// Points: [0, 1, 10, 11, 50]
    /// Complete linkage uses max distance, so merged cluster distances grow.
    #[test]
    fn linkage_complete_inactive_cluster_regression() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0], vec![50.0]];
        let z = linkage(&data, LinkageMethod::Complete).unwrap();
        assert_eq!(z.len(), 4);

        // Step 0: closest pair is (0,1) dist=1 or (2,3) dist=1; either valid
        assert!((z[0][2] - 1.0).abs() < 1e-12, "step 0 dist = {}", z[0][2]);
        assert_eq!(z[0][3], 2.0);

        // Step 1: next closest active pair, also dist=1
        assert!((z[1][2] - 1.0).abs() < 1e-12, "step 1 dist = {}", z[1][2]);
        assert_eq!(z[1][3], 2.0);

        // Step 2: merge the two size-2 clusters; complete linkage dist = max(11-0)=11
        assert!((z[2][2] - 11.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);

        // Step 3: merge point 4 with the size-4 cluster; complete dist = max(50-0)=50
        assert!((z[3][2] - 50.0).abs() < 1e-12, "step 3 dist = {}", z[3][2]);
        assert_eq!(z[3][3], 5.0);
    }

    /// Regression: `linkage_from_distances` with average method; ensures
    /// that after each merge step, only active clusters participate in
    /// distance updates and nearest-pair selection.
    #[test]
    fn linkage_from_distances_average_inactive_regression() {
        // 4 points with condensed distances: d(0,1)=2, d(0,2)=6, d(0,3)=10,
        //                                    d(1,2)=4, d(1,3)=8, d(2,3)=4
        let condensed = vec![2.0, 6.0, 10.0, 4.0, 8.0, 4.0];
        let z = linkage_from_distances(&condensed, 4, LinkageMethod::Average).unwrap();
        assert_eq!(z.len(), 3);

        // Step 0: closest pair is (0,1) dist=2
        assert_eq!(z[0][0], 0.0);
        assert_eq!(z[0][1], 1.0);
        assert!((z[0][2] - 2.0).abs() < 1e-12);
        assert_eq!(z[0][3], 2.0);

        // Step 1: cluster 4={0,1}. Average distances:
        //   d(4, 2) = (1*6 + 1*4)/2 = 5.0
        //   d(4, 3) = (1*10 + 1*8)/2 = 9.0
        //   d(2, 3) = 4.0  <- minimum among active pairs
        assert_eq!(z[1][0], 2.0);
        assert_eq!(z[1][1], 3.0);
        assert!((z[1][2] - 4.0).abs() < 1e-12);
        assert_eq!(z[1][3], 2.0);

        // Step 2: merge cluster 4={0,1} with cluster 5={2,3}
        //   d(4, 5) = average of (d(0,2), d(0,3), d(1,2), d(1,3)) via Lance-Williams:
        //   = (2*5.0 + 2*9.0) / (2+2) = 7.0
        // But Lance-Williams UPGMA: d(4,5) = (n4*d(4,2_merged) + n4_already?
        // Actually: d(new_cluster_4, k) was computed as average above.
        // d(4, 5={2,3}) = average(d(4,2), d(4,3)) weighted = (2*5 + 2*9)/4 = 7.0
        assert_eq!(z[2][0], 4.0);
        assert_eq!(z[2][1], 5.0);
        assert!((z[2][2] - 7.0).abs() < 1e-12, "step 2 dist = {}", z[2][2]);
        assert_eq!(z[2][3], 4.0);
    }

    #[test]
    fn fcluster_two_groups() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let z = linkage(&data, LinkageMethod::Complete).unwrap();
        let labels = fcluster(&z, 2).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn fcluster_unionfind_matches_relabel_reference() {
        // The union-find maxclust cut must reproduce the original O(n^2)
        // per-merge relabel exactly (same partition, same min-leaf labels, same
        // 1..k renumbering) across linkage methods, sizes, and cut counts.
        fn relabel(z: &[[f64; 4]], max_clusters: usize) -> Vec<usize> {
            let n = z.len() + 1;
            if max_clusters >= n || max_clusters == 0 {
                return (1..=n).collect();
            }
            let mut cluster_of = vec![0usize; 2 * n - 1];
            for (i, c) in cluster_of.iter_mut().enumerate().take(n) {
                *c = i;
            }
            for (step, row) in z.iter().enumerate().take(n - max_clusters) {
                let new_id = n + step;
                let (ci, cj) = (row[0] as usize, row[1] as usize);
                let label = cluster_of[ci].min(cluster_of[cj]);
                let (oi, oj) = (cluster_of[ci], cluster_of[cj]);
                for v in cluster_of.iter_mut().take(new_id + 1) {
                    if *v == oi || *v == oj {
                        *v = label;
                    }
                }
                cluster_of[new_id] = label;
            }
            let leaf: Vec<usize> = cluster_of[..n].to_vec();
            let mut u = leaf.clone();
            u.sort_unstable();
            u.dedup();
            leaf.iter()
                .map(|&l| u.binary_search(&l).unwrap() + 1)
                .collect()
        }
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 10.0
        };
        let methods = [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Ward,
            LinkageMethod::Centroid,
            LinkageMethod::Median,
        ];
        for &n in &[2usize, 5, 30, 90] {
            for &m in &methods {
                let data: Vec<Vec<f64>> = (0..n).map(|_| vec![next(), next()]).collect();
                let Ok(z) = linkage(&data, m) else { continue };
                for k in 1..=n {
                    assert_eq!(fcluster(&z, k).unwrap(), relabel(&z, k), "n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn fcluster_singleton_fast_path_is_one_based() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let z = linkage(&data, LinkageMethod::Complete).unwrap();

        assert_eq!(fcluster(&z, 0).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(fcluster(&z, 4).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(fcluster(&z, 5).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn fcluster_rejects_invalid_linkage_indices() {
        let invalid = [[0.0, 99.0, 1.0, 2.0]];
        assert!(matches!(
            fcluster(&invalid, 1),
            Err(ClusterError::InvalidArgument(msg)) if msg == "invalid linkage matrix"
        ));
    }

    #[test]
    fn dbscan_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
        ];
        let result = dbscan(&data, 0.5, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn dbscan_noise_detection() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![100.0, 100.0], // isolated noise point
        ];
        let result = dbscan(&data, 0.5, 2).unwrap();
        assert_eq!(result.labels[3], -1); // should be noise
    }

    #[test]
    fn dbscan_rejects_zero_eps() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = dbscan(&data, 0.0, 2).expect_err("should reject zero eps");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn dbscan_rejects_negative_eps() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = dbscan(&data, -1.0, 2).expect_err("should reject negative eps");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn dbscan_rejects_nan_eps() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = dbscan(&data, f64::NAN, 2).expect_err("should reject NaN eps");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn dbscan_rejects_zero_min_samples() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = dbscan(&data, 0.5, 0).expect_err("should reject zero min_samples");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn vq_rejects_nan_data() {
        let data = vec![vec![f64::NAN, 0.0]];
        let centroids = vec![vec![0.0, 0.0]];
        let err = vq(&data, &centroids).expect_err("should reject NaN data");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn vq_rejects_nan_centroids() {
        let data = vec![vec![0.0, 0.0]];
        let centroids = vec![vec![f64::NAN, 0.0]];
        let err = vq(&data, &centroids).expect_err("should reject NaN centroids");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn vq_rejects_empty_centroids() {
        let data = vec![vec![0.0, 0.0]];
        let centroids: Vec<Vec<f64>> = vec![];
        let err = vq(&data, &centroids).expect_err("should reject empty centroids");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn whiten_rejects_nan_input() {
        let data = vec![vec![1.0, f64::NAN]];
        let err = whiten(&data).expect_err("should reject NaN input");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn whiten_rejects_inf_input() {
        let data = vec![vec![f64::INFINITY, 1.0]];
        let err = whiten(&data).expect_err("should reject Inf input");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn silhouette_perfect_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels).unwrap();
        assert!(score > 0.9, "silhouette = {score}, expected > 0.9");
    }

    #[test]
    fn silhouette_score_three_clusters_pinned_value() {
        // /profiling-software-performance regression pin for the
        // O(N²·k) → O(N² + Nk) optimization (frankenscipy-ktpz0).
        //
        // 6 points, 3 perfectly-separated 2-element clusters along
        // the x-axis: {(0,0),(0.1,0)}, {(10,0),(10.1,0)}, {(20,0),(20.1,0)}.
        // For each anchor a(i)=0.1 (same-cluster partner). For b(i),
        // the asymmetry between "near edge" and "far edge" of each
        // cluster relative to its neighbour gives two distinct b
        // values:
        //   - 4 anchors (the inner-facing point of each end cluster
        //     plus both points of the middle cluster) see b = 9.95
        //     ⇒ s = 9.85/9.95
        //   - 2 anchors (the outer-facing point of each end cluster)
        //     see b = 10.05 ⇒ s = 9.95/10.05
        // Mean = (4·9.85/9.95 + 2·9.95/10.05) / 6.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![20.0, 0.0],
            vec![20.1, 0.0],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];
        let score = silhouette_score(&data, &labels).expect("silhouette");
        let expected = (4.0 * (9.85 / 9.95) + 2.0 * (9.95 / 10.05)) / 6.0;
        assert!(
            (score - expected).abs() < 1e-12,
            "silhouette_score = {score}, expected {expected}"
        );
    }

    #[test]
    fn gap_statistic_rejects_ragged_input_without_panic() {
        let data = vec![vec![0.0], vec![1.0, 2.0]];
        assert!(gap_statistic(&data, 2, 3, 42).is_empty());
    }

    #[test]
    fn gap_statistic_rejects_non_finite_input_without_panic() {
        let data = vec![vec![0.0, f64::NAN], vec![1.0, 2.0]];
        assert!(gap_statistic(&data, 2, 3, 42).is_empty());
    }

    #[test]
    fn cophenet_basic() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        let d = cophenet(&z);
        assert_eq!(d.len(), 3); // C(3,2)
        // cophenetic distance between 0 and 1 = merge distance = 1.0
        assert!((d[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn silhouette_handles_length_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0]; // too short
        let err = silhouette_score(&data, &labels).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn calinski_harabasz_handles_length_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0]; // too short
        let err =
            calinski_harabasz_score(&data, &labels).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn davies_bouldin_handles_length_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0]; // too short
        let err = davies_bouldin_score(&data, &labels).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn silhouette_samples_handles_length_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0]; // too short
        let err = silhouette_samples(&data, &labels).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn mean_shift_rejects_zero_bandwidth() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = mean_shift(&data, 0.0, 10).expect_err("should reject zero bandwidth");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn mean_shift_rejects_negative_bandwidth() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = mean_shift(&data, -1.0, 10).expect_err("should reject negative bandwidth");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn mean_shift_rejects_nan_bandwidth() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let err = mean_shift(&data, f64::NAN, 10).expect_err("should reject NaN bandwidth");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn silhouette_score_single_cluster_errors() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0, 0]; // all in cluster 0
        let err = silhouette_score(&data, &labels).expect_err("single cluster is undefined");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn silhouette_samples_single_cluster_errors() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let labels = vec![0, 0, 0];
        let err = silhouette_samples(&data, &labels).expect_err("single cluster is undefined");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn kmedoids_returns_correct_n_iter() {
        // Well-separated clusters should converge quickly
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let result = kmedoids(&data, 2, 100, 42).unwrap();
        // Should converge in far fewer than 100 iterations
        assert!(
            result.n_iter < 100,
            "kmedoids should converge early but reported n_iter={}",
            result.n_iter
        );
    }

    #[test]
    fn kmedoids_basic_clustering() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let result = kmedoids(&data, 2, 100, 42).unwrap();
        assert_eq!(result.labels.len(), 4);
        // First two in one cluster, last two in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn kmedoids_picks_medoid_minimizing_total_l1_distance() {
        // /profiling-software-performance regression for
        // [frankenscipy-ebx8l]: the medoid update step now
        // pre-computes the M×M intra-cluster distance matrix once
        // and selects the row-sum minimum. Verify the medoid is
        // truly a member of the cluster minimizing the total
        // sum of distances to other members.
        //
        // Tight cluster on the x-axis: {0, 1, 5, 6, 100}.
        // Medoid should be the geometric median ≈ 5 (sum of
        // distances |0-5| + |1-5| + |6-5| + |100-5| = 5+4+1+95 = 105
        // vs. medoid=6: |0-6| + |1-6| + |5-6| + |100-6| = 6+5+1+94 = 106
        // vs. medoid=1: 1+4+5+99 = 109).
        let data = vec![vec![0.0_f64], vec![1.0], vec![5.0], vec![6.0], vec![100.0]];
        let result = kmedoids(&data, 1, 100, 42).expect("kmedoids");
        assert_eq!(result.labels.len(), 5);
        // All points belong to the single cluster.
        for &lbl in &result.labels {
            assert_eq!(lbl, 0);
        }
        // The selected medoid must be the row whose total L1 sum is
        // minimal — for this dataset that's the index 2 (value 5.0).
        let centroid = &result.centroids[0];
        assert!(
            (centroid[0] - 5.0).abs() < 1e-12,
            "medoid should be the value-5 point, got {centroid:?}"
        );
    }

    #[test]
    fn is_valid_linkage_accepts_good_linkage() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert!(is_valid_linkage(&z));
    }

    #[test]
    fn is_valid_linkage_rejects_bad_indices() {
        // Invalid linkage: index 10 doesn't exist
        let bad = [[10.0, 0.0, 1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_negative_child_index() {
        let bad = [[-1.0, 1.0, 1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_nonfinite_child_index() {
        let bad = [[f64::NAN, 1.0, 1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_fractional_child_index() {
        let bad = [[0.5, 1.0, 1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_negative_distance() {
        let bad = [[0.0, 1.0, -1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_self_merge() {
        let bad = [[0.0, 0.0, 1.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_reused_child_cluster() {
        let bad = [[0.0, 1.0, 1.0, 2.0], [0.0, 2.0, 2.0, 2.0]];
        assert!(!is_valid_linkage(&bad));
        assert_eq!(num_obs_linkage(&bad), 0);
    }

    #[test]
    fn is_monotonic_detects_non_monotonic() {
        // distances: 5.0, 1.0 - not monotonic
        let z = [[0.0, 1.0, 5.0, 2.0], [2.0, 3.0, 1.0, 3.0]];
        assert!(!is_monotonic(&z));
    }

    #[test]
    fn is_monotonic_accepts_monotonic() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert!(is_monotonic(&z));
    }

    #[test]
    fn leaves_list_returns_correct_order() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        let leaves = leaves_list(&z);
        assert_eq!(leaves.len(), 3);
        // Check all indices present
        let mut sorted = leaves.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn leaves_list_rejects_reused_child_cluster() {
        let bad = [[0.0, 1.0, 1.0, 2.0], [0.0, 2.0, 2.0, 2.0]];
        assert!(leaves_list(&bad).is_empty());
    }

    #[test]
    fn num_obs_linkage_computes_correctly() {
        let data = vec![vec![0.0], vec![1.0], vec![5.0], vec![10.0]];
        let z = linkage(&data, LinkageMethod::Single).unwrap();
        assert_eq!(num_obs_linkage(&z), 4);
    }

    #[test]
    fn fclusterdata_combines_workflow() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let labels = fclusterdata(&data, 2, LinkageMethod::Complete).unwrap();
        assert_eq!(labels.len(), 4);
        // First two in one cluster, last two in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn leaves_list_empty_linkage() {
        let z: Vec<[f64; 4]> = vec![];
        let leaves = leaves_list(&z);
        // Empty linkage means 1 observation
        assert_eq!(leaves, vec![0]);
    }

    #[test]
    fn is_valid_linkage_empty_is_valid() {
        let z: Vec<[f64; 4]> = vec![];
        assert!(is_valid_linkage(&z));
    }

    #[test]
    fn is_monotonic_empty_is_monotonic() {
        let z: Vec<[f64; 4]> = vec![];
        assert!(is_monotonic(&z));
    }

    #[test]
    fn is_monotonic_single_is_monotonic() {
        let z = [[0.0, 1.0, 5.0, 2.0]];
        assert!(is_monotonic(&z));
    }

    #[test]
    fn is_valid_linkage_rejects_nan_distance() {
        let bad = [[0.0, 1.0, f64::NAN, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_inf_distance() {
        let bad = [[0.0, 1.0, f64::INFINITY, 2.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_zero_count() {
        let bad = [[0.0, 1.0, 1.0, 0.0]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn is_valid_linkage_rejects_fractional_count() {
        let bad = [[0.0, 1.0, 1.0, 2.5]];
        assert!(!is_valid_linkage(&bad));
    }

    #[test]
    fn adjusted_rand_score_bails_on_adversarial_label_magnitude() {
        // Per frankenscipy-n240: adversarial labels like
        // [0, 1_000_000_000] previously triggered a 1e9*k*8 byte
        // contingency-table allocation + OOM. Dense label remapping keeps
        // memory bounded by the number of unique labels.
        let labels_true = vec![0usize; 4];
        let labels_pred = vec![0, 1_000_000_000, 2, 3];
        let score = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn normalized_mutual_info_bails_on_adversarial_label_magnitude() {
        let labels_true = vec![0usize; 4];
        let labels_pred = vec![0, 1_000_000_000, 2, 3];
        let score = normalized_mutual_info(&labels_true, &labels_pred).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn adjusted_rand_score_rejects_length_mismatch() {
        let err = adjusted_rand_score(&[0, 1], &[0]).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn adjusted_rand_score_trivial_single_sample_is_perfect() {
        assert_eq!(adjusted_rand_score(&[7], &[99]).unwrap(), 1.0);
    }

    #[test]
    fn normalized_mutual_info_rejects_length_mismatch() {
        let err = normalized_mutual_info(&[0, 1], &[0]).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn homogeneity_rejects_length_mismatch() {
        let err = homogeneity_score(&[0, 1], &[0]).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn v_measure_propagates_length_mismatch() {
        let err = v_measure_score(&[0, 1], &[0]).expect_err("length mismatch should error");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn linkage_single_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.linkage([[0,0],[1,0],[0,1],[4,4],[5,4],[4,5]], method='single')
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![4.0, 4.0],
            vec![5.0, 4.0],
            vec![4.0, 5.0],
        ];
        let z = linkage(&data, LinkageMethod::Single).expect("linkage");
        let expected = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 5.0, 6.0],
        ];
        assert_eq!(z.len(), expected.len());
        for (i, row) in z.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "linkage[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn leaves_list_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.leaves_list on single linkage result
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 5.0, 6.0],
        ];
        let leaves = leaves_list(&z);
        let expected = [2, 0, 1, 5, 3, 4];
        assert_eq!(leaves, expected);
    }

    #[test]
    fn num_obs_linkage_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.num_obs_linkage
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 5.0, 6.0],
        ];
        let n = num_obs_linkage(&z);
        assert_eq!(n, 6);
    }

    #[test]
    fn is_valid_linkage_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.is_valid_linkage
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 5.0, 6.0],
        ];
        assert!(is_valid_linkage(&z));
    }

    #[test]
    fn cophenet_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.cophenet
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 5.0, 6.0],
        ];
        let cophenetic = cophenet(&z);
        let expected = [1.0, 1.0, 5.0, 5.0, 5.0, 1.0];
        for (i, val) in cophenetic.iter().take(6).enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "cophenet[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn fcluster_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.fcluster(Z, t=2, criterion='maxclust')
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 6.0, 1.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [5.0, 8.0, 1.0, 3.0],
            [7.0, 9.0, 6.4031242374328485, 6.0],
        ];
        let labels = fcluster(&z, 2).expect("fcluster should succeed");
        let expected = [1, 1, 1, 2, 2, 2];
        for (i, (&got, &want)) in labels.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "fcluster[{i}] got {got}, expected {want}");
        }
    }

    #[test]
    fn adjusted_rand_score_matches_scipy_reference_values() {
        // sklearn.metrics.adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
        // -> 1.0 (perfect agreement)
        let true_labels = [0, 0, 1, 1];
        let pred_labels = [0, 0, 1, 1];
        let score = adjusted_rand_score(&true_labels, &pred_labels)
            .expect("adjusted_rand_score should succeed");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "adjusted_rand_score got {score}, expected 1.0"
        );
    }

    #[test]
    fn adjusted_rand_score_partial_agreement_matches_scipy_reference_values() {
        // sklearn.metrics.adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
        // -> 0.5714285714285714
        let true_labels = [0, 0, 1, 2];
        let pred_labels = [0, 0, 1, 1];
        let score = adjusted_rand_score(&true_labels, &pred_labels)
            .expect("adjusted_rand_score should succeed");
        assert!(
            (score - 0.5714285714285714).abs() < 1e-10,
            "adjusted_rand_score got {score}, expected 0.5714285714285714"
        );
    }

    #[test]
    fn whiten_matches_scipy_reference_values() {
        // scipy.cluster.vq.whiten([[1,2,3], [4,5,6], [7,8,9]])
        // scipy uses: obs / obs.std(axis=0) where std uses ddof=0 (population std)
        // whiten([1,4,7]) with std = sqrt(((1-4)^2 + (4-4)^2 + (7-4)^2)/3) = sqrt(6)
        // -> [1/sqrt(6), 4/sqrt(6), 7/sqrt(6)] ≈ [0.408, 1.633, 2.858]
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = whiten(&data).expect("whiten should succeed");
        // Check that all columns are scaled by same factor (uniformly scaled)
        // scipy.cluster.vq.whiten uses population std (ddof=0)
        let expected_std = 6.0_f64.sqrt(); // std of [1,4,7] with ddof=0
        let expected_col0 = [1.0 / expected_std, 4.0 / expected_std, 7.0 / expected_std];
        for (i, (got, &want)) in result
            .iter()
            .map(|r| r[0])
            .zip(expected_col0.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-10,
                "row {i} col 0 got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn vq_matches_scipy_reference_values() {
        // scipy.cluster.vq.vq([[0,0], [1,1], [10,10]], [[0,0], [10,10]])
        // -> ([0, 0, 1], [0., sqrt(2), 0.])
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![10.0, 10.0]];
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let (labels, dists) = vq(&data, &centroids).expect("vq should succeed");
        assert_eq!(labels, vec![0, 0, 1], "labels don't match scipy");
        let expected_dists = [0.0, 2.0_f64.sqrt(), 0.0];
        for (i, (&got, &want)) in dists.iter().zip(expected_dists.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "dist[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn silhouette_score_matches_scipy_reference_values() {
        // scipy.metrics.silhouette_score([[0,0], [1,1], [10,10], [11,11]], [0,0,1,1])
        // Two tight clusters well-separated -> score close to 1
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels).expect("silhouette_score should succeed");
        // scipy gives approximately 0.8575 for this case
        // Exact value: a_i = intra-cluster distance, b_i = nearest-cluster distance
        // For point (0,0): a = sqrt(2) (to (1,1)), b = min(dist to cluster 1) = sqrt(200) or sqrt(242)
        // Actually let me compute: (b-a)/max(a,b)
        // This should give a high positive value (> 0.8)
        assert!(
            score > 0.8,
            "silhouette_score got {score}, expected > 0.8 (two well-separated clusters)"
        );
    }

    #[test]
    fn davies_bouldin_score_matches_scipy_reference_values() {
        // sklearn.metrics.davies_bouldin_score([[0,0], [1,1], [10,10], [11,11]], [0,0,1,1])
        // Lower is better - two well-separated clusters should give low score
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score =
            davies_bouldin_score(&data, &labels).expect("davies_bouldin_score should succeed");
        // sklearn gives approximately 0.1 for this well-separated case
        assert!(
            score < 0.3,
            "davies_bouldin_score got {score}, expected < 0.3 (two well-separated clusters)"
        );
    }

    #[test]
    fn kmeans_well_separated_clusters_matches_scipy_reference_values() {
        // scipy.cluster.vq.kmeans2([[0,0], [1,1], [0,1], [1,0], [10,10], [11,11], [10,11], [11,10]], 2)
        // Two well-separated clusters should converge to centroids near [0.5, 0.5] and [10.5, 10.5]
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![10.0, 11.0],
            vec![11.0, 10.0],
        ];
        let result = kmeans(&data, 2, 100, 42).expect("kmeans should succeed");
        // Check that we got 2 distinct clusters
        let labels = &result.labels;
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2, "should have exactly 2 clusters");
        // Check centroids are reasonable (close to cluster means)
        for centroid in &result.centroids {
            let dist_to_low = ((centroid[0] - 0.5).powi(2) + (centroid[1] - 0.5).powi(2)).sqrt();
            let dist_to_high = ((centroid[0] - 10.5).powi(2) + (centroid[1] - 10.5).powi(2)).sqrt();
            assert!(
                dist_to_low < 0.5 || dist_to_high < 0.5,
                "centroid {:?} should be near [0.5,0.5] or [10.5,10.5]",
                centroid
            );
        }
    }

    #[test]
    fn dbscan_two_clusters_matches_scipy_reference_values() {
        // sklearn.cluster.DBSCAN(eps=2.0, min_samples=2).fit_predict([[0,0], [1,1], [0,1], [1,0], [10,10], [11,11]])
        // Should find 2 clusters with no noise points
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let result = dbscan(&data, 2.0, 2).expect("dbscan should succeed");
        // All points should be assigned (no noise = -1)
        assert!(
            result.labels.iter().all(|&l| l >= 0),
            "should have no noise points"
        );
        // Should have exactly 2 clusters
        assert_eq!(result.n_clusters, 2, "should have exactly 2 clusters");
    }

    #[test]
    fn fclusterdata_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.fclusterdata([[0,0], [0,1], [4,4], [4,5]], t=2, criterion='maxclust')
        // -> [1, 1, 2, 2]
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![4.0, 4.0],
            vec![4.0, 5.0],
        ];
        let result = fclusterdata(&data, 2, LinkageMethod::Single).expect("fclusterdata");
        // First two points should be in same cluster, last two in another
        assert_eq!(result[0], result[1], "points 0,1 should be in same cluster");
        assert_eq!(result[2], result[3], "points 2,3 should be in same cluster");
        assert_ne!(
            result[0], result[2],
            "cluster 1 should differ from cluster 2"
        );
    }

    #[test]
    fn inconsistent_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.inconsistent(z, d=2) for z from linkage of [[0,0], [0,1], [4,4], [4,5]]
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![4.0, 4.0],
            vec![4.0, 5.0],
        ];
        let z = linkage(&data, LinkageMethod::Single).expect("linkage");
        let incon = inconsistent(&z, 2);
        assert_eq!(incon.len(), 3, "inconsistent should have 3 rows");
        // First two rows have 1 element each so std=0, R=0
        assert!(
            (incon[0][1] - 0.0).abs() < 1e-10,
            "std of row 0 should be 0"
        );
        assert!(
            (incon[1][1] - 0.0).abs() < 1e-10,
            "std of row 1 should be 0"
        );
        // Third row has multiple elements so std > 0
        assert!(incon[2][1] > 0.0, "std of row 2 should be > 0");
    }

    #[test]
    fn is_monotonic_matches_scipy_reference_values() {
        // scipy.cluster.hierarchy.is_monotonic(z) for properly formed linkage
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![4.0, 4.0],
            vec![4.0, 5.0],
        ];
        let z = linkage(&data, LinkageMethod::Single).expect("linkage");
        let z_arr: Vec<[f64; 4]> = z
            .iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect();
        assert!(is_monotonic(&z_arr), "valid linkage should be monotonic");
    }

    #[test]
    fn calinski_harabasz_score_matches_scipy_reference_values() {
        // sklearn.metrics.calinski_harabasz_score([[0,0], [0,1], [10,10], [10,11]], [0, 0, 1, 1])
        // Well-separated clusters should have high score
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![10.0, 11.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = calinski_harabasz_score(&data, &labels).expect("calinski_harabasz");
        // For well-separated clusters, score should be high (> 100)
        assert!(
            score > 100.0,
            "calinski_harabasz got {}, expected > 100",
            score
        );
    }

    #[test]
    fn fowlkes_mallows_score_matches_scipy_reference_values() {
        // sklearn.metrics.fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
        // Perfect agreement should give 1.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];
        let score = fowlkes_mallows_score(&labels_true, &labels_pred).expect("fowlkes_mallows");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "fowlkes_mallows perfect agreement got {}, expected 1.0",
            score
        );

        // sklearn.metrics.fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
        // Swapped labels should also give 1.0 (labels are arbitrary)
        let labels_pred_swapped = vec![1, 1, 0, 0];
        let score_swapped =
            fowlkes_mallows_score(&labels_true, &labels_pred_swapped).expect("fowlkes_mallows");
        assert!(
            (score_swapped - 1.0).abs() < 1e-10,
            "fowlkes_mallows swapped got {}, expected 1.0",
            score_swapped
        );
    }

    #[test]
    fn homogeneity_score_matches_scipy_reference_values() {
        // sklearn.metrics.homogeneity_score([0, 0, 1, 1], [0, 0, 1, 1]) = 1.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];
        let score = homogeneity_score(&labels_true, &labels_pred).expect("homogeneity");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "homogeneity perfect got {}, expected 1.0",
            score
        );
    }

    #[test]
    fn completeness_score_matches_scipy_reference_values() {
        // sklearn.metrics.completeness_score([0, 0, 1, 1], [0, 0, 1, 1]) = 1.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];
        let score = completeness_score(&labels_true, &labels_pred).expect("completeness");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "completeness perfect got {}, expected 1.0",
            score
        );
    }

    #[test]
    fn v_measure_score_matches_scipy_reference_values() {
        // sklearn.metrics.v_measure_score([0, 0, 1, 1], [0, 0, 1, 1]) = 1.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];
        let score = v_measure_score(&labels_true, &labels_pred).expect("v_measure");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "v_measure perfect got {}, expected 1.0",
            score
        );
    }

    #[test]
    fn normalized_mutual_info_matches_scipy_reference_values() {
        // sklearn.metrics.normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1]) = 1.0
        let labels_true = vec![0, 0, 1, 1];
        let labels_pred = vec![0, 0, 1, 1];
        let score = normalized_mutual_info(&labels_true, &labels_pred).expect("nmi");
        assert!(
            (score - 1.0).abs() < 1e-10,
            "nmi perfect got {}, expected 1.0",
            score
        );
    }
}
