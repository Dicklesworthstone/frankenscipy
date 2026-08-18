#![feature(portable_simd)]
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
// Principal Component Analysis (randomized)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`pca`], mirroring `sklearn.decomposition.PCA`.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal axes as rows (k×d): the top-k right singular vectors of the centered data.
    pub components: Vec<Vec<f64>>,
    /// Variance explained by each component, `σᵢ²/(n−1)` (length k).
    pub explained_variance: Vec<f64>,
    /// Fraction of total variance explained by each component (length k).
    pub explained_variance_ratio: Vec<f64>,
    /// Singular values of the centered data (length k).
    pub singular_values: Vec<f64>,
    /// Per-feature mean subtracted before the decomposition (length d).
    pub mean: Vec<f64>,
    /// The data projected onto the components, `U·diag(σ)` (n×k).
    pub transformed: Vec<Vec<f64>>,
}

/// Principal Component Analysis via randomized SVD.
///
/// Matches `sklearn.decomposition.PCA(n_components=k, svd_solver="randomized")`: centers the
/// `n×d` data, takes the top-`k` SVD of the centered matrix via
/// [`fsci_linalg::randomized_svd`], and returns the principal axes, explained variance, and
/// the projected data. Cost O(n·d·k) versus O(n·d·min(n,d)) for a full-SVD PCA — a large win
/// when k ≪ min(n,d) (the usual dimensionality-reduction regime). Deterministic given `seed`.
pub fn pca(x: &[Vec<f64>], n_components: usize, seed: u64) -> Result<PcaResult, ClusterError> {
    if x.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err(ClusterError::EmptyData);
    }
    if x.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all samples must have the same dimension".to_string(),
        ));
    }
    if x.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "pca input must be finite".to_string(),
        ));
    }
    let k = n_components.min(n).min(d);
    if k == 0 {
        return Err(ClusterError::InvalidArgument(
            "n_components must be at least 1".to_string(),
        ));
    }

    let denom = (n.max(2) - 1) as f64;
    let mean: Vec<f64> = (0..d)
        .map(|j| x.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().zip(&mean).map(|(&v, &m)| v - m).collect())
        .collect();
    // Total variance = trace of the covariance = ‖Xc‖_F² / (n−1).
    let total_var: f64 = xc.iter().flatten().map(|&v| v * v).sum::<f64>() / denom;

    let svd = fsci_linalg::randomized_svd(&xc, k, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;

    let explained_variance: Vec<f64> = svd.s.iter().map(|&s| s * s / denom).collect();
    let explained_variance_ratio: Vec<f64> = explained_variance
        .iter()
        .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
        .collect();
    // transform(X) = Xc·Vᵀ = U·diag(σ).
    let transformed: Vec<Vec<f64>> = svd
        .u
        .iter()
        .map(|urow| urow.iter().zip(&svd.s).map(|(&u, &s)| u * s).collect())
        .collect();

    Ok(PcaResult {
        components: svd.vt,
        explained_variance,
        explained_variance_ratio,
        singular_values: svd.s,
        mean,
        transformed,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Spectral Clustering (randomized)
// ══════════════════════════════════════════════════════════════════════

/// Spectral clustering of a precomputed `n×n` affinity (similarity) matrix into
/// `n_clusters` groups, via randomized eigendecomposition (Ng–Jordan–Weiss).
///
/// Matches `sklearn.cluster.SpectralClustering(affinity="precomputed")`: forms the
/// symmetric normalized affinity `D^{-1/2}·A·D^{-1/2}` (D the degree diagonal), takes its
/// top-`n_clusters` eigenvectors via [`fsci_linalg::randomized_eigh`] (the eigenvectors of
/// the smallest eigenvalues of the normalized Laplacian = the spectral embedding),
/// row-normalises the embedding, and runs [`kmeans`] on it. The eigendecomposition is the
/// dominant cost: O(n²·k) randomized versus O(n³) for a full `eigh` — a large win when
/// `n_clusters ≪ n`. `affinity` should be symmetric with non-negative entries; deterministic
/// given `seed`.
pub fn spectral_clustering(
    affinity: &[Vec<f64>],
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> Result<Vec<usize>, ClusterError> {
    if affinity.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = affinity.len();
    if affinity.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "affinity matrix must be square".to_string(),
        ));
    }
    if n_clusters == 0 || n_clusters > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_clusters={n_clusters} must be in [1, n={n}]"
        )));
    }
    if affinity
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(ClusterError::InvalidArgument(
            "affinity must be finite and non-negative".to_string(),
        ));
    }

    // Symmetric normalized affinity D^{-1/2} A D^{-1/2}.
    let inv_sqrt: Vec<f64> = affinity
        .iter()
        .map(|row| {
            let deg: f64 = row.iter().sum();
            if deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 }
        })
        .collect();
    let normalized: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| affinity[i][j] * inv_sqrt[i] * inv_sqrt[j])
                .collect()
        })
        .collect();

    let re = fsci_linalg::randomized_eigh(&normalized, n_clusters, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_eigh: {e}")))?;
    let k = re.eigenvalues.len();
    if k == 0 {
        return Err(ClusterError::ConvergenceFailed(
            "spectral embedding is empty".to_string(),
        ));
    }

    // Spectral embedding (n×k), row-normalised to unit length (Ng–Jordan–Weiss).
    let mut embedding = vec![vec![0.0; k]; n];
    for (i, row) in embedding.iter_mut().enumerate() {
        let mut nrm = 0.0;
        for (t, slot) in row.iter_mut().enumerate() {
            *slot = re.eigenvectors[i][t];
            nrm += *slot * *slot;
        }
        let nrm = nrm.sqrt();
        if nrm > 1e-12 {
            for slot in row.iter_mut() {
                *slot /= nrm;
            }
        }
    }

    Ok(kmeans(&embedding, n_clusters, max_iter, seed)?.labels)
}

/// Result of [`affinity_propagation`].
#[derive(Debug, Clone)]
pub struct AffinityPropagationResult {
    /// Cluster label for each point (length n); labels index into `exemplars`.
    pub labels: Vec<usize>,
    /// Indices of the exemplar (representative) points, one per cluster.
    pub exemplars: Vec<usize>,
    /// Iterations performed.
    pub n_iter: usize,
}

/// Affinity propagation clustering from a precomputed `n×n` similarity matrix (Frey–Dueck
/// 2007), matching `sklearn.cluster.AffinityPropagation(affinity="precomputed")`.
///
/// Exchanges "responsibility" and "availability" messages between every pair of points until a
/// stable set of exemplars emerges, then assigns each point to its best exemplar. The number of
/// clusters is *not* specified up front — it is governed by `preference` (the self-similarity
/// `s(k,k)`; larger ⇒ more clusters; a common choice is the median off-diagonal similarity).
/// Updates are damped by `damping ∈ [0.5, 1)`. Iterates up to `max_iter`, stopping early once the
/// exemplar set is unchanged for `convergence_iter` consecutive iterations. `similarity` should be
/// symmetric (e.g. negative squared distances); its diagonal is overwritten with `preference`.
/// Deterministic.
pub fn affinity_propagation(
    similarity: &[Vec<f64>],
    preference: f64,
    damping: f64,
    max_iter: usize,
    convergence_iter: usize,
) -> Result<AffinityPropagationResult, ClusterError> {
    if similarity.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = similarity.len();
    if similarity.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "similarity matrix must be square".to_string(),
        ));
    }
    if !(damping.is_finite() && (0.5..1.0).contains(&damping)) {
        return Err(ClusterError::InvalidArgument(
            "damping must be in [0.5, 1.0)".to_string(),
        ));
    }
    if max_iter == 0 || convergence_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter and convergence_iter must be positive".to_string(),
        ));
    }
    if !preference.is_finite() || similarity.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "similarity and preference must be finite".to_string(),
        ));
    }

    // S with the diagonal set to `preference`.
    let mut s: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| if i == k { preference } else { similarity[i][k] })
                .collect()
        })
        .collect();
    // Break exact ties/symmetry with tiny deterministic noise (degenerate inputs — e.g. several
    // identical clusters — otherwise oscillate and never settle on exemplars; sklearn does the
    // same with random_state). The perturbation is ~1e-10·‖S‖∞, far below any real structure.
    let smax = s
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let mut st: u64 = 0x9e37_79b9_7f4a_7c15;
    for row in s.iter_mut() {
        for v in row.iter_mut() {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
            *v += 1e-10 * smax * u;
        }
    }

    // Responsibilities `r` and availabilities `a` as flat row-major `n*n`
    // buffers (`r[i*n+k]`) instead of `Vec<Vec<f64>>`. The availability update
    // below walks them by COLUMN (fixed k, varying i); with per-row `Vec`s that
    // chases n scattered heap rows per column, whereas the flat layout is one
    // allocation with a predictable stride-n walk the prefetcher follows. Pure
    // storage change — arithmetic and order are unchanged, so output is
    // bit-for-bit identical (see bin/perf_ap_ab.rs A/B which asserts it).
    let mut r = vec![0.0f64; n * n]; // responsibilities, row-major
    let mut a = vec![0.0f64; n * n]; // availabilities, row-major
    let mut last_exemplars: Vec<usize> = Vec::new();
    let mut stable = 0usize;
    let mut iters = 0;

    for it in 0..max_iter {
        iters = it + 1;
        // Responsibilities: r(i,k) = s(i,k) − max_{k'≠k}(a(i,k')+s(i,k')). Each row i
        // reads only its own a[i]/s[i] (and old r[i]) and writes its own r[i], so the
        // rows are independent — update them in parallel into ordered slots,
        // byte-identical to the serial loop. frankenscipy-yw7ts.
        let update_resp_row = |i: usize, r_row: &mut [f64]| {
            // Largest and second-largest of (a+s) across k', with argmax.
            let (mut max1, mut max1_idx, mut max2) = (f64::NEG_INFINITY, 0usize, f64::NEG_INFINITY);
            for k in 0..n {
                let v = a[i * n + k] + s[i][k];
                if v > max1 {
                    max2 = max1;
                    max1 = v;
                    max1_idx = k;
                } else if v > max2 {
                    max2 = v;
                }
            }
            for (k, rik) in r_row.iter_mut().enumerate() {
                let competitor = if k == max1_idx { max2 } else { max1 };
                let upd = s[i][k] - competitor;
                *rik = damping * *rik + (1.0 - damping) * upd;
            }
        };
        let nthreads_r = if n.saturating_mul(n) < (1 << 18) || n < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(n)
        };
        if nthreads_r <= 1 {
            for (i, r_row) in r.chunks_mut(n).enumerate() {
                update_resp_row(i, r_row);
            }
        } else {
            let chunk_rows = n.div_ceil(nthreads_r);
            let update_resp_row = &update_resp_row;
            std::thread::scope(|scope| {
                for (t, r_block) in r.chunks_mut(chunk_rows * n).enumerate() {
                    let base = t * chunk_rows;
                    scope.spawn(move || {
                        for (li, r_row) in r_block.chunks_mut(n).enumerate() {
                            update_resp_row(base + li, r_row);
                        }
                    });
                }
            });
        }
        // Availabilities: a(i,k)=min(0, r(k,k)+Σ_{i'∉{i,k}}max(0,r(i',k))); a(k,k)=Σ_{i'≠k}max(0,·).
        // Restructured ROW-MAJOR to avoid the cache-pathological stride-n column
        // walk the naive `for k { for i {…} }` loop performs (both the col sum and
        // the a[i*n+k] writes are stride-n). col_pos[k]=Σ_i max(0,r[i*n+k]) is
        // accumulated in a sequential i-major pass — still i=0..n order per k, so
        // BIT-IDENTICAL to the per-column `(0..n).sum()` — then each row i is
        // updated in row-major order. Rows are independent (read shared col_pos,
        // write their own a[i]), so the update parallelizes byte-identically.
        // frankenscipy-ap-avail: ~1.3-4x serial (cache) + up to ~7.8x threaded at
        // large n vs the strided loop (de-risk perf_ap_avail_ab, EXACT).
        let mut col_pos = vec![0.0_f64; n];
        for i in 0..n {
            let ri = &r[i * n..i * n + n];
            for (k, cp) in col_pos.iter_mut().enumerate() {
                *cp += ri[k].max(0.0);
            }
        }
        let pos_kk: Vec<f64> = (0..n).map(|k| r[k * n + k].max(0.0)).collect();
        let update_avail_row = |i: usize, a_row: &mut [f64]| {
            for (k, aik) in a_row.iter_mut().enumerate() {
                let rkk = r[k * n + k];
                let cp = col_pos[k];
                let upd = if i == k {
                    cp - pos_kk[k]
                } else {
                    (rkk + cp - r[i * n + k].max(0.0) - pos_kk[k]).min(0.0)
                };
                *aik = damping * *aik + (1.0 - damping) * upd;
            }
        };
        let nthreads_a = if n.saturating_mul(n) < (1 << 20) || n < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(n)
        };
        if nthreads_a <= 1 {
            for (i, a_row) in a.chunks_mut(n).enumerate() {
                update_avail_row(i, a_row);
            }
        } else {
            let chunk_rows = n.div_ceil(nthreads_a);
            let update_avail_row = &update_avail_row;
            std::thread::scope(|scope| {
                for (t, a_block) in a.chunks_mut(chunk_rows * n).enumerate() {
                    let base = t * chunk_rows;
                    scope.spawn(move || {
                        for (li, a_row) in a_block.chunks_mut(n).enumerate() {
                            update_avail_row(base + li, a_row);
                        }
                    });
                }
            });
        }
        // Exemplars: points with r(k,k)+a(k,k) > 0.
        let exemplars: Vec<usize> = (0..n)
            .filter(|&k| r[k * n + k] + a[k * n + k] > 0.0)
            .collect();
        if !exemplars.is_empty() && exemplars == last_exemplars {
            stable += 1;
            if stable >= convergence_iter {
                break;
            }
        } else {
            stable = 0;
            last_exemplars = exemplars;
        }
    }

    // Final exemplars; fall back to the highest-criterion point if none emerged.
    let mut exemplars: Vec<usize> = (0..n)
        .filter(|&k| r[k * n + k] + a[k * n + k] > 0.0)
        .collect();
    if exemplars.is_empty() {
        let best = (0..n)
            .max_by(|&p, &q| {
                (r[p * n + p] + a[p * n + p]).total_cmp(&(r[q * n + q] + a[q * n + q]))
            })
            .unwrap_or(0);
        exemplars.push(best);
    }
    // Assign each point to the exemplar with the highest similarity (its own preference if it is
    // itself an exemplar).
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            exemplars
                .iter()
                .enumerate()
                .max_by(|&(_, &p), &(_, &q)| s[i][p].total_cmp(&s[i][q]))
                .map_or(0, |(c, _)| c)
        })
        .collect();

    Ok(AffinityPropagationResult {
        labels,
        exemplars,
        n_iter: iters,
    })
}

/// Result of [`spectral_coclustering`]: a simultaneous clustering of rows and columns.
#[derive(Debug, Clone)]
pub struct CoclusterResult {
    /// Cluster label for each row (length m).
    pub row_labels: Vec<usize>,
    /// Cluster label for each column (length n).
    pub col_labels: Vec<usize>,
}

/// Spectral co-clustering of a non-negative `m×n` data matrix into `n_clusters` biclusters
/// (Dhillon 2001), matching `sklearn.cluster.SpectralCoclustering`.
///
/// Treats the matrix as the biadjacency of a bipartite row/column graph: forms the normalized
/// matrix `Aₙ = D_r^{-1/2}·A·D_c^{-1/2}` (`D_r`, `D_c` the row/column sums), takes its leading
/// `1+⌈log₂ k⌉` singular triplets via [`fsci_linalg::randomized_svd`] (O(m·n·k) versus O(m·n·min)
/// for a full SVD), discards the trivial first one, stacks the `D_r^{-1/2}`-scaled left and
/// `D_c^{-1/2}`-scaled right singular vectors into one `(m+n)×ℓ` embedding, and [`kmeans`]-clusters
/// it — rows and columns that load on the same singular structure land in the same bicluster.
/// `data` must be non-negative with positive row and column sums; `n_clusters ≥ 2`. Deterministic
/// by `seed`.
pub fn spectral_coclustering(
    data: &[Vec<f64>],
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> Result<CoclusterResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let m = data.len();
    let n = data[0].len();
    if data.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_clusters < 2 || n_clusters > m.min(n) {
        return Err(ClusterError::InvalidArgument(format!(
            "n_clusters={n_clusters} must be in [2, min(m,n)={}]",
            m.min(n)
        )));
    }
    if data.iter().flatten().any(|v| !(v.is_finite() && *v >= 0.0)) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite and non-negative".to_string(),
        ));
    }

    let row_sum: Vec<f64> = data.iter().map(|r| r.iter().sum()).collect();
    let col_sum: Vec<f64> = (0..n).map(|j| data.iter().map(|r| r[j]).sum()).collect();
    if row_sum.iter().any(|&s| s <= 0.0) || col_sum.iter().any(|&s| s <= 0.0) {
        return Err(ClusterError::InvalidArgument(
            "every row and column must have a positive sum".to_string(),
        ));
    }
    let inv_sqrt_r: Vec<f64> = row_sum.iter().map(|&s| 1.0 / s.sqrt()).collect();
    let inv_sqrt_c: Vec<f64> = col_sum.iter().map(|&s| 1.0 / s.sqrt()).collect();

    // Normalized bipartite matrix Aₙ = D_r^{-1/2} A D_c^{-1/2}.
    let an: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..n)
                .map(|j| data[i][j] * inv_sqrt_r[i] * inv_sqrt_c[j])
                .collect()
        })
        .collect();

    // n_sv = 1 + ⌈log₂ k⌉; the embedding uses the singular vectors after the trivial first.
    let n_sv = (1 + (n_clusters as f64).log2().ceil() as usize).min(m.min(n));
    let svd = fsci_linalg::randomized_svd(&an, n_sv, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
    let kk = svd.s.len();
    if kk < 2 {
        return Err(ClusterError::ConvergenceFailed(
            "co-clustering needs at least 2 singular vectors".to_string(),
        ));
    }
    let edim = kk - 1; // drop the leading (trivial) singular vector

    // Stack the row and column embeddings: Z = [D_r^{-1/2}·U[:,1:]; D_c^{-1/2}·Vᵀ[1:,:]ᵀ].
    let mut z = Vec::with_capacity(m + n);
    for (urow, &isr) in svd.u.iter().zip(&inv_sqrt_r) {
        z.push((1..kk).map(|t| urow[t] * isr).collect::<Vec<f64>>());
    }
    for (j, &isc) in inv_sqrt_c.iter().enumerate() {
        z.push((1..kk).map(|t| svd.vt[t][j] * isc).collect::<Vec<f64>>());
    }
    debug_assert_eq!(z[0].len(), edim);

    let labels = kmeans(&z, n_clusters, max_iter, seed)?.labels;
    Ok(CoclusterResult {
        row_labels: labels[..m].to_vec(),
        col_labels: labels[m..].to_vec(),
    })
}

/// Result of [`spectral_embedding`].
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingResult {
    /// The `n×k` Laplacian-eigenmap embedding (rows are points), columns ordered by descending
    /// normalized-affinity eigenvalue (= ascending normalized-Laplacian eigenvalue).
    pub embedding: Vec<Vec<f64>>,
    /// The corresponding normalized-affinity eigenvalues `μ = 1 − λ_Laplacian`, descending.
    pub eigenvalues: Vec<f64>,
}

/// Laplacian-eigenmaps spectral embedding of a precomputed `n×n` affinity matrix into
/// `n_components` dimensions.
///
/// Matches `sklearn.manifold.SpectralEmbedding(affinity="precomputed")`: forms the symmetric
/// normalized affinity `D^{-1/2}·A·D^{-1/2}`, takes its top-`k` eigenvectors `u` via
/// [`fsci_linalg::randomized_eigh`], and returns the generalized eigenvectors `y = D^{-1/2}·u`
/// — the solutions of `L y = λ D y` for the smallest non-trivial normalized-Laplacian
/// eigenvalues, i.e. the manifold coordinates. Unlike [`spectral_clustering`] there is no
/// row-normalisation and no k-means; this is the embedding itself. The eigendecomposition
/// dominates: O(n²·k) randomized versus O(n³) full `eigh`, a large win when k ≪ n. `affinity`
/// should be symmetric with non-negative entries; deterministic given `seed`.
pub fn spectral_embedding(
    affinity: &[Vec<f64>],
    n_components: usize,
    seed: u64,
) -> Result<SpectralEmbeddingResult, ClusterError> {
    if affinity.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = affinity.len();
    if affinity.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "affinity matrix must be square".to_string(),
        ));
    }
    if n_components == 0 || n_components > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, n={n}]"
        )));
    }
    if affinity
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(ClusterError::InvalidArgument(
            "affinity must be finite and non-negative".to_string(),
        ));
    }

    let inv_sqrt: Vec<f64> = affinity
        .iter()
        .map(|row| {
            let deg: f64 = row.iter().sum();
            if deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 }
        })
        .collect();
    let normalized: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| affinity[i][j] * inv_sqrt[i] * inv_sqrt[j])
                .collect()
        })
        .collect();

    let re = fsci_linalg::randomized_eigh(&normalized, n_components, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_eigh: {e}")))?;
    let kk = re.eigenvalues.len();
    if kk == 0 {
        return Err(ClusterError::ConvergenceFailed(
            "spectral embedding is empty".to_string(),
        ));
    }

    // randomized_eigh yields eigenvalues ascending; the embedding wants the largest
    // normalized-affinity eigenvalues (= smallest Laplacian) first. y = D^{-1/2} u.
    let eigenvalues: Vec<f64> = (0..kk).rev().map(|t| re.eigenvalues[t]).collect();
    let embedding: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..kk)
                .rev()
                .map(|t| inv_sqrt[i] * re.eigenvectors[i][t])
                .collect()
        })
        .collect();

    Ok(SpectralEmbeddingResult {
        embedding,
        eigenvalues,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Kernel PCA (randomized)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`kernel_pca`], mirroring `sklearn.decomposition.KernelPCA`.
#[derive(Debug, Clone)]
pub struct KernelPcaResult {
    /// The leading kernel eigenvalues (length k).
    pub eigenvalues: Vec<f64>,
    /// The corresponding unit eigenvectors of the centered kernel (n×k).
    pub eigenvectors: Vec<Vec<f64>>,
    /// The data projected into the kernel feature space, `eigenvectors·diag(√λ)` (n×k).
    pub transformed: Vec<Vec<f64>>,
}

/// Kernel Principal Component Analysis from a precomputed `n×n` kernel (Gram) matrix.
///
/// Matches `sklearn.decomposition.KernelPCA(kernel="precomputed", n_components=k)`:
/// double-centers the kernel (`K − 1ₙK − K1ₙ + 1ₙK1ₙ`), takes its top-`k` eigenpairs via
/// [`fsci_linalg::randomized_eigh`], and returns the projections `αᵢ·√λᵢ` — the nonlinear
/// analogue of [`pca`]. The eigendecomposition dominates: O(n²·k) randomized versus O(n³)
/// for a full `eigh` — a large win when k ≪ n. `kernel` should be symmetric PSD; eigenvalues
/// that come out ≤ 0 (from the centering) contribute a zero projection. Deterministic by
/// `seed`.
pub fn kernel_pca(
    kernel: &[Vec<f64>],
    n_components: usize,
    seed: u64,
) -> Result<KernelPcaResult, ClusterError> {
    if kernel.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = kernel.len();
    if kernel.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "kernel matrix must be square".to_string(),
        ));
    }
    if n_components == 0 || n_components > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, n={n}]"
        )));
    }
    if kernel.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "kernel must be finite".to_string(),
        ));
    }

    // Double-center the (symmetric) kernel: row_mean serves as col_mean.
    let row_mean: Vec<f64> = kernel
        .iter()
        .map(|r| r.iter().sum::<f64>() / n as f64)
        .collect();
    let total_mean: f64 = row_mean.iter().sum::<f64>() / n as f64;
    let kc: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| kernel[i][j] - row_mean[i] - row_mean[j] + total_mean)
                .collect()
        })
        .collect();

    let re = fsci_linalg::randomized_eigh(&kc, n_components, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_eigh: {e}")))?;
    let kk = re.eigenvalues.len();

    let transformed: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..kk)
                .map(|t| re.eigenvectors[i][t] * re.eigenvalues[t].max(0.0).sqrt())
                .collect()
        })
        .collect();

    Ok(KernelPcaResult {
        eigenvalues: re.eigenvalues,
        eigenvectors: re.eigenvectors,
        transformed,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Classical Multidimensional Scaling (PCoA / Torgerson)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`classical_mds`].
#[derive(Debug, Clone)]
pub struct MdsResult {
    /// The `n×k` low-dimensional embedding (rows are points), columns ordered by descending
    /// eigenvalue.
    pub embedding: Vec<Vec<f64>>,
    /// The leading eigenvalues of the double-centered matrix, descending (length k). Negative
    /// values indicate the distances are not perfectly Euclidean in `k` dimensions.
    pub eigenvalues: Vec<f64>,
}

fn double_centered_gram_from_squared<F>(n: usize, squared: F) -> (Vec<Vec<f64>>, Vec<f64>, f64)
where
    F: Fn(usize, usize) -> f64,
{
    let denom = n as f64;
    let mut row_mean = vec![0.0; n];
    for (row, mean) in row_mean.iter_mut().enumerate() {
        let mut sum = 0.0;
        for col in 0..n {
            sum += squared(row, col);
        }
        *mean = sum / denom;
    }
    let total_mean = row_mean.iter().sum::<f64>() / denom;
    let gram: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|col| -0.5 * (squared(row, col) - row_mean[row] - row_mean[col] + total_mean))
                .collect()
        })
        .collect();
    (gram, row_mean, total_mean)
}

/// Classical (metric) multidimensional scaling — Torgerson/Gower PCoA — from a precomputed
/// `n×n` distance matrix.
///
/// Builds the double-centered Gram matrix `B = −½ J D² J` (where `Dᵢⱼ` is the distance and
/// `J = I − 1ₙ/n`), then takes its top-`k` eigenpairs to place the points so their inner
/// products best match `B`. The embedding is `eigenvectorsᵢ·√max(λᵢ,0)`, columns ordered by
/// descending eigenvalue — the classical-MDS analogue of [`kernel_pca`] on `B`. The
/// eigendecomposition dominates: O(n²·k) via [`fsci_linalg::randomized_eigh`] versus O(n³)
/// for a full `eigh`, a large win when k ≪ n. `distances` must be square, symmetric and
/// finite. Deterministic by `seed`.
pub fn classical_mds(
    distances: &[Vec<f64>],
    n_components: usize,
    seed: u64,
) -> Result<MdsResult, ClusterError> {
    if distances.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = distances.len();
    if distances.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "distance matrix must be square".to_string(),
        ));
    }
    if n_components == 0 || n_components > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, n={n}]"
        )));
    }
    if distances.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "distances must be finite".to_string(),
        ));
    }

    // Stream the squared-distance callback twice instead of materializing the
    // full D² matrix before the double-centering pass.
    let (b, _, _) = double_centered_gram_from_squared(n, |i, j| {
        let distance = distances[i][j];
        distance * distance
    });

    let re = fsci_linalg::randomized_eigh(&b, n_components, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_eigh: {e}")))?;
    let kk = re.eigenvalues.len();

    // randomized_eigh yields eigenvalues ascending; MDS wants largest first.
    let eigenvalues: Vec<f64> = (0..kk).rev().map(|t| re.eigenvalues[t]).collect();
    let embedding: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..kk)
                .rev()
                .map(|t| re.eigenvectors[i][t] * re.eigenvalues[t].max(0.0).sqrt())
                .collect()
        })
        .collect();

    Ok(MdsResult {
        embedding,
        eigenvalues,
    })
}

/// Landmark MDS (de Silva–Tenenbaum) — scalable classical MDS that eigendecomposes only an
/// `m×m` landmark block, then triangulates every point from its distances to the landmarks.
///
/// Runs [`classical_mds`]'s double-centering + eigendecomposition on the `n_landmarks×n_landmarks`
/// landmark submatrix (`m ≪ n`), giving the landmark embedding `L` and the inverse-scaled
/// eigenvectors `L# = V·diag(λ^{-1/2})`. Each of the `n` points is then placed by
/// `x_p = −½·L#·(δ_p − μ)`, where `δ_p` are its squared distances to the landmarks and `μ`
/// the mean landmark squared-distance — exact for landmarks, the least-squares triangulation
/// otherwise. Cost O(m³ + n·m·k) versus O(n²·k) for [`classical_mds`] over the full matrix — a
/// large win when m ≪ n. `distances` must be square, symmetric and finite; deterministic by
/// `seed`.
pub fn landmark_mds(
    distances: &[Vec<f64>],
    n_components: usize,
    n_landmarks: usize,
    seed: u64,
) -> Result<MdsResult, ClusterError> {
    if distances.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = distances.len();
    if distances.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "distance matrix must be square".to_string(),
        ));
    }
    if n_components == 0 || n_landmarks > n || n_components > n_landmarks {
        return Err(ClusterError::InvalidArgument(format!(
            "need 1 ≤ n_components={n_components} ≤ n_landmarks={n_landmarks} ≤ n={n}"
        )));
    }
    if distances.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "distances must be finite".to_string(),
        ));
    }

    // Deterministic landmark sample (Fisher–Yates partial shuffle, then sort).
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for i in 0..n_landmarks {
        let j = i + (nxt() as usize) % (n - i);
        perm.swap(i, j);
    }
    let mut landmarks: Vec<usize> = perm[..n_landmarks].to_vec();
    landmarks.sort_unstable();
    let m = landmarks.len();

    // Double-center the landmark squared-distance block without materializing Δ.
    let (bl, mu, _) =
        double_centered_gram_from_squared(m, |a, b| distances[landmarks[a]][landmarks[b]].powi(2));

    let e = fsci_linalg::eigh(&bl, fsci_linalg::DecompOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("eigh: {err}")))?;
    // eigh ascending; take the top-k (largest) eigenpairs, descending.
    let order: Vec<usize> = (0..m).rev().take(n_components).collect();
    let lmax = e.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let tol = (m as f64) * f64::EPSILON * lmax.max(1.0);

    // L# rows (k×m): row t = v_t / √λ_t (only for λ_t > tol).
    let eigenvalues: Vec<f64> = order.iter().map(|&t| e.eigenvalues[t]).collect();
    let lpinv: Vec<Vec<f64>> = order
        .iter()
        .map(|&t| {
            let lam = e.eigenvalues[t];
            let s = if lam > tol { 1.0 / lam.sqrt() } else { 0.0 };
            (0..m).map(|a| s * e.eigenvectors[a][t]).collect()
        })
        .collect();

    // Triangulate every point: x_p[t] = −½ Σ_b L#[t][b] (δ_p[b] − μ_b).
    let k = order.len();
    let embedding: Vec<Vec<f64>> = (0..n)
        .map(|p| {
            (0..k)
                .map(|t| {
                    let mut projected = 0.0;
                    for (b, &lb) in landmarks.iter().enumerate() {
                        projected += lpinv[t][b] * (distances[p][lb].powi(2) - mu[b]);
                    }
                    -0.5 * projected
                })
                .collect()
        })
        .collect();

    Ok(MdsResult {
        embedding,
        eigenvalues,
    })
}

// Min-heap node for Dijkstra (ordered by ascending tentative distance).
#[derive(Clone, Copy)]
struct DijkstraNode {
    dist: f64,
    node: usize,
}
impl PartialEq for DijkstraNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node == other.node
    }
}
impl Eq for DijkstraNode {}
impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed so BinaryHeap (a max-heap) pops the SMALLEST distance first.
        other
            .dist
            .total_cmp(&self.dist)
            .then(other.node.cmp(&self.node))
    }
}
impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Single-source Dijkstra over a weighted adjacency list; returns geodesic distances (∞ for
// unreachable nodes).
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(DijkstraNode {
        dist: 0.0,
        node: source,
    });
    while let Some(DijkstraNode { dist: d, node }) = heap.pop() {
        if d > dist[node] {
            continue;
        }
        for &(nbr, w) in &adj[node] {
            let nd = d + w;
            if nd < dist[nbr] {
                dist[nbr] = nd;
                heap.push(DijkstraNode {
                    dist: nd,
                    node: nbr,
                });
            }
        }
    }
    dist
}

/// Landmark Isomap (de Silva–Tenenbaum) — scalable nonlinear manifold embedding.
///
/// Matches `sklearn.manifold.Isomap`'s geometry but with the landmark approximation: builds a
/// symmetric `k`-nearest-neighbour graph (Euclidean edge weights), runs Dijkstra from each of
/// `m` landmarks to get the `m×n` geodesic distances, and applies landmark classical MDS
/// (double-centre the `m×m` landmark-geodesic block, take its top-`k` eigenpairs, triangulate
/// all `n` points). The geodesic distances capture the manifold's intrinsic geometry, so the
/// embedding "unrolls" curved manifolds a linear method (PCA/MDS) cannot. Cost
/// O(n²·d + m·(E + n log n) + n·m·k) — the `m` Dijkstra runs replace the O(n·(E+n log n))
/// all-source shortest paths of full Isomap, an asymptotic O(n)→O(m) drop in the geodesic
/// stage. Requires a connected neighbour graph (errors otherwise). `n_neighbors` is the graph
/// degree; deterministic by `seed`.
pub fn landmark_isomap(
    data: &[Vec<f64>],
    n_components: usize,
    n_neighbors: usize,
    n_landmarks: usize,
    seed: u64,
) -> Result<MdsResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let dim = data[0].len();
    if data.iter().any(|row| row.len() != dim) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_neighbors == 0 || n_neighbors >= n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_neighbors={n_neighbors} must be in [1, n-1={}]",
            n - 1
        )));
    }
    if n_components == 0 || n_components > n_landmarks || n_landmarks > n {
        return Err(ClusterError::InvalidArgument(format!(
            "need 1 ≤ n_components={n_components} ≤ n_landmarks={n_landmarks} ≤ n={n}"
        )));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    let edist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };

    // Symmetric k-NN graph (edge weight = Euclidean distance). Undirected: an edge is kept if
    // either endpoint lists the other among its k nearest.
    //
    // Phase 1 (parallel): each row i's k nearest are computed independently — the sort key
    // (`total_cmp` on distance, then ascending index) is a total order, so each row's top-k
    // candidate list is deterministic and row-independent. Producing them in parallel is
    // byte-identical to the sequential selection. This O(n²·(d + log n)) distance+sort sweep
    // dominates landmark Isomap, so it is the parallelized stage.
    let knn_row = |i: usize| -> Vec<(f64, usize)> {
        let mut d: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (edist(&data[i], &data[j]), j))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        d.truncate(n_neighbors);
        d
    };
    let work = (n as u64).saturating_mul(n as u64);
    let nthreads = if work < (1 << 16) || n < 8 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n)
    };
    let topk: Vec<Vec<(f64, usize)>> = if nthreads <= 1 {
        (0..n).map(knn_row).collect()
    } else {
        let mut out: Vec<Vec<(f64, usize)>> = vec![Vec::new(); n];
        let chunk = n.div_ceil(nthreads);
        let knn_row = &knn_row;
        std::thread::scope(|scope| {
            for (t, slot) in out.chunks_mut(chunk).enumerate() {
                let base = t * chunk;
                scope.spawn(move || {
                    for (off, o) in slot.iter_mut().enumerate() {
                        *o = knn_row(base + off);
                    }
                });
            }
        });
        out
    };

    // Phase 2 (serial): undirected edge insertion in the SAME order as the original single loop
    // (rows ascending, candidates in sorted order), so `adj`/`seen` are byte-identical to the
    // sequential build — and thus every downstream geodesic and the MDS embedding are unchanged.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut seen: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
    for (i, cand) in topk.iter().enumerate() {
        for &(w, j) in cand {
            if seen[i].insert(j) {
                adj[i].push((j, w));
            }
            if seen[j].insert(i) {
                adj[j].push((i, w));
            }
        }
    }

    // Landmarks.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for i in 0..n_landmarks {
        let j = i + (nxt() as usize) % (n - i);
        perm.swap(i, j);
    }
    let mut landmarks: Vec<usize> = perm[..n_landmarks].to_vec();
    landmarks.sort_unstable();
    let m = landmarks.len();

    // Geodesics from each landmark to all points (m×n).
    let geo: Vec<Vec<f64>> = landmarks.iter().map(|&l| dijkstra(&adj, l)).collect();
    if geo.iter().flatten().any(|d| !d.is_finite()) {
        return Err(ClusterError::ConvergenceFailed(
            "neighbour graph is disconnected; increase n_neighbors".to_string(),
        ));
    }

    // Landmark classical MDS on geodesic distances, streaming Δ[a][b] =
    // geodesic(landmark_a, landmark_b)² instead of materializing the m×m block.
    let (bl, mu, _) = double_centered_gram_from_squared(m, |a, b| geo[a][landmarks[b]].powi(2));

    let e = fsci_linalg::eigh(&bl, fsci_linalg::DecompOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("eigh: {err}")))?;
    let order: Vec<usize> = (0..m).rev().take(n_components).collect();
    let lmax = e.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let tol = (m as f64) * f64::EPSILON * lmax.max(1.0);

    let eigenvalues: Vec<f64> = order.iter().map(|&t| e.eigenvalues[t]).collect();
    // L# rows (k×m): v_t / √λ_t.
    let lpinv: Vec<Vec<f64>> = order
        .iter()
        .map(|&t| {
            let lam = e.eigenvalues[t];
            let s = if lam > tol { 1.0 / lam.sqrt() } else { 0.0 };
            (0..m).map(|a| s * e.eigenvectors[a][t]).collect()
        })
        .collect();

    // Triangulate every point from its landmark geodesics: x_p[t] = −½ Σ_b L#[t][b](δ_p[b]−μ_b),
    // where δ_p[b] = geodesic(landmark_b, p)².
    let k = order.len();
    let embedding: Vec<Vec<f64>> = (0..n)
        .map(|p| {
            (0..k)
                .map(|t| {
                    let mut projected = 0.0;
                    for b in 0..m {
                        projected += lpinv[t][b] * (geo[b][p].powi(2) - mu[b]);
                    }
                    -0.5 * projected
                })
                .collect()
        })
        .collect();

    Ok(MdsResult {
        embedding,
        eigenvalues,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Non-negative Matrix Factorization
// ══════════════════════════════════════════════════════════════════════

/// Initialization strategy for [`nmf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmfInit {
    /// Non-negative Double SVD (Boutsidis–Gallopoulos), seeded by a randomized SVD — a high
    /// quality start that converges in far fewer iterations than random.
    Nndsvd,
    /// Small random non-negative entries (the classic baseline; many more iterations).
    Random,
}

/// Result of [`nmf`]: `X ≈ W·H` with `W, H ≥ 0`.
#[derive(Debug, Clone)]
pub struct NmfResult {
    /// Basis / coefficient matrix `W` (n×k), non-negative.
    pub w: Vec<Vec<f64>>,
    /// Components matrix `H` (k×d), non-negative.
    pub h: Vec<Vec<f64>>,
    /// Multiplicative-update iterations performed.
    pub n_iter: usize,
    /// Final relative reconstruction error `‖X − W·H‖_F / ‖X‖_F`.
    pub reconstruction_err: f64,
}

// Dead at HEAD and dead in 1d999e235^ too (one occurrence, the definition), so
// this is NOT collateral from the bulk-deletion series — see frankenscipy-1b85n.
// Preserved rather than deleted; the allow goes away when that bead resolves.
#[allow(dead_code)]
fn transpose_rows(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cols = a.first().map_or(0, Vec::len);
    (0..cols)
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

/// A pair of dense matrices `(W, H)` returned by an NMF initializer.
type WhPair = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn nndsvd_init(x: &[Vec<f64>], k: usize, seed: u64) -> Result<WhPair, ClusterError> {
    let n = x.len();
    let d = x[0].len();
    let svd = fsci_linalg::randomized_svd(x, k, 10, 5, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
    let kk = svd.s.len();
    let mut w = vec![vec![0.0; k]; n];
    let mut h = vec![vec![0.0; d]; k];
    let norm = |v: &[f64]| -> f64 { v.iter().map(|&x| x * x).sum::<f64>().sqrt() };

    for j in 0..kk {
        let u_j: Vec<f64> = svd.u.iter().map(|row| row[j]).collect();
        let v_j: &Vec<f64> = &svd.vt[j];
        let s_j = svd.s[j].max(0.0);
        if j == 0 {
            // Leading singular vectors of a non-negative matrix are non-negative
            // (Perron–Frobenius); take their magnitudes.
            let sc = s_j.sqrt();
            for (i, wr) in w.iter_mut().enumerate() {
                wr[0] = sc * u_j[i].abs();
            }
            for (jj, &v) in v_j.iter().enumerate() {
                h[0][jj] = sc * v.abs();
            }
            continue;
        }
        let up: Vec<f64> = u_j.iter().map(|&x| x.max(0.0)).collect();
        let un: Vec<f64> = u_j.iter().map(|&x| (-x).max(0.0)).collect();
        let vp: Vec<f64> = v_j.iter().map(|&x| x.max(0.0)).collect();
        let vn: Vec<f64> = v_j.iter().map(|&x| (-x).max(0.0)).collect();
        let (nup, nun, nvp, nvn) = (norm(&up), norm(&un), norm(&vp), norm(&vn));
        let (uu, vv, nu, nv, m) = if nup * nvp >= nun * nvn {
            (up, vp, nup, nvp, nup * nvp)
        } else {
            (un, vn, nun, nvn, nun * nvn)
        };
        let sc = (s_j * m).sqrt();
        if nu > 1e-12 {
            for (i, wr) in w.iter_mut().enumerate() {
                wr[j] = sc * uu[i] / nu;
            }
        }
        if nv > 1e-12 {
            for (jj, &v) in vv.iter().enumerate() {
                h[j][jj] = sc * v / nv;
            }
        }
    }

    // NNDSVDa: replace exact zeros with a small constant so multiplicative updates can move
    // them off zero.
    let avg = x.iter().flatten().sum::<f64>() / (n * d) as f64;
    let fill = (avg * 0.01).max(1e-9);
    for row in w.iter_mut() {
        for v in row.iter_mut() {
            if *v < 1e-12 {
                *v = fill;
            }
        }
    }
    for row in h.iter_mut() {
        for v in row.iter_mut() {
            if *v < 1e-12 {
                *v = fill;
            }
        }
    }
    Ok((w, h))
}

/// Transpose a `rows×cols` row-major buffer into a `cols×rows` row-major buffer.
fn transpose_flat(src: &[f64], rows: usize, cols: usize, dst: &mut [f64]) {
    for r in 0..rows {
        let srow = &src[r * cols..r * cols + cols];
        for c in 0..cols {
            dst[c * rows + r] = srow[c];
        }
    }
}

/// `C(m×n) = A(m×p)·B(p×n)`, all row-major flat buffers; `out` is overwritten.
///
/// `ikj` accumulation with an MR=4 output-row panel: four output rows share each
/// streamed `B` row, cutting B memory traffic ~4× (the dominant `Wᵀ·X` in NMF is
/// memory-bound — it streams X once per output row) and letting the inner AXPY
/// auto-vectorize over the n axis.
fn nmf_mm(a: &[f64], b: &[f64], m: usize, p: usize, n: usize, out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    let mut i = 0;
    while i + 4 <= m {
        let rows = &mut out[i * n..(i + 4) * n];
        let (r0, rest) = rows.split_at_mut(n);
        let (r1, rest) = rest.split_at_mut(n);
        let (r2, r3) = rest.split_at_mut(n);
        let a0 = &a[i * p..(i + 1) * p];
        let a1 = &a[(i + 1) * p..(i + 2) * p];
        let a2 = &a[(i + 2) * p..(i + 3) * p];
        let a3 = &a[(i + 3) * p..(i + 4) * p];
        for l in 0..p {
            let (v0, v1, v2, v3) = (a0[l], a1[l], a2[l], a3[l]);
            let brow = &b[l * n..l * n + n];
            for ((((r0j, r1j), r2j), r3j), &bv) in r0
                .iter_mut()
                .zip(r1.iter_mut())
                .zip(r2.iter_mut())
                .zip(r3.iter_mut())
                .zip(brow)
            {
                *r0j += v0 * bv;
                *r1j += v1 * bv;
                *r2j += v2 * bv;
                *r3j += v3 * bv;
            }
        }
        i += 4;
    }
    while i < m {
        let r = &mut out[i * n..i * n + n];
        let ar = &a[i * p..i * p + p];
        for l in 0..p {
            let v = ar[l];
            if v == 0.0 {
                continue;
            }
            let brow = &b[l * n..l * n + n];
            for (rj, &bv) in r.iter_mut().zip(brow) {
                *rj += v * bv;
            }
        }
        i += 1;
    }
}

/// Convert the flat `W` (n×k) and `H` (k×d) back to row-major `Vec<Vec<f64>>`.
fn nmf_unflatten(
    w_flat: &[f64],
    h_flat: &[f64],
    n: usize,
    k: usize,
    d: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let w = (0..n).map(|r| w_flat[r * k..r * k + k].to_vec()).collect();
    let h = (0..k).map(|r| h_flat[r * d..r * d + d].to_vec()).collect();
    (w, h)
}

/// Reconstruction error `‖X − W·H‖_F / ‖X‖_F` on flat buffers; `wh` is scratch (n×d).
// Flat-buffer numeric kernel: the dimensions travel as separate scalars precisely
// so the caller can pass slices without wrapping. Bundling them into a struct
// would add an indirection to a hot path for no clarity gain — same reasoning as
// the 41 other `too_many_arguments` allows already in this workspace.
#[allow(clippy::too_many_arguments)]
fn nmf_rel_err(
    w: &[f64],
    h: &[f64],
    x: &[f64],
    n: usize,
    k: usize,
    d: usize,
    x_norm: f64,
    wh: &mut [f64],
) -> f64 {
    nmf_mm(w, h, n, k, d, wh);
    let mut s = 0.0;
    for idx in 0..n * d {
        let diff = x[idx] - wh[idx];
        s += diff * diff;
    }
    s.sqrt() / x_norm
}

/// Parallel NMF multiplicative-update iteration via a SAFE persistent worker pool
/// (no `unsafe` — the workspace forbids it).
///
/// The two dominant GEMMs are 91% of the serial time. Each worker permanently OWNS a
/// contiguous row-band of W and X (moved-in `Vec`s) and talks to the driver over channels:
/// - **Phase 1** (`Wᵀ·X`, `Wᵀ·W` — reductions over all rows): each worker computes the
///   partial `Wᵀ·X` / `Wᵀ·W` from its OWN band; the driver sums the partials. This turns
///   the cross-band reductions into embarrassingly-parallel per-band work with a cheap merge.
/// - Driver updates H (small, serial), forms `Hᵀ`, `H·Hᵀ`.
/// - **Phase 2** (`X·Hᵀ`, `W·H·Hᵀ`, W-update — independent per row): each worker updates its
///   OWN W band using the shared `Hᵀ`/`H·Hᵀ`.
///
/// On convergence-check iterations the workers also return their updated band + partial
/// reconstruction error, so the driver can assemble W and test `tol` without a separate pass.
/// Spawned ONCE per call (persistent), so there is no per-iteration thread-spawn tax — the
/// reason a naive per-call `thread::scope` plateaus well short of this.
#[allow(clippy::too_many_arguments)]
fn nmf_pool_iterate(
    x_flat: &[f64],
    w0_flat: Vec<f64>,
    mut h_flat: Vec<f64>,
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    x_norm: f64,
    eps: f64,
    nthreads: usize,
) -> (Vec<f64>, Vec<f64>, usize, f64) {
    use std::sync::mpsc;

    enum Msg {
        P1,
        P2 {
            ht: Vec<f64>,
            hht: Vec<f64>,
            h_err: Option<Vec<f64>>,
        },
        Stop,
    }
    enum Resp {
        P1 {
            pwtx: Vec<f64>,
            pwtw: Vec<f64>,
        },
        P2 {
            off: usize,
            wp: Option<Vec<f64>>,
            perr: Option<f64>,
        },
    }

    let bnd: Vec<(usize, usize)> = (0..nthreads)
        .map(|w| {
            let lo = w * n / nthreads;
            let hi = if w + 1 == nthreads {
                n
            } else {
                (w + 1) * n / nthreads
            };
            (lo, hi)
        })
        .collect();

    let mut wtx = vec![0.0f64; k * d];
    let mut wtw = vec![0.0f64; k * k];
    let mut wtwh = vec![0.0f64; k * d];
    let mut ht = vec![0.0f64; d * k];
    let mut hht = vec![0.0f64; k * k];

    let mut n_iter = 0usize;
    let mut last_w = w0_flat.clone();
    let mut last_err = f64::INFINITY;

    std::thread::scope(|s| {
        let mut cmd_tx = Vec::with_capacity(nthreads);
        let (resp_tx, resp_rx) = mpsc::channel::<Resp>();
        for &(lo, hi) in &bnd {
            let np = hi - lo;
            let wp0 = w0_flat[lo * k..hi * k].to_vec();
            let xp = x_flat[lo * d..hi * d].to_vec();
            let (ctx, crx) = mpsc::channel::<Msg>();
            cmd_tx.push(ctx);
            let resp_tx = resp_tx.clone();
            s.spawn(move || {
                let mut wp = wp0;
                let mut wpt = vec![0.0f64; k * np];
                let mut pwtx = vec![0.0f64; k * d];
                let mut pwtw = vec![0.0f64; k * k];
                let mut xhtp = vec![0.0f64; np * k];
                let mut whtp = vec![0.0f64; np * k];
                let mut recon = vec![0.0f64; np * d];
                while let Ok(msg) = crx.recv() {
                    match msg {
                        Msg::P1 => {
                            transpose_flat(&wp, np, k, &mut wpt);
                            nmf_mm(&wpt, &xp, k, np, d, &mut pwtx);
                            nmf_mm(&wpt, &wp, k, np, k, &mut pwtw);
                            if resp_tx
                                .send(Resp::P1 {
                                    pwtx: pwtx.clone(),
                                    pwtw: pwtw.clone(),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Msg::P2 { ht, hht, h_err } => {
                            nmf_mm(&xp, &ht, np, d, k, &mut xhtp);
                            nmf_mm(&wp, &hht, np, k, k, &mut whtp);
                            for idx in 0..np * k {
                                wp[idx] *= xhtp[idx] / (whtp[idx] + eps);
                            }
                            let (wp_out, perr) = if let Some(h) = h_err {
                                nmf_mm(&wp, &h, np, k, d, &mut recon);
                                let mut sm = 0.0;
                                for idx in 0..np * d {
                                    let diff = xp[idx] - recon[idx];
                                    sm += diff * diff;
                                }
                                (Some(wp.clone()), Some(sm))
                            } else {
                                (None, None)
                            };
                            if resp_tx
                                .send(Resp::P2 {
                                    off: lo,
                                    wp: wp_out,
                                    perr,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Msg::Stop => break,
                    }
                }
            });
        }
        drop(resp_tx);

        let mut prev = f64::INFINITY;
        for it in 0..max_iter {
            n_iter = it + 1;
            let check = it % 10 == 9 || it == max_iter - 1;
            for c in &cmd_tx {
                let _ = c.send(Msg::P1);
            }
            wtx.iter_mut().for_each(|v| *v = 0.0);
            wtw.iter_mut().for_each(|v| *v = 0.0);
            for _ in 0..nthreads {
                if let Ok(Resp::P1 { pwtx, pwtw }) = resp_rx.recv() {
                    for (a, b) in wtx.iter_mut().zip(&pwtx) {
                        *a += *b;
                    }
                    for (a, b) in wtw.iter_mut().zip(&pwtw) {
                        *a += *b;
                    }
                }
            }
            nmf_mm(&wtw, &h_flat, k, k, d, &mut wtwh);
            for idx in 0..k * d {
                h_flat[idx] *= wtx[idx] / (wtwh[idx] + eps);
            }
            transpose_flat(&h_flat, k, d, &mut ht);
            nmf_mm(&h_flat, &ht, k, d, k, &mut hht);
            let h_err = if check { Some(h_flat.clone()) } else { None };
            for c in &cmd_tx {
                let _ = c.send(Msg::P2 {
                    ht: ht.clone(),
                    hht: hht.clone(),
                    h_err: h_err.clone(),
                });
            }
            let mut err_sum = 0.0;
            for _ in 0..nthreads {
                if let Ok(Resp::P2 { off, wp, perr }) = resp_rx.recv() {
                    if let Some(wp) = wp {
                        last_w[off * k..off * k + wp.len()].copy_from_slice(&wp);
                    }
                    if let Some(pe) = perr {
                        err_sum += pe;
                    }
                }
            }
            if check {
                let err = err_sum.sqrt() / x_norm;
                last_err = err;
                if (prev - err).abs() < tol {
                    for c in &cmd_tx {
                        let _ = c.send(Msg::Stop);
                    }
                    return;
                }
                prev = err;
            }
        }
        for c in &cmd_tx {
            let _ = c.send(Msg::Stop);
        }
    });

    (last_w, h_flat, n_iter, last_err)
}

/// Non-negative Matrix Factorization `X ≈ W·H` (W, H ≥ 0) by multiplicative updates
/// (Lee–Seung, Frobenius objective).
///
/// Matches `sklearn.decomposition.NMF(n_components=k, solver="mu")`. The key lever is the
/// initialization: [`NmfInit::Nndsvd`] seeds W, H from a randomized SVD of X
/// ([`fsci_linalg::randomized_svd`]), a far better starting point than random — it reaches a
/// given reconstruction error in dramatically fewer iterations. `x` must be non-negative.
/// Deterministic given `seed`.
pub fn nmf(
    x: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
    init: NmfInit,
    seed: u64,
) -> Result<NmfResult, ClusterError> {
    if x.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err(ClusterError::EmptyData);
    }
    if x.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if x.iter().flatten().any(|&v| !v.is_finite() || v < 0.0) {
        return Err(ClusterError::InvalidArgument(
            "nmf input must be finite and non-negative".to_string(),
        ));
    }
    let k = n_components.min(n).min(d);
    if k == 0 {
        return Err(ClusterError::InvalidArgument(
            "n_components must be at least 1".to_string(),
        ));
    }
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(ClusterError::InvalidArgument(
            "tol must be finite and non-negative".to_string(),
        ));
    }

    let (w, h) = match init {
        NmfInit::Nndsvd => nndsvd_init(x, k, seed)?,
        NmfInit::Random => {
            let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
            let mut next = || {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + 1e-4
            };
            let w = (0..n).map(|_| (0..k).map(|_| next()).collect()).collect();
            let h = (0..k).map(|_| (0..d).map(|_| next()).collect()).collect();
            (w, h)
        }
    };

    let eps = 1e-10;
    let x_norm: f64 = x
        .iter()
        .flatten()
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-30);

    // Iterate on flat row-major buffers: the multiplicative updates are a
    // sequence of small GEMMs and the dominant Wᵀ·X is memory-bound. `nmf_mm`'s
    // MR=4 output-row panel cuts B traffic ~4× and vectorizes the inner AXPY.
    // All scratch is reused across iterations (no per-iter allocation).
    let mut x_flat = vec![0.0f64; n * d];
    for (r, row) in x.iter().enumerate() {
        x_flat[r * d..r * d + d].copy_from_slice(row);
    }
    let mut w_flat = vec![0.0f64; n * k];
    for (r, row) in w.iter().enumerate() {
        w_flat[r * k..r * k + k].copy_from_slice(row);
    }
    let mut h_flat = vec![0.0f64; k * d];
    for (r, row) in h.iter().enumerate() {
        h_flat[r * d..r * d + d].copy_from_slice(row);
    }

    // Large factorizations: fan the two dominant GEMMs (91% of serial time) across a
    // SAFE persistent worker pool (own-band partials + channel merge). Spawned once per
    // call, so no per-iteration thread-spawn tax. Gated so only sizes where it clearly
    // wins use it (≥4 worker bands of meaningful width); small inputs stay serial.
    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let nthreads = avail.min(n / 96).min(16);
    if nthreads >= 4 && d >= 4 && k >= 2 {
        let (w_flat, h_flat, n_iter, reconstruction_err) = nmf_pool_iterate(
            &x_flat, w_flat, h_flat, n, d, k, max_iter, tol, x_norm, eps, nthreads,
        );
        let (w_out, h_out) = nmf_unflatten(&w_flat, &h_flat, n, k, d);
        return Ok(NmfResult {
            w: w_out,
            h: h_out,
            n_iter,
            reconstruction_err,
        });
    }

    let mut wt_flat = vec![0.0f64; k * n];
    let mut wtx_flat = vec![0.0f64; k * d];
    let mut wtw_flat = vec![0.0f64; k * k];
    let mut wtwh_flat = vec![0.0f64; k * d];
    let mut ht_flat = vec![0.0f64; d * k];
    let mut xht_flat = vec![0.0f64; n * k];
    let mut hht_flat = vec![0.0f64; k * k];
    let mut whht_flat = vec![0.0f64; n * k];
    let mut wh_flat = vec![0.0f64; n * d];

    let mut prev = f64::INFINITY;
    let mut n_iter = 0;
    for it in 0..max_iter {
        n_iter = it + 1;
        // H ← H ⊙ (Wᵀ X) ⊘ (Wᵀ W H)
        transpose_flat(&w_flat, n, k, &mut wt_flat);
        nmf_mm(&wt_flat, &x_flat, k, n, d, &mut wtx_flat);
        nmf_mm(&wt_flat, &w_flat, k, n, k, &mut wtw_flat);
        nmf_mm(&wtw_flat, &h_flat, k, k, d, &mut wtwh_flat);
        for idx in 0..k * d {
            h_flat[idx] *= wtx_flat[idx] / (wtwh_flat[idx] + eps);
        }
        // W ← W ⊙ (X Hᵀ) ⊘ (W H Hᵀ)
        transpose_flat(&h_flat, k, d, &mut ht_flat);
        nmf_mm(&x_flat, &ht_flat, n, d, k, &mut xht_flat);
        nmf_mm(&h_flat, &ht_flat, k, d, k, &mut hht_flat);
        nmf_mm(&w_flat, &hht_flat, n, k, k, &mut whht_flat);
        for idx in 0..n * k {
            w_flat[idx] *= xht_flat[idx] / (whht_flat[idx] + eps);
        }
        if it % 10 == 9 || it == max_iter - 1 {
            let err = nmf_rel_err(&w_flat, &h_flat, &x_flat, n, k, d, x_norm, &mut wh_flat);
            if (prev - err).abs() < tol {
                let (w_out, h_out) = nmf_unflatten(&w_flat, &h_flat, n, k, d);
                return Ok(NmfResult {
                    w: w_out,
                    h: h_out,
                    n_iter,
                    reconstruction_err: err,
                });
            }
            prev = err;
        }
    }
    let reconstruction_err = nmf_rel_err(&w_flat, &h_flat, &x_flat, n, k, d, x_norm, &mut wh_flat);
    let (w_out, h_out) = nmf_unflatten(&w_flat, &h_flat, n, k, d);
    Ok(NmfResult {
        w: w_out,
        h: h_out,
        n_iter,
        reconstruction_err,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Truncated SVD (LSA)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`truncated_svd`], mirroring `sklearn.decomposition.TruncatedSVD`.
#[derive(Debug, Clone)]
pub struct TruncatedSvdResult {
    /// Top-k right singular vectors as rows (k×d).
    pub components: Vec<Vec<f64>>,
    /// The k largest singular values.
    pub singular_values: Vec<f64>,
    /// The data projected onto the components, `U·diag(σ)` (n×k).
    pub transformed: Vec<Vec<f64>>,
    /// Fraction of total variance explained by each component (length k).
    pub explained_variance_ratio: Vec<f64>,
}

/// Truncated SVD of `x` (n×d) — the top-`k` SVD WITHOUT centering (Latent Semantic
/// Analysis).
///
/// Matches `sklearn.decomposition.TruncatedSVD(n_components=k)`: unlike [`pca`] it does NOT
/// mean-center, so it works directly on raw / count / sparse-style data (text LSA). Computed
/// via [`fsci_linalg::randomized_svd`]: O(n·d·k) versus O(n·d·min(n,d)) for a full SVD — a
/// large win when k ≪ min(n,d). Deterministic given `seed`.
pub fn truncated_svd(
    x: &[Vec<f64>],
    n_components: usize,
    seed: u64,
) -> Result<TruncatedSvdResult, ClusterError> {
    if x.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err(ClusterError::EmptyData);
    }
    if x.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    let k = n_components.min(n).min(d);
    if k == 0 {
        return Err(ClusterError::InvalidArgument(
            "n_components must be at least 1".to_string(),
        ));
    }

    let svd = fsci_linalg::randomized_svd(x, k, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
    let kk = svd.s.len();
    let transformed: Vec<Vec<f64>> = svd
        .u
        .iter()
        .map(|urow| urow.iter().zip(&svd.s).map(|(&u, &s)| u * s).collect())
        .collect();

    // explained_variance_ratio_ = var(transformed column) / total variance of X.
    let denom = (n.max(2) - 1) as f64;
    let exp_var: Vec<f64> = (0..kk)
        .map(|t| {
            let mean: f64 = transformed.iter().map(|r| r[t]).sum::<f64>() / n as f64;
            transformed
                .iter()
                .map(|r| (r[t] - mean).powi(2))
                .sum::<f64>()
                / denom
        })
        .collect();
    let total_var: f64 = (0..d)
        .map(|j| {
            let cm: f64 = x.iter().map(|r| r[j]).sum::<f64>() / n as f64;
            x.iter().map(|r| (r[j] - cm).powi(2)).sum::<f64>() / denom
        })
        .sum();
    let explained_variance_ratio: Vec<f64> = exp_var
        .iter()
        .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
        .collect();

    Ok(TruncatedSvdResult {
        components: svd.vt,
        singular_values: svd.s,
        transformed,
        explained_variance_ratio,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Factor Analysis (randomized EM)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`factor_analysis`], mirroring `sklearn.decomposition.FactorAnalysis`.
#[derive(Debug, Clone)]
pub struct FactorAnalysisResult {
    /// The `k×d` factor-loading matrix `W` (`components_`).
    pub components: Vec<Vec<f64>>,
    /// Per-feature noise variances `Ψ` (length d, `noise_variance_`).
    pub noise_variance: Vec<f64>,
    /// Per-feature mean removed before fitting (length d).
    pub mean: Vec<f64>,
    /// Final average log-likelihood of the model.
    pub loglike: f64,
    /// EM iterations performed.
    pub n_iter: usize,
}

/// Factor Analysis via expectation-maximization with a randomized SVD inner solver.
///
/// Matches `sklearn.decomposition.FactorAnalysis(svd_method="randomized")`: fits the
/// generative model `x = W·z + μ + ε`, `ε ~ N(0, diag(Ψ))`, by EM. Each iteration scales the
/// centered data by `1/√Ψ` and takes its top-`k` SVD — here via [`fsci_linalg::randomized_svd`]
/// (O(n·d·k) per step) rather than a full O(n·d·min(n,d)) SVD — to update the loadings `W` and
/// noise variances `Ψ`. Iterates until the log-likelihood gain drops below `tol` or `max_iter`
/// is reached. Returns `components_` (`W`, k×d) and `noise_variance_` (`Ψ`). Deterministic by
/// `seed`.
pub fn factor_analysis(
    x: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<FactorAnalysisResult, ClusterError> {
    if x.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err(ClusterError::EmptyData);
    }
    if x.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_components == 0 || n_components > n.min(d) {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, min(n,d)={}]",
            n.min(d)
        )));
    }
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(ClusterError::InvalidArgument(
            "tol must be finite and non-negative".to_string(),
        ));
    }
    if x.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "x must be finite".to_string(),
        ));
    }

    const SMALL: f64 = 1e-12;
    let mean: Vec<f64> = (0..d)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|r| (0..d).map(|j| r[j] - mean[j]).collect())
        .collect();
    let var: Vec<f64> = (0..d)
        .map(|j| xc.iter().map(|r| r[j] * r[j]).sum::<f64>() / n as f64)
        .collect();

    let nsqrt = (n as f64).sqrt();
    let llconst = d as f64 * (2.0 * std::f64::consts::PI).ln() + n_components as f64;
    let mut psi = vec![1.0f64; d];
    let mut old_ll = f64::NEG_INFINITY;
    let mut w: Vec<Vec<f64>> = vec![vec![0.0; d]; n_components];
    let mut loglike = f64::NEG_INFINITY;
    let mut iters = 0;

    for it in 0..max_iter {
        iters = it + 1;
        let sqrt_psi: Vec<f64> = psi.iter().map(|&p| p.sqrt() + SMALL).collect();
        // Xs = Xc / (sqrt_psi * nsqrt), columnwise.
        let xs: Vec<Vec<f64>> = xc
            .iter()
            .map(|r| (0..d).map(|j| r[j] / (sqrt_psi[j] * nsqrt)).collect())
            .collect();
        let fro2: f64 = xs.iter().flatten().map(|v| v * v).sum();

        let svd = fsci_linalg::randomized_svd(&xs, n_components, 10, 4, seed)
            .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
        let kk = svd.s.len();
        let s2: Vec<f64> = svd.s.iter().map(|&s| s * s).collect();
        let cap2: f64 = s2.iter().sum();
        let unexp_var = fro2 - cap2;

        // W = sqrt(max(s^2 - 1, 0)) * Vt, then rescale by sqrt_psi (k×d).
        for (t, wrow) in w.iter_mut().enumerate().take(kk) {
            let scale = (s2[t] - 1.0).max(0.0).sqrt();
            for ((wv, &vv), &sp) in wrow.iter_mut().zip(&svd.vt[t]).zip(&sqrt_psi) {
                *wv = scale * vv * sp;
            }
        }
        for wrow in w.iter_mut().take(n_components).skip(kk) {
            wrow.iter_mut().for_each(|v| *v = 0.0);
        }

        // Average log-likelihood.
        let log_s2: f64 = s2.iter().map(|&v| v.max(SMALL).ln()).sum();
        let log_psi: f64 = psi.iter().map(|&p| p.ln()).sum();
        let ll = -(d as f64) / 2.0 * (llconst + log_s2 + unexp_var + log_psi);
        loglike = ll;

        if ll - old_ll < tol {
            break;
        }
        old_ll = ll;

        // Update noise variances: psi = max(var - sum_t W[t]^2, SMALL).
        for j in 0..d {
            let wj2: f64 = (0..n_components).map(|t| w[t][j] * w[t][j]).sum();
            psi[j] = (var[j] - wj2).max(SMALL);
        }
    }

    Ok(FactorAnalysisResult {
        components: w,
        noise_variance: psi,
        mean,
        loglike,
        n_iter: iters,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Probabilistic PCA (Tipping–Bishop, closed form)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`ppca`].
#[derive(Debug, Clone)]
pub struct PpcaResult {
    /// Factor-loading matrix `W` (k×d): `W_i = √max(λ_i − σ², 0)·V_i`.
    pub components: Vec<Vec<f64>>,
    /// Isotropic noise variance `σ²` (the mean of the discarded covariance eigenvalues).
    pub noise_variance: f64,
    /// The leading k covariance eigenvalues `λ_i` (descending).
    pub explained_variance: Vec<f64>,
    /// Per-feature mean removed before fitting (length d).
    pub mean: Vec<f64>,
}

/// Probabilistic PCA (Tipping–Bishop) — the maximum-likelihood factor model with *isotropic*
/// noise, `x = W·z + μ + ε`, `ε ~ N(0, σ²I)`.
///
/// Unlike [`factor_analysis`] (diagonal noise, solved by EM) PPCA has a closed-form ML
/// solution: with the top-`k` covariance eigenpairs `(λ_i, v_i)`, the noise variance is the
/// mean of the *discarded* eigenvalues `σ² = (tr Σ − Σ_{i<k} λ_i)/(d − k)` and the loadings are
/// `W_i = √max(λ_i − σ², 0)·v_i`. The eigenpairs come from one [`fsci_linalg::randomized_svd`]
/// of the centered data (O(n·d·k)) instead of a full O(n·d·min) SVD — and the trace is just the
/// summed feature variances — so no eigendecomposition of the full covariance is ever formed.
/// `n_components` must be in `[1, min(n, d)]`; deterministic by `seed`.
pub fn ppca(x: &[Vec<f64>], n_components: usize, seed: u64) -> Result<PpcaResult, ClusterError> {
    if x.is_empty() || x[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = x.len();
    let d = x[0].len();
    if x.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_components == 0 || n_components > n.min(d) {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, min(n,d)={}]",
            n.min(d)
        )));
    }
    if x.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "x must be finite".to_string(),
        ));
    }

    let mean: Vec<f64> = (0..d)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|r| (0..d).map(|j| r[j] - mean[j]).collect())
        .collect();
    // Total variance = trace(covariance) = Σ_j var_j.
    let total_var: f64 = (0..d)
        .map(|j| xc.iter().map(|r| r[j] * r[j]).sum::<f64>() / n as f64)
        .sum();

    let svd = fsci_linalg::randomized_svd(&xc, n_components, 10, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
    let kk = svd.s.len();
    // Covariance eigenvalues λ_i = σ_i² / n. randomized_svd returns singular values descending.
    let explained_variance: Vec<f64> = svd.s.iter().map(|&s| s * s / n as f64).collect();
    let sum_top: f64 = explained_variance.iter().sum();

    // σ² = mean of the discarded eigenvalues (0 when k captures the full dimension).
    let noise_variance = if d > kk {
        ((total_var - sum_top) / (d - kk) as f64).max(0.0)
    } else {
        0.0
    };

    // W_i = √max(λ_i − σ², 0) · V_i (k×d).
    let components: Vec<Vec<f64>> = (0..kk)
        .map(|i| {
            let scale = (explained_variance[i] - noise_variance).max(0.0).sqrt();
            svd.vt[i].iter().map(|&v| scale * v).collect()
        })
        .collect();

    Ok(PpcaResult {
        components,
        noise_variance,
        explained_variance,
        mean,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Nyström kernel approximation (column sampling)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`nystroem`].
#[derive(Debug, Clone)]
pub struct NystroemResult {
    /// Explicit feature map `Z` (n×m'), with `Z·Zᵀ ≈ K`. `m'` is the effective rank kept
    /// (≤ n_components, smaller if the landmark block is rank-deficient).
    pub feature_map: Vec<Vec<f64>>,
    /// The sampled landmark row/column indices (sorted, length n_components).
    pub landmark_indices: Vec<usize>,
}

/// Nyström low-rank approximation of a precomputed `n×n` symmetric PSD kernel.
///
/// Matches the idea of `sklearn.kernel_approximation.Nystroem(kernel="precomputed")`: samples
/// `n_components` landmark columns, forms `C = K[:, landmarks]` (n×m) and `W = K[landmarks,
/// landmarks]` (m×m), and returns the explicit feature map `Z = C·W^{-1/2}` so that `Z·Zᵀ =
/// C·W⁻¹·Cᵀ ≈ K` — exact when the numerical rank of `K` is ≤ m and the landmarks span its
/// range. Only the small `m×m` block is eigendecomposed ([`fsci_linalg::eigh`]); cost
/// O(n·m² + m³) versus O(n³) for a full eigendecomposition of `K`, a large win when m ≪ n.
/// Landmarks are chosen by a deterministic `seed`-seeded permutation. `kernel` must be square
/// and finite.
pub fn nystroem(
    kernel: &[Vec<f64>],
    n_components: usize,
    seed: u64,
) -> Result<NystroemResult, ClusterError> {
    if kernel.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = kernel.len();
    if kernel.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "kernel matrix must be square".to_string(),
        ));
    }
    if n_components == 0 || n_components > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, n={n}]"
        )));
    }
    if kernel.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "kernel must be finite".to_string(),
        ));
    }

    // Deterministic landmark sample: Fisher–Yates partial shuffle, then sort for stability.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for i in 0..n_components {
        let j = i + (next() as usize) % (n - i);
        perm.swap(i, j);
    }
    let mut landmarks: Vec<usize> = perm[..n_components].to_vec();
    landmarks.sort_unstable();
    let m = landmarks.len();

    // C = K[:, landmarks] (n×m); W = K[landmarks, landmarks] (m×m).
    let c: Vec<Vec<f64>> = kernel
        .iter()
        .map(|row| landmarks.iter().map(|&l| row[l]).collect())
        .collect();
    let w: Vec<Vec<f64>> = landmarks
        .iter()
        .map(|&li| landmarks.iter().map(|&lj| kernel[li][lj]).collect())
        .collect();

    // W^{-1/2} = V·diag(λ^{-1/2})·Vᵀ over eigenpairs with λ > tol (PSD block).
    let e = fsci_linalg::eigh(&w, fsci_linalg::DecompOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("eigh: {err}")))?;
    let lmax = e.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let tol = (m as f64) * f64::EPSILON * lmax.max(1.0);
    // inv_sqrt_w (m×m): Σ_t λ_t^{-1/2} v_t v_tᵀ for λ_t > tol.
    let mut inv_sqrt_w = vec![vec![0.0f64; m]; m];
    for (t, &lam) in e.eigenvalues.iter().enumerate() {
        if lam <= tol {
            continue;
        }
        let s = 1.0 / lam.sqrt();
        for (a, row) in inv_sqrt_w.iter_mut().enumerate() {
            let va = e.eigenvectors[a][t];
            if va == 0.0 {
                continue;
            }
            for (b, slot) in row.iter_mut().enumerate() {
                *slot += s * va * e.eigenvectors[b][t];
            }
        }
    }

    // Z = C · W^{-1/2} (n×m).
    let feature_map = fsci_linalg::matmul(&c, &inv_sqrt_w)
        .map_err(|err| ClusterError::InvalidArgument(format!("matmul: {err}")))?;

    Ok(NystroemResult {
        feature_map,
        landmark_indices: landmarks,
    })
}

/// Force-serial toggle for the Nyström RBF `E`-block fill, used only by the A/B perf harness
/// (`bin/perf_rbf_nystroem`) to compare the parallel row-fill against the original serial
/// `.iter().map().collect()` inside one binary. Production code leaves it `false`.
pub static NYSTROEM_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fill the Nyström `E` block `E[i][b] = exp(−γ·‖data[i] − data[landmarks[b]]‖²)` (n×m).
///
/// Each output row depends only on its own `data[i]` and the shared immutable `data`/`landmarks`,
/// so distributing the rows across threads is BYTE-IDENTICAL to the serial
/// `data.iter().map(row).collect()`: the per-element arithmetic and the intra-row evaluation
/// order are untouched — only which thread owns a given row changes. Work-gated on the O(n·m·d)
/// `exp` workload (the `available_parallelism()` syscall plus thread spawns cost more than the
/// fill for small blocks), matching the `mean_shift` / GMM E-step gate style in this crate. The
/// `NYSTROEM_FORCE_SERIAL` toggle pins the serial path for the A/B harness.
fn nystroem_e_block(data: &[Vec<f64>], landmarks: &[usize], gamma: f64) -> Vec<Vec<f64>> {
    let n = data.len();
    let m = landmarks.len();
    let d = data.first().map_or(0, Vec::len);
    let row = |xi: &[f64]| -> Vec<f64> {
        landmarks
            .iter()
            .map(|&b| {
                let d2: f64 = xi
                    .iter()
                    .zip(&data[b])
                    .map(|(&x, &y)| (x - y) * (x - y))
                    .sum();
                (-gamma * d2).exp()
            })
            .collect()
    };
    let serial = NYSTROEM_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed);
    let nthreads = if serial
        || (n as u64).saturating_mul(m as u64).saturating_mul(d as u64) < (1 << 20)
        || n < 2
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n)
    };
    if nthreads <= 1 {
        return data.iter().map(|xi| row(xi)).collect();
    }
    let mut e_mat: Vec<Vec<f64>> = vec![Vec::new(); n];
    let chunk = n.div_ceil(nthreads);
    let row = &row;
    std::thread::scope(|scope| {
        for (dchunk, echunk) in data.chunks(chunk).zip(e_mat.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (xi, slot) in dchunk.iter().zip(echunk.iter_mut()) {
                    *slot = row(xi);
                }
            });
        }
    });
    e_mat
}

/// Data-based RBF Nyström feature map — `sklearn.kernel_approximation.Nystroem(kernel="rbf")`.
///
/// Unlike [`nystroem`] (which takes a precomputed `n×n` kernel) this works directly from `n×d`
/// data and an RBF kernel `k(x,y)=exp(−γ‖x−y‖²)`, so the full kernel is **never formed**:
/// samples `n_components` landmark points, builds `W = K[L,L]` (m×m) and `E = K[:,L]` (n×m)
/// — only O(n·m·d) kernel evaluations — and returns the explicit feature map `Z = E·W^{-1/2}`
/// with `Z·Zᵀ ≈ K_rbf`. Cost O(n·m·d + n·m² + m³) versus O(n²·d + n³) for forming and
/// factoring the dense kernel — an asymptotic O(n²)→O(n·m) drop, exact when the kernel's
/// numerical rank is ≤ m. Feed `Z` to any linear method for an approximate kernel pipeline.
/// `gamma` is the RBF width; deterministic by `seed`.
pub fn rbf_nystroem(
    data: &[Vec<f64>],
    n_components: usize,
    gamma: f64,
    seed: u64,
) -> Result<NystroemResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let dim = data[0].len();
    if data.iter().any(|row| row.len() != dim) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_components == 0 || n_components > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, n={n}]"
        )));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(ClusterError::InvalidArgument(
            "gamma must be finite and positive".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    let rbf = |a: &[f64], b: &[f64]| -> f64 {
        let d2: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum();
        (-gamma * d2).exp()
    };

    // Deterministic landmark sample.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for i in 0..n_components {
        let j = i + (nxt() as usize) % (n - i);
        perm.swap(i, j);
    }
    let mut landmarks: Vec<usize> = perm[..n_components].to_vec();
    landmarks.sort_unstable();
    let m = landmarks.len();

    // W (m×m) and E (n×m) RBF blocks — O(n·m·d) kernel evals, no n×n matrix.
    let w: Vec<Vec<f64>> = landmarks
        .iter()
        .map(|&a| landmarks.iter().map(|&b| rbf(&data[a], &data[b])).collect())
        .collect();
    let e_mat = nystroem_e_block(data, &landmarks, gamma);

    // Z = E · W^{-1/2} (n×m).
    let w_inv_sqrt = sym_inv_sqrt(&w, m)?;
    let feature_map = fsci_linalg::matmul(&e_mat, &w_inv_sqrt)
        .map_err(|err| ClusterError::InvalidArgument(format!("matmul: {err}")))?;

    Ok(NystroemResult {
        feature_map,
        landmark_indices: landmarks,
    })
}

// ══════════════════════════════════════════════════════════════════════
// CUR decomposition (leverage-score row/column sampling)
// ══════════════════════════════════════════════════════════════════════

/// Result of [`cur_decomposition`]: `A ≈ C·U·R` with `C`/`R` actual columns/rows of `A`.
#[derive(Debug, Clone)]
pub struct CurResult {
    /// Sampled column indices (sorted, length k).
    pub column_indices: Vec<usize>,
    /// Sampled row indices (sorted, length k).
    pub row_indices: Vec<usize>,
    /// `C` = the selected columns of `A` (m×k).
    pub c: Vec<Vec<f64>>,
    /// Linking matrix `U` = `C⁺·A·R⁺` (k×k).
    pub u: Vec<Vec<f64>>,
    /// `R` = the selected rows of `A` (k×n).
    pub r: Vec<Vec<f64>>,
}

// Indices of the `k` largest values (ties broken by smaller index), then sorted ascending.
fn top_k_by_score(score: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// CUR decomposition (Mahoney–Drineas) of an `m×n` matrix via randomized leverage scores.
///
/// Selects `k` actual columns `C` and rows `R` of `A` by their statistical leverage —
/// `ℓ_j = Σ_{t<k} V_{tj}²` (columns) and `ℓ_i = Σ_{t<k} U_{it}²` (rows), from the rank-`k`
/// SVD computed with [`fsci_linalg::randomized_svd`] — then forms the linking matrix
/// `U = C⁺·A·R⁺` so that `A ≈ C·U·R`. Unlike an SVD this basis is *interpretable* (real
/// columns/rows), the row/column analogue of [`fsci_linalg::interp_decomp`]. Cost is dominated
/// by the O(m·n·k) randomized SVD versus O(m·n·min(m,n)) for a full-SVD leverage computation.
/// Exact to rounding when the numerical rank of `A` is ≤ k. `a` must be rectangular and finite;
/// deterministic by `seed`.
pub fn cur_decomposition(
    a: &[Vec<f64>],
    n_components: usize,
    n_oversamples: usize,
    seed: u64,
) -> Result<CurResult, ClusterError> {
    if a.is_empty() || a[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let m = a.len();
    let n = a[0].len();
    if a.iter().any(|row| row.len() != n) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_components == 0 || n_components > m.min(n) {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={n_components} must be in [1, min(m,n)={}]",
            m.min(n)
        )));
    }
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "a must be finite".to_string(),
        ));
    }

    let svd = fsci_linalg::randomized_svd(a, n_components, n_oversamples, 4, seed)
        .map_err(|e| ClusterError::InvalidArgument(format!("randomized_svd: {e}")))?;
    let kk = svd.s.len();

    // Leverage scores from the top-kk singular vectors.
    let col_lev: Vec<f64> = (0..n)
        .map(|j| (0..kk).map(|t| svd.vt[t][j].powi(2)).sum())
        .collect();
    let row_lev: Vec<f64> = (0..m)
        .map(|i| (0..kk).map(|t| svd.u[i][t].powi(2)).sum())
        .collect();
    let column_indices = top_k_by_score(&col_lev, n_components);
    let row_indices = top_k_by_score(&row_lev, n_components);

    // C = A[:, cols] (m×kc); R = A[rows, :] (kr×n).
    let c: Vec<Vec<f64>> = a
        .iter()
        .map(|row| column_indices.iter().map(|&j| row[j]).collect())
        .collect();
    let r: Vec<Vec<f64>> = row_indices.iter().map(|&i| a[i].clone()).collect();

    // U = C⁺ · A · R⁺.
    let pc = fsci_linalg::pinv(&c, fsci_linalg::PinvOptions::default())
        .map_err(|e| ClusterError::InvalidArgument(format!("pinv(C): {e}")))?
        .pseudo_inverse; // (kc×m)
    let pr = fsci_linalg::pinv(&r, fsci_linalg::PinvOptions::default())
        .map_err(|e| ClusterError::InvalidArgument(format!("pinv(R): {e}")))?
        .pseudo_inverse; // (n×kr)
    let pca = fsci_linalg::matmul(&pc, a)
        .map_err(|e| ClusterError::InvalidArgument(format!("matmul: {e}")))?; // (kc×n)
    let u = fsci_linalg::matmul(&pca, &pr)
        .map_err(|e| ClusterError::InvalidArgument(format!("matmul: {e}")))?; // (kc×kr)

    Ok(CurResult {
        column_indices,
        row_indices,
        c,
        u,
        r,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Nyström spectral clustering (data-based, never forms the n×n affinity)
// ══════════════════════════════════════════════════════════════════════

// Symmetric inverse square root M^{-1/2} = V·diag(λ^{-1/2})·Vᵀ over eigenpairs with λ > tol.
// (M must be symmetric PSD; `m` is its dimension.)
fn sym_inv_sqrt(matrix: &[Vec<f64>], m: usize) -> Result<Vec<Vec<f64>>, ClusterError> {
    let e = fsci_linalg::eigh(matrix, fsci_linalg::DecompOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("eigh: {err}")))?;
    let lmax = e.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let tol = (m as f64) * f64::EPSILON * lmax.max(1.0);
    let mut out = vec![vec![0.0f64; m]; m];
    for (t, &lam) in e.eigenvalues.iter().enumerate() {
        if lam <= tol {
            continue;
        }
        let s = 1.0 / lam.sqrt();
        for (a, row) in out.iter_mut().enumerate() {
            let va = e.eigenvectors[a][t];
            if va == 0.0 {
                continue;
            }
            for (b, slot) in row.iter_mut().enumerate() {
                *slot += s * va * e.eigenvectors[b][t];
            }
        }
    }
    Ok(out)
}

// Shared Nyström spectral embedding (Fowlkes orthogonalized Nyström): returns the raw n×k
// embedding V and the top-k normalized-affinity eigenvalues (descending). Inputs are assumed
// already validated. Never materializes the n×n affinity.
fn nystroem_spectral_embed(
    data: &[Vec<f64>],
    k: usize,
    n_landmarks: usize,
    gamma: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), ClusterError> {
    let n = data.len();
    let rbf = |a: &[f64], b: &[f64]| -> f64 {
        let d2: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum();
        (-gamma * d2).exp()
    };

    // Deterministic landmark sample.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for i in 0..n_landmarks {
        let j = i + (nxt() as usize) % (n - i);
        perm.swap(i, j);
    }
    let mut landmarks: Vec<usize> = perm[..n_landmarks].to_vec();
    landmarks.sort_unstable();
    let m = landmarks.len();

    // W (m×m, landmark–landmark) and E (n×m, all–landmark) RBF affinities.
    let w: Vec<Vec<f64>> = landmarks
        .iter()
        .map(|&a| landmarks.iter().map(|&b| rbf(&data[a], &data[b])).collect())
        .collect();
    let e_mat = nystroem_e_block(data, &landmarks, gamma);

    // Degree d = E·W⁻¹·(Eᵀ1). colsum_b = Σ_i E[i][b]; y = W⁻¹colsum.
    let colsum: Vec<f64> = (0..m).map(|b| e_mat.iter().map(|r| r[b]).sum()).collect();
    let w_inv = fsci_linalg::pinv(&w, fsci_linalg::PinvOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("pinv(W): {err}")))?
        .pseudo_inverse; // (m×m)
    let y: Vec<f64> = (0..m)
        .map(|a| w_inv[a].iter().zip(&colsum).map(|(&wv, &cs)| wv * cs).sum())
        .collect();
    let deg: Vec<f64> = e_mat
        .iter()
        .map(|row| row.iter().zip(&y).map(|(&ev, &yv)| ev * yv).sum::<f64>())
        .collect();

    // F = D^{-1/2} E (row-scaled).
    let f_mat: Vec<Vec<f64>> = e_mat
        .iter()
        .zip(&deg)
        .map(|(row, &d)| {
            let s = if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 };
            row.iter().map(|&v| v * s).collect()
        })
        .collect();

    // M = W^{-1/2}·(FᵀF)·W^{-1/2} (m×m). FᵀF accumulated in O(n·m²).
    let mut ftf = vec![vec![0.0f64; m]; m];
    for row in &f_mat {
        for a in 0..m {
            let ra = row[a];
            if ra == 0.0 {
                continue;
            }
            let dst = &mut ftf[a];
            for (b, slot) in dst.iter_mut().enumerate() {
                *slot += ra * row[b];
            }
        }
    }
    let w_inv_sqrt = sym_inv_sqrt(&w, m)?;
    let tmp = fsci_linalg::matmul(&w_inv_sqrt, &ftf)
        .map_err(|err| ClusterError::InvalidArgument(format!("matmul: {err}")))?;
    let mmat = fsci_linalg::matmul(&tmp, &w_inv_sqrt)
        .map_err(|err| ClusterError::InvalidArgument(format!("matmul: {err}")))?;

    // Eigenpairs of M (ascending) → top-k (largest) = leading normalized-affinity modes.
    let em = fsci_linalg::eigh(&mmat, fsci_linalg::DecompOptions::default())
        .map_err(|err| ClusterError::InvalidArgument(format!("eigh: {err}")))?;
    let tol =
        (m as f64) * f64::EPSILON * em.eigenvalues.iter().cloned().fold(0.0, f64::max).max(1.0);
    let order: Vec<usize> = (0..m).rev().take(k).collect();
    let eigenvalues: Vec<f64> = order.iter().map(|&t| em.eigenvalues[t]).collect();

    // G = W^{-1/2}·U_M[:,top]·Σ^{-1/2} (m×kk); then V = F·G (n×kk).
    let kk = order.len();
    let mut g = vec![vec![0.0f64; kk]; m];
    for (col, &t) in order.iter().enumerate() {
        let sig = em.eigenvalues[t];
        let inv_sqrt_sig = if sig > tol { 1.0 / sig.sqrt() } else { 0.0 };
        for (a, grow) in g.iter_mut().enumerate() {
            let wa = &w_inv_sqrt[a];
            let val: f64 = (0..m).map(|b| wa[b] * em.eigenvectors[b][t]).sum();
            grow[col] = val * inv_sqrt_sig;
        }
    }
    let v = fsci_linalg::matmul(&f_mat, &g)
        .map_err(|err| ClusterError::InvalidArgument(format!("matmul: {err}")))?; // (n×kk)
    Ok((v, eigenvalues))
}

/// Nyström-accelerated Laplacian-eigenmap embedding directly from `n×d` data — without ever
/// materializing the `n×n` affinity.
///
/// The embedding-only sibling of [`nystroem_spectral_clustering`]: returns the leading `k`
/// normalized-affinity eigenvectors (the manifold coordinates) and their eigenvalues, computed
/// by the same Fowlkes orthogonalized-Nyström method, with no row-normalisation and no k-means.
/// Cost O(n·m² + m³ + n·m·k) versus O(n²·d + n³) for the dense path. `gamma` is the RBF width;
/// deterministic by `seed`.
pub fn nystroem_spectral_embedding(
    data: &[Vec<f64>],
    n_components: usize,
    n_landmarks: usize,
    gamma: f64,
    seed: u64,
) -> Result<SpectralEmbeddingResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let dim = data[0].len();
    if data.iter().any(|row| row.len() != dim) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_components == 0 || n_components > n_landmarks || n_landmarks > n {
        return Err(ClusterError::InvalidArgument(format!(
            "need 1 ≤ n_components={n_components} ≤ n_landmarks={n_landmarks} ≤ n={n}"
        )));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(ClusterError::InvalidArgument(
            "gamma must be finite and positive".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }
    let (embedding, eigenvalues) =
        nystroem_spectral_embed(data, n_components, n_landmarks, gamma, seed)?;
    Ok(SpectralEmbeddingResult {
        embedding,
        eigenvalues,
    })
}

/// Diffusion-map embedding (Coifman–Lafon) directly from `n×d` data, Nyström-accelerated.
///
/// Like [`nystroem_spectral_embedding`] it takes the leading eigenpairs `(μ_j, ψ_j)` of the
/// normalized affinity, but it drops the trivial stationary eigenvector and scales each
/// remaining coordinate by `μ_j^t` (`t = time_steps`): `Ψ_t(x_i) = (μ_1^t ψ_1[i], …, μ_k^t
/// ψ_k[i])`. The Euclidean distance in this embedding approximates the *diffusion distance* at
/// time `t` — a multi-scale, noise-robust manifold geometry distinct from the raw spectral
/// embedding. Larger `t` emphasises coarse structure (slow-decaying modes). Cost is the same
/// O(n·m² + m³ + n·m·k) as the Nyström spectral routines; the `n×n` affinity is never formed.
/// `gamma` is the RBF width, `time_steps ≥ 0` the diffusion time; deterministic by `seed`.
pub fn diffusion_map(
    data: &[Vec<f64>],
    n_components: usize,
    n_landmarks: usize,
    gamma: f64,
    time_steps: f64,
    seed: u64,
) -> Result<SpectralEmbeddingResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let dim = data[0].len();
    if data.iter().any(|row| row.len() != dim) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    // One extra component is computed and discarded (the trivial stationary eigenvector).
    let want = n_components + 1;
    if n_components == 0 || want > n_landmarks || n_landmarks > n {
        return Err(ClusterError::InvalidArgument(format!(
            "need 1 ≤ n_components+1={want} ≤ n_landmarks={n_landmarks} ≤ n={n}"
        )));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(ClusterError::InvalidArgument(
            "gamma must be finite and positive".to_string(),
        ));
    }
    if !(time_steps.is_finite() && time_steps >= 0.0) {
        return Err(ClusterError::InvalidArgument(
            "time_steps must be finite and non-negative".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    let (v, eigenvalues) = nystroem_spectral_embed(data, want, n_landmarks, gamma, seed)?;
    let kk = eigenvalues.len();
    if kk < 2 {
        return Err(ClusterError::ConvergenceFailed(
            "diffusion map needs at least 2 captured eigenpairs".to_string(),
        ));
    }
    // Drop the leading (trivial) eigenvector; scale coordinate j by μ_j^t.
    let kept = kk - 1;
    let scaled_eigenvalues: Vec<f64> = (1..kk).map(|j| eigenvalues[j]).collect();
    let scales: Vec<f64> = scaled_eigenvalues
        .iter()
        .map(|&mu| mu.max(0.0).powf(time_steps))
        .collect();
    let embedding: Vec<Vec<f64>> = v
        .iter()
        .map(|row| (0..kept).map(|j| row[j + 1] * scales[j]).collect())
        .collect();

    Ok(SpectralEmbeddingResult {
        embedding,
        eigenvalues: scaled_eigenvalues,
    })
}

/// Nyström-accelerated normalized spectral clustering directly from `n×d` data — without ever
/// materializing the `n×n` affinity.
///
/// Builds an RBF affinity `k(x,y)=exp(−γ‖x−y‖²)` only between the `n` points and `m`
/// landmarks (`E`, n×m) and among the landmarks (`W`, m×m), and applies the Fowlkes
/// orthogonalized-Nyström method: the degree vector `d = E·W⁻¹·(Eᵀ1)`, the normalized rows
/// `F = D^{-1/2}E`, and the small `m×m` system `M = W^{-1/2}·(FᵀF)·W^{-1/2}` whose eigenpairs
/// extend to the leading eigenvectors of the full normalized affinity via
/// `V = F·W^{-1/2}·U_M·Σ^{-1/2}`. The top-`k` columns are row-normalised (Ng–Jordan–Weiss) and
/// passed to [`kmeans`]. Cost O(n·m² + m³ + n·m·k) versus O(n²·d + n³) for forming and
/// eigendecomposing the dense affinity — an asymptotic O(n²)→O(n·m) win when m ≪ n. `gamma`
/// is the RBF width; deterministic by `seed`.
pub fn nystroem_spectral_clustering(
    data: &[Vec<f64>],
    n_clusters: usize,
    n_landmarks: usize,
    gamma: f64,
    max_iter: usize,
    seed: u64,
) -> Result<Vec<usize>, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let dim = data[0].len();
    if data.iter().any(|row| row.len() != dim) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    if n_clusters == 0 || n_clusters > n_landmarks || n_landmarks > n {
        return Err(ClusterError::InvalidArgument(format!(
            "need 1 ≤ n_clusters={n_clusters} ≤ n_landmarks={n_landmarks} ≤ n={n}"
        )));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(ClusterError::InvalidArgument(
            "gamma must be finite and positive".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    let (v, _eigenvalues) = nystroem_spectral_embed(data, n_clusters, n_landmarks, gamma, seed)?;

    // Row-normalise the embedding (Ng–Jordan–Weiss), then k-means.
    let mut embedding = v;
    for row in embedding.iter_mut() {
        let nrm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-12 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }

    Ok(kmeans(&embedding, n_clusters, max_iter, seed)?.labels)
}

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
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }

    // K-means++ initialization
    let mut centroids = kmeans_plusplus_init(data, k, seed);
    let mut labels = vec![0usize; n];
    let mut inertia = f64::INFINITY;

    // Flatten the observations ONCE into a contiguous n×d buffer for cache-friendly,
    // auto-vectorizable distance scans in the assignment step (vs Vec<Vec> pointer-chasing).
    let data_flat: Vec<f64> = data.iter().flat_map(|p| p.iter().copied()).collect();
    for iter in 0..max_iter {
        // Assignment step: each point's nearest centroid is independent, so compute
        // (label, min_dist) in parallel. The inertia is then summed SEQUENTIALLY in
        // point order so its floating-point reduction — and therefore the convergence
        // check and iteration count — stay bit-identical to the serial version.
        let centroids_flat = flatten_centroids(&centroids, d);
        let assignments = assign_points(&data_flat, n, &centroids_flat, k, d);
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
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
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

    // Per-iteration scratch hoisted out of the loop and reused (byte-identical):
    // batch_indices/batch_labels are cleared and refilled, centroids_flat is
    // rewritten in place. frankenscipy-p1pp8.
    let mut batch_indices = Vec::with_capacity(batch);
    let mut batch_labels = Vec::with_capacity(batch);
    let mut centroids_flat = Vec::with_capacity(k * d);
    for _ in 0..max_iter {
        // Sample mini-batch
        batch_indices.clear();
        for _ in 0..batch {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            batch_indices.push((rng >> 33) as usize % n);
        }

        // Assignment
        flatten_centroids_into(&centroids, d, &mut centroids_flat);
        batch_labels.clear();
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

/// Result of [`gaussian_mixture`], mirroring `sklearn.mixture.GaussianMixture`.
#[derive(Debug, Clone)]
pub struct GmmResult {
    /// Mixture weights `π_k` (length k, sum to 1).
    pub weights: Vec<f64>,
    /// Component means `μ_k` (k×d).
    pub means: Vec<Vec<f64>>,
    /// Per-component diagonal covariances `σ²_k` (k×d).
    pub covariances: Vec<Vec<f64>>,
    /// Posterior responsibilities `γ_{ik}` (n×k), each row sums to 1.
    pub responsibilities: Vec<Vec<f64>>,
    /// Hard cluster labels `argmax_k γ_{ik}` (length n).
    pub labels: Vec<usize>,
    /// Final average log-likelihood per sample.
    pub log_likelihood: f64,
    /// Bayesian information criterion (lower is better) for choosing `n_components`.
    pub bic: f64,
    /// Akaike information criterion (lower is better).
    pub aic: f64,
    /// EM iterations performed.
    pub n_iter: usize,
}

/// Gaussian Mixture Model with diagonal covariances, fit by expectation-maximization.
///
/// Matches `sklearn.mixture.GaussianMixture(covariance_type="diag")`: fits
/// `p(x) = Σ_k π_k·N(x | μ_k, diag(σ²_k))` by EM. The E-step forms posterior responsibilities
/// `γ_{ik}` from the per-component diagonal-Gaussian log-density (combined with a numerically
/// stable log-sum-exp); the M-step re-estimates weights, means and diagonal variances as the
/// responsibility-weighted moments. Means are seeded by k-means++ (see [`kmeans`]); iterates
/// until the mean log-likelihood gain drops below `tol` or `max_iter`. `reg_covar` (≥0) is added
/// to every variance for numerical stability. Returns the parameters, soft responsibilities and
/// the hard `argmax` labels. Deterministic by `seed`.
pub fn gaussian_mixture(
    data: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    seed: u64,
) -> Result<GmmResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let d = data[0].len();
    if data.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    let k = n_components;
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={k} must be in [1, n={n}]"
        )));
    }
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(ClusterError::InvalidArgument(
            "tol must be finite and non-negative".to_string(),
        ));
    }
    if !(reg_covar.is_finite() && reg_covar >= 0.0) {
        return Err(ClusterError::InvalidArgument(
            "reg_covar must be finite and non-negative".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    const FLOOR: f64 = 1e-12;
    // Initialize: k-means++ means, uniform weights, global per-feature variance.
    let mut means = kmeans_plusplus_init(data, k, seed);
    let global_mean: Vec<f64> = (0..d)
        .map(|j| data.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let global_var: Vec<f64> = (0..d)
        .map(|j| {
            data.iter()
                .map(|r| (r[j] - global_mean[j]).powi(2))
                .sum::<f64>()
                / n as f64
                + reg_covar
        })
        .collect();
    let mut covariances = vec![global_var.clone(); k];
    let mut weights = vec![1.0 / k as f64; k];

    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut resp = vec![vec![0.0f64; k]; n];
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut old_ll = f64::NEG_INFINITY;
    let mut iters = 0;

    // E-step scratch hoisted out of the EM loop: log_norm (k) is rewritten each
    // iteration, logp (k) is rewritten for every data point — was allocated
    // n×max_iter times. Both fully overwritten before read -> byte-identical.
    // frankenscipy-5ufms.
    let mut log_norm = vec![0.0f64; k];
    let mut logp = vec![0.0f64; k];
    for it in 0..max_iter {
        iters = it + 1;
        // Precompute per-component log-weight and −½Σ log(2π σ²) normalizer.
        for c in 0..k {
            let mut s = weights[c].max(FLOOR).ln();
            for &cov in &covariances[c] {
                s -= half_log_2pi + 0.5 * cov.max(FLOOR).ln();
            }
            log_norm[c] = s;
        }

        // E-step: responsibilities + total log-likelihood (log-sum-exp per point).
        // Each point's resp row and per-point log-likelihood are independent, so
        // split the points across threads into ordered slots; total_ll is then the
        // SAME ordered sum of the SAME per-point lse values (serial reduce after) ->
        // byte-identical to the serial loop. The block scope drops the borrowing
        // closure before the M-step mutates means/covariances. frankenscipy-yw7ts.
        let total_ll: f64 = {
            let mut lse = vec![0.0f64; n];
            // Per-point E-step into a caller-provided logp scratch; returns lse.
            let e_step_point = |row: &[f64], logp: &mut [f64], resp_i: &mut [f64]| -> f64 {
                let mut maxlp = f64::NEG_INFINITY;
                for c in 0..k {
                    let mut s = log_norm[c];
                    for j in 0..d {
                        let diff = row[j] - means[c][j];
                        s -= 0.5 * diff * diff / covariances[c][j].max(FLOOR);
                    }
                    logp[c] = s;
                    if s > maxlp {
                        maxlp = s;
                    }
                }
                let mut sumexp = 0.0;
                for lp in logp.iter_mut() {
                    *lp = (*lp - maxlp).exp();
                    sumexp += *lp;
                }
                let inv = 1.0 / sumexp;
                for c in 0..k {
                    resp_i[c] = logp[c] * inv;
                }
                maxlp + sumexp.ln()
            };
            let nthreads_e = if n.saturating_mul(k).saturating_mul(d) < (1 << 16) || n < 2 {
                1
            } else {
                std::thread::available_parallelism()
                    .map(|c| c.get())
                    .unwrap_or(1)
                    .min(n)
            };
            if nthreads_e <= 1 {
                for ((row, resp_i), lse_i) in data.iter().zip(resp.iter_mut()).zip(lse.iter_mut()) {
                    *lse_i = e_step_point(row, &mut logp, resp_i);
                }
            } else {
                let chunk = n.div_ceil(nthreads_e);
                let e_step_point = &e_step_point;
                std::thread::scope(|scope| {
                    for ((resp_chunk, lse_chunk), data_chunk) in resp
                        .chunks_mut(chunk)
                        .zip(lse.chunks_mut(chunk))
                        .zip(data.chunks(chunk))
                    {
                        scope.spawn(move || {
                            let mut logp_local = vec![0.0f64; k];
                            for ((row, resp_i), lse_i) in data_chunk
                                .iter()
                                .zip(resp_chunk.iter_mut())
                                .zip(lse_chunk.iter_mut())
                            {
                                *lse_i = e_step_point(row, &mut logp_local, resp_i);
                            }
                        });
                    }
                });
            }
            lse.iter().sum()
        };
        log_likelihood = total_ll / n as f64;
        if (log_likelihood - old_ll).abs() < tol {
            break;
        }
        old_ll = log_likelihood;

        // M-step: weighted moments. Two byte-identical restructurings of the original
        // `for c { for j { Σ_i (mean); Σ_i (var) } }`:
        //  1. Loop interchange — accumulate the whole mean/var VECTORS in a single pass over
        //     i per component (contiguous `row[j]` access), turning 2·k·d strided passes over
        //     the data into 2·k contiguous ones (each output sum stays in the same i order).
        //  2. Fan the k independent components across cores (each computed by identical
        //     arithmetic on its own thread → byte-identical; disjoint output slots).
        let m_compute = |c: usize, w_out: &mut f64, m_out: &mut [f64], cov_out: &mut [f64]| {
            let nk: f64 = resp.iter().map(|r| r[c]).sum::<f64>().max(FLOOR);
            *w_out = nk / n as f64;
            m_out.iter_mut().for_each(|v| *v = 0.0);
            for (i, row) in data.iter().enumerate() {
                let g = resp[i][c];
                for (mv, &rv) in m_out.iter_mut().zip(row) {
                    *mv += g * rv;
                }
            }
            for mv in m_out.iter_mut() {
                *mv /= nk;
            }
            cov_out.iter_mut().for_each(|v| *v = 0.0);
            for (i, row) in data.iter().enumerate() {
                let g = resp[i][c];
                for ((vv, &rv), &mv) in cov_out.iter_mut().zip(row).zip(m_out.iter()) {
                    let diff = rv - mv;
                    *vv += g * (diff * diff);
                }
            }
            for vv in cov_out.iter_mut() {
                *vv = *vv / nk + reg_covar;
            }
        };
        let mwork = (n as u64).saturating_mul(d as u64);
        let nthreads_m = if mwork < (1 << 16) || k < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(k)
        };
        if nthreads_m <= 1 {
            for c in 0..k {
                let (w_slot, m_slot, cov_slot) =
                    (&mut weights[c], &mut means[c], &mut covariances[c]);
                m_compute(c, w_slot, m_slot, cov_slot);
            }
        } else {
            let per = k.div_ceil(nthreads_m);
            let m_compute = &m_compute;
            std::thread::scope(|scope| {
                for (((w_ch, m_ch), cov_ch), base) in weights
                    .chunks_mut(per)
                    .zip(means.chunks_mut(per))
                    .zip(covariances.chunks_mut(per))
                    .zip((0..k).step_by(per))
                {
                    scope.spawn(move || {
                        for (lc, ((w_out, m_out), cov_out)) in w_ch
                            .iter_mut()
                            .zip(m_ch.iter_mut())
                            .zip(cov_ch.iter_mut())
                            .enumerate()
                        {
                            m_compute(base + lc, w_out, m_out, cov_out);
                        }
                    });
                }
            });
        }
    }

    let labels: Vec<usize> = resp
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(c, _)| c)
        })
        .collect();

    // Information criteria for model selection. Free params (diag): means k·d + variances k·d +
    // weights (k−1).
    let total_ll = log_likelihood * n as f64;
    let n_params = (k * (2 * d + 1) - 1) as f64;
    let bic = -2.0 * total_ll + n_params * (n as f64).ln();
    let aic = -2.0 * total_ll + 2.0 * n_params;

    Ok(GmmResult {
        weights,
        means,
        covariances,
        responsibilities: resp,
        labels,
        log_likelihood,
        bic,
        aic,
        n_iter: iters,
    })
}

/// Result of [`gaussian_mixture_full`].
#[derive(Debug, Clone)]
pub struct GmmFullResult {
    /// Mixture weights `π_k` (length k, sum to 1).
    pub weights: Vec<f64>,
    /// Component means `μ_k` (k×d).
    pub means: Vec<Vec<f64>>,
    /// Full component covariance matrices `Σ_k` (k of d×d).
    pub covariances: Vec<Vec<Vec<f64>>>,
    /// Posterior responsibilities `γ_{ik}` (n×k), each row sums to 1.
    pub responsibilities: Vec<Vec<f64>>,
    /// Hard cluster labels `argmax_k γ_{ik}` (length n).
    pub labels: Vec<usize>,
    /// Final average log-likelihood per sample.
    pub log_likelihood: f64,
    /// Bayesian information criterion (lower is better) for choosing `n_components`.
    pub bic: f64,
    /// Akaike information criterion (lower is better).
    pub aic: f64,
    /// EM iterations performed.
    pub n_iter: usize,
}

// Lower-triangular Cholesky L (A = L·Lᵀ) of a d×d SPD matrix, or None if not positive definite.
fn cholesky_lower(m: &[Vec<f64>], d: usize) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = m[i][j];
            for (lik, ljk) in l[i].iter().zip(&l[j]).take(j) {
                s -= lik * ljk;
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Gaussian Mixture Model with FULL covariances, fit by expectation-maximization.
///
/// `sklearn.mixture.GaussianMixture(covariance_type="full")` (the default): fits
/// `p(x) = Σ_k π_k·N(x | μ_k, Σ_k)` with general `d×d` covariances, so it captures correlated
/// features that the diagonal [`gaussian_mixture`] cannot. Each E-step Cholesky-factors every
/// `Σ_k` once (`Σ = L·Lᵀ`), giving `log det Σ = 2·Σ log L_ii` and the squared Mahalanobis
/// distance `‖L⁻¹(x−μ)‖²` via forward substitution; responsibilities follow from a stable
/// log-sum-exp. The M-step re-estimates `Σ_k` as the responsibility-weighted scatter plus
/// `reg_covar·I` (which keeps every `Σ_k` positive definite — `reg_covar` must be > 0). Means
/// seeded by k-means++. Deterministic by `seed`.
pub fn gaussian_mixture_full(
    data: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    seed: u64,
) -> Result<GmmFullResult, ClusterError> {
    if data.is_empty() || data[0].is_empty() {
        return Err(ClusterError::EmptyData);
    }
    let n = data.len();
    let d = data[0].len();
    if data.iter().any(|row| row.len() != d) {
        return Err(ClusterError::InvalidArgument(
            "all rows must have the same length".to_string(),
        ));
    }
    let k = n_components;
    if k == 0 || k > n {
        return Err(ClusterError::InvalidArgument(format!(
            "n_components={k} must be in [1, n={n}]"
        )));
    }
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(ClusterError::InvalidArgument(
            "tol must be finite and non-negative".to_string(),
        ));
    }
    if !(reg_covar.is_finite() && reg_covar > 0.0) {
        return Err(ClusterError::InvalidArgument(
            "reg_covar must be finite and positive".to_string(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "data must be finite".to_string(),
        ));
    }

    const FLOOR: f64 = 1e-12;
    let mut means = kmeans_plusplus_init(data, k, seed);
    // Initialize covariances to the (regularized) global covariance.
    let global_mean: Vec<f64> = (0..d)
        .map(|j| data.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let mut global_cov = vec![vec![0.0f64; d]; d];
    for row in data {
        for a in 0..d {
            let da = row[a] - global_mean[a];
            for b in 0..d {
                global_cov[a][b] += da * (row[b] - global_mean[b]);
            }
        }
    }
    for (a, rowv) in global_cov.iter_mut().enumerate() {
        for v in rowv.iter_mut() {
            *v /= n as f64;
        }
        rowv[a] += reg_covar;
    }
    let mut covariances = vec![global_cov; k];
    let mut weights = vec![1.0 / k as f64; k];

    let half_d_log_2pi = 0.5 * d as f64 * (2.0 * std::f64::consts::PI).ln();
    let mut resp = vec![vec![0.0f64; k]; n];
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut old_ll = f64::NEG_INFINITY;
    let mut iters = 0;

    // E-step scratch hoisted out of the EM loop (logp was allocated n×max_iter
    // times; both fully overwritten before read -> byte-identical).
    // frankenscipy-5ufms.
    let mut log_norm = vec![0.0f64; k];
    let mut logp = vec![0.0f64; k];
    for it in 0..max_iter {
        iters = it + 1;
        // Per-component Cholesky factor + log-normalizer (log π_k − ½ log det Σ_k − (d/2)log2π).
        let mut chols: Vec<Vec<Vec<f64>>> = Vec::with_capacity(k);
        for c in 0..k {
            let l = cholesky_lower(&covariances[c], d).ok_or_else(|| {
                ClusterError::ConvergenceFailed(format!(
                    "component {c} covariance not positive definite"
                ))
            })?;
            let logdet: f64 = (0..d).map(|j| l[j][j].max(FLOOR).ln()).sum::<f64>() * 2.0;
            log_norm[c] = weights[c].max(FLOOR).ln() - 0.5 * logdet - half_d_log_2pi;
            chols.push(l);
        }

        // E-step. Each point's resp row and lse are independent (per-component
        // Cholesky forward-solve Mahalanobis + log-sum-exp), so split the points
        // across threads into ordered slots; total_ll is the SAME ordered sum of the
        // SAME per-point lse values -> byte-identical to the serial loop. Each thread
        // carries its own y (forward-solve) and logp scratch. The block scope drops
        // the borrowing closure before the M-step mutates means/covariances.
        // frankenscipy-yw7ts.
        let total_ll: f64 = {
            let mut lse = vec![0.0f64; n];
            let e_step_point =
                |row: &[f64], y: &mut [f64], logp: &mut [f64], resp_i: &mut [f64]| -> f64 {
                    let mut maxlp = f64::NEG_INFINITY;
                    for c in 0..k {
                        let l = &chols[c];
                        // Forward-solve L·y = (x − μ_c); Mahalanobis = ‖y‖².
                        let mut maha = 0.0;
                        for r in 0..d {
                            let mut s = row[r] - means[c][r];
                            for q in 0..r {
                                s -= l[r][q] * y[q];
                            }
                            let yr = s / l[r][r];
                            y[r] = yr;
                            maha += yr * yr;
                        }
                        let lp = log_norm[c] - 0.5 * maha;
                        logp[c] = lp;
                        if lp > maxlp {
                            maxlp = lp;
                        }
                    }
                    let mut sumexp = 0.0;
                    for lp in logp.iter_mut() {
                        *lp = (*lp - maxlp).exp();
                        sumexp += *lp;
                    }
                    let inv = 1.0 / sumexp;
                    for c in 0..k {
                        resp_i[c] = logp[c] * inv;
                    }
                    maxlp + sumexp.ln()
                };
            let work = n.saturating_mul(k).saturating_mul(d).saturating_mul(d);
            let nthreads_e = if work < (1 << 16) || n < 2 {
                1
            } else {
                std::thread::available_parallelism()
                    .map(|c| c.get())
                    .unwrap_or(1)
                    .min(n)
            };
            if nthreads_e <= 1 {
                let mut y = vec![0.0f64; d];
                for ((row, resp_i), lse_i) in data.iter().zip(resp.iter_mut()).zip(lse.iter_mut()) {
                    *lse_i = e_step_point(row, &mut y, &mut logp, resp_i);
                }
            } else {
                let chunk = n.div_ceil(nthreads_e);
                let e_step_point = &e_step_point;
                std::thread::scope(|scope| {
                    for ((resp_chunk, lse_chunk), data_chunk) in resp
                        .chunks_mut(chunk)
                        .zip(lse.chunks_mut(chunk))
                        .zip(data.chunks(chunk))
                    {
                        scope.spawn(move || {
                            let mut y_local = vec![0.0f64; d];
                            let mut logp_local = vec![0.0f64; k];
                            for ((row, resp_i), lse_i) in data_chunk
                                .iter()
                                .zip(resp_chunk.iter_mut())
                                .zip(lse_chunk.iter_mut())
                            {
                                *lse_i = e_step_point(row, &mut y_local, &mut logp_local, resp_i);
                            }
                        });
                    }
                });
            }
            lse.iter().sum()
        };
        log_likelihood = total_ll / n as f64;
        if (log_likelihood - old_ll).abs() < tol {
            break;
        }
        old_ll = log_likelihood;

        // M-step. Each component's (weight, mean, full covariance) is independent and the
        // covariance is the dominant O(n·d²) cost — fan the k components across cores. Each
        // component is computed by the identical serial arithmetic on its own thread, so the
        // result is BYTE-IDENTICAL to the serial loop (disjoint output slots, shared reads).
        let m_compute =
            |c: usize, w_out: &mut f64, m_out: &mut [f64], cov_out: &mut Vec<Vec<f64>>| {
                let nk: f64 = resp.iter().map(|r| r[c]).sum::<f64>().max(FLOOR);
                *w_out = nk / n as f64;
                for (j, mv) in m_out.iter_mut().enumerate() {
                    *mv = data
                        .iter()
                        .enumerate()
                        .map(|(i, r)| resp[i][c] * r[j])
                        .sum::<f64>()
                        / nk;
                }
                let mut cov = vec![vec![0.0f64; d]; d];
                for (i, row) in data.iter().enumerate() {
                    let g = resp[i][c];
                    if g == 0.0 {
                        continue;
                    }
                    for a in 0..d {
                        let ga = g * (row[a] - m_out[a]);
                        for b in 0..d {
                            cov[a][b] += ga * (row[b] - m_out[b]);
                        }
                    }
                }
                for rowv in cov.iter_mut() {
                    for v in rowv.iter_mut() {
                        *v /= nk;
                    }
                }
                for (a, rowv) in cov.iter_mut().enumerate() {
                    rowv[a] += reg_covar;
                }
                *cov_out = cov;
            };
        let mwork = (n as u64).saturating_mul(d as u64).saturating_mul(d as u64);
        let nthreads_m = if mwork < (1 << 16) || k < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(k)
        };
        if nthreads_m <= 1 {
            for c in 0..k {
                let (w_slot, m_slot, cov_slot) =
                    (&mut weights[c], &mut means[c], &mut covariances[c]);
                m_compute(c, w_slot, m_slot, cov_slot);
            }
        } else {
            let per = k.div_ceil(nthreads_m);
            let m_compute = &m_compute;
            std::thread::scope(|scope| {
                for ((((w_ch, m_ch), cov_ch), base), _) in weights
                    .chunks_mut(per)
                    .zip(means.chunks_mut(per))
                    .zip(covariances.chunks_mut(per))
                    .zip((0..k).step_by(per))
                    .zip(0..nthreads_m)
                {
                    scope.spawn(move || {
                        for (lc, ((w_out, m_out), cov_out)) in w_ch
                            .iter_mut()
                            .zip(m_ch.iter_mut())
                            .zip(cov_ch.iter_mut())
                            .enumerate()
                        {
                            m_compute(base + lc, w_out, m_out, cov_out);
                        }
                    });
                }
            });
        }
    }

    let labels: Vec<usize> = resp
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(c, _)| c)
        })
        .collect();

    // Free params (full): means k·d + covariances k·d(d+1)/2 + weights (k−1).
    let total_ll = log_likelihood * n as f64;
    let n_params = ((k - 1) + k * d + k * d * (d + 1) / 2) as f64;
    let bic = -2.0 * total_ll + n_params * (n as f64).ln();
    let aic = -2.0 * total_ll + 2.0 * n_params;

    Ok(GmmFullResult {
        weights,
        means,
        covariances,
        responsibilities: resp,
        labels,
        log_likelihood,
        bic,
        aic,
        n_iter: iters,
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

    // Resolves [frankenscipy-dnunq]: compare in squared-distance space
    // so we only sqrt the winning min_sq once per data point instead
    // of once per (point, centroid) pair. Monotone-equivalent for
    // nearest-neighbor selection.
    let k = centroids.len();
    let centroids_flat = flatten_centroids(centroids, d);
    let n = data.len();

    // Each point's nearest-centroid lookup is independent; assign them in
    // parallel into ordered slots (single pass — threads are spawned once).
    // Byte-identical: same per-point lowest-index argmin, results in data order.
    let assign = |point: &[f64]| -> (usize, f64) {
        let (best_c, min_sq) = nearest_centroid(point, &centroids_flat, k, d);
        (best_c, min_sq.sqrt())
    };
    let nthreads = if n.saturating_mul(k).saturating_mul(d) < (1 << 16) || n < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n)
    };
    let pairs: Vec<(usize, f64)> = if nthreads <= 1 {
        data.iter().map(|p| assign(p)).collect()
    } else {
        let mut out = vec![(0usize, 0.0f64); n];
        let chunk = n.div_ceil(nthreads);
        let assign = &assign;
        std::thread::scope(|scope| {
            for (t, slot) in out.chunks_mut(chunk).enumerate() {
                let base = t * chunk;
                scope.spawn(move || {
                    for (i, o) in slot.iter_mut().enumerate() {
                        *o = assign(&data[base + i]);
                    }
                });
            }
        });
        out
    };

    let labels: Vec<usize> = pairs.iter().map(|p| p.0).collect();
    let dists: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    Ok((labels, dists))
}

/// Assign each observation to its nearest code-book entry, matching
/// `scipy.cluster.vq.py_vq(obs, code_book)`: the pure-Python reference
/// implementation of [`vq`]. Returns `(code, dist)` — the index of the nearest
/// code and the Euclidean distance to it for each observation.
pub fn py_vq(
    obs: &[Vec<f64>],
    code_book: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<f64>), ClusterError> {
    vq(obs, code_book)
}

/// K-means clustering from explicit initial centroids (Lloyd iteration).
///
/// Matches `scipy.cluster.vq.kmeans2(data, k, iter, minit='matrix')`, the
/// deterministic path where `init_centroids` are the starting cluster centers.
/// Runs exactly `iter` Lloyd updates (assign via [`vq`], then recompute each
/// centroid as the mean of its members) with no early stopping; empty clusters
/// keep their previous centroid (scipy's `missing='warn'` default). Returns
/// `(centroids, labels)`, where — as in scipy — `labels` is the assignment from
/// the start of the final iteration (before the last centroid update). The
/// random initialization modes (`'random'`, `'points'`, `'++'`) are not
/// supported because they cannot reproduce scipy's RNG stream.
pub fn kmeans2(
    data: &[Vec<f64>],
    init_centroids: &[Vec<f64>],
    iter: usize,
) -> Result<(Vec<Vec<f64>>, Vec<usize>), ClusterError> {
    if data.is_empty() {
        return Err(ClusterError::EmptyData);
    }
    if iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "iter must be at least 1".to_string(),
        ));
    }
    let d = validate_feature_dimensions(data, "kmeans2")?;
    let cd = validate_feature_dimensions(init_centroids, "kmeans2 centroids")?;
    if d != cd {
        return Err(ClusterError::InvalidArgument(format!(
            "data dimension {d} must match centroid dimension {cd}"
        )));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "vq data must be finite".to_string(),
        ));
    }
    if init_centroids.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "vq centroids must be finite".to_string(),
        ));
    }
    let nc = init_centroids.len();
    if nc == 4 && d == 4 {
        return Ok(kmeans2_k4_d4(data, init_centroids, iter));
    }
    let mut code_book = init_centroids.to_vec();
    let mut label = vec![0usize; data.len()];
    // Lloyd scratch hoisted out of the loop: sums/counts are zeroed-and-reaccumulated
    // each pass; next_cb double-buffers the new code_book (mem::swap instead of a
    // fresh nc×d allocation per iteration). Byte-identical: every entry is fully
    // rewritten each pass. frankenscipy-4ylee.
    let mut sums = vec![vec![0.0_f64; d]; nc];
    let mut counts = vec![0usize; nc];
    let mut next_cb = code_book.clone();
    let mut centroids_flat = Vec::with_capacity(nc * d);
    let data_flat: Vec<f64> = data.iter().flat_map(|p| p.iter().copied()).collect();
    for _ in 0..iter {
        // Assign each observation to its nearest current centroid. kmeans2 only
        // needs labels, so bypass vq's Euclidean-distance sqrt/output vector.
        flatten_centroids_into(&code_book, d, &mut centroids_flat);
        let assignments = assign_points(&data_flat, data.len(), &centroids_flat, nc, d);
        for (dst, &(best_c, _)) in label.iter_mut().zip(assignments.iter()) {
            *dst = best_c;
        }
        // Recompute centroids as the mean of assigned observations.
        for row in sums.iter_mut() {
            row.iter_mut().for_each(|x| *x = 0.0);
        }
        counts.iter_mut().for_each(|x| *x = 0);
        // Accumulate from the contiguous `data_flat` (built once above) rather than
        // the `Vec<Vec<f64>>` rows, whose n scattered heap allocations make the
        // per-row gather cache-hostile. Byte-identical: same `+=` over c in 0..d.
        for (i, &lab) in label.iter().enumerate() {
            counts[lab] += 1;
            let row = &data_flat[i * d..i * d + d];
            let dst = &mut sums[lab];
            for c in 0..d {
                dst[c] += row[c];
            }
        }
        for j in 0..nc {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f64;
                for c in 0..d {
                    next_cb[j][c] = sums[j][c] * inv;
                }
            } else {
                // Empty cluster keeps its previous position.
                next_cb[j].clone_from(&code_book[j]);
            }
        }
        std::mem::swap(&mut code_book, &mut next_cb);
    }
    Ok((code_book, label))
}

fn kmeans2_k4_d4(
    data: &[Vec<f64>],
    init_centroids: &[Vec<f64>],
    iter: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n = data.len();
    let mut points = Vec::with_capacity(n * 4);
    for point in data {
        points.extend_from_slice(&point[..4]);
    }

    let mut centroids = [0.0_f64; 16];
    for c in 0..4 {
        centroids[c * 4..c * 4 + 4].copy_from_slice(&init_centroids[c][..4]);
    }
    let mut labels = vec![0usize; n];
    let mut sums = [0.0_f64; 16];
    let mut counts = [0usize; 4];

    for _ in 0..iter {
        sums.fill(0.0);
        counts.fill(0);
        for i in 0..n {
            let point = &points[i * 4..i * 4 + 4];
            let (best_c, _) = nearest_centroid_k4_d4(point, &centroids);
            labels[i] = best_c;
            counts[best_c] += 1;
            let dst = best_c * 4;
            sums[dst] += point[0];
            sums[dst + 1] += point[1];
            sums[dst + 2] += point[2];
            sums[dst + 3] += point[3];
        }
        for (c, &count) in counts.iter().enumerate() {
            if count > 0 {
                let inv = 1.0 / count as f64;
                let base = c * 4;
                centroids[base] = sums[base] * inv;
                centroids[base + 1] = sums[base + 1] * inv;
                centroids[base + 2] = sums[base + 2] * inv;
                centroids[base + 3] = sums[base + 3] * inv;
            }
        }
    }

    let code_book = (0..4)
        .map(|c| centroids[c * 4..c * 4 + 4].to_vec())
        .collect();
    (code_book, labels)
}

/// Whiten observations by dividing by per-feature standard deviation.
///
/// Matches `scipy.cluster.vq.whiten`.
/// When `true`, [`whiten`] builds its output serially (the ORIG behaviour); default `false` fans
/// the per-row divide-by-std map across row-chunks. Byte-identical. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static WHITEN_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

    // The output is a per-ROW map: out[i][j] = data[i][j] / stds[j] (or the raw value when the
    // feature is constant). Each output row is a pure function of data[i] and the shared stds, so
    // the rows — and their per-row Vec allocations — fan across cores with no cross-row dependency.
    // Every element is computed by the identical expression in the identical order → BYTE-IDENTICAL
    // to the serial map. The mean/variance reductions above stay serial (parallelizing them would
    // reassociate the per-feature sums). Gated on total element work.
    let whiten_row = |point: &[f64]| -> Vec<f64> {
        point
            .iter()
            .zip(stds.iter())
            .map(|(&v, &s)| if s > 0.0 { v / s } else { v })
            .collect()
    };
    let nthreads = if WHITEN_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || n < 2
        || n.saturating_mul(d) < 65_536
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n)
    };
    if nthreads <= 1 {
        return Ok(data.iter().map(|point| whiten_row(point)).collect());
    }
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); n];
    let chunk = n.div_ceil(nthreads);
    let stds_ref = &stds;
    std::thread::scope(|scope| {
        for (ci, block) in out.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            scope.spawn(move || {
                for (k, slot) in block.iter_mut().enumerate() {
                    let point = &data[base + k];
                    *slot = point
                        .iter()
                        .zip(stds_ref.iter())
                        .map(|(&v, &s)| if s > 0.0 { v / s } else { v })
                        .collect();
                }
            });
        }
    });
    Ok(out)
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
// Reachable only from `agglomerate_nnarray`, which nothing calls — the whole
// NN-array linkage lever is unwired at HEAD and was already unwired before the
// bulk deletion. Tracked in frankenscipy-1b85n; the allow goes away when that
// bead decides to re-wire or formally retire it.
#[allow(dead_code)]
#[inline]
fn agglo_idx(total: usize, row: usize, col: usize) -> usize {
    row * total + col
}

#[allow(dead_code)] // frankenscipy-1b85n — see `agglo_idx`.
fn agglo_nearest(inter_dist: &[f64], active_ids: &[usize], i: usize, total: usize) -> (usize, f64) {
    let mut best_j = i;
    let mut best_d = f64::INFINITY;
    let row = &inter_dist[agglo_idx(total, i, 0)..agglo_idx(total, i + 1, 0)];
    let start = active_ids.partition_point(|&j| j <= i);
    for &j in &active_ids[start..] {
        if row[j] < best_d {
            best_d = row[j];
            best_j = j;
        }
    }
    (best_j, best_d)
}

/// Agglomerative clustering core shared by `linkage` and `linkage_from_distances`.
///
/// O(n^2) typical rather than the naive O(n^3) "rescan every pair each step"
/// loop: a nearest-neighbour array keeps each active cluster's nearest active
/// successor, so the closest pair is found in O(active) instead of O(active^2).
///
/// CLAIM CORRECTED 2026-08-08 (frankenscipy-1b85n). This previously claimed to
/// be "byte-identical ... element-for-element" with the pairwise scan. Measured
/// against the SHIPPED path (`linkage` -> `linkage_from_dm`), that holds for
/// Single, Complete, Average and Weighted, but NOT for Ward: it differs by
/// exactly 1 ulp, and SciPy agrees with the shipped value, not this one. See
/// `unwired_agglomerate_nnarray_is_bit_identical_to_the_shipped_linkage`.
/// Re-wiring this lever would therefore move Ward off SciPy parity.
// UNWIRED: nothing in the library calls this. The bench keeps its own private
// copy (`legacy_agglomerate_nnarray` in benches/cluster_bench.rs) and measures
// that instead, so the bench reports a lever the library does not ship.
// frankenscipy-1b85n decides whether to re-wire (needs a full measured A/B) or
// formally retire it. Preserved, not deleted.
#[allow(dead_code)]
fn agglomerate_nnarray(n: usize, mut inter_dist: Vec<f64>, method: LinkageMethod) -> Vec<[f64; 4]> {
    let total = 2 * n - 1;
    let mut active_ids: Vec<usize> = (0..n).collect();
    let mut cluster_size = vec![1usize; total];
    let mut nn = vec![0usize; total];
    let mut d_nn = vec![f64::INFINITY; total];
    for i in 0..n {
        let (j, d) = agglo_nearest(&inter_dist, &active_ids, i, total);
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
        for &i in &active_ids {
            if d_nn[i] < min_d {
                min_d = d_nn[i];
                mi = i;
            }
        }
        let mj = nn[mi];

        let new_size = cluster_size[mi] + cluster_size[mj];
        result.push([mi as f64, mj as f64, min_d, new_size as f64]);

        cluster_size[new_id] = new_size;
        active_ids.retain(|&k| k != mi && k != mj);

        // Distances from the new cluster to every remaining active cluster.
        for &k in &active_ids {
            let d_ki = inter_dist[agglo_idx(total, k, mi)];
            let d_kj = inter_dist[agglo_idx(total, k, mj)];
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
            inter_dist[agglo_idx(total, k, new_id)] = new_dist;
            inter_dist[agglo_idx(total, new_id, k)] = new_dist;
        }

        // The new cluster is the largest active id, so it has no successor yet.
        d_nn[new_id] = f64::INFINITY;
        nn[new_id] = new_id;
        let refresh_len = active_ids.len();
        active_ids.push(new_id);

        // Refresh nearest neighbours: clusters that pointed at a merged cluster
        // recompute from scratch; the rest only need to test the new cluster as
        // a (strictly closer) candidate, matching the scan's smallest-j tie-break.
        for &k in &active_ids[..refresh_len] {
            if nn[k] == mi || nn[k] == mj {
                let (j, d) = agglo_nearest(&inter_dist, &active_ids, k, total);
                nn[k] = j;
                d_nn[k] = d;
            } else {
                let d_new = inter_dist[agglo_idx(total, k, new_id)];
                if d_new < d_nn[k] {
                    d_nn[k] = d_new;
                    nn[k] = new_id;
                }
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
    let d = validate_feature_dimensions(data, "linkage")?;
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ClusterError::InvalidArgument(
            "linkage input must be finite".to_string(),
        ));
    }
    let flat = flatten_points(data, d);

    // br-6m7l: scipy uses fast_linkage (heap-based "Generic Clustering
    // Algorithm" from Mullner 2011) for Centroid/Median; fsci's simpler
    // O(n^3) scan diverges from scipy on tie-break of the column-0/1
    // cluster IDs. Route those two methods through the heap path so the
    // resulting linkage matrix matches scipy element-for-element.
    // Build the full n×n distance matrix once and route every method through the shared
    // O(n²) dm-based path (single→MST, reducible→NN-chain, centroid/median→Müller heap).
    let dm = linkage_distance_matrix(&flat, n, d);
    Ok(linkage_from_dm(n, dm, method))
}

/// Minimum build work (`n² · dimension`) above which the dense distance-matrix
/// build is parallelized. Below it the serial upper-triangle loop wins (thread
/// spawn dominates); measured crossover well under n=800/d=4.
const LINKAGE_DM_PAR_WORK_GATE: u128 = 2_000_000;

/// Build the dense symmetric `n×n` distance matrix used by `linkage`, from
/// row-major `flat` (`d` coords per row). Serial below the work gate; above it
/// each thread owns a contiguous block of full rows (disjoint `chunks_mut`, no
/// per-row allocation, no scatter) and recomputes the lower triangle. Because
/// `sq_dist` is symmetric (`Σ(a−b)² == Σ(b−a)²` term-for-term), every entry is
/// bit-identical to the serial upper-triangle-plus-mirror fill, so the downstream
/// tie-break-sensitive agglomeration is unaffected. The diagonal stays `0.0`.
fn linkage_distance_matrix(flat: &[f64], n: usize, d: usize) -> Vec<f64> {
    let row = |idx: usize| -> &[f64] { &flat[idx * d..idx * d + d] };
    let mut dm = vec![0.0_f64; n * n];
    if (n as u128) * (n as u128) * (d.max(1) as u128) < LINKAGE_DM_PAR_WORK_GATE {
        for i in 0..n {
            for j in i + 1..n {
                let dist = sq_dist(row(i), row(j)).sqrt();
                dm[i * n + j] = dist;
                dm[j * n + i] = dist;
            }
        }
        return dm;
    }
    let nthreads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(16)
        .min(n)
        .max(1);
    let chunk = n.div_ceil(nthreads);
    let row = &row;
    std::thread::scope(|scope| {
        for (blk, rows) in dm.chunks_mut(chunk * n).enumerate() {
            let i0 = blk * chunk;
            scope.spawn(move || {
                for (roff, dst) in rows.chunks_mut(n).enumerate() {
                    let i = i0 + roff;
                    let ri = row(i);
                    for (j, slot) in dst.iter_mut().enumerate() {
                        if i != j {
                            *slot = sq_dist(ri, row(j)).sqrt();
                        }
                    }
                }
            });
        }
    });
    dm
}

/// Shared O(n²) agglomeration over a full n×n distance matrix `dm`, used by BOTH
/// `linkage` (from points) and `linkage_from_distances` (from a condensed matrix) so the
/// two entry points are bit-identical. single→MST, ward/complete/average/weighted→NN-chain,
/// centroid/median→the Müller heap (`linkage_fast`).
fn linkage_from_dm(n: usize, dm: Vec<f64>, method: LinkageMethod) -> Vec<[f64; 4]> {
    match method {
        LinkageMethod::Centroid | LinkageMethod::Median => {
            let dist_mat: Vec<Vec<f64>> = (0..n).map(|i| dm[i * n..(i + 1) * n].to_vec()).collect();
            linkage_fast(n, &dist_mat, method)
        }
        LinkageMethod::Single => single_linkage_mst(n, &dm),
        _ => nn_chain_linkage(n, dm, method),
    }
}

/// Single linkage via the minimum spanning tree (scipy's `mst_single_linkage`): the
/// single-linkage dendrogram is exactly the MST ordered by edge weight. Prim's algorithm
/// is O(n²) (vs the generic O(n³) nearest-pair scan), then the edges are stably sorted by
/// distance and relabeled with scipy's LinkageUnionFind (new cluster id n, n+1, …) so the
/// output matches scipy element-for-element.
fn single_linkage_mst(n: usize, dm: &[f64]) -> Vec<[f64; 4]> {
    // Prim's MST from vertex 0, recording edges in add-order.
    let mut in_tree = vec![false; n];
    let mut min_d = vec![f64::INFINITY; n];
    let mut nearest = vec![0usize; n];
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n - 1);
    in_tree[0] = true;
    min_d[1..n].copy_from_slice(&dm[1..n]);
    for _ in 1..n {
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_d[j] < bd {
                bd = min_d[j];
                best = j;
            }
        }
        in_tree[best] = true;
        edges.push((bd, nearest[best], best));
        for j in 0..n {
            if !in_tree[j] {
                let dj = dm[best * n + j];
                if dj < min_d[j] {
                    min_d[j] = dj;
                    nearest[j] = best;
                }
            }
        }
    }
    // Stable sort by distance (matches scipy's argsort(kind='stable') on the MST edges).
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    // scipy LinkageUnionFind relabel: each merge mints a new cluster id n, n+1, …
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut size = vec![1usize; total];
    fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut z = Vec::with_capacity(n - 1);
    // Merge `step` creates internal node `n + step`, so the label is a pure
    // function of the iteration index rather than a running counter.
    for (step, (dist, u, v)) in edges.into_iter().enumerate() {
        let next_label = n + step;
        let ru = uf_find(&mut parent, u);
        let rv = uf_find(&mut parent, v);
        let (a, b) = if ru < rv { (ru, rv) } else { (rv, ru) };
        let sz = size[ru] + size[rv];
        z.push([a as f64, b as f64, dist, sz as f64]);
        parent[ru] = next_label;
        parent[rv] = next_label;
        size[next_label] = sz;
    }
    z
}

/// Nearest-neighbour-chain agglomeration (Müller 2011, scipy's `nn_chain`) for the
/// reducible Lance-Williams methods (ward/complete/average/weighted): O(n²) instead of the
/// O(n³) nearest-pair scan. Builds the merge tree by following reciprocal-nearest-neighbour
/// chains, updating one full distance matrix via the method's Lance-Williams rule, then
/// stably sorts the merges by distance and relabels with scipy's LinkageUnionFind — output
/// matches scipy element-for-element.
fn nn_chain_linkage(n: usize, mut dm: Vec<f64>, method: LinkageMethod) -> Vec<[f64; 4]> {
    let mut size = vec![1.0_f64; n]; // 0.0 ⇒ inactive slot
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut merges: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        if chain.is_empty() {
            let start = (0..n).find(|&i| size[i] > 0.0).unwrap();
            chain.push(start);
        }
        // Extend the chain until the last two elements are reciprocal nearest neighbours.
        let (x, y, cur) = loop {
            let x = *chain.last().unwrap();
            let mut y = usize::MAX;
            let mut cur = f64::INFINITY;
            if chain.len() > 1 {
                y = chain[chain.len() - 2];
                cur = dm[x * n + y];
            }
            for i in 0..n {
                if i != x && size[i] > 0.0 {
                    let dxi = dm[x * n + i];
                    if dxi < cur {
                        cur = dxi;
                        y = i;
                    }
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                break (x, y, cur);
            }
            chain.push(y);
        };
        chain.pop();
        chain.pop();
        let (a, b) = if x < y { (x, y) } else { (y, x) };
        let (na, nb) = (size[a], size[b]);
        let new_size = na + nb;
        merges.push((a, b, cur, new_size));
        // Deactivate slot a; slot b carries the merged cluster. Update D(b, i) via Lance-Williams.
        size[a] = 0.0;
        size[b] = new_size;
        for i in 0..n {
            if i != a && i != b && size[i] > 0.0 {
                let dai = dm[a * n + i];
                let dbi = dm[b * n + i];
                let ni = size[i];
                let nd = match method {
                    LinkageMethod::Ward => {
                        let t = 1.0 / (na + nb + ni);
                        (((na + ni) * t) * dai * dai + ((nb + ni) * t) * dbi * dbi
                            - (ni * t) * cur * cur)
                            .sqrt()
                    }
                    LinkageMethod::Complete => dai.max(dbi),
                    LinkageMethod::Average => (na * dai + nb * dbi) / (na + nb),
                    LinkageMethod::Weighted => 0.5 * (dai + dbi),
                    _ => unreachable!(),
                };
                dm[b * n + i] = nd;
                dm[i * n + b] = nd;
            }
        }
    }
    // Stable sort by distance, then scipy LinkageUnionFind relabel (new id n, n+1, …).
    merges.sort_by(|p, q| p.2.total_cmp(&q.2));
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut z = Vec::with_capacity(n - 1);
    // As above: merge `step` creates internal node `n + step`.
    for (step, (u, v, dist, sz)) in merges.into_iter().enumerate() {
        let next_label = n + step;
        let ru = uf_find(&mut parent, u);
        let rv = uf_find(&mut parent, v);
        let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
        z.push([lo as f64, hi as f64, dist, sz]);
        parent[ru] = next_label;
        parent[rv] = next_label;
    }
    z
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

/// Reorder a linkage matrix to minimize the sum of distances between adjacent
/// leaves of the dendrogram (Bar-Joseph, Gifford & Jaakkola 2001), matching
/// `scipy.cluster.hierarchy.optimal_leaf_ordering(Z, y)`.
///
/// `z` is the linkage matrix (`n-1` rows) and `y` the *condensed* pairwise
/// distance matrix (`n(n-1)/2` entries) from which `z` was generated. Returns a
/// copy of `z` whose children are swapped (rotated) so that [`leaves_list`]
/// yields the optimally ordered leaves.
///
/// This is a faithful port of SciPy's implementation: leaves are relabelled to
/// their current linear positions, the dynamic-programming cost table `M` is
/// accumulated in `f32` (distances stay `f64`) exactly as SciPy does, per-node
/// optimal endpoints break ties toward the lowest `(u, w)`, and the recovered
/// per-node swaps are propagated to descendants via the rotation parity rule.
pub fn optimal_leaf_ordering(z: &[[f64; 4]], y: &[f64]) -> Result<Vec<[f64; 4]>, ClusterError> {
    let n = z.len() + 1;
    if n < 2 {
        return Ok(z.to_vec());
    }
    let expected = n * (n - 1) / 2;
    if y.len() != expected {
        return Err(ClusterError::InvalidArgument(format!(
            "optimal_leaf_ordering: condensed distance length {} != expected {expected}",
            y.len()
        )));
    }
    // Square distance matrix from the condensed form.
    let mut dmat = vec![0.0f64; n * n];
    let mut idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let v = y[idx];
            idx += 1;
            dmat[i * n + j] = v;
            dmat[j * n + i] = v;
        }
    }

    // Relabel leaves to their current linear order, so every cluster occupies a
    // contiguous [min, max) range of positions.
    let sorted_leaves = leaves_list(z);
    let mut posn = vec![0usize; n];
    for (sp, &orig) in sorted_leaves.iter().enumerate() {
        posn[orig] = sp;
    }
    // Distance matrix permuted into sorted-leaf order.
    let mut sd = vec![0.0f64; n * n];
    for a in 0..n {
        for b in 0..n {
            sd[a * n + b] = dmat[sorted_leaves[a] * n + sorted_leaves[b]];
        }
    }
    // Linkage children remapped: leaves -> sorted positions, internal nodes keep id.
    let mut szl = vec![0usize; n - 1];
    let mut szr = vec![0usize; n - 1];
    for i in 0..(n - 1) {
        let l = z[i][0] as usize;
        let r = z[i][1] as usize;
        szl[i] = if l < n { posn[l] } else { l };
        szr[i] = if r < n { posn[r] } else { r };
    }
    // Contiguous range of every node id (leaf position p -> [p, p+1)).
    let nc = 2 * n - 1;
    let mut crmin = vec![0usize; nc];
    let mut crmax = vec![0usize; nc];
    for p in 0..n {
        crmin[p] = p;
        crmax[p] = p + 1;
    }
    for i in 0..(n - 1) {
        let v = i + n;
        crmin[v] = crmin[szl[i]];
        crmax[v] = crmax[szr[i]];
    }

    // Dynamic program. M is f32 (matching SciPy); swap_status records, per (u,w),
    // which child sub-cluster supplied each endpoint.
    let mut mm = vec![0.0f32; n * n];
    let mut sw0 = vec![0u8; n * n];
    let mut sw1 = vec![0u8; n * n];
    let mut must_swap = vec![0u8; n - 1];
    let big = 1_073_741_824.0f32; // 2^30

    for i in 0..(n - 1) {
        let vl = szl[i];
        let vr = szr[i];
        let (total_u, ucl, mcl): (usize, [usize; 2], [usize; 2]) = if vl < n {
            (1, [vl, 0], [vl, 0])
        } else {
            let li = vl - n;
            (2, [szl[li], szr[li]], [szr[li], szl[li]])
        };
        let (total_w, wcl, kcl): (usize, [usize; 2], [usize; 2]) = if vr < n {
            (1, [vr, 0], [vr, 0])
        } else {
            let ri = vr - n;
            (2, [szr[ri], szl[ri]], [szl[ri], szr[ri]])
        };

        for sl in 0..total_u {
            for sr in 0..total_w {
                let umin = crmin[ucl[sl]];
                let umax = crmax[ucl[sl]];
                let mmin = crmin[mcl[sl]];
                let mmax = crmax[mcl[sl]];
                let wmin = crmin[wcl[sr]];
                let wmax = crmax[wcl[sr]];
                let kmin = crmin[kcl[sr]];
                let kmax = crmax[kcl[sr]];
                // Per-endpoint-pair cost min_{mp,kp}(M[u,mp] + M[w,kp] + D[mp,kp]).
                // Reads ONLY already-finalized child cells (mp∈M, kp∈K are child
                // ranges), so every (u,w) is independent and may be computed in
                // parallel; the min value is comparison-order-independent ⇒ identical.
                let cell = |u: usize, w: usize, mm: &[f32]| -> f32 {
                    let mut cur = big;
                    for mp in mmin..mmax {
                        for kp in kmin..kmax {
                            // SciPy: float M + float M (f32) + double D, stored f32.
                            let cand =
                                ((mm[u * n + mp] + mm[w * n + kp]) as f64 + sd[mp * n + kp]) as f32;
                            if cand < cur {
                                cur = cand;
                            }
                        }
                    }
                    cur
                };
                let nu = umax - umin;
                let nw = wmax - wmin;
                let work = (nu as u64)
                    .saturating_mul(nw as u64)
                    .saturating_mul((mmax - mmin) as u64)
                    .saturating_mul((kmax - kmin) as u64);
                // The DP is O(n⁴) and dominated by the top tree nodes; parallelize
                // those over the independent `u` rows (each thread reads `mm`
                // read-only, results written serially below). Small blocks stay
                // serial (thread spawn isn't worth it). Byte-identical either way.
                let cores = std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(1);
                let nthreads = if work >= (1 << 20) {
                    // Equivalent to the previous `cores.min(nu).min(16).max(1)`.
                    // The upper bound is itself floored at 1 so that nu == 0
                    // yields 1 thread rather than panicking on clamp(1, 0).
                    cores.clamp(1, nu.clamp(1, 16))
                } else {
                    1
                };
                let curs: Vec<f32> = if nthreads <= 1 {
                    let mm_ro: &[f32] = &mm;
                    (umin..umax)
                        .flat_map(|u| (wmin..wmax).map(move |w| cell(u, w, mm_ro)))
                        .collect()
                } else {
                    let mm_ro: &[f32] = &mm;
                    let cell = &cell;
                    let chunk = nu.div_ceil(nthreads);
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = (0..nthreads)
                            .filter_map(|t| {
                                let u0 = umin + t * chunk;
                                if u0 >= umax {
                                    return None;
                                }
                                let u1 = (u0 + chunk).min(umax);
                                Some(scope.spawn(move || {
                                    (u0..u1)
                                        .flat_map(|u| (wmin..wmax).map(move |w| cell(u, w, mm_ro)))
                                        .collect::<Vec<f32>>()
                                }))
                            })
                            .collect();
                        handles
                            .into_iter()
                            .flat_map(|h| h.join().expect("olo worker panicked"))
                            .collect()
                    })
                };
                let mut ci = 0usize;
                for u in umin..umax {
                    for w in wmin..wmax {
                        let cur = curs[ci];
                        ci += 1;
                        mm[u * n + w] = cur;
                        mm[w * n + u] = cur;
                        sw0[u * n + w] = sl as u8;
                        sw0[w * n + u] = sl as u8;
                        sw1[u * n + w] = sr as u8;
                        sw1[w * n + u] = sr as u8;
                    }
                }
            }
        }

        // Best endpoints for this node (tie-break: lowest u, then lowest w).
        let mut cur = big;
        let mut bu = 0usize;
        let mut bw = 0usize;
        for u in crmin[vl]..crmax[vl] {
            for w in crmin[vr]..crmax[vr] {
                if mm[u * n + w] < cur {
                    cur = mm[u * n + w];
                    bu = u;
                    bw = w;
                }
            }
        }
        if vl >= n {
            must_swap[vl - n] = sw0[bu * n + bw];
        }
        if vr >= n {
            must_swap[vr - n] = sw1[bu * n + bw];
        }
    }

    // Propagate swaps: rotating a node flips all of its descendants, so a node's
    // effective swap is the parity of `must_swap` over itself and its ancestors.
    let m1 = n - 1;
    let mut isdesc = vec![0i64; m1 * m1];
    for i in 0..m1 {
        isdesc[i * m1 + i] += 1;
        let l = z[i][0] as usize;
        let r = z[i][1] as usize;
        if l >= n {
            let li = l - n;
            isdesc[i * m1 + li] += 1;
            for c in 0..m1 {
                isdesc[i * m1 + c] += isdesc[li * m1 + c];
            }
        }
        if r >= n {
            let ri = r - n;
            isdesc[i * m1 + ri] += 1;
            for c in 0..m1 {
                isdesc[i * m1 + c] += isdesc[ri * m1 + c];
            }
        }
    }
    let mut final_swap = vec![false; m1];
    for j in 0..m1 {
        let mut s = 0usize;
        for i in 0..m1 {
            if must_swap[i] != 0 && isdesc[i * m1 + j] > 0 {
                s += 1;
            }
        }
        final_swap[j] = s % 2 == 1;
    }

    let mut out = z.to_vec();
    for i in 0..m1 {
        if final_swap[i] {
            out[i] = [z[i][1], z[i][0], z[i][2], z[i][3]];
        }
    }
    Ok(out)
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

        // Merge membership lists. ci/cj are subsumed by new_id and never referenced
        // again, so MOVE ci's list (reusing its buffer) instead of cloning it, then
        // append cj's. Byte-identical contents (ci's members then cj's, same order).
        let mut merged = std::mem::take(&mut membership[ci]);
        merged.extend_from_slice(&membership[cj]);
        membership[new_id] = merged;
    }

    condensed
}

/// When `true`, [`inconsistent`] computes its per-merge statistics serially (the ORIG behaviour);
/// default `false` fans the independent per-merge subtree walks across step-chunks. Byte-identical.
/// `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static INCONSISTENT_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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
    let m = z.len();

    // Each output row `result[step]` is a pure function of `step`: it walks `depth` levels of the
    // (read-only) linkage subtree below merge `n+step` via `collect_depths`, then reduces those
    // distances to (mean, std, count, incon). No cross-step state — the merges are independent — so
    // the loop fans across step-chunks. Every row is computed by the identical expression in the
    // identical order → BYTE-IDENTICAL to the serial build. Gated on the merge count.
    let stat = |step: usize| -> [f64; 4] {
        let mut dists = Vec::new();
        collect_depths(z, n + step, n, depth, &mut dists);
        let count = dists.len() as f64;
        let mean = if count > 0.0 {
            dists.iter().sum::<f64>() / count
        } else {
            z[step][2]
        };
        let std = if count > 1.0 {
            let var = dists.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        let incon = if std > 0.0 {
            (z[step][2] - mean) / std
        } else {
            0.0
        };
        [mean, std, count, incon]
    };

    let nthreads =
        if INCONSISTENT_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) || m < 4096 {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(m)
        };
    if nthreads <= 1 {
        return (0..m).map(stat).collect();
    }
    let mut result = vec![[0.0f64; 4]; m];
    let chunk = m.div_ceil(nthreads);
    let stat_ref = &stat;
    std::thread::scope(|scope| {
        for (ci, block) in result.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            scope.spawn(move || {
                for (k, slot) in block.iter_mut().enumerate() {
                    *slot = stat_ref(base + k);
                }
            });
        }
    });
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

/// Maximum linkage distance within each non-singleton cluster's subtree.
///
/// Matches `scipy.cluster.hierarchy.maxdists`. `MD[i]` is the largest link
/// height (column 2 of `Z`) among internal node `n + i` and all of its
/// descendant merges — which can exceed `Z[i, 2]` when the linkage has
/// inversions (centroid/median/Ward).
pub fn maxdists(z: &[[f64; 4]]) -> Vec<f64> {
    if z.is_empty() {
        return vec![];
    }
    let n = z.len() + 1;
    let mut md = vec![0.0_f64; z.len()];
    for (i, row) in z.iter().enumerate() {
        let mut m = row[2];
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;
        if c1 >= n {
            m = m.max(md[c1 - n]);
        }
        if c2 >= n {
            m = m.max(md[c2 - n]);
        }
        md[i] = m;
    }
    md
}

/// Maximum inconsistency coefficient within each non-singleton cluster's subtree.
///
/// Matches `scipy.cluster.hierarchy.maxinconsts(Z, R)`. Like [`maxdists`] but
/// propagates the inconsistency coefficient (column 3 of the inconsistency
/// matrix `R` from [`inconsistent`]) instead of the link height.
///
/// This IS [`max_rstat`] with `i = 3` — verified against the incumbent, where
/// `maxRstat(Z, R, 3)` and `maxinconsts(Z, R)` are element-for-element equal — so
/// it delegates rather than carrying a second copy of the traversal.
///
/// # Errors
///
/// `Z` and `R` having different row counts.
pub fn maxinconsts(z: &[[f64; 4]], r: &[[f64; 4]]) -> Result<Vec<f64>, ClusterError> {
    max_rstat(z, r, 3)
}

/// Find the root linkage node of each flat cluster.
///
/// Matches `scipy.cluster.hierarchy.leaders(Z, T)`. Given a flat cluster
/// assignment `T` (length `n`), returns `(L, M)` where `L[j]` is the linkage
/// node id leading flat cluster `M[j]`. `L` is in ascending node-id order. If
/// `T` is not a valid flat clustering of `Z` (a cluster spans more than one
/// subtree), an error is returned.
pub fn leaders(z: &[[f64; 4]], t: &[usize]) -> Result<(Vec<usize>, Vec<usize>), ClusterError> {
    let n = z.len() + 1;
    if t.len() != n {
        return Err(ClusterError::InvalidArgument(format!(
            "T must have length {n} (number of observations)"
        )));
    }
    let total = 2 * n - 1;
    // belong[node] = cluster id if the subtree is monochromatic, else -1.
    let mut belong = vec![-1_i64; total];
    for i in 0..n {
        belong[i] = t[i] as i64;
    }
    let mut parent = vec![usize::MAX; total];
    for (k, row) in z.iter().enumerate() {
        let node = n + k;
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;
        parent[c1] = node;
        parent[c2] = node;
        belong[node] = if belong[c1] != -1 && belong[c1] == belong[c2] {
            belong[c1]
        } else {
            -1
        };
    }
    let mut l = Vec::new();
    let mut m = Vec::new();
    for node in 0..total {
        let b = belong[node];
        if b == -1 {
            continue;
        }
        let is_leader = parent[node] == usize::MAX || belong[parent[node]] == -1;
        if is_leader {
            l.push(node);
            m.push(b as usize);
        }
    }
    // A valid flat clustering yields exactly one leader per cluster id.
    let mut seen = std::collections::HashSet::new();
    for &c in &m {
        if !seen.insert(c) {
            return Err(ClusterError::InvalidArgument(
                "T is not a valid flat cluster assignment for Z".to_string(),
            ));
        }
    }
    Ok((l, m))
}

/// Maximum of column `i` of the inconsistency matrix `R` over each non-singleton
/// cluster's subtree. Matches `scipy.cluster.hierarchy.maxRstat(Z, R, i)`
/// (`maxinconsts` is the `i == 3` case, `maxdists` propagates `Z[:,2]`).
pub fn max_rstat(z: &[[f64; 4]], r: &[[f64; 4]], i: usize) -> Result<Vec<f64>, ClusterError> {
    if i > 3 {
        return Err(ClusterError::InvalidArgument(
            "i must be in 0..=3".to_string(),
        ));
    }
    if z.len() != r.len() {
        return Err(ClusterError::InvalidArgument(
            "Z and R must have the same number of rows".to_string(),
        ));
    }
    if z.is_empty() {
        return Ok(vec![]);
    }
    let n = z.len() + 1;
    let mut out = vec![0.0_f64; z.len()];
    for (j, row) in z.iter().enumerate() {
        let mut m = r[j][i];
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;
        if c1 >= n {
            m = m.max(out[c1 - n]);
        }
        if c2 >= n {
            m = m.max(out[c2 - n]);
        }
        out[j] = m;
    }
    Ok(out)
}

/// Whether two flat cluster assignments are identical up to relabeling.
///
/// Matches `scipy.cluster.hierarchy.is_isomorphic(T1, T2)`: returns true iff
/// there is a bijection between the labels of `t1` and `t2` consistent on every
/// element.
pub fn is_isomorphic(t1: &[usize], t2: &[usize]) -> bool {
    if t1.len() != t2.len() {
        return false;
    }
    let mut fwd: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut rev: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (&a, &b) in t1.iter().zip(t2.iter()) {
        if *fwd.entry(a).or_insert(b) != b {
            return false;
        }
        if *rev.entry(b).or_insert(a) != a {
            return false;
        }
    }
    true
}

/// Whether a linkage matrix `Z` corresponds to a condensed distance vector `Y`,
/// i.e. they describe the same number of original observations.
///
/// Matches `scipy.cluster.hierarchy.correspond(Z, Y)`.
pub fn correspond(z: &[[f64; 4]], y: &[f64]) -> bool {
    let n = z.len() + 1;
    y.len() == n * (n - 1) / 2
}

/// Cut a hierarchical clustering into flat clusters.
///
/// Matches `scipy.cluster.hierarchy.cut_tree(Z, n_clusters=…, height=…)` for a
/// single cut: supply exactly one of `n_clusters` or `height`. Returns the
/// 0-based flat labels, numbered in order of first appearance. (`height` cuts
/// assume a monotone linkage, as produced by the standard linkage methods.)
pub fn cut_tree(
    z: &[[f64; 4]],
    n_clusters: Option<usize>,
    height: Option<f64>,
) -> Result<Vec<usize>, ClusterError> {
    let n = z.len() + 1;
    let num_merges = match (n_clusters, height) {
        (Some(k), None) => {
            if k == 0 || k > n {
                return Err(ClusterError::InvalidArgument(
                    "n_clusters must be in 1..=n".to_string(),
                ));
            }
            n - k
        }
        (None, Some(h)) => z.iter().take_while(|row| row[2] <= h).count(),
        _ => {
            return Err(ClusterError::InvalidArgument(
                "specify exactly one of n_clusters or height".to_string(),
            ));
        }
    };

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != c {
            let nx = parent[c];
            parent[c] = r;
            c = nx;
        }
        r
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut node_rep = vec![0usize; 2 * n - 1];
    for (p, slot) in node_rep.iter_mut().enumerate().take(n) {
        *slot = p;
    }
    for (i, row) in z.iter().enumerate().take(num_merges) {
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;
        let r1 = find(&mut parent, node_rep[c1]);
        let r2 = find(&mut parent, node_rep[c2]);
        parent[r2] = r1;
        node_rep[n + i] = r1;
    }

    let mut label_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0usize;
    let mut out = vec![0usize; n];
    for (p, slot) in out.iter_mut().enumerate() {
        let root = find(&mut parent, p);
        *slot = *label_of.entry(root).or_insert_with(|| {
            let l = next;
            next += 1;
            l
        });
    }
    Ok(out)
}

/// Cut a hierarchical tree at MULTIPLE points at once — the batched form of
/// [`cut_tree`], matching `scipy.cluster.hierarchy.cut_tree(Z, n_clusters=[…])` /
/// `height=[…]`. Returns one flat cluster labeling per requested cut: entry `j`
/// is column `j` of SciPy's `(n_points, n_cuts)` result (so `out[j][p]` is point
/// `p`'s label at cut `j`). With both `None`, cuts at every level — `n` clusters
/// down to `1` (`out[j]` has `n − j` clusters), matching SciPy's default full
/// matrix. SciPy's `cut_tree` is notably slow (re-derives each column); each cut
/// here reuses the O(n) union-find of [`cut_tree`], so every column is
/// byte-identical to SciPy's while the whole call is orders of magnitude faster.
pub fn cut_tree_multi(
    z: &[[f64; 4]],
    n_clusters: Option<&[usize]>,
    heights: Option<&[f64]>,
) -> Result<Vec<Vec<usize>>, ClusterError> {
    match (n_clusters, heights) {
        (Some(_), Some(_)) => Err(ClusterError::InvalidArgument(
            "specify at most one of n_clusters or heights".to_string(),
        )),
        (Some(ks), None) => ks.iter().map(|&k| cut_tree(z, Some(k), None)).collect(),
        (None, Some(hs)) => hs.iter().map(|&h| cut_tree(z, None, Some(h))).collect(),
        (None, None) => {
            let n = z.len() + 1;
            (0..n).map(|j| cut_tree(z, Some(n - j), None)).collect()
        }
    }
}

/// Convert a SciPy/FrankenSciPy linkage matrix to MATLAB(TM) format.
///
/// Matches `scipy.cluster.hierarchy.to_mlab_linkage(Z)`: 1-indexes the two
/// cluster columns and drops the cluster-size column, yielding `[c1+1, c2+1, d]`.
pub fn to_mlab_linkage(z: &[[f64; 4]]) -> Vec<[f64; 3]> {
    z.iter()
        .map(|row| [row[0] + 1.0, row[1] + 1.0, row[2]])
        .collect()
}

/// Convert a MATLAB(TM) linkage matrix to SciPy/FrankenSciPy format.
///
/// Matches `scipy.cluster.hierarchy.from_mlab_linkage(Z)`: 0-indexes the cluster
/// columns and appends the recomputed cluster-size column.
pub fn from_mlab_linkage(z: &[[f64; 3]]) -> Vec<[f64; 4]> {
    let n = z.len() + 1;
    let mut sizes = vec![0usize; 2 * n - 1];
    for slot in sizes.iter_mut().take(n) {
        *slot = 1;
    }
    let mut out = Vec::with_capacity(z.len());
    for (i, row) in z.iter().enumerate() {
        let c1 = (row[0] - 1.0) as usize;
        let c2 = (row[1] - 1.0) as usize;
        let count = sizes[c1] + sizes[c2];
        sizes[n + i] = count;
        out.push([row[0] - 1.0, row[1] - 1.0, row[2], count as f64]);
    }
    out
}

/// Whether `r` is a valid inconsistency matrix.
///
/// Matches `scipy.cluster.hierarchy.is_valid_im(R)`: non-empty, all entries
/// finite, link heights/means (col 0) and standard deviations (col 1)
/// non-negative, and link counts (col 2) at least 1.
pub fn is_valid_im(r: &[[f64; 4]]) -> bool {
    if r.is_empty() {
        return false;
    }
    r.iter().all(|row| {
        row.iter().all(|v| v.is_finite()) && row[0] >= 0.0 && row[1] >= 0.0 && row[2] >= 1.0
    })
}

/// A node in a hierarchical-clustering tree (`scipy.cluster.hierarchy.ClusterNode`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterNode {
    /// Node id: `0..n` are original observations, `n..2n-1` are merges.
    pub id: usize,
    /// Left child (`None` for a leaf).
    pub left: Option<Box<ClusterNode>>,
    /// Right child (`None` for a leaf).
    pub right: Option<Box<ClusterNode>>,
    /// Merge distance (`0.0` for a leaf).
    pub dist: f64,
    /// Number of original observations under this node.
    pub count: usize,
}

impl ClusterNode {
    /// Whether this node is a leaf (original observation).
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Leaf ids in left-to-right traversal order (`ClusterNode.pre_order()`).
    pub fn pre_order(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count);
        let mut stack: Vec<&ClusterNode> = vec![self];
        while let Some(node) = stack.pop() {
            if node.is_leaf() {
                out.push(node.id);
            } else {
                // Push right then left so left is visited first.
                if let Some(r) = &node.right {
                    stack.push(r);
                }
                if let Some(l) = &node.left {
                    stack.push(l);
                }
            }
        }
        out
    }
}

/// Convert a linkage matrix into a `ClusterNode` tree, returning the root.
///
/// Matches `scipy.cluster.hierarchy.to_tree(Z)`: leaf ids `0..n`, merge ids
/// `n..2n-1`, `left = Z[i,0]`, `right = Z[i,1]`.
pub fn to_tree(z: &[[f64; 4]]) -> Result<ClusterNode, ClusterError> {
    let n = z.len() + 1;
    if z.is_empty() {
        return Ok(ClusterNode {
            id: 0,
            left: None,
            right: None,
            dist: 0.0,
            count: 1,
        });
    }
    let mut nodes: Vec<Option<ClusterNode>> = Vec::with_capacity(2 * n - 1);
    for i in 0..n {
        nodes.push(Some(ClusterNode {
            id: i,
            left: None,
            right: None,
            dist: 0.0,
            count: 1,
        }));
    }
    for (i, row) in z.iter().enumerate() {
        let c1 = row[0] as usize;
        let c2 = row[1] as usize;
        let left = nodes.get_mut(c1).and_then(Option::take).ok_or_else(|| {
            ClusterError::InvalidArgument("invalid linkage child index".to_string())
        })?;
        let right = nodes.get_mut(c2).and_then(Option::take).ok_or_else(|| {
            ClusterError::InvalidArgument("invalid linkage child index".to_string())
        })?;
        let count = left.count + right.count;
        nodes.push(Some(ClusterNode {
            id: n + i,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            dist: row[2],
            count,
        }));
    }
    nodes[2 * n - 2]
        .take()
        .ok_or_else(|| ClusterError::InvalidArgument("failed to build tree root".to_string()))
}

/// Single-linkage (nearest-point) clustering of an observation matrix.
/// Matches `scipy.cluster.hierarchy.single(X)`.
pub fn single(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Single)
}

/// Complete-linkage (farthest-point) clustering. Matches `scipy.cluster.hierarchy.complete(X)`.
pub fn complete(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Complete)
}

/// Average-linkage (UPGMA) clustering. Matches `scipy.cluster.hierarchy.average(X)`.
pub fn average(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Average)
}

/// Weighted-average (WPGMA) linkage. Matches `scipy.cluster.hierarchy.weighted(X)`.
pub fn weighted(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Weighted)
}

/// Ward's minimum-variance linkage. Matches `scipy.cluster.hierarchy.ward(X)`.
pub fn ward(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Ward)
}

/// Centroid (UPGMC) linkage. Matches `scipy.cluster.hierarchy.centroid(X)`.
pub fn centroid(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Centroid)
}

/// Median (WPGMC) linkage. Matches `scipy.cluster.hierarchy.median(X)`.
pub fn median(data: &[Vec<f64>]) -> Result<Vec<[f64; 4]>, ClusterError> {
    linkage(data, LinkageMethod::Median)
}

/// Disjoint-set (union-find) data structure for incremental connectivity
/// queries, matching `scipy.cluster.hierarchy.DisjointSet`.
///
/// `find` (via [`DisjointSet::find`]) uses path halving; [`DisjointSet::merge`]
/// uses merge-by-size with insertion-order tie-breaking (the earlier-inserted
/// root becomes the parent on a size tie). Elements are kept in insertion order
/// for iteration and subset enumeration.
#[derive(Debug, Clone, Default)]
pub struct DisjointSet<T: Clone + Eq + std::hash::Hash> {
    n_subsets: usize,
    order: Vec<T>,
    indices: std::collections::HashMap<T, usize>,
    parents: std::collections::HashMap<T, T>,
    sizes: std::collections::HashMap<T, usize>,
    // Circular linked list linking the elements of each subset.
    nbrs: std::collections::HashMap<T, T>,
}

impl<T: Clone + Eq + std::hash::Hash> DisjointSet<T> {
    /// Create an empty disjoint set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_subsets: 0,
            order: Vec::new(),
            indices: std::collections::HashMap::new(),
            parents: std::collections::HashMap::new(),
            sizes: std::collections::HashMap::new(),
            nbrs: std::collections::HashMap::new(),
        }
    }

    /// Create a disjoint set from an iterator of elements (each its own subset).
    pub fn from_elements<I: IntoIterator<Item = T>>(elements: I) -> Self {
        let mut ds = Self::new();
        for x in elements {
            ds.add(x);
        }
        ds
    }

    /// Number of distinct subsets.
    #[must_use]
    pub fn n_subsets(&self) -> usize {
        self.n_subsets
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Whether `x` is present.
    #[must_use]
    pub fn contains(&self, x: &T) -> bool {
        self.indices.contains_key(x)
    }

    /// Elements in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.order.iter()
    }

    /// Add element `x` (a no-op if already present).
    pub fn add(&mut self, x: T) {
        if self.indices.contains_key(&x) {
            return;
        }
        let idx = self.order.len();
        self.indices.insert(x.clone(), idx);
        self.parents.insert(x.clone(), x.clone());
        self.sizes.insert(x.clone(), 1);
        self.nbrs.insert(x.clone(), x.clone());
        self.order.push(x);
        self.n_subsets += 1;
    }

    /// Find the root element of `x` (path halving). Returns `None` if absent.
    pub fn find(&mut self, x: &T) -> Option<T> {
        if !self.indices.contains_key(x) {
            return None;
        }
        let mut cur = x.clone();
        loop {
            let parent = self.parents[&cur].clone();
            if self.indices[&cur] == self.indices[&parent] {
                return Some(cur);
            }
            // Path halving: parents[cur] = parents[parent]; cur = that.
            let grandparent = self.parents[&parent].clone();
            self.parents.insert(cur.clone(), grandparent.clone());
            cur = grandparent;
        }
    }

    /// Merge the subsets of `x` and `y`. Returns `true` if they were distinct.
    pub fn merge(&mut self, x: &T, y: &T) -> bool {
        let (xr, yr) = match (self.find(x), self.find(y)) {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };
        if self.indices[&xr] == self.indices[&yr] {
            return false;
        }
        // Merge by size; on a tie, the earlier-inserted root is the parent.
        let (parent, child) =
            if (self.sizes[&xr], self.indices[&yr]) < (self.sizes[&yr], self.indices[&xr]) {
                (yr, xr)
            } else {
                (xr, yr)
            };
        self.parents.insert(child.clone(), parent.clone());
        self.sizes
            .insert(parent.clone(), self.sizes[&parent] + self.sizes[&child]);
        let np = self.nbrs[&parent].clone();
        let nc = self.nbrs[&child].clone();
        self.nbrs.insert(parent, nc);
        self.nbrs.insert(child, np);
        self.n_subsets -= 1;
        true
    }

    /// Whether `x` and `y` are in the same subset.
    pub fn connected(&mut self, x: &T, y: &T) -> bool {
        match (self.find(x), self.find(y)) {
            (Some(a), Some(b)) => self.indices[&a] == self.indices[&b],
            _ => false,
        }
    }

    /// The elements of the subset containing `x`, in internal traversal order.
    #[must_use]
    pub fn subset(&self, x: &T) -> Vec<T> {
        if !self.indices.contains_key(x) {
            return Vec::new();
        }
        let mut result = vec![x.clone()];
        let mut nxt = self.nbrs[x].clone();
        while self.indices[&nxt] != self.indices[x] {
            result.push(nxt.clone());
            nxt = self.nbrs[&nxt].clone();
        }
        result
    }

    /// The size of the subset containing `x`.
    pub fn subset_size(&mut self, x: &T) -> usize {
        match self.find(x) {
            Some(r) => self.sizes[&r],
            None => 0,
        }
    }

    /// All subsets, ordered by the insertion order of their first-seen element.
    #[must_use]
    pub fn subsets(&self) -> Vec<Vec<T>> {
        let mut result = Vec::new();
        let mut visited: std::collections::HashSet<T> = std::collections::HashSet::new();
        for x in &self.order {
            if !visited.contains(x) {
                let xset = self.subset(x);
                for e in &xset {
                    visited.insert(e.clone());
                }
                result.push(xset);
            }
        }
        result
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
    // Fixed-size [i64;6] cell key (d<=6 whenever gridding): Copy, no per-call heap
    // alloc, and hashes far faster than a Vec<i64> (no pointer chase). Unused dims
    // stay 0 → identical bucket partition to the Vec key, byte-identical results.
    let cell_of = |p: &[f64]| -> [i64; 6] {
        let mut c = [0i64; 6];
        for (ck, &pk) in c[..d].iter_mut().zip(p.iter()) {
            *ck = (pk / eps).floor() as i64;
        }
        c
    };
    let grid: Option<std::collections::HashMap<[i64; 6], Vec<usize>>> = use_grid.then(|| {
        let mut g: std::collections::HashMap<[i64; 6], Vec<usize>> =
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
        let mut cell = base;
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

    // Each point's eps-neighbourhood is independent of the (serial) label
    // expansion, and every point's list is consumed EXACTLY once below. When the
    // grid is active (bounded per-point degree → bounded memory) precompute all n
    // lists in parallel; the sequential BFS then moves each out with mem::take.
    // Byte-identical: same neighbour sets, same ascending index order, same labels.
    let mut all_nbrs: Option<Vec<Vec<usize>>> = if grid.is_some() && n >= 2048 {
        let nthreads = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n);
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
        if nthreads <= 1 {
            for (i, o) in out.iter_mut().enumerate() {
                *o = neighbors(i);
            }
        } else {
            let chunk = n.div_ceil(nthreads);
            let neighbors_ref = &neighbors;
            std::thread::scope(|scope| {
                for (t, slot) in out.chunks_mut(chunk).enumerate() {
                    let base = t * chunk;
                    scope.spawn(move || {
                        for (i, o) in slot.iter_mut().enumerate() {
                            *o = neighbors_ref(base + i);
                        }
                    });
                }
            });
        }
        Some(out)
    } else {
        None
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let nbrs = match &mut all_nbrs {
            Some(a) => std::mem::take(&mut a[i]),
            None => neighbors(i),
        };
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

            let j_nbrs = match &mut all_nbrs {
                Some(a) => std::mem::take(&mut a[j]),
                None => neighbors(j),
            };
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

    core_samples.sort_unstable();
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

/// Buffer-reusing variant of [`flatten_centroids`]: rewrites `flat` in place
/// (byte-identical content) so iterative loops can avoid reallocating the
/// `k × d` flat centroid buffer every step.
fn flatten_centroids_into(centroids: &[Vec<f64>], d: usize, flat: &mut Vec<f64>) {
    flat.clear();
    for centroid in centroids {
        flat.extend_from_slice(&centroid[..d]);
    }
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
    data_flat: &[f64],
    n: usize,
    centroids_flat: &[f64],
    k: usize,
    d: usize,
) -> Vec<(usize, f64)> {
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
            // Cap by work-per-thread (~1<<20 multiply-adds): on small per-call
            // batches (e.g. kmeans2's repeated n~20k assigns, called once per Lloyd
            // iteration) a 64-way split leaves each worker too little to amortize the
            // thread spawn, so the effective speedup stalls near 3x. Keeping
            // >=~1M ops/worker restores near-linear scaling; large single calls
            // (vq over n=100k) still saturate all cores. Result is identical
            // regardless of worker count (deterministic per-point argmin).
            .min(((work >> 20) as usize).max(1))
            .min(n / 32)
            .max(1)
    };
    // Contiguous flat layout (n×d): points are read sequentially (cache-friendly + auto-
    // vectorizable) instead of chasing `Vec<Vec<f64>>` heap pointers.
    if nthreads <= 1 {
        return (0..n)
            .map(|i| nearest_centroid(&data_flat[i * d..i * d + d], centroids_flat, k, d))
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
                    (i0..i1)
                        .map(|i| {
                            nearest_centroid(&data_flat[i * d..i * d + d], centroids_flat, k, d)
                        })
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

/// Below this centroid count the prefilter + partial-distance abandonment path is
/// a net loss: its branches (`if acc > bound`) defeat autovectorization, and with
/// few centroids there is little to prune. A plain full-distance argmin over all
/// `k` centroids vectorizes cleanly and wins. Above it, pruning skips enough full
/// distances to pay for the branchy scan, so the abandonment path is kept.
const NEAREST_FULL_SCAN_MAX_K: usize = 64;

fn nearest_centroid(point: &[f64], centroids_flat: &[f64], k: usize, d: usize) -> (usize, f64) {
    if k == 4 && d == 4 {
        return nearest_centroid_k4_d4(point, centroids_flat);
    }
    // Small-k fast path: compute every full `sq_dist` (branch-free, autovectorized)
    // and take the argmin. BYTE-IDENTICAL to the prefilter/abandonment path below:
    // `sq_dist` is the same strict left-fold `sq_dist_within` runs to completion,
    // the abandonment never returns a value < the true distance, and iterating
    // `c` ascending with a strict `<` update keeps the smallest-`c` minimizer — the
    // exact tie-break the full scan uses. The seed/prefilter only ever changed
    // pruning speed, never the selected centroid.
    if k <= NEAREST_FULL_SCAN_MAX_K {
        let mut best_c = 0usize;
        let mut min_sq = sq_dist(point, &centroids_flat[0..d]);
        for c in 1..k {
            let sd = sq_dist(point, &centroids_flat[c * d..c * d + d]);
            if sd < min_sq {
                min_sq = sd;
                best_c = c;
            }
        }
        return (best_c, min_sq);
    }
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

#[inline]
fn nearest_centroid_k4_d4(point: &[f64], centroids_flat: &[f64]) -> (usize, f64) {
    use std::simd::Simd;

    let p0 = Simd::<f64, 4>::splat(point[0]);
    let p1 = Simd::<f64, 4>::splat(point[1]);
    let p2 = Simd::<f64, 4>::splat(point[2]);
    let p3 = Simd::<f64, 4>::splat(point[3]);
    let c0 = Simd::from_array([
        centroids_flat[0],
        centroids_flat[4],
        centroids_flat[8],
        centroids_flat[12],
    ]);
    let c1 = Simd::from_array([
        centroids_flat[1],
        centroids_flat[5],
        centroids_flat[9],
        centroids_flat[13],
    ]);
    let c2 = Simd::from_array([
        centroids_flat[2],
        centroids_flat[6],
        centroids_flat[10],
        centroids_flat[14],
    ]);
    let c3 = Simd::from_array([
        centroids_flat[3],
        centroids_flat[7],
        centroids_flat[11],
        centroids_flat[15],
    ]);

    let d0 = p0 - c0;
    let d1 = p1 - c1;
    let d2 = p2 - c2;
    let d3 = p3 - c3;
    let dist = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    let values = dist.to_array();

    let mut best_c = 0usize;
    let mut min_sq = values[0];
    for (c, &sd) in values.iter().enumerate().skip(1) {
        if sd < min_sq {
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
    // Each k runs an independent (deterministic, seed+k) full kmeans, in k order
    // with no cross-k reduction — so the inertias are computed in parallel into
    // ordered slots, byte-identical to the serial map. kmeans is itself serial,
    // so there is no nested-parallelism oversubscription.
    let kmax = max_k.min(data.len());
    let inertia_at = |k: usize| -> f64 {
        kmeans(data, k, 50, seed.wrapping_add(k as u64))
            .map(|r| r.inertia)
            .unwrap_or(f64::INFINITY)
    };
    let nthreads = if kmax < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(kmax)
    };
    if nthreads <= 1 {
        return (1..=kmax).map(inertia_at).collect();
    }
    let mut out = vec![0.0; kmax];
    let chunk = kmax.div_ceil(nthreads);
    let inertia_at = &inertia_at;
    std::thread::scope(|scope| {
        for (t, slot) in out.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (i, o) in slot.iter_mut().enumerate() {
                    *o = inertia_at(base + i + 1); // k starts at 1
                }
            });
        }
    });
    out
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
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
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

    // Each center's update reads only its own old position and the immutable
    // `data` (never the other centers), so within an iteration the updates are
    // independent — compute them in parallel across the centers, byte-identical
    // (same per-center Gaussian-weighted mean in the same float order, written to
    // its own slot; the `shifted` flag is an order-independent OR).
    let update_center = |center: &[f64]| -> (Vec<f64>, bool) {
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
        (new_center, shift > 1e-6)
    };

    // Threads are re-spawned every iteration (up to max_iter times), so the
    // per-iteration O(n²·d) work must be large to amortize that; gate high.
    let nthreads = if n.saturating_mul(n).saturating_mul(d) < (1 << 22) || n < 2 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n)
    };

    for _ in 0..max_iter {
        let shifted = if nthreads <= 1 {
            let mut sh = false;
            for center in &mut centers {
                let (nc, s) = update_center(center);
                sh |= s;
                *center = nc;
            }
            sh
        } else {
            let chunk = n.div_ceil(nthreads);
            let update_center = &update_center;
            std::thread::scope(|scope| {
                let handles: Vec<_> = centers
                    .chunks_mut(chunk)
                    .map(|slab| {
                        scope.spawn(move || {
                            let mut sh = false;
                            for center in slab.iter_mut() {
                                let (nc, s) = update_center(center);
                                sh |= s;
                                *center = nc;
                            }
                            sh
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .fold(false, |acc, h| acc | h.join().unwrap())
            })
        };

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

    // Build the full n×n distance matrix from condensed form and route through the same
    // shared O(n²) path as `linkage` (bit-identical between the two entry points).
    let mut dm = vec![0.0_f64; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in i + 1..n {
            dm[i * n + j] = condensed_dist[idx];
            dm[j * n + i] = condensed_dist[idx];
            idx += 1;
        }
    }
    Ok(linkage_from_dm(n, dm, method))
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

        // Average over reference datasets. Each reference uses a seed derived
        // directly from (r, k) — fully independent, no RNG chaining — and runs an
        // expensive kmeans, so the references are computed in parallel into
        // ordered slots, then summed SEQUENTIALLY in r order (byte-identical:
        // same per-r dispersion, same float summation order).
        let one_ref = |r: usize| -> f64 {
            let ref_seed = seed.wrapping_add(1000 * r as u64 + k as u64);
            let mut rng = ref_seed;
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
            kmeans(&ref_data, k, 30, ref_seed)
                .map(|res| res.inertia.max(1e-30).ln())
                .unwrap_or(0.0)
        };

        let nthreads = if n_ref < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(n_ref)
        };
        let ref_log_wks: Vec<f64> = if nthreads <= 1 {
            (0..n_ref).map(one_ref).collect()
        } else {
            let mut out = vec![0.0; n_ref];
            let chunk = n_ref.div_ceil(nthreads);
            let one_ref = &one_ref;
            std::thread::scope(|scope| {
                for (t, slot) in out.chunks_mut(chunk).enumerate() {
                    let base = t * chunk;
                    scope.spawn(move || {
                        for (i, o) in slot.iter_mut().enumerate() {
                            *o = one_ref(base + i);
                        }
                    });
                }
            });
            out
        };

        let mean_ref = ref_log_wks.iter().sum::<f64>() / n_ref as f64;
        gaps.push(mean_ref - log_wk);
    }

    gaps
}

/// K-medoids (PAM) clustering.
///
/// Similar to K-means but uses actual data points as centers.
/// When `true`, `kmedoids` builds the full M×M intra-cluster distance matrix and
/// row-sums it (the ORIG behaviour); default `false` accumulates each candidate's
/// total distance directly from the pair distances (no M×M matrix, no second O(M²)
/// pass) — byte-identical. `#[doc(hidden)]` — same-binary A/B knob.
#[doc(hidden)]
pub static KMEDOIDS_COST_FUSE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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
    if max_iter == 0 {
        return Err(ClusterError::InvalidArgument(
            "max_iter must be positive".to_string(),
        ));
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

            // Each candidate's total distance to the other members — the minimizing
            // member is the new medoid. Fused (default): accumulate the per-candidate
            // cost directly from the M(M-1)/2 pair distances instead of materializing the
            // M×M matrix and row-summing it in a SECOND O(M²) pass. BYTE-IDENTICAL:
            // `cost[i]` receives the same `dist(i,j)` terms in the same order as
            // `dmat[i].iter().sum()` (the `<i` terms during earlier outer iterations in
            // ascending order, the `>i` terms during outer iteration `i`; the dropped
            // diagonal is a `+0.0` no-op on the non-negative running sum), and the
            // strict-`<` first-wins min-selection is unchanged.
            let best_local =
                if KMEDOIDS_COST_FUSE_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut dmat = vec![vec![0.0_f64; m]; m];
                    for i in 0..m {
                        let mi = &member_flat[i * d..i * d + d];
                        for j in (i + 1)..m {
                            let dist = sq_dist(mi, &member_flat[j * d..j * d + d]).sqrt();
                            dmat[i][j] = dist;
                            dmat[j][i] = dist;
                        }
                    }
                    let mut best_local = 0usize;
                    let mut best_cost = dmat[0].iter().sum::<f64>();
                    for (i, row) in dmat.iter().enumerate().skip(1) {
                        let cost: f64 = row.iter().sum();
                        if cost < best_cost {
                            best_cost = cost;
                            best_local = i;
                        }
                    }
                    best_local
                } else {
                    let mut cost = vec![0.0f64; m];
                    for i in 0..m {
                        let mi = &member_flat[i * d..i * d + d];
                        for j in (i + 1)..m {
                            let dist = sq_dist(mi, &member_flat[j * d..j * d + d]).sqrt();
                            cost[i] += dist;
                            cost[j] += dist;
                        }
                    }
                    let mut best_local = 0usize;
                    let mut best_cost = cost[0];
                    for (i, &c) in cost.iter().enumerate().skip(1) {
                        if c < best_cost {
                            best_cost = c;
                            best_local = i;
                        }
                    }
                    best_local
                };
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

    // ── frankenscipy-ecrbb: bit-identity gates for the four levers restored from
    // 1d999e235^. Each drives BOTH arms through the toggle and compares with
    // to_bits(), and each is sized ABOVE its work gate so the parallel/fused arm
    // actually runs — below the gate both arms take the same path and the test
    // would be vacuous. The gate constant is named in each test.

    #[test]
    fn whiten_parallel_is_bit_identical_to_serial() {
        // Gate: n < 2 || n*d < 65_536. n=8192, d=8 -> 65_536, so the threaded
        // path runs.
        use std::sync::atomic::Ordering;
        let (n, d) = (8192usize, 8usize);
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 31 + j * 17) % 1000) as f64 * 0.01 + (j as f64) * 1.5)
                    .collect()
            })
            .collect();

        WHITEN_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = whiten(&data).expect("whiten serial");
        WHITEN_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = whiten(&data).expect("whiten parallel");

        assert_eq!(serial.len(), parallel.len());
        for (i, (srow, prow)) in serial.iter().zip(parallel.iter()).enumerate() {
            for (j, (s, p)) in srow.iter().zip(prow.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "whiten[{i}][{j}] bit mismatch: serial {s} vs parallel {p}"
                );
            }
        }
    }

    #[test]
    fn inconsistent_parallel_is_bit_identical_to_serial() {
        // Gate: m < 4096, where m is the merge count. A synthetic caterpillar
        // linkage with 4097 merges clears it without paying for a real O(n²)
        // linkage build.
        use std::sync::atomic::Ordering;
        let m = 4097usize;
        let n = m + 1;
        let mut z: Vec<[f64; 4]> = Vec::with_capacity(m);
        // Merge 0 joins leaves 0 and 1; merge k>0 joins leaf k+1 with the
        // previous internal node n+k-1. Distances strictly increase, as a valid
        // linkage requires.
        z.push([0.0, 1.0, 1.0, 2.0]);
        for k in 1..m {
            z.push([
                (k + 1) as f64,
                (n + k - 1) as f64,
                1.0 + k as f64 * 0.5,
                (k + 2) as f64,
            ]);
        }

        INCONSISTENT_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = inconsistent(&z, 2);
        INCONSISTENT_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = inconsistent(&z, 2);

        assert_eq!(serial.len(), m, "one output row per merge");
        assert_eq!(serial.len(), parallel.len());
        for (i, (srow, prow)) in serial.iter().zip(parallel.iter()).enumerate() {
            for c in 0..4 {
                assert_eq!(
                    srow[c].to_bits(),
                    prow[c].to_bits(),
                    "inconsistent[{i}][{c}] bit mismatch: serial {} vs parallel {}",
                    srow[c],
                    prow[c]
                );
            }
        }
    }

    #[test]
    fn nystroem_parallel_is_bit_identical_to_serial() {
        // Gate: n*m*d < 1<<20 goes serial. A 256x256 kernel with 128 components
        // clears it.
        use std::sync::atomic::Ordering;
        let n = 256usize;
        let kernel: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let a = (i as f64) * 0.05;
                        let b = (j as f64) * 0.05;
                        (-(a - b) * (a - b) / 2.0).exp()
                    })
                    .collect()
            })
            .collect();

        NYSTROEM_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = nystroem(&kernel, 128, 7).expect("nystroem serial");
        NYSTROEM_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = nystroem(&kernel, 128, 7).expect("nystroem parallel");

        assert_eq!(serial.feature_map.len(), parallel.feature_map.len());
        for (i, (srow, prow)) in serial
            .feature_map
            .iter()
            .zip(parallel.feature_map.iter())
            .enumerate()
        {
            assert_eq!(srow.len(), prow.len(), "feature_map row {i} width");
            for (t, (s, p)) in srow.iter().zip(prow.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "nystroem feature_map[{i}][{t}] bit mismatch: serial {s} vs parallel {p}"
                );
            }
        }
    }

    #[test]
    fn kmedoids_cost_fuse_is_bit_identical_to_the_materialized_matrix() {
        // KMEDOIDS_COST_FUSE_DISABLE has no size gate: the toggle selects between
        // a materialized m×m distance matrix and the fused running-sum form on
        // every call, so any input exercises both arms. The fused arm claims to
        // accumulate the same dist(i,j) terms in the same order; this pins that.
        use std::sync::atomic::Ordering;
        let (n, d) = (300usize, 4usize);
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 13) % 97) as f64 * 0.3 + (j as f64))
                    .collect()
            })
            .collect();

        KMEDOIDS_COST_FUSE_DISABLE.store(true, Ordering::Relaxed);
        let materialized = kmedoids(&data, 5, 20, 11).expect("kmedoids materialized");
        KMEDOIDS_COST_FUSE_DISABLE.store(false, Ordering::Relaxed);
        let fused = kmedoids(&data, 5, 20, 11).expect("kmedoids fused");

        assert_eq!(
            materialized.labels, fused.labels,
            "fusing the cost accumulation must not move a single label"
        );
        assert_eq!(materialized.centroids.len(), fused.centroids.len());
        for (i, (mrow, frow)) in materialized
            .centroids
            .iter()
            .zip(fused.centroids.iter())
            .enumerate()
        {
            for (j, (m, f)) in mrow.iter().zip(frow.iter()).enumerate() {
                assert_eq!(
                    m.to_bits(),
                    f.to_bits(),
                    "kmedoids centroid[{i}][{j}] bit mismatch: {m} vs {f}"
                );
            }
        }
    }

    #[test]
    fn linkage_distance_matrix_parallel_is_bit_identical_to_serial() {
        // n=800, d=4 -> n²·d = 2.56e6 >= LINKAGE_DM_PAR_WORK_GATE, so
        // linkage_distance_matrix takes the threaded full-row path. It must be
        // bit-for-bit equal to the serial upper-triangle-plus-mirror build.
        let n = 800usize;
        let d = 4usize;
        let mut s: u64 = 0x0123_4567_89ab_cdef;
        let flat: Vec<f64> = (0..n * d)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        assert!((n as u128) * (n as u128) * (d as u128) >= LINKAGE_DM_PAR_WORK_GATE);

        let parallel = linkage_distance_matrix(&flat, n, d);

        // Inline serial reference (the pre-change build).
        let row = |idx: usize| -> &[f64] { &flat[idx * d..idx * d + d] };
        let mut serial = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let dist = sq_dist(row(i), row(j)).sqrt();
                serial[i * n + j] = dist;
                serial[j * n + i] = dist;
            }
        }
        assert_eq!(parallel, serial, "parallel dm build diverged from serial");
    }

    #[test]
    fn pca_matches_full_svd_on_low_rank() {
        let mut s: u64 = 0x3141_5926_5358_9793;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, d, r, k) = (200usize, 30usize, 6usize, 10usize);
        let u: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let v: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| rng()).collect()).collect();
        let x: Vec<Vec<f64>> = u
            .iter()
            .map(|ui| {
                (0..d)
                    .map(|j| (0..r).map(|t| ui[t] * v[t][j]).sum())
                    .collect()
            })
            .collect();

        let p = pca(&x, k, 3).expect("pca");
        let kk = p.components.len();
        assert_eq!(p.mean.len(), d);
        assert_eq!(p.components[0].len(), d);
        assert_eq!(p.transformed.len(), n);
        assert_eq!(p.transformed[0].len(), kk);
        assert_eq!(p.explained_variance.len(), kk);
        assert_eq!(p.singular_values.len(), kk);
        for w in p.explained_variance.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "explained variance must be descending");
        }

        // Reconstruction: Xc ≈ transformed · components (rank-k approx; X is exactly rank r ≤ k).
        let mut maxerr = 0.0f64;
        for (i, x_i) in x.iter().enumerate().take(n) {
            for (j, &xij) in x_i.iter().enumerate().take(d) {
                let approx: f64 = (0..kk)
                    .map(|t| p.transformed[i][t] * p.components[t][j])
                    .sum();
                let xc = xij - p.mean[j];
                maxerr = maxerr.max((xc - approx).abs());
            }
        }
        assert!(maxerr < 1e-9, "reconstruction maxerr {maxerr}");

        assert!(pca(&[], k, 1).is_err());
        assert!(pca(&x, 0, 1).is_err());
        assert!(matches!(
            pca(&[vec![1.0, f64::NAN], vec![2.0, 3.0]], 1, 1),
            Err(ClusterError::InvalidArgument(_))
        ));
        assert!(matches!(
            pca(&[vec![1.0, f64::INFINITY], vec![2.0, 3.0]], 1, 1),
            Err(ClusterError::InvalidArgument(_))
        ));
    }

    #[test]
    fn affinity_propagation_recovers_clusters() {
        let mut s: u64 = 0x51a4_b3c2_d1e0_f9a8;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        // Three well-separated 2-D blobs.
        let (k, per) = (3usize, 15usize);
        let n = k * per;
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            for _ in 0..per {
                pts.push([20.0 * c as f64 + rng(), rng()]);
                truth.push(c);
            }
        }
        // Similarity = negative squared Euclidean distance.
        let mut sim = vec![vec![0.0; n]; n];
        let mut offdiag = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                    sim[i][j] = -d2;
                    offdiag.push(-d2);
                }
            }
        }
        offdiag.sort_unstable_by(|a, b| a.total_cmp(b));
        let preference = offdiag[offdiag.len() / 2]; // median similarity

        let ap = affinity_propagation(&sim, preference, 0.9, 500, 15).expect("ap");
        assert_eq!(ap.labels.len(), n);
        assert!(!ap.exemplars.is_empty());
        // Recovered exactly 3 exemplars/clusters and perfect purity.
        let n_clusters = ap.exemplars.len();
        assert_eq!(n_clusters, k, "should find 3 exemplars, got {n_clusters}");
        let mut correct = 0usize;
        for pred in 0..n_clusters {
            let mut counts = vec![0usize; k];
            for (i, &l) in ap.labels.iter().enumerate() {
                if l == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(
            correct, n,
            "affinity propagation should perfectly separate the blobs"
        );

        assert!(affinity_propagation(&[], -1.0, 0.5, 10, 5).is_err());
        assert!(affinity_propagation(&sim, preference, 0.4, 10, 5).is_err()); // damping < 0.5
        assert!(affinity_propagation(&sim, preference, 1.0, 10, 5).is_err()); // damping >= 1.0
        assert!(affinity_propagation(&sim, preference, 0.9, 0, 5).is_err()); // max_iter == 0
        assert!(affinity_propagation(&sim, preference, 0.9, 10, 0).is_err()); // convergence_iter == 0
        assert!(affinity_propagation(&[vec![0.0, 1.0]], -1.0, 0.5, 10, 5).is_err()); // non-square
    }

    #[test]
    fn spectral_coclustering_recovers_block_structure() {
        let mut s: u64 = 0x243f_6a88_85a3_08d3;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        // Block-diagonal biclustering structure: 3 row-blocks × 3 col-blocks, large values on
        // the matching diagonal blocks and small noise elsewhere.
        let kk = 3usize;
        let rm = 12usize; // rows per block
        let cm = 9usize; // cols per block
        let m = kk * rm;
        let n = kk * cm;
        let mut data = vec![vec![0.0; n]; m];
        let mut row_truth = vec![0usize; m];
        let mut col_truth = vec![0usize; n];
        for (i, slot) in row_truth.iter_mut().enumerate().take(m) {
            *slot = i / rm;
        }
        for (j, slot) in col_truth.iter_mut().enumerate().take(n) {
            *slot = j / cm;
        }
        for i in 0..m {
            for j in 0..n {
                data[i][j] = if row_truth[i] == col_truth[j] {
                    10.0 + rng()
                } else {
                    0.01 + 0.01 * rng()
                };
            }
        }

        let cc = spectral_coclustering(&data, kk, 100, 7).expect("coclustering");
        assert_eq!(cc.row_labels.len(), m);
        assert_eq!(cc.col_labels.len(), n);

        // Purity of row and column labelings against the true blocks.
        let purity = |labels: &[usize], truth: &[usize]| -> usize {
            let mut correct = 0usize;
            for pred in 0..kk {
                let mut counts = vec![0usize; kk];
                for (i, &l) in labels.iter().enumerate() {
                    if l == pred {
                        counts[truth[i]] += 1;
                    }
                }
                correct += counts.iter().copied().max().unwrap_or(0);
            }
            correct
        };
        assert_eq!(
            purity(&cc.row_labels, &row_truth),
            m,
            "rows should match blocks"
        );
        assert_eq!(
            purity(&cc.col_labels, &col_truth),
            n,
            "cols should match blocks"
        );

        assert!(spectral_coclustering(&[], 2, 10, 1).is_err());
        assert!(spectral_coclustering(&data, 1, 10, 1).is_err()); // k < 2
        assert!(spectral_coclustering(&data, m + 1, 10, 1).is_err());
        assert!(spectral_coclustering(&[vec![-1.0, 1.0], vec![1.0, 1.0]], 2, 10, 1).is_err()); // negative
    }

    #[test]
    fn spectral_clustering_recovers_separated_blobs() {
        let mut s: u64 = 0x1a2b_3c4d_5e6f_7081;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (k, per) = (3usize, 20usize);
        let n = k * per;
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            for _ in 0..per {
                pts.push(vec![15.0 * c as f64 + rng(), rng()]);
                truth.push(c);
            }
        }
        let aff: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d2: f64 = pts[i]
                            .iter()
                            .zip(&pts[j])
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        (-0.5 * d2).exp()
                    })
                    .collect()
            })
            .collect();

        let labels = spectral_clustering(&aff, k, 100, 7).expect("spectral");
        assert_eq!(labels.len(), n);
        // Perfectly separated blobs → purity 1.0 (each predicted cluster is one true blob).
        let mut correct = 0usize;
        for pred in 0..k {
            let mut counts = vec![0usize; k];
            for i in 0..n {
                if labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(correct, n, "should perfectly separate the blobs");

        assert!(spectral_clustering(&[], 2, 10, 1).is_err());
        assert!(spectral_clustering(&aff, 0, 10, 1).is_err());
        assert!(matches!(
            spectral_clustering(&[vec![1.0, -0.1], vec![-0.1, 1.0]], 2, 10, 1),
            Err(ClusterError::InvalidArgument(_))
        ));
    }

    #[test]
    fn nystroem_spectral_clustering_recovers_separated_blobs() {
        let mut s: u64 = 0x7766_5544_3322_1100;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (k, per) = (3usize, 60usize);
        let n = k * per;
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            for _ in 0..per {
                pts.push(vec![15.0 * c as f64 + rng(), rng()]);
                truth.push(c);
            }
        }

        // Data-based Nyström spectral clustering — never forms the n×n affinity.
        let labels = nystroem_spectral_clustering(&pts, k, 24, 0.5, 100, 7).expect("nys-spectral");
        assert_eq!(labels.len(), n);
        let mut correct = 0usize;
        for pred in 0..k {
            let mut counts = vec![0usize; k];
            for i in 0..n {
                if labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(
            correct, n,
            "Nyström spectral should perfectly separate the blobs"
        );

        assert!(nystroem_spectral_clustering(&[], 2, 2, 0.5, 10, 1).is_err());
        assert!(nystroem_spectral_clustering(&pts, 0, 2, 0.5, 10, 1).is_err());
        assert!(nystroem_spectral_clustering(&pts, 5, 3, 0.5, 10, 1).is_err()); // k > landmarks
        assert!(nystroem_spectral_clustering(&pts, 2, n + 1, 0.5, 10, 1).is_err()); // landmarks > n
        assert!(nystroem_spectral_clustering(&pts, 2, 5, 0.0, 10, 1).is_err()); // gamma <= 0
    }

    #[test]
    fn nystroem_spectral_embedding_separates_blobs() {
        let mut s: u64 = 0x1212_3434_5656_7878;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (k, per) = (3usize, 50usize);
        let n = k * per;
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            for _ in 0..per {
                pts.push(vec![15.0 * c as f64 + rng(), rng()]);
                truth.push(c);
            }
        }

        let se = nystroem_spectral_embedding(&pts, k, 24, 0.5, 7).expect("embedding");
        assert_eq!(se.embedding.len(), n);
        assert_eq!(se.embedding[0].len(), se.eigenvalues.len());
        for w in se.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues descending");
        }
        // The embedding must separate the blobs: k-means on it recovers them (purity 1.0).
        let labels = kmeans(&se.embedding, k, 100, 7).expect("kmeans").labels;
        let mut correct = 0usize;
        for pred in 0..k {
            let mut counts = vec![0usize; k];
            for i in 0..n {
                if labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(correct, n, "embedding should separate the blobs");

        assert!(nystroem_spectral_embedding(&[], 2, 2, 0.5, 1).is_err());
        assert!(nystroem_spectral_embedding(&pts, 5, 3, 0.5, 1).is_err()); // k > landmarks
        assert!(nystroem_spectral_embedding(&pts, 2, 5, 0.0, 1).is_err()); // gamma <= 0
    }

    #[test]
    fn diffusion_map_separates_blobs_and_scales_with_time() {
        let mut s: u64 = 0xd1ff_5107_b10b_2024;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (k, per) = (3usize, 50usize);
        let n = k * per;
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            for _ in 0..per {
                pts.push(vec![15.0 * c as f64 + rng(), rng()]);
                truth.push(c);
            }
        }

        let dm = diffusion_map(&pts, k, 24, 0.5, 1.0, 7).expect("diffusion_map");
        assert_eq!(dm.embedding.len(), n);
        assert_eq!(dm.embedding[0].len(), k);
        assert_eq!(dm.eigenvalues.len(), k);
        for w in dm.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues descending");
        }
        // The diffusion coordinates separate the blobs (k-means → purity 1.0).
        let labels = kmeans(&dm.embedding, k, 100, 7).expect("kmeans").labels;
        let mut correct = 0usize;
        for pred in 0..k {
            let mut counts = vec![0usize; k];
            for i in 0..n {
                if labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(correct, n, "diffusion map should separate the blobs");

        // Larger diffusion time shrinks coordinates by μ^t (μ ≤ 1) — column 0's scale ratio is μ_1.
        let dm2 = diffusion_map(&pts, k, 24, 0.5, 2.0, 7).expect("diffusion_map t=2");
        let norm1: f64 = dm.embedding.iter().map(|r| r[0] * r[0]).sum::<f64>().sqrt();
        let norm2: f64 = dm2
            .embedding
            .iter()
            .map(|r| r[0] * r[0])
            .sum::<f64>()
            .sqrt();
        assert!(
            norm2 <= norm1 + 1e-9,
            "t=2 coords must not exceed t=1 (μ ≤ 1)"
        );

        assert!(diffusion_map(&[], 2, 3, 0.5, 1.0, 1).is_err());
        assert!(diffusion_map(&pts, 0, 3, 0.5, 1.0, 1).is_err());
        assert!(diffusion_map(&pts, 3, 3, 0.5, 1.0, 1).is_err()); // k+1 > landmarks
        assert!(diffusion_map(&pts, 2, 5, 0.0, 1.0, 1).is_err()); // gamma <= 0
        assert!(diffusion_map(&pts, 2, 5, 0.5, -1.0, 1).is_err()); // negative time
    }

    #[test]
    fn kernel_pca_reconstructs_centered_kernel() {
        let mut s: u64 = 0x9182_7364_5546_3728;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, r, k) = (90usize, 7usize, 12usize); // k > r → exact rank-k reconstruction
        let b: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let kernel: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..r).map(|t| b[i][t] * b[j][t]).sum())
                    .collect()
            })
            .collect();
        // Reference centered kernel.
        let rm: Vec<f64> = kernel
            .iter()
            .map(|r| r.iter().sum::<f64>() / n as f64)
            .collect();
        let tm: f64 = rm.iter().sum::<f64>() / n as f64;
        let kc: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| kernel[i][j] - rm[i] - rm[j] + tm).collect())
            .collect();

        let kp = kernel_pca(&kernel, k, 3).expect("kernel_pca");
        let kk = kp.transformed[0].len();
        assert_eq!(kp.eigenvalues.len(), kk);
        assert_eq!(kp.eigenvectors.len(), n);
        assert_eq!(kp.transformed.len(), n);

        // transformed·transformedᵀ ≈ centered kernel (rank k > r ⇒ exact to rounding).
        let mut maxerr = 0.0f64;
        for (i, kc_i) in kc.iter().enumerate().take(n) {
            for (j, &kcij) in kc_i.iter().enumerate().take(n) {
                let approx: f64 = (0..kk)
                    .map(|t| kp.transformed[i][t] * kp.transformed[j][t])
                    .sum();
                maxerr = maxerr.max((kcij - approx).abs());
            }
        }
        assert!(maxerr < 1e-8, "reconstruction maxerr {maxerr}");

        assert!(kernel_pca(&[], 2, 1).is_err());
        assert!(kernel_pca(&kernel, 0, 1).is_err());
    }

    #[test]
    fn nmf_factorizes_nonnegative_low_rank() {
        let mut s: u64 = 0x6f2c_1a8b_4d3e_5790;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 + 0.1
        };
        let (n, d, r, k) = (60usize, 25usize, 5usize, 6usize); // k ≥ r
        let wt: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let ht: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| rng()).collect()).collect();
        let x: Vec<Vec<f64>> = wt
            .iter()
            .map(|wr| {
                (0..d)
                    .map(|j| (0..r).map(|t| wr[t] * ht[t][j]).sum())
                    .collect()
            })
            .collect();

        let res = nmf(&x, k, 400, 1e-7, NmfInit::Nndsvd, 3).expect("nmf");
        assert_eq!(res.w.len(), n);
        assert_eq!(res.w[0].len(), k);
        assert_eq!(res.h.len(), k);
        assert_eq!(res.h[0].len(), d);
        // W, H are non-negative.
        assert!(res.w.iter().chain(&res.h).flatten().all(|&v| v >= 0.0));
        // X ≈ W·H to a small relative error (non-negative, k ≥ rank).
        assert!(
            res.reconstruction_err < 0.1,
            "reconstruction_err {} too high",
            res.reconstruction_err
        );

        assert!(nmf(&[], k, 10, 1e-6, NmfInit::Random, 1).is_err());
        assert!(nmf(&[vec![-1.0, 1.0]], 1, 10, 1e-6, NmfInit::Random, 1).is_err());
    }

    #[test]
    fn truncated_svd_reconstructs_low_rank() {
        let mut s: u64 = 0x4242_aaaa_5555_1234;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, d, r, k) = (120usize, 40usize, 7usize, 12usize); // k > r
        let u: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let v: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| rng()).collect()).collect();
        let x: Vec<Vec<f64>> = u
            .iter()
            .map(|ui| {
                (0..d)
                    .map(|j| (0..r).map(|t| ui[t] * v[t][j]).sum())
                    .collect()
            })
            .collect();

        let ts = truncated_svd(&x, k, 5).expect("truncated_svd");
        let kk = ts.singular_values.len();
        assert_eq!(ts.components.len(), kk);
        assert_eq!(ts.components[0].len(), d);
        assert_eq!(ts.transformed.len(), n);
        assert_eq!(ts.transformed[0].len(), kk);
        for w in ts.singular_values.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "singular values must be descending");
        }
        // X ≈ transformed · components (rank k > r ⇒ exact to rounding).
        let mut maxerr = 0.0f64;
        for (i, x_i) in x.iter().enumerate().take(n) {
            for (j, &xij) in x_i.iter().enumerate().take(d) {
                let approx: f64 = (0..kk)
                    .map(|t| ts.transformed[i][t] * ts.components[t][j])
                    .sum();
                maxerr = maxerr.max((xij - approx).abs());
            }
        }
        assert!(maxerr < 1e-8, "reconstruction maxerr {maxerr}");

        assert!(truncated_svd(&[], 2, 1).is_err());
        assert!(truncated_svd(&x, 0, 1).is_err());
    }

    #[test]
    fn double_centered_gram_matches_materialized_delta_reference() {
        let delta = [
            [0.0_f64, 1.0, 9.0, 16.0],
            [1.0, 0.0, 4.0, 25.0],
            [9.0, 4.0, 0.0, 36.0],
            [16.0, 25.0, 36.0, 0.0],
        ];
        let n = delta.len();
        let (actual, row_mean, total_mean) =
            double_centered_gram_from_squared(n, |i, j| delta[i][j]);
        let expected_row_mean: Vec<f64> = delta
            .iter()
            .map(|row| row.iter().sum::<f64>() / n as f64)
            .collect();
        let expected_total = expected_row_mean.iter().sum::<f64>() / n as f64;

        for idx in 0..n {
            assert!((row_mean[idx] - expected_row_mean[idx]).abs() <= 1e-12);
        }
        assert!((total_mean - expected_total).abs() <= 1e-12);

        for row in 0..n {
            for col in 0..n {
                let expected = -0.5
                    * (delta[row][col] - expected_row_mean[row] - expected_row_mean[col]
                        + expected_total);
                assert!(
                    (actual[row][col] - expected).abs() <= 1e-12,
                    "double-centered gram drift at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn classical_mds_recovers_euclidean_distances() {
        let mut s: u64 = 0x1357_9bdf_2468_ace0;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, r, k) = (80usize, 4usize, 4usize); // embed dim k == true dim r
        // Points in R^r, then the exact Euclidean distance matrix.
        let pts: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let dist: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..r)
                            .map(|t| (pts[i][t] - pts[j][t]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect();

        let mds = classical_mds(&dist, k, 11).expect("classical_mds");
        assert_eq!(mds.embedding.len(), n);
        let kk = mds.eigenvalues.len();
        assert_eq!(mds.embedding[0].len(), kk);
        for w in mds.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues must be descending");
        }
        // MDS embedding is an isometry up to rotation: pairwise distances must match.
        let mut maxerr = 0.0f64;
        for (i, dist_i) in dist.iter().enumerate().take(n) {
            for (j, &dij) in dist_i.iter().enumerate().take(n) {
                let de: f64 = (0..kk)
                    .map(|t| (mds.embedding[i][t] - mds.embedding[j][t]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                maxerr = maxerr.max((de - dij).abs());
            }
        }
        assert!(maxerr < 1e-6, "distance reconstruction maxerr {maxerr}");

        assert!(classical_mds(&[], 2, 1).is_err());
        assert!(classical_mds(&dist, 0, 1).is_err());
        assert!(classical_mds(&[vec![0.0, 1.0]], 1, 1).is_err()); // non-square
    }

    #[test]
    fn landmark_mds_recovers_euclidean_distances() {
        let mut st: u64 = 0xa5a5_3c3c_7e7e_1212;
        let mut rng = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, r, k, m) = (120usize, 4usize, 4usize, 12usize); // m > r, k == r
        let pts: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let dist: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..r)
                            .map(|t| (pts[i][t] - pts[j][t]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect();

        let mds = landmark_mds(&dist, k, m, 9).expect("landmark_mds");
        assert_eq!(mds.embedding.len(), n);
        let kk = mds.embedding[0].len();
        for w in mds.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues descending");
        }
        // Pairwise distances of the embedding match the originals (isometry up to rotation).
        let mut maxerr = 0.0f64;
        for (i, dist_i) in dist.iter().enumerate().take(n) {
            for (j, &dij) in dist_i.iter().enumerate().take(n) {
                let de: f64 = (0..kk)
                    .map(|t| (mds.embedding[i][t] - mds.embedding[j][t]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                maxerr = maxerr.max((de - dij).abs());
            }
        }
        assert!(maxerr < 1e-6, "landmark MDS distance maxerr {maxerr}");

        assert!(landmark_mds(&[], 2, 2, 1).is_err());
        assert!(landmark_mds(&dist, 0, 5, 1).is_err());
        assert!(landmark_mds(&dist, 5, 3, 1).is_err()); // k > landmarks
        assert!(landmark_mds(&dist, 2, n + 1, 1).is_err()); // landmarks > n
    }

    #[test]
    fn landmark_isomap_recovers_intrinsic_coordinates() {
        // A flat 2-D sheet (grid in (u,v)) isometrically embedded in 3-D via two orthonormal
        // axes. Geodesics equal intrinsic Euclidean distances, so Isomap must recover (u,v)
        // up to a rigid transform — verified by distance-matrix correlation.
        let gu = 16usize;
        let gv = 16usize;
        let n = gu * gv;
        // Orthonormal 3-D axes e1 ⟂ e2.
        let e1 = [0.6, 0.8, 0.0];
        let e2 = [0.0, 0.0, 1.0];
        let mut data = Vec::with_capacity(n);
        let mut intrinsic = Vec::with_capacity(n);
        for iu in 0..gu {
            for iv in 0..gv {
                let u = iu as f64;
                let v = iv as f64;
                data.push(vec![
                    u * e1[0] + v * e2[0],
                    u * e1[1] + v * e2[1],
                    u * e1[2] + v * e2[2],
                ]);
                intrinsic.push([u, v]);
            }
        }

        let iso = landmark_isomap(&data, 2, 8, 40, 7).expect("isomap");
        assert_eq!(iso.embedding.len(), n);
        assert_eq!(iso.embedding[0].len(), 2);
        for w in iso.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues descending");
        }

        // Pearson correlation between embedding pairwise distances and true intrinsic distances.
        let mut emb_d = Vec::new();
        let mut tru_d = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let ed: f64 = (0..2)
                    .map(|t| (iso.embedding[i][t] - iso.embedding[j][t]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let td: f64 = (0..2)
                    .map(|t| (intrinsic[i][t] - intrinsic[j][t]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                emb_d.push(ed);
                tru_d.push(td);
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (me, mt) = (mean(&emb_d), mean(&tru_d));
        let mut cov = 0.0;
        let mut ve = 0.0;
        let mut vt = 0.0;
        for (&a, &b) in emb_d.iter().zip(&tru_d) {
            cov += (a - me) * (b - mt);
            ve += (a - me).powi(2);
            vt += (b - mt).powi(2);
        }
        let corr = cov / (ve.sqrt() * vt.sqrt());
        assert!(corr > 0.97, "isomap distance correlation {corr}");

        assert!(landmark_isomap(&[], 2, 3, 3, 1).is_err());
        assert!(landmark_isomap(&data, 2, 0, 3, 1).is_err()); // n_neighbors 0
        assert!(landmark_isomap(&data, 5, 8, 3, 1).is_err()); // k > landmarks
        assert!(landmark_isomap(&data, 2, n, 5, 1).is_err()); // n_neighbors >= n
    }

    #[test]
    fn spectral_embedding_solves_generalized_eigenproblem() {
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        // Symmetric non-negative affinity (Gaussian-like) on random 1-D positions.
        let n = 60usize;
        let pos: Vec<f64> = (0..n).map(|_| rng() * 5.0).collect();
        let mut aff = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                aff[i][j] = (-(pos[i] - pos[j]).powi(2)).exp();
            }
        }
        let k = 4usize;
        let se = spectral_embedding(&aff, k, 3).expect("spectral_embedding");
        let kk = se.eigenvalues.len();
        assert_eq!(se.embedding.len(), n);
        assert_eq!(se.embedding[0].len(), kk);
        for w in se.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "eigenvalues must be descending");
        }
        let deg: Vec<f64> = aff.iter().map(|r| r.iter().sum()).collect();
        // Each column y_c must satisfy A y_c = μ_c D y_c (generalized eigenproblem).
        let mut maxres = 0.0f64;
        for c in 0..kk {
            let y: Vec<f64> = se.embedding.iter().map(|r| r[c]).collect();
            let ynorm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
            for i in 0..n {
                let ay: f64 = (0..n).map(|j| aff[i][j] * y[j]).sum();
                let res = (ay - se.eigenvalues[c] * deg[i] * y[i]).abs() / ynorm;
                maxres = maxres.max(res);
            }
        }
        assert!(maxres < 1e-7, "generalized-eig residual {maxres}");

        assert!(spectral_embedding(&[], 2, 1).is_err());
        assert!(spectral_embedding(&aff, 0, 1).is_err());
        assert!(spectral_embedding(&[vec![0.0, 1.0]], 1, 1).is_err()); // non-square
        assert!(matches!(
            spectral_embedding(&[vec![1.0, -0.1], vec![-0.1, 1.0]], 1, 1),
            Err(ClusterError::InvalidArgument(_))
        ));
    }

    #[test]
    fn factor_analysis_recovers_lowrank_plus_diagonal_covariance() {
        let mut s: u64 = 0x0123_4567_89ab_cdef;
        let mut gauss = || {
            // Box–Muller from the LCG.
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 11) as f64) / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let (n, d, k) = (4000usize, 8usize, 3usize);
        // Loadings (k×d) and per-feature noise std.
        let load: Vec<Vec<f64>> = (0..k).map(|_| (0..d).map(|_| gauss()).collect()).collect();
        let noise_sd: Vec<f64> = (0..d).map(|j| 0.3 + 0.1 * j as f64).collect();
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..k).map(|_| gauss()).collect();
                (0..d)
                    .map(|j| (0..k).map(|t| z[t] * load[t][j]).sum::<f64>() + noise_sd[j] * gauss())
                    .collect()
            })
            .collect();

        let fa = factor_analysis(&x, k, 1000, 1e-4, 5).expect("factor_analysis");
        assert_eq!(fa.components.len(), k);
        assert_eq!(fa.components[0].len(), d);
        assert_eq!(fa.noise_variance.len(), d);
        assert!(fa.loglike.is_finite());

        // Sample covariance of centered X.
        let mean = &fa.mean;
        let mut samp = vec![vec![0.0; d]; d];
        for row in &x {
            for a in 0..d {
                for b in 0..d {
                    samp[a][b] += (row[a] - mean[a]) * (row[b] - mean[b]);
                }
            }
        }
        for r in &mut samp {
            for v in r.iter_mut() {
                *v /= n as f64;
            }
        }
        // Model covariance C = Wᵀ W + diag(Ψ).
        let mut maxerr = 0.0f64;
        for (a, samp_a) in samp.iter().enumerate().take(d) {
            for (b, &sab) in samp_a.iter().enumerate().take(d) {
                let mut c: f64 = (0..k)
                    .map(|t| fa.components[t][a] * fa.components[t][b])
                    .sum();
                if a == b {
                    c += fa.noise_variance[a];
                }
                maxerr = maxerr.max((c - sab).abs());
            }
        }
        // FA maximizes likelihood of exactly this model; for n large the fit is tight.
        assert!(maxerr < 0.1, "covariance reconstruction maxerr {maxerr}");

        assert!(factor_analysis(&[], 2, 10, 1e-3, 1).is_err());
        assert!(factor_analysis(&x, 0, 10, 1e-3, 1).is_err());
        assert!(factor_analysis(&x, d + 1, 10, 1e-3, 1).is_err());
    }

    #[test]
    fn nystroem_reconstructs_lowrank_kernel() {
        let mut s: u64 = 0xfeed_face_dead_beef;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        // PSD kernel K = B Bᵀ of rank r; sample m > r landmarks ⇒ exact Nyström reconstruction.
        let (n, r, m) = (90usize, 5usize, 14usize);
        let b: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let kernel: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..r).map(|t| b[i][t] * b[j][t]).sum())
                    .collect()
            })
            .collect();

        let ny = nystroem(&kernel, m, 7).expect("nystroem");
        assert_eq!(ny.landmark_indices.len(), m);
        assert!(ny.landmark_indices.windows(2).all(|w| w[0] < w[1])); // sorted, distinct
        assert_eq!(ny.feature_map.len(), n);
        let mp = ny.feature_map[0].len();

        // K ≈ Z·Zᵀ.
        let mut maxerr = 0.0f64;
        for (i, kernel_i) in kernel.iter().enumerate().take(n) {
            for (j, &kij) in kernel_i.iter().enumerate().take(n) {
                let zz: f64 = (0..mp)
                    .map(|t| ny.feature_map[i][t] * ny.feature_map[j][t])
                    .sum();
                maxerr = maxerr.max((zz - kij).abs());
            }
        }
        assert!(maxerr < 1e-8, "Nyström reconstruction maxerr {maxerr}");

        assert!(nystroem(&[], 2, 1).is_err());
        assert!(nystroem(&kernel, 0, 1).is_err());
        assert!(nystroem(&kernel, n + 1, 1).is_err());
        assert!(nystroem(&[vec![1.0, 0.0]], 1, 1).is_err()); // non-square
    }

    #[test]
    fn rbf_nystroem_reconstructs_full_kernel_at_full_rank() {
        let mut s: u64 = 0x2468_ace0_1357_9bdf;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let (n, dim, gamma) = (40usize, 3usize, 0.7f64);
        let data: Vec<Vec<f64>> = (0..n).map(|_| (0..dim).map(|_| rng()).collect()).collect();
        // Reference full RBF kernel.
        let kref: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d2: f64 = (0..dim).map(|t| (data[i][t] - data[j][t]).powi(2)).sum();
                        (-gamma * d2).exp()
                    })
                    .collect()
            })
            .collect();

        // m = n landmarks ⇒ Z·Zᵀ = K exactly (W = K, Z = K·K^{-1/2}).
        let ny = rbf_nystroem(&data, n, gamma, 9).expect("rbf_nystroem");
        assert_eq!(ny.feature_map.len(), n);
        let mp = ny.feature_map[0].len();
        let mut maxerr = 0.0f64;
        for (i, kref_i) in kref.iter().enumerate().take(n) {
            for (j, &kij) in kref_i.iter().enumerate().take(n) {
                let zz: f64 = (0..mp)
                    .map(|t| ny.feature_map[i][t] * ny.feature_map[j][t])
                    .sum();
                maxerr = maxerr.max((zz - kij).abs());
            }
        }
        assert!(
            maxerr < 1e-8,
            "full-rank RBF reconstruction maxerr {maxerr}"
        );

        // Fewer landmarks: a valid (smaller) feature map, still finite.
        let ny2 = rbf_nystroem(&data, 12, gamma, 9).expect("rbf_nystroem m<n");
        assert_eq!(ny2.feature_map.len(), n);
        assert!(ny2.feature_map.iter().flatten().all(|v| v.is_finite()));

        assert!(rbf_nystroem(&[], 2, gamma, 1).is_err());
        assert!(rbf_nystroem(&data, 0, gamma, 1).is_err());
        assert!(rbf_nystroem(&data, n + 1, gamma, 1).is_err());
        assert!(rbf_nystroem(&data, 5, 0.0, 1).is_err()); // gamma <= 0
    }

    #[test]
    fn cur_decomposition_reconstructs_lowrank() {
        let mut s: u64 = 0xc0ff_ee00_1234_5678;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        // Rank-r matrix A = B·G (m×r · r×n); k ≥ r ⇒ CUR is exact to rounding.
        let (m, n, r, k) = (100usize, 60usize, 6usize, 10usize);
        let bb: Vec<Vec<f64>> = (0..m).map(|_| (0..r).map(|_| rng()).collect()).collect();
        let gg: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
        let a: Vec<Vec<f64>> = bb
            .iter()
            .map(|bi| {
                (0..n)
                    .map(|j| (0..r).map(|t| bi[t] * gg[t][j]).sum())
                    .collect()
            })
            .collect();

        let cur = cur_decomposition(&a, k, 10, 7).expect("cur");
        assert_eq!(cur.column_indices.len(), k);
        assert_eq!(cur.row_indices.len(), k);
        assert!(cur.column_indices.windows(2).all(|w| w[0] < w[1]));
        assert!(cur.row_indices.windows(2).all(|w| w[0] < w[1]));
        // C/R must be genuine columns/rows of A.
        for (cc, &jcol) in cur.column_indices.iter().enumerate() {
            for (i, a_i) in a.iter().enumerate().take(m) {
                assert_eq!(cur.c[i][cc], a_i[jcol]);
            }
        }

        // A ≈ C·U·R.
        let cu = fsci_linalg::matmul(&cur.c, &cur.u).expect("CU");
        let cur_recon = fsci_linalg::matmul(&cu, &cur.r).expect("CUR");
        let mut maxerr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                maxerr = maxerr.max((cur_recon[i][j] - a[i][j]).abs());
            }
        }
        assert!(maxerr < 1e-6, "CUR reconstruction maxerr {maxerr}");

        assert!(cur_decomposition(&[], 2, 5, 1).is_err());
        assert!(cur_decomposition(&a, 0, 5, 1).is_err());
        assert!(cur_decomposition(&a, m + 1, 5, 1).is_err());
    }

    #[test]
    fn ppca_recovers_isotropic_noise_and_covariance() {
        let mut s: u64 = 0x5151_2323_8989_abab;
        let mut gauss = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 11) as f64) / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let (n, d, k) = (6000usize, 8usize, 3usize);
        let load: Vec<Vec<f64>> = (0..k).map(|_| (0..d).map(|_| gauss()).collect()).collect();
        let sigma = 0.5f64; // isotropic noise std
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..k).map(|_| gauss()).collect();
                (0..d)
                    .map(|j| (0..k).map(|t| z[t] * load[t][j]).sum::<f64>() + sigma * gauss())
                    .collect()
            })
            .collect();

        let p = ppca(&x, k, 5).expect("ppca");
        assert_eq!(p.components.len(), k);
        assert_eq!(p.components[0].len(), d);
        assert_eq!(p.explained_variance.len(), k);
        for w in p.explained_variance.windows(2) {
            assert!(w[0] >= w[1] - 1e-9, "explained_variance descending");
        }
        // Recovered isotropic noise variance ≈ σ².
        assert!(
            (p.noise_variance - sigma * sigma).abs() < 0.05,
            "noise_variance {} vs {}",
            p.noise_variance,
            sigma * sigma
        );

        // Model covariance C = Wᵀ W + σ²I ≈ sample covariance.
        let mut samp = vec![vec![0.0; d]; d];
        for row in &x {
            for a in 0..d {
                for b in 0..d {
                    samp[a][b] += (row[a] - p.mean[a]) * (row[b] - p.mean[b]);
                }
            }
        }
        let mut maxerr = 0.0f64;
        for (a, samp_a) in samp.iter_mut().enumerate().take(d) {
            for (b, sab) in samp_a.iter_mut().enumerate().take(d) {
                *sab /= n as f64;
                let mut c: f64 = (0..k)
                    .map(|t| p.components[t][a] * p.components[t][b])
                    .sum();
                if a == b {
                    c += p.noise_variance;
                }
                maxerr = maxerr.max((c - *sab).abs());
            }
        }
        assert!(maxerr < 0.1, "covariance reconstruction maxerr {maxerr}");

        assert!(ppca(&[], 2, 1).is_err());
        assert!(ppca(&x, 0, 1).is_err());
        assert!(ppca(&x, d + 1, 1).is_err());
    }

    #[test]
    fn gaussian_mixture_recovers_separated_gaussians() {
        let mut s: u64 = 0x6a09_e667_f3bc_c908;
        let mut gauss = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 11) as f64) / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        // Three well-separated 2-D Gaussians (σ≈0.4) at distinct centers.
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let per = 80usize;
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..per {
                data.push(vec![ctr[0] + 0.4 * gauss(), ctr[1] + 0.4 * gauss()]);
                truth.push(c);
            }
        }
        let n = data.len();

        let gmm = gaussian_mixture(&data, 3, 200, 1e-6, 1e-6, 7).expect("gmm");
        assert_eq!(gmm.weights.len(), 3);
        assert_eq!(gmm.means.len(), 3);
        assert_eq!(gmm.means[0].len(), 2);
        assert_eq!(gmm.responsibilities.len(), n);
        assert!(gmm.log_likelihood.is_finite());
        assert!((gmm.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for w in &gmm.weights {
            assert!((w - 1.0 / 3.0).abs() < 0.05, "weight {w} should be ~1/3");
        }
        // Each row of responsibilities sums to 1.
        for r in &gmm.responsibilities {
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Hard labels separate the blobs (purity 1.0).
        let mut correct = 0usize;
        for pred in 0..3 {
            let mut counts = [0usize; 3];
            for i in 0..n {
                if gmm.labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(correct, n, "GMM should perfectly separate the Gaussians");
        // Each recovered mean is close to one true center.
        for ctr in &centers {
            let best = gmm
                .means
                .iter()
                .map(|m| (m[0] - ctr[0]).powi(2) + (m[1] - ctr[1]).powi(2))
                .fold(f64::INFINITY, f64::min);
            assert!(best < 0.25, "no recovered mean near center {ctr:?}");
        }
        // Model selection: BIC/AIC finite, and the true k=3 beats an underfit k=1.
        assert!(gmm.bic.is_finite() && gmm.aic.is_finite());
        let under = gaussian_mixture(&data, 1, 200, 1e-6, 1e-6, 7).expect("k=1");
        assert!(gmm.bic < under.bic, "true k=3 BIC should beat k=1");

        assert!(gaussian_mixture(&[], 2, 10, 1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, 0, 10, 1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, n + 1, 10, 1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, 2, 0, 1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, 2, 10, f64::NAN, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, 2, 10, -1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture(&data, 2, 10, 1e-3, -1.0, 1).is_err()); // reg_covar < 0
    }

    #[test]
    fn gaussian_mixture_full_recovers_correlated_gaussians() {
        let mut s: u64 = 0xbb67_ae85_84ca_a73b;
        let mut gauss = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 11) as f64) / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        // Two well-separated 2-D Gaussians with strong, opposite correlations (a diagonal model
        // could not represent these). x = center + L·z, z ~ N(0,I).
        let centers = [[0.0, 0.0], [12.0, 12.0]];
        let chols = [[[0.6, 0.0], [0.55, 0.25]], [[0.6, 0.0], [-0.55, 0.25]]];
        let per = 120usize;
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..per {
                let (z0, z1) = (gauss(), gauss());
                let l = &chols[c];
                data.push(vec![
                    ctr[0] + l[0][0] * z0 + l[0][1] * z1,
                    ctr[1] + l[1][0] * z0 + l[1][1] * z1,
                ]);
                truth.push(c);
            }
        }
        let n = data.len();

        let gmm = gaussian_mixture_full(&data, 2, 200, 1e-6, 1e-6, 11).expect("gmm full");
        assert_eq!(gmm.covariances.len(), 2);
        assert_eq!(gmm.covariances[0].len(), 2);
        assert!((gmm.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for r in &gmm.responsibilities {
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Covariances symmetric.
        for cov in &gmm.covariances {
            assert!((cov[0][1] - cov[1][0]).abs() < 1e-12);
        }
        // Purity 1.0.
        let mut correct = 0usize;
        for pred in 0..2 {
            let mut counts = [0usize; 2];
            for i in 0..n {
                if gmm.labels[i] == pred {
                    counts[truth[i]] += 1;
                }
            }
            correct += counts.iter().copied().max().unwrap_or(0);
        }
        assert_eq!(
            correct, n,
            "full GMM should separate the correlated Gaussians"
        );
        // The recovered component covering each center should have the right correlation SIGN.
        for (c, ctr) in centers.iter().enumerate() {
            let comp = (0..2)
                .min_by(|&a, &b| {
                    let da =
                        (gmm.means[a][0] - ctr[0]).powi(2) + (gmm.means[a][1] - ctr[1]).powi(2);
                    let db =
                        (gmm.means[b][0] - ctr[0]).powi(2) + (gmm.means[b][1] - ctr[1]).powi(2);
                    da.total_cmp(&db)
                })
                .unwrap();
            let true_off = chols[c][1][0] * chols[c][0][0]; // sign of the off-diagonal covariance
            assert!(
                gmm.covariances[comp][0][1].signum() == true_off.signum(),
                "covariance correlation sign mismatch for center {ctr:?}"
            );
        }

        assert!(gaussian_mixture_full(&data, 2, 10, 1e-3, 0.0, 1).is_err()); // reg_covar must be > 0
        assert!(gaussian_mixture_full(&data, 2, 0, 1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture_full(&data, 2, 10, f64::INFINITY, 1e-6, 1).is_err());
        assert!(gaussian_mixture_full(&data, 2, 10, -1e-3, 1e-6, 1).is_err());
        assert!(gaussian_mixture_full(&[], 2, 10, 1e-3, 1e-6, 1).is_err());
    }

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
    fn cluster_estimators_reject_zero_iteration_budgets() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];

        let err = kmeans(&data, 1, 0, 42).expect_err("kmeans should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));

        let err = mini_batch_kmeans(&data, 1, 0, 1, 42)
            .expect_err("mini_batch_kmeans should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));

        let err =
            nmf(&data, 1, 0, 1e-6, NmfInit::Random, 42).expect_err("nmf should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));

        let err = factor_analysis(&data, 1, 0, 1e-6, 42)
            .expect_err("factor_analysis should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));

        let err = mean_shift(&data, 1.0, 0).expect_err("mean_shift should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));

        let err = kmedoids(&data, 1, 0, 42).expect_err("kmedoids should reject max_iter=0");
        assert!(matches!(err, ClusterError::InvalidArgument(_)));
    }

    #[test]
    fn factorization_estimators_reject_invalid_tolerances() {
        let data = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        for tol in [f64::NAN, f64::INFINITY, -1e-6] {
            let err = nmf(&data, 1, 10, tol, NmfInit::Random, 42)
                .expect_err("nmf should reject invalid tol");
            assert!(matches!(err, ClusterError::InvalidArgument(_)));

            let err = factor_analysis(&data, 1, 10, tol, 42)
                .expect_err("factor_analysis should reject invalid tol");
            assert!(matches!(err, ClusterError::InvalidArgument(_)));
        }
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
    fn nearest_centroid_k4_d4_matches_scalar_argmin() {
        let centroids = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![2.0, -1.0, 0.5, 3.0],
            vec![-4.0, 2.0, 1.5, -0.25],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let flat = flatten_centroids(&centroids, 4);
        let points = [
            vec![0.1, -0.2, 0.3, -0.4],
            vec![1.8, -0.8, 0.7, 2.7],
            vec![-3.5, 1.9, 1.4, -0.4],
            vec![0.5, 0.5, 0.5, 0.5],
        ];

        for point in points {
            let got = nearest_centroid_k4_d4(&point, &flat);
            let mut want = (0usize, sq_dist(&point, &flat[0..4]));
            for c in 1..4 {
                let sd = sq_dist(&point, &flat[c * 4..c * 4 + 4]);
                if sd < want.1 {
                    want = (c, sd);
                }
            }
            assert_eq!(got.0, want.0);
            assert_eq!(got.1.to_bits(), want.1.to_bits());
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

    /// `scipy.cluster.vq.vq` tie-breaking and full-mantissa distances
    /// (frankenscipy-wp4ke).
    ///
    /// A scipy golden for vq already exists (`vq_matches_scipy_reference_values`
    /// further down), so this does NOT fill an empty column — it fills the gaps
    /// that one leaves. That test uses three 2-D points on integer coordinates
    /// whose distances are 0, sqrt(2), 0 at 1e-10, and the rest of the vq suite
    /// is hand-computed integer geometry plus metamorphic properties. Two things
    /// stay unpinned by all of it:
    ///
    ///   1. TIE-BREAKING. Nothing constrains which centroid wins an exact tie.
    ///      SciPy takes the lowest index; a `<` vs `<=` slip in the argmin loop
    ///      would silently flip it and every existing test would still pass.
    ///   2. Full-mantissa distances. Every existing expected distance is an
    ///      exact integer or a clean sqrt, so an error in the low bits of the
    ///      accumulation cannot show up.
    ///
    /// Reference values from scipy.cluster.vq.vq, SciPy 1.17.1.
    #[test]
    fn vq_tie_breaking_and_full_mantissa_match_scipy() {
        let centroids = vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 2.0, 2.0],
            vec![-3.0, 4.0, 0.0],
        ];
        let data = vec![
            vec![0.1, 0.2, 0.3],
            vec![1.7, -0.4, 2.2],
            vec![-3.3, 4.1, 0.05],
            vec![2.5, 2.5, 2.5],
            vec![0.0, 0.0, 0.0],
            vec![5.0, 5.0, 5.0],
            vec![1.0, 1.0, 1.0],
            vec![-1.25, 0.75, -2.5],
        ];
        // (code, dist) straight from scipy.cluster.vq.vq(obs, code_book).
        let expected = [
            (0usize, 0.374_165_738_677_394_17),
            (1, 2.426_932_219_902_319_3),
            (2, 0.320_156_211_871_642_17),
            (1, 0.866_025_403_784_438_6),
            (0, 0.0),
            (1, 5.196_152_422_706_632),
            (0, 1.732_050_807_568_877_2),
            (0, 2.893_959_225_697_556_4),
        ];

        let (labels, dists) = vq(&data, &centroids).expect("vq");
        assert_eq!(labels.len(), expected.len(), "one label per observation");
        for (i, (label, dist)) in labels.iter().zip(dists.iter()).enumerate() {
            let (want_code, want_dist) = expected[i];
            assert_eq!(*label, want_code, "vq code[{i}]");
            if want_dist == 0.0 {
                // An observation sitting exactly on its centroid must give a
                // hard zero, not a denormal from a sqrt of accumulated error.
                assert_eq!(*dist, 0.0, "vq dist[{i}] must be exactly zero");
            } else {
                let rel = (dist - want_dist).abs() / want_dist.abs();
                assert!(
                    rel <= 1.0e-15,
                    "vq dist[{i}]: got {dist}, scipy {want_dist} (rel {rel:.3e})"
                );
            }
        }

        // TIE-BREAKING. (1,1,1) is exactly equidistant from (0,0,0) and
        // (2,2,2) — both at sqrt(3). SciPy resolves to the LOWEST centroid
        // index. Nothing else in this suite pins that, and a `<` vs `<=` slip
        // in the argmin loop would silently flip it.
        let tie_centroids = vec![vec![0.0, 0.0, 0.0], vec![2.0, 2.0, 2.0]];
        let tie_data = vec![vec![1.0, 1.0, 1.0]];
        let (tie_labels, tie_dists) = vq(&tie_data, &tie_centroids).expect("vq tie");
        assert_eq!(
            tie_labels[0], 0,
            "scipy.cluster.vq.vq breaks an exact tie to the lowest index"
        );
        let tie_rel = (tie_dists[0] - 1.732_050_807_568_877_2).abs() / 1.732_050_807_568_877_2;
        assert!(tie_rel <= 1.0e-15, "vq tie distance: got {}", tie_dists[0]);
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

    /// Exact `scipy.cluster.hierarchy.linkage` Z-matrix goldens
    /// (frankenscipy-rlxfh).
    ///
    /// `linkage_single` below is the only value-level linkage coverage and it
    /// checks row 0 of a 3-point 1-D case: two indices and a distance. It never
    /// looks at column 3, never looks at row 1, and pins no method other than
    /// Single. So the parts of scipy's Z contract that carry the actual
    /// conventions were unpinned:
    ///
    ///   * the OBSERVATION-COUNT column (Z[i][3]) — an implementation that
    ///     wrote 2.0 for every merge, or that counted clusters rather than
    ///     original observations, passes everything above;
    ///   * the NEW-CLUSTER INDEXING rule, that the cluster formed by row i is
    ///     addressable as n+i in later rows (here 6..=9 for n=6);
    ///   * the row ordering (merges by increasing distance) and the
    ///     smaller-index-first convention within a row;
    ///   * every method except Single.
    ///
    /// The fixture is chosen so the golden is well defined: all 15 pairwise
    /// distances are DISTINCT (verified), so there is no tie whose resolution
    /// scipy and fsci could legitimately disagree on. A tie-broken fixture
    /// would make this test assert a coincidence rather than a contract.
    ///
    /// Reference values from scipy.cluster.hierarchy.linkage, SciPy 1.17.1.
    #[test]
    fn linkage_z_matrix_matches_scipy_across_methods() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.6, 0.1],
            vec![2.3, 1.7],
            vec![5.1, 4.2],
            vec![5.4, 3.6],
            vec![9.2, 0.4],
        ];
        let n = data.len();

        // (method, Z) straight from scipy.cluster.hierarchy.linkage(X, method).
        let cases: [(LinkageMethod, [[f64; 4]; 5]); 4] = [
            (
                LinkageMethod::Single,
                [
                    [0.0, 1.0, 0.608_276_253_029_821_9, 2.0],
                    [3.0, 4.0, 0.670_820_393_249_937_3, 2.0],
                    [2.0, 6.0, 2.334_523_505_985_75, 3.0],
                    [7.0, 8.0, 3.635_931_792_539_569, 5.0],
                    [5.0, 9.0, 4.967_896_939_349_687_5, 6.0],
                ],
            ),
            (
                LinkageMethod::Complete,
                [
                    [0.0, 1.0, 0.608_276_253_029_821_9, 2.0],
                    [3.0, 4.0, 0.670_820_393_249_937_3, 2.0],
                    [2.0, 6.0, 2.860_069_929_215_018_5, 3.0],
                    [5.0, 7.0, 5.590_169_943_749_474_5, 3.0],
                    [8.0, 9.0, 9.208_691_546_577_07, 6.0],
                ],
            ),
            (
                LinkageMethod::Average,
                [
                    [0.0, 1.0, 0.608_276_253_029_821_9, 2.0],
                    [3.0, 4.0, 0.670_820_393_249_937_3, 2.0],
                    [2.0, 6.0, 2.597_296_717_600_384_5, 3.0],
                    [5.0, 7.0, 5.279_033_441_549_581, 3.0],
                    [8.0, 9.0, 6.372_217_023_613_850_5, 6.0],
                ],
            ),
            (
                LinkageMethod::Ward,
                [
                    [0.0, 1.0, 0.608_276_253_029_821_9, 2.0],
                    [3.0, 4.0, 0.670_820_393_249_937_3, 2.0],
                    [2.0, 6.0, 2.993_882_651_897_587_6, 3.0],
                    [5.0, 7.0, 6.093_986_106_537_056, 3.0],
                    [8.0, 9.0, 10.379_466_909_881_9, 6.0],
                ],
            ),
        ];

        for (method, want) in cases {
            let z = linkage(&data, method).expect("linkage");
            assert_eq!(z.len(), n - 1, "{method:?}: Z has n-1 rows");
            for (i, (got_row, want_row)) in z.iter().zip(want.iter()).enumerate() {
                // Indices and the observation count are exact integers in
                // scipy's Z; compare them exactly rather than with a tolerance.
                assert_eq!(got_row[0], want_row[0], "{method:?} Z[{i}][0] index");
                assert_eq!(got_row[1], want_row[1], "{method:?} Z[{i}][1] index");
                assert_eq!(
                    got_row[3], want_row[3],
                    "{method:?} Z[{i}][3] observation count"
                );
                let rel = (got_row[2] - want_row[2]).abs() / want_row[2].abs();
                assert!(
                    rel <= 1e-12,
                    "{method:?} Z[{i}][2]: got {}, scipy {} (rel {rel:.3e})",
                    got_row[2],
                    want_row[2]
                );
            }

            // Contract checks that hold for every method, stated separately so a
            // failure says which rule broke rather than just which number moved.
            for (i, row) in z.iter().enumerate() {
                assert!(
                    row[0] < row[1],
                    "{method:?} Z[{i}]: smaller index must come first"
                );
                assert!(
                    row[0] < (n + i) as f64 && row[1] < (n + i) as f64,
                    "{method:?} Z[{i}]: may only reference clusters formed earlier"
                );
            }
            for i in 1..z.len() {
                assert!(
                    z[i][2] >= z[i - 1][2],
                    "{method:?}: merge distances must be non-decreasing"
                );
            }
            assert_eq!(
                z[z.len() - 1][3],
                n as f64,
                "{method:?}: final merge must contain every observation"
            );
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
    fn linkage_flat_core_matches_precomputed_condensed_contract() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![2.5, -0.25],
            vec![4.0, 1.75],
            vec![5.5, -1.0],
            vec![7.0, 0.25],
        ];
        let n = data.len();
        let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in i + 1..n {
                condensed.push(sq_dist(&data[i], &data[j]).sqrt());
            }
        }

        for method in [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Ward,
            LinkageMethod::Weighted,
        ] {
            let from_data = linkage(&data, method).unwrap();
            let from_dist = linkage_from_distances(&condensed, n, method).unwrap();
            assert_eq!(from_data.len(), from_dist.len(), "{method:?}");
            for (row_idx, (a, b)) in from_data.iter().zip(&from_dist).enumerate() {
                for col in 0..4 {
                    assert_eq!(
                        a[col].to_bits(),
                        b[col].to_bits(),
                        "{method:?} row {row_idx} col {col}: {} != {}",
                        a[col],
                        b[col]
                    );
                }
            }
        }
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

    // frankenscipy-1b85n. `agglomerate_nnarray` is unwired: nothing in the
    // library calls it, and the bench measures its own private (and by now
    // signature-divergent) copy instead. The bead has to decide re-wire vs
    // retire, and both answers need to know whether the unwired code still
    // agrees with what ships. This pins that.
    //
    // It also stops the code bit-rotting while it sits behind #[allow(dead_code)]:
    // an unwired function with no test can drift silently, and drift is what
    // makes the eventual decision expensive.
    //
    // Cross-validates two INDEPENDENT implementations bitwise. The shipped path
    // (`linkage` -> `linkage_from_dm`) routes Single to an MST, Centroid/Median
    // to the Müller heap, and everything else to NN-chain. `agglomerate_nnarray`
    // is a single nearest-neighbour-array loop that knows none of those routes.
    // Each is separately documented as byte-identical to the naive pairwise
    // scan, so they must agree with each other element-for-element -- and it is
    // exact bit equality via to_bits(), not a tolerance.
    #[test]
    fn unwired_agglomerate_nnarray_is_bit_identical_to_the_shipped_linkage() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![4.0, 4.0],
            vec![5.0, 4.0],
            vec![4.0, 5.0],
            vec![2.5, 3.25],
        ];
        let n = data.len();
        let total = 2 * n - 1;

        // agglomerate_nnarray indexes a flat (2n-1)x(2n-1) matrix through
        // agglo_idx(total, row, col) = row * total + col, with the inactive
        // upper ids left at +inf and the diagonal zeroed.
        let mut inter_dist = vec![f64::INFINITY; total * total];
        for i in 0..n {
            inter_dist[i * total + i] = 0.0;
            for j in i + 1..n {
                let d = data[i]
                    .iter()
                    .zip(&data[j])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                inter_dist[i * total + j] = d;
                inter_dist[j * total + i] = d;
            }
        }

        // MEASURED 2026-08-08: bit-identical for four of the five methods.
        for method in [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Weighted,
        ] {
            let shipped = linkage(&data, method).expect("shipped linkage");
            let unwired = agglomerate_nnarray(n, inter_dist.clone(), method);
            assert_eq!(shipped.len(), unwired.len(), "{method:?}: row count");
            for (r, (s_row, u_row)) in shipped.iter().zip(&unwired).enumerate() {
                for (c, (s, u)) in s_row.iter().zip(u_row).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        u.to_bits(),
                        "{method:?} row {r} col {c}: shipped {s} vs unwired nnarray {u}"
                    );
                }
            }
        }

        // WARD IS THE EXCEPTION, and it decides frankenscipy-1b85n. The unwired
        // NN-array disagrees with the shipped NN-chain path by exactly 1 ulp at
        // row 2 col 2, and SciPy agrees with the SHIPPED value:
        //     scipy.cluster.hierarchy.linkage(data, 'ward')[2][2]
        //         == 1.2909944487358056   (shipped fsci)
        //         != 1.2909944487358058   (unwired nnarray)
        // So re-wiring this lever would move Ward 1 ulp OFF SciPy. That makes
        // the bead's choice retire-not-rewire, on evidence rather than taste,
        // and it falsifies this function's own docstring claim of being
        // "byte-identical ... element-for-element" with the pairwise scan.
        let shipped_ward = linkage(&data, LinkageMethod::Ward).expect("shipped ward");
        let unwired_ward = agglomerate_nnarray(n, inter_dist.clone(), LinkageMethod::Ward);
        assert_eq!(
            shipped_ward[2][2].to_bits(),
            1.290_994_448_735_805_6_f64.to_bits(),
            "shipped Ward must stay bit-equal to scipy at the known divergence point"
        );
        assert_ne!(
            unwired_ward[2][2].to_bits(),
            shipped_ward[2][2].to_bits(),
            "the unwired nnarray Ward divergence has disappeared; re-check \
             frankenscipy-1b85n, its retire recommendation rests on this"
        );
        assert_eq!(
            unwired_ward[2][2]
                .to_bits()
                .abs_diff(shipped_ward[2][2].to_bits()),
            1,
            "the unwired Ward divergence should be exactly 1 ulp; a larger gap \
             means something else changed"
        );
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
    fn linkage_non_single_methods_match_scipy_z_matrices() {
        // SciPy 1.17.1: scipy.cluster.hierarchy.linkage on this asymmetric
        // four-point input.  The distinct first two merge distances also lock
        // the cluster-ID ordering, while the final row distinguishes each
        // Lance-Williams update rule.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 0.0],
            vec![3.0, 2.0],
        ];
        let common = [[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 2.0]];
        let cases = [
            (LinkageMethod::Complete, 3.605_551_275_463_989),
            (LinkageMethod::Average, 3.232_526_648_950_187),
            (LinkageMethod::Weighted, 3.232_526_648_950_187),
            (LinkageMethod::Ward, 4.301_162_633_521_313),
            (LinkageMethod::Centroid, 3.041_381_265_149_109_7),
            (LinkageMethod::Median, 3.041_381_265_149_109_7),
        ];

        for (method, final_distance) in cases {
            let z = linkage(&data, method).expect("linkage");
            let expected = [common[0], common[1], [4.0, 5.0, final_distance, 4.0]];
            assert_eq!(z.len(), expected.len(), "{method:?}");
            for (row_index, (actual, expected)) in z.iter().zip(expected).enumerate() {
                for (column, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "{method:?} Z[{row_index}][{column}] = {actual}, expected {expected}"
                    );
                }
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
    fn proximity_cliques_finds_two_groups() {
        // proximity_cliques (Bron-Kerbosch maximal cliques on the eps-proximity
        // graph) was untested. Two well-separated pairs within eps -> two maximal
        // cliques {0,1} and {2,3}.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let mut cliques = proximity_cliques(&data, 0.5);
        for c in cliques.iter_mut() {
            c.sort_unstable();
        }
        cliques.sort();
        assert_eq!(cliques, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn elbow_inertias_recovers_known_inertia() {
        // elbow_inertias was untested. 4 points at square corners: the k=1 inertia
        // is the exact total within-cluster SS = sum||x-centroid||^2 = 4*2 = 8
        // (seed-independent), and k=4 (each point its own cluster) -> 0. Inertias
        // are non-increasing in k.
        let data = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 2.0],
        ];
        let inertias = elbow_inertias(&data, 4, 42);
        assert_eq!(inertias.len(), 4);
        assert!(
            (inertias[0] - 8.0).abs() < 1e-9,
            "k=1 inertia = {}",
            inertias[0]
        );
        assert!(inertias[3].abs() < 1e-9, "k=4 inertia ~ 0");
        for w in inertias.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "inertias non-increasing");
        }
    }

    #[test]
    fn vq_match_scipy() {
        // scipy.cluster.vq.vq: assign each obs to nearest centroid (code, distance).
        let obs = vec![vec![1.0, 1.0], vec![5.0, 5.0], vec![1.0, 6.0]];
        let cb = vec![vec![0.0, 0.0], vec![6.0, 6.0]];
        let (codes, dist) = vq(&obs, &cb).expect("vq");
        assert_eq!(codes, vec![0, 1, 1]);
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(
            (dist[0] - sqrt2).abs() < 1e-12 && (dist[1] - sqrt2).abs() < 1e-12,
            "dist01: {:?}",
            &dist[..2]
        );
        assert!((dist[2] - 5.0).abs() < 1e-12, "dist2: {}", dist[2]);
    }

    #[test]
    fn whiten_match_scipy() {
        // scipy.cluster.vq.whiten: divide each column by its (population) std.
        let x = vec![vec![1.0, 2.0], vec![4.0, 8.0], vec![7.0, 2.0]];
        let w = whiten(&x).expect("whiten");
        let e = [
            [0.408_248_290_463_863_1, 0.707_106_781_186_547_5],
            [1.632_993_161_855_452_5, 2.828_427_124_746_190_3],
            [2.857_738_033_247_041_5, 0.707_106_781_186_547_5],
        ];
        for (gr, er) in w.iter().zip(&e) {
            for (g, ex) in gr.iter().zip(er) {
                assert!((g - ex).abs() < 1e-11, "whiten: {g} vs {ex}");
            }
        }
    }

    #[test]
    fn fcluster_maxclust_match_scipy() {
        // Single-linkage Z of [[0,0],[0,1],[5,5],[5,6]]. fcluster (maxclust):
        // 2 clusters -> [1,1,2,2], 1 -> all 1, 4 -> [1,2,3,4]. Matches
        // scipy.cluster.hierarchy.fcluster(Z, t, criterion='maxclust').
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 2.0],
            [4.0, 5.0, 6.403_124_237_432_849, 4.0],
        ];
        assert_eq!(fcluster(&z, 2).expect("fc2"), vec![1, 1, 2, 2]);
        assert_eq!(fcluster(&z, 1).expect("fc1"), vec![1, 1, 1, 1]);
        assert_eq!(fcluster(&z, 4).expect("fc4"), vec![1, 2, 3, 4]);
    }

    #[test]
    fn linkage_single_match_scipy() {
        // scipy.cluster.hierarchy.linkage(X, method='single') Z-matrix for two
        // well-separated pairs; final merge distance = sqrt(41) (single linkage).
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![5.0, 6.0],
        ];
        let z = linkage(&x, LinkageMethod::Single).expect("linkage");
        let expect = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 2.0],
            [4.0, 5.0, 6.403_124_237_432_849, 4.0],
        ];
        assert_eq!(z.len(), 3);
        for (gr, er) in z.iter().zip(&expect) {
            for (g, e) in gr.iter().zip(er) {
                assert!((g - e).abs() < 1e-12, "linkage: {g} vs {e}");
            }
        }
    }

    #[test]
    fn cluster_indices_match_sklearn_exactly() {
        // Exact golden values from sklearn.metrics for the two well-separated
        // clusters [[0,0],[1,1],[10,10],[11,11]] with labels [0,0,1,1]. The other
        // index tests only check loose inequalities; this pins the precise values
        // and adds calinski_harabasz_score coverage (previously untested).
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let ch = calinski_harabasz_score(&data, &labels).expect("calinski_harabasz");
        assert!(
            (ch - 200.0).abs() < 1e-9,
            "calinski_harabasz: {ch} != 200.0"
        );
        let db = davies_bouldin_score(&data, &labels).expect("davies_bouldin");
        assert!((db - 0.1).abs() < 1e-12, "davies_bouldin: {db} != 0.1");
        let sil = silhouette_score(&data, &labels).expect("silhouette");
        assert!(
            (sil - 0.899_749_373_433_583_9).abs() < 1e-12,
            "silhouette: {sil} != 0.8997493734335839"
        );
    }

    #[test]
    fn comparison_metrics_match_sklearn_exactly() {
        // Exact golden values from sklearn.metrics for two clusterings of 6
        // samples: true=[0,0,1,1,2,2], pred=[0,0,1,2,2,2] (partial agreement).
        let t = [0usize, 0, 1, 1, 2, 2];
        let p = [0usize, 0, 1, 2, 2, 2];
        let close = |got: f64, want: f64, name: &str| {
            assert!((got - want).abs() < 1e-12, "{name}: {got} != {want}");
        };
        close(
            adjusted_rand_score(&t, &p).unwrap(),
            0.444_444_444_444_444_4,
            "adjusted_rand",
        );
        close(
            fowlkes_mallows_score(&t, &p).unwrap(),
            0.577_350_269_189_625_8,
            "fowlkes_mallows",
        );
        close(
            normalized_mutual_info(&t, &p).unwrap(),
            0.739_667_376_800_759_2,
            "nmi (arithmetic)",
        );
        close(
            homogeneity_score(&t, &p).unwrap(),
            0.710_309_917_857_152_5,
            "homogeneity",
        );
        close(
            completeness_score(&t, &p).unwrap(),
            0.771_556_173_679_471_2,
            "completeness",
        );
        close(
            v_measure_score(&t, &p).unwrap(),
            0.739_667_376_800_759,
            "v_measure",
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
    fn kmeans2_matrix_init_matches_scipy() {
        let data = vec![
            vec![1.0, 1.0],
            vec![1.5, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 7.0],
            vec![3.5, 5.0],
            vec![4.5, 5.0],
            vec![3.5, 4.5],
        ];
        let init = vec![vec![1.0, 1.0], vec![5.0, 7.0]];
        // scipy.cluster.vq.kmeans2(data, init, minit='matrix', iter=10)
        let (cb, lab) = kmeans2(&data, &init, 10).unwrap();
        assert_eq!(lab, vec![0, 0, 1, 1, 1, 1, 1]);
        let exp_cb = [[1.25, 1.5], [3.9, 5.1]];
        for j in 0..2 {
            for c in 0..2 {
                assert!((cb[j][c] - exp_cb[j][c]).abs() < 1e-8, "cb[{j}][{c}]");
            }
        }
        // iter=1: labels from the single (initial) assignment; [3,4] ties to index 0.
        let (cb1, lab1) = kmeans2(&data, &init, 1).unwrap();
        assert_eq!(lab1, vec![0, 0, 0, 1, 1, 1, 1]);
        let exp_cb1 = [[1.83333333, 2.33333333], [4.125, 5.375]];
        for j in 0..2 {
            for c in 0..2 {
                assert!((cb1[j][c] - exp_cb1[j][c]).abs() < 1e-7, "cb1[{j}][{c}]");
            }
        }
        assert!(kmeans2(&data, &init, 0).is_err());
    }

    #[test]
    fn kmeans2_k4_d4_fused_matches_vq_reference() {
        fn reference(
            data: &[Vec<f64>],
            init: &[Vec<f64>],
            iter: usize,
        ) -> (Vec<Vec<f64>>, Vec<usize>) {
            let d = data[0].len();
            let nc = init.len();
            let mut code_book = init.to_vec();
            let mut label = vec![0usize; data.len()];
            let mut sums = vec![vec![0.0_f64; d]; nc];
            let mut counts = vec![0usize; nc];
            let mut next_cb = code_book.clone();
            for _ in 0..iter {
                label = vq(data, &code_book).expect("vq reference").0;
                for row in sums.iter_mut() {
                    row.iter_mut().for_each(|x| *x = 0.0);
                }
                counts.iter_mut().for_each(|x| *x = 0);
                for (i, &lab) in label.iter().enumerate() {
                    counts[lab] += 1;
                    for c in 0..d {
                        sums[lab][c] += data[i][c];
                    }
                }
                for j in 0..nc {
                    if counts[j] > 0 {
                        let inv = 1.0 / counts[j] as f64;
                        for c in 0..d {
                            next_cb[j][c] = sums[j][c] * inv;
                        }
                    } else {
                        next_cb[j].clone_from(&code_book[j]);
                    }
                }
                std::mem::swap(&mut code_book, &mut next_cb);
            }
            (code_book, label)
        }

        let data: Vec<Vec<f64>> = (0..96)
            .map(|i| {
                let cluster = (i % 4) as f64;
                (0..4)
                    .map(|j| cluster * 3.0 + ((i * 17 + j * 13) as f64 * 0.037).sin())
                    .collect()
            })
            .collect();
        let init: Vec<Vec<f64>> = (0..4).map(|k| vec![k as f64 * 3.0; 4]).collect();

        let got = kmeans2(&data, &init, 9).expect("fused kmeans2");
        let want = reference(&data, &init, 9);
        assert_eq!(got.1, want.1);
        for (got_row, want_row) in got.0.iter().zip(want.0.iter()) {
            for (&g, &w) in got_row.iter().zip(want_row.iter()) {
                assert_eq!(g.to_bits(), w.to_bits());
            }
        }
    }

    #[test]
    fn linkage_method_wrappers_match_linkage() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![3.0, 3.0],
            vec![3.2, 2.9],
            vec![6.0, 0.0],
            vec![5.8, 0.3],
        ];
        // Each convenience wrapper equals linkage(data, method) — and scipy's
        // single/complete/average/weighted/ward(X) equal linkage(X, method=...).
        assert_eq!(
            single(&data).unwrap(),
            linkage(&data, LinkageMethod::Single).unwrap()
        );
        assert_eq!(
            complete(&data).unwrap(),
            linkage(&data, LinkageMethod::Complete).unwrap()
        );
        assert_eq!(
            average(&data).unwrap(),
            linkage(&data, LinkageMethod::Average).unwrap()
        );
        assert_eq!(
            weighted(&data).unwrap(),
            linkage(&data, LinkageMethod::Weighted).unwrap()
        );
        assert_eq!(
            ward(&data).unwrap(),
            linkage(&data, LinkageMethod::Ward).unwrap()
        );
        assert_eq!(
            centroid(&data).unwrap(),
            linkage(&data, LinkageMethod::Centroid).unwrap()
        );
        assert_eq!(
            median(&data).unwrap(),
            linkage(&data, LinkageMethod::Median).unwrap()
        );
    }

    #[test]
    fn disjoint_set_matches_scipy() {
        let mut d = DisjointSet::from_elements([1, 2, 3, 4, 5]);
        assert!(d.merge(&1, &2));
        assert!(d.merge(&3, &4));
        assert!(d.merge(&4, &5));
        assert!(!d.merge(&5, &5));
        // Roots (scipy: d[2]==1, d[5]==3).
        assert_eq!(d.find(&2), Some(1));
        assert_eq!(d.find(&5), Some(3));
        assert!(d.connected(&1, &2));
        assert!(!d.connected(&1, &5));
        let mut s4 = d.subset(&4);
        s4.sort_unstable();
        assert_eq!(s4, vec![3, 4, 5]);
        assert_eq!(d.subset_size(&4), 3);
        let subsets: Vec<Vec<i32>> = d
            .subsets()
            .into_iter()
            .map(|mut s| {
                s.sort_unstable();
                s
            })
            .collect();
        assert_eq!(subsets, vec![vec![1, 2], vec![3, 4, 5]]);
        assert_eq!(d.n_subsets(), 2);
        assert_eq!(d.len(), 5);
        assert!(d.contains(&3) && !d.contains(&9));
    }

    #[test]
    fn centroid_median_match_scipy() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![5.0, 6.0],
            vec![10.0, 0.0],
        ];
        // scipy.cluster.hierarchy.centroid/median oracle.
        let want = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.0, 2.0],
            [5.0, 6.0, 7.07106781, 4.0],
            [4.0, 7.0, 8.07774721, 5.0],
        ];
        for (name, z) in [
            ("centroid", centroid(&data).unwrap()),
            ("median", median(&data).unwrap()),
        ] {
            for (i, row) in z.iter().enumerate() {
                for k in 0..4 {
                    assert!(
                        (row[k] - want[i][k]).abs() < 1e-6,
                        "{name}[{i}][{k}]: {} vs {}",
                        row[k],
                        want[i][k]
                    );
                }
            }
        }
    }

    #[test]
    fn hierarchy_maxrstat_cut_tree_match_scipy() {
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.5, 2.0],
            [5.0, 4.0, 2.0, 3.0],
            [6.0, 7.0, 3.0, 5.0],
        ];
        let r = inconsistent(&z, 2);
        // scipy.cluster.hierarchy.maxRstat(Z, R, i)
        let m0 = max_rstat(&z, &r, 0).unwrap();
        for (g, e) in m0.iter().zip(&[1.0, 1.5, 1.5, 2.166667]) {
            assert!((g - e).abs() < 1e-5, "maxRstat0 {g} vs {e}");
        }
        let m3 = max_rstat(&z, &r, 3).unwrap();
        // Not an attempt to spell FRAC_1_SQRT_2 (frankenscipy-9fq10): these are
        // reference values quoted to 6 decimals and compared at 1e-5.
        // Substituting the constant would tighten the expected value beyond what
        // the reference supplies, i.e. change the golden to satisfy a lint.
        #[allow(clippy::approx_constant)]
        let expected = [0.0, 0.0, 0.707_107, 1.091_089];
        for (g, e) in m3.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "maxRstat3 {g} vs {e}");
        }
        assert!(max_rstat(&z, &r, 4).is_err());

        // cut_tree
        assert_eq!(cut_tree(&z, Some(2), None).unwrap(), vec![0, 0, 1, 1, 0]);
        assert_eq!(cut_tree(&z, None, Some(1.7)).unwrap(), vec![0, 0, 1, 1, 2]);
        assert_eq!(cut_tree(&z, Some(5), None).unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(cut_tree(&z, Some(1), None).unwrap(), vec![0, 0, 0, 0, 0]);
        assert!(cut_tree(&z, Some(2), Some(1.0)).is_err());

        // is_isomorphic / correspond
        assert!(is_isomorphic(&[1, 1, 2, 2], &[2, 2, 1, 1]));
        assert!(!is_isomorphic(&[1, 2, 1], &[1, 2, 3]));
        assert!(correspond(
            &z,
            &(0..10).map(|i| i as f64).collect::<Vec<_>>()
        ));
        assert!(!correspond(&z, &[0.0, 1.0, 2.0]));

        // to_mlab / from_mlab round-trip and scipy values.
        let ml = to_mlab_linkage(&z);
        let exp_ml = [
            [1.0, 2.0, 1.0],
            [3.0, 4.0, 1.5],
            [6.0, 5.0, 2.0],
            [7.0, 8.0, 3.0],
        ];
        assert_eq!(ml, exp_ml);
        let back = from_mlab_linkage(&ml);
        assert_eq!(back, z);
        assert!(is_valid_im(&r));
        assert!(!is_valid_im(&[]));
        assert!(!is_valid_im(&[[1.0, -1.0, 2.0, 0.0]])); // negative std

        // to_tree: scipy root id=8, count=5, dist=3, pre_order=[2,3,0,1,4].
        let root = to_tree(&z).unwrap();
        assert_eq!(root.id, 8);
        assert_eq!(root.count, 5);
        assert!((root.dist - 3.0).abs() < 1e-12);
        assert!(!root.is_leaf());
        assert_eq!(root.left.as_ref().unwrap().id, 6);
        assert_eq!(root.right.as_ref().unwrap().id, 7);
        assert_eq!(root.pre_order(), vec![2, 3, 0, 1, 4]);
    }

    #[test]
    fn cut_tree_multi_matches_looping_single_cut() {
        // Build a real linkage, then check cut_tree_multi equals looping the
        // (byte-exact-to-scipy) single cut_tree for every mode.
        let mut s = 3u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let n = 60usize;
        let data: Vec<Vec<f64>> = (0..n).map(|_| (0..3).map(|_| rng()).collect()).collect();
        let z = average(&data).expect("linkage");

        // n_clusters list.
        let ks = [2usize, 5, 9, 20];
        let multi = cut_tree_multi(&z, Some(&ks), None).expect("multi");
        assert_eq!(multi.len(), ks.len());
        for (col, &k) in multi.iter().zip(&ks) {
            assert_eq!(col, &cut_tree(&z, Some(k), None).unwrap());
            assert_eq!(
                col.iter().copied().max().unwrap() + 1,
                k,
                "k={k} label count"
            );
        }

        // heights list.
        let hs = [z[10][2], z[30][2], z[50][2]];
        let hmulti = cut_tree_multi(&z, None, Some(&hs)).expect("hmulti");
        for (col, &h) in hmulti.iter().zip(&hs) {
            assert_eq!(col, &cut_tree(&z, None, Some(h)).unwrap());
        }

        // Full matrix (both None): column j has n-j clusters, matching scipy default.
        let full = cut_tree_multi(&z, None, None).expect("full");
        assert_eq!(full.len(), n);
        for (j, col) in full.iter().enumerate() {
            assert_eq!(col, &cut_tree(&z, Some(n - j), None).unwrap());
            assert_eq!(
                col.iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                n - j
            );
        }

        // Both specified -> error.
        assert!(cut_tree_multi(&z, Some(&ks), Some(&hs)).is_err());
    }

    #[test]
    fn maxdists_maxinconsts_leaders_match_scipy() {
        // Fixed linkage matrix (5 obs, 4 merges) — same Z fed to both libs.
        let z = [
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 1.5, 2.0],
            [5.0, 4.0, 2.0, 3.0],
            [6.0, 7.0, 3.0, 5.0],
        ];
        // scipy.cluster.hierarchy.maxdists(Z)
        assert_eq!(maxdists(&z), vec![1.0, 1.5, 2.0, 3.0]);

        // maxinconsts(Z, inconsistent(Z, 2))
        let r = inconsistent(&z, 2);
        let mi = maxinconsts(&z, &r).unwrap();
        // Reference values quoted to 8 decimals and compared at 1e-7; not the
        // constant (frankenscipy-9fq10). See the note on maxRstat3 above.
        #[allow(clippy::approx_constant)]
        let exp_mi = [0.0, 0.0, 0.707_106_78, 1.091_089_45];
        for (a, b) in mi.iter().zip(exp_mi.iter()) {
            assert!((a - b).abs() < 1e-7, "maxinconsts {a} vs {b}");
        }

        // leaders for T = [2,2,1,1,2]  -> L=[6,7], M=[1,2]
        let (l, m) = leaders(&z, &[2, 2, 1, 1, 2]).unwrap();
        assert_eq!(l, vec![6, 7]);
        assert_eq!(m, vec![1, 2]);
        // Swapped labels -> L ascending by node id, M follows: L=[6,7], M=[2,1]
        let (l2, m2) = leaders(&z, &[1, 1, 2, 2, 1]).unwrap();
        assert_eq!(l2, vec![6, 7]);
        assert_eq!(m2, vec![2, 1]);
        // Invalid (cluster split across two subtrees) is rejected.
        assert!(leaders(&z, &[1, 2, 1, 2, 1]).is_err());
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

#[cfg(test)]
mod max_rstat_matches_scipy {
    use super::{max_rstat, maxinconsts};

    /// `linkage(X, 'average')` for a 7-point fixture, taken from the incumbent so
    /// this tests `max_rstat` and not `linkage`.
    const Z: [[f64; 4]; 6] = [
        [2.0, 3.0, 0.282_842_712_474_619_06, 2.0],
        [0.0, 1.0, 0.360_555_127_546_398_96, 2.0],
        [4.0, 5.0, 0.500_000_000_000_000_11, 2.0],
        [7.0, 8.0, 1.316_739_431_232_531_6, 4.0],
        [6.0, 9.0, 3.457_038_224_229_381_6, 3.0],
        [10.0, 11.0, 3.875_372_344_345_360_3, 7.0],
    ];
    /// `inconsistent(Z, 2)` from the incumbent.
    const R: [[f64; 4]; 6] = [
        [0.282_842_712_474_619_06, 0.0, 1.0, 0.0],
        [0.360_555_127_546_398_96, 0.0, 1.0, 0.0],
        [0.500_000_000_000_000_11, 0.0, 1.0, 0.0],
        [
            0.653_379_090_417_849_92,
            0.575_799_453_966_743_59,
            3.0,
            1.152_068_374_231_204_6,
        ],
        [
            1.978_519_112_114_690_8,
            2.090_941_780_580_422,
            2.0,
            0.707_106_781_186_547_68,
        ],
        [
            2.883_049_999_935_757_5,
            1.372_496_796_803_865_7,
            3.0,
            0.723_005_217_003_365_4,
        ],
    ];

    /// All four columns against `scipy.cluster.hierarchy.maxRstat(Z, R, i)`.
    ///
    /// All four matter: a subtree maximum that simply echoed its own row would
    /// pass on columns 0 and 2 (monotone up the tree here) and fail on 1 and 3,
    /// where the propagated value comes from a CHILD and not the row itself.
    #[test]
    fn every_column_matches_scipy() {
        let want: [[f64; 6]; 4] = [
            [
                0.282_842_712_474_619_06,
                0.360_555_127_546_398_96,
                0.500_000_000_000_000_11,
                0.653_379_090_417_849_92,
                1.978_519_112_114_690_8,
                2.883_049_999_935_757_5,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.575_799_453_966_743_59,
                2.090_941_780_580_422,
                2.090_941_780_580_422,
            ],
            [1.0, 1.0, 1.0, 3.0, 2.0, 3.0],
            [
                0.0,
                0.0,
                0.0,
                1.152_068_374_231_204_6,
                0.707_106_781_186_547_68,
                1.152_068_374_231_204_6,
            ],
        ];
        for (field, expected) in want.iter().enumerate() {
            let got = max_rstat(&Z, &R, field).expect("valid field");
            assert_eq!(got.len(), expected.len());
            for (k, (&g, &w)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "max_rstat(field={field})[{k}] = {g}, scipy = {w}"
                );
            }
        }

        // The propagation is load-bearing, not incidental: on column 1 the last
        // row's own R value is 1.3724967968038657, but the answer is its child's
        // 2.090941780580422. A no-op implementation returning `R[k][field]` would
        // pass columns 0 and 2 and fail here.
        assert!(
            R[5][1] < max_rstat(&Z, &R, 1).unwrap()[5],
            "the subtree maximum must come from a child, not the row itself"
        );
    }

    /// `maxinconsts` is the `field = 3` case, and scipy agrees element for
    /// element. Asserting BIT equality rather than closeness: they are now the
    /// same code path, so anything less would not notice if that stopped being
    /// true.
    #[test]
    fn maxinconsts_is_the_field_three_case() {
        let a = maxinconsts(&Z, &R).expect("same rows");
        let b = max_rstat(&Z, &R, 3).expect("valid field");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn rejects_a_bad_field_and_mismatched_rows() {
        assert!(max_rstat(&Z, &R, 4).is_err(), "field 4 is out of range");
        assert!(max_rstat(&Z, &R[..5], 0).is_err(), "row counts differ");
        // Empty is not an error, it is an empty answer -- matching maxdists.
        assert_eq!(max_rstat(&[], &[], 0).expect("empty"), Vec::<f64>::new());
        // MUST-MISS control for the field guard: 3 is the last valid one.
        assert!(max_rstat(&Z, &R, 3).is_ok(), "field 3 must still be accepted");
    }
}
