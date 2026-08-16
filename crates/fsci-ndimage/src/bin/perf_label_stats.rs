//! Same-process A/B for label-indexed measurements (`mean`/`sum`/...).
//!
//! The shared core `measurement_label_groups` used to bucket each of the N
//! input elements with an O(K) linear `position` scan over the K requested
//! labels — O(N*K). The first fix built a label->position map once and bucketed
//! in O(1), but still materialized one `Vec` per label. The second fix streamed
//! directly into flat sum/count arrays but still hashed every element label.
//! The third fix used a dense table for compact integer labels but still called
//! `fract()` in the hot probe. The fourth fix cheapened that dense probe. This
//! bin reconstructs the historical routes, proves byte-identical means, and
//! times them against the current public `mean`.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_label_stats`.

use fsci_ndimage::{NdArray, mean};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Verbatim old behavior: per element, linear `position` scan over `index`.
fn old_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); index.len()];
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if let Some(pos) = index
            .iter()
            .position(|&wanted| label_value == wanted as f64)
        {
            groups[pos].push(value);
        }
    }
    groups
        .iter()
        .map(|g| {
            if g.is_empty() {
                0.0
            } else {
                g.iter().sum::<f64>() / g.len() as f64
            }
        })
        .collect()
}

/// Previous shipped behavior: O(N+K) label lookup, but still materializes
/// every label bucket before computing means.
fn bucketed_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); index.len()];
    let mut label_to_pos: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::with_capacity(index.len());
    for (pos, &wanted_label) in index.iter().enumerate() {
        label_to_pos
            .entry(label_key(wanted_label as f64))
            .or_insert(pos);
    }
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if let Some(&pos) = label_to_pos.get(&label_key(label_value)) {
            groups[pos].push(value);
        }
    }
    groups
        .iter()
        .map(|g| {
            if g.is_empty() {
                f64::NAN
            } else {
                g.iter().sum::<f64>() / g.len() as f64
            }
        })
        .collect()
}

/// Previous shipped `mean` behavior: O(N+K) label lookup with direct flat
/// sum/count accumulation, but one HashMap probe per input element.
fn flat_hash_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let mut sums = vec![0.0f64; index.len()];
    let mut counts = vec![0usize; index.len()];
    let mut label_to_pos: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::with_capacity(index.len());
    for (pos, &wanted_label) in index.iter().enumerate() {
        label_to_pos
            .entry(label_key(wanted_label as f64))
            .or_insert(pos);
    }
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if let Some(&pos) = label_to_pos.get(&label_key(label_value)) {
            sums[pos] += value;
            counts[pos] += 1;
        }
    }
    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect()
}

/// Previous shipped dense lookup: direct flat accumulation with a dense
/// label->position table, but the hot probe used `is_finite` + `fract`.
fn dense_fract_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let Some(max_label) = index.iter().copied().max() else {
        return Vec::new();
    };
    let mut label_to_pos = vec![usize::MAX; max_label + 1];
    for (pos, &wanted_label) in index.iter().enumerate() {
        if label_to_pos[wanted_label] == usize::MAX {
            label_to_pos[wanted_label] = pos;
        }
    }

    let mut sums = vec![0.0f64; index.len()];
    let mut counts = vec![0usize; index.len()];
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if !label_value.is_finite() || label_value < 0.0 || label_value.fract() != 0.0 {
            continue;
        }
        let label = label_value as usize;
        if let Some(&pos) = label_to_pos.get(label)
            && pos != usize::MAX
        {
            sums[pos] += value;
            counts[pos] += 1;
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect()
}

/// Previous shipped dense lookup after the cheap-probe fix: direct flat
/// accumulation with a dense label->position table and bounded integer cast.
fn dense_table_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let Some(max_label) = index.iter().copied().max() else {
        return Vec::new();
    };
    let mut label_to_pos = vec![usize::MAX; max_label + 1];
    for (pos, &wanted_label) in index.iter().enumerate() {
        if label_to_pos[wanted_label] == usize::MAX {
            label_to_pos[wanted_label] = pos;
        }
    }

    let mut sums = vec![0.0f64; index.len()];
    let mut counts = vec![0usize; index.len()];
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if !(label_value >= 0.0 && label_value < label_to_pos.len() as f64) {
            continue;
        }
        let label = label_value as usize;
        if label as f64 != label_value {
            continue;
        }
        let pos = label_to_pos[label];
        if pos != usize::MAX {
            sums[pos] += value;
            counts[pos] += 1;
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect()
}

fn label_key(x: f64) -> u64 {
    if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

fn main() {
    for &(side, k) in &[(256usize, 512usize), (512, 1024), (512, 2048), (768, 4096)] {
        let n = side * side;
        let mut r = Lcg(0xA11CE ^ (side as u64) ^ ((k as u64) << 20));
        // Random label in 1..=k for every cell; random values.
        let labels_data: Vec<f64> = (0..n)
            .map(|_| (1 + (r.next_u64() as usize) % k) as f64)
            .collect();
        let values: Vec<f64> = (0..n).map(|_| (r.next_u64() >> 11) as f64 * 1e-9).collect();
        let input = NdArray::new(values, vec![side, side]).unwrap();
        let labels = NdArray::new(labels_data, vec![side, side]).unwrap();
        let index: Vec<usize> = (1..=k).collect();

        // Correctness: byte-identical to the old linear-scan grouping.
        let new_means = mean(&input, Some(&labels), Some(&index)).unwrap();
        let old_means = old_mean_by_label(&input, &labels, &index);
        let bucketed_means = bucketed_mean_by_label(&input, &labels, &index);
        let hash_means = flat_hash_mean_by_label(&input, &labels, &index);
        let dense_fract_means = dense_fract_mean_by_label(&input, &labels, &index);
        let dense_table_means = dense_table_mean_by_label(&input, &labels, &index);
        let mismatches = new_means
            .iter()
            .zip(&old_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let bucketed_mismatches = new_means
            .iter()
            .zip(&bucketed_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let hash_mismatches = new_means
            .iter()
            .zip(&hash_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let dense_fract_mismatches = new_means
            .iter()
            .zip(&dense_fract_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let dense_table_mismatches = new_means
            .iter()
            .zip(&dense_table_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();

        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..3 {
            acc += old_mean_by_label(&input, &labels, &index)[0];
        }
        let old_t = t0.elapsed() / 3;

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += bucketed_mean_by_label(&input, &labels, &index)[0];
        }
        let bucketed_t = t1.elapsed() / 3;

        let t2 = Instant::now();
        for _ in 0..3 {
            acc += flat_hash_mean_by_label(&input, &labels, &index)[0];
        }
        let hash_t = t2.elapsed() / 3;

        let t3 = Instant::now();
        for _ in 0..3 {
            acc += dense_fract_mean_by_label(&input, &labels, &index)[0];
        }
        let dense_fract_t = t3.elapsed() / 3;

        let t4 = Instant::now();
        for _ in 0..3 {
            acc += dense_table_mean_by_label(&input, &labels, &index)[0];
        }
        let dense_table_t = t4.elapsed() / 3;

        let t5 = Instant::now();
        for _ in 0..3 {
            acc += mean(&input, Some(&labels), Some(&index)).unwrap()[0];
        }
        let one_based_t = t5.elapsed() / 3;

        let old_ratio = old_t.as_secs_f64() / one_based_t.as_secs_f64();
        let bucketed_ratio = bucketed_t.as_secs_f64() / one_based_t.as_secs_f64();
        let hash_ratio = hash_t.as_secs_f64() / one_based_t.as_secs_f64();
        let dense_fract_ratio = dense_fract_t.as_secs_f64() / one_based_t.as_secs_f64();
        let dense_table_ratio = dense_table_t.as_secs_f64() / one_based_t.as_secs_f64();
        println!(
            "N={:>7} K={:>5}  old(O(N*K))={:>10.3?}  bucketed(O(N+K))={:>10.3?}  hashflat(O(N+K))={:>10.3?}  dense_fract(O(N+K))={:>10.3?}  dense_table(O(N+K))={:>10.3?}  one_based(O(N+K))={:>10.3?}  old/fast={old_ratio:>7.1}x  bucketed/fast={bucketed_ratio:>6.2}x  hash/fast={hash_ratio:>6.2}x  fract/fast={dense_fract_ratio:>6.2}x  table/fast={dense_table_ratio:>6.2}x  mism={mismatches}/{bucketed_mismatches}/{hash_mismatches}/{dense_fract_mismatches}/{dense_table_mismatches}  (acc={acc:.3})",
            n, k, old_t, bucketed_t, hash_t, dense_fract_t, dense_table_t, one_based_t
        );
    }
}
