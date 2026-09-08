#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::{find_peaks, FindPeaksOptions};
use libfuzzer_sys::fuzz_target;

// Signal peak detection oracle:
// Tests find_peaks for correctness properties:
//
// 1. All returned peaks are local maxima
// 2. Peak indices are within bounds
// 3. Peak indices are sorted and unique
// 4. Height filter is respected
// 5. Distance filter is respected (no two peaks closer than min_distance)
// 6. Prominence values are non-negative

const MAX_LEN: usize = 256;

#[derive(Debug, Arbitrary)]
struct FindPeaksInput {
    signal: Vec<f64>,
    height: Option<f64>,
    distance: Option<u8>,
    prominence: Option<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fuzz_target!(|input: FindPeaksInput| {
    let n = input.signal.len().min(MAX_LEN);
    if n < 3 {
        return;
    }

    let signal: Vec<f64> = input.signal.iter().take(n).map(|&v| sanitize(v)).collect();

    let height = input.height.map(sanitize);
    let distance = input.distance.map(|d| (d as usize).max(1));
    let prominence = input.prominence.map(|p| sanitize(p).abs());

    let options = FindPeaksOptions {
        height,
        distance,
        prominence,
        width: None,
    };

    let result = find_peaks(&signal, options);

    if result.peaks.len() != result.peak_heights.len() {
        panic!(
            "find_peaks peaks count {} != peak_heights count {}",
            result.peaks.len(),
            result.peak_heights.len()
        );
    }

    if result.peaks.len() != result.prominences.len() {
        panic!(
            "find_peaks peaks count {} != prominences count {}",
            result.peaks.len(),
            result.prominences.len()
        );
    }

    for (i, &peak_idx) in result.peaks.iter().enumerate() {
        if peak_idx >= n {
            panic!(
                "find_peaks peak index {} out of bounds (n={})",
                peak_idx, n
            );
        }

        if peak_idx == 0 || peak_idx == n - 1 {
            panic!(
                "find_peaks peak at boundary index {} (n={})",
                peak_idx, n
            );
        }

        let peak_val = signal[peak_idx];

        if peak_val < signal[peak_idx - 1] || peak_val < signal[peak_idx + 1] {
            panic!(
                "find_peaks index {} is not a local maximum: val={}, left={}, right={}",
                peak_idx,
                peak_val,
                signal[peak_idx - 1],
                signal[peak_idx + 1]
            );
        }

        if let Some(min_height) = height
            && peak_val < min_height
        {
            panic!(
                "find_peaks peak {} height {} below min_height {}",
                peak_idx, peak_val, min_height
            );
        }

        if (result.peak_heights[i] - peak_val).abs() > 1e-10 {
            panic!(
                "find_peaks peak_heights[{}]={} != signal[{}]={}",
                i, result.peak_heights[i], peak_idx, peak_val
            );
        }

        if result.prominences[i] < 0.0 {
            panic!(
                "find_peaks prominence {} at peak {} is negative",
                result.prominences[i], peak_idx
            );
        }
    }

    for i in 1..result.peaks.len() {
        if result.peaks[i] <= result.peaks[i - 1] {
            panic!(
                "find_peaks peaks not sorted/unique: peaks[{}]={} <= peaks[{}]={}",
                i,
                result.peaks[i],
                i - 1,
                result.peaks[i - 1]
            );
        }
    }

    if let Some(min_dist) = distance {
        for i in 1..result.peaks.len() {
            let dist = result.peaks[i] - result.peaks[i - 1];
            if dist < min_dist {
                panic!(
                    "find_peaks distance violation: peaks[{}]={} and peaks[{}]={} are {} apart, min_distance={}",
                    i - 1,
                    result.peaks[i - 1],
                    i,
                    result.peaks[i],
                    dist,
                    min_dist
                );
            }
        }
    }
});
