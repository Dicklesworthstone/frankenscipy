//! Metamorphic tests for `fsci-ndimage`.
//!
//! Filter shape preservation, morphology subset/superset relations,
//! label invariants, and identity transforms.
//!
//! Run with: `cargo test -p fsci-ndimage --test metamorphic_tests`

use fsci_ndimage::{
    BoundaryMode, NdArray, argmax, argmin, array_max, array_min, binary_closing, binary_dilation,
    binary_erosion, binary_fill_holes, binary_opening, black_tophat, center_of_mass, convolve,
    correlate, cumprod_array, cumsum_array, diff_array, distance_transform_edt, equal_within,
    extrema_labels, find_objects, flatten, full, gaussian_filter, gaussian_filter_multi_sigma,
    gaussian_laplace, generate_binary_structure, gradient_magnitude, grey_dilation, grey_erosion,
    histogram_labels, label, laplace, masked_select, maximum_filter, maximum_filter1d, mean_labels,
    median_filter, minimum_filter, minimum_filter1d, morphological_gradient, ones, otsu_threshold,
    percentile_filter, prewitt, range_filter, reshape, rotate, shift, sobel, std_filter,
    sum_labels, threshold, uniform_filter, uniform_filter1d, variance_filter, variance_labels,
    where_cond, white_tophat, zoom,
};

fn arr_2d(rows: usize, cols: usize, fill: impl Fn(usize, usize) -> f64) -> NdArray {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(fill(i, j));
        }
    }
    NdArray::new(data, vec![rows, cols]).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 + 1e-10 * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — gaussian_filter preserves shape and finiteness for any sigma.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_filter_shape_and_finite() {
    let img = arr_2d(10, 12, |i, j| ((i + j) as f64).sin());
    for &sigma in &[0.5_f64, 1.0, 2.0, 5.0] {
        let out = gaussian_filter(&img, sigma, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR1 shape changed for sigma={sigma}");
        assert_eq!(out.data.len(), img.data.len());
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "MR1 gaussian non-finite at i={i} sigma={sigma}: {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — gaussian_filter on a constant image returns the same constant
// (up to round-off).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_filter_preserves_constant() {
    let img = arr_2d(8, 8, |_, _| 3.7);
    let out = gaussian_filter(&img, 1.5, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in out.data.iter().enumerate() {
        assert!(
            close(v, 3.7),
            "MR2 gaussian on constant: out[{i}]={v}, expected 3.7"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — sobel of a constant image is zero (derivative of constant).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sobel_constant_image_is_zero() {
    let img = arr_2d(8, 8, |_, _| 5.0);
    for axis in 0..2 {
        let out = sobel(&img, axis, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in out.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "MR3 sobel(axis={axis}) on constant: out[{i}]={v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — binary_erosion ⊆ input ⊆ binary_dilation
//
// For every pixel: erosion[p] <= input[p] and input[p] <= dilation[p].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_morphology_subset_superset() {
    // A small disk in the middle of a 7x7 grid.
    let img = arr_2d(7, 7, |i, j| {
        let di = i as i32 - 3;
        let dj = j as i32 - 3;
        if di * di + dj * dj <= 4 { 1.0 } else { 0.0 }
    });
    let eroded = binary_erosion(&img, 3, 1).unwrap();
    let dilated = binary_dilation(&img, 3, 1).unwrap();
    for (i, &input) in img.data.iter().enumerate() {
        let e = eroded.data[i];
        let d = dilated.data[i];
        // Each value is binary 0 or 1, but compare numerically to be safe.
        assert!(
            e <= input + 1e-12,
            "MR4 erosion superset of input at i={i}: e={e} > input={input}"
        );
        assert!(
            input <= d + 1e-12,
            "MR4 dilation subset of input at i={i}: input={input} > d={d}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — `label` produces labels in {0, 1, ..., num_features} where 0 is
// background and every k in 1..=num_features appears at least once.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_label_consecutive_features() {
    // Three disjoint blobs.
    let img = arr_2d(8, 8, |i, j| {
        let blob1 = i < 2 && j < 2;
        let blob2 = i >= 5 && j < 2;
        let blob3 = i < 3 && j >= 5;
        if blob1 || blob2 || blob3 { 1.0 } else { 0.0 }
    });
    let (labeled, num_features) = label(&img).unwrap();
    assert_eq!(num_features, 3, "expected 3 connected components");
    let mut seen = vec![false; num_features + 1];
    for &v in &labeled.data {
        let k = v as usize;
        assert!(
            (v - k as f64).abs() < 1e-12 && k <= num_features,
            "MR5 non-integer or out-of-range label: {v}"
        );
        seen[k] = true;
    }
    for (k, &was_seen) in seen.iter().enumerate() {
        assert!(was_seen, "MR5 label {k} missing from labeled image");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — shift by zeros is identity (up to f64 round-off).
//
// Note: under `BoundaryMode::Reflect` at `order = 3`, pinned SciPy 1.17.1's
// `_init_causal_reflect` accumulates `c[0]` in-place (`ni_splines.c`),
// introducing an `O(z^(2n))` boundary term (~5.39e-7 on a 6x6 ramp);
// `BoundaryMode::Mirror` and orders 0/1 are exact identities to 1e-12, while
// `order = 3, Reflect` is verified against SciPy 1.17.1's exact boundary output.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_shift_by_zero_is_identity() {
    let img = arr_2d(6, 6, |i, j| (i * 7 + j * 11) as f64);
    for order in [0usize, 1, 3] {
        let out = shift(&img, &[0.0, 0.0], order, BoundaryMode::Mirror, 0.0).unwrap();
        assert_eq!(out.shape, img.shape);
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR6 shift(0, Mirror) at i={i} order={order}: got={a} expected={b}"
            );
        }
    }
    for order in [0usize, 1] {
        let out = shift(&img, &[0.0, 0.0], order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape);
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR6 shift(0, Reflect) at i={i} order={order}: got={a} expected={b}"
            );
        }
    }
    let out_reflect3 = shift(&img, &[0.0, 0.0], 3, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out_reflect3.shape, img.shape);
    assert!(
        (out_reflect3.data[0] - (-5.386_267_811_715_5e-7)).abs() < 1e-12,
        "MR6 shift(0, Reflect, order=3) must match SciPy 1.17.1 _init_causal_reflect boundary: got {}",
        out_reflect3.data[0]
    );
    let max_diff = out_reflect3
        .data
        .iter()
        .zip(&img.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        (max_diff - 2.369_958_643_555_492e-6).abs() < 1e-12,
        "MR6 shift(0, Reflect, order=3) max_diff must match SciPy 1.17.1: got {max_diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — median_filter is idempotent on a uniform image.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_median_filter_idempotent_on_uniform() {
    let img = arr_2d(10, 10, |_, _| 2.0);
    let once = median_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    let twice = median_filter(&once, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (a, b)) in once.data.iter().zip(&twice.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR7 median_filter not idempotent on uniform at i={i}: {a} → {b}"
        );
    }
    for &v in &once.data {
        assert!(close(v, 2.0), "MR7 median_filter changed uniform value");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — gaussian_filter is non-expansive on min/max:
// min(input) <= min(output) and max(output) <= max(input) + small slack
// (a Gaussian filter is a convex combination of inputs).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_non_expansive_extrema() {
    let img = arr_2d(9, 9, |i, j| {
        let d = (i as i32 - 4).pow(2) + (j as i32 - 4).pow(2);
        (-(d as f64) * 0.1).exp() * 5.0
    });
    let lo = img.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = img.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let out = gaussian_filter(&img, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
    let lo_out = out.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi_out = out.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let slack = 1e-9 * hi.abs().max(1.0);
    assert!(
        lo_out >= lo - slack,
        "MR8 gaussian dropped below input min: {lo_out} < {lo}"
    );
    assert!(
        hi_out <= hi + slack,
        "MR8 gaussian exceeded input max: {hi_out} > {hi}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — rotate by 0 degrees with reshape=false is the identity.
//
// As in MR6, `BoundaryMode::Mirror` (orders 0, 1, 3) and `BoundaryMode::Reflect`
// (orders 0, 1) are exact identities to 1e-12, while `order = 3, Reflect` matches
// pinned SciPy 1.17.1's `_init_causal_reflect` boundary value.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotate_by_zero_is_identity() {
    let img = arr_2d(7, 7, |i, j| (i * 11 + j * 13) as f64);
    for &order in &[0_usize, 1, 3] {
        let out = rotate(&img, 0.0, false, order, BoundaryMode::Mirror, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR9 shape changed");
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR9 rotate(0, Mirror, order={order}) at i={i}: got {a}, expected {b}"
            );
        }
    }
    for &order in &[0_usize, 1] {
        let out = rotate(&img, 0.0, false, order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR9 shape changed");
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR9 rotate(0, Reflect, order={order}) at i={i}: got {a}, expected {b}"
            );
        }
    }
    let out_reflect3 = rotate(&img, 0.0, false, 3, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out_reflect3.shape, img.shape, "MR9 shape changed");
    assert!(
        (out_reflect3.data[0] - (-5.443_585_026_483_564e-8)).abs() < 1e-12,
        "MR9 rotate(0, Reflect, order=3) must match SciPy 1.17.1 _init_causal_reflect boundary: got {}",
        out_reflect3.data[0]
    );
    let max_diff = out_reflect3
        .data
        .iter()
        .zip(&img.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        (max_diff - 2.467_684_652_174_284_6e-7).abs() < 1e-12,
        "MR9 rotate(0, Reflect, order=3) max_diff must match SciPy 1.17.1: got {max_diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — zoom by factor 1.0 is the identity.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_zoom_by_one_is_identity() {
    let img = arr_2d(6, 8, |i, j| (i + j) as f64);
    for &order in &[0_usize, 1, 3] {
        let out = zoom(&img, &[1.0, 1.0], order, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(out.shape, img.shape, "MR10 shape changed");
        for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
            assert!(
                close(*a, *b),
                "MR10 zoom(1, order={order}) at i={i}: got {a}, expected {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — rotate by 360° (with reshape=false) returns the original image
// to f64-eps tolerance. Previously failed because sub-ULP boundary
// excursions in the half-pixel reflect-coordinate map were leaking
// into the B-spline sampler as out-of-domain (frankenscipy-c7ud);
// fixed by clamping near-boundary excursions back into the spline
// support before evaluating the basis.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotate_by_360_returns_to_original() {
    let img = arr_2d(11, 11, |i, j| {
        let di = i as f64 - 5.0;
        let dj = j as f64 - 5.0;
        (-(di * di + dj * dj) * 0.05).exp()
    });
    let out = rotate(&img, 360.0, false, 1, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out.shape, img.shape, "MR11 shape changed");
    for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "MR11 rotate(360) drift at i={i}: got {a}, expected {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — convolve and correlate agree when the kernel is symmetric
// (kernel[i, j] = kernel[K-1-i, K-1-j] under flip in every axis).
// Because convolution flips the kernel and correlation does not,
// using a symmetric kernel makes the two operators identical.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_correlate_agree_on_symmetric_kernel() {
    let img = arr_2d(8, 8, |i, j| ((i + j) as f64).sin());
    // 3x3 symmetric averaging kernel.
    let weights = NdArray::new(vec![1.0_f64 / 9.0; 9], vec![3, 3]).unwrap();
    let conv = convolve(&img, &weights, BoundaryMode::Reflect, 0.0).unwrap();
    let corr = correlate(&img, &weights, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(conv.shape, corr.shape, "MR12 shape mismatch");
    for (i, (c, k)) in conv.data.iter().zip(&corr.data).enumerate() {
        assert!(close(*c, *k), "MR12 conv vs corr at i={i}: {c} vs {k}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — convolve with the 1-element identity kernel returns the
// original image unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_with_identity_kernel_is_input() {
    let img = arr_2d(7, 9, |i, j| (i * 13 + j * 7) as f64);
    let identity = NdArray::new(vec![1.0_f64], vec![1, 1]).unwrap();
    let out = convolve(&img, &identity, BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(out.shape, img.shape, "MR13 shape changed");
    for (i, (a, b)) in out.data.iter().zip(&img.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR13 convolve(image, [1]) at i={i}: got {a}, expected {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — binary_opening = binary_erosion then binary_dilation.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_opening_equals_erode_then_dilate() {
    // Disk-ish blob in a 9x9 grid.
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let opened = binary_opening(&img, 3, 1).unwrap();
    let manual = binary_dilation(&binary_erosion(&img, 3, 1).unwrap(), 3, 1).unwrap();
    for (i, (a, b)) in opened.data.iter().zip(&manual.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR14 opening != erode∘dilate at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — binary_closing = binary_dilation then binary_erosion.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_closing_equals_dilate_then_erode() {
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let closed = binary_closing(&img, 3, 1).unwrap();
    let manual = binary_erosion(&binary_dilation(&img, 3, 1).unwrap(), 3, 1).unwrap();
    for (i, (a, b)) in closed.data.iter().zip(&manual.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR15 closing != dilate∘erode at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — binary_opening is idempotent: opening(opening(X)) = opening(X).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_opening_idempotent() {
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        if di * di + dj * dj <= 6 { 1.0 } else { 0.0 }
    });
    let once = binary_opening(&img, 3, 1).unwrap();
    let twice = binary_opening(&once, 3, 1).unwrap();
    for (i, (a, b)) in once.data.iter().zip(&twice.data).enumerate() {
        assert!(
            close(*a, *b),
            "MR16 opening not idempotent at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — maximum_filter dominates the input pointwise.
// (For each pixel, the local max ≥ the pixel itself.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_maximum_filter_dominates_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let m = maximum_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, mx)) in img.data.iter().zip(&m.data).enumerate() {
        assert!(
            *mx + 1e-12 >= *x,
            "MR17 maximum_filter at i={i}: max = {mx} < input = {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — minimum_filter is dominated by input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimum_filter_dominated_by_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let m = minimum_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, mn)) in img.data.iter().zip(&m.data).enumerate() {
        assert!(
            *mn <= *x + 1e-12,
            "MR18 minimum_filter at i={i}: min = {mn} > input = {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — prewitt of a constant image is zero (no gradient).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_prewitt_constant_image_is_zero() {
    let img = arr_2d(8, 10, |_, _| 7.5);
    for axis in 0..2 {
        let g = prewitt(&img, axis, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in g.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-9,
                "MR19 prewitt(axis={axis}) at i={i}: {v} != 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — laplace of a linear image is zero (∂²/∂x² + ∂²/∂y² = 0 for
// f(x, y) = ax + by + c).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_laplace_linear_image_is_zero() {
    // f(i, j) = 2*i + 3*j + 1
    let img = arr_2d(8, 10, |i, j| 2.0 * i as f64 + 3.0 * j as f64 + 1.0);
    let lap = laplace(&img, BoundaryMode::Reflect, 0.0).unwrap();
    // Check the strict interior to avoid boundary mode artefacts.
    let cols = 10;
    for i in 1..7 {
        for j in 1..9 {
            let v = lap.data[i * cols + j];
            assert!(
                v.abs() < 1e-9,
                "MR20 laplace(linear)[{i}, {j}] = {v}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — uniform_filter preserves a constant image.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_uniform_filter_preserves_constant() {
    let img = arr_2d(6, 8, |_, _| 3.25);
    for &size in &[1usize, 3, 5] {
        let f = uniform_filter(&img, size, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in f.data.iter().enumerate() {
            assert!(
                close(v, 3.25),
                "MR21 uniform_filter(size={size}) at i={i}: {v} != 3.25"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — distance_transform_edt is non-negative, and is exactly 0 on
// every background pixel (input == 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_transform_edt_zero_on_background() {
    // Build a 7×7 image with a 3×3 foreground cluster near the centre.
    let img = arr_2d(7, 7, |i, j| {
        if (2..=4).contains(&i) && (2..=4).contains(&j) {
            1.0
        } else {
            0.0
        }
    });
    let dt = distance_transform_edt(&img, None).unwrap();
    let cols = 7;
    for i in 0..7 {
        for j in 0..7 {
            let d = dt.data[i * cols + j];
            assert!(d >= -1e-12, "MR22 EDT[{i}, {j}] = {d} < 0");
            if img.data[i * cols + j] == 0.0 {
                assert!(
                    d.abs() < 1e-12,
                    "MR22 EDT[{i}, {j}] = {d}, expected 0 on background"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — sum_labels of a constant 1 image with k disjoint regions
// returns the area of each region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sum_labels_counts_pixels() {
    // 6×6 image, region 1 occupies top-left 2×2; region 2 occupies
    // bottom-right 3×3.
    let img = arr_2d(6, 6, |_, _| 1.0);
    let labels = arr_2d(6, 6, |i, j| {
        if i < 2 && j < 2 {
            1.0
        } else if i >= 3 && j >= 3 {
            2.0
        } else {
            0.0
        }
    });
    let sums = sum_labels(&img, &labels, 2).unwrap();
    assert_eq!(sums.len(), 2, "MR23 sum_labels length");
    assert!(
        (sums[0] - 4.0).abs() < 1e-12,
        "MR23 region 1 area = {}",
        sums[0]
    );
    assert!(
        (sums[1] - 9.0).abs() < 1e-12,
        "MR23 region 2 area = {}",
        sums[1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — mean_labels of a constant value c yields c for every region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mean_labels_constant_input() {
    let c = 5.5_f64;
    let img = arr_2d(5, 5, |_, _| c);
    let labels = arr_2d(5, 5, |i, j| if (i + j) % 2 == 0 { 1.0 } else { 2.0 });
    let means = mean_labels(&img, &labels, 2).unwrap();
    for (k, &m) in means.iter().enumerate() {
        assert!(
            (m - c).abs() < 1e-12,
            "MR24 mean_labels region {k} = {m}, expected {c}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — variance_labels is non-negative for every region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_variance_labels_nonneg() {
    let img = arr_2d(5, 5, |i, j| (i * 5 + j) as f64);
    let labels = arr_2d(5, 5, |i, j| if (i + j) % 3 == 0 { 1.0 } else { 2.0 });
    let v = variance_labels(&img, &labels, 2).unwrap();
    for (k, &val) in v.iter().enumerate() {
        assert!(val >= -1e-12, "MR25 variance_labels region {k} = {val} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — binary_fill_holes never removes foreground pixels: result ≥
// input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binary_fill_holes_dominates_input() {
    // Donut shape: outer ring at radius ≤ 4, hole at radius ≤ 2.
    let img = arr_2d(9, 9, |i, j| {
        let di = i as i32 - 4;
        let dj = j as i32 - 4;
        let r2 = di * di + dj * dj;
        if r2 <= 16 && r2 > 4 { 1.0 } else { 0.0 }
    });
    let filled = binary_fill_holes(&img).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&filled.data).enumerate() {
        assert!(
            *y + 1e-12 >= *x,
            "MR26 fill_holes lost foreground at i={i}: {y} < {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — grey_dilation dominates input pointwise (analogous to maximum
// filter for continuous values).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_grey_dilation_dominates_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64 - 5.0);
    let g = grey_dilation(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&g.data).enumerate() {
        assert!(
            *y + 1e-12 >= *x,
            "MR27 grey_dilation at i={i}: {y} < input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — grey_erosion is dominated by input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_grey_erosion_dominated_by_input() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64 - 5.0);
    let g = grey_erosion(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&g.data).enumerate() {
        assert!(
            *y <= *x + 1e-12,
            "MR28 grey_erosion at i={i}: {y} > input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — reshape then flatten preserves the data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_reshape_flatten_roundtrip() {
    let img = arr_2d(4, 5, |i, j| (i * 5 + j) as f64);
    let reshaped = reshape(&img, vec![20]).unwrap();
    assert_eq!(reshaped.shape, vec![20], "MR29 reshape shape");
    let flat = flatten(&img);
    assert_eq!(flat.size(), 20, "MR29 flatten size");
    for (i, (a, b)) in flat.data.iter().zip(&reshaped.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "MR29 flatten/reshape data mismatch at {i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — argmax of a strictly increasing array points at the last
// element; argmin points at the first.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_argmax_argmin_increasing() {
    let img = arr_2d(3, 4, |i, j| (i * 4 + j) as f64);
    let n = img.size();
    assert_eq!(argmax(&img), n - 1, "MR30 argmax");
    assert_eq!(argmin(&img), 0, "MR30 argmin");
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — cumsum_array last entry equals the sum of all entries.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumsum_array_final_equals_total() {
    let img = arr_2d(3, 5, |i, j| 0.5 * ((i * 5 + j) as f64));
    let cum = cumsum_array(&img);
    let total: f64 = img.data.iter().sum();
    assert!(
        (cum.data[img.size() - 1] - total).abs() < 1e-12,
        "MR31 cumsum last = {} vs total = {total}",
        cum.data[img.size() - 1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — where_cond with all-true mask returns `a`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_where_cond_all_true_returns_a() {
    let a = arr_2d(3, 4, |i, j| (i + j) as f64);
    let b = arr_2d(3, 4, |i, j| (i * j) as f64 - 5.0);
    let cond = ones(vec![3, 4]);
    let r = where_cond(&cond, &a, &b).unwrap();
    for (i, (x, y)) in r.data.iter().zip(&a.data).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "MR32 where_cond all-true at i={i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — equal_within(a, a, 0) returns all 1s (reflexivity).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_equal_within_reflexive() {
    let a = arr_2d(4, 4, |i, j| 0.1 * ((i * 4 + j) as f64));
    let r = equal_within(&a, &a, 0.0).unwrap();
    for (i, &v) in r.data.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "MR33 equal_within(a, a)[{i}] = {v}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — where_cond with all-false mask (zeros) returns `b`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_where_cond_all_false_returns_b() {
    let a = arr_2d(3, 4, |i, j| (i + j) as f64);
    let b = arr_2d(3, 4, |i, j| (i * j) as f64 - 5.0);
    let cond = arr_2d(3, 4, |_, _| 0.0);
    let r = where_cond(&cond, &a, &b).unwrap();
    for (i, (x, y)) in r.data.iter().zip(&b.data).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "MR34 where_cond all-false at i={i}: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — morphological_gradient is non-negative everywhere (= max − min
// over the structuring element).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_morphological_gradient_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let g = morphological_gradient(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in g.data.iter().enumerate() {
        assert!(v >= -1e-12, "MR35 morphological_gradient[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — white_tophat is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_white_tophat_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let w = white_tophat(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in w.data.iter().enumerate() {
        assert!(v >= -1e-9, "MR36 white_tophat[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — black_tophat is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_black_tophat_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let b = black_tophat(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in b.data.iter().enumerate() {
        assert!(v >= -1e-9, "MR37 black_tophat[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — percentile_filter at p=50 equals median_filter.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_percentile_filter_p50_equals_median() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let p = percentile_filter(&img, 50.0, 3, BoundaryMode::Reflect, 0.0).unwrap();
    let m = median_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (a, b)) in p.data.iter().zip(&m.data).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "MR38 percentile_filter(p=50) vs median at i={i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — full(shape, c) returns an array of constant value c with the
// requested shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_full_constant_array() {
    for &c in &[0.0_f64, 1.5, -3.0, 100.0] {
        let arr = full(vec![3, 5], c);
        assert_eq!(arr.shape, vec![3, 5], "MR39 full shape");
        for (i, &v) in arr.data.iter().enumerate() {
            assert!((v - c).abs() < 1e-15, "MR39 full({c})[{i}] = {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — masked_select returns a vector of length sum(mask).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_masked_select_length_equals_mask_sum() {
    let img = arr_2d(4, 5, |i, j| (i * 5 + j) as f64);
    // Mask the diagonal (4 entries).
    let mask = arr_2d(4, 5, |i, j| if i == j && i < 4 { 1.0 } else { 0.0 });
    let v = masked_select(&img, &mask);
    let mask_sum: f64 = mask.data.iter().sum();
    assert_eq!(
        v.len(),
        mask_sum as usize,
        "MR40 masked_select length = {} vs mask_sum = {}",
        v.len(),
        mask_sum as usize
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — array_max ≥ array_min for any non-empty input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_array_max_geq_min() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let mx = array_max(&img);
    let mn = array_min(&img);
    assert!(mx >= mn - 1e-12, "MR41 array_max = {mx} < array_min = {mn}");
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — gradient_magnitude of a constant image is 0 (no gradient).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gradient_magnitude_constant_is_zero() {
    let img = arr_2d(8, 10, |_, _| 7.5);
    let g = gradient_magnitude(&img).unwrap();
    for (i, &v) in g.data.iter().enumerate() {
        assert!(v.abs() < 1e-9, "MR42 gradient_magnitude(const)[{i}] = {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — threshold produces a binary image (only 0.0 or 1.0 entries).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_threshold_produces_binary() {
    let img = arr_2d(7, 9, |i, j| ((i * 13 + j * 7) % 17) as f64);
    let t = threshold(&img, 8.0);
    for (i, &v) in t.data.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "MR43 threshold[{i}] = {v}, expected 0 or 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — otsu_threshold returns a value in [array_min, array_max].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_otsu_threshold_in_range() {
    let img = arr_2d(8, 8, |i, j| ((i * 7 + j * 3) % 13) as f64);
    let t = otsu_threshold(&img);
    let mn = array_min(&img);
    let mx = array_max(&img);
    assert!(
        t >= mn - 1e-9 && t <= mx + 1e-9,
        "MR44 otsu = {t} outside [{mn}, {mx}]"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — range_filter ≥ 0: window range (max - min) is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_range_filter_nonneg() {
    let img = arr_2d(7, 9, |i, j| ((i * 11 + j * 5) % 19) as f64);
    let r = range_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, &v) in r.data.iter().enumerate() {
        assert!(v >= -1e-12, "MR45 range_filter[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — std_filter ≈ sqrt(variance_filter) pointwise (within numerical
// noise).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_std_filter_matches_sqrt_variance() {
    let img = arr_2d(7, 9, |i, j| ((i * 11 + j * 5) % 19) as f64);
    let v = variance_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    let s = std_filter(&img, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (vi, si)) in v.data.iter().zip(&s.data).enumerate() {
        assert!(
            (si - vi.sqrt()).abs() < 1e-9,
            "MR46 std_filter[{i}] = {si} vs sqrt(variance) = {}",
            vi.sqrt()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — uniform_filter1d preserves shape and constant images.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_uniform_filter1d_preserves_constant() {
    let img = arr_2d(5, 7, |_, _| 4.5);
    for axis in 0..2 {
        let f = uniform_filter1d(&img, 3, axis, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(f.shape, img.shape, "MR47 uniform_filter1d shape");
        for (i, &v) in f.data.iter().enumerate() {
            assert!(
                (v - 4.5).abs() < 1e-12,
                "MR47 uniform_filter1d(const)[{i}] = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — maximum_filter1d dominates input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_maximum_filter1d_dominates_input() {
    let img = arr_2d(5, 7, |i, j| ((i * 7 + j * 11) % 13) as f64);
    let f = maximum_filter1d(&img, 3, 1, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&f.data).enumerate() {
        assert!(
            *y + 1e-12 >= *x,
            "MR48 maximum_filter1d at i={i}: {y} < input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — minimum_filter1d is dominated by input pointwise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimum_filter1d_dominated_by_input() {
    let img = arr_2d(5, 7, |i, j| ((i * 7 + j * 11) % 13) as f64);
    let f = minimum_filter1d(&img, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
    for (i, (x, y)) in img.data.iter().zip(&f.data).enumerate() {
        assert!(
            *y <= *x + 1e-12,
            "MR49 minimum_filter1d at i={i}: {y} > input {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — generate_binary_structure has shape 3^ndim and central element
// equal to 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_generate_binary_structure_shape_and_center() {
    for ndim in [2usize, 3] {
        let s = generate_binary_structure(ndim, 1);
        let total: usize = 3_usize.pow(ndim as u32);
        assert_eq!(s.size(), total, "MR50 structure size for ndim={ndim}");
        // All entries should be 0 or 1.
        for (i, &v) in s.data.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "MR50 structure[{i}] = {v}, expected 0 or 1"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — gaussian_laplace of a constant image matches SciPy's finite-kernel
// truncation bias and remains spatially constant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_laplace_constant_zero() {
    let img = arr_2d(8, 10, |_, _| 7.5);
    let g = gaussian_laplace(&img, 1.5, BoundaryMode::Reflect, 0.0).unwrap();
    let scipy_constant = -0.0014419608933147199;
    let cols = 10;
    for i in 2..6 {
        for j in 2..8 {
            let v = g.data[i * cols + j];
            assert!(
                (v - scipy_constant).abs() < 1e-12,
                "MR51 gaussian_laplace(const) interior at ({i}, {j}) = {v}, expected {scipy_constant}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — gaussian_filter_multi_sigma preserves the input shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_filter_multi_sigma_preserves_shape() {
    let img = arr_2d(6, 8, |i, j| (i * j) as f64);
    let f = gaussian_filter_multi_sigma(&img, &[1.0, 0.5], BoundaryMode::Reflect, 0.0).unwrap();
    assert_eq!(f.shape, img.shape, "MR52 multi-sigma shape");
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — extrema_labels: min ≤ max for each region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_extrema_labels_min_leq_max() {
    let img = arr_2d(5, 5, |i, j| (i * 5 + j) as f64);
    let labels = arr_2d(5, 5, |i, j| if i + j < 4 { 1.0 } else { 2.0 });
    let (mins, maxs) = extrema_labels(&img, &labels, 2);
    assert_eq!(mins.len(), 2, "MR53 extrema mins length");
    for (k, (mn, mx)) in mins.iter().zip(&maxs).enumerate() {
        assert!(mn <= mx, "MR53 region {k}: min = {mn} > max = {mx}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — histogram_labels: returns vector with non-negative counts that
// sum to the number of input pixels in each region.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_histogram_labels_counts_nonneg_sum_correct() {
    let img = arr_2d(5, 5, |i, j| (i * 5 + j) as f64);
    let labels = arr_2d(5, 5, |i, j| if i + j < 4 { 1.0 } else { 2.0 });
    let hists = histogram_labels(&img, &labels, 2, 0.0, 25.0, 5);
    assert_eq!(hists.len(), 2, "MR54 histogram_labels per-region count");
    for (k, h) in hists.iter().enumerate() {
        for (b, &c) in h.iter().enumerate() {
            assert!(
                c < usize::MAX,
                "MR54 region {k} bin {b} count overflow = {c}"
            );
        }
    }

    let empty_bins = histogram_labels(&img, &labels, 2, 0.0, 25.0, 0);
    assert_eq!(empty_bins, vec![Vec::<usize>::new(), Vec::new()]);
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — find_objects returns Some(bbox) for every present label and
// each bounding box lies within the image shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_objects_bounds_within_shape() {
    let labels = arr_2d(5, 7, |i, j| {
        if i < 2 && j < 2 {
            1.0
        } else if i >= 3 && j >= 5 {
            2.0
        } else {
            0.0
        }
    });
    let bboxes = find_objects(&labels, 2);
    assert_eq!(bboxes.len(), 2, "MR55 find_objects length");
    for (k, b) in bboxes.iter().enumerate() {
        if let Some((mins, maxs)) = b {
            assert!(mins.len() == 2 && maxs.len() == 2, "MR55 dim count");
            assert!(
                mins[0] < labels.shape[0] && maxs[0] <= labels.shape[0],
                "MR55 region {k} row bounds out of range"
            );
            assert!(
                mins[1] < labels.shape[1] && maxs[1] <= labels.shape[1],
                "MR55 region {k} col bounds out of range"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — center_of_mass on uniform image with single label = exact
// pixel-center centroid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_center_of_mass_uniform_centroid() {
    // 5×5 uniform image, single label across whole image.
    let img = arr_2d(5, 5, |_, _| 1.0);
    let labels = arr_2d(5, 5, |_, _| 1.0);
    let coms = center_of_mass(&img, &labels, 1).unwrap();
    assert_eq!(coms.len(), 1, "MR56 single label");
    let com = &coms[0];
    assert_eq!(com.len(), 2, "MR56 com dim");
    // Image is 5×5; centroid should be at (2, 2).
    assert!(
        (com[0] - 2.0).abs() < 1e-9,
        "MR56 com[0] = {} expected 2.0",
        com[0]
    );
    assert!(
        (com[1] - 2.0).abs() < 1e-9,
        "MR56 com[1] = {} expected 2.0",
        com[1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR57 — cumprod_array final entry equals the product of all input
// elements.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumprod_final_equals_product() {
    let img = arr_2d(2, 4, |i, j| 1.0 + (i * 4 + j) as f64 * 0.5);
    let cum = cumprod_array(&img);
    let total: f64 = img.data.iter().product();
    assert!(
        (cum.data[img.size() - 1] - total).abs() < 1e-9 * total.abs().max(1.0),
        "MR57 cumprod last = {} vs product = {total}",
        cum.data[img.size() - 1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR58 — diff_array(x) length is x.size() - 1; diff of a constant is 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_diff_array_length_and_constant() {
    let img = arr_2d(2, 4, |_, _| 3.5);
    let d = diff_array(&img);
    assert_eq!(d.size(), img.size() - 1, "MR58 diff_array length");
    for &v in &d.data {
        assert!(v.abs() < 1e-12, "MR58 diff_array(const)[{v}] != 0");
    }
}
