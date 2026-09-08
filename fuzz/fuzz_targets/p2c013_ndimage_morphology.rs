#![no_main]

use arbitrary::Arbitrary;
use fsci_ndimage::{
    binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening, label,
    NdArray,
};
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 16;
const MAX_STRUCT_SIZE: usize = 5;
const MAX_ITERATIONS: usize = 3;

#[derive(Debug, Arbitrary)]
struct MorphInput {
    raw_data: Vec<u8>,
    width: u8,
    height: u8,
    structure_size: u8,
    iterations: u8,
}

fn to_binary_array(raw: &[u8], size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            if i < raw.len() && raw[i] > 127 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn arrays_equal(a: &NdArray, b: &NdArray) -> bool {
    if a.shape != b.shape {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|(&x, &y)| (x - y).abs() < 1e-10)
}

fn is_subset(subset: &NdArray, superset: &NdArray) -> bool {
    if subset.shape != superset.shape {
        return false;
    }
    subset
        .data
        .iter()
        .zip(superset.data.iter())
        .all(|(&s, &sup)| {
            let s_bin = s > 0.5;
            let sup_bin = sup > 0.5;
            !s_bin || sup_bin
        })
}

fn complement(arr: &NdArray) -> NdArray {
    let data: Vec<f64> = arr.data.iter().map(|&v| if v > 0.5 { 0.0 } else { 1.0 }).collect();
    NdArray::new(data, arr.shape.clone()).unwrap()
}

fuzz_target!(|input: MorphInput| {
    let w = (input.width as usize).clamp(1, MAX_DIM);
    let h = (input.height as usize).clamp(1, MAX_DIM);
    let struct_size = (input.structure_size as usize).clamp(1, MAX_STRUCT_SIZE);
    let iterations = (input.iterations as usize).clamp(1, MAX_ITERATIONS);

    let size = w * h;
    let data = to_binary_array(&input.raw_data, size);
    let array = match NdArray::new(data, vec![h, w]) {
        Ok(a) => a,
        Err(_) => return,
    };

    // Oracle 1: opening is idempotent — opening(opening(X)) == opening(X)
    if let Ok(opened1) = binary_opening(&array, struct_size, iterations)
        && let Ok(opened2) = binary_opening(&opened1, struct_size, iterations)
    {
        assert!(
            arrays_equal(&opened1, &opened2),
            "binary_opening not idempotent: shape={:?}, struct_size={}, iterations={}",
            array.shape,
            struct_size,
            iterations
        );
    }

    // Oracle 2: closing is idempotent — closing(closing(X)) == closing(X)
    if let Ok(closed1) = binary_closing(&array, struct_size, iterations)
        && let Ok(closed2) = binary_closing(&closed1, struct_size, iterations)
    {
        assert!(
            arrays_equal(&closed1, &closed2),
            "binary_closing not idempotent: shape={:?}, struct_size={}, iterations={}",
            array.shape,
            struct_size,
            iterations
        );
    }

    // Oracle 3: opening(X) ⊆ X (erosion removes pixels)
    if let Ok(opened) = binary_opening(&array, struct_size, iterations) {
        assert!(
            is_subset(&opened, &array),
            "binary_opening not subset of input: shape={:?}",
            array.shape
        );
    }

    // Oracle 4: X ⊆ closing(X) (dilation adds pixels)
    if let Ok(closed) = binary_closing(&array, struct_size, iterations) {
        assert!(
            is_subset(&array, &closed),
            "input not subset of binary_closing: shape={:?}",
            array.shape
        );
    }

    // Oracle 5: erosion duality — erosion(X) == complement(dilation(complement(X)))
    if let (Ok(eroded), Ok(dilated_comp)) = (
        binary_erosion(&array, struct_size, 1),
        binary_dilation(&complement(&array), struct_size, 1),
    ) {
        let comp_dilated_comp = complement(&dilated_comp);
        assert!(
            arrays_equal(&eroded, &comp_dilated_comp),
            "erosion/dilation duality violated: shape={:?}, struct_size={}",
            array.shape,
            struct_size
        );
    }

    // Oracle 6: label returns contiguous labels 1..num_features
    if let Ok((labeled, num_features)) = label(&array) {
        assert_eq!(
            labeled.shape, array.shape,
            "label output shape mismatch"
        );

        let mut seen_labels = std::collections::HashSet::new();
        for &v in &labeled.data {
            let lbl = v as usize;
            if lbl > 0 {
                seen_labels.insert(lbl);
            }
        }

        assert!(
            seen_labels.len() <= num_features,
            "label returned more distinct labels ({}) than num_features ({})",
            seen_labels.len(),
            num_features
        );

        for lbl in 1..=num_features {
            assert!(
                seen_labels.contains(&lbl) || num_features == 0,
                "label missing expected label {}, num_features={}, seen={:?}",
                lbl,
                num_features,
                seen_labels
            );
        }
    }

    // Oracle 7: binary_fill_holes on all-zeros returns all-zeros
    let zeros = NdArray::zeros(array.shape.clone());
    if let Ok(filled_zeros) = binary_fill_holes(&zeros) {
        let all_zero = filled_zeros.data.iter().all(|&v| v == 0.0);
        assert!(
            all_zero,
            "binary_fill_holes on zeros should return zeros"
        );
    }

    // Oracle 8: binary_fill_holes on all-ones returns all-ones
    let ones_data: Vec<f64> = vec![1.0; array.size()];
    let ones = NdArray::new(ones_data, array.shape.clone()).unwrap();
    if let Ok(filled_ones) = binary_fill_holes(&ones) {
        let all_one = filled_ones.data.iter().all(|&v| v == 1.0);
        assert!(
            all_one,
            "binary_fill_holes on ones should return ones"
        );
    }

    // Oracle 9: erosion shrinks or maintains count (never grows)
    if let Ok(eroded) = binary_erosion(&array, struct_size, iterations) {
        let orig_count: usize = array.data.iter().filter(|&&v| v > 0.5).count();
        let eroded_count: usize = eroded.data.iter().filter(|&&v| v > 0.5).count();
        assert!(
            eroded_count <= orig_count,
            "erosion increased pixel count: {} -> {}",
            orig_count,
            eroded_count
        );
    }

    // Oracle 10: dilation grows or maintains count (never shrinks)
    if let Ok(dilated) = binary_dilation(&array, struct_size, iterations) {
        let orig_count: usize = array.data.iter().filter(|&&v| v > 0.5).count();
        let dilated_count: usize = dilated.data.iter().filter(|&&v| v > 0.5).count();
        assert!(
            dilated_count >= orig_count,
            "dilation decreased pixel count: {} -> {}",
            orig_count,
            dilated_count
        );
    }
});
