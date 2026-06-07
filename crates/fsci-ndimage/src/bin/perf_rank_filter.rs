//! Deterministic golden output for ndimage rank/median filter optimization work.

use fsci_ndimage::{
    BoundaryMode, NdArray, median_filter, median_filter_axes, rank_filter, rank_filter_axes,
};
use std::fmt::Write as _;
use std::path::Path;

fn deterministic_image(rows: usize, cols: usize) -> NdArray {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let value = match i % 41 {
            0 => -0.0,
            1 => 0.0,
            2 => f64::from_bits(0x7ff8_0000_0000_0001),
            3 => f64::NEG_INFINITY,
            4 => f64::INFINITY,
            _ => {
                let raw = ((i * 37 + (i / cols) * 11 + (i % cols) * 17) % 211) as i64 - 105;
                raw as f64 / 8.0
            }
        };
        data.push(value);
    }
    NdArray::new(data, vec![rows, cols]).expect("deterministic image")
}

fn digest_array(arr: &NdArray) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &dim in &arr.shape {
        h ^= dim as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for (i, &value) in arr.data.iter().enumerate() {
        h ^= (i as u64).rotate_left(17) ^ value.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn append_case(out: &mut String, label: &str, arr: &NdArray) {
    let mid = arr.data.len() / 2;
    writeln!(
        out,
        "{label} shape={:?} len={} digest={:016x} first={:016x} mid={:016x} last={:016x}",
        arr.shape,
        arr.data.len(),
        digest_array(arr),
        arr.data[0].to_bits(),
        arr.data[mid].to_bits(),
        arr.data[arr.data.len() - 1].to_bits()
    )
    .expect("write golden line");
}

fn golden_text() -> String {
    let image = deterministic_image(34, 29);
    let mut out = String::from("fsci-ndimage rank-filter golden v1\n");

    append_case(
        &mut out,
        "median_all_reflect_size7",
        &median_filter(&image, 7, BoundaryMode::Reflect, -13.0).expect("median reflect"),
    );
    append_case(
        &mut out,
        "median_all_constant_size15",
        &median_filter(
            &image,
            15,
            BoundaryMode::Constant,
            f64::from_bits(0xfff8_0000_0000_0041),
        )
        .expect("median constant"),
    );
    append_case(
        &mut out,
        "rank_all_reflect_size7_rank12",
        &rank_filter(&image, 12, 7, BoundaryMode::Reflect, -13.0).expect("rank reflect"),
    );
    append_case(
        &mut out,
        "rank_all_constant_size15_rank56",
        &rank_filter(&image, 56, 15, BoundaryMode::Constant, -19.0).expect("rank constant"),
    );
    append_case(
        &mut out,
        "median_axis_last_wrap_size9",
        &median_filter_axes(&image, 9, &[-1], BoundaryMode::Wrap, -13.0).expect("median axes"),
    );
    append_case(
        &mut out,
        "rank_axis_first_nearest_size9_rank3",
        &rank_filter_axes(&image, 3, 9, &[-2], BoundaryMode::Nearest, -13.0).expect("rank axes"),
    );

    out
}

fn main() {
    let output = golden_text();
    if let Some(path) = std::env::args().nth(1) {
        if let Some(parent) = Path::new(&path).parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        println!("{}", output.trim_end());
    }
}
