//! Verify BoundaryMode::Mirror against scipy.ndimage mode='mirror' (gitignored).
use fsci_ndimage::{
    BoundaryMode, NdArray, correlate1d, gaussian_filter1d, map_coordinates, minimum_filter1d,
    sobel, uniform_filter1d,
};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

fn main() {
    let m = BoundaryMode::Mirror;
    let input = NdArray::new(vec![1.0, 2.0, 4.0, 7.0, 3.0, 9.0, 5.0, 8.0], vec![8]).unwrap();
    dump(
        "corr_k3",
        &correlate1d(&input, &[1.0, 2.0, -1.0], 0, m, 0.0)
            .unwrap()
            .data,
    );
    dump(
        "corr_k5",
        &correlate1d(&input, &[1.0, -2.0, 3.0, -2.0, 1.0], 0, m, 0.0)
            .unwrap()
            .data,
    );
    dump(
        "gauss_s1",
        &gaussian_filter1d(&input, 1.0, 0, 0, m, 0.0).unwrap().data,
    );
    dump(
        "gauss_s2_o1",
        &gaussian_filter1d(&input, 2.0, 0, 1, m, 0.0).unwrap().data,
    );
    dump(
        "uniform3",
        &uniform_filter1d(&input, 3, 0, m, 0.0).unwrap().data,
    );
    dump(
        "minimum3",
        &minimum_filter1d(&input, 3, 0, m, 0.0).unwrap().data,
    );
    // 2-D sobel
    let img = NdArray::new(
        vec![1.0, 2.0, 4.0, 7.0, 3.0, 9.0, 5.0, 8.0, 1.0, 4.0, 3.0, 7.0],
        vec![3, 4],
    )
    .unwrap();
    dump("sobel0", &sobel(&img, 0, m, 0.0).unwrap().data);
    dump("sobel1", &sobel(&img, 1, m, 0.0).unwrap().data);
    // interpolation order 0 and 1 with mirror
    let coords = vec![-0.7, 0.5, 2.3, 5.5, 7.9, 9.0];
    dump(
        "mapcoord_o0",
        &map_coordinates(&input, std::slice::from_ref(&coords), 0, m, 0.0).unwrap(),
    );
    dump(
        "mapcoord_o1",
        &map_coordinates(&input, &[coords], 1, m, 0.0).unwrap(),
    );
    // order>=2 mirror must error
    match map_coordinates(&input, &[vec![2.3]], 3, m, 0.0) {
        Ok(_) => println!("mapcoord_o3|UNEXPECTED_OK"),
        Err(_) => println!("mapcoord_o3|ERR_AS_EXPECTED"),
    }
}
