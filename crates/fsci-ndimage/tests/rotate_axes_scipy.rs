#![forbid(unsafe_code)]

//! N-D rotation regressions for frankenscipy-o4zkq. Set
//! FSCI_REQUIRE_SCIPY_ORACLE=1 to require pinned SciPy 1.17.1 / NumPy 2.4.3.
//! FSCI_PYTHON (or SCIPY_PYTHON) selects the interpreter.

use fsci_ndimage::{BoundaryMode, NdArray, NdimageError, rotate, rotate_axes};
use std::fmt::Write;
use std::process::Command;

fn coordinates(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    index
}

fn input_for_shape(shape: &[usize]) -> NdArray {
    let data = (0..shape.iter().product::<usize>())
        .map(|i| (i % 17) as f64 * 0.25 - 2.0 + ((i / 7) % 5) as f64 * 0.125)
        .collect();
    NdArray::new(data, shape.to_vec()).unwrap()
}

#[test]
fn zero_angle_preserves_cardinal_samples_and_singleton_planes() {
    for shape in [vec![3, 2, 3], vec![1, 2, 3]] {
        let input = input_for_shape(&shape);
        for order in 0..=5 {
            for mode in [
                BoundaryMode::Constant,
                BoundaryMode::Nearest,
                BoundaryMode::Mirror,
                BoundaryMode::Wrap,
            ] {
                let output = rotate_axes(&input, 0.0, &[0, 2], true, order, mode, -0.75).unwrap();
                assert_eq!(output.shape, input.shape);
                for (&actual, &expected) in output.data.iter().zip(&input.data) {
                    assert!(
                        (actual - expected).abs() < 1e-10,
                        "shape={shape:?}, order={order}, mode={mode:?}: {actual} != {expected}"
                    );
                }
            }
        }
    }
    // Short reflect lines use SciPy's causal initial condition, which does
    // not reproduce the input exactly. The live zero-angle cases below check
    // those values against SciPy for every order instead of assuming identity.
}

#[test]
fn mirror_nearest_rounds_half_grid_coordinates_before_folding_taps() {
    let input = NdArray::new((0..10).map(f64::from).collect(), vec![2, 5]).unwrap();
    let output = rotate(&input, 90.0, false, 0, BoundaryMode::Mirror, -0.75).unwrap();
    // Output (0, 3) samples source (1.5, 2.5). SciPy retains 1.5 until
    // rounding, then folds integer row 2 to 0. Folding the continuous
    // coordinate first would select row 1 and incorrectly return 8.
    assert_eq!(output.shape, vec![2, 5]);
    assert_eq!(output.data, vec![8., 8., 8., 3., 8., 7., 7., 7., 2., 7.]);
}

#[test]
fn quarter_turns_permute_every_four_dimensional_plane() {
    let shape = vec![2, 3, 4, 5];
    let input = NdArray::new((0..120).map(f64::from).collect(), shape.clone()).unwrap();
    for row_axis in 0..shape.len() {
        for col_axis in row_axis + 1..shape.len() {
            for angle in [90.0, -90.0, 180.0, 270.0] {
                let rotated = rotate_axes(
                    &input,
                    angle,
                    &[row_axis as isize, col_axis as isize],
                    true,
                    0,
                    BoundaryMode::Constant,
                    -1.0,
                )
                .unwrap();
                let mut expected_shape = shape.clone();
                if angle != 180.0 {
                    expected_shape.swap(row_axis, col_axis);
                }
                assert_eq!(rotated.shape, expected_shape);
                for (flat, &value) in rotated.data.iter().enumerate() {
                    let output = coordinates(flat, &rotated.shape);
                    let mut source = output.clone();
                    if angle == 90.0 {
                        source[row_axis] = output[col_axis];
                        source[col_axis] = shape[col_axis] - 1 - output[row_axis];
                    } else if angle == 180.0 {
                        source[row_axis] = shape[row_axis] - 1 - output[row_axis];
                        source[col_axis] = shape[col_axis] - 1 - output[col_axis];
                    } else {
                        source[row_axis] = shape[row_axis] - 1 - output[col_axis];
                        source[col_axis] = output[row_axis];
                    }
                    assert_eq!(
                        value,
                        input.get(&source),
                        "axes=({row_axis}, {col_axis}), angle={angle}, output={output:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn default_plane_and_signed_axis_normalization() {
    let input = input_for_shape(&[2, 3, 4]);
    let default = rotate(&input, 90.0, true, 0, BoundaryMode::Constant, 0.0).unwrap();
    assert_eq!(default.shape, vec![3, 2, 4]);
    for row in 0..3 {
        for col in 0..2 {
            for channel in 0..4 {
                assert_eq!(
                    default.get(&[row, col, channel]),
                    input.get(&[col, 2 - row, channel])
                );
            }
        }
    }
    let reference =
        rotate_axes(&input, 27.5, &[0, 2], false, 3, BoundaryMode::Reflect, 0.0).unwrap();
    for axes in [[2, 0], [-1, -3], [-3, 2]] {
        let result =
            rotate_axes(&input, 27.5, &axes, false, 3, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(result.shape, reference.shape);
        for (&a, &b) in result.data.iter().zip(&reference.data) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn spline_prefilter_does_not_mix_unselected_axes() {
    let mut input = NdArray::zeros(vec![5, 3, 6, 2]);
    for row in 0..5 {
        for col in 0..6 {
            input.set(&[row, 1, col, 0], (row * row + col * col + 1) as f64);
        }
    }
    for order in 2..=5 {
        let rotated = rotate_axes(
            &input,
            31.0,
            &[2, 0],
            true,
            order,
            BoundaryMode::Reflect,
            0.0,
        )
        .unwrap();
        let mut active_sum = 0.0;
        for (flat, &value) in rotated.data.iter().enumerate() {
            let index = coordinates(flat, &rotated.shape);
            if index[1] == 1 && index[3] == 0 {
                active_sum += value.abs();
            } else {
                assert_eq!(value, 0.0, "order={order}, index={index:?}");
            }
        }
        assert!(active_sum > 1.0);
    }
}

#[test]
fn rotation_rejects_invalid_planes_and_array_metadata() {
    let input = input_for_shape(&[2, 3, 4]);
    for axes in [
        vec![],
        vec![0],
        vec![0, 1, 2],
        vec![0, 0],
        vec![0, -3],
        vec![0, 3],
        vec![-4, 1],
        vec![isize::MIN, 1],
    ] {
        assert!(
            rotate_axes(&input, 30.0, &axes, true, 3, BoundaryMode::Constant, 0.0).is_err(),
            "axes={axes:?}"
        );
    }
    for angle in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(rotate(&input, angle, true, 1, BoundaryMode::Nearest, 0.0).is_err());
    }
    assert!(rotate(&input, 30.0, true, 6, BoundaryMode::Nearest, 0.0).is_err());
    assert!(
        rotate(
            &input_for_shape(&[4]),
            0.0,
            false,
            0,
            BoundaryMode::Nearest,
            0.0
        )
        .is_err()
    );
    let mut bad_strides = input.clone();
    bad_strides.strides[0] = usize::MAX;
    assert!(rotate(&bad_strides, 0.0, false, 0, BoundaryMode::Nearest, 0.0).is_err());
    let overflow = NdArray {
        data: vec![],
        shape: vec![usize::MAX, 2],
        strides: vec![2, 1],
    };
    assert!(rotate(&overflow, 45.0, true, 0, BoundaryMode::Nearest, 0.0).is_err());
    let mismatch = NdArray {
        data: vec![],
        shape: vec![2, 2],
        strides: vec![2, 1],
    };
    assert!(rotate(&mismatch, 0.0, false, 0, BoundaryMode::Nearest, 0.0).is_err());
}

#[test]
fn rotation_handles_empty_batches_without_sampling_empty_planes() {
    let input = NdArray::new(vec![], vec![2, 0, 3]).unwrap();
    let result = rotate_axes(&input, 90.0, &[0, 2], true, 3, BoundaryMode::Nearest, 0.0).unwrap();
    assert_eq!(result.shape, vec![3, 0, 2]);
    assert!(result.data.is_empty());
    let empty_plane = NdArray::new(vec![], vec![0, 3]).unwrap();
    assert_eq!(
        rotate(&empty_plane, 0.0, false, 0, BoundaryMode::Nearest, 0.0).unwrap_err(),
        NdimageError::EmptyInput
    );
}

#[derive(Debug)]
struct Case {
    shape: Vec<usize>,
    axes: [isize; 2],
    angle: f64,
    reshape: bool,
    order: usize,
    mode: BoundaryMode,
}

#[test]
fn nd_rotation_matches_live_scipy_across_planes_orders_and_boundaries() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON")
        .or_else(|_| std::env::var("SCIPY_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let probe = Command::new(&python)
        .args([
            "-c",
            "import scipy, numpy, scipy.ndimage; print(scipy.__version__, numpy.__version__)",
        ])
        .output();
    let versions = match probe {
        Ok(output) if output.status.success() => {
            String::from_utf8(output.stdout).expect("SciPy version probe UTF-8")
        }
        unavailable => {
            assert!(
                !required,
                "required SciPy rotation oracle is unavailable: {unavailable:?}"
            );
            eprintln!("SKIP N-D rotation: {unavailable:?}; compared cases = 0");
            return;
        }
    };
    eprintln!(
        "N-D rotation oracle: {python}, SciPy/NumPy {}",
        versions.trim()
    );
    if required {
        assert_eq!(
            versions.trim(),
            "1.17.1 2.4.3",
            "required SciPy rotation oracle must use the repository pins"
        );
    }

    let scenarios: [(&[usize], [isize; 2], f64); 12] = [
        (&[4, 5, 3], [0, 1], 31.0),
        (&[3, 4, 5], [2, 0], -27.5),
        (&[2, 5, 6, 3], [-3, -1], 40.25),
        (&[2, 3, 5, 4], [0, 2], 90.0),
        (&[1, 4, 3], [0, 2], 37.0),
        (&[3, 5, 4], [1, 2], 180.0),
        (&[2, 3, 4], [0, 2], 1.0e12),
        (&[2, 3, 4], [0, 2], 1.0e15),
        (&[3, 4, 2], [0, 1], 90.000_000_000_5),
        (&[2, 3, 3], [1, 2], 0.0),
        (&[1, 3, 4], [0, 2], 0.0),
        (&[2, 3, 6], [0, 2], 90.0),
    ];
    let mut cases = Vec::new();
    let mut payload = String::new();
    for (shape, axes, angle) in scenarios {
        for reshape in [false, true] {
            for order in 0..=5 {
                for (mode, name) in [
                    (BoundaryMode::Constant, "constant"),
                    (BoundaryMode::Nearest, "nearest"),
                    (BoundaryMode::Reflect, "reflect"),
                    (BoundaryMode::Mirror, "mirror"),
                    (BoundaryMode::Wrap, "wrap"),
                ] {
                    let shape_text = shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(
                        payload,
                        "{}|{shape_text}|{},{}|{angle:.17e}|{}|{order}|{name}",
                        cases.len(),
                        axes[0],
                        axes[1],
                        usize::from(reshape)
                    )
                    .unwrap();
                    cases.push(Case {
                        shape: shape.to_vec(),
                        axes,
                        angle,
                        reshape,
                        order,
                        mode,
                    });
                }
            }
        }
    }
    let script = r#"
import sys
import numpy as np
from scipy.ndimage import rotate
for line in sys.argv[1].splitlines():
    ident, shape, axes, angle, reshape, order, mode = line.split('|')
    shape = tuple(map(int, shape.split(',')))
    axes = tuple(map(int, axes.split(',')))
    data = np.array([(i % 17)*0.25 - 2.0 + ((i//7) % 5)*0.125
                     for i in range(int(np.prod(shape)))], dtype=np.float64).reshape(shape)
    result = rotate(data, float(angle), axes=axes, reshape=bool(int(reshape)),
                    order=int(order), mode=mode, cval=-0.75, prefilter=True)
    print(ident + '|' + ','.join(map(str, result.shape)) + '|' +
          ','.join(format(float(x), '.17e') for x in result.ravel()))
"#;
    let output = Command::new(&python)
        .args(["-c", script, &payload])
        .output()
        .expect("launch SciPy rotation oracle");
    assert!(
        output.status.success(),
        "SciPy rotation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("rotation oracle UTF-8");
    let mut compared = 0;
    for (index, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line.split('|').collect();
        assert_eq!(fields.len(), 3, "invalid oracle row: {line}");
        assert_eq!(fields[0].parse::<usize>().unwrap(), index);
        let case = &cases[index];
        let expected_shape: Vec<usize> = fields[1]
            .split(',')
            .map(|value| value.parse().unwrap())
            .collect();
        let expected: Vec<f64> = fields[2]
            .split(',')
            .filter(|value| !value.is_empty())
            .map(|value| value.parse().unwrap())
            .collect();
        let input = input_for_shape(&case.shape);
        let result = rotate_axes(
            &input,
            case.angle,
            &case.axes,
            case.reshape,
            case.order,
            case.mode,
            -0.75,
        )
        .unwrap_or_else(|error| panic!("{case:?}: {error}"));
        assert_eq!(result.shape, expected_shape, "{case:?}");
        assert_eq!(result.data.len(), expected.len(), "{case:?}");
        for (pixel, (&actual, &reference)) in result.data.iter().zip(&expected).enumerate() {
            assert!(
                actual.is_finite()
                    && reference.is_finite()
                    && (actual - reference).abs() <= 5e-10 * (1.0 + reference.abs()),
                "{case:?}, pixel {pixel}: {actual:.17e} != SciPy {reference:.17e}"
            );
        }
        compared += 1;
    }
    assert_eq!(compared, cases.len(), "the oracle must compare every case");
    assert_eq!(compared, 720);
    eprintln!("N-D rotation: compared {compared} cases against live SciPy");
}
