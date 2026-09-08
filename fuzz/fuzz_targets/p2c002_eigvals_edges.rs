#![no_main]

use arbitrary::Arbitrary;
use fsci_linalg::{DecompOptions, eig, eigh, eigvals, eigvalsh};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 8;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum EdgeF64 {
    Finite(f64),
    Zero,
    NegZero,
    One,
    NegOne,
    Tiny,
    NegTiny,
    PosInf,
    NegInf,
    Nan,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e4, 1.0e4),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
        }
    }

    fn finite(self) -> f64 {
        let value = self.raw();
        if value.is_finite() { value } else { 0.0 }
    }
}

#[derive(Debug, Arbitrary)]
struct EigvalsInput {
    dim: u8,
    extra_cols: u8,
    hardened: bool,
    check_finite: bool,
    ragged: bool,
    symmetric: bool,
    values: Vec<EdgeF64>,
}

fn mode_from_flag(hardened: bool) -> RuntimeMode {
    if hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    }
}

fn build_matrix(input: &EigvalsInput) -> Vec<Vec<f64>> {
    let rows = usize::from(input.dim) % (MAX_DIM + 1);
    let cols = if input.extra_cols.is_multiple_of(5) {
        rows.saturating_add(1).min(MAX_DIM + 1)
    } else {
        rows
    };
    let mut matrix = vec![vec![0.0; cols]; rows];
    let mut cursor = 0;

    for row in &mut matrix {
        for value in row {
            let edge = input.values.get(cursor).copied().unwrap_or(EdgeF64::Zero);
            *value = if input.symmetric {
                edge.finite()
            } else {
                edge.raw()
            };
            cursor += 1;
        }
    }

    if input.symmetric && rows == cols {
        for i in 0..rows {
            for j in 0..i {
                let value = matrix[j][i];
                matrix[i][j] = value;
            }
        }
    }

    if input.ragged && rows > 0 {
        matrix[rows - 1].push(0.0);
    }

    matrix
}

fn has_nonfinite(matrix: &[Vec<f64>]) -> bool {
    matrix.iter().flatten().any(|value| !value.is_finite())
}

fn assert_sorted(values: &[f64]) {
    for pair in values.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "eigvalsh/eigh eigenvalues not sorted: {values:?}"
        );
    }
}

fuzz_target!(|input: EigvalsInput| {
    let mode = mode_from_flag(input.hardened);
    let options = DecompOptions {
        mode,
        check_finite: input.check_finite,
    };
    let matrix = build_matrix(&input);
    let rows = matrix.len();
    let square = matrix.iter().all(|row| row.len() == rows);
    let must_reject_nonfinite = (input.check_finite || input.hardened) && has_nonfinite(&matrix);

    let eigvals_result = eigvals(&matrix, options);
    if must_reject_nonfinite {
        assert!(
            eigvals_result.is_err(),
            "eigvals accepted non-finite input with finite checking enabled"
        );
    }
    if let Ok((real, imag)) = eigvals_result {
        assert_eq!(real.len(), rows, "eigvals real length mismatch");
        assert_eq!(imag.len(), rows, "eigvals imag length mismatch");
        assert!(square, "eigvals returned Ok for non-square/ragged matrix");
    }

    if let Ok(result) = eig(&matrix, options) {
        assert_eq!(
            result.eigenvalues_re.len(),
            rows,
            "eig real length mismatch"
        );
        assert_eq!(
            result.eigenvalues_im.len(),
            rows,
            "eig imag length mismatch"
        );
        assert_eq!(
            result.eigenvectors.len(),
            rows,
            "eig eigenvector row mismatch"
        );
        assert!(square, "eig returned Ok for non-square/ragged matrix");
    }

    if input.symmetric {
        if let Ok(result) = eigh(&matrix, options) {
            assert_eq!(result.eigenvalues.len(), rows, "eigh value length mismatch");
            assert_eq!(result.eigenvectors.len(), rows, "eigh vector row mismatch");
            assert_sorted(&result.eigenvalues);
        }
        if let Ok(values) = eigvalsh(&matrix, options) {
            assert_eq!(values.len(), rows, "eigvalsh length mismatch");
            assert_sorted(&values);
        }
    }
});
