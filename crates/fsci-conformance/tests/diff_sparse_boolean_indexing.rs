#![forbid(unsafe_code)]
//! Cover CsrMatrix boolean indexing methods.
//!
//! Resolves [frankenscipy-zlkqh]. Verifies:
//!   * boolean_row_index keeps rows where mask=true (and only those)
//!   * boolean_col_index keeps cols where mask=true (and only those)
//!   * boolean_index combines row+col masks
//!   * boolean_mask_values returns the values at mask=true cells
//!   * Each value at the resulting positions equals the original
//!     dense value at the selected (row, col)
//!   * Mask-length mismatches error

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{CooMatrix, CscMatrix, FormatConvertible, Shape2D};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
/// One ledger arm per check, each one fixed case recorded under its own name. The `*_values` and
/// `all_true_*` arms compare against the source matrix's dense entries (flattened through
/// `slices`, which rejects a length mismatch or a non-finite element); the `*_errors` arms are
/// refusals; the shape arms are structural.
const ARMS: [&str; 12] = [
    "row_index_shape",
    "row_index_values",
    "col_index_shape",
    "col_index_values",
    "full_index_shape",
    "full_index_values",
    "mask_values_correct",
    "row_mask_length_mismatch_errors",
    "col_mask_length_mismatch_errors",
    "mask_values_shape_mismatch_errors",
    "all_true_row_returns_original",
    "all_false_row_zero_rows",
];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create bool_index diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn csc_to_dense(m: &CscMatrix) -> Vec<Vec<f64>> {
    let s = m.shape();
    let mut d = vec![vec![0.0_f64; s.cols]; s.rows];
    let indptr = m.indptr();
    let indices = m.indices();
    let data = m.data();
    for j in 0..s.cols {
        for idx in indptr[j]..indptr[j + 1] {
            d[indices[idx]][j] = data[idx];
        }
    }
    d
}

#[test]
fn diff_sparse_boolean_indexing() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_sparse_boolean_indexing", &ARMS);
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // 3×4 test matrix with diverse values
    let trips: Vec<(usize, usize, f64)> = vec![
        (0, 0, 1.0),
        (0, 2, 2.0),
        (1, 1, 3.0),
        (1, 3, 4.0),
        (2, 0, 5.0),
        (2, 2, 6.0),
        (2, 3, 7.0),
    ];
    let data: Vec<f64> = trips.iter().map(|t| t.2).collect();
    let rs: Vec<usize> = trips.iter().map(|t| t.0).collect();
    let cs: Vec<usize> = trips.iter().map(|t| t.1).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(3, 4), data, rs, cs, true).expect("coo");
    let csc = coo.to_csc().expect("csc");
    let dense = csc_to_dense(&csc);

    // === boolean_row_index: keep rows 0, 2 ===
    {
        let mask = [true, false, true];
        let result = csc.boolean_row_index(&mask).expect("row_index");
        let result_dense = csc_to_dense(&result);
        let shape_ok = result.shape().rows == 2 && result.shape().cols == 4;
        ledger.compared("row_index_shape", "row_index_shape", shape_ok);
        check(
            "row_index_shape",
            shape_ok,
            format!("shape={:?}", result.shape()),
        );
        // Rows in result should equal rows 0 and 2 of original
        let expected = vec![dense[0].clone(), dense[2].clone()];
        let (expected_flat, got_flat) = (expected.concat(), result_dense.concat());
        if ledger
            .slices(
                "row_index_values",
                "row_index_values",
                Some(expected_flat.as_slice()),
                Some(got_flat.as_slice()),
            )
            .is_some()
        {
            ledger.compared(
                "row_index_values",
                "row_index_values",
                result_dense == expected,
            );
        }
        check(
            "row_index_values",
            result_dense == expected,
            format!("got={result_dense:?} expected={expected:?}"),
        );
    }

    // === boolean_col_index: keep cols 1, 2 ===
    {
        let mask = [false, true, true, false];
        let result = csc.boolean_col_index(&mask).expect("col_index");
        let result_dense = csc_to_dense(&result);
        let shape_ok = result.shape().rows == 3 && result.shape().cols == 2;
        ledger.compared("col_index_shape", "col_index_shape", shape_ok);
        check(
            "col_index_shape",
            shape_ok,
            format!("shape={:?}", result.shape()),
        );
        let expected: Vec<Vec<f64>> = dense.iter().map(|row| vec![row[1], row[2]]).collect();
        let (expected_flat, got_flat) = (expected.concat(), result_dense.concat());
        if ledger
            .slices(
                "col_index_values",
                "col_index_values",
                Some(expected_flat.as_slice()),
                Some(got_flat.as_slice()),
            )
            .is_some()
        {
            ledger.compared(
                "col_index_values",
                "col_index_values",
                result_dense == expected,
            );
        }
        check(
            "col_index_values",
            result_dense == expected,
            format!("got={result_dense:?} expected={expected:?}"),
        );
    }

    // === boolean_index: combined row+col mask ===
    {
        let row_mask = [true, false, true];
        let col_mask = [true, false, true, false];
        let result = csc.boolean_index(&row_mask, &col_mask).expect("full_index");
        let result_dense = csc_to_dense(&result);
        let shape_ok = result.shape().rows == 2 && result.shape().cols == 2;
        ledger.compared("full_index_shape", "full_index_shape", shape_ok);
        check(
            "full_index_shape",
            shape_ok,
            format!("shape={:?}", result.shape()),
        );
        // Expected: take rows [0, 2] then cols [0, 2]
        let expected = vec![
            vec![dense[0][0], dense[0][2]],
            vec![dense[2][0], dense[2][2]],
        ];
        let (expected_flat, got_flat) = (expected.concat(), result_dense.concat());
        if ledger
            .slices(
                "full_index_values",
                "full_index_values",
                Some(expected_flat.as_slice()),
                Some(got_flat.as_slice()),
            )
            .is_some()
        {
            ledger.compared(
                "full_index_values",
                "full_index_values",
                result_dense == expected,
            );
        }
        check(
            "full_index_values",
            result_dense == expected,
            format!("got={result_dense:?} expected={expected:?}"),
        );
    }

    // === boolean_mask_values: 2D mask returns values at true positions ===
    {
        // Select positions (0,0), (1,3), (2,2) → values 1.0, 4.0, 6.0
        let mask = vec![
            vec![true, false, false, false],
            vec![false, false, false, true],
            vec![false, false, true, false],
        ];
        let result = csc.boolean_mask_values(&mask).expect("mask_values");
        let expected = vec![1.0_f64, 4.0, 6.0];
        if ledger
            .slices(
                "mask_values_correct",
                "mask_values_correct",
                Some(expected.as_slice()),
                Some(result.as_slice()),
            )
            .is_some()
        {
            ledger.compared(
                "mask_values_correct",
                "mask_values_correct",
                result == expected,
            );
        }
        check(
            "mask_values_correct",
            result == expected,
            format!("got={result:?} expected={expected:?}"),
        );
    }

    // === Error: row_mask length mismatch ===
    {
        let bad_mask = [true, false]; // shorter than 3 rows
        let r = csc.boolean_row_index(&bad_mask);
        // The designed answer is a refusal (SciPy raises IndexError on a wrong-length mask).
        ledger.expected_raise(
            "row_mask_length_mismatch_errors",
            "row_mask_length_mismatch_errors",
            r.is_err(),
        );
        check(
            "row_mask_length_mismatch_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // === Error: col_mask length mismatch ===
    {
        let bad_mask = [true, false, true]; // shorter than 4 cols
        let r = csc.boolean_col_index(&bad_mask);
        ledger.expected_raise(
            "col_mask_length_mismatch_errors",
            "col_mask_length_mismatch_errors",
            r.is_err(),
        );
        check(
            "col_mask_length_mismatch_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // === Error: 2D mask shape mismatch ===
    {
        let bad_mask = vec![vec![true, false], vec![false, true]]; // 2×2 not 3×4
        let r = csc.boolean_mask_values(&bad_mask);
        ledger.expected_raise(
            "mask_values_shape_mismatch_errors",
            "mask_values_shape_mismatch_errors",
            r.is_err(),
        );
        check(
            "mask_values_shape_mismatch_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // === Edge: all-true mask returns original matrix ===
    {
        let all_true = [true; 3];
        let r = csc.boolean_row_index(&all_true).expect("all-true row");
        let r_dense = csc_to_dense(&r);
        let same = r_dense == dense && r.shape() == csc.shape();
        let (dense_flat, r_flat) = (dense.concat(), r_dense.concat());
        if ledger
            .slices(
                "all_true_row_returns_original",
                "all_true_row_returns_original",
                Some(dense_flat.as_slice()),
                Some(r_flat.as_slice()),
            )
            .is_some()
        {
            ledger.compared(
                "all_true_row_returns_original",
                "all_true_row_returns_original",
                same,
            );
        }
        check("all_true_row_returns_original", same, String::new());
    }

    // === Edge: all-false mask returns 0-row matrix ===
    {
        let all_false = [false; 3];
        let r = csc.boolean_row_index(&all_false).expect("all-false row");
        let shape_ok = r.shape().rows == 0 && r.shape().cols == 4;
        ledger.compared(
            "all_false_row_zero_rows",
            "all_false_row_zero_rows",
            shape_ok,
        );
        check(
            "all_false_row_zero_rows",
            shape_ok,
            format!("shape={:?}", r.shape()),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_boolean_indexing".into(),
        category: "fsci_sparse::CsrMatrix::boolean_* coverage".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("bool_index mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "boolean indexing coverage failed: {} cases",
        diffs.len()
    );
    // Every arm is one fixed case, so each must have compared it.
    ledger.finish(1);
}
