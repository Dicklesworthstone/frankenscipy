use fsci_signal::filtfilt_gust;
use std::{fmt::Write, process::Command};

#[test]
fn short_record_uses_minimum_norm_initial_states() {
    // SciPy 1.17.1, mixed FIR/IIR orders: three observations and six unknown states.
    let actual = filtfilt_gust(
        &[0.2, 0.35, 0.1, 0.05],
        &[1.0, -0.4, 0.12],
        &[2.0, -1.0, 3.0],
        None,
    )
    .unwrap();
    let expected = [0.06987954008664604, 0.15981776821024546, 0.3536527284506676];
    for (value, reference) in actual.iter().zip(expected) {
        assert!((value - reference).abs() < 1e-13, "{actual:?}");
    }
}

#[test]
fn scalar_gain_and_filter_validation() {
    assert_eq!(
        filtfilt_gust(&[6.0], &[2.0], &[1.0, -2.0], None).unwrap(),
        [9.0, -18.0]
    );
    for (b, a, x, irlen) in [
        (&[][..], &[1.0][..], &[1.0][..], None),
        (&[1.0], &[0.0], &[1.0], None),
        (&[1.0], &[1.0], &[f64::NAN], None),
        (&[1.0], &[1.0], &[], None),
        (&[1.0], &[1.0], &[1.0], Some(0)),
    ] {
        assert!(filtfilt_gust(b, a, x, irlen).is_err());
    }
}

#[test]
fn full_and_truncated_records_match_live_scipy() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut script = String::from(
        "import sys\ntry:\n import scipy, numpy\n from scipy.signal import filtfilt\nexcept ImportError:\n sys.exit(77)\n",
    );
    if required {
        script.push_str("assert (scipy.__version__, numpy.__version__) == ('1.17.1','2.4.3')\n");
    }
    let mut outputs = Vec::new();
    for (b, a) in [
        (&[0.1, 0.1][..], &[1.0, -0.8][..]),
        (&[0.25, 0.5, 0.25], &[1.0]),
        (&[0.27, 0.54, 0.27], &[4.0, -4.57, 1.65]),
        (&[0.5, -0.5], &[1.0, -1.0]), // cancelled pole, rank-deficient system
    ] {
        for n in [1, 3, 10, 79] {
            let x: Vec<_> = (0..n).map(|i| f64::from((i * 31 + 3) % 17 - 8)).collect();
            for irlen in [None, Some(3)] {
                let limit = irlen.map_or_else(|| "None".into(), |m| m.to_string());
                writeln!(
                    script,
                    "print(*filtfilt({b:?},{a:?},{x:?},method='gust',irlen={limit}))"
                )
                .unwrap();
                outputs.push(filtfilt_gust(b, a, &x, irlen).unwrap());
            }
        }
    }
    let result = Command::new(&python).args(["-c", &script]).output();
    if matches!(&result, Err(e) if e.kind() == std::io::ErrorKind::NotFound)
        || matches!(&result, Ok(o) if o.status.code() == Some(77))
    {
        assert!(!required, "required SciPy unavailable via {python}");
        return;
    }
    let result = result.unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    let rows: Vec<_> = stdout.lines().collect();
    assert_eq!(rows.len(), 32);
    assert_eq!(outputs.len(), rows.len());
    for (case, (values, row)) in outputs.iter().zip(rows).enumerate() {
        let references: Vec<f64> = row.split_whitespace().map(|v| v.parse().unwrap()).collect();
        assert_eq!(values.len(), references.len());
        for (i, (value, reference)) in values.iter().zip(references).enumerate() {
            assert!(
                (value - reference).abs() < 5e-12,
                "case {case}, sample {i}: {value} vs {reference}"
            );
        }
    }
}
