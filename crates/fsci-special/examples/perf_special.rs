#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;
use fsci_special::{SpecialTensor, ellipeinc, ellipkinc, erf, erfc};
use std::error::Error;
use std::f64::consts::PI;
use std::io::{Error as IoError, ErrorKind};

type GoldenResult = Result<(), Box<dyn Error>>;

const ERROR_INPUTS: &[f64] = &[
    f64::NEG_INFINITY,
    -3.0,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    3.0,
    f64::INFINITY,
    f64::NAN,
];

fn main() -> GoldenResult {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "perf_special".to_string());
    match args.next().as_deref() {
        Some("golden-error") => print_error_golden(),
        Some("golden-elliptic") => print_elliptic_golden(),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("usage: {program} <golden-error|golden-elliptic>"),
        )
        .into()),
    }
}

fn print_error_golden() -> GoldenResult {
    for &x in ERROR_INPUTS {
        print_scalar(
            "erf",
            x,
            erf(&SpecialTensor::RealScalar(x), RuntimeMode::Strict),
        )?;
        print_scalar(
            "erfc",
            x,
            erfc(&SpecialTensor::RealScalar(x), RuntimeMode::Strict),
        )?;
    }

    let vector_inputs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    print_vector(
        "erf_vec",
        erf(
            &SpecialTensor::RealVec(vector_inputs.to_vec()),
            RuntimeMode::Strict,
        ),
    )?;
    print_vector(
        "erfc_vec",
        erfc(
            &SpecialTensor::RealVec(vector_inputs.to_vec()),
            RuntimeMode::Strict,
        ),
    )
}

fn print_elliptic_golden() -> GoldenResult {
    let scalar_cases = [(PI / 6.0, 0.0), (PI / 4.0, 0.5), (PI / 3.0, 0.9)];
    for (phi, m) in scalar_cases {
        print_binary_scalar(
            "ellipkinc",
            phi,
            m,
            ellipkinc(
                &SpecialTensor::RealScalar(phi),
                &SpecialTensor::RealScalar(m),
                RuntimeMode::Strict,
            ),
        )?;
        print_binary_scalar(
            "ellipeinc",
            phi,
            m,
            ellipeinc(
                &SpecialTensor::RealScalar(phi),
                &SpecialTensor::RealScalar(m),
                RuntimeMode::Strict,
            ),
        )?;
    }

    print_vector(
        "ellipkinc_broadcast_m",
        ellipkinc(
            &SpecialTensor::RealScalar(PI / 3.0),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        ),
    )?;
    print_vector(
        "ellipeinc_pairwise",
        ellipeinc(
            &SpecialTensor::RealVec(vec![PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.1]),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        ),
    )
}

fn print_scalar(
    function: &str,
    input: f64,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealScalar(value) => {
            println!(
                "{function} input_bits={:016x} output_bits={:016x} output={value:.17e}",
                input.to_bits(),
                value.to_bits(),
            );
            Ok(())
        }
        other => Err(unexpected_tensor("RealScalar", other)),
    }
}

fn print_binary_scalar(
    function: &str,
    left: f64,
    right: f64,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealScalar(value) => {
            println!(
                "{function} left_bits={:016x} right_bits={:016x} output_bits={:016x} output={value:.17e}",
                left.to_bits(),
                right.to_bits(),
                value.to_bits(),
            );
            Ok(())
        }
        other => Err(unexpected_tensor("RealScalar", other)),
    }
}

fn print_vector(
    function: &str,
    result: Result<SpecialTensor, fsci_special::SpecialError>,
) -> GoldenResult {
    match result? {
        SpecialTensor::RealVec(values) => {
            print!("{function}");
            for value in values {
                print!(" {:016x}", value.to_bits());
            }
            println!();
            Ok(())
        }
        other => Err(unexpected_tensor("RealVec", other)),
    }
}

fn unexpected_tensor(expected: &str, actual: SpecialTensor) -> Box<dyn Error> {
    IoError::new(
        ErrorKind::InvalidData,
        format!("expected {expected}, got {actual:?}"),
    )
    .into()
}
