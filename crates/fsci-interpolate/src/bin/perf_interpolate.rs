use std::{fmt::Write as _, fs, process::ExitCode};

use fsci_interpolate::{
    GriddataMethod, RectBivariateSpline, RegularGridInterpolator, RegularGridMethod, griddata,
};

fn grid_1d(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn query_1d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.001 + 0.998 * t
        })
        .collect()
}

fn rect_grid(side: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let x = grid_1d(side);
    let y = grid_1d(side);
    let z = x
        .iter()
        .map(|&xi| {
            y.iter()
                .map(|&yi| (xi * 6.0).sin() * (yi * 4.0).cos())
                .collect()
        })
        .collect();
    (x, y, z)
}

fn points_2d(side: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut points = Vec::with_capacity(side * side);
    let mut values = Vec::with_capacity(side * side);
    for iy in 0..side {
        for ix in 0..side {
            let x = (ix as f64 + 0.3 * (iy % 3) as f64) / side as f64;
            let y = (iy as f64 + 0.2 * (ix % 5) as f64) / side as f64;
            points.push(vec![x, y]);
            values.push((x * 5.0).sin() + (y * 3.0).cos());
        }
    }
    (points, values)
}

fn queries_2d(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = ((i * 37) % 997) as f64 / 997.0;
            let y = ((i * 53 + 17) % 991) as f64 / 991.0;
            vec![x, y]
        })
        .collect()
}

fn regular_grid_values(points: &[Vec<f64>]) -> Vec<f64> {
    let nx = points[0].len();
    let ny = points[1].len();
    let nz = points[2].len();
    let mut values = Vec::with_capacity(nx * ny * nz);
    for &x in &points[0] {
        for &y in &points[1] {
            for &z in &points[2] {
                values.push((x * 5.0).sin() + (y * 3.0).cos() + z * z);
            }
        }
    }
    values
}

fn regular_grid_queries(n: usize) -> Vec<Vec<f64>> {
    queries_2d(n)
        .into_iter()
        .map(|q| vec![q[0], q[1], (q[0] * 0.7 + q[1] * 0.3).fract()])
        .collect()
}

fn regular_grid_case() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let points = vec![grid_1d(32), grid_1d(32), grid_1d(16)];
    let values = regular_grid_values(&points);
    let queries = regular_grid_queries(4096);
    (points, values, queries)
}

fn write_or_print(path: Option<&str>, contents: &str) -> Result<(), String> {
    if let Some(path) = path {
        fs::write(path, contents).map_err(|err| format!("failed to write {path}: {err}"))
    } else {
        print!("{contents}");
        Ok(())
    }
}

fn print_rect_eval_grid_golden() -> Result<(), String> {
    let (x, y, z) = rect_grid(32);
    let spline = RectBivariateSpline::new(&x, &y, &z, 3, 3)
        .map_err(|err| format!("failed to construct spline: {err}"))?;
    let xi = query_1d(64);
    let yi = query_1d(64);
    let values = spline.eval_grid(&xi, &yi);

    println!("fsci-interpolate rect_eval_grid golden v1");
    println!("input_grid=32 query_grid=64 kx=3 ky=3 order=x-major,y-inner");
    for (i, row) in values.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            println!("{i:02} {j:02} {:016x}", value.to_bits());
        }
    }
    Ok(())
}

fn print_regular_grid_linear_golden(path: Option<&str>) -> Result<(), String> {
    let (points, values, queries) = regular_grid_case();
    let interpolator =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
            .map_err(|err| format!("failed to construct regular-grid interpolator: {err}"))?;
    let values = interpolator
        .eval_many(&queries)
        .map_err(|err| format!("failed to evaluate regular-grid linear: {err}"))?;

    let mut out = String::new();
    writeln!(&mut out, "fsci-interpolate regular_grid_linear golden v1")
        .map_err(|err| err.to_string())?;
    writeln!(
        &mut out,
        "grid=32x32x16 queries=4096 method=linear order=query-input bits=f64"
    )
    .map_err(|err| err.to_string())?;
    for (i, value) in values.iter().enumerate() {
        writeln!(&mut out, "{i:04} {:016x}", value.to_bits()).map_err(|err| err.to_string())?;
    }

    write_or_print(path, &out)
}

fn print_regular_grid_nearest_golden(path: Option<&str>) -> Result<(), String> {
    let (points, values, queries) = regular_grid_case();
    let interpolator =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
            .map_err(|err| format!("failed to construct regular-grid interpolator: {err}"))?;
    let values = interpolator
        .eval_many(&queries)
        .map_err(|err| format!("failed to evaluate regular-grid nearest: {err}"))?;

    let mut out = String::new();
    writeln!(&mut out, "fsci-interpolate regular_grid_nearest golden v1")
        .map_err(|err| err.to_string())?;
    writeln!(
        &mut out,
        "grid=32x32x16 queries=4096 method=nearest order=query-input bits=f64"
    )
    .map_err(|err| err.to_string())?;
    for (i, value) in values.iter().enumerate() {
        writeln!(&mut out, "{i:04} {:016x}", value.to_bits()).map_err(|err| err.to_string())?;
    }

    write_or_print(path, &out)
}

fn print_griddata_linear_golden() -> Result<(), String> {
    let (points, values) = points_2d(24);
    let queries = queries_2d(1024);
    let values = griddata(&points, &values, &queries, GriddataMethod::Linear)
        .map_err(|err| format!("failed to evaluate griddata linear: {err}"))?;

    println!("fsci-interpolate griddata_linear golden v1");
    println!("points=576 queries=1024 method=linear order=query-input");
    for (i, value) in values.iter().enumerate() {
        println!("{i:04} {:016x}", value.to_bits());
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "golden".to_string());
    let output_path = args.next();
    if args.next().is_some() {
        eprintln!(
            "usage: perf_interpolate [golden|griddata-linear|regular-grid-linear|regular-grid-nearest] [output-path]"
        );
        return ExitCode::from(2);
    }
    match mode.as_str() {
        "griddata-linear" => match print_griddata_linear_golden() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        "regular-grid-linear" => match print_regular_grid_linear_golden(output_path.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        "regular-grid-nearest" => match print_regular_grid_nearest_golden(output_path.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        "golden" => match print_rect_eval_grid_golden() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!(
                "usage: perf_interpolate [golden|griddata-linear|regular-grid-linear|regular-grid-nearest] [output-path]"
            );
            ExitCode::from(2)
        }
    }
}
