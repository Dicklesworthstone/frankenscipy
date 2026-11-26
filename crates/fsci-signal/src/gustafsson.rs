use crate::{SignalError, lfilter};
use fsci_linalg::{LstsqOptions, lstsq};

/// Filter a real signal forward and backward using Gustafsson's initial states.
///
/// Matches `scipy.signal.filtfilt(b, a, x, method="gust", irlen=...)`.
/// This avoids extending the record. It picks the two filter states by minimizing
/// the difference between forward-backward and backward-forward filtering.
/// The full solve accepts short signals, including records shorter than the filter.
///
/// For long records and a decaying impulse response, `Some(irlen)` limits the
/// initial-state calculation and correction to the first and last `irlen`
/// samples. Choose a length beyond which the impulse response is negligible.
/// `None` uses the entire record. The filter still runs over every sample.
pub fn filtfilt_gust(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    irlen: Option<usize>,
) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() || irlen == Some(0) {
        return Err(SignalError::InvalidParameter {
            detail: "Gustafsson filtering requires a nonempty signal and positive irlen".into(),
        });
    }

    // Run the ordinary zero-state passes first. lfilter also validates the
    // coefficients, leading denominator and samples before we allocate a system.
    let mut forward = lfilter(b, a, x, None)?;
    forward.reverse();
    let mut forward_backward = lfilter(b, a, &forward, None)?;
    forward_backward.reverse();
    let order = a.len().max(b.len()) - 1;
    if order == 0 {
        return Ok(forward_backward);
    }
    let n = x.len();
    let m = irlen.filter(|&len| len <= (n - 1) / 2).unwrap_or(n);
    let truncated = m < n;
    let rows = if truncated { 2 * m } else { m };
    let mut backward = lfilter(b, a, &x.iter().rev().copied().collect::<Vec<_>>(), None)?;
    backward.reverse();
    let backward_forward = lfilter(b, a, &backward, None)?;
    let difference: Vec<_> = (0..rows)
        .map(|row| {
            let i = if truncated && row >= m {
                n - rows + row
            } else {
                row
            };
            backward_forward[i] - forward_backward[i]
        })
        .collect();

    // O maps a delay state to its zero-input output. In transposed direct
    // form II each column is a delayed copy of the first. Let R reverse time
    // and S = filter(R O). Gustafsson's system is [R S - O, R O - S].
    // With truncation its two blocks act on separate ends of the signal.
    let zeros = vec![0.0; m];
    let mut unit_state = vec![0.0; order];
    unit_state[0] = 1.0;
    let response = lfilter(b, a, &zeros, Some(&unit_state))?;
    let mut matrix = vec![vec![0.0; 2 * order]; rows];
    for column in 0..order.min(m) {
        let observed = |i: usize| {
            if i >= column {
                response[i - column]
            } else {
                0.0
            }
        };
        let reversed: Vec<_> = (0..m).rev().map(observed).collect();
        let filtered = lfilter(b, a, &reversed, None)?;
        for i in 0..m {
            matrix[i][column] = filtered[m - 1 - i] - observed(i);
            let row = if truncated { m + i } else { i };
            matrix[row][order + column] = reversed[i] - filtered[i];
        }
    }
    let solution = lstsq(
        &matrix,
        &difference,
        LstsqOptions {
            cond: Some(f64::EPSILON),
            ..Default::default()
        },
    )
    .map_err(|err| {
        SignalError::NumericalFailure(format!("Gustafsson initial-state solve: {err}"))
    })?;
    let (start, end) = solution.x.split_at(order);

    // Apply the two state corrections R S start and R O end directly; a
    // second dense matrix would repeat the work of the filter recurrences.
    let mut start_response = lfilter(b, a, &zeros, Some(start))?;
    start_response.reverse();
    let left = lfilter(b, a, &start_response, None)?;
    let right = lfilter(b, a, &zeros, Some(end))?;
    for i in 0..m {
        forward_backward[i] += left[m - 1 - i];
        forward_backward[n - m + i] += right[m - 1 - i];
    }
    Ok(forward_backward)
}
