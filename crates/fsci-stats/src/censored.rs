#![forbid(unsafe_code)]

//! Censored data representation for survival analysis and distribution fitting.
//!
//! Matches `scipy.stats.CensoredData`.

/// Represents censored data for survival analysis and distribution fitting.
///
/// Observations can be:
/// - Uncensored: exact event time observed.
/// - Right-censored: event occurred after the recorded time.
/// - Left-censored: event occurred before the recorded time.
/// - Interval-censored: event occurred within `[low, high]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CensoredData {
    /// Exact observed event times.
    pub uncensored: Vec<f64>,
    /// Left-censored observations (event happened before this time).
    pub left: Vec<f64>,
    /// Right-censored observations (event happened after this time).
    pub right: Vec<f64>,
    /// Interval-censored observations: pairs of (low, high).
    pub interval: Vec<(f64, f64)>,
}

impl CensoredData {
    /// Create a new CensoredData instance with specified components.
    #[must_use]
    pub fn new(
        uncensored: Vec<f64>,
        left: Vec<f64>,
        right: Vec<f64>,
        interval: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            uncensored,
            left,
            right,
            interval,
        }
    }

    /// Construct from observations and a boolean mask where `true` indicates right-censoring.
    pub fn right_censored(x: &[f64], censored: &[bool]) -> Result<Self, String> {
        if x.len() != censored.len() {
            return Err(format!(
                "x and censored must have the same length (got {} and {})",
                x.len(),
                censored.len()
            ));
        }
        let mut uncensored = Vec::new();
        let mut right = Vec::new();
        for (&val, &is_cens) in x.iter().zip(censored.iter()) {
            if is_cens {
                right.push(val);
            } else {
                uncensored.push(val);
            }
        }
        Ok(Self {
            uncensored,
            left: Vec::new(),
            right,
            interval: Vec::new(),
        })
    }

    /// Construct from observations and a boolean mask where `true` indicates left-censoring.
    pub fn left_censored(x: &[f64], censored: &[bool]) -> Result<Self, String> {
        if x.len() != censored.len() {
            return Err(format!(
                "x and censored must have the same length (got {} and {})",
                x.len(),
                censored.len()
            ));
        }
        let mut uncensored = Vec::new();
        let mut left = Vec::new();
        for (&val, &is_cens) in x.iter().zip(censored.iter()) {
            if is_cens {
                left.push(val);
            } else {
                uncensored.push(val);
            }
        }
        Ok(Self {
            uncensored,
            left,
            right: Vec::new(),
            interval: Vec::new(),
        })
    }

    /// Construct from lower and upper bounds of interval-censored data.
    pub fn interval_censored(low: &[f64], high: &[f64]) -> Result<Self, String> {
        if low.len() != high.len() {
            return Err(format!(
                "low and high must have the same length (got {} and {})",
                low.len(),
                high.len()
            ));
        }
        let mut interval = Vec::with_capacity(low.len());
        for (i, (&lo, &hi)) in low.iter().zip(high.iter()).enumerate() {
            if lo > hi {
                return Err(format!("interval bound low[{i}]={lo} > high[{i}]={hi}"));
            }
            interval.push((lo, hi));
        }
        Ok(Self {
            uncensored: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            interval,
        })
    }

    /// Total number of observations (both uncensored and censored).
    #[must_use]
    pub fn len(&self) -> usize {
        self.uncensored.len() + self.left.len() + self.right.len() + self.interval.len()
    }

    /// Whether there are zero observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of censored observations.
    #[must_use]
    pub fn num_censored(&self) -> usize {
        self.left.len() + self.right.len() + self.interval.len()
    }
}
