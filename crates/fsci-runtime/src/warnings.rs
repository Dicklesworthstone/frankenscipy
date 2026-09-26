#![forbid(unsafe_code)]

//! SciPy's warnings, the non-fatal conditions SciPy reports through `warnings.warn`.
//!
//! A routine that SciPy would warn from calls [`warn`] under the same condition and carries
//! on with the value SciPy returns. [`catch_warnings`] is the analogue of
//! `warnings.catch_warnings(record=True)`: it runs a closure and hands back every warning
//! the closure raised on the calling thread. A warning raised outside any
//! `catch_warnings` scope is written to stderr the first time its category and message
//! are seen in the process, as Python's default filter prints the first occurrence and
//! suppresses repeats.
//!
//! Capture is per thread. The emitting routines decide whether to warn on the thread that
//! called them, before any work they fan out, so a capture on the caller sees them.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Mutex, OnceLock};

/// A SciPy warning class. Each variant is named after the class SciPy raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningCategory {
    /// `scipy.stats.DegenerateDataWarning`: the data make the result undefined, as when a
    /// BCa bootstrap interval cannot be formed.
    DegenerateDataWarning,
    /// `scipy.stats.ConstantInputWarning` (a `DegenerateDataWarning`): a correlation of a
    /// constant input, whose coefficient is NaN.
    ConstantInputWarning,
    /// `scipy.stats.NearConstantInputWarning` (a `DegenerateDataWarning`): an input so
    /// close to constant that the computed coefficient may be inaccurate.
    NearConstantInputWarning,
    /// `scipy.optimize.OptimizeWarning`.
    OptimizeWarning,
    /// `scipy.integrate.IntegrationWarning`: QUADPACK stopped short of the requested
    /// accuracy.
    IntegrationWarning,
    /// `scipy.signal.BadCoefficients`: leading numerator coefficients that are zero to
    /// within 1e-14 after normalisation, which `normalize` strips.
    BadCoefficients,
    /// `scipy.special.SpecialFunctionWarning`: an `sf_error` condition under `errstate`
    /// mode "warn".
    SpecialFunctionWarning,
}

impl WarningCategory {
    /// The SciPy class name, as `warnings` prints it.
    #[must_use]
    pub const fn scipy_name(self) -> &'static str {
        match self {
            Self::DegenerateDataWarning => "DegenerateDataWarning",
            Self::ConstantInputWarning => "ConstantInputWarning",
            Self::NearConstantInputWarning => "NearConstantInputWarning",
            Self::OptimizeWarning => "OptimizeWarning",
            Self::IntegrationWarning => "IntegrationWarning",
            Self::BadCoefficients => "BadCoefficients",
            Self::SpecialFunctionWarning => "SpecialFunctionWarning",
        }
    }

    /// Whether a SciPy `except`/filter on `other` catches this category, following
    /// SciPy's class hierarchy (the two constant-input warnings are `DegenerateDataWarning`s).
    #[must_use]
    pub const fn is_a(self, other: Self) -> bool {
        matches!(
            (self, other),
            (
                Self::ConstantInputWarning | Self::NearConstantInputWarning,
                Self::DegenerateDataWarning
            )
        ) || self as u8 == other as u8
    }
}

impl fmt::Display for WarningCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.scipy_name())
    }
}

/// One raised warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub category: WarningCategory,
    pub message: String,
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.category, self.message)
    }
}

thread_local! {
    /// One buffer per open `catch_warnings` scope, innermost last.
    static CAPTURES: RefCell<Vec<Vec<Warning>>> = const { RefCell::new(Vec::new()) };
}

fn printed_once() -> &'static Mutex<HashSet<(WarningCategory, String)>> {
    static PRINTED: OnceLock<Mutex<HashSet<(WarningCategory, String)>>> = OnceLock::new();
    PRINTED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Raise a warning: record it in the innermost `catch_warnings` scope on this thread, or,
/// with none open, print it to stderr unless this category and message were printed before.
pub fn warn(category: WarningCategory, message: impl Into<String>) {
    let warning = Warning {
        category,
        message: message.into(),
    };
    let unrecorded = CAPTURES.with(|captures| {
        let mut captures = captures.borrow_mut();
        match captures.last_mut() {
            Some(scope) => {
                scope.push(warning);
                None
            }
            None => Some(warning),
        }
    });
    if let Some(warning) = unrecorded {
        let first = match printed_once().lock() {
            Ok(mut seen) => seen.insert((warning.category, warning.message.clone())),
            Err(_) => true,
        };
        if first {
            eprintln!("{warning}");
        }
    }
}

/// Closes a capture scope even when the closure unwinds, so a panic inside one
/// `catch_warnings` cannot leave later warnings on this thread captured into a dead scope.
struct ScopeGuard {
    depth: usize,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CAPTURES.with(|captures| captures.borrow_mut().truncate(self.depth));
    }
}

/// Run `f` and return its value with every warning it raised on this thread, in order, like
/// `warnings.catch_warnings(record=True)` with the "always" filter. Scopes nest: a warning
/// goes to the innermost open scope only.
pub fn catch_warnings<T>(f: impl FnOnce() -> T) -> (T, Vec<Warning>) {
    let depth = CAPTURES.with(|captures| {
        let mut captures = captures.borrow_mut();
        captures.push(Vec::new());
        captures.len() - 1
    });
    let guard = ScopeGuard { depth };
    let value = f();
    let recorded = CAPTURES.with(|captures| {
        captures
            .borrow_mut()
            .get_mut(depth)
            .map(std::mem::take)
            .unwrap_or_default()
    });
    drop(guard);
    (value, recorded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_records_warnings_in_order_and_nothing_else() {
        let ((), recorded) = catch_warnings(|| {
            warn(WarningCategory::ConstantInputWarning, "first");
            warn(WarningCategory::OptimizeWarning, "second");
        });
        assert_eq!(
            recorded,
            vec![
                Warning {
                    category: WarningCategory::ConstantInputWarning,
                    message: "first".into(),
                },
                Warning {
                    category: WarningCategory::OptimizeWarning,
                    message: "second".into(),
                },
            ]
        );
        let (value, recorded) = catch_warnings(|| 7);
        assert_eq!(value, 7);
        assert!(recorded.is_empty());
    }

    #[test]
    fn nested_scopes_capture_into_the_innermost_only() {
        let (inner, outer) = catch_warnings(|| {
            warn(WarningCategory::IntegrationWarning, "outer before");
            let ((), inner) = catch_warnings(|| warn(WarningCategory::BadCoefficients, "inner"));
            warn(WarningCategory::IntegrationWarning, "outer after");
            inner
        });
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].category, WarningCategory::BadCoefficients);
        let outer: Vec<&str> = outer.iter().map(|w| w.message.as_str()).collect();
        assert_eq!(outer, ["outer before", "outer after"]);
    }

    #[test]
    fn a_panicking_scope_is_closed_on_unwind() {
        let unwound = std::panic::catch_unwind(|| {
            catch_warnings(|| {
                warn(WarningCategory::BadCoefficients, "lost");
                std::panic::resume_unwind(Box::new("inside the scope"));
            })
        });
        assert!(unwound.is_err());
        // The dead scope must not swallow this one's warning.
        let ((), recorded) = catch_warnings(|| warn(WarningCategory::BadCoefficients, "kept"));
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].message, "kept");
        CAPTURES.with(|captures| assert!(captures.borrow().is_empty()));
    }

    #[test]
    fn capture_is_per_thread() {
        let ((), recorded) = catch_warnings(|| {
            std::thread::spawn(|| {
                let ((), there) =
                    catch_warnings(|| warn(WarningCategory::DegenerateDataWarning, "there"));
                assert_eq!(there.len(), 1);
            })
            .join()
            .expect("worker");
        });
        assert!(
            recorded.is_empty(),
            "another thread's warning leaked: {recorded:?}"
        );
    }

    #[test]
    fn hierarchy_follows_scipy_classes() {
        use WarningCategory as W;
        assert!(W::ConstantInputWarning.is_a(W::DegenerateDataWarning));
        assert!(W::NearConstantInputWarning.is_a(W::DegenerateDataWarning));
        assert!(W::DegenerateDataWarning.is_a(W::DegenerateDataWarning));
        assert!(!W::DegenerateDataWarning.is_a(W::ConstantInputWarning));
        assert!(!W::OptimizeWarning.is_a(W::IntegrationWarning));
        assert_eq!(W::IntegrationWarning.to_string(), "IntegrationWarning");
    }
}
