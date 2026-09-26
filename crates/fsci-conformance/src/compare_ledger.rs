//! Compared-case ledger for the live-SciPy differential tests (frankenscipy-olv0j.1).
//!
//! Observed defect class: nearly every `tests/diff_*.rs` ends in
//! `diffs.iter().all(|d| d.pass)`, which is true over an empty iterator, and reaches it through
//! silent skips: the Python oracle maps an exception to `None`, the Rust loop does
//! `Err(_) => continue`, and `is_finite()` guards drop a case whose fsci value is NaN. A test whose
//! oracle raised on every case, or whose Rust call always failed, compared nothing and passed.
//! Four such columns shipped under SciPy 1.17.1 (frankenscipy-olv0j.2).
//!
//! A test declares its arms, records exactly one outcome per (arm, case), and ends with
//! [`CompareLedger::finish`]. That panics, with a table, unless every arm compared at least the
//! minimum number of cases, every compared case passed, no case had a finite SciPy value against
//! a failed fsci call, and, under `FSCI_REQUIRE_SCIPY_ORACLE=1`, SciPy produced a value for every
//! case that was not declared to raise.
//!
//! The shape of a test loop: every SciPy value and every fsci call goes through the ledger, with
//! no `continue` that bypasses it, and the counts go into the diff log.
//!
//! ```
//! use fsci_conformance::CompareLedger;
//!
//! // (case id, SciPy's value or None when it raised, fsci's result)
//! let cases = [("a", Some(1.0), Ok(1.0)), ("b", Some(2.0), Ok(2.0 + 1e-15))];
//! let n = cases.len();
//! let mut ledger = CompareLedger::new("diff_example", &["value"]);
//! for (case_id, scipy, fsci) in cases {
//!     let fsci: Result<f64, String> = fsci;
//!     let Some((s, f)) = ledger.pair("value", case_id, scipy, fsci.ok()) else {
//!         continue; // recorded: SciPy gave nothing, fsci failed, or a non-finite outcome
//!     };
//!     ledger.compared("value", case_id, (f - s).abs() <= 1e-12);
//! }
//! let counts = ledger.finish(n); // every case must have been compared
//! assert_eq!(counts["value"].compared_cases, 2);
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;
use serde::de::{self, Deserializer};

/// Set to `1` in CI: a case SciPy produced no value for is then a failure, not a skip.
pub const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// How many problem lines a failing [`CompareLedger::finish`] prints.
const PROBLEMS_SHOWN: usize = 20;

/// Deserializes an oracle scalar that is `null` (SciPy raised or gave nothing), a number, or a
/// non-finite value sent as the string `"nan"`, `"inf"` or `"-inf"` (JSON has no NaN). With
/// `#[serde(default, deserialize_with = "fsci_conformance::compare_ledger::oracle_f64")]` on an
/// `Option<f64>` field, a documented NaN answer (e.g. `nan_policy='propagate'`) reaches
/// [`CompareLedger::pair`] as NaN, a matching refusal, instead of looking like a missing oracle.
/// The Python side:
/// `def fval(v): v = float(v); return v if math.isfinite(v) else ("nan" if math.isnan(v) else ("inf" if v > 0 else "-inf"))`
///
/// # Errors
/// A value that is neither null, a number, nor one of the sentinels.
pub fn oracle_f64<'de, D: Deserializer<'de>>(de: D) -> Result<Option<f64>, D::Error> {
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = Option<f64>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("null, a number, or \"nan\" / \"inf\" / \"-inf\"")
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D2: Deserializer<'de>>(self, de: D2) -> Result<Self::Value, D2::Error> {
            de.deserialize_any(V)
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v))
        }
        #[allow(clippy::cast_precision_loss)] // oracle integers are small counts and ranks
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as f64))
        }
        #[allow(clippy::cast_precision_loss)] // oracle integers are small counts and ranks
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v as f64))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            match v {
                "nan" => Ok(Some(f64::NAN)),
                "inf" => Ok(Some(f64::INFINITY)),
                "-inf" => Ok(Some(f64::NEG_INFINITY)),
                other => Err(E::custom(format!(
                    "expected an oracle number, got {other:?}"
                ))),
            }
        }
    }
    de.deserialize_option(V)
}

/// Per-arm outcome counts, written into each diff log so the compared coverage is readable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ArmCounts {
    /// Cases both sides computed, including matching non-finite values and declared raises.
    pub compared_cases: usize,
    /// Compared cases outside the test's tolerance.
    pub compared_failed: usize,
    /// SciPy gave a definite value; fsci returned an error, nothing, or a non-finite value.
    pub rust_failed: usize,
    /// SciPy produced no value: it raised, or the oracle failed.
    pub oracle_missing: usize,
    /// Known discrepancies excused by the bead that owns them.
    pub allowlisted: usize,
}

#[derive(Debug, Clone)]
struct Problem {
    arm: String,
    case_id: String,
    what: String,
}

/// One test's record of what it actually compared against SciPy.
#[derive(Debug, Clone)]
pub struct CompareLedger {
    test_id: String,
    arms: BTreeMap<String, ArmCounts>,
    problems: Vec<Problem>,
}

impl CompareLedger {
    /// A ledger for `test_id` over the named arms (functions, or outputs of one function). An arm
    /// that records nothing is reported by `finish`, so declare every arm the test means to check.
    /// Recording under an undeclared arm panics, so a misspelled arm cannot hide.
    ///
    /// # Panics
    /// When `arms` is empty.
    #[must_use]
    pub fn new(test_id: &str, arms: &[&str]) -> Self {
        assert!(
            !arms.is_empty(),
            "{test_id}: a compared-case ledger needs at least one arm"
        );
        Self {
            test_id: test_id.to_owned(),
            arms: arms
                .iter()
                .map(|arm| ((*arm).to_owned(), ArmCounts::default()))
                .collect(),
            problems: Vec::new(),
        }
    }

    fn record(&mut self, arm: &str, case_id: &str, problem: Option<String>) -> &mut ArmCounts {
        let Self {
            test_id,
            arms,
            problems,
        } = self;
        let counts = arms
            .get_mut(arm)
            .unwrap_or_else(|| panic!("{test_id}: arm `{arm}` was not declared in the ledger"));
        if let Some(what) = problem {
            problems.push(Problem {
                arm: arm.to_owned(),
                case_id: case_id.to_owned(),
                what,
            });
        }
        counts
    }

    /// A case both sides computed; `pass` is the test's own tolerance verdict.
    pub fn compared(&mut self, arm: &str, case_id: &str, pass: bool) {
        let problem = (!pass).then(|| "outside tolerance".to_owned());
        let counts = self.record(arm, case_id, problem);
        counts.compared_cases += 1;
        if !pass {
            counts.compared_failed += 1;
        }
    }

    /// SciPy raised or returned nothing for this case.
    pub fn oracle_missing(&mut self, arm: &str, case_id: &str, reason: &str) {
        self.record(arm, case_id, Some(format!("SciPy gave no value: {reason}")))
            .oracle_missing += 1;
    }

    /// SciPy gave a definite value and fsci returned an error, nothing, or a non-finite value.
    pub fn rust_failed(&mut self, arm: &str, case_id: &str, reason: &str) {
        self.record(arm, case_id, Some(format!("fsci failed: {reason}")))
            .rust_failed += 1;
    }

    /// SciPy is documented to raise for this case. fsci refusing too is a compared, passing case;
    /// fsci returning a value is a compared, failing one.
    pub fn expected_raise(&mut self, arm: &str, case_id: &str, fsci_refused: bool) {
        let problem =
            (!fsci_refused).then(|| "SciPy raises here but fsci returned a value".to_owned());
        let counts = self.record(arm, case_id, problem);
        counts.compared_cases += 1;
        if !fsci_refused {
            counts.compared_failed += 1;
        }
    }

    /// A known discrepancy owned by `bead` (a `frankenscipy-` id). It does not count as compared,
    /// and a failing verdict lists it beside the problems.
    ///
    /// # Panics
    /// When `bead` is not a bead id.
    pub fn allowlisted(&mut self, arm: &str, case_id: &str, bead: &str, reason: &str) {
        assert!(
            bead.starts_with("frankenscipy-") && bead.len() > "frankenscipy-".len(),
            "{}: allowlisting {arm}/{case_id} needs the owning bead id, got `{bead}`",
            self.test_id
        );
        self.record(
            arm,
            case_id,
            Some(format!("allowlisted under {bead}: {reason}")),
        )
        .allowlisted += 1;
    }

    /// The common scalar shape: SciPy's value (`None` when it raised or returned nothing) against
    /// fsci's (`None` when it returned an error or nothing).
    ///
    /// Returns both values when both are finite; the caller computes its metric and records it
    /// with [`compared`](Self::compared). Every other combination is recorded here:
    /// - SciPy gave no value: `oracle_missing`.
    /// - SciPy finite or infinite, fsci missing or not matching that class: `rust_failed`
    ///   (for an infinite SciPy value, fsci's finite answer is a compared failure instead).
    /// - SciPy NaN: compared, passing when fsci also refused (NaN or no value), failing when fsci
    ///   invented a finite value.
    /// - Both infinite: compared, passing when the signs agree.
    pub fn pair(
        &mut self,
        arm: &str,
        case_id: &str,
        scipy: Option<f64>,
        fsci: Option<f64>,
    ) -> Option<(f64, f64)> {
        let Some(s) = scipy else {
            self.oracle_missing(arm, case_id, "no value");
            return None;
        };
        match fsci {
            Some(f) if s.is_finite() && f.is_finite() => Some((s, f)),
            _ if s.is_nan() => {
                self.compared(arm, case_id, fsci.is_none_or(f64::is_nan));
                None
            }
            Some(f) if s.is_infinite() && !f.is_nan() => {
                self.compared(arm, case_id, f == s);
                None
            }
            _ => {
                let got = fsci.map_or_else(|| "no value".to_owned(), |f| format!("{f}"));
                self.rust_failed(arm, case_id, &format!("{got} against SciPy {s}"));
                None
            }
        }
    }

    /// The presence check for a value of any shape: SciPy's (`None` when it raised or returned
    /// nothing) and fsci's (`None` when it failed). Records `oracle_missing` or `rust_failed` and
    /// returns both only when both exist; the caller compares them and calls
    /// [`compared`](Self::compared).
    pub fn both<S, F>(
        &mut self,
        arm: &str,
        case_id: &str,
        scipy: Option<S>,
        fsci: Option<F>,
    ) -> Option<(S, F)> {
        let Some(s) = scipy else {
            self.oracle_missing(arm, case_id, "no value");
            return None;
        };
        let Some(f) = fsci else {
            self.rust_failed(arm, case_id, "no value");
            return None;
        };
        Some((s, f))
    }

    /// Element-wise vectors: [`both`](Self::both), then a length mismatch or a non-finite element
    /// that does not match SciPy's (NaN for NaN, the same infinity) is a compared failure, and a
    /// non-finite fsci element where SciPy's is finite is an fsci failure. Without this a NaN
    /// vanishes in the usual `fold(0.0, f64::max)` of differences and the case passes. Returns
    /// both slices when every element is either finite on both sides or a matching non-finite.
    pub fn slices<'s, 'f>(
        &mut self,
        arm: &str,
        case_id: &str,
        scipy: Option<&'s [f64]>,
        fsci: Option<&'f [f64]>,
    ) -> Option<(&'s [f64], &'f [f64])> {
        let (s, f) = self.both(arm, case_id, scipy, fsci)?;
        if s.len() != f.len() {
            let what = format!("fsci length {} against SciPy {}", f.len(), s.len());
            self.compared_failure(arm, case_id, what);
            return None;
        }
        for (i, (&si, &fi)) in s.iter().zip(f).enumerate() {
            if si.is_finite() && !fi.is_finite() {
                self.rust_failed(
                    arm,
                    case_id,
                    &format!("element {i} is {fi} against SciPy {si}"),
                );
                return None;
            }
            let same_class = if si.is_nan() { fi.is_nan() } else { fi == si };
            if !si.is_finite() && !same_class {
                let what = format!("element {i} is {fi} against SciPy {si}");
                self.compared_failure(arm, case_id, what);
                return None;
            }
        }
        Some((s, f))
    }

    fn compared_failure(&mut self, arm: &str, case_id: &str, what: String) {
        let counts = self.record(arm, case_id, Some(what));
        counts.compared_cases += 1;
        counts.compared_failed += 1;
    }

    /// Counts per arm, for the test's diff log.
    #[must_use]
    pub fn counts(&self) -> &BTreeMap<String, ArmCounts> {
        &self.arms
    }

    /// `Err(table)` unless every arm compared at least `min_per_arm` cases, none of them failed,
    /// no fsci call failed against a definite SciPy value, and (when `require_oracle`) SciPy gave
    /// a value for every case.
    ///
    /// # Errors
    /// The report `finish` panics with.
    ///
    /// # Panics
    /// When `min_per_arm` is 0: a zero minimum is the vacuous pass this ledger exists to stop.
    pub fn verdict(&self, min_per_arm: usize, require_oracle: bool) -> Result<(), String> {
        assert!(
            min_per_arm >= 1,
            "{}: min_per_arm must be at least 1",
            self.test_id
        );
        let mut reasons = Vec::new();
        for (arm, c) in &self.arms {
            if c.compared_cases < min_per_arm {
                reasons.push(format!(
                    "arm `{arm}` compared {} case(s); the minimum is {min_per_arm}",
                    c.compared_cases
                ));
            }
            if c.compared_failed > 0 {
                reasons.push(format!(
                    "arm `{arm}`: {} case(s) outside tolerance",
                    c.compared_failed
                ));
            }
            if c.rust_failed > 0 {
                reasons.push(format!(
                    "arm `{arm}`: fsci failed on {} case(s) where SciPy gave a value",
                    c.rust_failed
                ));
            }
            if require_oracle && c.oracle_missing > 0 {
                reasons.push(format!(
                    "arm `{arm}`: SciPy gave no value for {} case(s) under {REQUIRE_SCIPY_ENV}=1",
                    c.oracle_missing
                ));
            }
        }
        if reasons.is_empty() {
            return Ok(());
        }
        let mut report = format!("{}: compared-case ledger failed\n", self.test_id);
        let _ = writeln!(
            report,
            "  {:<20} {:>9} {:>7} {:>12} {:>15} {:>12}",
            "arm", "compared", "failed", "rust_failed", "oracle_missing", "allowlisted"
        );
        for (arm, c) in &self.arms {
            let _ = writeln!(
                report,
                "  {arm:<20} {:>9} {:>7} {:>12} {:>15} {:>12}",
                c.compared_cases, c.compared_failed, c.rust_failed, c.oracle_missing, c.allowlisted
            );
        }
        for reason in &reasons {
            let _ = writeln!(report, "  - {reason}");
        }
        let _ = writeln!(
            report,
            "  problems (first {} of {}):",
            PROBLEMS_SHOWN.min(self.problems.len()),
            self.problems.len()
        );
        for p in self.problems.iter().take(PROBLEMS_SHOWN) {
            let _ = writeln!(report, "    {} {}: {}", p.arm, p.case_id, p.what);
        }
        Err(report)
    }

    /// Ends the test: panics with the [`verdict`](Self::verdict) table on any failure, reading
    /// `FSCI_REQUIRE_SCIPY_ORACLE` for the oracle rule. Returns the counts for the diff log.
    ///
    /// # Panics
    /// When the verdict is an error.
    pub fn finish(&self, min_per_arm: usize) -> BTreeMap<String, ArmCounts> {
        let require = std::env::var(REQUIRE_SCIPY_ENV).is_ok_and(|v| v == "1");
        if let Err(report) = self.verdict(min_per_arm, require) {
            panic!("{report}");
        }
        self.arms.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{ArmCounts, CompareLedger};

    fn counts(ledger: &CompareLedger, arm: &str) -> ArmCounts {
        ledger.counts()[arm]
    }

    #[test]
    fn an_arm_that_compared_nothing_fails_even_when_nothing_else_went_wrong() {
        let mut ledger = CompareLedger::new("t", &["pmf", "cdf"]);
        ledger.compared("pmf", "a", true);
        ledger.oracle_missing("cdf", "a", "raised");
        let report = ledger.verdict(1, false).expect_err("cdf compared nothing");
        assert!(report.contains("arm `cdf` compared 0 case(s)"), "{report}");
        assert!(
            report.contains("cdf a: SciPy gave no value: raised"),
            "{report}"
        );
    }

    #[test]
    fn every_arm_comparing_and_passing_is_accepted() {
        let mut ledger = CompareLedger::new("t", &["pmf", "cdf"]);
        ledger.compared("pmf", "a", true);
        ledger.compared("cdf", "a", true);
        ledger.oracle_missing("cdf", "b", "raised");
        assert_eq!(ledger.verdict(1, false), Ok(()));
        assert_eq!(
            counts(&ledger, "cdf"),
            ArmCounts {
                compared_cases: 1,
                oracle_missing: 1,
                ..ArmCounts::default()
            }
        );
    }

    #[test]
    fn a_rust_failure_against_a_scipy_value_fails_even_beside_passing_cases() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        for case in ["a", "b", "c"] {
            ledger.compared("x", case, true);
        }
        ledger.rust_failed("x", "d", "Err(Domain)");
        let report = ledger.verdict(1, false).expect_err("one fsci failure");
        assert!(report.contains("fsci failed on 1 case(s)"), "{report}");
    }

    #[test]
    fn a_missing_oracle_value_fails_only_when_scipy_is_required() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        ledger.compared("x", "a", true);
        ledger.oracle_missing("x", "b", "raised");
        assert_eq!(ledger.verdict(1, false), Ok(()));
        let report = ledger.verdict(1, true).expect_err("required oracle");
        assert!(
            report.contains("SciPy gave no value for 1 case(s)"),
            "{report}"
        );
    }

    #[test]
    fn the_minimum_is_per_arm_and_out_of_tolerance_cases_fail() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        ledger.compared("x", "a", true);
        assert!(ledger.verdict(2, false).is_err());
        ledger.compared("x", "b", false);
        let report = ledger
            .verdict(2, false)
            .expect_err("b is outside tolerance");
        assert!(report.contains("1 case(s) outside tolerance"), "{report}");
    }

    #[test]
    fn pair_classifies_every_combination() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        assert_eq!(
            ledger.pair("x", "both", Some(1.0), Some(2.0)),
            Some((1.0, 2.0))
        );
        assert_eq!(
            counts(&ledger, "x"),
            ArmCounts::default(),
            "the caller records it"
        );

        let rows: [(Option<f64>, Option<f64>, ArmCounts); 8] = [
            (
                None,
                Some(1.0),
                ArmCounts {
                    oracle_missing: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(1.0),
                None,
                ArmCounts {
                    rust_failed: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(1.0),
                Some(f64::NAN),
                ArmCounts {
                    rust_failed: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(f64::NAN),
                None,
                ArmCounts {
                    compared_cases: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(f64::NAN),
                Some(3.0),
                ArmCounts {
                    compared_cases: 1,
                    compared_failed: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(f64::INFINITY),
                Some(f64::INFINITY),
                ArmCounts {
                    compared_cases: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
                ArmCounts {
                    compared_cases: 1,
                    compared_failed: 1,
                    ..ArmCounts::default()
                },
            ),
            (
                Some(f64::NEG_INFINITY),
                None,
                ArmCounts {
                    rust_failed: 1,
                    ..ArmCounts::default()
                },
            ),
        ];
        for (scipy, fsci, want) in rows {
            let mut one = CompareLedger::new("t", &["x"]);
            assert_eq!(
                one.pair("x", "c", scipy, fsci),
                None,
                "{scipy:?} vs {fsci:?}"
            );
            assert_eq!(counts(&one, "x"), want, "{scipy:?} vs {fsci:?}");
        }
    }

    #[test]
    fn slices_catch_what_a_max_fold_of_differences_swallows() {
        let scipy = [1.0, 2.0, 3.0];
        // must-miss: 0.0_f64.max(NaN) is 0.0, so the usual fold would call this case exact
        let nan_diff = [1.0, f64::NAN, 3.0]
            .iter()
            .zip(&scipy)
            .map(|(a, b): (&f64, &f64)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert_eq!(nan_diff, 0.0);
        let mut ledger = CompareLedger::new("t", &["v"]);
        assert_eq!(
            ledger.slices("v", "nan", Some(&scipy), Some(&[1.0, f64::NAN, 3.0])),
            None
        );
        assert_eq!(counts(&ledger, "v").rust_failed, 1);
        assert_eq!(
            ledger.slices("v", "short", Some(&scipy), Some(&[1.0, 2.0])),
            None
        );
        assert_eq!(counts(&ledger, "v").compared_failed, 1);
        assert_eq!(ledger.slices("v", "none", Some(&scipy), None), None);
        assert_eq!(counts(&ledger, "v").rust_failed, 2);
        assert_eq!(ledger.slices("v", "gone", None, Some(&scipy)), None);
        assert_eq!(counts(&ledger, "v").oracle_missing, 1);
        // a NaN SciPy element must be matched by NaN, not by a number
        let with_nan = [1.0, f64::NAN];
        assert_eq!(
            ledger.slices("v", "invented", Some(&with_nan), Some(&[1.0, 0.0])),
            None
        );
        assert_eq!(counts(&ledger, "v").compared_failed, 2);
        // must-hit: finite everywhere, or matching non-finite, is handed back to compare
        let got = ledger.slices("v", "ok", Some(&with_nan), Some(&[1.0 + 1e-15, f64::NAN]));
        assert!(got.is_some());
        let inf = [f64::NEG_INFINITY, 0.5];
        assert!(ledger.slices("v", "inf", Some(&inf), Some(&inf)).is_some());
    }

    #[test]
    fn both_records_the_missing_side() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        assert_eq!(ledger.both("x", "a", Some("s"), Some(2)), Some(("s", 2)));
        assert_eq!(ledger.both::<&str, i32>("x", "b", None, None), None);
        assert_eq!(ledger.both("x", "c", Some("s"), None::<i32>), None);
        assert_eq!(
            counts(&ledger, "x"),
            ArmCounts {
                oracle_missing: 1,
                rust_failed: 1,
                ..ArmCounts::default()
            }
        );
    }

    #[test]
    fn oracle_f64_tells_a_nan_answer_from_a_missing_one() {
        #[derive(serde::Deserialize)]
        struct Row {
            #[serde(default, deserialize_with = "super::oracle_f64")]
            v: Option<f64>,
        }
        let parse = |s: &str| serde_json::from_str::<Row>(s).map(|r| r.v);
        assert_eq!(parse(r#"{"v": null}"#).unwrap(), None);
        assert_eq!(parse("{}").unwrap(), None);
        assert_eq!(parse(r#"{"v": 1.5}"#).unwrap(), Some(1.5));
        assert_eq!(parse(r#"{"v": 3}"#).unwrap(), Some(3.0));
        assert!(parse(r#"{"v": "nan"}"#).unwrap().is_some_and(f64::is_nan));
        assert_eq!(parse(r#"{"v": "-inf"}"#).unwrap(), Some(f64::NEG_INFINITY));
        assert!(parse(r#"{"v": "NaN?"}"#).is_err());
        // must-hit: a NaN answer is a matching refusal, not a missing oracle
        let mut ledger = CompareLedger::new("t", &["x"]);
        ledger.pair("x", "a", parse(r#"{"v": "nan"}"#).unwrap(), None);
        ledger.pair("x", "b", parse(r#"{"v": null}"#).unwrap(), None);
        assert_eq!(
            counts(&ledger, "x"),
            ArmCounts {
                compared_cases: 1,
                oracle_missing: 1,
                ..ArmCounts::default()
            }
        );
    }

    #[test]
    fn a_declared_raise_matches_only_a_refusal() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        ledger.expected_raise("x", "a", true);
        assert_eq!(ledger.verdict(1, true), Ok(()));
        ledger.expected_raise("x", "b", false);
        assert!(ledger.verdict(1, true).is_err());
    }

    #[test]
    fn allowlisted_cases_do_not_count_as_compared() {
        let mut ledger = CompareLedger::new("t", &["x"]);
        ledger.allowlisted("x", "a", "frankenscipy-3u8ze", "zeta tail truncation");
        assert_eq!(counts(&ledger, "x").allowlisted, 1);
        assert!(ledger.verdict(1, false).is_err());
    }

    #[test]
    #[should_panic(expected = "needs the owning bead id")]
    fn an_allowlist_entry_without_a_bead_is_rejected() {
        CompareLedger::new("t", &["x"]).allowlisted("x", "a", "later", "reason");
    }

    #[test]
    #[should_panic(expected = "arm `cfd` was not declared")]
    fn a_misspelled_arm_is_rejected() {
        CompareLedger::new("t", &["cdf"]).compared("cfd", "a", true);
    }

    #[test]
    #[should_panic(expected = "compared-case ledger failed")]
    fn finish_panics_on_an_empty_comparison() {
        let _ = CompareLedger::new("t", &["x"]).finish(1);
    }
}
