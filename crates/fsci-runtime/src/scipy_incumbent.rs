//! One canonical answer to "where is the live SciPy incumbent, and is it the pinned one?".
//!
//! # Why this exists
//!
//! Every vs-SciPy harness in this workspace spawns a Python child and compares against it.
//! Until 2026-09-01 each one carried its own copy of the answer, and every copy named two
//! absolute paths:
//!
//! ```text
//! /usr/bin/python3.13
//! /data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages
//! ```
//!
//! Both of those are gone from `thinkstation1`. Neither disappearance is checked by anything
//! that runs before the fixture is written to the child's stdin, so the failure surfaced as
//! `BrokenPipe` several lines *after* a well-formed provenance header had already printed --
//! a missing incumbent wearing the costume of a flaky pipe.
//!
//! `frankenscipy-m5s54` read that symptom as "the pinned scipy 1.17.1 incumbent is
//! unreachable". That conclusion was half wrong, and the wrong half is the expensive one:
//!
//! ```text
//! /home/ubuntu/.local/bin/python3.13 -> scipy 1.17.1, numpy 2.4.3, fsci_loaded=False
//! ```
//!
//! The pinned pair is present on the local host and is exactly what every certified row in
//! the ledger already cites. What broke was DISCOVERY, not the incumbent. So the fix is not
//! to relax a version gate; it is to stop hard-coding paths and prove the import instead.
//!
//! # What is pinned, and why it stayed 1.17.1
//!
//! [`PINNED_SCIPY`]/[`PINNED_NUMPY`] are the recorded incumbent. Three facts decided it:
//!
//! 1. The pair is reachable on `thinkstation1`, so the pin is satisfiable rather than
//!    aspirational.
//! 2. Every previously certified row cites it. Moving the pin would silently make new rows
//!    incomparable to the entire existing ledger while every row still looked well-formed.
//! 3. The obvious alternative -- follow the rch fleet to scipy 1.18.1 -- is not one pin. The
//!    sampled workers disagree with each other on numpy (2.3.5 on the `vmi` pair, 2.5.2 on
//!    `ovh-a`), so worker rows would not be comparable to worker rows.
//!
//! The consequence is a real constraint and is stated rather than papered over: live-incumbent
//! rows are LOCAL-ONLY until the fleet is provisioned with the pinned pair. A harness on a
//! worker will resolve an interpreter, report `genuine=false`, and refuse -- which is the
//! correct behaviour and is not a bug to be tuned away.
//!
//! # The probe runs under the caller's environment, not a convenient one
//!
//! [`ScipyIncumbent::resolve_with`] takes the same environment overlay the caller will later
//! spawn the oracle under, and [`ScipyIncumbent::apply_to`] replays exactly what the probe
//! proved. This is load-bearing rather than tidy. Several harnesses spawn with
//! `PYTHONNOUSERSITE=1`, and on this host the only SciPy lives in the *user* site directory;
//! a probe that omitted that variable would report a working interpreter and hand back one
//! that cannot import SciPy at all. A probe reproducing a symptom is not the same thing as a
//! probe sharing a code path.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};

/// The recorded incumbent SciPy version. See the module header for why it is still 1.17.1.
pub const PINNED_SCIPY: &str = "1.17.1";
/// The recorded incumbent NumPy version.
///
/// NumPy is part of the pin because it is part of the answer: SciPy dispatches array work to
/// it, and the sampled fleet disagreed with itself about which NumPy it had. A row that names
/// only SciPy cannot be compared to another row that names only SciPy.
pub const PINNED_NUMPY: &str = "2.4.3";

/// Site-packages directories that have held the pinned incumbent, newest-known first.
///
/// Ordering is deliberate: the entry that no longer exists is kept so that a host which has
/// been re-provisioned with the pinned tree is preferred over one relying on a user site.
/// Non-existent entries are skipped before probing, so a stale entry costs nothing.
pub const SITE_PACKAGES_CANDIDATES: [&str; 2] = [
    "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages",
    "/home/ubuntu/.local/lib/python3.13/site-packages",
];

/// Interpreters to try, in order, when `SCIPY_PYTHON` is unset.
///
/// `python3` is last on purpose. On `thinkstation1` it is 3.14 with no SciPy at all, and it is
/// precisely the fall-through that turned a missing incumbent into a `BrokenPipe`. It stays in
/// the list because a correctly provisioned host may well have the incumbent there, but it
/// must never win over a named 3.13.
pub const PYTHON_CANDIDATES: [&str; 3] = [
    "/usr/bin/python3.13",
    "/home/ubuntu/.local/bin/python3.13",
    "python3",
];

/// Environment every oracle spawn should carry, so the incumbent is single-threaded and its
/// BLAS does not silently recruit the whole box.
pub const SINGLE_THREAD_ENV: [(&str, &str); 6] = [
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
];

/// Emitted by the probe child. One line, `key=value` pairs, so a partially-written line from a
/// dying interpreter parses as a failure rather than as a plausible answer.
const PROBE_SOURCE: &str = r#"
import sys

try:
    import scipy
    import numpy
    for _name in sys.argv[1:]:
        __import__(_name)
except BaseException as exc:
    sys.stdout.write("FSCI_PROBE ok=0 error=%r\n" % (exc,))
else:
    _fsci = any(
        name.startswith("fsci") or name.startswith("franken") for name in sys.modules
    )
    sys.stdout.write(
        "FSCI_PROBE ok=1 scipy=%s numpy=%s fsci_loaded=%s executable=%s\n"
        % (scipy.__version__, numpy.__version__, int(_fsci), sys.executable)
    )
sys.stdout.flush()
"#;

/// A resolved, proven live-SciPy incumbent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScipyIncumbent {
    /// Interpreter that was proven to import SciPy under `env`.
    pub python: String,
    /// `PYTHONPATH` the proof used, if one was needed. `None` means the interpreter found
    /// SciPy on its own default path.
    pub pythonpath: Option<String>,
    /// Environment overlay the proof ran under, and that [`Self::apply_to`] replays.
    pub env: BTreeMap<String, String>,
    /// Reported `scipy.__version__`.
    pub scipy_version: String,
    /// Reported `numpy.__version__`.
    pub numpy_version: String,
    /// Whether any FrankenSciPy module was resident in the oracle interpreter. A `true` here
    /// means the "incumbent" is measuring us, so it is not an incumbent.
    pub fsci_loaded: bool,
    /// Interpreter's own `sys.executable`, which is not always the path we invoked.
    pub executable: String,
    /// Every `(interpreter, pythonpath) -> outcome` pair the resolver tried, in order.
    pub probe_trail: Vec<String>,
}

impl ScipyIncumbent {
    /// Is this the recorded incumbent, uncontaminated by FrankenSciPy?
    ///
    /// All three clauses matter and none is redundant: a wrong SciPy is a different library, a
    /// wrong NumPy is a different array engine underneath the same library, and a loaded
    /// `fsci` means the oracle is not independent of the thing it is judging.
    #[must_use]
    pub fn genuine(&self) -> bool {
        self.scipy_version == PINNED_SCIPY
            && self.numpy_version == PINNED_NUMPY
            && !self.fsci_loaded
    }

    /// Why [`Self::genuine`] is false, phrased for a refusal message. `None` when it is true.
    #[must_use]
    pub fn disagreement(&self) -> Option<String> {
        if self.genuine() {
            return None;
        }
        let mut parts = Vec::new();
        if self.scipy_version != PINNED_SCIPY {
            parts.push(format!(
                "scipy {} != pinned {PINNED_SCIPY}",
                self.scipy_version
            ));
        }
        if self.numpy_version != PINNED_NUMPY {
            parts.push(format!(
                "numpy {} != pinned {PINNED_NUMPY}",
                self.numpy_version
            ));
        }
        if self.fsci_loaded {
            parts.push("FrankenSciPy modules are resident in the oracle interpreter".to_string());
        }
        Some(parts.join("; "))
    }

    /// Provenance for a ledger row. Names BOTH versions, because a row naming one cannot be
    /// compared to a row naming the other.
    #[must_use]
    pub fn provenance_line(&self) -> String {
        format!(
            "scipy_incumbent: python={} pythonpath={} scipy={} numpy={} fsci_loaded={} \
             genuine={} pinned_scipy={PINNED_SCIPY} pinned_numpy={PINNED_NUMPY}",
            self.python,
            self.pythonpath.as_deref().unwrap_or("<default>"),
            self.scipy_version,
            self.numpy_version,
            self.fsci_loaded,
            self.genuine(),
        )
    }

    /// Configure `command` to run this exact interpreter under this exact environment.
    ///
    /// The caller must not re-derive the interpreter or the `PYTHONPATH` itself; doing so is
    /// how the probed configuration and the timed configuration come apart.
    pub fn apply_to(&self, command: &mut Command) {
        for (key, value) in &self.env {
            command.env(key, value);
        }
        match self.pythonpath.as_deref() {
            Some(path) => {
                command.env("PYTHONPATH", path);
            }
            None => {
                command.env_remove("PYTHONPATH");
            }
        }
    }

    /// A `Command` for this interpreter, already carrying the proven environment.
    #[must_use]
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.python);
        self.apply_to(&mut command);
        command
    }

    /// Resolve under [`SINGLE_THREAD_ENV`], requiring only `scipy` and `numpy` to import.
    ///
    /// # Errors
    /// Returns [`ResolveError`] when no candidate interpreter can import SciPy.
    pub fn resolve() -> Result<Self, ResolveError> {
        Self::resolve_with(&[], &[])
    }

    /// Resolve under [`SINGLE_THREAD_ENV`] plus `extra_env`, additionally proving that each
    /// module in `required_modules` imports.
    ///
    /// Pass the submodules the oracle actually uses (`scipy.sparse.linalg`,
    /// `scipy.integrate`, ...). A bare `import scipy` can succeed on an installation whose
    /// compiled submodules do not load, and that difference only shows up mid-timing.
    ///
    /// `extra_env` must be whatever the caller will later spawn the oracle under. It is applied
    /// to the probe as well, so the probe cannot pass under conditions the oracle will not get.
    ///
    /// # Errors
    /// Returns [`ResolveError`] when no candidate interpreter can import SciPy and every
    /// module in `required_modules` under this environment.
    pub fn resolve_with(
        extra_env: &[(&str, &str)],
        required_modules: &[&str],
    ) -> Result<Self, ResolveError> {
        let mut env: BTreeMap<String, String> = SINGLE_THREAD_ENV
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect();
        for (key, value) in extra_env {
            env.insert((*key).to_string(), (*value).to_string());
        }

        // `None` first: an interpreter that already has the incumbent on its default path
        // should be used as-is. Prepending a foreign site-packages onto a working interpreter
        // is how version skew gets introduced by the very code meant to prevent it.
        let mut pythonpaths: Vec<Option<String>> = vec![None];
        pythonpaths.extend(
            SITE_PACKAGES_CANDIDATES
                .iter()
                .filter(|path| Path::new(*path).is_dir())
                .map(|path| Some((*path).to_string())),
        );

        let pinned = std::env::var("SCIPY_PYTHON").ok().filter(|v| !v.is_empty());
        let candidates: Vec<String> = match pinned {
            // An explicit pin is a deliberate act. It is still probed, so a typo fails with the
            // interpreter named instead of as a broken pipe, but nothing routes around it.
            Some(python) => vec![python],
            None => PYTHON_CANDIDATES
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        };

        let mut probe_trail = Vec::new();
        for python in &candidates {
            for pythonpath in &pythonpaths {
                match probe(python, pythonpath.as_deref(), &env, required_modules) {
                    Ok(report) => {
                        probe_trail.push(format!(
                            "{python}+{}=ok",
                            pythonpath.as_deref().unwrap_or("<default>")
                        ));
                        return Ok(Self {
                            python: python.clone(),
                            pythonpath: pythonpath.clone(),
                            env,
                            scipy_version: report.scipy_version,
                            numpy_version: report.numpy_version,
                            fsci_loaded: report.fsci_loaded,
                            executable: report.executable,
                            probe_trail,
                        });
                    }
                    Err(reason) => probe_trail.push(format!(
                        "{python}+{}={reason}",
                        pythonpath.as_deref().unwrap_or("<default>")
                    )),
                }
            }
        }
        Err(ResolveError { probe_trail })
    }
}

/// Does `python`, under this `PYTHONPATH`, actually import SciPy and every module in
/// `required_modules`?
///
/// Split out of the resolver so BOTH of its answers can be exercised. The must-MISS arm is
/// the one that matters: the selection this replaced was `Path::exists`, a predicate that
/// cannot fail for the reason we care about, because `python3` exists on every host here and
/// imports SciPy on almost none of them. A probe that only ever returns `true` reads exactly
/// like a working one, right up until it routes an afternoon of rows into a broken pipe.
#[must_use]
pub fn interpreter_can_import_scipy(
    python: &str,
    pythonpath: Option<&str>,
    required_modules: &[&str],
) -> bool {
    let env: BTreeMap<String, String> = SINGLE_THREAD_ENV
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    probe(python, pythonpath, &env, required_modules).is_ok()
}

/// No interpreter on this host could import the incumbent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolveError {
    /// Every candidate tried, with the reason it was rejected.
    pub probe_trail: Vec<String>,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "no interpreter on this host can import the live SciPy incumbent, so every ratio \
             against it would be meaningless. Set SCIPY_PYTHON to one that can. Probed: {}",
            self.probe_trail.join(" ")
        )
    }
}

impl std::error::Error for ResolveError {}

#[derive(Debug)]
struct ProbeReport {
    scipy_version: String,
    numpy_version: String,
    fsci_loaded: bool,
    executable: String,
}

fn probe(
    python: &str,
    pythonpath: Option<&str>,
    env: &BTreeMap<String, String>,
    required_modules: &[&str],
) -> Result<ProbeReport, String> {
    let mut command = Command::new(python);
    for (key, value) in env {
        command.env(key, value);
    }
    match pythonpath {
        Some(path) => {
            command.env("PYTHONPATH", path);
        }
        None => {
            command.env_remove("PYTHONPATH");
        }
    }
    let output = command
        .arg("-c")
        .arg(PROBE_SOURCE)
        .args(required_modules)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|error| format!("spawn:{error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("FSCI_PROBE "))
        .ok_or_else(|| "no-probe-line".to_string())?;
    parse_probe_line(line)
}

fn parse_probe_line(line: &str) -> Result<ProbeReport, String> {
    let mut fields: BTreeMap<&str, &str> = BTreeMap::new();
    for token in line.trim_end().split(' ').skip(1) {
        if let Some((key, value)) = token.split_once('=') {
            fields.insert(key, value);
        }
    }
    if fields.get("ok").copied() != Some("1") {
        return Err(format!(
            "import-failed:{}",
            fields.get("error").copied().unwrap_or("unknown")
        ));
    }
    Ok(ProbeReport {
        scipy_version: (*fields.get("scipy").ok_or("probe-line-missing-scipy")?).to_string(),
        numpy_version: (*fields.get("numpy").ok_or("probe-line-missing-numpy")?).to_string(),
        fsci_loaded: fields.get("fsci_loaded").copied() == Some("1"),
        executable: (*fields.get("executable").unwrap_or(&"<unknown>")).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn incumbent(scipy: &str, numpy: &str, fsci_loaded: bool) -> ScipyIncumbent {
        ScipyIncumbent {
            python: "python3.13".to_string(),
            pythonpath: None,
            env: BTreeMap::new(),
            scipy_version: scipy.to_string(),
            numpy_version: numpy.to_string(),
            fsci_loaded,
            executable: "python3.13".to_string(),
            probe_trail: Vec::new(),
        }
    }

    /// Each clause of `genuine` is exercised in both directions. A predicate only ever shown
    /// its passing case reports `true` whether it is testing anything or nothing at all.
    #[test]
    fn genuine_requires_all_three_clauses() {
        assert!(incumbent(PINNED_SCIPY, PINNED_NUMPY, false).genuine());
        assert!(!incumbent("1.18.1", PINNED_NUMPY, false).genuine());
        assert!(!incumbent(PINNED_SCIPY, "2.5.2", false).genuine());
        assert!(!incumbent(PINNED_SCIPY, PINNED_NUMPY, true).genuine());
    }

    /// The fleet skew that produced `frankenscipy-m5s54`, kept as a regression: relaxing the
    /// pin to 1.18.1 would admit two workers that disagree with each other about NumPy.
    #[test]
    fn sampled_fleet_versions_are_all_refused_and_say_why() {
        for (scipy, numpy) in [("1.18.1", "2.5.2"), ("1.18.1", "2.3.5")] {
            let reason = incumbent(scipy, numpy, false)
                .disagreement()
                .expect("fleet skew must be refused");
            assert!(reason.contains("scipy 1.18.1"), "{reason}");
            assert!(reason.contains(numpy), "{reason}");
        }
        assert_eq!(
            incumbent(PINNED_SCIPY, PINNED_NUMPY, false).disagreement(),
            None
        );
    }

    #[test]
    fn provenance_names_both_versions_and_the_pin() {
        let line = incumbent(PINNED_SCIPY, PINNED_NUMPY, false).provenance_line();
        for expected in [
            "scipy=1.17.1",
            "numpy=2.4.3",
            "pinned_scipy=1.17.1",
            "pinned_numpy=2.4.3",
            "genuine=true",
            "fsci_loaded=false",
        ] {
            assert!(line.contains(expected), "{line} is missing {expected}");
        }
    }

    /// Both arms of the parser, so "it parsed" is distinguishable from "it matched anything".
    #[test]
    fn probe_line_parser_accepts_a_report_and_refuses_the_failures() {
        let good = parse_probe_line(
            "FSCI_PROBE ok=1 scipy=1.17.1 numpy=2.4.3 fsci_loaded=0 executable=/bin/python3.13",
        )
        .expect("well-formed report");
        assert_eq!(good.scipy_version, "1.17.1");
        assert_eq!(good.numpy_version, "2.4.3");
        assert!(!good.fsci_loaded);

        let contaminated = parse_probe_line(
            "FSCI_PROBE ok=1 scipy=1.17.1 numpy=2.4.3 fsci_loaded=1 executable=/bin/python3.13",
        )
        .expect("well-formed report");
        assert!(contaminated.fsci_loaded);

        let failed = parse_probe_line("FSCI_PROBE ok=0 error=ModuleNotFoundError('scipy')")
            .expect_err("an import failure is not a report");
        assert!(failed.starts_with("import-failed:"), "{failed}");

        // A truncated line from a dying interpreter must not read as a plausible answer.
        parse_probe_line("FSCI_PROBE ok=1 scipy=1.17.1")
            .expect_err("a half-written line is not a report");
    }

    /// The probe answers BOTH ways -- the two-arm control. One arm proves nothing: blindness
    /// and blanket-matching both print clean numbers.
    #[test]
    fn interpreter_probe_answers_both_ways() {
        // MUST MISS: an interpreter that is not on the box at all. This arm needs no SciPy
        // anywhere, so it runs identically on a bare worker.
        assert!(
            !interpreter_can_import_scipy("/nonexistent/bin/python-not-installed", None, &[]),
            "probe claimed a nonexistent interpreter can import scipy"
        );
        // MUST MISS for the same reason even when handed a plausible PYTHONPATH: it is the
        // IMPORT that is proven, never the existence of a directory.
        assert!(
            !interpreter_can_import_scipy(
                "/nonexistent/bin/python-not-installed",
                Some("/tmp"),
                &[]
            ),
            "probe was satisfied by a path rather than by an import"
        );
        // MUST MISS on a real, working interpreter asked for a module that cannot exist.
        // Without this arm, a probe that ignored `required_modules` entirely would pass.
        assert!(
            !interpreter_can_import_scipy(
                "python3",
                None,
                &["scipy.this_submodule_does_not_exist"]
            ),
            "probe ignored required_modules"
        );
        // MUST HIT, but only where an incumbent is actually installed. Asserting
        // unconditionally would make this a host check rather than a probe check.
        match ScipyIncumbent::resolve() {
            Ok(found) => {
                assert!(
                    interpreter_can_import_scipy(&found.python, found.pythonpath.as_deref(), &[]),
                    "probe is not repeatable on {}",
                    found.python
                );
                println!("must-hit arm observed on {}", found.python);
            }
            Err(_) => println!(
                "must-hit arm SKIPPED: no candidate on this host imports scipy, so only the \
                 must-miss arms ran here"
            ),
        }
    }

    /// End-to-end against whatever this host actually has, and it must stay meaningful on a
    /// host WITHOUT the incumbent -- which is most rch workers. So it asserts the property
    /// that holds either way: whatever `resolve` proved, `command()` must reproduce.
    ///
    /// The success arm re-runs the import through `command()` rather than trusting the
    /// resolver's own report. That is the whole point of `apply_to` existing, and a resolver
    /// that returned a working interpreter alongside an environment that breaks it would pass
    /// every other test in this file.
    ///
    /// The failure arm asserts the resolver actually exhausted its candidates instead of
    /// bailing on the first miss, because "nothing worked" and "nothing was tried" produce
    /// the same `Err` and only one of them is a real answer.
    #[test]
    fn resolve_agrees_with_this_host_whichever_way_it_goes() {
        match ScipyIncumbent::resolve() {
            Ok(found) => {
                assert!(!found.scipy_version.is_empty(), "{found:?}");
                assert!(!found.numpy_version.is_empty(), "{found:?}");
                let line = found.provenance_line();
                assert!(
                    line.contains(&format!("scipy={}", found.scipy_version)),
                    "{line}"
                );
                assert!(
                    line.contains(&format!("numpy={}", found.numpy_version)),
                    "{line}"
                );

                let replayed = found
                    .command()
                    .arg("-c")
                    .arg("import scipy, numpy")
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .expect("replayed interpreter must be spawnable");
                assert!(
                    replayed.success(),
                    "resolve() proved an import that command() cannot reproduce: {found:?}"
                );
                println!("{line}");
            }
            Err(error) => {
                // Every interpreter candidate must appear in the trail. Existing site-packages
                // multiply the trail, so this is a lower bound, not an equality.
                assert!(
                    error.probe_trail.len() >= PYTHON_CANDIDATES.len(),
                    "resolver gave up early: {:?}",
                    error.probe_trail
                );
                for python in PYTHON_CANDIDATES {
                    assert!(
                        error
                            .probe_trail
                            .iter()
                            .any(|entry| entry.starts_with(python)),
                        "{python} was never tried: {:?}",
                        error.probe_trail
                    );
                }
                println!("no incumbent on this host: {error}");
            }
        }
    }

    /// `apply_to` must replay the probed configuration exactly, including CLEARING an
    /// inherited `PYTHONPATH` when the proof did not use one. Inheriting a stray `PYTHONPATH`
    /// from the agent's shell is a way to spawn a different SciPy than the one proven.
    #[test]
    fn apply_to_replays_the_probed_environment() {
        let mut resolved = incumbent(PINNED_SCIPY, PINNED_NUMPY, false);
        resolved
            .env
            .insert("OMP_NUM_THREADS".to_string(), "1".to_string());
        resolved.pythonpath = Some("/some/site-packages".to_string());
        let with_path = resolved.command();
        let replayed: BTreeMap<_, _> = with_path
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert_eq!(
            replayed.get("PYTHONPATH"),
            Some(&Some("/some/site-packages".to_string()))
        );
        assert_eq!(
            replayed.get("OMP_NUM_THREADS"),
            Some(&Some("1".to_string()))
        );

        resolved.pythonpath = None;
        let without_path = resolved.command();
        let cleared: BTreeMap<_, _> = without_path
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert_eq!(cleared.get("PYTHONPATH"), Some(&None));
    }
}
