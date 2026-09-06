//! Lint conformance fixture cases against `artifacts/TOLERANCE_POLICY.md`.
//!
//! For each FSCI-P2C-*.json fixture, walks every case and compares its
//! `(rtol, atol)` against the per-packet baseline tier from §2 of the policy.
//! Cases looser than baseline that lack a `rationale` field are violations.
//!
//! Usage:
//!     tolerance_lint                       # default fixture dir
//!     tolerance_lint --fixtures-dir DIR    # override fixture dir
//!     tolerance_lint --json                # emit JSON report
//!     tolerance_lint --max-violations N    # exit 1 if violations exceed N (default 0 = strict)
//!     tolerance_lint --baseline N          # exit 1 if violations regress past starting point N
//!     tolerance_lint --write-baseline      # anchor/refresh the harness ratchet
//!                                          # (refuses to adopt LOOSENED values)
//!
//! Harness ratchet (frankenscipy-6a5s9): the scan also covers every
//! `tests/diff_*.rs` tolerance contract — `const <NAME>_TOL: f64 = <literal>`
//! declarations and bare `<lhs-diff> <= <float literal>` comparisons — against
//! the committed baseline `fixtures/tolerance_baseline.json`. A tolerance
//! larger than its baseline, a harness missing from the baseline, or a
//! baseline entry whose file is gone is a violation (exit 1). Tightening is
//! never a violation; `--write-baseline` adopts tightenings, new contracts,
//! and removals, but refuses to run when any value loosened — loosening
//! requires a hand-edited baseline in the same commit.
//!
//! Exit codes:
//!     0 — within budget
//!     1 — violation count over the configured threshold
//!     2 — IO/parse error

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::Value;

/// Per-packet baseline tolerance tier (rtol). Cases looser than this require
/// an explicit `rationale` field per `artifacts/TOLERANCE_POLICY.md` §2.
/// `None` means the packet has no numeric baseline (structural / Tnone).
fn packet_baseline_rtol(packet: &str) -> Option<f64> {
    match packet {
        "FSCI-P2C-001" => None,
        "FSCI-P2C-002" => Some(1e-12),
        "FSCI-P2C-003" => None,
        "FSCI-P2C-004" => Some(1e-10),
        "FSCI-P2C-005" => Some(1e-10),
        "FSCI-P2C-006" => Some(1e-12),
        "FSCI-P2C-007" => None,
        "FSCI-P2C-008" => None,
        "FSCI-P2C-009" => Some(1e-10),
        "FSCI-P2C-010" => Some(1e-10),
        "FSCI-P2C-011" => Some(1e-10),
        "FSCI-P2C-012" => Some(1e-10),
        "FSCI-P2C-013" => Some(1e-10),
        "FSCI-P2C-014" => Some(1e-12),
        "FSCI-P2C-015" => Some(1e-12),
        "FSCI-P2C-016" => Some(1e-15),
        "FSCI-P2C-017" => Some(1e-12),
        _ => None,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct Violation {
    packet: String,
    case_id: String,
    rtol: f64,
    atol: f64,
    baseline_rtol: f64,
    multiple: f64,
}

fn extract_case_tolerance(case: &Value) -> Option<(f64, f64, Option<String>)> {
    let case_obj = case.as_object()?;
    let expected = case_obj.get("expected").and_then(|v| v.as_object());
    let rtol = expected
        .and_then(|e| e.get("rtol"))
        .or_else(|| case_obj.get("rtol"))
        .and_then(|v| v.as_f64())?;
    let atol = expected
        .and_then(|e| e.get("atol"))
        .or_else(|| case_obj.get("atol"))
        .and_then(|v| v.as_f64())?;
    let rationale = case_obj
        .get("rationale")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| {
            expected
                .and_then(|e| e.get("rationale"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
        });
    Some((rtol, atol, rationale))
}

fn lint_fixture(path: &PathBuf) -> Result<Vec<Violation>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: Value =
        serde_json::from_str(&content).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let cases = parsed
        .as_object()
        .and_then(|o| o.get("cases"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let packet = stem.split('_').next().unwrap_or("").to_owned();
    let Some(baseline) = packet_baseline_rtol(&packet) else {
        return Ok(Vec::new());
    };

    let mut violations = Vec::new();
    for case in &cases {
        let Some((rtol, atol, rationale)) = extract_case_tolerance(case) else {
            continue;
        };
        if rationale.is_some() {
            continue;
        }
        if rtol > baseline * 1.001 {
            let case_id = case
                .as_object()
                .and_then(|o| o.get("case_id").or_else(|| o.get("operation")))
                .and_then(|v| v.as_str())
                .unwrap_or("(unnamed)")
                .to_owned();
            let multiple = if baseline > 0.0 {
                rtol / baseline
            } else {
                f64::INFINITY
            };
            violations.push(Violation {
                packet: packet.clone(),
                case_id,
                rtol,
                atol,
                baseline_rtol: baseline,
                multiple,
            });
        }
    }
    Ok(violations)
}

fn print_text_report(violations: &[Violation]) {
    let mut by_packet: std::collections::BTreeMap<String, Vec<&Violation>> =
        std::collections::BTreeMap::new();
    for v in violations {
        by_packet.entry(v.packet.clone()).or_default().push(v);
    }
    println!("Tolerance Lint Report");
    println!("=====================");
    println!("Reference: artifacts/TOLERANCE_POLICY.md §2 baseline tiers");
    println!();
    for (packet, items) in &by_packet {
        println!("--- {packet} ({} violations) ---", items.len());
        for v in items {
            println!(
                "  {:50} rtol={:.0e} (×{:>6.1} baseline {:.0e})",
                v.case_id, v.rtol, v.multiple, v.baseline_rtol
            );
        }
        println!();
    }
    println!(
        "Total: {} violations across {} packets",
        violations.len(),
        by_packet.len()
    );
}

// ── Harness tolerance ratchet (tests/diff_*.rs) ─────────────────────────
//
// G9 previously walked only the FSCI-P2C-*.json fixtures. The 731 diff_*.rs
// harnesses each carry their own tolerance contracts (`const *_TOL: f64`
// declarations; every harness in the corpus uses that form as of 2026-09-06)
// that no gate watched — a contributor could loosen ABS_TOL from 1e-12 to
// 1e-3 invisibly. This section ratchets them against a committed baseline
// (`fixtures/tolerance_baseline.json`: file name -> {contract -> value}).

const TESTS_DIR_REL: &str = "crates/fsci-conformance/tests";
const HARNESS_BASELINE_REL: &str = "crates/fsci-conformance/fixtures/tolerance_baseline.json";

type HarnessBaseline = std::collections::BTreeMap<String, std::collections::BTreeMap<String, f64>>;

#[derive(Debug, Clone, serde::Serialize)]
struct HarnessViolation {
    file: String,
    name: String,
    /// "loosened" | "not_in_baseline" | "file_not_in_baseline" | "file_removed"
    kind: &'static str,
    current: Option<f64>,
    baseline: Option<f64>,
}

/// Parse the tail of a `const <NAME>: f64 = <literal>;` declaration whose
/// NAME contains `_TOL` (ABS_TOL, REL_TOL, ABS_TOL_CHI, PDF_TOL, …). The
/// `: f64` type guard keeps non-f64 consts whose names merely contain TOL
/// out of the ratchet. `None` when the line is not a tolerance const at all.
fn parse_tol_const(tail: &str) -> Option<(String, f64)> {
    let name = tail.split(':').next()?.trim();
    if !name.contains("_TOL") {
        return None;
    }
    let type_seg = tail.split(':').nth(1)?;
    if !type_seg.trim_start().starts_with("f64") {
        return None;
    }
    let value_str = tail.split('=').nth(1)?;
    // A trailing `// comment` after the literal is common
    // (`= 0.5; // Coarse grid resolution`); it is not part of the value.
    let value_str = value_str.split("//").next().unwrap_or("");
    let value_str = value_str.trim().trim_end_matches(';').trim();
    let value: f64 = value_str.parse().ok()?;
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    Some((name.to_string(), value))
}

/// Bare tolerance comparisons: `<lhs ending in "diff"> <= <float literal>`.
/// The literal must be complete — a trailing arithmetic or call continuation
/// (`4.0 * se.max(1e-3)`, `2.0 * tolerance`) disqualifies it, which keeps
/// composite statistical rules out of the ratchet.
fn parse_bare_float(rhs: &str) -> Option<f64> {
    let t = rhs.trim_start();
    let literal: String = t
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
        .collect();
    let literal = literal.trim_end_matches(['-', '+', 'e', 'E', '.']);
    let value: f64 = literal.parse().ok()?;
    let after = t[literal.len()..].trim_start();
    let boundary = after.is_empty()
        || after.starts_with(';')
        || after.starts_with(')')
        || after.starts_with(',')
        || after.starts_with('|')
        || after.starts_with('&')
        || after.starts_with('{');
    if boundary && value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn parse_inline_diff_thresholds(line: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(pos) = rest.find("<=") {
        let lhs_raw = &rest[..pos];
        let rhs_raw = &rest[pos + 2..];
        let lhs: String = lhs_raw
            .trim_end()
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if lhs.ends_with("diff")
            && let Some(value) = parse_bare_float(rhs_raw)
        {
            out.push((lhs, value));
        }
        rest = rhs_raw;
    }
    out
}

/// Collect every tolerance contract in one harness source file. Keys are the
/// const name, or `@<lhs>` for a bare inline comparison (disambiguated
/// `@<lhs>#2`, `#3`, … by first appearance when one lhs carries multiple
/// distinct values). Errors are hard: a `*_TOL` const whose value is not a
/// plain literal must not silently escape the ratchet.
fn extract_harness_tolerances(
    file: &str,
    src: &str,
) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let mut out = std::collections::BTreeMap::new();
    for (idx, line) in src.lines().enumerate() {
        let t = line.trim();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        if let Some(tail) = t.strip_prefix("const ") {
            match parse_tol_const(tail) {
                Some((name, value)) => {
                    if out
                        .insert(name.clone(), value)
                        .is_some_and(|prev| prev != value)
                    {
                        return Err(format!(
                            "{file}:{}: conflicting values for tolerance `{name}`",
                            idx + 1
                        ));
                    }
                }
                None => {
                    let name = tail.split(':').next().unwrap_or("").trim();
                    if name.contains("_TOL") && tail.contains(": f64") {
                        return Err(format!(
                            "{file}:{}: tolerance const `{name}` is not a plain literal; keep \
                             tolerance contracts in `const <NAME>_TOL: f64 = <literal>;` form",
                            idx + 1
                        ));
                    }
                }
            }
            continue;
        }
        for (lhs, value) in parse_inline_diff_thresholds(t) {
            let mut key = format!("@{lhs}");
            let mut k = 2;
            while out.get(&key).is_some_and(|&prev| prev != value) {
                key = format!("@{lhs}#{k}");
                k += 1;
            }
            out.entry(key).or_insert(value);
        }
    }
    Ok(out)
}

/// Compare the scanned contracts against the committed baseline. Tightenings
/// come back separately: they are never violations, only a nudge to re-run
/// with `--write-baseline`.
fn compare_harness(
    current: &HarnessBaseline,
    baseline: &HarnessBaseline,
) -> (Vec<HarnessViolation>, Vec<String>) {
    let mut violations = Vec::new();
    let mut tightened = Vec::new();
    for (file, tols) in current {
        let Some(base) = baseline.get(file) else {
            violations.push(HarnessViolation {
                file: file.clone(),
                name: String::new(),
                kind: "file_not_in_baseline",
                current: None,
                baseline: None,
            });
            continue;
        };
        for (name, value) in tols {
            match base.get(name) {
                None => violations.push(HarnessViolation {
                    file: file.clone(),
                    name: name.clone(),
                    kind: "not_in_baseline",
                    current: Some(*value),
                    baseline: None,
                }),
                Some(bv) if *value > *bv => violations.push(HarnessViolation {
                    file: file.clone(),
                    name: name.clone(),
                    kind: "loosened",
                    current: Some(*value),
                    baseline: Some(*bv),
                }),
                Some(bv) if *value < *bv => tightened.push(format!(
                    "{file}:{name} {bv:e} -> {value:e} (adopt with --write-baseline)"
                )),
                _ => {}
            }
        }
    }
    for file in baseline.keys() {
        if !current.contains_key(file) {
            violations.push(HarnessViolation {
                file: file.clone(),
                name: String::new(),
                kind: "file_removed",
                current: None,
                baseline: None,
            });
        }
    }
    (violations, tightened)
}

fn load_harness_baseline(path: &PathBuf) -> Result<Option<HarnessBaseline>, String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    serde_json::from_str(&content).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn write_harness_baseline(path: &PathBuf, baseline: &HarnessBaseline) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let pretty = serde_json::to_string_pretty(baseline).map_err(|e| e.to_string())?;
    fs::write(path, pretty + "\n").map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: tolerance_lint [OPTIONS]");
        println!();
        println!("Options:");
        println!(
            "  --fixtures-dir DIR     Override fixture directory (default: crates/fsci-conformance/fixtures)"
        );
        println!(
            "  --tests-dir DIR        Override diff-harness directory (default: {TESTS_DIR_REL})"
        );
        println!(
            "  --baseline-file FILE   Override harness baseline path (default: {HARNESS_BASELINE_REL})"
        );
        println!("  --write-baseline       Anchor/refresh the harness tolerance baseline");
        println!("  --json                 Emit JSON instead of text report");
        println!("  --max-violations N     Pass if fixture violation count <= N (default 0)");
        println!(
            "  --baseline N           Pass if fixture violation count <= N (alias for --max-violations)"
        );
        println!("  -h, --help             Show this help");
        return ExitCode::SUCCESS;
    }

    let fixtures_dir = args
        .iter()
        .position(|a| a == "--fixtures-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/fsci-conformance/fixtures"));
    let tests_dir = args
        .iter()
        .position(|a| a == "--tests-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(TESTS_DIR_REL));
    let baseline_path = args
        .iter()
        .position(|a| a == "--baseline-file")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(HARNESS_BASELINE_REL));
    let write_baseline = args.iter().any(|a| a == "--write-baseline");
    let emit_json = args.iter().any(|a| a == "--json");
    let max_violations = args
        .iter()
        .position(|a| a == "--max-violations" || a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let entries = match fs::read_dir(&fixtures_dir) {
        Ok(it) => it,
        Err(e) => {
            eprintln!("error: read {}: {e}", fixtures_dir.display());
            return ExitCode::from(2);
        }
    };

    let mut all = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("FSCI-P2C-") && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    for path in &paths {
        match lint_fixture(path) {
            Ok(vs) => all.extend(vs),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }

    // ── Harness tolerance ratchet ──
    let test_entries = match fs::read_dir(&tests_dir) {
        Ok(it) => it,
        Err(e) => {
            eprintln!("error: read {}: {e}", tests_dir.display());
            return ExitCode::from(2);
        }
    };
    let mut test_paths: Vec<PathBuf> = test_entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("diff_") && n.ends_with(".rs"))
                .unwrap_or(false)
        })
        .collect();
    test_paths.sort();

    let mut current: HarnessBaseline = std::collections::BTreeMap::new();
    for path in &test_paths {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: read {}: {e}", path.display());
                return ExitCode::from(2);
            }
        };
        match extract_harness_tolerances(&file_name, &src) {
            Ok(tols) => {
                current.insert(file_name, tols);
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }

    if write_baseline {
        let existing = match load_harness_baseline(&baseline_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        if let Some(existing) = &existing {
            let (violations, _) = compare_harness(&current, existing);
            let loosened: Vec<&HarnessViolation> =
                violations.iter().filter(|v| v.kind == "loosened").collect();
            if !loosened.is_empty() {
                for v in &loosened {
                    eprintln!(
                        "loosened: {}:{} {:e} -> {:e}",
                        v.file,
                        v.name,
                        v.baseline.unwrap_or_default(),
                        v.current.unwrap_or_default()
                    );
                }
                eprintln!(
                    "REFUSING to write baseline: {} tolerance(s) loosened. Loosening a \
                     tolerance contract requires hand-editing {} in the same commit so \
                     reviewers can grep for it.",
                    loosened.len(),
                    baseline_path.display()
                );
                return ExitCode::from(1);
            }
        }
        let total: usize = current.values().map(|t| t.len()).sum();
        if let Err(e) = write_harness_baseline(&baseline_path, &current) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
        println!(
            "wrote {}: {} files, {} tolerance contracts anchored",
            baseline_path.display(),
            current.len(),
            total
        );
        return ExitCode::SUCCESS;
    }

    let baseline = match load_harness_baseline(&baseline_path) {
        Ok(Some(b)) => b,
        Ok(None) => {
            eprintln!(
                "error: harness tolerance baseline {} is missing; run \
                 `tolerance_lint --write-baseline` once and commit the file",
                baseline_path.display()
            );
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let (harness_violations, tightened) = compare_harness(&current, &baseline);

    if emit_json {
        let payload = serde_json::json!({
            "total_violations": all.len(),
            "max_violations": max_violations,
            "violations": all,
            "harness_files_scanned": current.len(),
            "harness_violations": harness_violations,
            "harness_tightened": tightened,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).unwrap_or_default()
        );
    } else {
        print_text_report(&all);
        println!(
            "harness tolerances: {} diff files scanned against baseline",
            current.len()
        );
        for v in &harness_violations {
            match v.kind {
                "loosened" => println!(
                    "LOOSENED  {}:{} {:e} -> {:e}",
                    v.file,
                    v.name,
                    v.baseline.unwrap_or_default(),
                    v.current.unwrap_or_default()
                ),
                "not_in_baseline" => println!(
                    "UNANCHORED {}:{} = {:e} (add to baseline via --write-baseline)",
                    v.file,
                    v.name,
                    v.current.unwrap_or_default()
                ),
                "file_not_in_baseline" => {
                    println!("UNANCHORED FILE {} (anchor via --write-baseline)", v.file)
                }
                "file_removed" => println!(
                    "BASELINE STALE {} (file removed; prune via --write-baseline)",
                    v.file
                ),
                _ => println!("VIOLATION {}:{} unknown kind {}", v.file, v.name, v.kind),
            }
        }
        for t in &tightened {
            println!("tightened: {t}");
        }
    }

    if all.len() > max_violations {
        eprintln!(
            "FAIL: {} fixture violations > {} threshold",
            all.len(),
            max_violations
        );
        return ExitCode::from(1);
    }
    if !harness_violations.is_empty() {
        eprintln!(
            "FAIL: {} harness tolerance violation(s): loosening or unanchored contracts \
             in tests/diff_*.rs — see the report above",
            harness_violations.len()
        );
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod harness_ratchet_tests {
    use super::*;

    const SAMPLE: &str = "\
const PACKET_ID: &str = \"FSCI-P2C-006\";
const ABS_TOL: f64 = 1.0e-10;
pub const REL_TOL: f64 = 2.5e-12;
const ABS_TOL_CHI: f64 = 1.0e-12;
const MAX_CASES: usize = 44;
let pass = abs_diff <= 5.0e-6 || rel_diff <= 5.0e-6;
let geo = x[0] * x[0] + x[1] * x[1] <= 1.0;
let margin = diff <= 4.0 * se.max(1e-3);
let py = p <= 0.0 or ref_p <= 0.0;
";

    fn baseline_from(pairs: &[(&str, f64)]) -> HarnessBaseline {
        [(
            "diff_x.rs".to_string(),
            pairs
                .iter()
                .map(|(n, v)| (n.to_string(), *v))
                .collect::<std::collections::BTreeMap<_, _>>(),
        )]
        .into_iter()
        .collect()
    }

    #[test]
    fn extracts_consts_and_bare_inline_thresholds_only() {
        let m = extract_harness_tolerances("diff_x.rs", SAMPLE).unwrap();
        assert_eq!(m["ABS_TOL"], 1.0e-10);
        assert_eq!(m["REL_TOL"], 2.5e-12);
        assert_eq!(m["ABS_TOL_CHI"], 1.0e-12);
        // Non-tolerance consts, geometry predicates, composite expressions,
        // and foreign (oracle-script) lines stay out.
        assert!(
            !m.keys()
                .any(|k| !k.starts_with("REL_") && k.contains("MAX"))
        );
        assert_eq!(m["@abs_diff"], 5.0e-6);
        assert_eq!(m["@rel_diff"], 5.0e-6);
        assert_eq!(m.len(), 5);
    }

    #[test]
    fn loosening_is_a_violation_and_tightening_is_not() {
        let current = extract_harness_tolerances("diff_x.rs", SAMPLE).unwrap();
        let current = [("diff_x.rs".to_string(), current)]
            .into_iter()
            .collect::<HarnessBaseline>();
        // ABS_TOL loosened 1e-12 -> 1e-10; @abs_diff tightened 1e-4 -> 5e-6.
        let baseline = baseline_from(&[
            ("ABS_TOL", 1.0e-12),
            ("REL_TOL", 2.5e-12),
            ("ABS_TOL_CHI", 1.0e-12),
            ("@abs_diff", 1.0e-4),
            ("@rel_diff", 5.0e-6),
        ]);
        let (violations, tightened) = compare_harness(&current, &baseline);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].kind, "loosened");
        assert_eq!(violations[0].name, "ABS_TOL");
        assert_eq!(tightened.len(), 1);
        assert!(tightened[0].contains("@abs_diff"));
    }

    #[test]
    fn equal_values_produce_nothing() {
        let current = extract_harness_tolerances("diff_x.rs", SAMPLE).unwrap();
        let current = [("diff_x.rs".to_string(), current)]
            .into_iter()
            .collect::<HarnessBaseline>();
        let baseline = baseline_from(&[
            ("ABS_TOL", 1.0e-10),
            ("REL_TOL", 2.5e-12),
            ("ABS_TOL_CHI", 1.0e-12),
            ("@abs_diff", 5.0e-6),
            ("@rel_diff", 5.0e-6),
        ]);
        let (violations, tightened) = compare_harness(&current, &baseline);
        assert!(violations.is_empty());
        assert!(tightened.is_empty());
    }

    #[test]
    fn new_names_and_unanchored_files_violate() {
        let current = extract_harness_tolerances("diff_x.rs", SAMPLE).unwrap();
        let current = [("diff_x.rs".to_string(), current)]
            .into_iter()
            .collect::<HarnessBaseline>();
        // Baseline missing @abs_diff and the whole file diff_other.rs.
        let mut baseline = baseline_from(&[
            ("ABS_TOL", 1.0e-10),
            ("REL_TOL", 2.5e-12),
            ("ABS_TOL_CHI", 1.0e-12),
            ("@rel_diff", 5.0e-6),
        ]);
        baseline.insert("diff_other.rs".to_string(), Default::default());
        let (violations, _) = compare_harness(&current, &baseline);
        assert!(
            violations
                .iter()
                .any(|v| v.kind == "not_in_baseline" && v.name == "@abs_diff")
        );
        assert!(
            violations
                .iter()
                .any(|v| v.kind == "file_removed" && v.file == "diff_other.rs")
        );
        // And a current file absent from the baseline entirely:
        let empty: HarnessBaseline = Default::default();
        let (violations, _) = compare_harness(&current, &empty);
        assert_eq!(violations[0].kind, "file_not_in_baseline");
    }

    #[test]
    fn non_literal_tol_const_is_rejected_not_ignored() {
        let src = "const ABS_TOL: f64 = f64::EPSILON * 8.0;\n";
        let err = extract_harness_tolerances("diff_y.rs", src).unwrap_err();
        assert!(err.contains("not a plain literal"), "{err}");
    }

    #[test]
    fn duplicate_identical_consts_dedup_but_conflicting_ones_error() {
        let ok = "const ABS_TOL: f64 = 1.0e-10;\nconst ABS_TOL: f64 = 1.0e-10;\n";
        assert_eq!(
            extract_harness_tolerances("diff_z.rs", ok).unwrap()["ABS_TOL"],
            1.0e-10
        );
        let bad = "const ABS_TOL: f64 = 1.0e-10;\nconst ABS_TOL: f64 = 1.0e-9;\n";
        assert!(extract_harness_tolerances("diff_z.rs", bad).is_err());
    }
}
