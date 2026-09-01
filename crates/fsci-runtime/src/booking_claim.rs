#![forbid(unsafe_code)]

//! One canonical answer to "was this timed run actually serialised against the fleet?".
//!
//! # Why this exists
//!
//! The standing operating rule is that measurements are one-at-a-time fleet-wide: take the
//! agent-mail build slot before any timed run, because nine projects benchmarking at once took
//! this host to run queue 122 and every ratio measured in that window was a contention artifact.
//!
//! That rule names a tool that does not work. `acquire_build_slot` answers:
//!
//! ```text
//! Build slots are disabled. Enable WORKTREES_ENABLED to use this tool.
//! ```
//!
//! categorically, for every slot name (`frankenscipy-fr78g`). The fallback the campaign adopted
//! is an addressed, durably persisted agent-mail booking message, whose id each harness carries
//! in `TRJ_BOOKING_CLAIM_MESSAGE_ID` and prints into the row.
//!
//! # The defect this module closes
//!
//! Every harness that requires that variable validates it like this:
//!
//! ```text
//! let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
//! ```
//!
//! and `required_env` is, depending on the harness, "the variable is set" or "the variable is set
//! and not whitespace". Nothing resolves the id. Nothing checks that a message exists, that it
//! belongs to this project, that the measuring agent sent it, that it is a booking rather than an
//! unrelated automated notice, or that it is from this measurement rather than one last month.
//!
//! This is precisely the tautological-gate defect the campaign already found and fixed for
//! `BINARY_BUILD_ROUTE`, which hard-required the free-text string to contain `"rch"`, `"base"`,
//! `"clean-overlay"` and `"no-overlay"` -- a substring match on a string the operator types, which
//! cannot distinguish a genuine remote build from a local one. The same reasoning applies here
//! with more force, because the booking claim is not one provenance field among many: it is the
//! *substitute for the disabled build slot*, so it is the only evidence in the row that the run
//! was serialised at all.
//!
//! Two facts, both observed rather than argued:
//!
//! 1. `required_env` accepts any string. In `perf_spsolve` it accepts the empty string, because
//!    that copy is a bare `std::env::var`.
//! 2. The recorded claims are not auditable after the fact. The `2026-07-30` bdf and gmres rows
//!    cite `6983` and `7131`. This host's archive begins at `2026-08-27T20:27` with id `276` and
//!    holds no July messages at all, so `6983` now resolves to an unrelated automated
//!    `[ATC] acknowledgment requested` notice and `7131` resolves to nothing.
//!
//! Fact 2 is *not* evidence that those bookings were fabricated -- ids were renumbered by an
//! archive rebuild, so the original messages are simply gone. It is evidence of something else
//! that matters as much: a claim id recorded in a ledger row cannot be checked later, by anyone,
//! including the agent that wrote it. An unverifiable provenance field is worse than an absent one,
//! because it reads as evidence.
//!
//! # What replaces it
//!
//! The mechanism does not need a server feature flag, which is why it is implementable at all:
//! a booking is an ordinary agent-mail message, addressed to at least one other agent, whose
//! subject carries [`BOOKING_SUBJECT_MARKER`]. Agent-mail already persists every message to the
//! git mailbox archive as a file named `<ISO-timestamp>__<slug>__<id>.md` carrying a JSON header.
//! So the claim becomes checkable by construction, and [`BookingClaim::resolve_in`] checks it:
//!
//! * the id parses and resolves to a message in **this project's** archive,
//! * the message is a booking (subject marker) and is **addressed** to someone,
//! * the **measuring agent sent it** -- a booking someone else made is not yours,
//! * it was created within [`DEFAULT_MAX_AGE_SECS`] of the run, so a stale id cannot be reused.
//!
//! [`BookingClaim::fleet_conflicts_in`] then restores the part of `acquire_build_slot` that
//! actually did the work: it scans the *sibling project archives* for other live bookings in the
//! same window, so a harness can report -- or refuse on -- another project measuring concurrently.
//! That is the contention the rule exists to prevent, and nothing in the repository has been
//! looking for it since the slot was disabled.
//!
//! This module verifies claims; it deliberately does not create them, so that the act of booking
//! stays an ordinary addressed message a human can read in the mailbox.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Environment variable each timed harness carries its booking id in.
pub const CLAIM_ENV: &str = "TRJ_BOOKING_CLAIM_MESSAGE_ID";

/// Subject marker that distinguishes a measurement booking from ordinary mail.
///
/// Matched case-insensitively so an agent writing `[measurement booking]` still books.
pub const BOOKING_SUBJECT_MARKER: &str = "[MEASUREMENT BOOKING]";

/// How recent a booking must be to cover the run that cites it.
///
/// Six hours is long enough for a slow remote build plus a many-round paired schedule, and short
/// enough that last week's id cannot be pasted into today's row.
pub const DEFAULT_MAX_AGE_SECS: u64 = 6 * 60 * 60;

/// Default archive root, matching where agent-mail persists its git mailbox.
const DEFAULT_ARCHIVE_ENV: &str = "MCP_AGENT_MAIL_ARCHIVE_ROOT";
const DEFAULT_ARCHIVE_SUFFIX: &str = ".mcp_agent_mail_git_mailbox_repo";

/// A booking claim that resolved to a real, current, self-sent booking message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookingClaim {
    pub message_id: u64,
    pub subject: String,
    pub from: String,
    pub to: Vec<String>,
    pub project: String,
    pub created: String,
    pub age_seconds: u64,
    pub path: PathBuf,
}

impl BookingClaim {
    /// One line naming everything a later auditor needs to re-resolve this claim by hand.
    pub fn provenance_line(&self) -> String {
        format!(
            "trj_booking_claim: message_id={} verified=true from={} to={} created={} \
             age_seconds={} subject={:?}",
            self.message_id,
            self.from,
            self.to.join(","),
            self.created,
            self.age_seconds,
            self.subject,
        )
    }

    /// Resolve and verify the claim `id` for `agent_name` in `project_slug`'s archive.
    ///
    /// `now_unix` is passed rather than read so the freshness rule is testable; `max_age_secs`
    /// bounds how long before the run the booking may have been made.
    pub fn resolve_in(
        archive_root: &Path,
        project_slug: &str,
        agent_name: &str,
        id: u64,
        now_unix: i64,
        max_age_secs: u64,
    ) -> Result<Self, ClaimRejection> {
        let messages = archive_root
            .join("projects")
            .join(project_slug)
            .join("messages");
        if !messages.is_dir() {
            return Err(ClaimRejection::NoArchive {
                path: messages.clone(),
            });
        }

        let path = find_message_file(&messages, id).ok_or(ClaimRejection::Unresolvable {
            id,
            searched: messages.clone(),
        })?;

        let raw = std::fs::read_to_string(&path).map_err(|error| ClaimRejection::Unreadable {
            path: path.clone(),
            detail: error.to_string(),
        })?;
        let header = parse_json_header(&raw).ok_or_else(|| ClaimRejection::Unreadable {
            path: path.clone(),
            detail: "message has no leading ---json header".to_string(),
        })?;

        let subject = header_string(&header, "subject").unwrap_or_default();
        let from = header_string(&header, "from").unwrap_or_default();
        let project = header_string(&header, "project").unwrap_or_default();
        let created = header_string(&header, "created").unwrap_or_default();
        let to = header_strings(&header, "to");

        if !subject
            .to_ascii_uppercase()
            .contains(BOOKING_SUBJECT_MARKER)
        {
            return Err(ClaimRejection::NotABooking { id, subject });
        }
        if to.is_empty() {
            return Err(ClaimRejection::Unaddressed { id, subject });
        }
        if from != agent_name {
            return Err(ClaimRejection::NotSentByAgent {
                id,
                sender: from,
                agent: agent_name.to_string(),
            });
        }

        let created_unix =
            parse_iso8601_unix(&created).ok_or_else(|| ClaimRejection::Unreadable {
                path: path.clone(),
                detail: format!("unparseable created timestamp {created:?}"),
            })?;
        if created_unix > now_unix {
            return Err(ClaimRejection::FromTheFuture { id, created });
        }
        let age_seconds = (now_unix - created_unix) as u64;
        if age_seconds > max_age_secs {
            return Err(ClaimRejection::Stale {
                id,
                age_seconds,
                max_age_secs,
            });
        }

        Ok(Self {
            message_id: id,
            subject,
            from,
            to,
            project,
            created,
            age_seconds,
            path,
        })
    }

    /// Other projects' live bookings overlapping this one's window, newest first.
    ///
    /// This is the fleet-wide serialisation `acquire_build_slot` used to provide. An empty result
    /// is the quiet case; a non-empty one names who else believes they are measuring right now.
    pub fn fleet_conflicts_in(
        archive_root: &Path,
        own_project_slug: &str,
        now_unix: i64,
        max_age_secs: u64,
    ) -> Vec<FleetBooking> {
        let projects = archive_root.join("projects");
        let Ok(entries) = std::fs::read_dir(&projects) else {
            return Vec::new();
        };

        let mut conflicts = Vec::new();
        for entry in entries.flatten() {
            let slug = entry.file_name().to_string_lossy().into_owned();
            if slug == own_project_slug || !entry.path().is_dir() {
                continue;
            }
            for path in recent_message_files(&entry.path().join("messages")) {
                let Ok(raw) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let Some(header) = parse_json_header(&raw) else {
                    continue;
                };
                let subject = header_string(&header, "subject").unwrap_or_default();
                if !subject
                    .to_ascii_uppercase()
                    .contains(BOOKING_SUBJECT_MARKER)
                {
                    continue;
                }
                let created = header_string(&header, "created").unwrap_or_default();
                let Some(created_unix) = parse_iso8601_unix(&created) else {
                    continue;
                };
                if created_unix > now_unix {
                    continue;
                }
                let age_seconds = (now_unix - created_unix) as u64;
                if age_seconds > max_age_secs {
                    continue;
                }
                conflicts.push(FleetBooking {
                    project_slug: slug.clone(),
                    agent: header_string(&header, "from").unwrap_or_default(),
                    subject,
                    created,
                    age_seconds,
                });
            }
        }
        conflicts.sort_by_key(|booking| booking.age_seconds);
        conflicts
    }
}

/// A live booking held by another project on this fleet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FleetBooking {
    pub project_slug: String,
    pub agent: String,
    pub subject: String,
    pub created: String,
    pub age_seconds: u64,
}

/// Why a booking claim was refused. Every variant names what was checked and what was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimRejection {
    Missing,
    MissingAgent,
    Malformed {
        value: String,
    },
    NoArchive {
        path: PathBuf,
    },
    Unresolvable {
        id: u64,
        searched: PathBuf,
    },
    Unreadable {
        path: PathBuf,
        detail: String,
    },
    NotABooking {
        id: u64,
        subject: String,
    },
    Unaddressed {
        id: u64,
        subject: String,
    },
    NotSentByAgent {
        id: u64,
        sender: String,
        agent: String,
    },
    FromTheFuture {
        id: u64,
        created: String,
    },
    Stale {
        id: u64,
        age_seconds: u64,
        max_age_secs: u64,
    },
}

impl std::fmt::Display for ClaimRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAgent => write!(
                formatter,
                "AGENT_NAME is required: a booking is verified against the agent that sent it"
            ),
            Self::Missing => write!(
                formatter,
                "{CLAIM_ENV} is required: send an addressed agent-mail message whose subject \
                 contains {BOOKING_SUBJECT_MARKER} and pass its id"
            ),
            Self::Malformed { value } => write!(
                formatter,
                "{CLAIM_ENV}={value:?} is not an agent-mail message id"
            ),
            Self::NoArchive { path } => write!(
                formatter,
                "no agent-mail archive for this project at {}; the booking cannot be verified, \
                 so no timed row may be taken",
                path.display()
            ),
            Self::Unresolvable { id, searched } => write!(
                formatter,
                "booking claim {id} resolves to no message under {}; a claim that cannot be \
                 resolved is not evidence",
                searched.display()
            ),
            Self::Unreadable { path, detail } => write!(
                formatter,
                "booking claim message {} could not be read: {detail}",
                path.display()
            ),
            Self::NotABooking { id, subject } => write!(
                formatter,
                "message {id} is not a measurement booking: subject {subject:?} does not contain \
                 {BOOKING_SUBJECT_MARKER}"
            ),
            Self::Unaddressed { id, subject } => write!(
                formatter,
                "booking {id} ({subject:?}) is addressed to nobody; a booking no other agent \
                 receives serialises nothing"
            ),
            Self::NotSentByAgent { id, sender, agent } => write!(
                formatter,
                "booking {id} was sent by {sender:?}, not by the measuring agent {agent:?}"
            ),
            Self::FromTheFuture { id, created } => write!(
                formatter,
                "booking {id} is dated {created}, which is after this run started"
            ),
            Self::Stale {
                id,
                age_seconds,
                max_age_secs,
            } => write!(
                formatter,
                "booking {id} is {age_seconds}s old, beyond the {max_age_secs}s window; book the \
                 run you are actually taking"
            ),
        }
    }
}

impl std::error::Error for ClaimRejection {}

/// The archive root agent-mail persists to, honouring an explicit override.
pub fn default_archive_root() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var(DEFAULT_ARCHIVE_ENV)
        && !explicit.trim().is_empty()
    {
        return Some(PathBuf::from(explicit));
    }
    std::env::var("HOME")
        .ok()
        .map(|home| PathBuf::from(home).join(DEFAULT_ARCHIVE_SUFFIX))
}

/// Agent-mail's project slug for a project key such as `/data/projects/frankenscipy`.
pub fn project_slug(project_key: &str) -> String {
    project_key
        .trim_matches('/')
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

/// The project every harness in this workspace books against.
pub const PROJECT_KEY: &str = "/data/projects/frankenscipy";

/// The one call a timed harness makes: verify this run's booking, or refuse.
///
/// The measuring agent comes from `AGENT_NAME` because a booking is only evidence if it names
/// who made it -- "some booking exists" was the old gate, and it is what let any string pass.
pub fn verify_for_harness() -> Result<BookingClaim, ClaimRejection> {
    let agent = std::env::var("AGENT_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or(ClaimRejection::MissingAgent)?;
    verify_from_env(PROJECT_KEY, agent.trim())
}

/// Other projects' live bookings right now, for a harness to report into its row.
///
/// Reported rather than fatal: refusing here would let one project's stale booking block the
/// fleet, which is a policy change for other repositories to make, not a check to smuggle in.
pub fn fleet_conflicts_now() -> Vec<FleetBooking> {
    let Some(root) = default_archive_root() else {
        return Vec::new();
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);
    BookingClaim::fleet_conflicts_in(&root, &project_slug(PROJECT_KEY), now, DEFAULT_MAX_AGE_SECS)
}

/// Read [`CLAIM_ENV`] and verify it, against the real archive and the current clock.
pub fn verify_from_env(
    project_key: &str,
    agent_name: &str,
) -> Result<BookingClaim, ClaimRejection> {
    let raw = std::env::var(CLAIM_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or(ClaimRejection::Missing)?;
    let id = raw
        .trim()
        .parse::<u64>()
        .map_err(|_| ClaimRejection::Malformed { value: raw.clone() })?;
    let root = default_archive_root().ok_or(ClaimRejection::NoArchive {
        path: PathBuf::from(DEFAULT_ARCHIVE_SUFFIX),
    })?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);
    BookingClaim::resolve_in(
        &root,
        &project_slug(project_key),
        agent_name,
        id,
        now,
        DEFAULT_MAX_AGE_SECS,
    )
}

// ── archive plumbing ──────────────────────────────────────────────────────────────────────────

/// Locate `*__<id>.md` under a `messages/` tree, which agent-mail nests as `<year>/<month>/`.
fn find_message_file(messages: &Path, id: u64) -> Option<PathBuf> {
    let needle = format!("__{id}.md");
    for year in read_dirs(messages) {
        for month in read_dirs(&year) {
            let Ok(entries) = std::fs::read_dir(&month) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(&needle))
                {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Every message file in a `messages/` tree. Bookings are rare, so this stays a plain walk.
fn recent_message_files(messages: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for year in read_dirs(messages) {
        for month in read_dirs(&year) {
            let Ok(entries) = std::fs::read_dir(&month) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                    files.push(path);
                }
            }
        }
    }
    files
}

fn read_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}

/// Extract the leading `---json { .. } ---` header agent-mail writes at the top of each message.
fn parse_json_header(raw: &str) -> Option<BTreeMap<String, serde_json::Value>> {
    let body = raw
        .strip_prefix("---json")?
        .trim_start_matches(['\r', '\n']);
    let end = body.find("\n---")?;
    serde_json::from_str(&body[..end]).ok()
}

fn header_string(header: &BTreeMap<String, serde_json::Value>, key: &str) -> Option<String> {
    header
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn header_strings(header: &BTreeMap<String, serde_json::Value>, key: &str) -> Vec<String> {
    header
        .get(key)
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.fraction][Z|+00:00]` to a Unix timestamp.
///
/// Agent-mail writes UTC, and a dependency on a date crate would be a poor trade for one format.
fn parse_iso8601_unix(value: &str) -> Option<i64> {
    let bytes = value.as_bytes();
    if bytes.len() < 19 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return None;
    }
    let year: i64 = value.get(0..4)?.parse().ok()?;
    let month: i64 = value.get(5..7)?.parse().ok()?;
    let day: i64 = value.get(8..10)?.parse().ok()?;
    let hour: i64 = value.get(11..13)?.parse().ok()?;
    let minute: i64 = value.get(14..16)?.parse().ok()?;
    let second: i64 = value.get(17..19)?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

/// Howard Hinnant's `days_from_civil`: civil date to days since 1970-01-01.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch archive. Kept hand-rolled so this crate gains no dev-dependency.
    struct Archive {
        root: PathBuf,
    }

    impl Archive {
        fn new(label: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0);
            let root = std::env::temp_dir().join(format!("fsci-booking-{label}-{unique}"));
            std::fs::create_dir_all(&root).expect("scratch archive");
            Self { root }
        }

        /// Write a message exactly as agent-mail persists one.
        fn write(
            &self,
            slug: &str,
            id: u64,
            from: &str,
            to: &[&str],
            subject: &str,
            created: &str,
        ) {
            let dir = self
                .root
                .join("projects")
                .join(slug)
                .join("messages")
                .join(&created[0..4])
                .join(&created[5..7]);
            std::fs::create_dir_all(&dir).expect("message dir");
            let recipients = to
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let header = format!(
                "---json\n{{\n  \"id\": {id},\n  \"from\": \"{from}\",\n  \"to\": [{recipients}],\n  \
                 \"subject\": \"{subject}\",\n  \"created\": \"{created}\",\n  \
                 \"project_slug\": \"{slug}\",\n  \"project\": \"/data/projects/frankenscipy\"\n}}\n---\n\nbody\n"
            );
            std::fs::write(dir.join(format!("{created}__slug__{id}.md")), header).expect("write");
        }
    }

    impl Drop for Archive {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    const SLUG: &str = "data-projects-frankenscipy";
    const AGENT: &str = "BlackThrush";
    /// 2026-09-01T12:00:00Z, the reference "now" every case below is judged against.
    const NOW: i64 = 1_788_264_000;

    fn resolve(archive: &Archive, id: u64) -> Result<BookingClaim, ClaimRejection> {
        BookingClaim::resolve_in(&archive.root, SLUG, AGENT, id, NOW, DEFAULT_MAX_AGE_SECS)
    }

    /// MUST-HIT arm: a real, current, self-sent, addressed booking is accepted.
    #[test]
    fn genuine_booking_verifies() {
        let archive = Archive::new("hit");
        archive.write(
            SLUG,
            9001,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] spsolve periodic cuboid",
            "2026-09-01T11-30-00Z",
        );
        let claim = resolve(&archive, 9001).expect("genuine booking must verify");
        assert_eq!(claim.message_id, 9001);
        assert_eq!(claim.from, AGENT);
        assert_eq!(claim.to, vec!["SilverRiver".to_string()]);
        assert_eq!(claim.age_seconds, 1_800);
        assert!(claim.provenance_line().contains("verified=true"));
    }

    /// The marker is matched case-insensitively, so lower-case booking subjects still work.
    #[test]
    fn lowercase_marker_still_books() {
        let archive = Archive::new("lower");
        archive.write(
            SLUG,
            9002,
            AGENT,
            &["SilverRiver"],
            "[measurement booking] eigh sweep",
            "2026-09-01T11-59-00Z",
        );
        assert!(resolve(&archive, 9002).is_ok());
    }

    /// MUST-MISS: the exact failure the old gate could not see -- an id naming no message.
    #[test]
    fn fabricated_id_is_refused() {
        let archive = Archive::new("miss-absent");
        archive.write(
            SLUG,
            9001,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] real one",
            "2026-09-01T11-30-00Z",
        );
        assert!(matches!(
            resolve(&archive, 6983),
            Err(ClaimRejection::Unresolvable { id: 6983, .. })
        ));
    }

    /// MUST-MISS: a real message that is not a booking. This is the `6983` case exactly.
    #[test]
    fn unrelated_message_is_refused() {
        let archive = Archive::new("miss-kind");
        archive.write(
            SLUG,
            6983,
            AGENT,
            &["GentleCoast"],
            "[ATC] acknowledgment requested from GentleCoast",
            "2026-09-01T11-30-00Z",
        );
        assert!(matches!(
            resolve(&archive, 6983),
            Err(ClaimRejection::NotABooking { id: 6983, .. })
        ));
    }

    /// MUST-MISS: someone else's booking is not yours.
    #[test]
    fn another_agents_booking_is_refused() {
        let archive = Archive::new("miss-agent");
        archive.write(
            SLUG,
            9003,
            "SilverRiver",
            &[AGENT],
            "[MEASUREMENT BOOKING] their run",
            "2026-09-01T11-30-00Z",
        );
        assert!(matches!(
            resolve(&archive, 9003),
            Err(ClaimRejection::NotSentByAgent { id: 9003, .. })
        ));
    }

    /// MUST-MISS: a booking addressed to nobody serialises nothing.
    #[test]
    fn unaddressed_booking_is_refused() {
        let archive = Archive::new("miss-unaddressed");
        archive.write(
            SLUG,
            9004,
            AGENT,
            &[],
            "[MEASUREMENT BOOKING] talking to myself",
            "2026-09-01T11-30-00Z",
        );
        assert!(matches!(
            resolve(&archive, 9004),
            Err(ClaimRejection::Unaddressed { id: 9004, .. })
        ));
    }

    /// MUST-MISS: last week's id cannot cover today's run.
    #[test]
    fn stale_booking_is_refused() {
        let archive = Archive::new("miss-stale");
        archive.write(
            SLUG,
            9005,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] yesterday",
            "2026-08-30T11-30-00Z",
        );
        assert!(matches!(
            resolve(&archive, 9005),
            Err(ClaimRejection::Stale { id: 9005, .. })
        ));
    }

    /// The freshness window is a boundary, so both sides of it are pinned.
    #[test]
    fn freshness_window_boundary_is_exact() {
        let archive = Archive::new("boundary");
        archive.write(
            SLUG,
            9006,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] edge",
            "2026-09-01T06-00-00Z",
        );
        let inside =
            BookingClaim::resolve_in(&archive.root, SLUG, AGENT, 9006, NOW, DEFAULT_MAX_AGE_SECS);
        assert_eq!(inside.expect("exactly at the window").age_seconds, 21_600);

        let outside = BookingClaim::resolve_in(
            &archive.root,
            SLUG,
            AGENT,
            9006,
            NOW,
            DEFAULT_MAX_AGE_SECS - 1,
        );
        assert!(matches!(outside, Err(ClaimRejection::Stale { .. })));
    }

    /// MUST-MISS: a booking filed under another project does not cover a run here.
    ///
    /// This project's archive exists and is searched -- otherwise the case would pass for the
    /// uninteresting reason that there was nothing to search.
    #[test]
    fn foreign_project_booking_is_refused() {
        let archive = Archive::new("miss-project");
        archive.write(
            SLUG,
            9008,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] ours, a different run",
            "2026-09-01T11-30-00Z",
        );
        archive.write(
            "data-projects-frankentorch",
            9007,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] other project",
            "2026-09-01T11-30-00Z",
        );
        assert!(resolve(&archive, 9008).is_ok(), "our own archive resolves");
        assert!(matches!(
            resolve(&archive, 9007),
            Err(ClaimRejection::Unresolvable { id: 9007, .. })
        ));
    }

    /// A missing archive refuses rather than passing vacuously.
    #[test]
    fn absent_archive_refuses() {
        let archive = Archive::new("no-archive");
        assert!(matches!(
            resolve(&archive, 9001),
            Err(ClaimRejection::NoArchive { .. })
        ));
    }

    /// Fleet conflicts: MUST-HIT on another project measuring, MUST-MISS on our own booking.
    #[test]
    fn fleet_conflicts_see_other_projects_only() {
        let archive = Archive::new("fleet");
        archive.write(
            SLUG,
            9010,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] ours",
            "2026-09-01T11-30-00Z",
        );
        archive.write(
            "data-projects-frankentorch",
            9011,
            "GentleCoast",
            &["BlackThrush"],
            "[MEASUREMENT BOOKING] gauntlet lane sweep",
            "2026-09-01T11-45-00Z",
        );
        archive.write(
            "data-projects-frankenfs",
            9012,
            "PinkOak",
            &["BlackThrush"],
            "ordinary status mail",
            "2026-09-01T11-50-00Z",
        );

        let conflicts =
            BookingClaim::fleet_conflicts_in(&archive.root, SLUG, NOW, DEFAULT_MAX_AGE_SECS);
        assert_eq!(
            conflicts.len(),
            1,
            "only the other project's BOOKING counts"
        );
        assert_eq!(conflicts[0].project_slug, "data-projects-frankentorch");
        assert_eq!(conflicts[0].agent, "GentleCoast");
    }

    /// A quiet fleet reports no conflicts -- the negative arm of the check above.
    #[test]
    fn quiet_fleet_has_no_conflicts() {
        let archive = Archive::new("quiet");
        archive.write(
            SLUG,
            9013,
            AGENT,
            &["SilverRiver"],
            "[MEASUREMENT BOOKING] ours",
            "2026-09-01T11-30-00Z",
        );
        archive.write(
            "data-projects-frankentorch",
            9014,
            "GentleCoast",
            &["BlackThrush"],
            "[MEASUREMENT BOOKING] last week",
            "2026-08-25T11-45-00Z",
        );
        assert!(
            BookingClaim::fleet_conflicts_in(&archive.root, SLUG, NOW, DEFAULT_MAX_AGE_SECS)
                .is_empty()
        );
    }

    /// Real bytes, both arms: agent-mail's own output, not a fixture I shaped to parse.
    ///
    /// The synthetic cases above prove the predicates. They cannot prove this module reads what
    /// agent-mail actually writes -- a parser and a hand-written fixture can agree with each
    /// other and both be wrong about production. These two files were copied verbatim out of
    /// this host's archive:
    ///
    /// * `40984` -- a genuine booking this agent sent (MUST-HIT),
    /// * `6983` -- the genuine automated `[ATC] acknowledgment requested` notice that the
    ///   `2026-07-30` bdf row's claim id now lands on (MUST-MISS).
    ///
    /// The second is the case that matters: under the old gate, `TRJ_BOOKING_CLAIM_MESSAGE_ID`
    /// naming this message was indistinguishable from naming a booking.
    #[test]
    fn real_agent_mail_bytes_both_arms() {
        const BOOKING: &str = include_str!("../tests/fixtures/agent_mail/booking__40984.md");
        const NOT_A_BOOKING: &str =
            include_str!("../tests/fixtures/agent_mail/not-a-booking__6983.md");

        let archive = Archive::new("real-bytes");
        let dir = archive
            .root
            .join("projects")
            .join(SLUG)
            .join("messages")
            .join("2026")
            .join("09");
        std::fs::create_dir_all(&dir).expect("message dir");
        std::fs::write(dir.join("real__40984.md"), BOOKING).expect("booking");
        std::fs::write(dir.join("real__6983.md"), NOT_A_BOOKING).expect("notice");

        // Judged one minute after the booking was sent, so the case never ages out.
        let header = parse_json_header(BOOKING).expect("agent-mail writes a ---json header");
        let created = header_string(&header, "created").expect("created");
        let now = parse_iso8601_unix(&created).expect("agent-mail timestamp parses") + 60;

        // MUST-HIT: the real booking verifies, and its real recipients survive the round trip.
        let claim =
            BookingClaim::resolve_in(&archive.root, SLUG, AGENT, 40984, now, DEFAULT_MAX_AGE_SECS)
                .expect("a genuine self-sent booking must verify");
        assert_eq!(claim.from, AGENT);
        assert_eq!(claim.age_seconds, 60);
        assert!(claim.to.contains(&"SilverRiver".to_string()));
        assert!(claim.subject.starts_with(BOOKING_SUBJECT_MARKER));

        // MUST-MISS: the real non-booking is refused for being the wrong kind of message.
        let notice =
            BookingClaim::resolve_in(&archive.root, SLUG, AGENT, 6983, now, DEFAULT_MAX_AGE_SECS);
        assert!(
            matches!(notice, Err(ClaimRejection::NotABooking { id: 6983, .. })),
            "an ordinary ATC notice must not pass as a booking, got {notice:?}"
        );

        // And it is refused for the right reason rather than by failing to parse at all.
        let notice_header =
            parse_json_header(NOT_A_BOOKING).expect("the notice parses as a real message");
        assert_eq!(
            header_string(&notice_header, "from").as_deref(),
            Some("AirTrafficControl")
        );
    }

    /// Live control against the REAL agent-mail archive, both arms, in one test.
    ///
    /// Synthetic fixtures prove the predicates; they cannot prove this module parses what
    /// agent-mail actually writes. This case does, and is env-gated rather than `#[ignore]`d
    /// because ignored diagnostics rot silently. It is LOCAL-ONLY: the archive lives under
    /// `$HOME` on `thinkstation1` and is not synced to an rch worker, so on a worker there is
    /// no archive and the case declines rather than pretending to have checked something.
    ///
    /// ```text
    /// FSCI_BOOKING_LIVE_CLAIM=<id of a booking you sent> \
    /// FSCI_BOOKING_LIVE_AGENT=<your agent name> \
    ///   cargo test -p fsci-runtime --lib live_archive_two_arm_control -- --nocapture
    /// ```
    #[test]
    fn live_archive_two_arm_control() {
        let (Ok(raw_id), Ok(agent)) = (
            std::env::var("FSCI_BOOKING_LIVE_CLAIM"),
            std::env::var("FSCI_BOOKING_LIVE_AGENT"),
        ) else {
            eprintln!("live control declined: set FSCI_BOOKING_LIVE_CLAIM and _AGENT");
            return;
        };
        let id: u64 = raw_id.trim().parse().expect("claim id must be numeric");
        let root = default_archive_root().expect("archive root");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs() as i64;

        // MUST-HIT: the real booking, sent by this agent, verifies against the real archive.
        let claim = BookingClaim::resolve_in(&root, SLUG, &agent, id, now, DEFAULT_MAX_AGE_SECS)
            .expect("a real self-sent booking must verify");
        assert_eq!(claim.message_id, id);
        assert_eq!(claim.from, agent);
        assert!(!claim.to.is_empty());
        eprintln!("live MUST-HIT: {}", claim.provenance_line());

        // MUST-MISS: the same real message, claimed by an agent who did not send it.
        let impostor =
            BookingClaim::resolve_in(&root, SLUG, "NotTheSender", id, now, DEFAULT_MAX_AGE_SECS);
        assert!(
            matches!(impostor, Err(ClaimRejection::NotSentByAgent { .. })),
            "another agent's booking must not verify, got {impostor:?}"
        );

        // MUST-MISS: an id far past the archive's high-water mark resolves to nothing.
        let absent =
            BookingClaim::resolve_in(&root, SLUG, &agent, u64::MAX, now, DEFAULT_MAX_AGE_SECS);
        assert!(
            matches!(absent, Err(ClaimRejection::Unresolvable { .. })),
            "a fabricated id must not verify, got {absent:?}"
        );
        eprintln!("live MUST-MISS arms both refused");
    }

    #[test]
    fn project_slug_matches_agent_mail() {
        assert_eq!(
            project_slug("/data/projects/frankenscipy"),
            "data-projects-frankenscipy"
        );
        assert_eq!(
            project_slug("/data/projects/mcp_agent_mail_rust"),
            "data-projects-mcp-agent-mail-rust"
        );
    }

    #[test]
    fn iso8601_parses_against_known_instants() {
        assert_eq!(parse_iso8601_unix("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_iso8601_unix("2026-09-01T12:00:00Z"), Some(NOW));
        // The archive's own dashed spelling, and a fractional-second stamp.
        assert_eq!(parse_iso8601_unix("2026-09-01T12-00-00Z"), Some(NOW));
        assert_eq!(
            parse_iso8601_unix("2026-08-28T09:18:33.952936Z"),
            Some(1_787_908_713)
        );
        assert_eq!(parse_iso8601_unix("not-a-timestamp"), None);
        assert_eq!(parse_iso8601_unix("2026-13-01T00:00:00Z"), None);
    }
}
