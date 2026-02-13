use fsci_conformance::dashboard::{
    DashboardPanel, DashboardState, load_dashboard_state_from_artifact_root,
};
use fsci_conformance::{HarnessConfig, style_for_case_result, style_for_packet_summary};
use ftui::core::event::{Event, KeyCode, KeyEvent, KeyEventKind, Modifiers};
use ftui::layout::{Constraint, Flex, Rect};
use ftui::render::frame::Frame;
use ftui::runtime::{App, Cmd, Model, ScreenMode};
use ftui::widgets::Widget;
use ftui::widgets::paragraph::Paragraph;
use ftui::{PackedRgba, Style};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct CliArgs {
    artifact_root: PathBuf,
    packet_filter: Option<String>,
    inline_height: u16,
}

#[derive(Debug, Clone)]
enum CliParseError {
    Help,
    Message(String),
}

#[derive(Debug, Clone)]
struct DashboardApp {
    artifact_root: PathBuf,
    packet_filter: Option<String>,
    state: DashboardState,
    status_line: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DashboardMsg {
    Up,
    Down,
    Left,
    Right,
    NextPanel,
    PrevPanel,
    Reload,
    Quit,
    Noop,
}

impl From<Event> for DashboardMsg {
    fn from(event: Event) -> Self {
        match event {
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press,
            }) => match code {
                KeyCode::Char('c') if modifiers.contains(Modifiers::CTRL) => Self::Quit,
                KeyCode::Char('q') | KeyCode::Escape => Self::Quit,
                KeyCode::Up | KeyCode::Char('k') => Self::Up,
                KeyCode::Down | KeyCode::Char('j') => Self::Down,
                KeyCode::Left | KeyCode::Char('h') => Self::Left,
                KeyCode::Right | KeyCode::Char('l') => Self::Right,
                KeyCode::Tab => Self::NextPanel,
                KeyCode::BackTab => Self::PrevPanel,
                KeyCode::Char('r') => Self::Reload,
                _ => Self::Noop,
            },
            _ => Self::Noop,
        }
    }
}

impl Model for DashboardApp {
    type Message = DashboardMsg;

    fn update(&mut self, msg: Self::Message) -> Cmd<Self::Message> {
        match msg {
            DashboardMsg::Quit => return Cmd::Quit,
            DashboardMsg::Up => self.state.move_up(),
            DashboardMsg::Down => self.state.move_down(),
            DashboardMsg::Left | DashboardMsg::PrevPanel => self.state.prev_panel(),
            DashboardMsg::Right | DashboardMsg::NextPanel => self.state.next_panel(),
            DashboardMsg::Reload => {
                match load_dashboard_state_from_artifact_root(
                    &self.artifact_root,
                    self.packet_filter.clone(),
                ) {
                    Ok(next_state) => {
                        self.state = next_state;
                        self.status_line =
                            Some(format!("reloaded from {}", self.artifact_root.display()));
                    }
                    Err(error) => {
                        self.status_line = Some(format!("reload failed: {error}"));
                    }
                }
            }
            DashboardMsg::Noop => {}
        }
        Cmd::None
    }

    fn view(&self, frame: &mut Frame) {
        let area = Rect::from_size(frame.buffer.width(), frame.buffer.height());
        if area.is_empty() {
            return;
        }

        let rows = Flex::vertical()
            .constraints([
                Constraint::Fixed(2),
                Constraint::Min(0),
                Constraint::Fixed(2),
            ])
            .split(area);

        self.render_header(rows[0], frame);

        let cols = Flex::horizontal()
            .constraints([
                Constraint::Percentage(34.0),
                Constraint::Percentage(33.0),
                Constraint::Fill,
            ])
            .split(rows[1]);

        self.render_packets_panel(cols[0], frame);
        self.render_cases_panel(cols[1], frame);
        self.render_summary_panel(cols[2], frame);

        self.render_footer(rows[2], frame);
    }
}

impl DashboardApp {
    fn render_header(&self, area: Rect, frame: &mut Frame) {
        if area.is_empty() {
            return;
        }

        let metrics = self.state.metrics();
        let title = format!(
            "FrankenSciPy Conformance Dashboard | packets={} failed_packets={} failed_cases={}/{} ({:.2}%)",
            metrics.total_packets,
            metrics.failed_packets,
            metrics.failed_cases,
            metrics.total_cases,
            metrics.failed_case_pct
        );

        Paragraph::new(truncate_for_width(&title, area.width))
            .style(Style::new().fg(PackedRgba::rgb(125, 225, 250)).bold())
            .render(single_line(area, 0), frame);

        let subtitle = format!(
            "artifact_root={} filter={}",
            self.artifact_root.display(),
            self.packet_filter.as_deref().unwrap_or("<none>")
        );
        Paragraph::new(truncate_for_width(&subtitle, area.width))
            .style(Style::new().fg(PackedRgba::rgb(170, 170, 170)))
            .render(single_line(area, 1), frame);
    }

    fn render_packets_panel(&self, area: Rect, frame: &mut Frame) {
        let mut lines = Vec::new();
        let mut styles = Vec::new();

        let header_style = panel_header_style(self.state.active_panel() == DashboardPanel::Packets);
        lines.push(String::from("PACKETS"));
        styles.push(header_style);

        for (idx, summary) in self.state.summaries().iter().enumerate() {
            let prefix = if idx == self.state.selected_packet_index() {
                ">"
            } else {
                " "
            };
            let line = format!(
                "{prefix} {} pass={} fail={}",
                summary.packet_id, summary.passed_cases, summary.failed_cases
            );

            let mut style = style_for_packet_summary(summary);
            if idx == self.state.selected_packet_index() {
                style = style.bg(PackedRgba::rgb(25, 35, 60));
            }
            lines.push(line);
            styles.push(style);
        }

        if self.state.summaries().is_empty() {
            lines.push(String::from("  no packets found"));
            styles.push(Style::new().fg(PackedRgba::rgb(170, 170, 170)));
        }

        render_lines(area, frame, &lines, &styles);
    }

    fn render_cases_panel(&self, area: Rect, frame: &mut Frame) {
        let mut lines = Vec::new();
        let mut styles = Vec::new();

        let header_style = panel_header_style(self.state.active_panel() == DashboardPanel::Cases);
        lines.push(String::from("CASES"));
        styles.push(header_style);

        if let Some(packet) = self.state.selected_packet() {
            for (idx, case) in packet.case_results.iter().enumerate() {
                let prefix = if idx == self.state.selected_case_index() {
                    ">"
                } else {
                    " "
                };
                let status = if case.passed { "PASS" } else { "FAIL" };
                let line = format!("{prefix} {} [{}]", case.case_id, status);

                let mut style = style_for_case_result(case);
                if idx == self.state.selected_case_index() {
                    style = style.bg(PackedRgba::rgb(35, 35, 65));
                }
                lines.push(line);
                styles.push(style);
            }

            if packet.case_results.is_empty() {
                lines.push(String::from("  selected packet has no cases"));
                styles.push(Style::new().fg(PackedRgba::rgb(170, 170, 170)));
            }
        } else {
            lines.push(String::from("  no packet selected"));
            styles.push(Style::new().fg(PackedRgba::rgb(170, 170, 170)));
        }

        render_lines(area, frame, &lines, &styles);
    }

    fn render_summary_panel(&self, area: Rect, frame: &mut Frame) {
        let mut lines = Vec::new();
        let mut styles = Vec::new();

        let header_style = panel_header_style(self.state.active_panel() == DashboardPanel::Summary);
        lines.push(String::from("SUMMARY"));
        styles.push(header_style);

        let metrics = self.state.metrics();
        lines.push(format!(
            "failed_packets={}/{}",
            metrics.failed_packets, metrics.total_packets
        ));
        styles.push(Style::new().fg(PackedRgba::rgb(220, 220, 220)));

        lines.push(format!(
            "failed_cases={}/{} ({:.2}%)",
            metrics.failed_cases, metrics.total_cases, metrics.failed_case_pct
        ));
        styles.push(Style::new().fg(PackedRgba::rgb(220, 220, 220)));

        lines.push(format!(
            "sidecars={} decode_proofs={} oracle_captures={}",
            metrics.packets_with_sidecar,
            metrics.packets_with_decode_proof,
            metrics.packets_with_oracle_capture
        ));
        styles.push(Style::new().fg(PackedRgba::rgb(220, 220, 220)));

        if let Some(packet) = self.state.selected_packet() {
            lines.push(format!("packet={}", packet.packet_id));
            styles.push(Style::new().fg(PackedRgba::rgb(180, 230, 255)));

            if let Some(artifact) = self.state.selected_packet_artifacts() {
                lines.push(format!(
                    "artifacts: report={} sidecar={} decode={} oracle={} oracle_err={}",
                    bool_as_flag(artifact.has_report),
                    bool_as_flag(artifact.has_sidecar),
                    bool_as_flag(artifact.has_decode_proof),
                    bool_as_flag(artifact.has_oracle_capture),
                    bool_as_flag(artifact.has_oracle_error)
                ));
                styles.push(Style::new().fg(PackedRgba::rgb(150, 210, 180)));
            }
        }

        if let Some(case) = self.state.selected_case() {
            lines.push(format!("case={} passed={}", case.case_id, case.passed));
            styles.push(style_for_case_result(case));

            lines.push(format!("message={}", case.message));
            styles.push(Style::new().fg(PackedRgba::rgb(195, 195, 195)));
        }

        for warning in self.state.warnings().iter().take(3) {
            lines.push(format!("warning: {warning}"));
            styles.push(Style::new().fg(PackedRgba::rgb(250, 180, 75)));
        }

        render_lines(area, frame, &lines, &styles);
    }

    fn render_footer(&self, area: Rect, frame: &mut Frame) {
        if area.is_empty() {
            return;
        }

        let help = "keys: q/esc quit | j/k or arrows move | tab/backtab panels | r reload";
        Paragraph::new(truncate_for_width(help, area.width))
            .style(Style::new().fg(PackedRgba::rgb(160, 160, 160)))
            .render(single_line(area, 0), frame);

        let status = self.status_line.as_deref().unwrap_or("ready");
        Paragraph::new(truncate_for_width(status, area.width))
            .style(Style::new().fg(PackedRgba::rgb(120, 200, 160)))
            .render(single_line(area, 1), frame);
    }
}

fn render_lines(area: Rect, frame: &mut Frame, lines: &[String], styles: &[Style]) {
    if area.is_empty() {
        return;
    }

    for row in 0..area.height as usize {
        let Some(text) = lines.get(row) else {
            break;
        };
        let style = styles
            .get(row)
            .copied()
            .unwrap_or_else(|| Style::new().fg(PackedRgba::rgb(220, 220, 220)));
        let line_area = single_line(area, row as u16);
        Paragraph::new(truncate_for_width(text, line_area.width))
            .style(style)
            .render(line_area, frame);
    }
}

fn panel_header_style(active: bool) -> Style {
    if active {
        Style::new()
            .fg(PackedRgba::rgb(255, 230, 120))
            .bg(PackedRgba::rgb(55, 45, 20))
            .bold()
    } else {
        Style::new().fg(PackedRgba::rgb(160, 160, 160)).bold()
    }
}

fn bool_as_flag(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn single_line(area: Rect, row: u16) -> Rect {
    Rect::new(
        area.x,
        area.y.saturating_add(row),
        area.width,
        if row < area.height { 1 } else { 0 },
    )
}

fn truncate_for_width(input: &str, width: u16) -> String {
    if width == 0 {
        return String::new();
    }

    let max = width as usize;
    let count = input.chars().count();
    if count <= max {
        return input.to_owned();
    }

    if max <= 1 {
        return String::from("…");
    }

    let mut out = String::with_capacity(max);
    for ch in input.chars().take(max - 1) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, CliParseError> {
    let mut artifact_root: Option<PathBuf> = None;
    let mut packet_filter: Option<String> = None;
    let mut inline_height = 24u16;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--artifact-root" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err(CliParseError::Message(
                        "missing value for --artifact-root".to_owned(),
                    ));
                };
                artifact_root = Some(PathBuf::from(value));
            }
            "--packet" | "--packet-filter" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err(CliParseError::Message(
                        "missing value for --packet-filter".to_owned(),
                    ));
                };
                packet_filter = Some(value.clone());
            }
            "--inline-height" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err(CliParseError::Message(
                        "missing value for --inline-height".to_owned(),
                    ));
                };
                inline_height = value.parse::<u16>().map_err(|error| {
                    CliParseError::Message(format!(
                        "invalid --inline-height value `{value}`: {error}"
                    ))
                })?;
            }
            "-h" | "--help" => return Err(CliParseError::Help),
            unknown => {
                return Err(CliParseError::Message(format!(
                    "unknown argument `{unknown}`"
                )));
            }
        }
        index += 1;
    }

    let default_artifact_root = HarnessConfig::default_paths()
        .fixture_root
        .join("artifacts");

    Ok(CliArgs {
        artifact_root: artifact_root.unwrap_or(default_artifact_root),
        packet_filter,
        inline_height: inline_height.max(6),
    })
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} [--artifact-root <path>] [--packet-filter <query>] [--inline-height <rows>]"
    );
    eprintln!("  --artifact-root <path>  root containing packet artifact directories");
    eprintln!("  --packet-filter <query> case-insensitive packet/family filter");
    eprintln!("  --inline-height <rows>  inline TUI height (default 24)");
}

fn main() -> std::io::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "conformance_dashboard".to_owned());

    let args = match parse_cli_args(&argv[1..]) {
        Ok(args) => args,
        Err(CliParseError::Help) => {
            print_usage(&program);
            return Ok(());
        }
        Err(CliParseError::Message(message)) => {
            eprintln!("{message}");
            print_usage(&program);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                message,
            ));
        }
    };

    let state =
        load_dashboard_state_from_artifact_root(&args.artifact_root, args.packet_filter.clone())
            .map_err(|error| {
                std::io::Error::other(format!("failed to load dashboard data: {error}"))
            })?;

    let app = DashboardApp {
        artifact_root: args.artifact_root,
        packet_filter: args.packet_filter,
        state,
        status_line: Some("dashboard ready".to_owned()),
    };

    App::new(app)
        .screen_mode(ScreenMode::Inline {
            ui_height: args.inline_height,
        })
        .run()
}

#[cfg(test)]
mod tests {
    use super::{parse_cli_args, truncate_for_width};

    #[test]
    fn parse_cli_defaults_are_stable() {
        let args = parse_cli_args(&[]).expect("defaults parse");
        assert!(args.inline_height >= 6);
    }

    #[test]
    fn parse_cli_reads_overrides() {
        let args = parse_cli_args(&[
            "--artifact-root".to_owned(),
            "/tmp/x".to_owned(),
            "--packet-filter".to_owned(),
            "P2C-002".to_owned(),
            "--inline-height".to_owned(),
            "14".to_owned(),
        ])
        .expect("override parse");
        assert_eq!(args.artifact_root.to_string_lossy(), "/tmp/x");
        assert_eq!(args.packet_filter.as_deref(), Some("P2C-002"));
        assert_eq!(args.inline_height, 14);
    }

    #[test]
    fn truncate_preserves_short_and_ellipsizes_long() {
        assert_eq!(truncate_for_width("abc", 5), "abc");
        assert_eq!(truncate_for_width("abcdef", 4), "abc…");
    }
}
