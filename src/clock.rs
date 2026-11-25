//! The machine clock for one model call.
//!
//! The line is volatile, so a later call in the same turn reads the clock again.

use jiff::Timestamp;
use jiff::tz::TimeZone;

use crate::prompt::PromptBlocks;

pub const CLOCK_PREFIX: &str = "Current time:";

/// Format `now` in `zone` and in UTC.
///
/// The line tells the model that this is the machine clock.
pub fn clock_line(now: Timestamp, zone: &TimeZone) -> String {
    let local = now.to_zoned(zone.clone());
    let zone_name = zone.iana_name().unwrap_or("local");
    format!(
        "{CLOCK_PREFIX} {} {zone_name} (UTC{}). UTC: {}. This is the machine clock now. Use it for the time and the date.",
        local.strftime("%Y-%m-%d %H:%M:%S"),
        local.offset(),
        now.strftime("%Y-%m-%d %H:%M:%S"),
    )
}

/// Replace a clock line with the clock at this call.
pub fn refresh_clock(blocks: &PromptBlocks) -> PromptBlocks {
    let line = clock_line(Timestamp::now(), &TimeZone::system());
    PromptBlocks {
        cached: blocks.cached.clone(),
        volatile: blocks
            .volatile
            .iter()
            .map(|text| {
                if text.starts_with(CLOCK_PREFIX) {
                    line.clone()
                } else {
                    text.clone()
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{CLOCK_PREFIX, clock_line, refresh_clock};
    use crate::prompt::PromptBlocks;
    use jiff::Timestamp;
    use jiff::tz::TimeZone;

    #[test]
    fn utc_clock_names_the_machine_time() {
        let Ok(now) = "2026-09-29T14:58:00Z".parse::<Timestamp>() else {
            panic!("test timestamp must be valid");
        };
        assert_eq!(
            clock_line(now, &TimeZone::UTC),
            "Current time: 2026-09-29 14:58:00 UTC (UTC+00). UTC: 2026-09-29 14:58:00. This is the machine clock now. Use it for the time and the date."
        );
    }

    #[test]
    fn refresh_replaces_only_the_clock_line() {
        let blocks = PromptBlocks {
            cached: vec!["core".to_string()],
            volatile: vec![
                "plugin".to_string(),
                format!("{CLOCK_PREFIX} stale"),
                "session".to_string(),
            ],
        };
        let fresh = refresh_clock(&blocks);
        assert_eq!(fresh.cached, ["core"]);
        assert_eq!(fresh.volatile[0], "plugin");
        assert!(fresh.volatile[1].starts_with(CLOCK_PREFIX));
        assert!(fresh.volatile[1].contains("machine clock now"));
        assert_ne!(fresh.volatile[1], format!("{CLOCK_PREFIX} stale"));
        assert_eq!(fresh.volatile[2], "session");
    }
}
