//! Time: the server's clock (which tests can move forward), and the
//! formats the API uses: RFC 3339 times, 'YYYY-MM' months, all in UTC.

use std::sync::atomic::{AtomicI64, Ordering};
use time::{Date, Month, OffsetDateTime, format_description::well_known::Rfc3339};

/// The clock: the system's, plus an offset that only tests set.
#[derive(Debug, Default)]
pub struct Clock {
    offset: AtomicI64,
}

impl Clock {
    /// Unix seconds now.
    pub fn now(&self) -> i64 {
        OffsetDateTime::now_utc().unix_timestamp() + self.offset.load(Ordering::Relaxed)
    }

    /// Move the clock forward by `secs` (tests).
    pub fn advance(&self, secs: i64) {
        self.offset.fetch_add(secs, Ordering::Relaxed);
    }
}

fn utc(ts: i64) -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(ts).unwrap_or(OffsetDateTime::UNIX_EPOCH)
}

/// `ts` as an RFC 3339 time in UTC, to the second.
pub fn rfc3339(ts: i64) -> String {
    utc(ts).format(&Rfc3339).unwrap_or_default()
}

/// The calendar month (UTC) of `ts`, as 'YYYY-MM'.
pub fn period_of(ts: i64) -> String {
    let d = utc(ts);
    format!("{:04}-{:02}", d.year(), d.month() as u8)
}

/// Whether `s` is a month as 'YYYY-MM'.
pub fn is_period(s: &str) -> bool {
    let Some((y, m)) = s.split_once('-') else { return false };
    y.len() == 4 && m.len() == 2 && y.parse::<i32>().is_ok() && m.parse::<u8>().is_ok_and(|m| (1..=12).contains(&m))
}

/// Unix seconds at midnight UTC of a 'YYYY-MM-DD' date.
pub fn parse_date(s: &str) -> Option<i64> {
    let mut it = s.split('-');
    let (y, m, d) = (it.next()?.parse::<i32>().ok()?, it.next()?.parse::<u8>().ok()?, it.next()?.parse::<u8>().ok()?);
    if it.next().is_some() || s.len() != 10 {
        return None;
    }
    let date = Date::from_calendar_date(y, Month::try_from(m).ok()?, d).ok()?;
    Some(date.midnight().assume_utc().unix_timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_and_months_are_utc() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(period_of(1_793_491_199), "2026-10");
        assert_eq!(period_of(1_793_491_200), "2026-11");
        assert!(is_period("2026-10") && !is_period("2026-13") && !is_period("26-10") && !is_period("2026-1"));
        assert_eq!(parse_date("2026-11-01"), Some(1_793_491_200));
        assert_eq!(parse_date("2026-02-30"), None);
        assert_eq!(parse_date("2026-1-01"), None);
    }

    #[test]
    fn the_clock_moves_forward_when_told() {
        let c = Clock::default();
        let a = c.now();
        c.advance(86_400 * 31);
        assert!(c.now() - a >= 86_400 * 31);
    }
}
