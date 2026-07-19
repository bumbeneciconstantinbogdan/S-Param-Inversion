//! Custom Askama filters for the web templates.

/// Format a float with the given number of decimal places.
pub fn format_f64(val: &f64, decimals: &usize) -> askama::Result<String> {
    Ok(format!("{val:.decimals$}", decimals = decimals))
}

/// Format a duration in seconds as a human-readable string.
pub fn format_duration(secs: &f64) -> askama::Result<String> {
    if *secs < 60.0 {
        Ok(format!("{secs:.1}s"))
    } else if *secs < 3600.0 {
        let mins = secs / 60.0;
        Ok(format!("{mins:.1}m"))
    } else {
        let hours = secs / 3600.0;
        Ok(format!("{hours:.1}h"))
    }
}

/// Format a percentage value.
pub fn format_pct(val: &f64) -> askama::Result<String> {
    Ok(format!("{val:.1}%"))
}
