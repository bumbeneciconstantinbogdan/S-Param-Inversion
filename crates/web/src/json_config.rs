//! Tiny zero-dep helpers for reading typed values out of a parsed
//! `serde_json::Value`.  Used by `/evaluate`, `/infer` and `tasks` to
//! pull stacked toggles + scheduler / optimiser params out of the
//! per-run `config_json` blob without re-parsing the same string four
//! times in a row (the pre-refactor pattern these replace).
//!
//! All helpers fall back to a caller-supplied default rather than
//! propagating an error.  The persisted JSON for older runs predates
//! the stack fields entirely, so missing-key is the *normal* case for
//! anything M10-related — silent fallback is the desired semantics.

/// Look up `obj[key]` and coerce to `u64`.  Returns `default` if the
/// key is absent or the value is not a JSON number representable as
/// `u64`.
pub fn get_u64_or(obj: &serde_json::Value, key: &str, default: u64) -> u64 {
    obj.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

/// Look up `obj[key]` and coerce to `f64`.  Returns `default` if the
/// key is absent or the value is not numeric.
pub fn get_f64_or(obj: &serde_json::Value, key: &str, default: f64) -> f64 {
    obj.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Look up `obj[key]` and return its string slice (no allocation).
/// Returns `default` if the key is absent or the value isn't a string.
pub fn get_str_or<'a>(obj: &'a serde_json::Value, key: &str, default: &'a str) -> &'a str {
    obj.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Parse `text` as JSON.  Returns `Value::Object({})` on parse failure
/// so callers can chain `get_*_or` without an `Option<Value>` wrapper.
/// The empty-object fallback is observably equivalent to "every key
/// missing", which is what `get_*_or` already handles.
pub fn parse_or_default(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_or_default_returns_empty_obj_on_garbage() {
        let v = parse_or_default("not json");
        assert_eq!(v, serde_json::json!({}));
    }

    #[test]
    fn get_u64_or_falls_back_when_missing() {
        let v = serde_json::json!({"a": 5});
        assert_eq!(get_u64_or(&v, "a", 0), 5);
        assert_eq!(get_u64_or(&v, "b", 7), 7);
    }

    #[test]
    fn get_str_or_returns_borrow_when_present() {
        let v = serde_json::json!({"name": "alice"});
        assert_eq!(get_str_or(&v, "name", "?"), "alice");
        assert_eq!(get_str_or(&v, "missing", "?"), "?");
    }
}
