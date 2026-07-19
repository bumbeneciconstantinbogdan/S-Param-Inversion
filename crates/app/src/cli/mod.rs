//! Command-line interface subcommands for the S-parameter inversion pipeline.
//!
//! Each submodule implements a single CLI subcommand using pure static dispatch.

pub mod benchmark;
pub mod evaluate;
pub mod generate;
pub mod hpo;
mod shared;
pub mod train;

#[cfg(test)]
mod tests {
    use sparam_core::io::{ensure_dir, write_json};

    // -----------------------------------------------------------------------
    // ensure_dir
    // -----------------------------------------------------------------------

    #[test]
    fn ensure_dir_creates_nested_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        ensure_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_dir_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("exists");
        std::fs::create_dir(&dir).unwrap();
        ensure_dir(&dir).unwrap();
        assert!(dir.is_dir());
    }

    // -----------------------------------------------------------------------
    // write_json
    // -----------------------------------------------------------------------

    #[test]
    fn write_json_creates_file_and_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sub").join("out.json");
        write_json(&serde_json::json!({"key": 42}), &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"key\": 42"));
    }
}
