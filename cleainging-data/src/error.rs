//! Error types for CSV processing operations.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during CSV processing.
#[derive(Debug, Error)]
pub enum CsvError {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    Parse(#[from] csv::Error),
}
