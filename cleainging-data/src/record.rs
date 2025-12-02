//! Data types for news article records and processing results.

use serde::{Deserialize, Serialize};

/// A single record from the CSV file containing news article data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Record {
    pub title: String,
    pub text: String,
    pub subject: String,
    pub date: String,
}

/// Result of the deduplication process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeduplicationResult {
    /// The deduplicated records with first occurrences preserved.
    pub records: Vec<Record>,
    /// The number of duplicate records that were removed.
    pub duplicates_removed: usize,
}

/// Result of the balancing process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BalanceResult {
    /// The first dataset after balancing.
    pub dataset_a: Vec<Record>,
    /// The second dataset after balancing.
    pub dataset_b: Vec<Record>,
    /// The final size of both datasets.
    pub final_size: usize,
}
