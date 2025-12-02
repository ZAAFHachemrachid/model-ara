//! NLP-Fack: A CLI tool for preparing news article datasets for NLP/ML tasks.

mod error;
mod io;
mod processing;
mod record;

pub use error::CsvError;
pub use record::{BalanceResult, DeduplicationResult, Record};

use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), CsvError> {
    // Define input and output paths
    let fake_input = Path::new("fake.csv");
    let true_input = Path::new("true.csv");
    let fake_output = Path::new("fake_clean.csv");
    let true_output = Path::new("true_clean.csv");

    // Step 1: Load CSV files
    println!("=== Step 1: Loading CSV files ===");
    let fake_records = io::load(fake_input)?;
    let true_records = io::load(true_input)?;
    let fake_initial = fake_records.len();
    let true_initial = true_records.len();

    println!("  Fake news: {} rows loaded", fake_initial);
    println!("  True news: {} rows loaded", true_initial);
    println!();

    // Step 2: Deduplicate datasets
    println!("=== Step 2: Deduplication ===");
    println!("Before:");
    println!("  Fake news: {} rows", fake_initial);
    println!("  True news: {} rows", true_initial);

    let fake_dedup = processing::deduplicate(fake_records);
    let true_dedup = processing::deduplicate(true_records);
    let fake_after_dedup = fake_dedup.records.len();
    let true_after_dedup = true_dedup.records.len();

    println!("After:");
    println!("  Fake news: {} rows ({} duplicates removed)", fake_after_dedup, fake_dedup.duplicates_removed);
    println!("  True news: {} rows ({} duplicates removed)", true_after_dedup, true_dedup.duplicates_removed);
    println!();

    // Step 3: Balance datasets
    println!("=== Step 3: Balancing ===");
    println!("Before:");
    println!("  Fake news: {} rows", fake_after_dedup);
    println!("  True news: {} rows", true_after_dedup);

    let mut rng = rand::thread_rng();
    let balance_result = processing::balance(fake_dedup.records, true_dedup.records, &mut rng);

    println!("After:");
    println!("  Fake news: {} rows", balance_result.dataset_a.len());
    println!("  True news: {} rows", balance_result.dataset_b.len());
    println!();

    // Step 4: Write output files
    println!("=== Step 4: Writing output files ===");
    io::write(fake_output, &balance_result.dataset_a)?;
    io::write(true_output, &balance_result.dataset_b)?;

    println!("  {} ({} rows)", fake_output.display(), balance_result.dataset_a.len());
    println!("  {} ({} rows)", true_output.display(), balance_result.dataset_b.len());
    println!();

    // Final summary
    let total_input = fake_initial + true_initial;
    let total_output = balance_result.dataset_a.len() + balance_result.dataset_b.len();

    println!("=== Final Summary ===");
    println!("Input:  {} total rows (fake: {}, true: {})", total_input, fake_initial, true_initial);
    println!("Output: {} total rows (fake: {}, true: {})", total_output, balance_result.dataset_a.len(), balance_result.dataset_b.len());
    println!("Duplicates removed: {}", fake_dedup.duplicates_removed + true_dedup.duplicates_removed);
    println!("Processing complete!");

    Ok(())
}
