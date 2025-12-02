//! Data processing operations for deduplication and balancing.

use std::collections::HashSet;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::record::{BalanceResult, DeduplicationResult, Record};

/// Removes duplicate records based on matching title and text fields.
///
/// # Arguments
/// * `records` - Vector of records to deduplicate
///
/// # Returns
/// * `DeduplicationResult` containing deduplicated records and count of duplicates removed
///
/// # Requirements
/// - 2.1: Identifies duplicates based on matching title and text fields
/// - 2.2: Retains only the first occurrence and removes subsequent duplicates
/// - 2.3: Reports the number of duplicates removed
pub fn deduplicate(records: Vec<Record>) -> DeduplicationResult {
    let original_count = records.len();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut deduplicated: Vec<Record> = Vec::new();

    for record in records {
        let key = (record.title.clone(), record.text.clone());
        if seen.insert(key) {
            // This is the first occurrence of this (title, text) pair
            deduplicated.push(record);
        }
        // If insert returns false, we've seen this key before - skip the duplicate
    }

    DeduplicationResult {
        duplicates_removed: original_count - deduplicated.len(),
        records: deduplicated,
    }
}

/// Balances two datasets to have equal sizes by randomly sampling the larger one.
///
/// # Arguments
/// * `dataset_a` - First dataset of records
/// * `dataset_b` - Second dataset of records
/// * `rng` - Random number generator for sampling
///
/// # Returns
/// * `BalanceResult` containing both balanced datasets and the final size
///
/// # Requirements
/// - 3.1: Reduces the larger dataset to match the size of the smaller dataset
/// - 3.2: Randomly samples records to retain from the larger dataset
/// - 3.3: Reports the final size of both datasets
pub fn balance<R: Rng>(
    dataset_a: Vec<Record>,
    dataset_b: Vec<Record>,
    rng: &mut R,
) -> BalanceResult {
    let final_size = dataset_a.len().min(dataset_b.len());

    let balanced_a = if dataset_a.len() > final_size {
        // Randomly sample from dataset_a
        let mut indices: Vec<usize> = (0..dataset_a.len()).collect();
        indices.shuffle(rng);
        indices.truncate(final_size);
        indices.sort(); // Preserve relative order of selected records
        indices.into_iter().map(|i| dataset_a[i].clone()).collect()
    } else {
        dataset_a
    };

    let balanced_b = if dataset_b.len() > final_size {
        // Randomly sample from dataset_b
        let mut indices: Vec<usize> = (0..dataset_b.len()).collect();
        indices.shuffle(rng);
        indices.truncate(final_size);
        indices.sort(); // Preserve relative order of selected records
        indices.into_iter().map(|i| dataset_b[i].clone()).collect()
    } else {
        dataset_b
    };

    BalanceResult {
        dataset_a: balanced_a,
        dataset_b: balanced_b,
        final_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    fn make_record(title: &str, text: &str) -> Record {
        Record {
            title: title.to_string(),
            text: text.to_string(),
            subject: "subject".to_string(),
            date: "date".to_string(),
        }
    }

    /// Strategy to generate arbitrary Record instances for property testing
    fn arb_record() -> impl Strategy<Value = Record> {
        (
            any::<String>(),
            any::<String>(),
            any::<String>(),
            any::<String>(),
        )
            .prop_map(|(title, text, subject, date)| Record {
                title,
                text,
                subject,
                date,
            })
    }

    /// Strategy to generate a vector of Records
    fn arb_records() -> impl Strategy<Value = Vec<Record>> {
        prop::collection::vec(arb_record(), 0..50)
    }

    // **Feature: csv-data-prep, Property 2: Deduplication Uniqueness**
    // **Validates: Requirements 2.1**
    //
    // *For any* collection of Records, after deduplication, no two Records
    // in the result SHALL have the same (title, text) pair.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_deduplication_uniqueness(records in arb_records()) {
            let result = deduplicate(records);

            // Collect all (title, text) pairs from the deduplicated result
            let mut seen_keys: HashSet<(String, String)> = HashSet::new();

            for record in &result.records {
                let key = (record.title.clone(), record.text.clone());
                // If insert returns false, we found a duplicate - this should never happen
                prop_assert!(
                    seen_keys.insert(key.clone()),
                    "Found duplicate (title, text) pair after deduplication: {:?}",
                    key
                );
            }

            // Additional check: the number of unique keys should equal the number of records
            prop_assert_eq!(
                seen_keys.len(),
                result.records.len(),
                "Number of unique keys should equal number of deduplicated records"
            );
        }
    }

    // **Feature: csv-data-prep, Property 3: Deduplication Preserves First Occurrence**
    // **Validates: Requirements 2.2**
    //
    // *For any* collection of Records containing duplicates, the deduplicated result
    // SHALL contain the first occurrence of each unique (title, text) pair in the original order.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_deduplication_preserves_first_occurrence(records in arb_records()) {
            let result = deduplicate(records.clone());

            // Build a map of (title, text) -> first occurrence index and record from original
            let mut first_occurrences: std::collections::HashMap<(String, String), (usize, &Record)> =
                std::collections::HashMap::new();

            for (idx, record) in records.iter().enumerate() {
                let key = (record.title.clone(), record.text.clone());
                first_occurrences.entry(key).or_insert((idx, record));
            }

            // Verify each record in the result matches the first occurrence from the original
            for result_record in &result.records {
                let key = (result_record.title.clone(), result_record.text.clone());
                let (_, first_record) = first_occurrences
                    .get(&key)
                    .expect("Result record should have a corresponding first occurrence");

                // The result record should be identical to the first occurrence
                prop_assert_eq!(
                    result_record, *first_record,
                    "Deduplicated record should match the first occurrence"
                );
            }

            // Verify the order is preserved: result records should appear in the same
            // relative order as their first occurrences in the original
            let result_indices: Vec<usize> = result.records
                .iter()
                .map(|r| {
                    let key = (r.title.clone(), r.text.clone());
                    first_occurrences.get(&key).unwrap().0
                })
                .collect();

            // Check that indices are strictly increasing (preserves original order)
            for window in result_indices.windows(2) {
                prop_assert!(
                    window[0] < window[1],
                    "Result records should preserve original order: index {} should be < {}",
                    window[0],
                    window[1]
                );
            }
        }
    }

    // Deduplication tests

    #[test]
    fn test_deduplicate_no_duplicates() {
        let records = vec![
            make_record("Title1", "Text1"),
            make_record("Title2", "Text2"),
            make_record("Title3", "Text3"),
        ];

        let result = deduplicate(records);
        assert_eq!(result.records.len(), 3);
        assert_eq!(result.duplicates_removed, 0);
    }

    #[test]
    fn test_deduplicate_with_duplicates() {
        let records = vec![
            make_record("Title1", "Text1"),
            make_record("Title2", "Text2"),
            make_record("Title1", "Text1"), // Duplicate
            make_record("Title3", "Text3"),
        ];

        let result = deduplicate(records);
        assert_eq!(result.records.len(), 3);
        assert_eq!(result.duplicates_removed, 1);
    }

    #[test]
    fn test_deduplicate_preserves_first_occurrence() {
        let records = vec![
            Record {
                title: "Title".to_string(),
                text: "Text".to_string(),
                subject: "First".to_string(),
                date: "Date1".to_string(),
            },
            Record {
                title: "Title".to_string(),
                text: "Text".to_string(),
                subject: "Second".to_string(),
                date: "Date2".to_string(),
            },
        ];

        let result = deduplicate(records);
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].subject, "First");
        assert_eq!(result.duplicates_removed, 1);
    }

    #[test]
    fn test_deduplicate_empty_input() {
        let records: Vec<Record> = vec![];
        let result = deduplicate(records);
        assert_eq!(result.records.len(), 0);
        assert_eq!(result.duplicates_removed, 0);
    }

    #[test]
    fn test_deduplicate_all_duplicates() {
        let records = vec![
            make_record("Same", "Same"),
            make_record("Same", "Same"),
            make_record("Same", "Same"),
        ];

        let result = deduplicate(records);
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.duplicates_removed, 2);
    }

    #[test]
    fn test_deduplicate_same_title_different_text() {
        // Records with same title but different text should NOT be considered duplicates
        let records = vec![make_record("Title", "Text1"), make_record("Title", "Text2")];

        let result = deduplicate(records);
        assert_eq!(result.records.len(), 2);
        assert_eq!(result.duplicates_removed, 0);
    }

    // Balance tests

    #[test]
    fn test_balance_equal_sizes() {
        let dataset_a = vec![make_record("A1", "TextA1"), make_record("A2", "TextA2")];
        let dataset_b = vec![make_record("B1", "TextB1"), make_record("B2", "TextB2")];

        let mut rng = rand::thread_rng();
        let result = balance(dataset_a, dataset_b, &mut rng);

        assert_eq!(result.dataset_a.len(), 2);
        assert_eq!(result.dataset_b.len(), 2);
        assert_eq!(result.final_size, 2);
    }

    #[test]
    fn test_balance_first_larger() {
        let dataset_a = vec![
            make_record("A1", "TextA1"),
            make_record("A2", "TextA2"),
            make_record("A3", "TextA3"),
            make_record("A4", "TextA4"),
        ];
        let dataset_b = vec![make_record("B1", "TextB1"), make_record("B2", "TextB2")];

        let mut rng = rand::thread_rng();
        let result = balance(dataset_a.clone(), dataset_b.clone(), &mut rng);

        assert_eq!(result.dataset_a.len(), 2);
        assert_eq!(result.dataset_b.len(), 2);
        assert_eq!(result.final_size, 2);
        // Verify subset property - all records in result should be from original
        for record in &result.dataset_a {
            assert!(dataset_a.contains(record));
        }
    }

    #[test]
    fn test_balance_second_larger() {
        let dataset_a = vec![make_record("A1", "TextA1"), make_record("A2", "TextA2")];
        let dataset_b = vec![
            make_record("B1", "TextB1"),
            make_record("B2", "TextB2"),
            make_record("B3", "TextB3"),
            make_record("B4", "TextB4"),
        ];

        let mut rng = rand::thread_rng();
        let result = balance(dataset_a.clone(), dataset_b.clone(), &mut rng);

        assert_eq!(result.dataset_a.len(), 2);
        assert_eq!(result.dataset_b.len(), 2);
        assert_eq!(result.final_size, 2);
        // Verify subset property - all records in result should be from original
        for record in &result.dataset_b {
            assert!(dataset_b.contains(record));
        }
    }

    #[test]
    fn test_balance_empty_datasets() {
        let dataset_a: Vec<Record> = vec![];
        let dataset_b: Vec<Record> = vec![];

        let mut rng = rand::thread_rng();
        let result = balance(dataset_a, dataset_b, &mut rng);

        assert_eq!(result.dataset_a.len(), 0);
        assert_eq!(result.dataset_b.len(), 0);
        assert_eq!(result.final_size, 0);
    }

    #[test]
    fn test_balance_one_empty() {
        let dataset_a = vec![make_record("A1", "TextA1"), make_record("A2", "TextA2")];
        let dataset_b: Vec<Record> = vec![];

        let mut rng = rand::thread_rng();
        let result = balance(dataset_a, dataset_b, &mut rng);

        assert_eq!(result.dataset_a.len(), 0);
        assert_eq!(result.dataset_b.len(), 0);
        assert_eq!(result.final_size, 0);
    }
}
