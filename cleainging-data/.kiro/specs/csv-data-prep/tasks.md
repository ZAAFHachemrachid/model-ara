# Implementation Plan

- [x] 1. Set up project dependencies and core data structures
  - [x] 1.1 Add required dependencies to Cargo.toml (csv, serde, thiserror, rand, proptest)
    - Add csv crate for CSV parsing/writing
    - Add serde with derive feature for serialization
    - Add thiserror for error handling
    - Add rand for random sampling
    - Add proptest as dev dependency for property testing
    - _Requirements: 1.1, 4.1_
  - [x] 1.2 Create Record struct and error types
    - Define Record struct with title, text, subject, date fields
    - Implement Serialize/Deserialize derives
    - Define CsvError enum with FileNotFound, Io, and Parse variants
    - _Requirements: 1.1, 1.3_

- [x] 2. Implement CSV loading functionality
  - [x] 2.1 Implement CSV file loader
    - Create load function that reads CSV from path
    - Handle file not found errors
    - Skip malformed rows and continue processing
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [ ]* 2.2 Write property test for CSV round-trip
    - **Property 1: CSV Round-Trip Consistency**
    - **Validates: Requirements 1.1, 1.4, 4.1, 4.2**

- [-] 3. Implement deduplication functionality
  - [x] 3.1 Implement content-based deduplicator
    - Create deduplicate function using HashSet on (title, text) tuple
    - Return DeduplicationResult with records and duplicates_removed count
    - Preserve first occurrence ordering
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 3.2 Write property test for deduplication uniqueness
    - **Property 2: Deduplication Uniqueness**
    - **Validates: Requirements 2.1**
  - [x] 3.3 Write property test for first occurrence preservation
    - **Property 3: Deduplication Preserves First Occurrence**
    - **Validates: Requirements 2.2**
  - [ ]* 3.4 Write property test for deduplication count accuracy
    - **Property 4: Deduplication Count Accuracy**
    - **Validates: Requirements 2.3**

- [x] 4. Implement balancing functionality
  - [x] 4.1 Implement random balancer
    - Create balance function that takes two Vec<Record> and RNG
    - Reduce larger dataset to match smaller using random sampling
    - Return BalanceResult with both datasets and final_size
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ]* 4.2 Write property test for balance size equality
    - **Property 5: Balance Size Equality**
    - **Validates: Requirements 3.1, 3.3**
  - [ ]* 4.3 Write property test for balance subset property
    - **Property 6: Balance Subset Property**
    - **Validates: Requirements 3.2**

- [x] 5. Implement CSV writing functionality
  - [x] 5.1 Implement CSV file writer
    - Create write function that outputs records to CSV file
    - Properly escape special characters and handle multi-line text
    - _Requirements: 4.1, 4.2, 4.3_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement main processing pipeline
  - [x] 7.1 Create main function with full pipeline
    - Load both CSV files (fake.csv, true.csv)
    - Display initial record counts
    - Deduplicate each dataset and display results
    - Balance datasets and display final counts
    - Write output files (fake_clean.csv, true_clean.csv)
    - Display final summary
    - _Requirements: 4.3, 5.1, 5.2, 5.3_

- [ ] 8. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
