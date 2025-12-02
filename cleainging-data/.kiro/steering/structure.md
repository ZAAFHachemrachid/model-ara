# Project Structure

```
nlp-fack/
├── src/
│   └── main.rs          # All application code (single-file architecture)
├── Cargo.toml           # Project manifest and dependencies
├── Cargo.lock           # Locked dependency versions
├── fake.csv             # Input: fake news dataset (large)
├── true.csv             # Input: true news dataset (large)
├── fake_clean.csv       # Output: processed fake news
├── true_clean.csv       # Output: processed true news
└── .kiro/
    └── specs/           # Feature specifications
```

## Code Organization (main.rs)

The codebase follows a single-file architecture with clear separation:

1. **Data Types**: `Record`, `DeduplicationResult`, `BalanceResult`
2. **Error Handling**: `CsvError` enum with thiserror
3. **Core Functions**:
   - `load()` - CSV file loading with error handling
   - `deduplicate()` - Remove duplicate records
   - `balance()` - Equalize dataset sizes
   - `write()` - CSV output
4. **Entry Point**: `main()` and `run()` orchestration
5. **Tests**: Inline `#[cfg(test)]` module with unit tests

## Conventions
- Public functions have doc comments with `# Arguments`, `# Returns`, `# Requirements` sections
- Tests use `tempfile` for file-based testing
- Error handling uses `Result<T, CsvError>` pattern
