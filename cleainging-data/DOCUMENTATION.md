# Data Cleaning Documentation

## 📋 Overview

**Data Cleaning** is a Rust command-line tool designed for preparing news article datasets for Natural Language Processing (NLP) and Machine Learning (ML) tasks, specifically fake news detection.

The tool processes CSV files containing news articles, cleans them by removing duplicates, balances dataset sizes, and outputs cleaned data ready for ML training.

---

## 🎯 Purpose & Use Case

| Aspect | Description |
|--------|-------------|
| **Domain** | Fake news detection / NLP data preparation |
| **Input** | Raw CSV datasets (`fake.csv`, `true.csv`) |
| **Output** | Cleaned, balanced datasets (`fake_clean.csv`, `true_clean.csv`) |
| **Goal** | Prepare balanced training data for ML models |

---

## 🏗️ Architecture

### Project Structure

```
data-cleaning/
├── src/
│   ├── main.rs          # Entry point & orchestration
│   ├── error.rs         # Custom error types
│   ├── io.rs            # CSV file I/O operations
│   ├── processing.rs    # Deduplication & balancing logic
│   └── record.rs        # Data structures
├── Cargo.toml           # Dependencies & project config
├── fake.csv             # Input: fake news dataset
├── true.csv             # Input: true news dataset
├── fake_clean.csv       # Output: processed fake news
└── true_clean.csv       # Output: processed true news
```

### Module Dependency Graph

```
main.rs
   ├── error.rs      (CsvError)
   ├── io.rs         (load, write)
   ├── processing.rs (deduplicate, balance)
   └── record.rs     (Record, DeduplicationResult, BalanceResult)
```

---

## 📦 Dependencies

### Runtime Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `csv` | 1.3 | CSV parsing and writing |
| `serde` | 1.0 | Serialization/deserialization with `derive` feature |
| `thiserror` | 1.0 | Ergonomic error type derivation |
| `rand` | 0.8 | Random sampling for dataset balancing |

### Dev Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `proptest` | 1.4 | Property-based testing |
| `tempfile` | 3.10 | Temporary files for I/O tests |

---

## 🔧 Core Components

### 1. Data Types (`record.rs`)

#### `Record`
Represents a single news article from the CSV file.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Record {
    pub title: String,    // Article headline
    pub text: String,     // Article body content
    pub subject: String,  // Category/topic
    pub date: String,     // Publication date
}
```

#### `DeduplicationResult`
Contains the output of the deduplication process.

```rust
pub struct DeduplicationResult {
    pub records: Vec<Record>,      // Deduplicated records
    pub duplicates_removed: usize, // Count of removed duplicates
}
```

#### `BalanceResult`
Contains the output of the balancing process.

```rust
pub struct BalanceResult {
    pub dataset_a: Vec<Record>,  // First balanced dataset
    pub dataset_b: Vec<Record>,  // Second balanced dataset
    pub final_size: usize,       // Size of both datasets
}
```

---

### 2. Error Handling (`error.rs`)

Custom error type using `thiserror` for ergonomic error handling:

```rust
#[derive(Debug, Error)]
pub enum CsvError {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    Parse(#[from] csv::Error),
}
```

**Error Variants:**
- `FileNotFound` - Input file doesn't exist
- `Io` - General I/O errors (permissions, disk full, etc.)
- `Parse` - CSV format/parsing errors

---

### 3. File I/O (`io.rs`)

#### `load(path: &Path) -> Result<Vec<Record>, CsvError>`

Loads records from a CSV file.

**Behavior:**
- Validates file existence before parsing
- Parses CSV with headers: `title`, `text`, `subject`, `date`
- Skips malformed rows (graceful degradation)
- Preserves special characters and newlines in text fields

**Example:**
```rust
let records = io::load(Path::new("fake.csv"))?;
println!("Loaded {} records", records.len());
```

#### `write(path: &Path, records: &[Record]) -> Result<(), CsvError>`

Writes records to a CSV file.

**Behavior:**
- Creates valid CSV with proper headers
- Properly escapes quotes, commas, and newlines
- Overwrites existing files

**Example:**
```rust
io::write(Path::new("output.csv"), &records)?;
```

---

### 4. Processing (`processing.rs`)

#### `deduplicate(records: Vec<Record>) -> DeduplicationResult`

Removes duplicate records based on `(title, text)` matching.

**Algorithm:**
1. Iterate through records in order
2. Track seen `(title, text)` pairs in a `HashSet`
3. Keep first occurrence, discard subsequent duplicates
4. Return deduplicated records + removal count

**Complexity:** O(n) time, O(n) space

**Example:**
```rust
let result = processing::deduplicate(records);
println!("Removed {} duplicates", result.duplicates_removed);
```

#### `balance<R: Rng>(dataset_a, dataset_b, rng) -> BalanceResult`

Balances two datasets to equal sizes via random sampling.

**Algorithm:**
1. Determine target size = `min(len_a, len_b)`
2. If a dataset is larger, randomly sample indices
3. Sort selected indices to preserve relative order
4. Return balanced datasets

**Complexity:** O(n log n) due to sorting

**Example:**
```rust
let mut rng = rand::thread_rng();
let result = processing::balance(fake_records, true_records, &mut rng);
assert_eq!(result.dataset_a.len(), result.dataset_b.len());
```

---

## 🚀 Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        NLP-Fack Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: LOAD                    Step 2: DEDUPLICATE
┌──────────────┐               ┌──────────────────────┐
│  fake.csv    │──────────────▶│  Remove duplicates   │
│  (N rows)    │               │  based on title+text │
└──────────────┘               └──────────┬───────────┘
                                          │
┌──────────────┐               ┌──────────▼───────────┐
│  true.csv    │──────────────▶│  Remove duplicates   │
│  (M rows)    │               │  based on title+text │
└──────────────┘               └──────────┬───────────┘
                                          │
                               Step 3: BALANCE
                               ┌──────────▼───────────┐
                               │  Random sampling to  │
                               │  equalize sizes      │
                               └──────────┬───────────┘
                                          │
                               Step 4: WRITE
                               ┌──────────▼───────────┐
                               │  fake_clean.csv      │
                               │  true_clean.csv      │
                               │  (K rows each)       │
                               └──────────────────────┘
```

---

## 💻 Usage

### Build

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Run

```bash
# Run with debug build
cargo run

# Run release build directly
./target/release/nlp-fack
```

### Example Output

```
=== Step 1: Loading CSV files ===
  Fake news: 23481 rows loaded
  True news: 21417 rows loaded

=== Step 2: Deduplication ===
Before:
  Fake news: 23481 rows
  True news: 21417 rows
After:
  Fake news: 23470 rows (11 duplicates removed)
  True news: 21417 rows (0 duplicates removed)

=== Step 3: Balancing ===
Before:
  Fake news: 23470 rows
  True news: 21417 rows
After:
  Fake news: 21417 rows
  True news: 21417 rows

=== Step 4: Writing output files ===
  fake_clean.csv (21417 rows)
  true_clean.csv (21417 rows)

=== Final Summary ===
Input:  44898 total rows (fake: 23481, true: 21417)
Output: 42834 total rows (fake: 21417, true: 21417)
Duplicates removed: 11
Processing complete!
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_deduplicate_with_duplicates
```

### Test Categories

| Category | Location | Description |
|----------|----------|-------------|
| Unit Tests | `io.rs`, `processing.rs` | Function-level tests |
| Property Tests | `processing.rs` | Proptest-based invariant testing |
| Integration | `io.rs` | Round-trip file I/O tests |

### Property-Based Tests

The codebase uses `proptest` for property-based testing:

1. **Deduplication Uniqueness** - After deduplication, no two records have the same `(title, text)` pair
2. **First Occurrence Preservation** - Deduplicated results contain the first occurrence of each unique pair in original order

---

## 📊 CSV Format

### Input/Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `title` | String | Article headline |
| `text` | String | Full article body (may contain newlines) |
| `subject` | String | Category (e.g., "politicsNews", "worldnews") |
| `date` | String | Publication date |
import pandas as pd
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the translation pipeline
print("Loading translation model...")
pipe = pipeline(
    "translation",
    model="facebook/nllb-200-3.3B",
    device=device,
    src_lang="eng_Latn",  # English
    tgt_lang="arb_Arab"   # Arabic
)

# Load the datasets
print("Loading datasets...")
fake_df = pd.read_csv('./fake.csv')
true_df = pd.read_csv('./true.csv')

# Function to translate text in batches
def translate_column(df, column_name, batch_size=8):
    """Translate a column of text to Arabic"""
    translations = []
    texts = df[column_name].fillna("").tolist()
    
    print(f"Translating {len(texts)} texts from column '{column_name}'...")
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Filter out empty strings
        batch = [text if text else "" for text in batch]
        
        try:
            # Translate batch
            results = pipe(batch, max_length=512)
            batch_translations = [r['translation_text'] for r in results]
            translations.extend(batch_translations)
            
            if (i + batch_size) % 50 == 0:
                print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Add empty translations for failed batch
            translations.extend([""] * len(batch))
    
    return translations

# Translate title and text columns for fake news
print("\n=== Translating Fake News ===")
if 'title' in fake_df.columns:
    fake_df['title_ar'] = translate_column(fake_df, 'title')
if 'text' in fake_df.columns:
    fake_df['text_ar'] = translate_column(fake_df, 'text')

# Translate title and text columns for true news
print("\n=== Translating True News ===")
if 'title' in true_df.columns:
    true_df['title_ar'] = translate_column(true_df, 'title')
if 'text' in true_df.columns:
    true_df['text_ar'] = translate_column(true_df, 'text')

# Save the translated datasets
print("\nSaving translated datasets...")
fake_df.to_csv('fake_translated.csv', index=False)
true_df.to_csv('true_translated.csv', index=False)

print("\n✅ Translation completed!")
print(f"Fake news dataset shape: {fake_df.shape}")
print(f"True news dataset shape: {true_df.shape}")

# Display sample translations
print("\n=== Sample Translations ===")
if 'title' in fake_df.columns and 'title_ar' in fake_df.columns:
    print("\nOriginal (Fake):", fake_df['title'].iloc[0][:100])
    print("Arabic:", fake_df['title_ar'].iloc[0][:100])
### Example CSV

```csv
title,text,subject,date
"Breaking News","Article content here...",politicsNews,January 1, 2024
"World Update","More content with ""quotes""...",worldnews,January 2, 2024
```

---

## ⚙️ Configuration

Currently, the tool uses hardcoded paths:

| Setting | Value |
|---------|-------|
| Fake input | `fake.csv` |
| True input | `true.csv` |
| Fake output | `fake_clean.csv` |
| True output | `true_clean.csv` |

To modify paths, edit `src/main.rs`:

```rust
let fake_input = Path::new("fake.csv");
let true_input = Path::new("true.csv");
let fake_output = Path::new("fake_clean.csv");
let true_output = Path::new("true_clean.csv");
```

---

## 🔒 Error Handling

The application handles errors gracefully:

| Scenario | Behavior |
|----------|----------|
| Missing input file | Exits with descriptive error message |
| Malformed CSV rows | Skips row, continues processing |
| Write permission denied | Exits with I/O error |
| Empty input files | Processes normally (outputs empty files) |

---

## 📈 Performance Considerations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Load | O(n) | Linear scan of CSV |
| Deduplicate | O(n) | HashSet-based lookup |
| Balance | O(n log n) | Shuffle + sort for order preservation |
| Write | O(n) | Linear write |

**Memory:** Entire datasets are loaded into memory. For very large files (millions of rows), consider streaming approaches.

---

## 🛠️ Development

### Code Quality

```bash
# Check for errors
cargo check

# Lint with Clippy
cargo clippy

# Format code
cargo fmt
```

### Adding New Features

1. Add data types to `record.rs`
2. Add processing logic to `processing.rs`
3. Add I/O operations to `io.rs`
4. Update orchestration in `main.rs`
5. Add tests alongside implementation

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

1. Follow Rust 2021 edition conventions
2. Add doc comments for public functions
3. Include unit tests for new functionality
4. Run `cargo fmt` and `cargo clippy` before committing
