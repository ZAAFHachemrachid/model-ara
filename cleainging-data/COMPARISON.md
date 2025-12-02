# Rust vs Python Implementation Comparison

## Overview

| Aspect | Python | Rust |
|--------|--------|------|
| **Lines of Code** | ~230 lines (single file) | ~450 lines (5 modules) |
| **Dependencies** | pandas, scipy, numpy, imblearn, emoji, re | csv, serde, thiserror, rand |
| **Memory Model** | Loads entire DataFrames in memory | Streams/loads records into Vec |

## Architecture

### Python
Single procedural script with inline logic. Everything runs top-to-bottom with print statements for progress.

### Rust
Modular design with separation of concerns:
- `record.rs` - Data types
- `error.rs` - Error handling
- `io.rs` - File operations
- `processing.rs` - Business logic
- `main.rs` - Orchestration

## Key Differences

### 1. Deduplication

**Python** - uses pandas built-in:
```python
df = df.drop_duplicates()
```

**Rust** - explicit HashSet-based deduplication on (title, text):
```rust
let mut seen: HashSet<(String, String)> = HashSet::new();
for record in records {
    if seen.insert(key) { deduplicated.push(record); }
}
```

### 2. Balancing Strategy

| | Python | Rust |
|-|--------|------|
| **Approach** | Upsamples smaller dataset | Downsamples larger dataset |
| **Method** | Duplicates rows to match larger | Randomly samples to match smaller |

This is a significant behavioral difference!

### 3. Text Cleaning

| | Python | Rust |
|-|--------|------|
| Lowercase | ✅ | ❌ |
| Strip whitespace | ✅ | ❌ |
| Remove URLs | ✅ | ❌ |
| Remove emojis | ✅ | ❌ |
| Normalize whitespace | ✅ | ❌ |

Rust preserves original content exactly.

### 4. Error Handling

**Python** - try/except with exit():
```python
try:
    fake_df = pd.read_csv("./DataSet/fake.csv")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()
```

**Rust** - Result type with custom error enum:
```rust
pub fn load(path: &Path) -> Result<Vec<Record>, CsvError> {
    if !path.exists() {
        return Err(CsvError::FileNotFound(path.to_path_buf()));
    }
    // ...
}
```

### 5. Testing

| | Python | Rust |
|-|--------|------|
| Unit tests | ❌ | ✅ |
| Property-based tests | ❌ | ✅ (proptest) |

## Performance Considerations

- **Rust** will be significantly faster for large datasets due to zero-cost abstractions and no GC
- **Python** has higher memory overhead (pandas DataFrames) but offers more data manipulation flexibility
- Rust's `csv` crate handles streaming better for very large files

## Output Files

| Python | Rust |
|--------|------|
| `fake_cleaned.csv`, `true_cleaned.csv` | `fake_clean.csv`, `true_clean.csv` |
| `fake_balanced.csv`, `true_balanced.csv` | (same as cleaned) |
| `combined_balanced.csv` | (not produced) |
| `cleaning_report.txt` | (console output only) |

## Summary

| Aspect | Python | Rust |
|--------|--------|------|
| Feature-rich | ✅ | ❌ |
| Type-safe | ❌ | ✅ |
| Well-tested | ❌ | ✅ |
| Performance | Slower | Faster |
| Text cleaning | Extensive | None |
| Balancing | Upsample | Downsample |
