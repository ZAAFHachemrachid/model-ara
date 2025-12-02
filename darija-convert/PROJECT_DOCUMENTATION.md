# Darija Convert - Project Documentation

## Executive Summary

**Darija Convert** is a high-performance data processing tool built in Rust that transforms and cleans Arabic news articles from CSV datasets. The project focuses on extracting, normalizing, and preparing multilingual news content for analysis and machine learning applications.

---

## Project Overview

### Purpose
The tool processes large CSV files containing news articles in both English and Arabic (Darija dialect), extracting relevant content while removing noise, URLs, mentions, emojis, and excessive punctuation. The cleaned data is then exported to standardized CSV format for downstream analysis.

### Key Objectives
- Extract and clean Arabic text from raw news datasets
- Remove social media artifacts (URLs, mentions, emojis)
- Normalize punctuation and whitespace
- Maintain data integrity through proper CSV handling
- Process large datasets efficiently with minimal memory overhead

---

## Technical Architecture

### Technology Stack
- **Language**: Rust 2021 Edition
- **Dependencies**: 
  - `regex` v1.x - Pattern matching for filename generation
  - Standard library for I/O operations

### Core Components

#### 1. **Main Processing Pipeline** (`main.rs`)
The application follows a streaming architecture:

```
Input CSV → Line-by-line parsing → Field extraction → 
Text cleaning → CSV escaping → Output CSV
```

#### 2. **CSV Parser** (`parse_csv_line`)
- Handles quoted fields with escaped quotes
- Supports comma-separated values with proper quote handling
- Preserves field integrity during parsing

#### 3. **Text Cleaning Engine** (`clean_text`)
Multi-stage cleaning process:

**Stage 1: URL & Mention Removal**
- Filters HTTP/HTTPS URLs
- Removes Twitter-specific URLs (pic.twitter, t.co)
- Eliminates @mentions and parenthetical mentions

**Stage 2: Character Normalization**
- Removes emoji characters (emoticons, symbols, transport, flags, dingbats)
- Handles excessive punctuation (max 2 consecutive identical punctuation marks)
- Normalizes whitespace

**Stage 3: Output Formatting**
- Joins cleaned words with single spaces
- Ensures consistent formatting

#### 4. **CSV Output Handler** (`escape_csv`)
- Properly escapes fields containing commas, quotes, or newlines
- Doubles internal quotes for CSV compliance
- Maintains RFC 4180 CSV standard compliance

#### 5. **Filename Generation** (`generate_output_filename`)
- Intelligently parses input filenames
- Preserves numbering patterns (e.g., `_01`, `_02`)
- Generates output as `{base}_done{number}.csv`

---

## Data Processing Workflow

### Input Format
```csv
title,text,subject,date,عنوان,نص
"English Title","English Text","Category","Date","Arabic Title","Arabic Text"
```

### Processing Steps

1. **Read Input**: Stream CSV file line-by-line
2. **Parse Fields**: Extract 6 fields per record
3. **Extract Arabic Content**: Fields 4 (title) and 5 (text)
4. **Clean Text**: Apply multi-stage cleaning
5. **Format Output**: Create standardized CSV
6. **Write Output**: Stream to new file with `_done` suffix

### Output Format
```csv
title,text,subject,date
"Cleaned Arabic Title","Cleaned Arabic Text","Category","Date"
```

---

## Dataset Overview

### Available Datasets

The project includes multiple dataset variants for testing and validation:

#### True Datasets (Ground Truth)
- `true_clean_translated_01.csv` - Part 1
- `true_clean_translated_02.csv` - Part 2
- `true_clean_translated_part3.csv` - Part 3
- `true_clean_translated_part4.csv` - Part 4
- `true_clean_translated_part5.csv` - Part 5

#### Fake Datasets (Test/Comparison)
- `fake_clean_translated_01.csv` - Part 1
- `fake_clean_translated_02.csv` - Part 2
- `fake_clean_translated_part3.csv` - Part 3
- `fake_clean_translated_part4.csv` - Part 4

#### Processed Outputs
- `true_clean_translated_done_01.csv` - Processed Part 1
- `true_clean_translated_done_02.csv` - Processed Part 2
- `true_clean_translated_part_done3.csv` - Processed Part 3
- `true_clean_translated_part_done4.csv` - Processed Part 4
- `true_clean_translated_part_done5.csv` - Processed Part 5

### Data Characteristics
- **Content**: News articles from Reuters and other news sources
- **Languages**: English + Arabic (Darija dialect)
- **Categories**: Politics, News
- **Date Range**: December 2017 - January 2018
- **Sample Size**: Multiple articles per dataset file

---

## Key Features

### 1. Intelligent Text Cleaning
- **URL Removal**: Eliminates all HTTP/HTTPS links and Twitter-specific URLs
- **Mention Filtering**: Removes @mentions in various formats
- **Emoji Handling**: Strips emoji characters across all Unicode ranges
- **Punctuation Normalization**: Limits consecutive identical punctuation to 2 occurrences
- **Whitespace Normalization**: Collapses multiple spaces into single spaces

### 2. Robust CSV Handling
- Proper quote escaping for RFC 4180 compliance
- Handles fields with special characters
- Preserves data integrity during round-trip processing

### 3. Efficient Processing
- Streaming architecture for memory efficiency
- Line-by-line processing suitable for large files
- Minimal memory footprint regardless of file size

### 4. Smart Filename Management
- Preserves original numbering schemes
- Automatically generates output filenames
- Supports multiple file naming patterns

---

## Usage

### Building the Project
```bash
cargo build --release
```

### Running the Tool
```bash
cargo run --release -- <input.csv>
```

### Example
```bash
cargo run --release -- true_clean_translated_01.csv
# Output: true_clean_translated_done_01.csv
```

### Command Line Interface
```
Usage: darija-convert <input.csv>

Arguments:
  <input.csv>  Path to input CSV file

Output:
  {input}_done.csv  Processed CSV file with cleaned content
```

---

## Performance Characteristics

### Efficiency Metrics
- **Memory Usage**: O(1) - Constant memory regardless of file size
- **Processing Speed**: Streaming line-by-line processing
- **I/O Pattern**: Sequential read/write for optimal disk performance

### Scalability
- Handles files of any size without memory constraints
- Suitable for batch processing large datasets
- Can process multiple files sequentially

---

## Text Cleaning Examples

### Example 1: URL and Mention Removal
**Input:**
```
Check this out https://t.co/abc123 @user and pic.twitter.com/xyz
```

**Output:**
```
Check this out and
```

### Example 2: Emoji and Punctuation Normalization
**Input:**
```
Great news!!! 😊😊 Really??? Amazing...
```

**Output:**
```
Great news!! Really?? Amazing..
```

### Example 3: Whitespace Normalization
**Input:**
```
Multiple    spaces    between    words
```

**Output:**
```
Multiple spaces between words
```

---

## Project Structure

```
darija-convert/
├── Cargo.toml                          # Project manifest
├── Cargo.lock                          # Dependency lock file
├── src/
│   └── main.rs                         # Main application code
├── true_clean_translated_*.csv         # Ground truth datasets
├── fake_clean_translated_*.csv         # Test datasets
├── true_clean_translated_done_*.csv    # Processed outputs
└── fake_clean_translated_done_*.csv    # Processed test outputs
```

---

## Implementation Details

### Emoji Detection
The tool detects emojis across multiple Unicode ranges:
- Emoticons (U+1F600-U+1F64F)
- Miscellaneous Symbols (U+1F300-U+1F5FF)
- Transport & Map Symbols (U+1F680-U+1F6FF)
- Flags (U+1F1E0-U+1F1FF)
- Miscellaneous Symbols (U+2600-U+26FF)
- Dingbats (U+2700-U+27BF)
- Variation Selectors (U+FE00-U+FE0F)
- Supplemental Symbols (U+1F900-U+1F9FF)

### Punctuation Handling
Supported punctuation marks: `! ? . , ; : - _`

The algorithm:
1. Tracks consecutive identical punctuation
2. Allows up to 2 consecutive occurrences
3. Skips additional occurrences
4. Resets counter on different punctuation

---

## Quality Assurance

### Data Validation
- Input validation for CSV format
- Field count verification (minimum 6 fields required)
- Proper error handling for file I/O operations

### Output Verification
- CSV compliance checking
- Field escaping validation
- Filename generation accuracy

### Testing Approach
- Multiple dataset variants (true vs. fake)
- Processed output files for comparison
- Batch processing capability for validation

---

## Use Cases

### 1. News Article Preprocessing
Clean raw news data for NLP and machine learning pipelines

### 2. Multilingual Dataset Preparation
Prepare Arabic-English parallel corpora for translation models

### 3. Social Media Data Cleaning
Remove social media artifacts from news aggregation datasets

### 4. Data Quality Improvement
Standardize and normalize text data for analysis

### 5. Batch Processing
Process multiple dataset files in sequence for large-scale operations

---

## Future Enhancements

### Potential Improvements
1. **Parallel Processing**: Multi-threaded processing for multiple files
2. **Configuration Options**: Customizable cleaning rules via config file
3. **Language Detection**: Automatic language identification
4. **Advanced NLP**: Tokenization and lemmatization support
5. **Progress Reporting**: Real-time processing statistics
6. **Error Recovery**: Graceful handling of malformed records
7. **Output Formats**: Support for JSON, Parquet, and other formats

---

## Dependencies & Licensing

### Direct Dependencies
- `regex` v1.x - Pattern matching library

### Build Requirements
- Rust 1.56+ (2021 edition)
- Cargo package manager

---

## Conclusion

Darija Convert is a production-ready tool for processing and cleaning multilingual news datasets. Its streaming architecture ensures efficient processing of large files while maintaining data integrity through proper CSV handling and comprehensive text cleaning. The tool is particularly valuable for preparing Arabic news content for machine learning and NLP applications.

---

## Contact & Support

For questions or issues regarding this project, please refer to the project repository or contact the development team.

**Project Name**: Darija Convert  
**Version**: 0.1.0  
**Edition**: Rust 2021  
**Status**: Active Development
