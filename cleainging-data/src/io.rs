//! File I/O operations for CSV loading and writing.

use std::path::Path;

use crate::error::CsvError;
use crate::record::Record;

/// Loads records from a CSV file at the given path.
///
/// # Arguments
/// * `path` - Path to the CSV file to load
///
/// # Returns
/// * `Ok(Vec<Record>)` - Successfully parsed records (malformed rows are skipped)
/// * `Err(CsvError)` - If the file doesn't exist or cannot be opened
///
/// # Requirements
/// - 1.1: Parses all records with columns: title, text, subject, and date
/// - 1.2: Skips malformed rows and continues processing
/// - 1.3: Returns error with descriptive message for non-existent files
/// - 1.4: Preserves original text content including special characters and newlines
pub fn load(path: &Path) -> Result<Vec<Record>, CsvError> {
    // Check if file exists first to provide a clear error message
    if !path.exists() {
        return Err(CsvError::FileNotFound(path.to_path_buf()));
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true) // Allow varying number of fields
        .from_path(path)?;

    let mut records = Vec::new();

    for result in reader.deserialize() {
        match result {
            Ok(record) => records.push(record),
            Err(_) => {
                // Skip malformed rows and continue processing (Requirement 1.2)
                continue;
            }
        }
    }

    Ok(records)
}

/// Writes records to a CSV file at the given path.
///
/// # Arguments
/// * `path` - Path to the output CSV file
/// * `records` - Slice of records to write
///
/// # Returns
/// * `Ok(())` - Successfully wrote all records
/// * `Err(CsvError)` - If the file cannot be created or written to
///
/// # Requirements
/// - 4.1: Creates valid CSV files with the same column structure as input (title, text, subject, date)
/// - 4.2: Properly escapes special characters and handles multi-line text fields
/// - 4.3: Outputs processed data to CSV files
pub fn write(path: &Path, records: &[Record]) -> Result<(), CsvError> {
    let mut writer = csv::Writer::from_path(path)?;

    for record in records {
        writer.serialize(record)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_file_not_found() {
        let result = load(Path::new("nonexistent_file.csv"));
        assert!(matches!(result, Err(CsvError::FileNotFound(_))));
    }

    #[test]
    fn test_load_valid_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,text,subject,date").unwrap();
        writeln!(file, "Title1,Text1,Subject1,Date1").unwrap();
        writeln!(file, "Title2,Text2,Subject2,Date2").unwrap();

        let records = load(file.path()).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].title, "Title1");
        assert_eq!(records[1].title, "Title2");
    }

    #[test]
    fn test_load_skips_malformed_rows() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,text,subject,date").unwrap();
        writeln!(file, "Title1,Text1,Subject1,Date1").unwrap();
        writeln!(file, "MalformedRow").unwrap(); // Missing columns
        writeln!(file, "Title2,Text2,Subject2,Date2").unwrap();

        let records = load(file.path()).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_load_preserves_special_characters() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,text,subject,date").unwrap();
        writeln!(file, r#""Title with ""quotes""","Text with
newline",Subject,Date"#).unwrap();

        let records = load(file.path()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].title, r#"Title with "quotes""#);
        assert!(records[0].text.contains('\n'));
    }

    #[test]
    fn test_load_empty_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "title,text,subject,date").unwrap();

        let records = load(file.path()).unwrap();
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_write_valid_records() {
        let records = vec![
            Record {
                title: "Title1".to_string(),
                text: "Text1".to_string(),
                subject: "subject".to_string(),
                date: "date".to_string(),
            },
            Record {
                title: "Title2".to_string(),
                text: "Text2".to_string(),
                subject: "subject".to_string(),
                date: "date".to_string(),
            },
        ];

        let file = NamedTempFile::new().unwrap();
        write(file.path(), &records).unwrap();

        // Read back and verify
        let loaded = load(file.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].title, "Title1");
        assert_eq!(loaded[1].title, "Title2");
    }

    #[test]
    fn test_write_empty_records() {
        let records: Vec<Record> = vec![];

        let file = NamedTempFile::new().unwrap();
        write(file.path(), &records).unwrap();

        // Read back and verify
        let loaded = load(file.path()).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_write_special_characters() {
        let records = vec![Record {
            title: r#"Title with "quotes""#.to_string(),
            text: "Text with\nnewline".to_string(),
            subject: "Subject, with, commas".to_string(),
            date: "2024-01-01".to_string(),
        }];

        let file = NamedTempFile::new().unwrap();
        write(file.path(), &records).unwrap();

        // Read back and verify special characters are preserved
        let loaded = load(file.path()).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].title, r#"Title with "quotes""#);
        assert!(loaded[0].text.contains('\n'));
        assert!(loaded[0].subject.contains(','));
    }

    #[test]
    fn test_write_round_trip() {
        // Create records with various content
        let original = vec![
            Record {
                title: "Simple Title".to_string(),
                text: "Simple text content".to_string(),
                subject: "News".to_string(),
                date: "2024-01-15".to_string(),
            },
            Record {
                title: "Complex \"Title\"".to_string(),
                text: "Multi\nline\ntext".to_string(),
                subject: "Politics, World".to_string(),
                date: "2024-02-20".to_string(),
            },
        ];

        let file = NamedTempFile::new().unwrap();
        write(file.path(), &original).unwrap();

        let loaded = load(file.path()).unwrap();
        assert_eq!(loaded.len(), original.len());
        for (orig, load) in original.iter().zip(loaded.iter()) {
            assert_eq!(orig.title, load.title);
            assert_eq!(orig.text, load.text);
            assert_eq!(orig.subject, load.subject);
            assert_eq!(orig.date, load.date);
        }
    }
}
