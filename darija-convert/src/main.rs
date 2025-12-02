use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::env;
use std::path::Path;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.csv>", args[0]);
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = generate_output_filename(input_path);

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let output = File::create(&output_path)?;
    let mut writer = BufWriter::new(output);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let fields = parse_csv_line(&line);

        if i == 0 {
            writeln!(writer, "title,text,subject,date")?;
        } else if fields.len() >= 6 {
            let arabic_title = clean_text(&fields[4]);
            let arabic_text = clean_text(&fields[5]);
            let subject = &fields[2];
            let date = &fields[3];

            writeln!(
                writer,
                "{},{},{},{}",
                escape_csv(&arabic_title),
                escape_csv(&arabic_text),
                escape_csv(subject),
                escape_csv(date)
            )?;
        }
    }

    println!("Done! Output written to {}", output_path);
    Ok(())
}

fn clean_text(s: &str) -> String {
    // First pass: remove URLs and mentions using regex-like logic
    let words: Vec<&str> = s.split_whitespace().collect();
    let mut cleaned_words = Vec::new();

    for word in words {
        // Skip URLs
        if word.starts_with("http://") || word.starts_with("https://") 
           || word.starts_with("pic.twitter") || word.starts_with("t.co") 
           || word.contains("twitter.com") {
            continue;
        }
        // Skip mentions (@username) including ones in parentheses like (@user)
        if word.starts_with('@') || word.starts_with("(@") || word.contains('@') {
            continue;
        }
        cleaned_words.push(word.to_string());
    }

    let text = cleaned_words.join(" ");
    
    // Second pass: clean characters
    let mut result = String::with_capacity(text.len());
    let mut last_punct: Option<char> = None;
    let mut punct_count = 0;

    for c in text.chars() {
        // Skip emojis
        if is_emoji(c) {
            continue;
        }

        // Handle excessive punctuation
        if is_punct(c) {
            if last_punct == Some(c) {
                punct_count += 1;
                if punct_count >= 2 {
                    continue; // Skip if more than 2 of same punct
                }
            } else {
                punct_count = 1;
                last_punct = Some(c);
            }
        } else {
            last_punct = None;
            punct_count = 0;
        }

        result.push(c);
    }

    // Normalize whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_emoji(c: char) -> bool {
    let cp = c as u32;
    (0x1F600..=0x1F64F).contains(&cp) ||  // Emoticons
    (0x1F300..=0x1F5FF).contains(&cp) ||  // Misc symbols
    (0x1F680..=0x1F6FF).contains(&cp) ||  // Transport
    (0x1F1E0..=0x1F1FF).contains(&cp) ||  // Flags
    (0x2600..=0x26FF).contains(&cp) ||    // Misc symbols
    (0x2700..=0x27BF).contains(&cp) ||    // Dingbats
    (0xFE00..=0xFE0F).contains(&cp) ||    // Variation selectors
    (0x1F900..=0x1F9FF).contains(&cp)     // Supplemental
}

fn is_punct(c: char) -> bool {
    matches!(c, '!' | '?' | '.' | ',' | ';' | ':' | '-' | '_')
}



fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn generate_output_filename(input: &str) -> String {
    let path = Path::new(input);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path.extension().map(|e| e.to_string_lossy()).unwrap_or_default();
    
    // Look for pattern like _01, _02, etc. at the end of the stem
    let re_pattern = regex::Regex::new(r"^(.+?)(\s*_?\d+\s*)$").ok();
    
    if let Some(re) = re_pattern {
        if let Some(caps) = re.captures(&stem) {
            let base = caps.get(1).map(|m| m.as_str()).unwrap_or(&stem);
            let num = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            return format!("{}_done{}.{}", base.trim_end(), num, ext);
        }
    }
    
    // Fallback: just append _done before extension
    format!("{}_done.{}", stem, ext)
}
