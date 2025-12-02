# Product Overview

NLP-Fack is a Rust CLI tool for preparing news article datasets for NLP/machine learning tasks, specifically fake news detection.

## Purpose
- Processes CSV files containing news articles (fake and true news datasets)
- Cleans and prepares data by removing duplicates and balancing dataset sizes
- Outputs cleaned, balanced datasets ready for ML training

## Core Workflow
1. Load CSV files with news articles (title, text, subject, date columns)
2. Deduplicate records based on title + text matching
3. Balance datasets to equal sizes via random sampling
4. Export cleaned data to new CSV files

## Input/Output
- Input: `fake.csv`, `true.csv` (large news article datasets)
- Output: `fake_clean.csv`, `true_clean.csv` (processed, balanced datasets)
