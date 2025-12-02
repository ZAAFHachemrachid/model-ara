# Product Overview

model-ara is a machine learning project for Arabic text classification, specifically focused on news validity detection (fake news vs. real news).

## Purpose
- Binary classification of Arabic news articles
- Distinguishes between valid (1) and invalid (0) news content
- Dataset includes political news and general news articles

## Data Structure
The dataset contains:
- `title`: Article headline (Arabic)
- `text`: Full article content (Arabic)
- `subject`: Category (News, politicsNews)
- `date`: Publication date
- `validity`: Target label (0 = fake, 1 = real)
