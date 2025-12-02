# Farasa Integration Design Document

## Overview

This document describes the design for integrating the Farasa Arabic NLP toolkit into the existing feature extraction system. Farasa provides specialized Arabic text processing capabilities that will enhance the fake news classification pipeline with linguistically-informed features.

The integration adds a new `FarasaExtractor` component that provides:
- **Segmentation**: Breaking Arabic words into morphological components
- **Stemming**: Reducing words to their stem form
- **Lemmatization**: Finding dictionary forms of words
- **POS Tagging**: Extracting grammatical patterns as features
- **Named Entity Recognition**: Counting entity mentions by category

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FeatureExtractor (Enhanced)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐              │
│  │ TfidfExtractor  │  │LinguisticExtractor│  │SentimentExtractor│           │
│  └─────────────────┘  └──────────────────┘  └────────────────┘              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    FarasaExtractor (NEW)                    │            │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │            │
│  │  │ Segmenter    │  │ Stemmer      │  │ Lemmatizer   │       │            │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │            │
│  │  ┌──────────────┐  ┌──────────────┐                         │            │
│  │  │ POSTagger    │  │ NERTagger    │                         │            │
│  │  └──────────────┘  └──────────────┘                         │            │
│  └─────────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                   │
│  - fit(texts) -> self                                                       │
│  - transform(texts) -> sparse matrix                                        │
│  - fit_transform(texts) -> sparse matrix                                    │
│  - save(path) -> None                                                       │
│  - load(path) -> FeatureExtractor                                           │
│  - get_feature_names() -> list[str]                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. FarasaExtractor

Main class that wraps Farasa functionality and extracts features.

```python
class FarasaExtractor:
    def __init__(
        self,
        enable_segmentation: bool = True,
        enable_stemming: bool = True,
        enable_lemmatization: bool = True,
        enable_pos: bool = True,
        enable_ner: bool = True
    ):
        """Initialize Farasa extractor with configuration."""
    
    def segment(self, text: str) -> list[str]:
        """Segment Arabic text into morphological components."""
    
    def stem(self, text: str) -> list[str]:
        """Extract stems from Arabic text."""
    
    def lemmatize(self, text: str) -> list[str]:
        """Extract lemmas from Arabic text."""
    
    def pos_tag(self, text: str) -> list[tuple[str, str]]:
        """Extract POS tags for each token."""
    
    def ner(self, text: str) -> list[tuple[str, str]]:
        """Extract named entities with their categories."""
    
    def extract_features(self, text: str) -> dict[str, float]:
        """
        Extract all Farasa-based features from text.
        
        Returns dict with keys:
        - pos_noun_ratio: float
        - pos_verb_ratio: float
        - pos_adj_ratio: float
        - pos_other_ratio: float
        - ner_person_count: int
        - ner_location_count: int
        - ner_organization_count: int
        - ner_total_count: int
        - avg_segments_per_word: float
        """
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to Farasa feature matrix."""
    
    def get_feature_names(self) -> list[str]:
        """Return list of Farasa feature names."""
    
    def get_segmented_text(self, text: str) -> str:
        """Return segmented text as a string for TF-IDF."""
    
    def is_available(self) -> bool:
        """Check if Farasa is properly installed and available."""
```

### 2. FarasaSegmentPrinter

Utility class for serializing and parsing segmented text.

```python
class FarasaSegmentPrinter:
    SEGMENT_SEPARATOR: str = "+"
    WORD_SEPARATOR: str = " "
    
    @staticmethod
    def print_segments(segments: list[list[str]]) -> str:
        """Convert segments to printable string format."""
    
    @staticmethod
    def parse_segments(text: str) -> list[list[str]]:
        """Parse printed segments back to list format."""
```

### 3. Enhanced FeatureExtractor

Updates to the main FeatureExtractor class.

```python
class FeatureExtractor:
    def __init__(
        self,
        max_tfidf_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_farasa: bool = True,           # NEW
        use_segmented_tfidf: bool = False  # NEW
    ):
        """Initialize the feature extraction pipeline."""
    
    # Existing methods remain unchanged
    # New Farasa features are automatically included when use_farasa=True
```

## Data Models

### Farasa Feature Output Structure

```python
# Farasa features: dense array (n_samples, 9)
# Columns:
#   - pos_noun_ratio: float (0-1)
#   - pos_verb_ratio: float (0-1)
#   - pos_adj_ratio: float (0-1)
#   - pos_other_ratio: float (0-1)
#   - ner_person_count: int
#   - ner_location_count: int
#   - ner_organization_count: int
#   - ner_total_count: int
#   - avg_segments_per_word: float

# Combined features with Farasa: sparse matrix (n_samples, max_features + 18)
# Order: [TF-IDF..., linguistic (7)..., sentiment (2)..., farasa (9)...]
```

### Configuration Model

```python
@dataclass
class FarasaConfig:
    enable_segmentation: bool = True
    enable_stemming: bool = True
    enable_lemmatization: bool = True
    enable_pos: bool = True
    enable_ner: bool = True
    use_segmented_tfidf: bool = False
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Segmentation produces valid output
*For any* Arabic text, segmentation SHALL produce a list of segments where each word maps to one or more morphological components, and non-Arabic characters are preserved unchanged.
**Validates: Requirements 1.1, 1.3**

### Property 2: Segment round-trip consistency
*For any* segmented Arabic text, printing then parsing the segments SHALL produce an equivalent segment structure.
**Validates: Requirements 1.4**

### Property 3: Stemming produces output for all words
*For any* non-empty Arabic text, stemming SHALL produce exactly one stem per word in the input.
**Validates: Requirements 2.1**

### Property 4: Lemmatization produces output for all words
*For any* non-empty Arabic text, lemmatization SHALL produce exactly one lemma per word in the input.
**Validates: Requirements 3.1**

### Property 5: POS tagging assigns tags to all tokens
*For any* non-empty Arabic text, POS tagging SHALL assign exactly one tag per token.
**Validates: Requirements 4.1**

### Property 6: POS distribution is valid probability distribution
*For any* text, the POS ratio features (noun, verb, adj, other) SHALL sum to 1.0 (within floating point tolerance).
**Validates: Requirements 4.2, 4.3**

### Property 7: NER features are non-negative integers
*For any* text, the NER count features (person, location, organization, total) SHALL be non-negative integers, and total SHALL equal the sum of individual counts.
**Validates: Requirements 5.2, 5.3**

### Property 8: Combined feature matrix has correct dimensions
*For any* list of texts with Farasa enabled, the combined feature matrix SHALL have shape (len(texts), tfidf_features + 18) where 18 = linguistic (7) + sentiment (2) + farasa (9).
**Validates: Requirements 6.1**

### Property 9: Backward compatibility without Farasa
*For any* list of texts with Farasa disabled, the feature matrix SHALL have the same shape and values as the original pipeline (tfidf_features + 9).
**Validates: Requirements 6.2**

### Property 10: Segmented TF-IDF produces different vocabulary
*For any* Arabic text corpus, TF-IDF with segmentation enabled SHALL produce a vocabulary containing morphological segments rather than full words.
**Validates: Requirements 7.1, 7.2**

## Error Handling

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Farasa not installed | Log warning, set is_available() to False, skip Farasa features |
| Empty text input | Return zero values for ratios, zero counts for NER |
| Non-Arabic text | Pass through unchanged for segmentation, return neutral features |
| Farasa API timeout | Retry once, then return default features with warning |
| Invalid segment format | Raise ValueError with descriptive message |

## Testing Strategy

### Dual Testing Approach

The system will use both unit tests and property-based tests:

1. **Unit Tests**: Verify specific examples, edge cases, and integration points
2. **Property-Based Tests**: Verify universal properties using Hypothesis library

### Property-Based Testing Framework

- **Library**: `hypothesis` (Python property-based testing library)
- **Minimum iterations**: 100 per property test
- **Test annotation format**: `**Feature: farasa-integration, Property {number}: {property_text}**`

### Unit Test Coverage

- Empty text handling
- Non-Arabic text handling
- Known Arabic word segmentation examples
- Farasa unavailable graceful degradation
- Configuration enable/disable options
- Integration with existing FeatureExtractor

### Property Test Coverage

Each correctness property (1-10) will have a corresponding property-based test that:
1. Generates random valid inputs using Hypothesis strategies
2. Executes the Farasa extraction
3. Asserts the property holds for all generated inputs

### Test File Structure

```
tests/
├── __init__.py
├── test_farasa_extractor.py      # Unit tests for FarasaExtractor
├── test_farasa_properties.py     # Property-based tests
└── test_feature_extractor.py     # Updated integration tests
```

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "farasapy>=0.0.14",
]
```

Note: Farasa requires Java Runtime Environment (JRE) to be installed on the system.
