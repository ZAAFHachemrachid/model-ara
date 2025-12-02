# Feature Extraction Design Document

## Overview

This document describes the design for a feature extraction system that transforms Arabic news text into numerical features for fake news classification. The system extracts three types of features:

1. **TF-IDF Features**: Capture word importance using term frequency-inverse document frequency
2. **Linguistic Features**: Capture writing style patterns (length, punctuation, sensationalism indicators)
3. **Sentiment Features**: Capture emotional tone (polarity, subjectivity)

The extracted features are combined into a unified feature matrix suitable for machine learning models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FeatureExtractor (Main Class)                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ TfidfExtractor  │  │LinguisticExtractor│  │SentimentExtractor│
│  │                 │  │                  │  │                │  │
│  │ - vectorizer    │  │ - keywords       │  │ - analyzer     │  │
│  │ - fit()         │  │ - extract()      │  │ - extract()    │  │
│  │ - transform()   │  │                  │  │                │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Methods:                                                       │
│  - fit(texts) -> self                                          │
│  - transform(texts) -> sparse matrix                           │
│  - fit_transform(texts) -> sparse matrix                       │
│  - save(path) -> None                                          │
│  - load(path) -> FeatureExtractor                              │
│  - get_feature_names() -> list[str]                            │
│  - analyze_features(features, labels) -> dict                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. TfidfExtractor

Wraps scikit-learn's TfidfVectorizer for text vectorization.

```python
class TfidfExtractor:
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """Initialize TF-IDF extractor with configuration."""
    
    def fit(self, texts: list[str]) -> "TfidfExtractor":
        """Fit the vectorizer on training texts."""
    
    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Transform texts to TF-IDF feature matrix."""
    
    def get_feature_names(self) -> list[str]:
        """Return list of feature names (vocabulary terms)."""
```

### 2. LinguisticExtractor

Extracts writing style features from text.

```python
class LinguisticExtractor:
    SENSATIONALISM_KEYWORDS: list[str] = [
        'shocking', 'unbelievable', "you won't believe", 'breaking',
        'urgent', 'alert', 'must see', 'goes viral', 'exposed',
        # Arabic equivalents
        'صادم', 'لن تصدق', 'عاجل', 'خطير', 'فضيحة', 'مفاجأة'
    ]
    
    def extract(self, text: str) -> dict[str, float]:
        """
        Extract linguistic features from a single text.
        
        Returns dict with keys:
        - text_length: int
        - word_count: int
        - uppercase_ratio: float
        - exclamation_count: int
        - question_count: int
        - punctuation_ratio: float
        - fake_keyword_count: int
        """
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to linguistic feature matrix."""
    
    def get_feature_names(self) -> list[str]:
        """Return list of linguistic feature names."""
```

### 3. SentimentExtractor

Extracts sentiment features using TextBlob.

```python
class SentimentExtractor:
    def extract(self, text: str) -> dict[str, float | str]:
        """
        Extract sentiment features from a single text.
        
        Returns dict with keys:
        - polarity: float (-1 to 1)
        - subjectivity: float (0 to 1)
        - sentiment: str ('positive', 'negative', 'neutral')
        """
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to sentiment feature matrix (polarity, subjectivity)."""
    
    def get_feature_names(self) -> list[str]:
        """Return list of sentiment feature names."""
```

### 4. FeatureExtractor (Main Class)

Orchestrates all extractors and combines features.

```python
class FeatureExtractor:
    def __init__(
        self,
        max_tfidf_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """Initialize the feature extraction pipeline."""
    
    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """Fit TF-IDF extractor on training texts."""
    
    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Transform texts to combined feature matrix."""
    
    def fit_transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Fit and transform in one step."""
    
    def get_feature_names(self) -> list[str]:
        """Return all feature names in column order."""
    
    def save(self, path: str) -> None:
        """Save fitted pipeline to disk using joblib."""
    
    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load fitted pipeline from disk."""
    
    def analyze_features(
        self,
        features: scipy.sparse.csr_matrix,
        labels: np.ndarray
    ) -> dict:
        """Compute feature statistics grouped by label."""
    
    def get_top_tfidf_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Return top N TF-IDF features by total weight."""
```

## Data Models

### Feature Output Structure

```python
# TF-IDF features: sparse matrix (n_samples, max_features)
# Shape: (num_documents, 5000) by default

# Linguistic features: dense array (n_samples, 7)
# Columns: text_length, word_count, uppercase_ratio, 
#          exclamation_count, question_count, punctuation_ratio, fake_keyword_count

# Sentiment features: dense array (n_samples, 2)
# Columns: polarity, subjectivity

# Combined features: sparse matrix (n_samples, max_features + 9)
# Order: [TF-IDF features..., linguistic features..., sentiment features...]
```

### Configuration Model

```python
@dataclass
class FeatureExtractorConfig:
    max_tfidf_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sensationalism_keywords: list[str] = field(default_factory=list)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following properties can be combined and refined:

### Property 1: TF-IDF output shape consistency
*For any* list of texts and configured max_features, the TF-IDF transform output SHALL have shape (len(texts), <= max_features) and be a sparse matrix.
**Validates: Requirements 1.1, 1.2**

### Property 2: TF-IDF vocabulary persistence
*For any* fitted TF-IDF extractor, transforming new text SHALL use the same vocabulary as the training data, producing consistent feature dimensions.
**Validates: Requirements 1.5**

### Property 3: Linguistic feature text metrics
*For any* non-empty text, the extracted text_length SHALL equal len(text) and word_count SHALL equal len(text.split()).
**Validates: Requirements 2.1, 2.2**

### Property 4: Linguistic feature ratio bounds
*For any* text, uppercase_ratio and punctuation_ratio SHALL be in the range [0, 1].
**Validates: Requirements 2.3, 2.5**

### Property 5: Linguistic feature punctuation counts
*For any* text, exclamation_count SHALL equal text.count('!') and question_count SHALL equal text.count('?').
**Validates: Requirements 2.4**

### Property 6: Sentiment polarity bounds
*For any* text, the computed polarity SHALL be in the range [-1, 1].
**Validates: Requirements 3.1**

### Property 7: Sentiment subjectivity bounds
*For any* text, the computed subjectivity SHALL be in the range [0, 1].
**Validates: Requirements 3.2**

### Property 8: Sentiment classification consistency
*For any* polarity value, the sentiment label SHALL be 'positive' if polarity > 0.1, 'negative' if polarity < -0.1, and 'neutral' otherwise.
**Validates: Requirements 3.3, 3.4, 3.5**

### Property 9: Combined feature matrix dimensions
*For any* list of texts, the combined feature matrix SHALL have shape (len(texts), tfidf_features + 9) where 9 is the count of linguistic (7) and sentiment (2) features.
**Validates: Requirements 4.1**

### Property 10: Pipeline serialization round-trip
*For any* fitted FeatureExtractor and input text, saving then loading the pipeline SHALL produce identical feature vectors when transforming the same text.
**Validates: Requirements 5.2, 5.4**

### Property 11: Top TF-IDF features ordering
*For any* request for top N features, the returned list SHALL be sorted by weight in descending order.
**Validates: Requirements 6.2**

## Error Handling

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Empty text input | Return zero values for ratios, empty counts |
| Empty text list | Return empty sparse matrix with correct shape |
| Transform before fit | Raise `NotFittedError` |
| Invalid file path for save/load | Raise `FileNotFoundError` or `IOError` |
| Corrupted saved model | Raise `ValueError` with descriptive message |

## Testing Strategy

### Dual Testing Approach

The system will use both unit tests and property-based tests:

1. **Unit Tests**: Verify specific examples, edge cases, and integration points
2. **Property-Based Tests**: Verify universal properties using Hypothesis library

### Property-Based Testing Framework

- **Library**: `hypothesis` (Python property-based testing library)
- **Minimum iterations**: 100 per property test
- **Test annotation format**: `**Feature: feature-extraction, Property {number}: {property_text}**`

### Unit Test Coverage

- Empty text handling
- Single document processing
- Multiple document batch processing
- Pipeline save/load functionality
- Feature name retrieval
- Integration with pandas DataFrames

### Property Test Coverage

Each correctness property (1-11) will have a corresponding property-based test that:
1. Generates random valid inputs using Hypothesis strategies
2. Executes the feature extraction
3. Asserts the property holds for all generated inputs

### Test File Structure

```
tests/
├── __init__.py
├── test_tfidf_extractor.py
├── test_linguistic_extractor.py
├── test_sentiment_extractor.py
├── test_feature_extractor.py
└── test_properties.py  # Property-based tests
```
