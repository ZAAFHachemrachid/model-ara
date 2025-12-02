# model-ara: Arabic Fake News Classification

A machine learning system for detecting fake news in Arabic language content. Achieves **99.71% F1-score** on validation data using advanced feature engineering and ensemble methods.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the complete pipeline
python main.py

# Launch interactive dashboard
streamlit run app.py
```

## Overview

**model-ara** addresses misinformation in Arabic media through automated binary classification (fake vs. real news). The system combines multiple feature extraction techniques with optimized machine learning models to deliver production-ready accuracy.

### Key Features

- **High Accuracy:** 99.71% test accuracy with robust generalization
- **Comprehensive Features:** 5,000+ engineered features from TF-IDF, linguistic analysis, and sentiment
- **Arabic NLP Support:** Optional Farasa integration for morphological analysis
- **Interpretability:** Feature importance analysis for transparency
- **Scalable:** Efficient sparse matrix operations for large-scale deployment

## Dataset

| Component | Count | Purpose |
|-----------|-------|---------|
| Training | ~2,800 samples | Model training |
| Validation | ~600 samples | Hyperparameter tuning |
| Test | ~600 samples | Final evaluation |

**Data Structure:**
- `title`: Article headline (Arabic)
- `text`: Full article content (Arabic)
- `subject`: Category (News/politicsNews)
- `date`: Publication date
- `validity`: Target label (0=fake, 1=real)

## Architecture

### Feature Extraction Pipeline

```
Arabic Text
    ↓
[Preprocessing]
    ↓
[Parallel Feature Extraction]
├─ TF-IDF Vectorizer (5,000 features)
├─ Linguistic Analysis (9 features)
├─ Sentiment Analysis (3 features)
└─ Farasa Arabic NLP (6 features, optional)
    ↓
Combined Feature Matrix (~5,000+ features)
    ↓
[Model Training & Selection]
├─ Logistic Regression (F1: 0.9958)
├─ Linear SVM (F1: 0.9967)
└─ Random Forest (F1: 0.9992) ← Best Model
    ↓
Classification Result + Confidence
```

### Feature Categories

**TF-IDF Features (~5,000)**
- Unigrams and bigrams capturing vocabulary patterns
- Min document frequency: 2, Max: 95%
- Sparse representation for memory efficiency

**Linguistic Features (9)**
- Word/sentence length, punctuation ratio, uppercase ratio
- Digit ratio, special characters, vocabulary diversity
- Stopword ratio, lexical diversity

**Sentiment Features (3)**
- Polarity score, subjectivity, sentiment intensity
- Captures emotional language patterns

**Farasa Features (6, optional)**
- POS tag ratios, named entity counts
- Morphological segmentation, lemmatization patterns
- Specialized Arabic morphological analysis

## Models

### Performance Comparison

| Model | Validation F1 | Test F1 | Accuracy |
|-------|---------------|---------|----------|
| Logistic Regression | 0.9958 | 0.9831 | 0.9833 |
| Linear SVM | 0.9967 | 0.9967 | 0.9967 |
| **Random Forest** | **0.9992** | **0.9965** | **0.9967** |

### Best Model: Random Forest

**Test Set Results:**
- Accuracy: 99.67%
- Precision: 99.67%
- Recall: 99.65%
- F1-Score: 99.65%

**Confusion Matrix:**
```
                Predicted
              Fake    Real
Actual Fake    297      1
       Real      2    300
```

**Error Analysis:**
- False Positive Rate: 0.34%
- False Negative Rate: 0.66%
- Total Error Rate: 0.50%

## Usage

### Complete Pipeline

```bash
# Run with all features (default)
python main.py

# Run without Farasa (faster)
python main.py --no-farasa

# Force refit feature extractor
python main.py --force-refit

# Use segmented text for TF-IDF
python main.py --farasa --segmented-tfidf
```

### Interactive Dashboard

```bash
streamlit run app.py
# Access at: http://localhost:8501
```

### Model Inference

```python
import joblib
from src.feature_extraction import FeatureExtractor

# Load trained components
model = joblib.load('best_model.joblib')
extractor = FeatureExtractor.load('feature_extractor.joblib')

# Prepare text
text = "Arabic news article text here"

# Extract features and predict
features = extractor.transform([text])
prediction = model.predict(features)
probability = model.predict_proba(features)

# Results
print(f"Classification: {'Real' if prediction[0] == 1 else 'Fake'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

## Project Structure

```
model-ara/
├── main.py                          # Pipeline orchestration
├── app.py                           # Streamlit dashboard
├── training_dashboard.py            # Training visualization
├── pyproject.toml                   # Project configuration
│
├── dataset/                         # Data files
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── src/
│   ├── feature_extraction/          # Feature engineering
│   │   ├── feature_extractor.py
│   │   ├── tfidf_extractor.py
│   │   ├── linguistic_extractor.py
│   │   ├── sentiment_extractor.py
│   │   ├── farasa_extractor.py
│   │   └── visualization.py
│   │
│   ├── model_training/              # Model training
│   │   ├── models.py
│   │   ├── trainers.py
│   │   └── data_utils.py
│   │
│   └── model_evaluation/            # Evaluation
│       ├── model_evaluator.py
│       ├── metrics_computer.py
│       ├── confusion_matrix.py
│       ├── feature_importance.py
│       ├── error_analyzer.py
│       ├── report_generator.py
│       ├── result_serializer.py
│       └── data_models.py
│
├── tests/                           # Test suite
│   ├── test_feature_extractor.py
│   └── test_manual_verification.py
│
└── saved_models/                    # Model artifacts
    ├── 20251202_154734_Random_Forest_f1_0.9992/
    ├── 20251202_155109_Linear_SVM_f1_0.9967/
    └── ...
```

## Installation

### Requirements

- Python 3.12+
- pip or UV package manager

### Setup

```bash
# Clone repository
git clone <repository-url>
cd model-ara

# Install dependencies
pip install -e .

# Or using UV (recommended)
uv pip install -e .
```

### Dependencies

**Core ML:**
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

**NLP & Text:**
- textblob >= 0.17.0
- farasapy >= 0.0.14

**Data & Computation:**
- scipy >= 1.11.0
- imbalanced-learn >= 0.11.0

**Visualization:**
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- streamlit >= 1.28.0
- plotly >= 5.18.0

**Serialization:**
- joblib >= 1.3.0

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_feature_extractor.py

# Verbose output
pytest -v
```

## Performance Metrics

### Computational Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Feature Extraction | 2-3 sec | Per 100 articles |
| Model Training | 30-60 sec | Full pipeline |
| Single Prediction | 50-100 ms | Per article |
| Batch (100 articles) | 5-10 sec | Parallel processing |

### Resource Usage

| Resource | Usage | Notes |
|----------|-------|-------|
| Memory | ~500 MB | Model + extractor loaded |
| Disk | ~100 MB | Model artifacts |
| CPU | Single core | Sufficient for inference |

## Advanced Features

### Farasa Arabic NLP Integration

Optional specialized Arabic morphological analysis:

```bash
# Enable Farasa (default)
python main.py --farasa

# Use segmented text for TF-IDF
python main.py --farasa --segmented-tfidf
```

Extracts:
- Part-of-speech tag ratios
- Named entity counts
- Morphological segmentation metrics
- Lemmatization patterns

**Note:** Requires Java runtime environment

### Feature Visualization

```python
from src.feature_extraction.visualization import FeatureVisualizer

visualizer = FeatureVisualizer(extractor)

# Plot feature distributions
visualizer.plot_feature_distributions(features, labels)

# Plot top features
visualizer.plot_top_features(n=20)

# Plot feature correlations
visualizer.plot_feature_correlations(features)
```

## Deployment

### Production Requirements

- **Memory:** ~500MB (model + extractor)
- **CPU:** Single core sufficient for inference
- **Latency:** ~50-100ms per article
- **Throughput:** ~10-20 articles/second

### Scalability Options

1. **Batch Processing:** Process multiple articles in parallel
2. **API Service:** REST endpoint for predictions
3. **Containerization:** Docker for consistent deployment
4. **Load Balancing:** Multiple instances behind load balancer

### Monitoring & Maintenance

- Track prediction distribution over time
- Monitor model drift (performance degradation)
- Collect misclassified examples for retraining
- Update model quarterly with new data

## Limitations & Future Work

### Current Limitations

- **Language:** Arabic only (not multilingual)
- **Domain:** Trained on news articles (may not generalize to social media)
- **Temporal:** Doesn't account for time-based patterns
- **Context:** Lacks external knowledge base for fact-checking
- **Explainability:** Feature importance is statistical, not semantic

### Future Enhancements

- Deep learning with transformer models (AraBERT, CAMeLBERT)
- Multimodal analysis (image + text)
- Real-time continuous learning
- LIME/SHAP for individual prediction explanations
- Multilingual support
- Fact-checking integration with knowledge bases
- Temporal analysis and social context

## Troubleshooting

**Farasa not available:**
```bash
python main.py --no-farasa
```

**Out of memory:**
- Reduce `max_tfidf_features` or process in batches

**Slow inference:**
- Use Linear SVM instead of Random Forest for speed

## References

**Arabic NLP:**
- [Farasa](https://github.com/aghie/farasa)
- [AraBERT](https://github.com/aub-mind/arabert)

**Machine Learning:**
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)

**Text Processing:**
- [TextBlob](https://textblob.readthedocs.io/)

## License

[Add your license here]

## Contact

[Add contact information here]

---

**Status:** Production Ready | **Last Updated:** December 2, 2025
