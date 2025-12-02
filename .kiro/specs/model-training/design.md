# Model Training Design Document

## Overview

This document describes the design for a model training system that trains and evaluates machine learning classifiers for Arabic fake news detection. The system uses the combined feature matrix from the feature extraction phase and trains three models:

1. **Logistic Regression**: Linear classifier with balanced class weights
2. **Linear SVM**: Margin-based classifier with balanced class weights
3. **Random Forest**: Ensemble classifier with SMOTE oversampling

The system performs hyperparameter tuning, handles class imbalance, and selects the best model based on F1-score.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ModelTrainer (Main Class)                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  DataSplitter   │  │ ClassAnalyzer   │  │  ModelEvaluator     │  │
│  │                 │  │                 │  │                     │  │
│  │ - split()       │  │ - analyze()     │  │ - compute_f1()      │  │
│  │ - stratify      │  │ - get_counts()  │  │ - classification_   │  │
│  │                 │  │ - get_ratio()   │  │   report()          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   LRTrainer     │  │   SVMTrainer    │  │    RFTrainer        │  │
│  │                 │  │                 │  │                     │  │
│  │ - class_weight  │  │ - class_weight  │  │ - SMOTE pipeline    │  │
│  │ - C search      │  │ - C search      │  │ - n_estimators      │  │
│  │ - train()       │  │ - train()       │  │ - train()           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Methods:                                                           │
│  - split_data(X, y) -> (X_train, X_val, y_train, y_val)            │
│  - analyze_class_distribution(y) -> dict                            │
│  - train_logistic_regression(X_train, y_train, X_val, y_val)       │
│  - train_linear_svm(X_train, y_train, X_val, y_val)                │
│  - train_random_forest(X_train, y_train, X_val, y_val)             │
│  - compare_models(models) -> dict                                   │
│  - get_best_model(models) -> model                                  │
│  - save_model(model, path) -> None                                  │
│  - load_model(path) -> model                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. ModelTrainer (Main Class)

Orchestrates the entire training pipeline.

```python
class ModelTrainer:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        c_values: list[float] = [0.1, 1.0, 10.0],
        n_estimators_values: list[int] = [100, 200]
    ):
        """Initialize the model trainer with configuration."""
    
    def split_data(
        self,
        X: scipy.sparse.csr_matrix,
        y: np.ndarray
    ) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
        """Split data into training and validation sets with stratification."""
    
    def analyze_class_distribution(self, y: np.ndarray) -> dict:
        """Analyze and return class distribution statistics."""
    
    def train_all_models(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> dict[str, TrainedModel]:
        """Train all three model types and return results."""
    
    def compare_models(self, models: dict[str, TrainedModel]) -> list[dict]:
        """Compare models and return ranked list by F1-score."""
    
    def get_best_model(self, models: dict[str, TrainedModel]) -> TrainedModel:
        """Return the model with highest validation F1-score."""
    
    def save_model(self, model: TrainedModel, path: str) -> None:
        """Save trained model with metadata to disk."""
    
    @classmethod
    def load_model(cls, path: str) -> TrainedModel:
        """Load trained model from disk."""
```

### 2. TrainedModel (Data Class)

Holds a trained model with its metadata.

```python
@dataclass
class TrainedModel:
    name: str
    model: Any  # sklearn estimator or imblearn Pipeline
    best_params: dict
    val_f1_score: float
    classification_report: str
```

### 3. LogisticRegressionTrainer

Trains Logistic Regression with hyperparameter search.

```python
class LogisticRegressionTrainer:
    def __init__(self, c_values: list[float] = [0.1, 1.0, 10.0]):
        """Initialize with C values to search."""
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train LR with balanced class weights, return best model."""
```

### 4. LinearSVMTrainer

Trains Linear SVM with hyperparameter search.

```python
class LinearSVMTrainer:
    def __init__(self, c_values: list[float] = [0.1, 1.0, 10.0]):
        """Initialize with C values to search."""
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train LinearSVC with balanced class weights, return best model."""
```

### 5. RandomForestTrainer

Trains Random Forest with SMOTE pipeline.

```python
class RandomForestTrainer:
    def __init__(self, n_estimators_values: list[int] = [100, 200]):
        """Initialize with n_estimators values to search."""
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train RF with SMOTE pipeline, return best model."""
```

## Data Models

### Input Data Structure

```python
# Feature matrix from feature extraction phase
X_combined: scipy.sparse.csr_matrix  # Shape: (n_samples, n_features)

# Target labels
y: np.ndarray  # Shape: (n_samples,), values: 0 (fake) or 1 (real)
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    c_values: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    n_estimators_values: list[int] = field(default_factory=lambda: [100, 200])
    pos_label: int = 1  # Label for F1-score calculation
```

### Class Distribution Analysis

```python
@dataclass
class ClassDistribution:
    counts: dict[int, int]  # {0: count_fake, 1: count_real}
    percentages: dict[int, float]  # {0: pct_fake, 1: pct_real}
    imbalance_ratio: float  # majority_count / minority_count
```

### Model Comparison Result

```python
@dataclass
class ModelComparison:
    rankings: list[dict]  # [{"name": str, "f1_score": float}, ...]
    best_model_name: str
    best_f1_score: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Split ratio consistency
*For any* feature matrix X and labels y, splitting with test_size=0.2 SHALL produce a training set with approximately 80% of samples and validation set with approximately 20% of samples (within rounding tolerance).
**Validates: Requirements 1.1**

### Property 2: Stratified split preserves class distribution
*For any* feature matrix X and labels y with class distribution D, after stratified splitting, both training and validation sets SHALL have class distributions approximately equal to D (within 5% tolerance).
**Validates: Requirements 1.2**

### Property 3: Split reproducibility with fixed seed
*For any* feature matrix X and labels y, splitting twice with the same random_state SHALL produce identical training and validation sets.
**Validates: Requirements 1.3**

### Property 4: Class counts sum to total
*For any* label array y, the sum of class counts SHALL equal len(y) and the sum of class percentages SHALL equal 1.0.
**Validates: Requirements 2.1, 2.2**

### Property 5: F1-score bounds
*For any* model predictions and true labels, the computed F1-score SHALL be in the range [0.0, 1.0].
**Validates: Requirements 3.3, 4.3, 5.4**

### Property 6: Best model selection
*For any* set of trained models with F1-scores, the selected best model SHALL have an F1-score greater than or equal to all other models' F1-scores.
**Validates: Requirements 3.4, 4.4, 5.5, 6.1, 6.3**

### Property 7: SMOTE applied only to training data
*For any* Random Forest training with SMOTE, the validation set size SHALL remain unchanged after training (SMOTE only affects training data).
**Validates: Requirements 5.1**

### Property 8: Model serialization round-trip
*For any* trained model and input features, saving then loading the model SHALL produce identical predictions when applied to the same input features.
**Validates: Requirements 7.2, 7.4**

## Error Handling

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Empty feature matrix | Raise `ValueError` with descriptive message |
| Mismatched X and y lengths | Raise `ValueError` with shape information |
| Invalid test_size (not in 0-1) | Raise `ValueError` with valid range |
| Single class in labels | Raise `ValueError` - cannot stratify |
| Model not fitted before predict | Raise `NotFittedError` |
| Invalid file path for save/load | Raise `FileNotFoundError` or `IOError` |
| Corrupted saved model | Raise `ValueError` with descriptive message |
| SMOTE fails (too few samples) | Fall back to class weights for RF |

## Testing Strategy

### Dual Testing Approach

The system will use both unit tests and property-based tests:

1. **Unit Tests**: Verify specific examples, edge cases, and integration points
2. **Property-Based Tests**: Verify universal properties using Hypothesis library

### Property-Based Testing Framework

- **Library**: `hypothesis` (Python property-based testing library)
- **Minimum iterations**: 100 per property test
- **Test annotation format**: `**Feature: model-training, Property {number}: {property_text}**`

### Unit Test Coverage

- Data splitting with various sizes
- Class distribution analysis
- Individual model training (LR, SVM, RF)
- Model comparison and ranking
- Model serialization/deserialization
- Error handling for edge cases

### Property Test Coverage

Each correctness property (1-8) will have a corresponding property-based test that:
1. Generates random valid inputs using Hypothesis strategies
2. Executes the model training operation
3. Asserts the property holds for all generated inputs

### Test File Structure

```
tests/
├── __init__.py
├── test_model_trainer.py
├── test_data_splitting.py
├── test_class_analysis.py
├── test_model_serialization.py
└── test_training_properties.py  # Property-based tests
```

### Integration Testing

- End-to-end training pipeline with sample data
- Integration with feature extraction output
- Model persistence and reload workflow
