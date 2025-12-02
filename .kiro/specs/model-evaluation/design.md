# Model Evaluation Design Document

## Overview

This document describes the design for a model evaluation and interpretation system for Arabic fake news classification. The system evaluates the best-performing model from the training phase, generates comprehensive metrics, visualizes results, and provides explainability through feature importance analysis.

The evaluation pipeline consists of:
1. **Model Loading**: Load the best trained model from disk
2. **Prediction Generation**: Generate predictions on the test set
3. **Metrics Computation**: Calculate accuracy, precision, recall, F1-score
4. **Confusion Matrix**: Visualize prediction outcomes
5. **Error Analysis**: Analyze false positives and false negatives
6. **Feature Importance**: Extract and display influential features
7. **Report Generation**: Create comprehensive evaluation report

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ModelEvaluator (Main Class)                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ MetricsComputer │  │ ConfusionMatrix │  │  ErrorAnalyzer      │  │
│  │                 │  │   Visualizer    │  │                     │  │
│  │ - accuracy()    │  │ - compute()     │  │ - get_fp_fn()       │  │
│  │ - precision()   │  │ - plot()        │  │ - fp_rate()         │  │
│  │ - recall()      │  │ - save()        │  │ - fn_rate()         │  │
│  │ - f1_score()    │  │                 │  │ - interpret()       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ FeatureImport.  │  │ ReportGenerator │  │  ResultSerializer   │  │
│  │   Extractor     │  │                 │  │                     │  │
│  │ - coefficients  │  │ - generate()    │  │ - save()            │  │
│  │ - importances   │  │ - format()      │  │ - load()            │  │
│  │ - top_features  │  │ - export()      │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Methods:                                                           │
│  - evaluate(model, X_test, y_test) -> EvaluationResult             │
│  - compute_metrics(y_true, y_pred) -> MetricsResult                │
│  - compute_confusion_matrix(y_true, y_pred) -> ConfusionMatrixData │
│  - plot_confusion_matrix(cm_data, save_path) -> Figure             │
│  - analyze_errors(cm_data) -> ErrorAnalysis                        │
│  - extract_feature_importance(model, feature_names) -> FeatureImportance │
│  - generate_report(eval_result) -> EvaluationReport                │
│  - save_results(eval_result, path) -> None                         │
│  - load_results(path) -> EvaluationResult                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. ModelEvaluator (Main Class)

Orchestrates the entire evaluation pipeline.

```python
class ModelEvaluator:
    def __init__(
        self,
        class_names: list[str] = ["Real News (0)", "Fake News (1)"],
        pos_label: int = 1
    ):
        """Initialize the model evaluator with configuration."""
    
    def evaluate(
        self,
        model: Any,
        X_test: csr_matrix,
        y_test: np.ndarray,
        feature_names: list[str] | None = None
    ) -> EvaluationResult:
        """Run complete evaluation pipeline on test set."""
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> MetricsResult:
        """Compute accuracy, precision, recall, F1-score."""
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ConfusionMatrixData:
        """Compute confusion matrix with TP, TN, FP, FN."""
    
    def plot_confusion_matrix(
        self,
        cm_data: ConfusionMatrixData,
        save_path: str | None = None
    ) -> plt.Figure:
        """Generate confusion matrix heatmap visualization."""
    
    def analyze_errors(
        self,
        cm_data: ConfusionMatrixData
    ) -> ErrorAnalysis:
        """Analyze false positives and false negatives."""
    
    def extract_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 10
    ) -> FeatureImportance:
        """Extract feature importance from model."""
    
    def generate_report(
        self,
        eval_result: EvaluationResult
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
    
    def save_results(
        self,
        eval_result: EvaluationResult,
        path: str
    ) -> None:
        """Save evaluation results to disk."""
    
    @classmethod
    def load_results(cls, path: str) -> EvaluationResult:
        """Load evaluation results from disk."""
```

### 2. MetricsComputer

Computes classification metrics.

```python
class MetricsComputer:
    def __init__(self, pos_label: int = 1):
        """Initialize with positive label for binary classification."""
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> MetricsResult:
        """Compute all metrics at once."""
    
    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list[str]
    ) -> str:
        """Generate sklearn classification report string."""
```

### 3. ConfusionMatrixVisualizer

Handles confusion matrix computation and visualization.

```python
class ConfusionMatrixVisualizer:
    def __init__(self, class_names: list[str]):
        """Initialize with class names for axis labels."""
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ConfusionMatrixData:
        """Compute confusion matrix values."""
    
    def plot(
        self,
        cm_data: ConfusionMatrixData,
        figsize: tuple[int, int] = (6, 5)
    ) -> plt.Figure:
        """Create heatmap visualization."""
    
    def save(
        self,
        fig: plt.Figure,
        path: str
    ) -> None:
        """Save figure to file."""
```

### 4. FeatureImportanceExtractor

Extracts feature importance from different model types.

```python
class FeatureImportanceExtractor:
    def extract(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 10
    ) -> FeatureImportance:
        """Extract feature importance based on model type."""
    
    def _extract_linear_coefficients(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int
    ) -> FeatureImportance:
        """Extract coefficients from linear models."""
    
    def _extract_tree_importances(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int
    ) -> FeatureImportance:
        """Extract importances from tree-based models."""
```

## Data Models

### MetricsResult

```python
@dataclass
class MetricsResult:
    accuracy: float
    precision: float  # For positive class (fake news)
    recall: float     # For positive class (fake news)
    f1_score: float   # For positive class (fake news)
    classification_report: str
```

### ConfusionMatrixData

```python
@dataclass
class ConfusionMatrixData:
    matrix: np.ndarray  # 2x2 array
    tn: int  # True Negatives
    fp: int  # False Positives
    fn: int  # False Negatives
    tp: int  # True Positives
```

### ErrorAnalysis

```python
@dataclass
class ErrorAnalysis:
    fp_count: int
    fn_count: int
    fp_rate: float  # FP / (TN + FP)
    fn_rate: float  # FN / (TP + FN)
    interpretation: str
```

### FeatureImportance

```python
@dataclass
class FeatureImportance:
    model_type: str  # "linear" or "tree"
    top_fake_features: list[tuple[str, float]]  # Features pushing towards fake
    top_real_features: list[tuple[str, float]]  # Features pushing towards real
    all_importances: dict[str, float] | None = None
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    model_name: str
    timestamp: str
    metrics: MetricsResult
    confusion_matrix: ConfusionMatrixData
    error_analysis: ErrorAnalysis
    feature_importance: FeatureImportance | None
```

### EvaluationReport

```python
@dataclass
class EvaluationReport:
    summary: str
    metrics_table: str
    confusion_breakdown: str
    error_analysis_text: str
    feature_importance_text: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Prediction count matches test set size
*For any* model and test set (X_test, y_test), the number of predictions generated SHALL equal len(y_test).
**Validates: Requirements 1.2**

### Property 2: Metrics bounds
*For any* true labels y_true and predictions y_pred, all computed metrics (accuracy, precision, recall, F1-score) SHALL be in the range [0.0, 1.0].
**Validates: Requirements 1.3**

### Property 3: Confusion matrix sum equals total samples
*For any* true labels y_true and predictions y_pred, the sum TP + TN + FP + FN SHALL equal len(y_true).
**Validates: Requirements 2.1**

### Property 4: Error rates bounds and computation
*For any* confusion matrix data, the false positive rate (FP / (TN + FP)) and false negative rate (FN / (TP + FN)) SHALL be in the range [0.0, 1.0].
**Validates: Requirements 3.2, 3.3**

### Property 5: Feature coefficients shape matches feature count
*For any* linear model with n features, the extracted coefficients array SHALL have length n.
**Validates: Requirements 4.1**

### Property 6: Random Forest importances sum to one
*For any* Random Forest model, the extracted feature importances SHALL sum to approximately 1.0 (within tolerance 0.001).
**Validates: Requirements 4.2**

### Property 7: Top features are sorted by importance
*For any* model and feature names, the top N features returned for fake news classification SHALL be sorted in descending order by absolute importance value.
**Validates: Requirements 4.3, 4.4**

### Property 8: Report completeness
*For any* evaluation result, the generated report SHALL contain all required fields: metrics (accuracy, precision, recall, F1), confusion matrix breakdown (TP, TN, FP, FN), error rates (FP rate, FN rate), and feature importances.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 9: Evaluation results serialization round-trip
*For any* evaluation result, saving then loading SHALL produce an evaluation result with identical metric values.
**Validates: Requirements 6.2, 6.4**

## Error Handling

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Model file not found | Raise `FileNotFoundError` with path information |
| Invalid model object | Raise `ValueError` with model type information |
| Empty test set | Raise `ValueError` with descriptive message |
| Mismatched X_test and y_test lengths | Raise `ValueError` with shape information |
| Model without coef_ or feature_importances_ | Return None for feature importance, log warning |
| Invalid save path | Raise `IOError` with path information |
| Corrupted saved results | Raise `ValueError` with descriptive message |
| Division by zero in rate calculation | Return 0.0 and log warning |

## Testing Strategy

### Dual Testing Approach

The system will use both unit tests and property-based tests:

1. **Unit Tests**: Verify specific examples, edge cases, and integration points
2. **Property-Based Tests**: Verify universal properties using Hypothesis library

### Property-Based Testing Framework

- **Library**: `hypothesis` (Python property-based testing library)
- **Minimum iterations**: 100 per property test
- **Test annotation format**: `**Feature: model-evaluation, Property {number}: {property_text}**`

### Unit Test Coverage

- Metrics computation with known inputs
- Confusion matrix computation
- Error analysis calculations
- Feature importance extraction for different model types
- Report generation
- Serialization/deserialization
- Visualization output

### Property Test Coverage

Each correctness property (1-9) will have a corresponding property-based test that:
1. Generates random valid inputs using Hypothesis strategies
2. Executes the evaluation operation
3. Asserts the property holds for all generated inputs

### Test File Structure

```
tests/
├── __init__.py
├── test_model_evaluator.py
├── test_metrics_computation.py
├── test_confusion_matrix.py
├── test_error_analysis.py
├── test_feature_importance.py
├── test_evaluation_serialization.py
└── test_evaluation_properties.py  # Property-based tests
```

### Integration Testing

- End-to-end evaluation pipeline with trained model
- Integration with model training output
- Visualization file generation
- Report export workflow
