"""Data models for model evaluation results.

This module defines dataclasses for storing evaluation metrics,
confusion matrix data, error analysis, feature importance, and
comprehensive evaluation results.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class MetricsResult:
    """Classification metrics for model evaluation.
    
    Attributes:
        accuracy: Overall accuracy (correct predictions / total)
        precision: Precision for positive class (TP / (TP + FP))
        recall: Recall for positive class (TP / (TP + FN))
        f1_score: F1-score for positive class (harmonic mean of precision and recall)
        classification_report: Formatted string with per-class metrics
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    classification_report: str


@dataclass
class ConfusionMatrixData:
    """Confusion matrix data with individual cell values.
    
    Attributes:
        matrix: 2x2 numpy array with confusion matrix values
        tn: True Negatives count
        fp: False Positives count
        fn: False Negatives count
        tp: True Positives count
    """
    matrix: np.ndarray
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass
class ErrorAnalysis:
    """Error analysis results with rates and interpretation.
    
    Attributes:
        fp_count: Number of false positives
        fn_count: Number of false negatives
        fp_rate: False positive rate (FP / (TN + FP))
        fn_rate: False negative rate (FN / (TP + FN))
        interpretation: Text explaining error criticality for fake news detection
    """
    fp_count: int
    fn_count: int
    fp_rate: float
    fn_rate: float
    interpretation: str


@dataclass
class FeatureImportance:
    """Feature importance data from model.
    
    Attributes:
        model_type: Type of model ("linear" or "tree")
        top_fake_features: List of (feature_name, importance) tuples for fake news
        top_real_features: List of (feature_name, importance) tuples for real news
        all_importances: Optional dict mapping feature names to importance values
    """
    model_type: str
    top_fake_features: list[tuple[str, float]]
    top_real_features: list[tuple[str, float]]
    all_importances: dict[str, float] | None = None


@dataclass
class EvaluationResult:
    """Complete evaluation result combining all metrics and analysis.
    
    Attributes:
        model_name: Name/identifier of the evaluated model
        timestamp: ISO format timestamp of evaluation
        metrics: Classification metrics (accuracy, precision, recall, F1)
        confusion_matrix: Confusion matrix data with TP, TN, FP, FN
        error_analysis: Error analysis with rates and interpretation
        feature_importance: Optional feature importance data
    """
    model_name: str
    timestamp: str
    metrics: MetricsResult
    confusion_matrix: ConfusionMatrixData
    error_analysis: ErrorAnalysis
    feature_importance: FeatureImportance | None = None


@dataclass
class EvaluationReport:
    """Formatted evaluation report for display and export.
    
    Attributes:
        summary: Brief summary of evaluation results
        metrics_table: Formatted table with accuracy, precision, recall, F1
        confusion_breakdown: Formatted confusion matrix breakdown
        error_analysis_text: Formatted error analysis text
        feature_importance_text: Formatted feature importance text
    """
    summary: str
    metrics_table: str
    confusion_breakdown: str
    error_analysis_text: str
    feature_importance_text: str
