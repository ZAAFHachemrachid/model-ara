# Model Evaluation Package
"""
Model evaluation and interpretation system for Arabic fake news classification.

This package provides:
- MetricsComputer: Compute accuracy, precision, recall, F1-score
- ConfusionMatrixVisualizer: Generate and visualize confusion matrices
- ErrorAnalyzer: Analyze prediction errors and compute error rates
- FeatureImportanceExtractor: Extract feature importance from models
- ModelEvaluator: Main orchestrator for evaluation pipeline
"""

from src.model_evaluation.data_models import (
    MetricsResult,
    ConfusionMatrixData,
    ErrorAnalysis,
    FeatureImportance,
    EvaluationResult,
    EvaluationReport,
)
from src.model_evaluation.metrics_computer import MetricsComputer
from src.model_evaluation.confusion_matrix import ConfusionMatrixVisualizer
from src.model_evaluation.error_analyzer import ErrorAnalyzer
from src.model_evaluation.feature_importance import FeatureImportanceExtractor
from src.model_evaluation.report_generator import ReportGenerator

__all__ = [
    "MetricsResult",
    "ConfusionMatrixData",
    "ErrorAnalysis",
    "FeatureImportance",
    "EvaluationResult",
    "EvaluationReport",
    "MetricsComputer",
    "ConfusionMatrixVisualizer",
    "ErrorAnalyzer",
    "FeatureImportanceExtractor",
    "ReportGenerator",
]
