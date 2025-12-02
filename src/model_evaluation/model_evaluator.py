"""Main ModelEvaluator class for model evaluation pipeline.

This module provides the ModelEvaluator class that orchestrates the entire
evaluation pipeline, combining metrics computation, confusion matrix visualization,
error analysis, feature importance extraction, and report generation.
"""

from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

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
from src.model_evaluation.result_serializer import ResultSerializer


class ModelEvaluator:
    """Orchestrates the entire model evaluation pipeline.
    
    This class combines all evaluation components to provide a complete
    evaluation workflow for Arabic fake news classification models.
    
    Attributes:
        class_names: List of class names for display
        pos_label: Positive class label for binary classification
        metrics_computer: Component for computing classification metrics
        cm_visualizer: Component for confusion matrix computation and visualization
        error_analyzer: Component for error analysis
        feature_extractor: Component for feature importance extraction
        report_generator: Component for report generation
    """
    
    def __init__(
        self,
        class_names: list[str] | None = None,
        pos_label: int = 1
    ):
        """Initialize the ModelEvaluator with configuration.
        
        Args:
            class_names: List of class names for axis labels and reports.
                        Default is ["Real News (0)", "Fake News (1)"].
            pos_label: The label for the positive class (fake news).
                      Default is 1.
        
        Requirements: 1.1, 2.1, 3.1, 4.1
        """
        if class_names is None:
            class_names = ["Real News (0)", "Fake News (1)"]
        
        self.class_names = class_names
        self.pos_label = pos_label
        
        # Instantiate all component classes
        self.metrics_computer = MetricsComputer(pos_label=pos_label)
        self.cm_visualizer = ConfusionMatrixVisualizer(class_names=class_names)
        self.error_analyzer = ErrorAnalyzer()
        self.feature_extractor = FeatureImportanceExtractor()
        self.report_generator = ReportGenerator()


    def evaluate(
        self,
        model: Any,
        X_test: csr_matrix | np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        feature_names: list[str] | None = None
    ) -> EvaluationResult:
        """Run complete evaluation pipeline on test set.
        
        Generates predictions, computes all metrics, creates confusion matrix,
        analyzes errors, and extracts feature importance (if feature_names provided).
        
        Args:
            model: Trained sklearn model with predict() method
            X_test: Test set features (sparse matrix or numpy array)
            y_test: Test set labels as numpy array
            model_name: Name/identifier for the model being evaluated
            feature_names: Optional list of feature names for importance extraction
        
        Returns:
            Complete EvaluationResult with all metrics and analysis
        
        Raises:
            ValueError: If model doesn't have predict method or inputs are invalid
        
        Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 4.1
        """
        # Validate model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError(
                f"Model {type(model).__name__} does not have a predict method"
            )
        
        # Validate test set
        if len(y_test) == 0:
            raise ValueError("Test set is empty")
        
        # Get number of samples from X_test
        n_samples = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
        if n_samples != len(y_test):
            raise ValueError(
                f"X_test and y_test must have the same number of samples. "
                f"Got {n_samples} and {len(y_test)}."
            )
        
        # Generate predictions on test set (Requirement 1.2)
        y_pred = model.predict(X_test)
        
        # Compute all metrics (Requirement 1.3)
        metrics = self.compute_metrics(y_test, y_pred)
        
        # Compute confusion matrix (Requirement 2.1)
        cm_data = self.compute_confusion_matrix(y_test, y_pred)
        
        # Analyze errors (Requirement 3.1)
        error_analysis = self.analyze_errors(cm_data)
        
        # Extract feature importance if feature_names provided (Requirement 4.1)
        feature_importance = None
        if feature_names is not None:
            try:
                feature_importance = self.extract_feature_importance(
                    model, feature_names
                )
            except ValueError:
                # Model doesn't support feature importance extraction
                feature_importance = None
        
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        return EvaluationResult(
            model_name=model_name,
            timestamp=timestamp,
            metrics=metrics,
            confusion_matrix=cm_data,
            error_analysis=error_analysis,
            feature_importance=feature_importance,
        )
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> MetricsResult:
        """Compute accuracy, precision, recall, F1-score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            MetricsResult with all computed metrics
        """
        return self.metrics_computer.compute_all(
            y_true, y_pred, target_names=self.class_names
        )
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ConfusionMatrixData:
        """Compute confusion matrix with TP, TN, FP, FN.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            ConfusionMatrixData with matrix and individual values
        """
        return self.cm_visualizer.compute(y_true, y_pred)
    
    def plot_confusion_matrix(
        self,
        cm_data: ConfusionMatrixData,
        save_path: str | None = None
    ) -> plt.Figure:
        """Generate confusion matrix heatmap visualization.
        
        Args:
            cm_data: ConfusionMatrixData to visualize
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure with the heatmap
        """
        fig = self.cm_visualizer.plot(cm_data)
        
        if save_path is not None:
            self.cm_visualizer.save(fig, save_path)
        
        return fig
    
    def analyze_errors(
        self,
        cm_data: ConfusionMatrixData
    ) -> ErrorAnalysis:
        """Analyze false positives and false negatives.
        
        Args:
            cm_data: ConfusionMatrixData with TP, TN, FP, FN values
        
        Returns:
            ErrorAnalysis with counts, rates, and interpretation
        """
        return self.error_analyzer.analyze_errors(cm_data)
    
    def extract_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 10
    ) -> FeatureImportance:
        """Extract feature importance from model.
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            FeatureImportance with top features for fake/real classification
        
        Raises:
            ValueError: If model type is not supported
        """
        return self.feature_extractor.extract(model, feature_names, top_n)
    
    def generate_report(
        self,
        eval_result: EvaluationResult
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report.
        
        Args:
            eval_result: Complete evaluation result
        
        Returns:
            EvaluationReport with formatted text for all sections
        """
        return self.report_generator.generate_report(eval_result)
    
    def save_results(
        self,
        eval_result: EvaluationResult,
        path: str
    ) -> None:
        """Save evaluation results to disk.
        
        Args:
            eval_result: The evaluation result to save
            path: File path to save the results to
        """
        ResultSerializer.save_results(eval_result, path)
    
    @classmethod
    def load_results(cls, path: str) -> EvaluationResult:
        """Load evaluation results from disk.
        
        Args:
            path: File path to load the results from
        
        Returns:
            The loaded EvaluationResult
        """
        return ResultSerializer.load_results(path)
