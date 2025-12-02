"""Metrics computation for model evaluation.

This module provides the MetricsComputer class for computing
classification metrics including accuracy, precision, recall, and F1-score.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.model_evaluation.data_models import MetricsResult


class MetricsComputer:
    """Computes classification metrics for model evaluation.
    
    This class computes accuracy, precision, recall, and F1-score
    for binary classification tasks, with focus on the positive class
    (fake news detection).
    
    Attributes:
        pos_label: The label for the positive class (default: 1 for fake news)
    """
    
    def __init__(self, pos_label: int = 1):
        """Initialize MetricsComputer with positive label configuration.
        
        Args:
            pos_label: The label value for the positive class.
                       Default is 1 (fake news in our classification task).
        """
        self.pos_label = pos_label
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list[str] | None = None,
    ) -> MetricsResult:
        """Compute all classification metrics at once.
        
        Computes accuracy, precision, recall, and F1-score for the
        positive class, along with a full classification report.
        
        Args:
            y_true: Ground truth labels as numpy array.
            y_pred: Predicted labels as numpy array.
            target_names: Optional list of class names for the report.
                         Default is ["Real News (0)", "Fake News (1)"].
        
        Returns:
            MetricsResult containing all computed metrics.
        
        Raises:
            ValueError: If y_true and y_pred have different lengths.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )
        
        if target_names is None:
            target_names = ["Real News (0)", "Fake News (1)"]
        
        # Compute metrics for positive class
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, pos_label=self.pos_label, zero_division=0.0
        )
        recall = recall_score(
            y_true, y_pred, pos_label=self.pos_label, zero_division=0.0
        )
        f1 = f1_score(
            y_true, y_pred, pos_label=self.pos_label, zero_division=0.0
        )
        
        # Generate classification report
        report = self.classification_report(y_true, y_pred, target_names)
        
        return MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            classification_report=report,
        )
    
    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list[str],
    ) -> str:
        """Generate sklearn classification report string.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            target_names: List of class names for the report.
        
        Returns:
            Formatted classification report string with per-class metrics.
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0.0,
        )
