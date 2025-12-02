"""Confusion matrix computation and visualization.

This module provides the ConfusionMatrixVisualizer class for computing
confusion matrices and generating heatmap visualizations.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model_evaluation.data_models import ConfusionMatrixData


class ConfusionMatrixVisualizer:
    """Computes and visualizes confusion matrices for model evaluation.
    
    This class handles confusion matrix computation from predictions,
    extraction of TP, TN, FP, FN values, and heatmap visualization.
    
    Attributes:
        class_names: List of class names for axis labels.
    """
    
    def __init__(self, class_names: list[str] | None = None):
        """Initialize ConfusionMatrixVisualizer with class names.
        
        Args:
            class_names: List of class names for axis labels.
                        Default is ["Real News (0)", "Fake News (1)"].
        """
        if class_names is None:
            class_names = ["Real News (0)", "Fake News (1)"]
        self.class_names = class_names
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ConfusionMatrixData:
        """Compute confusion matrix values from predictions.
        
        Computes the confusion matrix and extracts TP, TN, FP, FN values.
        For binary classification with labels [0, 1]:
        - TN: True Negatives (actual=0, predicted=0)
        - FP: False Positives (actual=0, predicted=1)
        - FN: False Negatives (actual=1, predicted=0)
        - TP: True Positives (actual=1, predicted=1)
        
        Args:
            y_true: Ground truth labels as numpy array.
            y_pred: Predicted labels as numpy array.
        
        Returns:
            ConfusionMatrixData containing the matrix and individual values.
        
        Raises:
            ValueError: If y_true and y_pred have different lengths.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )
        
        # Compute confusion matrix using sklearn
        # Returns [[TN, FP], [FN, TP]] for binary classification
        matrix = confusion_matrix(y_true, y_pred)
        
        # Extract individual values
        # sklearn confusion_matrix returns: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = matrix.ravel()
        
        return ConfusionMatrixData(
            matrix=matrix,
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )
    
    def plot(
        self,
        cm_data: ConfusionMatrixData,
        figsize: tuple[int, int] = (6, 5),
    ) -> plt.Figure:
        """Create heatmap visualization of confusion matrix.
        
        Generates a seaborn heatmap with:
        - Actual labels on y-axis
        - Predicted labels on x-axis
        - Annotated cell values
        
        Args:
            cm_data: ConfusionMatrixData containing the matrix to visualize.
            figsize: Figure size as (width, height) tuple.
        
        Returns:
            matplotlib Figure object containing the heatmap.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap using seaborn
        sns.heatmap(
            cm_data.matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        
        # Set axis labels per requirements
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix")
        
        plt.tight_layout()
        return fig
    
    def save(
        self,
        fig: plt.Figure,
        path: str,
        dpi: int = 150,
    ) -> None:
        """Save confusion matrix figure to file.
        
        Args:
            fig: matplotlib Figure to save.
            path: File path for saving the figure.
            dpi: Resolution in dots per inch.
        
        Raises:
            IOError: If the file cannot be saved.
        """
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except Exception as e:
            raise IOError(f"Failed to save confusion matrix to {path}: {e}")
