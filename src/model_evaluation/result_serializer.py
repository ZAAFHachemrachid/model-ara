"""Result serialization for model evaluation.

This module provides functionality to save and load EvaluationResult
objects to/from disk using joblib for efficient serialization.
"""

import joblib
from pathlib import Path
from typing import Any

from src.model_evaluation.data_models import (
    EvaluationResult,
    MetricsResult,
    ConfusionMatrixData,
    ErrorAnalysis,
    FeatureImportance,
)


class ResultSerializer:
    """Handles serialization and deserialization of evaluation results.
    
    Uses joblib for efficient serialization of numpy arrays and
    complex nested dataclass structures.
    """
    
    @staticmethod
    def save_results(eval_result: EvaluationResult, path: str) -> None:
        """Save evaluation results to disk using joblib.
        
        Serializes the complete EvaluationResult including model name,
        timestamp, and all computed metrics.
        
        Args:
            eval_result: The evaluation result to save
            path: File path to save the results to
            
        Raises:
            IOError: If the file cannot be written
            
        Requirements: 6.1, 6.3
        """
        save_path = Path(path)
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(eval_result, save_path)
        except Exception as e:
            raise IOError(f"Failed to save evaluation results to {path}: {e}")

    @classmethod
    def load_results(cls, path: str) -> EvaluationResult:
        """Load evaluation results from disk.
        
        Restores the EvaluationResult from a previously saved file
        and validates the loaded structure.
        
        Args:
            path: File path to load the results from
            
        Returns:
            The loaded EvaluationResult
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the loaded data is not a valid EvaluationResult
            
        Requirements: 6.2
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {path}")
        
        try:
            result = joblib.load(load_path)
        except Exception as e:
            raise ValueError(f"Failed to load evaluation results from {path}: {e}")
        
        # Validate loaded result structure
        cls._validate_result(result)
        
        return result
    
    @staticmethod
    def _validate_result(result: Any) -> None:
        """Validate that the loaded object is a valid EvaluationResult.
        
        Args:
            result: The loaded object to validate
            
        Raises:
            ValueError: If the object is not a valid EvaluationResult
        """
        if not isinstance(result, EvaluationResult):
            raise ValueError(
                f"Loaded object is not an EvaluationResult, got {type(result).__name__}"
            )
        
        # Validate required fields exist and have correct types
        if not isinstance(result.model_name, str):
            raise ValueError("EvaluationResult.model_name must be a string")
        
        if not isinstance(result.timestamp, str):
            raise ValueError("EvaluationResult.timestamp must be a string")
        
        if not isinstance(result.metrics, MetricsResult):
            raise ValueError("EvaluationResult.metrics must be a MetricsResult")
        
        if not isinstance(result.confusion_matrix, ConfusionMatrixData):
            raise ValueError(
                "EvaluationResult.confusion_matrix must be a ConfusionMatrixData"
            )
        
        if not isinstance(result.error_analysis, ErrorAnalysis):
            raise ValueError(
                "EvaluationResult.error_analysis must be an ErrorAnalysis"
            )
        
        # feature_importance can be None or FeatureImportance
        if result.feature_importance is not None:
            if not isinstance(result.feature_importance, FeatureImportance):
                raise ValueError(
                    "EvaluationResult.feature_importance must be a FeatureImportance or None"
                )
