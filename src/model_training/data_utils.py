"""
Data utilities for model training.

Provides data splitting, class distribution analysis, and model serialization functionality.
"""

import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from .models import ClassDistribution, TrainedModel


def split_data(
    X: csr_matrix,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Split data into training and validation sets with stratification.
    
    Uses stratified sampling to preserve class distribution in both sets.
    
    Args:
        X: Feature matrix (sparse CSR matrix)
        y: Target labels array
        test_size: Proportion of data for validation set (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
        
    Raises:
        ValueError: If X and y have mismatched lengths, or if test_size is invalid
        
    Requirements: 1.1, 1.2, 1.3
    """
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y have mismatched lengths: X has {X.shape[0]} samples, "
            f"y has {len(y)} samples"
        )
    
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if len(y) == 0:
        raise ValueError("Cannot split empty dataset")
    
    # Check for single class
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Cannot perform stratified split with single class. "
            f"Found only class(es): {unique_classes}"
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )
    
    return X_train, X_val, y_train, y_val


def analyze_class_distribution(y: np.ndarray) -> ClassDistribution:
    """Analyze and return class distribution statistics.
    
    Computes counts, percentages, and imbalance ratio for each class.
    
    Args:
        y: Target labels array
        
    Returns:
        ClassDistribution with counts, percentages, and imbalance_ratio
        
    Raises:
        ValueError: If y is empty
        
    Requirements: 2.1, 2.2, 2.3
    """
    if len(y) == 0:
        raise ValueError("Cannot analyze empty label array")
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    total_samples = len(y)
    
    # Build counts dict
    counts = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
    
    # Build percentages dict
    percentages = {
        int(cls): float(count / total_samples) 
        for cls, count in zip(unique_classes, class_counts)
    }
    
    # Calculate imbalance ratio (majority / minority)
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float('inf')
    
    return ClassDistribution(
        counts=counts,
        percentages=percentages,
        imbalance_ratio=imbalance_ratio
    )


def compare_models(models: dict[str, TrainedModel]) -> list[dict]:
    """Compare models and return ranked list by F1-score.
    
    Accepts a dictionary of TrainedModel objects, sorts them by validation
    F1-score in descending order, and returns a ranked list.
    
    Args:
        models: Dictionary mapping model names to TrainedModel objects
        
    Returns:
        List of dicts with 'name' and 'f1_score' keys, sorted by F1-score descending
        
    Raises:
        ValueError: If models dict is empty
        
    Requirements: 6.1, 6.2
    """
    if not models:
        raise ValueError("Cannot compare empty models dictionary")
    
    # Create list of (name, f1_score) tuples
    model_scores = [
        {"name": name, "f1_score": model.val_f1_score}
        for name, model in models.items()
    ]
    
    # Sort by F1-score descending (Req 6.1)
    ranked = sorted(model_scores, key=lambda x: x["f1_score"], reverse=True)
    
    return ranked


def get_best_model(models: dict[str, TrainedModel]) -> TrainedModel:
    """Return the model with highest validation F1-score.
    
    Args:
        models: Dictionary mapping model names to TrainedModel objects
        
    Returns:
        TrainedModel with the highest val_f1_score
        
    Raises:
        ValueError: If models dict is empty
        
    Requirements: 6.3, 6.4
    """
    if not models:
        raise ValueError("Cannot get best model from empty dictionary")
    
    # Find model with highest F1-score (Req 6.3)
    best_model = max(models.values(), key=lambda m: m.val_f1_score)
    
    return best_model


def save_model(model: TrainedModel, path: str) -> None:
    """Save trained model with metadata to disk using joblib.
    
    Serializes the complete TrainedModel including the fitted model object
    and all metadata (name, hyperparameters, F1-score, classification report).
    
    Args:
        model: TrainedModel object to save
        path: File path where the model will be saved
        
    Raises:
        ValueError: If model is None
        IOError: If the file cannot be written
        
    Requirements: 7.1, 7.3
    """
    if model is None:
        raise ValueError("Cannot save None model")
    
    # Create serializable dict with all metadata (Req 7.3)
    model_data = {
        "name": model.name,
        "model": model.model,
        "best_params": model.best_params,
        "val_f1_score": model.val_f1_score,
        "classification_report": model.classification_report,
    }
    
    # Serialize to disk using joblib (Req 7.1)
    joblib.dump(model_data, path)


def load_model(path: str) -> TrainedModel:
    """Load trained model from disk.
    
    Restores a TrainedModel from a file saved with save_model().
    Validates that the loaded data has the expected structure.
    
    Args:
        path: File path to load the model from
        
    Returns:
        TrainedModel restored from disk
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the loaded data is corrupted or has invalid structure
        
    Requirements: 7.2
    """
    # Load the serialized data
    model_data = joblib.load(path)
    
    # Validate loaded model structure
    required_keys = {"name", "model", "best_params", "val_f1_score", "classification_report"}
    if not isinstance(model_data, dict):
        raise ValueError(
            f"Corrupted model file: expected dict, got {type(model_data).__name__}"
        )
    
    missing_keys = required_keys - set(model_data.keys())
    if missing_keys:
        raise ValueError(
            f"Corrupted model file: missing required keys: {missing_keys}"
        )
    
    # Reconstruct TrainedModel from loaded data
    return TrainedModel(
        name=model_data["name"],
        model=model_data["model"],
        best_params=model_data["best_params"],
        val_f1_score=model_data["val_f1_score"],
        classification_report=model_data["classification_report"],
    )
