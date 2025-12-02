"""
Data models for model training.

Contains dataclasses for training configuration and results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainedModel:
    """Holds a trained model with its metadata.
    
    Requirements: 6.4, 7.3
    """
    name: str
    model: Any  # sklearn estimator or imblearn Pipeline
    best_params: dict
    val_f1_score: float
    classification_report: str


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Requirements: 1.1, 3.1, 4.1, 5.1
    """
    test_size: float = 0.2
    random_state: int = 42
    c_values: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    n_estimators_values: list[int] = field(default_factory=lambda: [100, 200])
    pos_label: int = 1  # Label for F1-score calculation


@dataclass
class ClassDistribution:
    """Class distribution analysis results.
    
    Requirements: 2.1, 2.2, 2.3
    """
    counts: dict[int, int]  # {0: count_fake, 1: count_real}
    percentages: dict[int, float]  # {0: pct_fake, 1: pct_real}
    imbalance_ratio: float  # majority_count / minority_count
