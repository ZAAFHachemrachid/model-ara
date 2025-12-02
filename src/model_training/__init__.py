"""
Model Training Package for Arabic Fake News Classification.

This package provides functionality for training and evaluating machine learning
classifiers for Arabic fake news detection, including:
- Data splitting with stratification
- Class distribution analysis
- Logistic Regression with balanced class weights
- Linear SVM with balanced class weights
- Random Forest with SMOTE oversampling
- Model comparison and selection
- Model serialization

Requirements: 3.1, 4.1, 5.1
"""

# Import dataclasses from models module
from .models import TrainedModel, TrainingConfig, ClassDistribution

# Import functions from submodules
from .data_utils import (
    split_data,
    analyze_class_distribution,
    compare_models,
    get_best_model,
    save_model,
    load_model,
)

# Import trainer classes
from .trainers import (
    LogisticRegressionTrainer,
    LinearSVMTrainer,
    RandomForestTrainer,
    ModelTrainer,
)

# Public exports
__all__ = [
    "TrainedModel",
    "TrainingConfig",
    "ClassDistribution",
    "split_data",
    "analyze_class_distribution",
    "compare_models",
    "get_best_model",
    "save_model",
    "load_model",
    "LogisticRegressionTrainer",
    "LinearSVMTrainer",
    "RandomForestTrainer",
    "ModelTrainer",
]
