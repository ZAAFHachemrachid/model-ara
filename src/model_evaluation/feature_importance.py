"""Feature importance extraction for model evaluation.

This module provides functionality to extract feature importance
from different model types (linear models and tree-based models).
"""

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.model_evaluation.data_models import FeatureImportance


logger = logging.getLogger(__name__)


class FeatureImportanceExtractor:
    """Extracts feature importance from different model types.
    
    Supports:
    - Linear models (LogisticRegression, LinearSVC): Uses coefficients
    - Tree-based models (RandomForest): Uses feature_importances_
    """
    
    def extract(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 10
    ) -> FeatureImportance:
        """Extract feature importance based on model type.
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names matching model features
            top_n: Number of top features to return for each class
            
        Returns:
            FeatureImportance with top features for fake/real classification
            
        Raises:
            ValueError: If model type is not supported or feature count mismatch
        """
        # Detect model type and extract accordingly
        if self._is_linear_model(model):
            return self._extract_linear_coefficients(model, feature_names, top_n)
        elif self._is_tree_model(model):
            return self._extract_tree_importances(model, feature_names, top_n)
        else:
            raise ValueError(
                f"Unsupported model type: {type(model).__name__}. "
                "Supported types: LogisticRegression, LinearSVC, RandomForestClassifier"
            )

    
    def _is_linear_model(self, model: Any) -> bool:
        """Check if model is a linear model with coefficients."""
        return isinstance(model, (LogisticRegression, LinearSVC))
    
    def _is_tree_model(self, model: Any) -> bool:
        """Check if model is a tree-based model with feature importances."""
        return isinstance(model, RandomForestClassifier)
    
    def _extract_linear_coefficients(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int
    ) -> FeatureImportance:
        """Extract coefficients from linear models.
        
        For linear models, positive coefficients push towards the positive class
        (fake news, label=1) and negative coefficients push towards the negative
        class (real news, label=0).
        
        Args:
            model: Linear model with coef_ attribute
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            FeatureImportance with coefficients as importance values
        """
        if not hasattr(model, 'coef_'):
            raise ValueError(f"Model {type(model).__name__} does not have coef_ attribute")
        
        coefficients = model.coef_.flatten()
        
        if len(coefficients) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: model has {len(coefficients)} coefficients, "
                f"but {len(feature_names)} feature names provided"
            )
        
        # Create importance dict
        all_importances = {
            name: float(coef) for name, coef in zip(feature_names, coefficients)
        }
        
        # Sort by coefficient value
        # Positive coefficients -> push towards fake news (class 1)
        # Negative coefficients -> push towards real news (class 0)
        sorted_features = sorted(
            zip(feature_names, coefficients),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top features for fake news (highest positive coefficients)
        top_fake_features = [
            (name, float(coef)) for name, coef in sorted_features[:top_n]
            if coef > 0
        ]
        
        # Top features for real news (most negative coefficients)
        top_real_features = [
            (name, float(abs(coef))) for name, coef in sorted_features[-top_n:]
            if coef < 0
        ]
        # Reverse to show most important first (most negative)
        top_real_features = sorted(top_real_features, key=lambda x: x[1], reverse=True)
        
        return FeatureImportance(
            model_type="linear",
            top_fake_features=top_fake_features,
            top_real_features=top_real_features,
            all_importances=all_importances
        )

    
    def _extract_tree_importances(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int
    ) -> FeatureImportance:
        """Extract importances from tree-based models.
        
        For tree-based models like Random Forest, feature_importances_ gives
        the relative importance of each feature. These importances sum to 1.0.
        
        Note: Tree importances don't indicate direction (fake vs real),
        so we return the same top features for both classes, sorted by
        absolute importance.
        
        Args:
            model: Tree-based model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            FeatureImportance with importances as importance values
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(
                f"Model {type(model).__name__} does not have feature_importances_ attribute"
            )
        
        importances = model.feature_importances_
        
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: model has {len(importances)} importances, "
                f"but {len(feature_names)} feature names provided"
            )
        
        # Create importance dict
        all_importances = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        
        # Sort by importance (descending)
        sorted_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        # For tree models, importances don't indicate direction
        # Return top N features for both fake and real
        top_features = [
            (name, float(imp)) for name, imp in sorted_features[:top_n]
        ]
        
        return FeatureImportance(
            model_type="tree",
            top_fake_features=top_features,
            top_real_features=top_features,
            all_importances=all_importances
        )
