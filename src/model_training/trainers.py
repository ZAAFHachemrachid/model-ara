"""
Model trainers for Arabic fake news classification.

Provides trainer classes for different ML models with hyperparameter search.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .models import TrainedModel


class LogisticRegressionTrainer:
    """Trains Logistic Regression with hyperparameter search.
    
    Uses balanced class weights to address class imbalance and searches
    over regularization strength C values to find the best configuration.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    def __init__(self, c_values: list[float] | None = None):
        """Initialize with C values to search.
        
        Args:
            c_values: List of regularization strength values to search.
                     Defaults to [0.1, 1.0, 10.0]
        """
        self.c_values = c_values if c_values is not None else [0.1, 1.0, 10.0]
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train Logistic Regression with balanced class weights.
        
        Performs hyperparameter search over C values and selects the
        configuration with the highest F1-score on the validation set.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix
            y_val: Validation labels
            
        Returns:
            TrainedModel with best configuration
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
        """
        best_model = None
        best_f1 = -1.0
        best_c = None
        
        # Hyperparameter search over C values
        for c in self.c_values:
            # Create model with balanced class weights (Req 3.1)
            model = LogisticRegression(
                C=c,
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs',
                random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Compute F1-score on validation set (Req 3.3)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Track best model (Req 3.4)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_c = c
        
        # Generate classification report for best model (Req 3.5)
        y_pred_best = best_model.predict(X_val)
        report = classification_report(y_val, y_pred_best)
        
        return TrainedModel(
            name="LogisticRegression",
            model=best_model,
            best_params={"C": best_c},
            val_f1_score=best_f1,
            classification_report=report
        )


class LinearSVMTrainer:
    """Trains Linear SVM with hyperparameter search.
    
    Uses balanced class weights to address class imbalance and searches
    over regularization strength C values to find the best configuration.
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    
    def __init__(self, c_values: list[float] | None = None):
        """Initialize with C values to search.
        
        Args:
            c_values: List of regularization strength values to search.
                     Defaults to [0.1, 1.0, 10.0]
        """
        self.c_values = c_values if c_values is not None else [0.1, 1.0, 10.0]
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train Linear SVM with balanced class weights.
        
        Performs hyperparameter search over C values and selects the
        configuration with the highest F1-score on the validation set.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix
            y_val: Validation labels
            
        Returns:
            TrainedModel with best configuration
            
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        best_model = None
        best_f1 = -1.0
        best_c = None
        
        # Hyperparameter search over C values (Req 4.2)
        for c in self.c_values:
            # Create model with balanced class weights (Req 4.1)
            model = LinearSVC(
                C=c,
                class_weight='balanced',
                max_iter=1000,
                dual='auto',
                random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Compute F1-score on validation set (Req 4.3)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Track best model (Req 4.4)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_c = c
        
        # Generate classification report for best model (Req 4.5)
        y_pred_best = best_model.predict(X_val)
        report = classification_report(y_val, y_pred_best)
        
        return TrainedModel(
            name="LinearSVM",
            model=best_model,
            best_params={"C": best_c},
            val_f1_score=best_f1,
            classification_report=report
        )


class RandomForestTrainer:
    """Trains Random Forest with SMOTE pipeline.
    
    Uses SMOTE oversampling on training data only via imblearn Pipeline
    to prevent data leakage. Searches over n_estimators values to find
    the best configuration.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    
    def __init__(self, n_estimators_values: list[int] | None = None):
        """Initialize with n_estimators values to search.
        
        Args:
            n_estimators_values: List of n_estimators values to search.
                                Defaults to [100, 200]
        """
        self.n_estimators_values = n_estimators_values if n_estimators_values is not None else [100, 200]
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> TrainedModel:
        """Train Random Forest with SMOTE pipeline.
        
        Creates an imblearn Pipeline with SMOTE -> RandomForestClassifier.
        SMOTE is applied only to training data during fit(), preventing
        data leakage to the validation set.
        
        Performs hyperparameter search over n_estimators values and selects
        the configuration with the highest F1-score on the validation set.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix
            y_val: Validation labels
            
        Returns:
            TrainedModel with best configuration
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
        """
        best_pipeline = None
        best_f1 = -1.0
        best_n_estimators = None
        
        # Hyperparameter search over n_estimators values (Req 5.3)
        for n_estimators in self.n_estimators_values:
            # Create SMOTE -> RandomForest pipeline (Req 5.1, 5.2)
            # Using ImbPipeline ensures SMOTE is applied only to training data
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('rf', RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Train the pipeline (SMOTE applied only to training data)
            pipeline.fit(X_train, y_train)
            
            # Compute F1-score on validation set (Req 5.4)
            # Note: SMOTE is NOT applied to validation data during predict
            y_pred = pipeline.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Track best pipeline (Req 5.5)
            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipeline
                best_n_estimators = n_estimators
        
        # Generate classification report for best model (Req 5.6)
        y_pred_best = best_pipeline.predict(X_val)
        report = classification_report(y_val, y_pred_best)
        
        return TrainedModel(
            name="RandomForest",
            model=best_pipeline,
            best_params={"n_estimators": best_n_estimators},
            val_f1_score=best_f1,
            classification_report=report
        )


class ModelTrainer:
    """Main orchestrator class for the model training pipeline.
    
    Coordinates data splitting, class analysis, and training of all model types.
    Provides a unified interface for training multiple classifiers and selecting
    the best performing model.
    
    Requirements: 1.1, 3.1, 4.1, 5.1
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        c_values: list[float] | None = None,
        n_estimators_values: list[int] | None = None
    ):
        """Initialize the model trainer with configuration.
        
        Args:
            test_size: Proportion of data for validation set (default 0.2)
            random_state: Random seed for reproducibility
            c_values: List of C values for LR and SVM hyperparameter search.
                     Defaults to [0.1, 1.0, 10.0]
            n_estimators_values: List of n_estimators for RF hyperparameter search.
                                Defaults to [100, 200]
                                
        Requirements: 1.1, 3.1, 4.1, 5.1
        """
        self.test_size = test_size
        self.random_state = random_state
        self.c_values = c_values if c_values is not None else [0.1, 1.0, 10.0]
        self.n_estimators_values = n_estimators_values if n_estimators_values is not None else [100, 200]
        
        # Instantiate all trainer components
        self.lr_trainer = LogisticRegressionTrainer(c_values=self.c_values)
        self.svm_trainer = LinearSVMTrainer(c_values=self.c_values)
        self.rf_trainer = RandomForestTrainer(n_estimators_values=self.n_estimators_values)
    
    def train_all_models(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray
    ) -> dict[str, TrainedModel]:
        """Train all three model types and return results.
        
        Calls each trainer's train() method, collects results, and prints
        progress and F1-scores during training.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix
            y_val: Validation labels
            
        Returns:
            Dictionary mapping model names to TrainedModel objects
            
        Requirements: 3.1, 4.1, 5.1
        """
        models: dict[str, TrainedModel] = {}
        
        # Train Logistic Regression (Req 3.1)
        print("Training Logistic Regression...")
        lr_model = self.lr_trainer.train(X_train, y_train, X_val, y_val)
        models[lr_model.name] = lr_model
        print(f"  Best C: {lr_model.best_params['C']}, F1-score: {lr_model.val_f1_score:.4f}")
        
        # Train Linear SVM (Req 4.1)
        print("Training Linear SVM...")
        svm_model = self.svm_trainer.train(X_train, y_train, X_val, y_val)
        models[svm_model.name] = svm_model
        print(f"  Best C: {svm_model.best_params['C']}, F1-score: {svm_model.val_f1_score:.4f}")
        
        # Train Random Forest with SMOTE (Req 5.1)
        print("Training Random Forest with SMOTE...")
        rf_model = self.rf_trainer.train(X_train, y_train, X_val, y_val)
        models[rf_model.name] = rf_model
        print(f"  Best n_estimators: {rf_model.best_params['n_estimators']}, F1-score: {rf_model.val_f1_score:.4f}")
        
        print("\nAll models trained successfully!")
        return models
