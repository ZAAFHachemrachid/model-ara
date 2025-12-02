# Requirements Document

## Introduction

This document specifies the requirements for a model training system for Arabic fake news classification. The system trains and evaluates three classic machine learning models—Logistic Regression, Support Vector Machine (SVM), and Random Forest—using the combined feature matrix from the feature extraction phase. The system addresses class imbalance through class weights and SMOTE, performs hyperparameter tuning, and selects the best performing model based on F1-score.

## Glossary

- **Feature Matrix**: A numerical array where rows represent documents and columns represent features (output from feature extraction phase)
- **Class Imbalance**: A condition where one class (fake or real news) has significantly more samples than the other
- **Class Weights**: Weights assigned to classes inversely proportional to their frequency to address imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique)**: An oversampling method that creates synthetic samples for the minority class
- **Hyperparameter**: A model configuration parameter set before training (e.g., regularization strength C)
- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics
- **Regularization Strength (C)**: A hyperparameter controlling the trade-off between fitting training data and model simplicity
- **Stratified Split**: A data split that preserves the class distribution in both training and validation sets
- **Cross-Validation**: A technique for evaluating model performance by training on multiple data subsets
- **Pipeline**: A sequence of data processing steps combined into a single estimator

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to split the combined feature matrix into training and validation sets, so that I can train models and evaluate their performance on unseen data.

#### Acceptance Criteria

1. WHEN splitting the data THEN the Model Trainer SHALL divide the feature matrix into 80% training and 20% validation sets
2. WHEN splitting the data THEN the Model Trainer SHALL use stratified sampling to preserve class distribution in both sets
3. WHEN splitting the data THEN the Model Trainer SHALL use a fixed random seed for reproducibility
4. WHEN the split is complete THEN the Model Trainer SHALL report the shape of training and validation feature matrices

### Requirement 2

**User Story:** As a data scientist, I want to analyze class distribution in the training data, so that I can understand the imbalance and apply appropriate techniques.

#### Acceptance Criteria

1. WHEN analyzing class distribution THEN the Model Trainer SHALL compute the count of samples per class in the training set
2. WHEN analyzing class distribution THEN the Model Trainer SHALL compute the percentage of each class in the training set
3. WHEN class imbalance is detected THEN the Model Trainer SHALL report the imbalance ratio

### Requirement 3

**User Story:** As a data scientist, I want to train a Logistic Regression model with class weight balancing, so that I can classify fake news while handling class imbalance.

#### Acceptance Criteria

1. WHEN training Logistic Regression THEN the Model Trainer SHALL use balanced class weights to address imbalance
2. WHEN tuning hyperparameters THEN the Model Trainer SHALL search over regularization strength C values [0.1, 1.0, 10.0]
3. WHEN evaluating each hyperparameter configuration THEN the Model Trainer SHALL compute F1-score on the validation set
4. WHEN hyperparameter search is complete THEN the Model Trainer SHALL select the configuration with the highest F1-score
5. WHEN the best model is selected THEN the Model Trainer SHALL generate a classification report with precision, recall, and F1-score per class

### Requirement 4

**User Story:** As a data scientist, I want to train a Linear SVM model with class weight balancing, so that I can leverage margin-based classification for fake news detection.

#### Acceptance Criteria

1. WHEN training Linear SVM THEN the Model Trainer SHALL use balanced class weights to address imbalance
2. WHEN tuning hyperparameters THEN the Model Trainer SHALL search over regularization strength C values [0.1, 1.0, 10.0]
3. WHEN evaluating each hyperparameter configuration THEN the Model Trainer SHALL compute F1-score on the validation set
4. WHEN hyperparameter search is complete THEN the Model Trainer SHALL select the configuration with the highest F1-score
5. WHEN the best model is selected THEN the Model Trainer SHALL generate a classification report with precision, recall, and F1-score per class

### Requirement 5

**User Story:** As a data scientist, I want to train a Random Forest model with SMOTE oversampling, so that I can use ensemble methods with synthetic minority samples.

#### Acceptance Criteria

1. WHEN training Random Forest THEN the Model Trainer SHALL apply SMOTE to the training data only
2. WHEN applying SMOTE THEN the Model Trainer SHALL use a pipeline to prevent data leakage to validation set
3. WHEN tuning hyperparameters THEN the Model Trainer SHALL search over n_estimators values [100, 200]
4. WHEN evaluating each hyperparameter configuration THEN the Model Trainer SHALL compute F1-score on the validation set
5. WHEN hyperparameter search is complete THEN the Model Trainer SHALL select the configuration with the highest F1-score
6. WHEN the best model is selected THEN the Model Trainer SHALL generate a classification report with precision, recall, and F1-score per class

### Requirement 6

**User Story:** As a data scientist, I want to compare all trained models and select the best one, so that I can identify the most effective classifier for fake news detection.

#### Acceptance Criteria

1. WHEN comparing models THEN the Model Trainer SHALL rank all models by their validation F1-score
2. WHEN displaying comparison THEN the Model Trainer SHALL show model name and F1-score for each model
3. WHEN comparison is complete THEN the Model Trainer SHALL identify the best performing model
4. WHEN the best model is identified THEN the Model Trainer SHALL return the fitted model object for further use

### Requirement 7

**User Story:** As a data scientist, I want to serialize the best trained model, so that I can reuse it for predictions on new data.

#### Acceptance Criteria

1. WHEN saving a trained model THEN the Model Trainer SHALL serialize the model to disk using joblib
2. WHEN loading a saved model THEN the Model Trainer SHALL restore the exact model state for consistent predictions
3. WHEN serializing THEN the Model Trainer SHALL include model metadata (name, hyperparameters, validation F1-score)
4. WHEN deserializing a model THEN the Model Trainer SHALL produce identical predictions for identical input features

