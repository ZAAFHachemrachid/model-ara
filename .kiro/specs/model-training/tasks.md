# Implementation Plan

- [x] 1. Set up project dependencies and structure
  - Add required dependencies to pyproject.toml: imbalanced-learn (for SMOTE)
  - Create src/model_training/ package directory structure
  - Create __init__.py with public exports
  - _Requirements: 3.1, 4.1, 5.1_

- [x] 2. Implement data splitting and class analysis
  - [x] 2.1 Create data splitting functionality
    - Implement split_data() using train_test_split with stratify parameter
    - Use test_size=0.2 and configurable random_state
    - Return X_train, X_val, y_train, y_val tuple
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ]* 2.2 Write property test for split ratio
    - **Property 1: Split ratio consistency**
    - **Validates: Requirements 1.1**
  - [ ]* 2.3 Write property test for stratified distribution
    - **Property 2: Stratified split preserves class distribution**
    - **Validates: Requirements 1.2**
  - [ ]* 2.4 Write property test for reproducibility
    - **Property 3: Split reproducibility with fixed seed**
    - **Validates: Requirements 1.3**
  - [x] 2.5 Implement class distribution analysis
    - Implement analyze_class_distribution() to compute counts and percentages
    - Calculate imbalance ratio (majority/minority)
    - Return ClassDistribution dataclass
    - _Requirements: 2.1, 2.2, 2.3_
  - [ ]* 2.6 Write property test for class counts
    - **Property 4: Class counts sum to total**
    - **Validates: Requirements 2.1, 2.2**

- [x] 3. Implement TrainedModel dataclass
  - [x] 3.1 Create TrainedModel dataclass
    - Define fields: name, model, best_params, val_f1_score, classification_report
    - Add type hints for all fields
    - _Requirements: 6.4, 7.3_

- [x] 4. Implement Logistic Regression trainer
  - [x] 4.1 Create LogisticRegressionTrainer class
    - Initialize with c_values list [0.1, 1.0, 10.0]
    - Implement train() method with hyperparameter search loop
    - Use class_weight='balanced' for all models
    - Compute F1-score for each C value and select best
    - Return TrainedModel with best configuration
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 5. Implement Linear SVM trainer
  - [x] 5.1 Create LinearSVMTrainer class
    - Initialize with c_values list [0.1, 1.0, 10.0]
    - Implement train() method with hyperparameter search loop
    - Use LinearSVC with class_weight='balanced'
    - Compute F1-score for each C value and select best
    - Return TrainedModel with best configuration
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Implement Random Forest trainer with SMOTE
  - [x] 6.1 Create RandomForestTrainer class
    - Initialize with n_estimators_values list [100, 200]
    - Implement train() method using ImbPipeline from imblearn
    - Create pipeline: SMOTE -> RandomForestClassifier
    - Compute F1-score for each n_estimators value and select best
    - Return TrainedModel with best configuration
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [ ]* 6.2 Write property test for SMOTE training data only
    - **Property 7: SMOTE applied only to training data**
    - **Validates: Requirements 5.1**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement model comparison and selection
  - [x] 8.1 Implement compare_models method
    - Accept dict of TrainedModel objects
    - Sort by val_f1_score descending
    - Return ranked list with model name and F1-score
    - _Requirements: 6.1, 6.2_
  - [x] 8.2 Implement get_best_model method
    - Return TrainedModel with highest val_f1_score
    - _Requirements: 6.3, 6.4_
  - [ ]* 8.3 Write property test for F1-score bounds
    - **Property 5: F1-score bounds**
    - **Validates: Requirements 3.3, 4.3, 5.4**
  - [ ]* 8.4 Write property test for best model selection
    - **Property 6: Best model selection**
    - **Validates: Requirements 3.4, 4.4, 5.5, 6.1, 6.3**

- [x] 9. Implement model serialization
  - [x] 9.1 Implement save_model method
    - Serialize TrainedModel to disk using joblib
    - Include all metadata (name, params, F1-score)
    - _Requirements: 7.1, 7.3_
  - [x] 9.2 Implement load_model class method
    - Restore TrainedModel from disk
    - Validate loaded model structure
    - _Requirements: 7.2_
  - [ ]* 9.3 Write property test for serialization round-trip
    - **Property 8: Model serialization round-trip**
    - **Validates: Requirements 7.2, 7.4**

- [x] 10. Implement ModelTrainer main class
  - [x] 10.1 Create ModelTrainer class
    - Initialize with TrainingConfig parameters
    - Instantiate all trainer components
    - _Requirements: 1.1, 3.1, 4.1, 5.1_
  - [x] 10.2 Implement train_all_models method
    - Call each trainer's train() method
    - Collect results into dict[str, TrainedModel]
    - Print progress and F1-scores during training
    - _Requirements: 3.1, 4.1, 5.1_

- [x] 11. Integration and main entry point
  - [x] 11.1 Update main.py with model training workflow
    - Load feature extractor and transform data
    - Split data using ModelTrainer
    - Analyze class distribution
    - Train all models
    - Compare and select best model
    - Save best model to disk
    - Display final comparison results
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1_

- [ ] 12. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
