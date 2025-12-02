"""Main entry point for Arabic fake news classification.

This module demonstrates the complete workflow:
1. Load dataset from CSV files
2. Create and fit FeatureExtractor on training data (or load existing)
3. Transform train/validation/test sets
4. Split data for model training
5. Analyze class distribution
6. Train all models (Logistic Regression, Linear SVM, Random Forest)
7. Compare and select best model
8. Save best model to disk
9. Display final comparison results
10. Evaluate best model on test set
11. Generate and display evaluation report

Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1
Model Evaluation Requirements: 1.1, 1.2, 1.3, 2.2, 3.1, 4.3, 5.5, 6.1
Farasa Integration Requirements: 6.4
"""

import argparse
import os
import pandas as pd
import numpy as np

from src.feature_extraction import FeatureExtractor
from src.model_training import (
    ModelTrainer,
    split_data,
    analyze_class_distribution,
    compare_models,
    get_best_model,
    save_model,
    load_model,
)
from src.model_evaluation import ModelEvaluator


def load_dataset(data_dir: str = "dataset") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files.
        
    Returns:
        Tuple of (train_df, validation_df, test_df) DataFrames.
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    validation_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    return train_df, validation_df, test_df


def get_texts(df: pd.DataFrame) -> list[str]:
    """Extract text content from DataFrame.
    
    Combines title and text columns for feature extraction.
    
    Args:
        df: DataFrame with 'title' and 'text' columns.
        
    Returns:
        List of combined text strings.
    """
    # Combine title and text for richer feature extraction
    # Handle missing values by filling with empty string
    titles = df["title"].fillna("")
    texts = df["text"].fillna("")
    combined = titles + " " + texts
    return combined.tolist()


def display_feature_analysis(
    extractor: FeatureExtractor,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Display feature analysis statistics.
    
    Args:
        extractor: Fitted FeatureExtractor instance.
        features: Feature matrix from transform().
        labels: Array of labels (0=fake, 1=real).
    """
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    analysis = extractor.analyze_features(features, labels)
    
    # Get feature names for linguistic and sentiment features
    feature_names = extractor.get_feature_names()
    n_tfidf = len(extractor._tfidf_extractor.get_feature_names())
    
    # Focus on linguistic and sentiment features (last 9 features)
    manual_feature_names = feature_names[n_tfidf:]
    
    label_names = {0: "Fake News", 1: "Real News"}
    
    for label, stats in analysis.items():
        print(f"\n{label_names.get(label, f'Label {label}')}:")
        print("-" * 40)
        
        # Display stats for manual features only (more interpretable)
        for i, name in enumerate(manual_feature_names):
            idx = n_tfidf + i
            mean_val = stats["mean"][idx]
            std_val = stats["std"][idx]
            print(f"  {name:20s}: mean={mean_val:8.3f}, std={std_val:8.3f}")


def display_top_tfidf_features(extractor: FeatureExtractor, n: int = 20) -> None:
    """Display top TF-IDF features by weight.
    
    Args:
        extractor: Fitted FeatureExtractor instance.
        n: Number of top features to display.
    """
    print("\n" + "=" * 60)
    print(f"TOP {n} TF-IDF FEATURES BY WEIGHT")
    print("=" * 60)
    
    top_features = extractor.get_top_tfidf_features(n=n)
    
    for i, (name, weight) in enumerate(top_features, 1):
        print(f"  {i:2d}. {name:30s} (weight: {weight:.4f})")


def display_farasa_statistics(
    extractor: FeatureExtractor,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Display Farasa feature statistics.
    
    Shows statistics for Farasa-specific features including POS ratios,
    NER counts, and segmentation metrics, grouped by label.
    
    Args:
        extractor: Fitted FeatureExtractor instance with Farasa enabled.
        features: Feature matrix from transform().
        labels: Array of labels (0=fake, 1=real).
        
    Requirements: 6.4
    """
    # Check if Farasa is available
    if not extractor._farasa_available:
        print("\n" + "=" * 60)
        print("FARASA FEATURE STATISTICS")
        print("=" * 60)
        print("\n  ⚠️  Farasa is not available. Farasa features disabled.")
        return
    
    print("\n" + "=" * 60)
    print("FARASA FEATURE STATISTICS")
    print("=" * 60)
    
    # Get feature names and identify Farasa feature indices
    feature_names = extractor.get_feature_names()
    farasa_feature_names = extractor._farasa_extractor.get_feature_names()
    
    # Find the starting index of Farasa features
    n_tfidf = len(extractor._tfidf_extractor.get_feature_names())
    n_linguistic = len(extractor._linguistic_extractor.get_feature_names())
    n_sentiment = len(extractor._sentiment_extractor.get_feature_names())
    farasa_start_idx = n_tfidf + n_linguistic + n_sentiment
    
    # Convert sparse matrix to dense for Farasa features only
    dense_features = features.toarray()
    labels_array = np.asarray(labels)
    
    label_names = {0: "Fake News", 1: "Real News"}
    
    print("\n  Farasa Features Summary:")
    print("-" * 60)
    
    for label in [0, 1]:
        mask = labels_array == label
        label_features = dense_features[mask]
        
        print(f"\n  {label_names.get(label, f'Label {label}')}:")
        print("  " + "-" * 50)
        
        for i, name in enumerate(farasa_feature_names):
            idx = farasa_start_idx + i
            values = label_features[:, idx]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"    {name:25s}: mean={mean_val:7.3f}, std={std_val:6.3f}, "
                  f"min={min_val:6.2f}, max={max_val:7.2f}")
    
    # Display overall Farasa feature comparison between classes
    print("\n  Feature Differences (Real - Fake):")
    print("  " + "-" * 50)
    
    fake_mask = labels_array == 0
    real_mask = labels_array == 1
    
    for i, name in enumerate(farasa_feature_names):
        idx = farasa_start_idx + i
        fake_mean = np.mean(dense_features[fake_mask, idx])
        real_mean = np.mean(dense_features[real_mask, idx])
        diff = real_mean - fake_mean
        
        # Indicate direction with arrow
        if abs(diff) > 0.01:
            direction = "↑" if diff > 0 else "↓"
        else:
            direction = "≈"
        
        print(f"    {name:25s}: {diff:+7.3f} {direction}")


def display_class_distribution(distribution) -> None:
    """Display class distribution analysis.
    
    Args:
        distribution: ClassDistribution object with counts, percentages, and imbalance_ratio.
        
    Requirements: 2.1, 2.2, 2.3
    """
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    label_names = {0: "Fake News", 1: "Real News"}
    
    print("\n  Class Counts:")
    for cls, count in distribution.counts.items():
        name = label_names.get(cls, f"Class {cls}")
        pct = distribution.percentages[cls] * 100
        print(f"    {name}: {count:,} samples ({pct:.1f}%)")
    
    print(f"\n  Imbalance Ratio: {distribution.imbalance_ratio:.2f}")
    
    if distribution.imbalance_ratio > 1.5:
        print("  ⚠️  Class imbalance detected - using balanced class weights and SMOTE")
    else:
        print("  ✓ Classes are relatively balanced")


def display_model_comparison(rankings: list[dict], best_model) -> None:
    """Display model comparison results.
    
    Args:
        rankings: List of dicts with 'name' and 'f1_score' keys, sorted by F1-score.
        best_model: The TrainedModel with highest F1-score.
        
    Requirements: 6.1, 6.2, 6.3
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    print("\n  Ranking by Validation F1-Score:")
    print("-" * 50)
    
    for rank, model_info in enumerate(rankings, 1):
        name = model_info["name"]
        f1 = model_info["f1_score"]
        marker = " 🏆" if rank == 1 else ""
        print(f"    {rank}. {name:20s}: F1-Score = {f1:.4f}{marker}")
    
    print("\n" + "-" * 50)
    print(f"  Best Model: {best_model.name}")
    print(f"  Best Parameters: {best_model.best_params}")
    print(f"  Validation F1-Score: {best_model.val_f1_score:.4f}")
    
    print("\n  Classification Report:")
    print(best_model.classification_report)


def display_evaluation_metrics(report) -> None:
    """Display evaluation metrics from the report.
    
    Args:
        report: EvaluationReport with formatted metrics.
        
    Requirements: 1.3, 5.1
    """
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(report.metrics_table)


def display_confusion_matrix_breakdown(report) -> None:
    """Display confusion matrix breakdown from the report.
    
    Args:
        report: EvaluationReport with confusion breakdown.
        
    Requirements: 2.1, 5.2
    """
    print("\n" + report.confusion_breakdown)


def display_error_analysis(report) -> None:
    """Display error analysis from the report.
    
    Args:
        report: EvaluationReport with error analysis text.
        
    Requirements: 3.1, 5.3
    """
    print("\n" + report.error_analysis_text)


def display_feature_importance(report) -> None:
    """Display feature importance from the report.
    
    Args:
        report: EvaluationReport with feature importance text.
        
    Requirements: 4.3, 5.4
    """
    print("\n" + report.feature_importance_text)


def run_feature_extraction(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extractor_path: str = "feature_extractor.joblib",
    force_refit: bool = False,
    use_farasa: bool = True,
    use_segmented_tfidf: bool = False,
) -> tuple[FeatureExtractor, np.ndarray, np.ndarray, np.ndarray]:
    """Run feature extraction workflow.
    
    Loads existing feature extractor if available, otherwise fits a new one.
    
    Args:
        train_df: Training DataFrame.
        validation_df: Validation DataFrame.
        test_df: Test DataFrame.
        extractor_path: Path to saved feature extractor.
        force_refit: If True, refit even if saved extractor exists.
        use_farasa: Enable Farasa Arabic NLP features (default True).
        use_segmented_tfidf: Use segmented text for TF-IDF (default False).
        
    Returns:
        Tuple of (extractor, train_features, validation_features, test_features).
        
    Requirements: 6.4
    """
    # Extract texts
    train_texts = get_texts(train_df)
    validation_texts = get_texts(validation_df)
    test_texts = get_texts(test_df)
    
    # Load or create feature extractor
    if os.path.exists(extractor_path) and not force_refit:
        print(f"  Loading existing feature extractor from {extractor_path}...")
        extractor = FeatureExtractor.load(extractor_path)
        # Display Farasa status from loaded extractor
        if extractor.use_farasa:
            if extractor._farasa_available:
                print("  ✓ Farasa features enabled (from saved extractor)")
            else:
                print("  ⚠️  Farasa was enabled but is not available")
        else:
            print("  Farasa features disabled (from saved extractor)")
    else:
        print("  Creating and fitting new FeatureExtractor...")
        print(f"  Farasa features: {'enabled' if use_farasa else 'disabled'}")
        if use_farasa:
            print(f"  Segmented TF-IDF: {'enabled' if use_segmented_tfidf else 'disabled'}")
        
        extractor = FeatureExtractor(
            max_tfidf_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            use_farasa=use_farasa,
            use_segmented_tfidf=use_segmented_tfidf,
        )
        extractor.fit(train_texts)
        
        # Display Farasa availability status
        if use_farasa:
            if extractor._farasa_available:
                print("  ✓ Farasa initialized successfully")
            else:
                print("  ⚠️  Farasa not available - continuing without Farasa features")
        
        extractor.save(extractor_path)
        print(f"  Feature extractor saved to {extractor_path}")
    
    # Transform all datasets
    print("  Transforming datasets...")
    train_features = extractor.transform(train_texts)
    validation_features = extractor.transform(validation_texts)
    test_features = extractor.transform(test_texts)
    
    return extractor, train_features, validation_features, test_features


def run_model_evaluation(
    model,
    model_name: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    feature_names: list[str],
    confusion_matrix_path: str = "confusion_matrix.png",
    results_path: str = "evaluation_results.joblib",
) -> None:
    """Run model evaluation workflow on test set.
    
    Evaluates the model, displays all metrics and analysis, saves confusion
    matrix plot and evaluation results to disk.
    
    Args:
        model: Trained sklearn model with predict() method.
        model_name: Name/identifier for the model.
        test_features: Test set feature matrix.
        test_labels: Test set labels.
        feature_names: List of feature names for importance extraction.
        confusion_matrix_path: Path to save confusion matrix plot.
        results_path: Path to save evaluation results.
        
    Requirements: 1.1, 1.2, 1.3, 2.2, 3.1, 4.3, 5.5, 6.1
    """
    print("\n" + "=" * 60)
    print("[PHASE 4] MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Step 1: Create evaluator and run evaluation (Req 1.1, 1.2, 1.3)
    print("\n[Step 1] Running model evaluation...")
    evaluator = ModelEvaluator(
        class_names=["Real News (0)", "Fake News (1)"],
        pos_label=1
    )
    
    eval_result = evaluator.evaluate(
        model=model,
        X_test=test_features,
        y_test=test_labels,
        model_name=model_name,
        feature_names=feature_names,
    )
    
    # Step 2: Generate report
    print("[Step 2] Generating evaluation report...")
    report = evaluator.generate_report(eval_result)
    
    # Step 3: Display metrics summary (Req 1.3, 5.1)
    print("\n[Step 3] Displaying evaluation results...")
    display_evaluation_metrics(report)
    
    # Step 4: Display confusion matrix breakdown (Req 2.1, 5.2)
    display_confusion_matrix_breakdown(report)
    
    # Step 5: Plot and save confusion matrix (Req 2.2)
    print(f"\n[Step 4] Saving confusion matrix plot to {confusion_matrix_path}...")
    evaluator.plot_confusion_matrix(
        eval_result.confusion_matrix,
        save_path=confusion_matrix_path
    )
    print(f"  Confusion matrix saved to: {confusion_matrix_path}")
    
    # Step 6: Display error analysis (Req 3.1, 5.3)
    display_error_analysis(report)
    
    # Step 7: Display feature importance (Req 4.3, 5.4)
    display_feature_importance(report)
    
    # Step 8: Save evaluation results to disk (Req 5.5, 6.1)
    print(f"\n[Step 5] Saving evaluation results to {results_path}...")
    evaluator.save_results(eval_result, results_path)
    print(f"  Evaluation results saved to: {results_path}")
    
    return eval_result


def run_model_training(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    model_path: str = "best_model.joblib",
) -> None:
    """Run model training workflow.
    
    Trains all models, compares them, selects the best, and saves it.
    
    Args:
        train_features: Training feature matrix.
        train_labels: Training labels.
        validation_features: Validation feature matrix.
        validation_labels: Validation labels.
        model_path: Path to save the best model.
        
    Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1
    """
    # Step 1: Analyze class distribution (Req 2.1, 2.2, 2.3)
    print("\n[Step 1] Analyzing class distribution...")
    distribution = analyze_class_distribution(train_labels)
    display_class_distribution(distribution)
    
    # Step 2: Create ModelTrainer and train all models
    print("\n[Step 2] Training models...")
    print("-" * 50)
    
    trainer = ModelTrainer(
        test_size=0.2,
        random_state=42,
        c_values=[0.1, 1.0, 10.0],
        n_estimators_values=[100, 200],
    )
    
    # Train all models (Req 3.1, 4.1, 5.1)
    trained_models = trainer.train_all_models(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    
    # Step 3: Compare models and select best (Req 6.1, 6.2, 6.3)
    print("\n[Step 3] Comparing models...")
    rankings = compare_models(trained_models)
    best_model = get_best_model(trained_models)
    
    display_model_comparison(rankings, best_model)
    
    # Step 4: Save best model (Req 7.1, 7.3)
    print("\n[Step 4] Saving best model...")
    save_model(best_model, model_path)
    print(f"  Best model saved to: {model_path}")
    
    return best_model, trained_models


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
        
    Requirements: 6.4
    """
    parser = argparse.ArgumentParser(
        description="Arabic Fake News Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with Farasa enabled (default)
  python main.py --no-farasa              # Run without Farasa features
  python main.py --farasa --segmented-tfidf  # Use segmented text for TF-IDF
  python main.py --force-refit            # Force refit feature extractor
        """
    )
    
    # Farasa options
    farasa_group = parser.add_argument_group("Farasa Options")
    farasa_group.add_argument(
        "--farasa",
        dest="use_farasa",
        action="store_true",
        default=True,
        help="Enable Farasa Arabic NLP features (default: enabled)"
    )
    farasa_group.add_argument(
        "--no-farasa",
        dest="use_farasa",
        action="store_false",
        help="Disable Farasa Arabic NLP features"
    )
    farasa_group.add_argument(
        "--segmented-tfidf",
        dest="use_segmented_tfidf",
        action="store_true",
        default=False,
        help="Use morphologically segmented text for TF-IDF (requires Farasa)"
    )
    
    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--force-refit",
        dest="force_refit",
        action="store_true",
        default=False,
        help="Force refit feature extractor even if saved version exists"
    )
    general_group.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default="dataset",
        help="Directory containing train/validation/test CSV files (default: dataset)"
    )
    
    return parser.parse_args()


def main():
    """Main workflow for Arabic fake news classification.
    
    Runs the complete pipeline:
    1. Load dataset
    2. Feature extraction (with optional Farasa integration)
    3. Model training
    4. Model comparison and selection
    5. Save best model
    
    Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1
    Farasa Integration Requirements: 6.4
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("ARABIC FAKE NEWS CLASSIFICATION")
    print("=" * 60)
    
    # Display configuration
    print("\n  Configuration:")
    print(f"    Farasa features: {'enabled' if args.use_farasa else 'disabled'}")
    if args.use_farasa:
        print(f"    Segmented TF-IDF: {'enabled' if args.use_segmented_tfidf else 'disabled'}")
    print(f"    Force refit: {'yes' if args.force_refit else 'no'}")
    print(f"    Data directory: {args.data_dir}")
    
    # =========================================================================
    # PHASE 1: Load Dataset
    # =========================================================================
    print("\n[PHASE 1] Loading dataset...")
    train_df, validation_df, test_df = load_dataset(args.data_dir)
    
    print(f"  Training samples:   {len(train_df):,}")
    print(f"  Validation samples: {len(validation_df):,}")
    print(f"  Test samples:       {len(test_df):,}")
    
    # Extract labels
    train_labels = train_df["validity"].values
    validation_labels = validation_df["validity"].values
    test_labels = test_df["validity"].values
    
    print(f"\n  Training label distribution:")
    print(f"    Fake (0): {(train_labels == 0).sum():,}")
    print(f"    Real (1): {(train_labels == 1).sum():,}")
    
    # =========================================================================
    # PHASE 2: Feature Extraction
    # =========================================================================
    print("\n[PHASE 2] Feature extraction...")
    extractor, train_features, validation_features, test_features = run_feature_extraction(
        train_df, 
        validation_df, 
        test_df,
        force_refit=args.force_refit,
        use_farasa=args.use_farasa,
        use_segmented_tfidf=args.use_segmented_tfidf,
    )
    
    print(f"\n  Feature matrix shapes:")
    print(f"    Training:   {train_features.shape}")
    print(f"    Validation: {validation_features.shape}")
    print(f"    Test:       {test_features.shape}")
    
    # Display feature analysis
    display_feature_analysis(extractor, train_features, train_labels)
    display_top_tfidf_features(extractor, n=10)
    
    # Display Farasa statistics if enabled
    if args.use_farasa:
        display_farasa_statistics(extractor, train_features, train_labels)
    
    # =========================================================================
    # PHASE 3: Model Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("[PHASE 3] MODEL TRAINING")
    print("=" * 60)
    
    best_model, trained_models = run_model_training(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        model_path="best_model.joblib",
    )
    
    # =========================================================================
    # PHASE 4: Model Evaluation on Test Set
    # =========================================================================
    # Get feature names for feature importance extraction
    feature_names = extractor.get_feature_names()
    
    # Run evaluation workflow (Req 1.1, 1.2, 1.3, 2.2, 3.1, 4.3, 5.5, 6.1)
    eval_result = run_model_evaluation(
        model=best_model.model,
        model_name=best_model.name,
        test_features=test_features,
        test_labels=test_labels,
        feature_names=feature_names,
        confusion_matrix_path="confusion_matrix.png",
        results_path="evaluation_results.joblib",
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total features: {train_features.shape[1]:,}")
    print(f"    - TF-IDF features: {len(extractor._tfidf_extractor.get_feature_names()):,}")
    print(f"    - Linguistic features: {len(extractor._linguistic_extractor.get_feature_names())}")
    print(f"    - Sentiment features: {len(extractor._sentiment_extractor.get_feature_names())}")
    
    # Display Farasa feature count if enabled
    if extractor.use_farasa:
        if extractor._farasa_available:
            print(f"    - Farasa features: {len(extractor._farasa_extractor.get_feature_names())} ✓")
        else:
            print(f"    - Farasa features: 0 (unavailable)")
    else:
        print(f"    - Farasa features: disabled")
    
    print(f"\n  Best Model: {best_model.name}")
    print(f"  Validation F1-Score: {best_model.val_f1_score:.4f}")
    print(f"  Test F1-Score: {eval_result.metrics.f1_score:.4f}")
    print(f"  Test Accuracy: {eval_result.metrics.accuracy:.4f}")
    print(f"  Best Parameters: {best_model.best_params}")
    print(f"\n  Artifacts saved:")
    print(f"    - Model: best_model.joblib")
    print(f"    - Feature extractor: feature_extractor.joblib")
    print(f"    - Confusion matrix: confusion_matrix.png")
    print(f"    - Evaluation results: evaluation_results.joblib")
    print("\nClassification pipeline complete!")


if __name__ == "__main__":
    main()
