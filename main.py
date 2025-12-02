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

Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1
"""

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
)


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


def run_feature_extraction(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extractor_path: str = "feature_extractor.joblib",
    force_refit: bool = False,
) -> tuple[FeatureExtractor, np.ndarray, np.ndarray, np.ndarray]:
    """Run feature extraction workflow.
    
    Loads existing feature extractor if available, otherwise fits a new one.
    
    Args:
        train_df: Training DataFrame.
        validation_df: Validation DataFrame.
        test_df: Test DataFrame.
        extractor_path: Path to saved feature extractor.
        force_refit: If True, refit even if saved extractor exists.
        
    Returns:
        Tuple of (extractor, train_features, validation_features, test_features).
    """
    # Extract texts
    train_texts = get_texts(train_df)
    validation_texts = get_texts(validation_df)
    test_texts = get_texts(test_df)
    
    # Load or create feature extractor
    if os.path.exists(extractor_path) and not force_refit:
        print(f"  Loading existing feature extractor from {extractor_path}...")
        extractor = FeatureExtractor.load(extractor_path)
    else:
        print("  Creating and fitting new FeatureExtractor...")
        extractor = FeatureExtractor(
            max_tfidf_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        extractor.fit(train_texts)
        extractor.save(extractor_path)
        print(f"  Feature extractor saved to {extractor_path}")
    
    # Transform all datasets
    print("  Transforming datasets...")
    train_features = extractor.transform(train_texts)
    validation_features = extractor.transform(validation_texts)
    test_features = extractor.transform(test_texts)
    
    return extractor, train_features, validation_features, test_features


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


def main():
    """Main workflow for Arabic fake news classification.
    
    Runs the complete pipeline:
    1. Load dataset
    2. Feature extraction
    3. Model training
    4. Model comparison and selection
    5. Save best model
    
    Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1
    """
    print("=" * 60)
    print("ARABIC FAKE NEWS CLASSIFICATION")
    print("=" * 60)
    
    # =========================================================================
    # PHASE 1: Load Dataset
    # =========================================================================
    print("\n[PHASE 1] Loading dataset...")
    train_df, validation_df, test_df = load_dataset()
    
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
        train_df, validation_df, test_df
    )
    
    print(f"\n  Feature matrix shapes:")
    print(f"    Training:   {train_features.shape}")
    print(f"    Validation: {validation_features.shape}")
    print(f"    Test:       {test_features.shape}")
    
    # Display feature analysis
    display_feature_analysis(extractor, train_features, train_labels)
    display_top_tfidf_features(extractor, n=10)
    
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
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total features: {train_features.shape[1]:,}")
    print(f"    - TF-IDF features: {len(extractor._tfidf_extractor.get_feature_names()):,}")
    print(f"    - Linguistic features: {len(extractor._linguistic_extractor.get_feature_names())}")
    print(f"    - Sentiment features: {len(extractor._sentiment_extractor.get_feature_names())}")
    print(f"\n  Best Model: {best_model.name}")
    print(f"  Best F1-Score: {best_model.val_f1_score:.4f}")
    print(f"  Best Parameters: {best_model.best_params}")
    print(f"\n  Model saved to: best_model.joblib")
    print(f"  Feature extractor saved to: feature_extractor.joblib")
    print("\nClassification pipeline complete!")


if __name__ == "__main__":
    main()
