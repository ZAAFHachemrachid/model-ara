"""Feature extraction module for Arabic fake news classification.

This module provides components for extracting TF-IDF, linguistic, and sentiment
features from Arabic text for machine learning classification.
"""

from src.feature_extraction.tfidf_extractor import TfidfExtractor
from src.feature_extraction.linguistic_extractor import LinguisticExtractor
from src.feature_extraction.sentiment_extractor import SentimentExtractor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.feature_extraction.visualization import (
    plot_feature_distributions,
    plot_linguistic_features,
    plot_sentiment_features,
    save_feature_plots,
)

__all__ = [
    "TfidfExtractor",
    "LinguisticExtractor",
    "SentimentExtractor",
    "FeatureExtractor",
    "plot_feature_distributions",
    "plot_linguistic_features",
    "plot_sentiment_features",
    "save_feature_plots",
]
