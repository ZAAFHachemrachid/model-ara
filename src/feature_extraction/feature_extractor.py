"""Main feature extractor that orchestrates all extraction components."""

import numpy as np
import scipy.sparse
from scipy.sparse import hstack, csr_matrix
import joblib

from src.feature_extraction.tfidf_extractor import TfidfExtractor
from src.feature_extraction.linguistic_extractor import LinguisticExtractor
from src.feature_extraction.sentiment_extractor import SentimentExtractor


class FeatureExtractor:
    """Orchestrates all feature extractors and combines features.
    
    Combines TF-IDF, linguistic, and sentiment features into a unified
    sparse feature matrix for machine learning classification.
    """

    def __init__(
        self,
        max_tfidf_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """Initialize the feature extraction pipeline.
        
        Args:
            max_tfidf_features: Maximum number of TF-IDF features (default 5000).
            ngram_range: Range of n-grams for TF-IDF (default unigrams and bigrams).
            min_df: Minimum document frequency for TF-IDF terms (default 2).
            max_df: Maximum document frequency ratio for TF-IDF terms (default 0.95).
        """
        # Store configuration
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize component extractors
        self._tfidf_extractor = TfidfExtractor(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self._linguistic_extractor = LinguisticExtractor()
        self._sentiment_extractor = SentimentExtractor()
        
        self._is_fitted = False


    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """Fit TF-IDF extractor on training texts.
        
        Only the TF-IDF extractor requires fitting. Linguistic and sentiment
        extractors are stateless.
        
        Args:
            texts: List of text documents to fit on.
            
        Returns:
            Self for method chaining.
        """
        self._tfidf_extractor.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Transform texts to combined feature matrix.
        
        Extracts TF-IDF, linguistic, and sentiment features and combines
        them into a single sparse matrix using scipy.sparse.hstack for
        memory efficiency.
        
        Args:
            texts: List of text documents to transform.
            
        Returns:
            Sparse matrix with combined features.
            Column order: [TF-IDF features, linguistic features, sentiment features]
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Extract TF-IDF features (already sparse)
        tfidf_features = self._tfidf_extractor.transform(texts)
        
        # Extract linguistic features (dense) and convert to sparse
        linguistic_features = self._linguistic_extractor.transform(texts)
        linguistic_sparse = csr_matrix(linguistic_features)
        
        # Extract sentiment features (dense) and convert to sparse
        sentiment_features = self._sentiment_extractor.transform(texts)
        sentiment_sparse = csr_matrix(sentiment_features)
        
        # Combine all features using sparse hstack for memory efficiency
        combined = hstack([tfidf_features, linguistic_sparse, sentiment_sparse])
        
        return combined.tocsr()

    def fit_transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Fit and transform in one step.
        
        Convenience method that fits the extractor on the provided texts
        and then transforms them.
        
        Args:
            texts: List of text documents to fit and transform.
            
        Returns:
            Sparse matrix with combined features.
        """
        self.fit(texts)
        return self.transform(texts)


    def get_feature_names(self) -> list[str]:
        """Return all feature names in column order.
        
        Combines feature names from all extractors in the same order
        as the combined feature matrix columns.
        
        Returns:
            List of feature names: [TF-IDF names, linguistic names, sentiment names]
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")
        
        # Combine feature names in the same order as transform()
        tfidf_names = self._tfidf_extractor.get_feature_names()
        linguistic_names = self._linguistic_extractor.get_feature_names()
        sentiment_names = self._sentiment_extractor.get_feature_names()
        
        return tfidf_names + linguistic_names + sentiment_names

    def save(self, path: str) -> None:
        """Save fitted pipeline to disk using joblib.
        
        Serializes all fitted components and configuration to enable
        restoring the exact pipeline state later.
        
        Args:
            path: File path to save the pipeline to.
            
        Raises:
            ValueError: If the extractor has not been fitted.
            IOError: If the file cannot be written.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
        
        # Bundle all state needed to restore the pipeline
        state = {
            "config": {
                "max_tfidf_features": self.max_tfidf_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
            },
            "tfidf_extractor": self._tfidf_extractor,
            "is_fitted": self._is_fitted,
        }
        
        try:
            joblib.dump(state, path)
        except Exception as e:
            raise IOError(f"Failed to save pipeline to {path}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load fitted pipeline from disk.
        
        Restores a previously saved pipeline to its exact state,
        enabling consistent transformations on new data.
        
        Args:
            path: File path to load the pipeline from.
            
        Returns:
            Restored FeatureExtractor instance.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted or invalid.
        """
        try:
            state = joblib.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load pipeline from {path}: {e}") from e
        
        # Validate loaded state
        required_keys = {"config", "tfidf_extractor", "is_fitted"}
        if not required_keys.issubset(state.keys()):
            raise ValueError(f"Corrupted pipeline file: missing required keys")
        
        # Create new instance with saved configuration
        config = state["config"]
        instance = cls(
            max_tfidf_features=config["max_tfidf_features"],
            ngram_range=config["ngram_range"],
            min_df=config["min_df"],
            max_df=config["max_df"],
        )
        
        # Restore fitted state
        instance._tfidf_extractor = state["tfidf_extractor"]
        instance._is_fitted = state["is_fitted"]
        
        return instance

    def analyze_features(
        self,
        features: scipy.sparse.csr_matrix,
        labels: np.ndarray,
    ) -> dict:
        """Compute feature statistics grouped by label.
        
        Analyzes the feature matrix and computes summary statistics
        (mean, std, min, max) for each feature, grouped by label.
        
        Args:
            features: Sparse feature matrix from transform().
            labels: Array of labels corresponding to each row in features.
            
        Returns:
            Dictionary with structure:
            {
                label_value: {
                    'mean': array of mean values per feature,
                    'std': array of std values per feature,
                    'min': array of min values per feature,
                    'max': array of max values per feature,
                }
            }
            
        Raises:
            ValueError: If features and labels have mismatched lengths.
        """
        if features.shape[0] != len(labels):
            raise ValueError(
                f"Features and labels must have same length: "
                f"{features.shape[0]} vs {len(labels)}"
            )
        
        # Convert to dense for statistics computation
        # Note: For very large matrices, consider computing stats on sparse directly
        dense_features = features.toarray()
        labels_array = np.asarray(labels)
        
        unique_labels = np.unique(labels_array)
        result = {}
        
        for label in unique_labels:
            mask = labels_array == label
            label_features = dense_features[mask]
            
            result[label] = {
                'mean': np.mean(label_features, axis=0),
                'std': np.std(label_features, axis=0),
                'min': np.min(label_features, axis=0),
                'max': np.max(label_features, axis=0),
            }
        
        return result

    def get_top_tfidf_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Return top N TF-IDF features by total weight.
        
        Computes the sum of IDF weights for each term in the vocabulary
        and returns the top N features sorted by weight in descending order.
        
        Args:
            n: Number of top features to return (default 20).
            
        Returns:
            List of (feature_name, weight) tuples sorted by weight descending.
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting top features")
        
        # Access the underlying TfidfVectorizer's IDF weights
        vectorizer = self._tfidf_extractor._vectorizer
        idf_weights = vectorizer.idf_
        feature_names = self._tfidf_extractor.get_feature_names()
        
        # Create list of (name, weight) tuples
        feature_weights = list(zip(feature_names, idf_weights))
        
        # Sort by weight descending and take top N
        sorted_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)
        
        return sorted_features[:n]
