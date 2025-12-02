"""Main feature extractor that orchestrates all extraction components."""

import logging
import numpy as np
import scipy.sparse
from scipy.sparse import hstack, csr_matrix
import joblib

from src.feature_extraction.tfidf_extractor import TfidfExtractor
from src.feature_extraction.linguistic_extractor import LinguisticExtractor
from src.feature_extraction.sentiment_extractor import SentimentExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates all feature extractors and combines features.
    
    Combines TF-IDF, linguistic, sentiment, and optionally Farasa features 
    into a unified sparse feature matrix for machine learning classification.
    """

    def __init__(
        self,
        max_tfidf_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_farasa: bool = True,
        use_segmented_tfidf: bool = False,
    ):
        """Initialize the feature extraction pipeline.
        
        Args:
            max_tfidf_features: Maximum number of TF-IDF features (default 5000).
            ngram_range: Range of n-grams for TF-IDF (default unigrams and bigrams).
            min_df: Minimum document frequency for TF-IDF terms (default 2).
            max_df: Maximum document frequency ratio for TF-IDF terms (default 0.95).
            use_farasa: Enable Farasa Arabic NLP features (default True).
            use_segmented_tfidf: Use segmented text for TF-IDF when Farasa enabled (default False).
        """
        # Store configuration
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_farasa = use_farasa
        self.use_segmented_tfidf = use_segmented_tfidf
        
        # Initialize component extractors
        self._tfidf_extractor = TfidfExtractor(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self._linguistic_extractor = LinguisticExtractor()
        self._sentiment_extractor = SentimentExtractor()
        
        # Initialize Farasa extractor with graceful degradation
        self._farasa_extractor = None
        self._farasa_available = False
        if use_farasa:
            self._initialize_farasa()
        
        self._is_fitted = False
    
    def _initialize_farasa(self) -> None:
        """Initialize Farasa extractor with graceful degradation.
        
        If Farasa is unavailable (not installed or Java not available),
        logs a warning and continues without Farasa features.
        """
        try:
            from src.feature_extraction.farasa_extractor import FarasaExtractor
            self._farasa_extractor = FarasaExtractor()
            
            if self._farasa_extractor.is_available():
                self._farasa_available = True
                logger.info("Farasa extractor initialized successfully")
            else:
                logger.warning(
                    "Farasa extractor created but not available. "
                    "Farasa features will be disabled."
                )
                self._farasa_available = False
        except ImportError as e:
            logger.warning(f"Could not import FarasaExtractor: {e}. Farasa features disabled.")
            self._farasa_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Farasa: {e}. Farasa features disabled.")
            self._farasa_available = False


    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """Fit TF-IDF extractor on training texts.
        
        Only the TF-IDF extractor requires fitting. Linguistic, sentiment,
        and Farasa extractors are stateless.
        
        When use_segmented_tfidf is enabled and Farasa is available,
        the TF-IDF extractor is fitted on segmented text instead of raw text.
        
        Args:
            texts: List of text documents to fit on.
            
        Returns:
            Self for method chaining.
        """
        # Use segmented text for TF-IDF if configured and Farasa available
        if self.use_segmented_tfidf and self._farasa_available:
            segmented_texts = [
                self._farasa_extractor.get_segmented_text(text) for text in texts
            ]
            self._tfidf_extractor.fit(segmented_texts)
        else:
            self._tfidf_extractor.fit(texts)
        
        self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Transform texts to combined feature matrix.
        
        Extracts TF-IDF, linguistic, sentiment, and optionally Farasa features
        and combines them into a single sparse matrix using scipy.sparse.hstack
        for memory efficiency.
        
        Args:
            texts: List of text documents to transform.
            
        Returns:
            Sparse matrix with combined features.
            Column order: [TF-IDF features, linguistic features, sentiment features, farasa features]
            When Farasa is disabled: [TF-IDF features, linguistic features, sentiment features]
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Use segmented text for TF-IDF if configured and Farasa available
        if self.use_segmented_tfidf and self._farasa_available:
            tfidf_texts = [
                self._farasa_extractor.get_segmented_text(text) for text in texts
            ]
        else:
            tfidf_texts = texts
        
        # Extract TF-IDF features (already sparse)
        tfidf_features = self._tfidf_extractor.transform(tfidf_texts)
        
        # Extract linguistic features (dense) and convert to sparse
        linguistic_features = self._linguistic_extractor.transform(texts)
        linguistic_sparse = csr_matrix(linguistic_features)
        
        # Extract sentiment features (dense) and convert to sparse
        sentiment_features = self._sentiment_extractor.transform(texts)
        sentiment_sparse = csr_matrix(sentiment_features)
        
        # Build list of feature matrices to combine
        feature_matrices = [tfidf_features, linguistic_sparse, sentiment_sparse]
        
        # Add Farasa features if enabled and available
        if self.use_farasa and self._farasa_available:
            farasa_features = self._farasa_extractor.transform(texts)
            farasa_sparse = csr_matrix(farasa_features)
            feature_matrices.append(farasa_sparse)
        
        # Combine all features using sparse hstack for memory efficiency
        combined = hstack(feature_matrices)
        
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
            List of feature names: [TF-IDF names, linguistic names, sentiment names, farasa names]
            When Farasa is disabled: [TF-IDF names, linguistic names, sentiment names]
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")
        
        # Combine feature names in the same order as transform()
        tfidf_names = self._tfidf_extractor.get_feature_names()
        linguistic_names = self._linguistic_extractor.get_feature_names()
        sentiment_names = self._sentiment_extractor.get_feature_names()
        
        feature_names = tfidf_names + linguistic_names + sentiment_names
        
        # Add Farasa feature names if enabled and available
        if self.use_farasa and self._farasa_available:
            farasa_names = self._farasa_extractor.get_feature_names()
            feature_names = feature_names + farasa_names
        
        return feature_names

    def save(self, path: str) -> None:
        """Save fitted pipeline to disk using joblib.
        
        Serializes all fitted components and configuration to enable
        restoring the exact pipeline state later. Includes FarasaExtractor
        configuration for consistent feature extraction on load.
        
        Args:
            path: File path to save the pipeline to.
            
        Raises:
            ValueError: If the extractor has not been fitted.
            IOError: If the file cannot be written.
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
        
        # Capture FarasaExtractor configuration if available
        farasa_config = None
        if self._farasa_extractor is not None:
            farasa_config = {
                "enable_segmentation": self._farasa_extractor.enable_segmentation,
                "enable_stemming": self._farasa_extractor.enable_stemming,
                "enable_lemmatization": self._farasa_extractor.enable_lemmatization,
                "enable_pos": self._farasa_extractor.enable_pos,
                "enable_ner": self._farasa_extractor.enable_ner,
            }
        
        # Bundle all state needed to restore the pipeline
        state = {
            "config": {
                "max_tfidf_features": self.max_tfidf_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "use_farasa": self.use_farasa,
                "use_segmented_tfidf": self.use_segmented_tfidf,
            },
            "tfidf_extractor": self._tfidf_extractor,
            "is_fitted": self._is_fitted,
            "farasa_available": self._farasa_available,
            "farasa_config": farasa_config,
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
        
        Handles backward compatibility with pipelines saved without
        Farasa configuration. When loading a pipeline with Farasa enabled,
        recreates the FarasaExtractor with the same configuration settings.
        
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
        # Handle backward compatibility for pipelines without Farasa config
        config = state["config"]
        use_farasa = config.get("use_farasa", False)
        
        # Create instance without initializing Farasa (we'll do it manually)
        instance = cls.__new__(cls)
        
        # Restore configuration
        instance.max_tfidf_features = config["max_tfidf_features"]
        instance.ngram_range = config["ngram_range"]
        instance.min_df = config["min_df"]
        instance.max_df = config["max_df"]
        instance.use_farasa = use_farasa
        instance.use_segmented_tfidf = config.get("use_segmented_tfidf", False)
        
        # Restore fitted TF-IDF extractor
        instance._tfidf_extractor = state["tfidf_extractor"]
        instance._is_fitted = state["is_fitted"]
        
        # Recreate stateless extractors
        instance._linguistic_extractor = LinguisticExtractor()
        instance._sentiment_extractor = SentimentExtractor()
        
        # Restore Farasa extractor with saved configuration
        instance._farasa_extractor = None
        instance._farasa_available = False
        
        if use_farasa:
            farasa_config = state.get("farasa_config")
            instance._restore_farasa_extractor(farasa_config, state.get("farasa_available", False))
        
        return instance
    
    def _restore_farasa_extractor(
        self, 
        farasa_config: dict | None, 
        saved_farasa_available: bool
    ) -> None:
        """Restore FarasaExtractor from saved configuration.
        
        Attempts to recreate the FarasaExtractor with the same configuration
        as when the pipeline was saved. Handles graceful degradation if
        Farasa is no longer available.
        
        Args:
            farasa_config: Saved FarasaExtractor configuration dict, or None.
            saved_farasa_available: Whether Farasa was available when saved.
        """
        try:
            from src.feature_extraction.farasa_extractor import FarasaExtractor
            
            if farasa_config is not None:
                # Restore with exact saved configuration
                self._farasa_extractor = FarasaExtractor(
                    enable_segmentation=farasa_config.get("enable_segmentation", True),
                    enable_stemming=farasa_config.get("enable_stemming", True),
                    enable_lemmatization=farasa_config.get("enable_lemmatization", True),
                    enable_pos=farasa_config.get("enable_pos", True),
                    enable_ner=farasa_config.get("enable_ner", True),
                )
            else:
                # Backward compatibility: use default configuration
                self._farasa_extractor = FarasaExtractor()
            
            if self._farasa_extractor.is_available():
                self._farasa_available = True
                logger.info("Farasa extractor restored successfully")
            else:
                # Farasa was available when saved but not now
                if saved_farasa_available:
                    logger.warning(
                        "Farasa was available when pipeline was saved but is not "
                        "available now. Farasa features will be disabled."
                    )
                self._farasa_available = False
                
        except ImportError as e:
            logger.warning(f"Could not import FarasaExtractor: {e}. Farasa features disabled.")
            self._farasa_available = False
        except Exception as e:
            logger.warning(f"Failed to restore Farasa: {e}. Farasa features disabled.")
            self._farasa_available = False

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
