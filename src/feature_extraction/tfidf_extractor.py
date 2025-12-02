"""TF-IDF feature extractor component."""

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


class TfidfExtractor:
    """Wraps scikit-learn's TfidfVectorizer for text vectorization.
    
    Extracts TF-IDF features from text data, supporting unigrams and bigrams
    with configurable feature limits and document frequency thresholds.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """Initialize TF-IDF extractor with configuration.
        
        Args:
            max_features: Maximum number of features to extract (default 5000).
            ngram_range: Range of n-grams to extract (default unigrams and bigrams).
            min_df: Minimum document frequency for terms (default 2).
            max_df: Maximum document frequency ratio for terms (default 0.95).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> "TfidfExtractor":
        """Fit the vectorizer on training texts.
        
        Args:
            texts: List of text documents to fit on.
            
        Returns:
            Self for method chaining.
        """
        self._vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Transform texts to TF-IDF feature matrix.
        
        Args:
            texts: List of text documents to transform.
            
        Returns:
            Sparse matrix of TF-IDF features.
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("TfidfExtractor must be fitted before transform")
        return self._vectorizer.transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names (vocabulary terms).
        
        Returns:
            List of vocabulary terms in feature order.
            
        Raises:
            ValueError: If the extractor has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("TfidfExtractor must be fitted before getting feature names")
        return list(self._vectorizer.get_feature_names_out())
