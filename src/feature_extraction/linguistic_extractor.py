"""Linguistic feature extractor component."""

import string
import numpy as np


class LinguisticExtractor:
    """Extracts writing style features from text.
    
    Features extracted:
    - text_length: Number of characters in text
    - word_count: Number of words in text
    - uppercase_ratio: Ratio of uppercase characters to total alphabetic characters
    - exclamation_count: Number of exclamation marks
    - question_count: Number of question marks
    - punctuation_ratio: Ratio of punctuation characters to total characters
    - fake_keyword_count: Count of sensationalism keywords
    """
    
    SENSATIONALISM_KEYWORDS: list[str] = [
        
        # Arabic keywords
        'صادم', 'لن تصدق', 'عاجل', 'خطير', 'فضيحة', 'مفاجأة'
    ]
    
    FEATURE_NAMES: list[str] = [
        'text_length',
        'word_count',
        'uppercase_ratio',
        'exclamation_count',
        'question_count',
        'punctuation_ratio',
        'fake_keyword_count'
    ]
    
    def extract(self, text: str) -> dict[str, float]:
        """Extract linguistic features from a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with feature names as keys and computed values
        """
        # Text length in characters
        text_length = len(text)
        
        # Word count
        word_count = len(text.split()) if text else 0
        
        # Uppercase ratio (ratio of uppercase to total alphabetic characters)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            uppercase_count = sum(1 for c in alpha_chars if c.isupper())
            uppercase_ratio = uppercase_count / len(alpha_chars)
        else:
            uppercase_ratio = 0.0

        # Exclamation and question mark counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Punctuation ratio
        if text:
            punctuation_count = sum(1 for c in text if c in string.punctuation)
            punctuation_ratio = punctuation_count / len(text)
        else:
            punctuation_ratio = 0.0
        
        # Fake keyword count (case-insensitive matching)
        text_lower = text.lower()
        fake_keyword_count = sum(
            1 for keyword in self.SENSATIONALISM_KEYWORDS
            if keyword.lower() in text_lower
        )
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'uppercase_ratio': uppercase_ratio,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'punctuation_ratio': punctuation_ratio,
            'fake_keyword_count': fake_keyword_count
        }
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to linguistic feature matrix.
        
        Args:
            texts: List of input text strings
            
        Returns:
            NumPy array of shape (len(texts), 7) with linguistic features
        """
        features = [self.extract(text) for text in texts]
        return np.array([
            [f[name] for name in self.FEATURE_NAMES]
            for f in features
        ])
    
    def get_feature_names(self) -> list[str]:
        """Return list of linguistic feature names.
        
        Returns:
            List of feature name strings in column order
        """
        return self.FEATURE_NAMES.copy()
