"""Sentiment feature extractor component."""

import numpy as np
from textblob import TextBlob


class SentimentExtractor:
    """Extracts sentiment features from text using TextBlob.
    
    Extracts polarity (-1 to 1) and subjectivity (0 to 1) scores,
    and classifies sentiment as positive, negative, or neutral.
    """
    
    # Thresholds for sentiment classification
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1
    
    def extract(self, text: str) -> dict[str, float | str]:
        """Extract sentiment features from a single text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dict with keys:
            - polarity: float (-1 to 1)
            - subjectivity: float (0 to 1)
            - sentiment: str ('positive', 'negative', 'neutral')
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment based on polarity thresholds
        if polarity > self.POSITIVE_THRESHOLD:
            sentiment = "positive"
        elif polarity < self.NEGATIVE_THRESHOLD:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment,
        }
    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to sentiment feature matrix.
        
        Args:
            texts: List of input texts.
            
        Returns:
            numpy array of shape (len(texts), 2) with columns [polarity, subjectivity].
        """
        features = []
        for text in texts:
            result = self.extract(text)
            features.append([result["polarity"], result["subjectivity"]])
        return np.array(features)
    
    def get_feature_names(self) -> list[str]:
        """Return list of sentiment feature names.
        
        Returns:
            List of feature column names.
        """
        return ["polarity", "subjectivity"]
