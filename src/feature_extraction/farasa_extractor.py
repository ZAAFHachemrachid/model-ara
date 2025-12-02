"""Farasa Arabic NLP extractor for feature extraction.

This module provides the FarasaExtractor class that wraps Farasa functionality
to extract Arabic-specific NLP features including segmentation, stemming,
lemmatization, POS tagging, and Named Entity Recognition.
"""

import logging
import numpy as np
from typing import Optional

from src.feature_extraction.farasa_segment_printer import FarasaSegmentPrinter

logger = logging.getLogger(__name__)


class FarasaExtractor:
    """Extracts Arabic NLP features using the Farasa toolkit.
    
    This class provides methods for Arabic text processing including:
    - Segmentation: Breaking words into morphological components
    - Stemming: Reducing words to their stem form
    - Lemmatization: Finding dictionary forms of words
    - POS Tagging: Extracting grammatical patterns
    - Named Entity Recognition: Identifying entities
    
    Attributes:
        enable_segmentation: Whether segmentation is enabled.
        enable_stemming: Whether stemming is enabled.
        enable_lemmatization: Whether lemmatization is enabled.
        enable_pos: Whether POS tagging is enabled.
        enable_ner: Whether NER is enabled.
    """
    
    # Feature names for the 9 Farasa features
    FEATURE_NAMES = [
        "pos_noun_ratio",
        "pos_verb_ratio",
        "pos_adj_ratio",
        "pos_other_ratio",
        "ner_person_count",
        "ner_location_count",
        "ner_organization_count",
        "ner_total_count",
        "avg_segments_per_word",
    ]
    
    def __init__(
        self,
        enable_segmentation: bool = True,
        enable_stemming: bool = True,
        enable_lemmatization: bool = True,
        enable_pos: bool = True,
        enable_ner: bool = True,
    ):
        """Initialize Farasa extractor with configuration.
        
        Args:
            enable_segmentation: Enable morphological segmentation.
            enable_stemming: Enable stem extraction.
            enable_lemmatization: Enable lemma extraction.
            enable_pos: Enable POS tagging.
            enable_ner: Enable Named Entity Recognition.
        """
        self.enable_segmentation = enable_segmentation
        self.enable_stemming = enable_stemming
        self.enable_lemmatization = enable_lemmatization
        self.enable_pos = enable_pos
        self.enable_ner = enable_ner
        
        self._segmenter: Optional[object] = None
        self._stemmer: Optional[object] = None
        self._lemmatizer: Optional[object] = None
        self._pos_tagger: Optional[object] = None
        self._ner_tagger: Optional[object] = None
        self._available: Optional[bool] = None
        
        self._initialize_farasa()

    
    def _initialize_farasa(self) -> None:
        """Initialize Farasa components based on configuration.
        
        Attempts to import and initialize Farasa components. If Farasa
        is not available, logs a warning and sets _available to False.
        """
        try:
            from farasa.segmenter import FarasaSegmenter
            from farasa.stemmer import FarasaStemmer
            from farasa.pos import FarasaPOSTagger
            from farasa.ner import FarasaNamedEntityRecognizer
            
            if self.enable_segmentation:
                self._segmenter = FarasaSegmenter(interactive=True)
            
            if self.enable_stemming:
                self._stemmer = FarasaStemmer(interactive=True)
            
            # Farasa doesn't have a separate lemmatizer - use stemmer for lemmatization
            if self.enable_lemmatization:
                # Reuse stemmer if already initialized, otherwise create new
                if self._stemmer is None:
                    self._stemmer = FarasaStemmer(interactive=True)
            
            if self.enable_pos:
                self._pos_tagger = FarasaPOSTagger(interactive=True)
            
            if self.enable_ner:
                self._ner_tagger = FarasaNamedEntityRecognizer(interactive=True)
            
            self._available = True
            logger.info("Farasa initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Farasa not available: {e}. Farasa features will be disabled.")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Farasa: {e}. Farasa features will be disabled.")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Farasa is properly installed and available.
        
        Returns:
            True if Farasa is available and initialized, False otherwise.
        """
        return self._available is True

    
    def _is_arabic(self, char: str) -> bool:
        """Check if a character is Arabic.
        
        Args:
            char: A single character to check.
            
        Returns:
            True if the character is Arabic, False otherwise.
        """
        # Arabic Unicode range: U+0600 to U+06FF (Arabic)
        # Extended Arabic: U+0750 to U+077F (Arabic Supplement)
        # Arabic Presentation Forms: U+FB50 to U+FDFF, U+FE70 to U+FEFF
        code = ord(char)
        return (
            (0x0600 <= code <= 0x06FF) or
            (0x0750 <= code <= 0x077F) or
            (0xFB50 <= code <= 0xFDFF) or
            (0xFE70 <= code <= 0xFEFF)
        )
    
    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains any Arabic characters.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text contains Arabic characters, False otherwise.
        """
        return any(self._is_arabic(char) for char in text)
    
    def segment(self, text: str) -> list[list[str]]:
        """Segment Arabic text into morphological components.
        
        Breaks Arabic words into prefixes, stems, and suffixes.
        Non-Arabic characters are passed through unchanged.
        
        Args:
            text: Arabic text to segment.
            
        Returns:
            A list of words, where each word is a list of its segments.
            Example: [["و", "ال", "كتاب"], ["جميل"]]
        """
        if not text or not text.strip():
            return []
        
        if not self.is_available() or self._segmenter is None:
            # Return each word as a single segment when Farasa unavailable
            return [[word] for word in text.split() if word]
        
        try:
            # Farasa returns segmented text with '+' between segments
            segmented = self._segmenter.segment(text)
            
            # Parse the segmented output
            result = []
            for word in segmented.split():
                if word:
                    # Split by '+' to get individual segments
                    segments = word.split('+')
                    result.append(segments)
            
            return result
            
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}. Returning unsegmented words.")
            return [[word] for word in text.split() if word]
    
    def get_segmented_text(self, text: str) -> str:
        """Return segmented text as a string for TF-IDF integration.
        
        Converts text to segmented form where morphological segments
        are separated by spaces, suitable for TF-IDF vectorization.
        
        Args:
            text: Arabic text to segment.
            
        Returns:
            Segmented text with segments separated by spaces.
            Example: "و ال كتاب جميل" for input "والكتاب جميل"
        """
        if not text or not text.strip():
            return ""
        
        segments = self.segment(text)
        
        # Flatten segments and join with spaces
        all_segments = []
        for word_segments in segments:
            all_segments.extend(word_segments)
        
        return " ".join(all_segments)

    
    def stem(self, text: str) -> list[str]:
        """Extract stems from Arabic text.
        
        Reduces Arabic words to their stem form, normalizing
        morphological variants to a common representation.
        
        Args:
            text: Arabic text to stem.
            
        Returns:
            A list of stems, one per word in the input.
            Returns empty list for empty text.
        """
        if not text or not text.strip():
            return []
        
        if not self.is_available() or self._stemmer is None:
            # Return original words when Farasa unavailable
            return text.split()
        
        try:
            # Farasa stemmer returns space-separated stems
            stemmed = self._stemmer.stem(text)
            return stemmed.split()
            
        except Exception as e:
            logger.warning(f"Stemming failed: {e}. Returning original words.")
            return text.split()

    
    def lemmatize(self, text: str) -> list[str]:
        """Extract lemmas from Arabic text.
        
        Finds the dictionary form (lemma) for each word in the text.
        Returns the original word when no lemma is found.
        
        Note: Farasa doesn't have a dedicated lemmatizer, so this uses
        the stemmer as an approximation. For true lemmatization, consider
        using additional Arabic NLP tools.
        
        Args:
            text: Arabic text to lemmatize.
            
        Returns:
            A list of lemmas, one per word in the input.
            Returns empty list for empty text.
        """
        if not text or not text.strip():
            return []
        
        if not self.is_available() or self._stemmer is None:
            # Return original words when Farasa unavailable
            return text.split()
        
        try:
            # Use stemmer as approximation for lemmatization
            # Farasa's stemmer provides a reasonable approximation
            lemmatized = self._stemmer.stem(text)
            
            original_words = text.split()
            lemmas = lemmatized.split()
            
            # Ensure we return original word if lemma is empty
            result = []
            for i, lemma in enumerate(lemmas):
                if lemma and lemma.strip():
                    result.append(lemma)
                elif i < len(original_words):
                    result.append(original_words[i])
                else:
                    result.append(lemma)
            
            return result
            
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}. Returning original words.")
            return text.split()

    
    def pos_tag(self, text: str) -> list[tuple[str, str]]:
        """Extract POS tags for each token.
        
        Assigns a Part-of-Speech tag to each token in the text.
        
        Args:
            text: Arabic text to tag.
            
        Returns:
            A list of (token, tag) tuples.
        """
        if not text or not text.strip():
            return []
        
        if not self.is_available() or self._pos_tagger is None:
            # Return tokens with 'UNK' tag when Farasa unavailable
            return [(word, "UNK") for word in text.split() if word]
        
        try:
            # Farasa POS tagger returns tagged text
            tagged = self._pos_tagger.tag(text)
            
            # Parse the output - format is "word/TAG word/TAG ..."
            result = []
            for item in tagged.split():
                if '/' in item:
                    parts = item.rsplit('/', 1)
                    if len(parts) == 2:
                        result.append((parts[0], parts[1]))
                    else:
                        result.append((item, "UNK"))
                else:
                    result.append((item, "UNK"))
            
            return result
            
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}. Returning unknown tags.")
            return [(word, "UNK") for word in text.split() if word]
    
    def ner(self, text: str) -> list[tuple[str, str]]:
        """Extract named entities with their categories.
        
        Identifies named entities and classifies them into categories:
        PERSON, LOCATION, ORGANIZATION.
        
        Args:
            text: Arabic text to analyze.
            
        Returns:
            A list of (entity, category) tuples.
        """
        if not text or not text.strip():
            return []
        
        if not self.is_available() or self._ner_tagger is None:
            return []
        
        try:
            # Farasa NER returns tagged entities
            ner_result = self._ner_tagger.recognize(text)
            
            # Parse the output - format varies, typically "word/TAG"
            result = []
            for item in ner_result.split():
                if '/' in item:
                    parts = item.rsplit('/', 1)
                    if len(parts) == 2:
                        entity, tag = parts
                        # Only include actual named entities (not 'O' tags)
                        if tag and tag != 'O':
                            # Normalize tag names
                            normalized_tag = self._normalize_ner_tag(tag)
                            if normalized_tag:
                                result.append((entity, normalized_tag))
            
            return result
            
        except Exception as e:
            logger.warning(f"NER failed: {e}. Returning empty list.")
            return []
    
    def _normalize_ner_tag(self, tag: str) -> str:
        """Normalize NER tag to standard categories.
        
        Args:
            tag: Raw NER tag from Farasa.
            
        Returns:
            Normalized tag (PERSON, LOCATION, ORGANIZATION) or empty string.
        """
        tag_upper = tag.upper()
        
        # Map various tag formats to standard categories
        if 'PER' in tag_upper or 'PERSON' in tag_upper:
            return 'PERSON'
        elif 'LOC' in tag_upper or 'LOCATION' in tag_upper or 'GPE' in tag_upper:
            return 'LOCATION'
        elif 'ORG' in tag_upper or 'ORGANIZATION' in tag_upper:
            return 'ORGANIZATION'
        
        return ''

    
    def extract_features(self, text: str) -> dict[str, float]:
        """Extract all Farasa-based features from text.
        
        Combines POS ratios, NER counts, and segment statistics
        into a feature dictionary.
        
        Args:
            text: Arabic text to extract features from.
            
        Returns:
            Dictionary with keys:
            - pos_noun_ratio: float (0-1)
            - pos_verb_ratio: float (0-1)
            - pos_adj_ratio: float (0-1)
            - pos_other_ratio: float (0-1)
            - ner_person_count: int
            - ner_location_count: int
            - ner_organization_count: int
            - ner_total_count: int
            - avg_segments_per_word: float
        """
        # Initialize default features
        features = {
            "pos_noun_ratio": 0.0,
            "pos_verb_ratio": 0.0,
            "pos_adj_ratio": 0.0,
            "pos_other_ratio": 0.0,
            "ner_person_count": 0,
            "ner_location_count": 0,
            "ner_organization_count": 0,
            "ner_total_count": 0,
            "avg_segments_per_word": 0.0,
        }
        
        if not text or not text.strip():
            return features
        
        # Extract POS features
        if self.enable_pos:
            pos_tags = self.pos_tag(text)
            features.update(self._compute_pos_features(pos_tags))
        
        # Extract NER features
        if self.enable_ner:
            entities = self.ner(text)
            features.update(self._compute_ner_features(entities))
        
        # Extract segmentation features
        if self.enable_segmentation:
            segments = self.segment(text)
            features.update(self._compute_segment_features(segments))
        
        return features
    
    def _compute_pos_features(self, pos_tags: list[tuple[str, str]]) -> dict[str, float]:
        """Compute POS distribution features.
        
        Args:
            pos_tags: List of (token, tag) tuples.
            
        Returns:
            Dictionary with POS ratio features.
        """
        if not pos_tags:
            return {
                "pos_noun_ratio": 0.0,
                "pos_verb_ratio": 0.0,
                "pos_adj_ratio": 0.0,
                "pos_other_ratio": 0.0,
            }
        
        noun_count = 0
        verb_count = 0
        adj_count = 0
        other_count = 0
        
        for _, tag in pos_tags:
            tag_upper = tag.upper()
            if 'NOUN' in tag_upper or tag_upper.startswith('N'):
                noun_count += 1
            elif 'VERB' in tag_upper or tag_upper.startswith('V'):
                verb_count += 1
            elif 'ADJ' in tag_upper or tag_upper.startswith('A'):
                adj_count += 1
            else:
                other_count += 1
        
        total = len(pos_tags)
        
        return {
            "pos_noun_ratio": noun_count / total,
            "pos_verb_ratio": verb_count / total,
            "pos_adj_ratio": adj_count / total,
            "pos_other_ratio": other_count / total,
        }
    
    def _compute_ner_features(self, entities: list[tuple[str, str]]) -> dict[str, int]:
        """Compute NER count features.
        
        Args:
            entities: List of (entity, category) tuples.
            
        Returns:
            Dictionary with NER count features.
        """
        person_count = 0
        location_count = 0
        organization_count = 0
        
        for _, category in entities:
            if category == 'PERSON':
                person_count += 1
            elif category == 'LOCATION':
                location_count += 1
            elif category == 'ORGANIZATION':
                organization_count += 1
        
        total = person_count + location_count + organization_count
        
        return {
            "ner_person_count": person_count,
            "ner_location_count": location_count,
            "ner_organization_count": organization_count,
            "ner_total_count": total,
        }
    
    def _compute_segment_features(self, segments: list[list[str]]) -> dict[str, float]:
        """Compute segmentation statistics.
        
        Args:
            segments: List of words, each as a list of segments.
            
        Returns:
            Dictionary with segment statistics.
        """
        if not segments:
            return {"avg_segments_per_word": 0.0}
        
        total_segments = sum(len(word_segments) for word_segments in segments)
        avg_segments = total_segments / len(segments)
        
        return {"avg_segments_per_word": avg_segments}

    
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform multiple texts to Farasa feature matrix.
        
        Args:
            texts: List of Arabic texts to transform.
            
        Returns:
            NumPy array of shape (len(texts), 9) with Farasa features.
        """
        if not texts:
            return np.array([]).reshape(0, len(self.FEATURE_NAMES))
        
        features_list = []
        for text in texts:
            features = self.extract_features(text)
            # Convert to list in consistent order
            feature_values = [features[name] for name in self.FEATURE_NAMES]
            features_list.append(feature_values)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> list[str]:
        """Return list of Farasa feature names.
        
        Returns:
            List of 9 feature names in consistent order.
        """
        return self.FEATURE_NAMES.copy()
