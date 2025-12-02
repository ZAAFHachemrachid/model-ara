"""Utility class for serializing and parsing Farasa segmented text.

This module provides the FarasaSegmentPrinter class for converting between
segment list representations and string formats, enabling round-trip
consistency for segmented Arabic text.
"""


class FarasaSegmentPrinter:
    """Utility class for serializing and parsing segmented Arabic text.
    
    This class provides methods to convert segment lists to printable string
    format and parse them back, ensuring round-trip consistency as required
    by Requirements 1.4.
    
    Attributes:
        SEGMENT_SEPARATOR: Character used to separate segments within a word.
        WORD_SEPARATOR: Character used to separate words.
    """
    
    SEGMENT_SEPARATOR: str = "+"
    WORD_SEPARATOR: str = " "
    
    @staticmethod
    def print_segments(segments: list[list[str]]) -> str:
        """Convert segment lists to printable string format.
        
        Takes a list of words where each word is represented as a list of
        its morphological segments, and converts it to a string representation
        where segments are joined by SEGMENT_SEPARATOR and words by WORD_SEPARATOR.
        
        Args:
            segments: A list of words, where each word is a list of segment strings.
                     Example: [["و", "ال", "كتاب"], ["جميل"]]
        
        Returns:
            A string representation of the segments.
            Example: "و+ال+كتاب جميل"
        
        Raises:
            ValueError: If segments is None or contains None values.
        """
        if segments is None:
            raise ValueError("segments cannot be None")
        
        word_strings = []
        for word_segments in segments:
            if word_segments is None:
                raise ValueError("Word segments cannot be None")
            # Join segments within a word with the segment separator
            word_str = FarasaSegmentPrinter.SEGMENT_SEPARATOR.join(word_segments)
            word_strings.append(word_str)
        
        # Join words with the word separator
        return FarasaSegmentPrinter.WORD_SEPARATOR.join(word_strings)
    
    @staticmethod
    def parse_segments(text: str) -> list[list[str]]:
        """Parse printed segments back to list format.
        
        Takes a string representation of segmented text and converts it back
        to a list of words where each word is a list of its segments.
        
        Args:
            text: A string representation of segmented text.
                  Example: "و+ال+كتاب جميل"
        
        Returns:
            A list of words, where each word is a list of segment strings.
            Example: [["و", "ال", "كتاب"], ["جميل"]]
        
        Raises:
            ValueError: If text is None.
        """
        if text is None:
            raise ValueError("text cannot be None")
        
        # Handle empty string case
        if not text:
            return []
        
        # Split by word separator first
        words = text.split(FarasaSegmentPrinter.WORD_SEPARATOR)
        
        result = []
        for word in words:
            # Handle empty words (from multiple consecutive spaces)
            if not word:
                result.append([])
            else:
                # Split each word by segment separator
                segments = word.split(FarasaSegmentPrinter.SEGMENT_SEPARATOR)
                result.append(segments)
        
        return result
