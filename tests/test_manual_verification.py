"""Manual verification tests for feature extraction components."""

from src.feature_extraction import TfidfExtractor, LinguisticExtractor, SentimentExtractor


def test_linguistic_extractor():
    """Test LinguisticExtractor with various inputs."""
    print('=== LinguisticExtractor Tests ===')
    ling = LinguisticExtractor()

    # Test with normal text
    text = 'Hello World! How are you?'
    features = ling.extract(text)
    print(f'Text: {repr(text)}')
    print(f'  text_length: {features["text_length"]} (expected: {len(text)})')
    print(f'  word_count: {features["word_count"]} (expected: {len(text.split())})')
    print(f'  exclamation_count: {features["exclamation_count"]} (expected: {text.count("!")})')
    print(f'  question_count: {features["question_count"]} (expected: {text.count("?")})')
    print(f'  uppercase_ratio: {features["uppercase_ratio"]:.4f} (should be in [0,1])')
    print(f'  punctuation_ratio: {features["punctuation_ratio"]:.4f} (should be in [0,1])')
    
    # Assertions
    assert features["text_length"] == len(text)
    assert features["word_count"] == len(text.split())
    assert features["exclamation_count"] == text.count("!")
    assert features["question_count"] == text.count("?")
    assert 0 <= features["uppercase_ratio"] <= 1
    assert 0 <= features["punctuation_ratio"] <= 1

    # Test with empty text
    empty_features = ling.extract('')
    print(f'\nEmpty text features:')
    print(f'  text_length: {empty_features["text_length"]} (expected: 0)')
    print(f'  uppercase_ratio: {empty_features["uppercase_ratio"]} (expected: 0.0)')
    print(f'  punctuation_ratio: {empty_features["punctuation_ratio"]} (expected: 0.0)')
    
    assert empty_features["text_length"] == 0
    assert empty_features["uppercase_ratio"] == 0.0
    assert empty_features["punctuation_ratio"] == 0.0
    
    print('LinguisticExtractor tests passed!')


def test_sentiment_extractor():
    """Test SentimentExtractor with various inputs."""
    print('\n=== SentimentExtractor Tests ===')
    sent = SentimentExtractor()

    # Test positive sentiment
    pos_text = 'I love this! It is amazing and wonderful!'
    pos_features = sent.extract(pos_text)
    print(f'Positive text: {repr(pos_text)}')
    print(f'  polarity: {pos_features["polarity"]:.4f} (should be > 0.1)')
    print(f'  subjectivity: {pos_features["subjectivity"]:.4f} (should be in [0,1])')
    print(f'  sentiment: {pos_features["sentiment"]} (expected: positive)')
    
    assert pos_features["polarity"] > 0.1
    assert 0 <= pos_features["subjectivity"] <= 1
    assert pos_features["sentiment"] == "positive"

    # Test negative sentiment
    neg_text = 'This is terrible and awful. I hate it.'
    neg_features = sent.extract(neg_text)
    print(f'\nNegative text: {repr(neg_text)}')
    print(f'  polarity: {neg_features["polarity"]:.4f} (should be < -0.1)')
    print(f'  subjectivity: {neg_features["subjectivity"]:.4f} (should be in [0,1])')
    print(f'  sentiment: {neg_features["sentiment"]} (expected: negative)')
    
    assert neg_features["polarity"] < -0.1
    assert 0 <= neg_features["subjectivity"] <= 1
    assert neg_features["sentiment"] == "negative"

    # Test neutral sentiment
    neutral_text = 'The meeting is at 3pm.'
    neutral_features = sent.extract(neutral_text)
    print(f'\nNeutral text: {repr(neutral_text)}')
    print(f'  polarity: {neutral_features["polarity"]:.4f} (should be in [-0.1, 0.1])')
    print(f'  subjectivity: {neutral_features["subjectivity"]:.4f} (should be in [0,1])')
    print(f'  sentiment: {neutral_features["sentiment"]} (expected: neutral)')
    
    assert -0.1 <= neutral_features["polarity"] <= 0.1
    assert 0 <= neutral_features["subjectivity"] <= 1
    assert neutral_features["sentiment"] == "neutral"
    
    print('SentimentExtractor tests passed!')


def test_tfidf_extractor():
    """Test TfidfExtractor with sample texts."""
    print('\n=== TfidfExtractor Tests ===')
    tfidf = TfidfExtractor(max_features=100)
    
    # Sample texts for fitting
    texts = [
        'This is a sample text about news',
        'Another text with different words',
        'News article about politics and government',
        'Sports news and entertainment updates',
        'Technology and science discoveries',
    ]
    
    # Fit and transform
    tfidf.fit(texts)
    features = tfidf.transform(texts)
    
    print(f'Fitted on {len(texts)} texts')
    print(f'Feature matrix shape: {features.shape}')
    print(f'Number of features: {len(tfidf.get_feature_names())}')
    
    # Assertions
    assert features.shape[0] == len(texts)
    assert features.shape[1] <= 100  # max_features
    assert len(tfidf.get_feature_names()) == features.shape[1]
    
    # Test vocabulary persistence
    new_texts = ['New text to transform']
    new_features = tfidf.transform(new_texts)
    assert new_features.shape[1] == features.shape[1]
    
    print('TfidfExtractor tests passed!')


if __name__ == '__main__':
    test_linguistic_extractor()
    test_sentiment_extractor()
    test_tfidf_extractor()
    print('\n=== All manual verification tests passed! ===')
