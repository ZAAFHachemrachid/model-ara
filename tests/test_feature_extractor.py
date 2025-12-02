"""Tests for the main FeatureExtractor class."""

import os
import tempfile
import numpy as np
from src.feature_extraction import FeatureExtractor


def test_feature_extractor_fit_transform():
    """Test FeatureExtractor fit and transform methods."""
    print('=== FeatureExtractor Fit/Transform Tests ===')
    
    # Sample texts
    texts = [
        'This is a sample news article about politics',
        'Another article discussing sports and entertainment',
        'Breaking news about technology and science',
        'Political news from around the world',
        'Entertainment updates and celebrity news',
    ]
    
    # Create extractor with small max_features for testing
    extractor = FeatureExtractor(max_tfidf_features=50)
    
    # Fit and transform
    features = extractor.fit_transform(texts)
    
    print(f'Input texts: {len(texts)}')
    print(f'Feature matrix shape: {features.shape}')
    
    # Expected: tfidf_features + 7 linguistic + 2 sentiment
    expected_cols = features.shape[1]  # Will be <= 50 + 9
    assert features.shape[0] == len(texts)
    assert features.shape[1] <= 50 + 9  # max_tfidf + linguistic + sentiment
    
    # Test feature names
    feature_names = extractor.get_feature_names()
    assert len(feature_names) == features.shape[1]
    
    # Verify linguistic and sentiment feature names are present
    assert 'text_length' in feature_names
    assert 'word_count' in feature_names
    assert 'polarity' in feature_names
    assert 'subjectivity' in feature_names
    
    print('FeatureExtractor fit/transform tests passed!')


def test_feature_extractor_serialization():
    """Test FeatureExtractor save and load methods."""
    print('\n=== FeatureExtractor Serialization Tests ===')
    
    # Sample texts
    texts = [
        'Sample text for testing serialization',
        'Another text with different content',
        'Third text for variety in the corpus',
    ]
    
    # Create and fit extractor
    extractor = FeatureExtractor(max_tfidf_features=30)
    original_features = extractor.fit_transform(texts)
    original_names = extractor.get_feature_names()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name
    
    try:
        extractor.save(temp_path)
        print(f'Saved pipeline to: {temp_path}')
        
        # Load from file
        loaded_extractor = FeatureExtractor.load(temp_path)
        print('Loaded pipeline successfully')
        
        # Transform same texts with loaded extractor
        loaded_features = loaded_extractor.transform(texts)
        loaded_names = loaded_extractor.get_feature_names()
        
        # Verify identical results
        print(f'Original shape: {original_features.shape}')
        print(f'Loaded shape: {loaded_features.shape}')
        
        assert original_features.shape == loaded_features.shape
        assert original_names == loaded_names
        
        # Compare feature values (convert to dense for comparison)
        original_dense = original_features.toarray()
        loaded_dense = loaded_features.toarray()
        assert np.allclose(original_dense, loaded_dense)
        
        print('Serialization round-trip test passed!')
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_feature_extractor_analyze_features():
    """Test FeatureExtractor analyze_features method."""
    print('\n=== FeatureExtractor Analyze Features Tests ===')
    
    texts = [
        'Positive news about good things happening',
        'Negative news about bad events',
        'Neutral factual reporting',
        'More positive content here',
    ]
    labels = np.array([1, 0, 1, 1])  # 1=real, 0=fake
    
    extractor = FeatureExtractor(max_tfidf_features=20)
    features = extractor.fit_transform(texts)
    
    # Analyze features
    analysis = extractor.analyze_features(features, labels)
    
    print(f'Labels analyzed: {np.unique(labels)}')
    print(f'Analysis keys: {list(analysis.keys())}')
    
    # Verify structure
    assert 0 in analysis
    assert 1 in analysis
    
    for label in [0, 1]:
        assert 'mean' in analysis[label]
        assert 'std' in analysis[label]
        assert 'min' in analysis[label]
        assert 'max' in analysis[label]
        
        # Verify shapes
        assert len(analysis[label]['mean']) == features.shape[1]
    
    print('Analyze features test passed!')


def test_feature_extractor_top_tfidf_features():
    """Test FeatureExtractor get_top_tfidf_features method."""
    print('\n=== FeatureExtractor Top TF-IDF Features Tests ===')
    
    texts = [
        'News about politics and government',
        'Sports news and entertainment',
        'Technology updates and science',
        'Political news from the capital',
        'Entertainment industry updates',
    ]
    
    extractor = FeatureExtractor(max_tfidf_features=50)
    extractor.fit_transform(texts)
    
    # Get top features
    top_features = extractor.get_top_tfidf_features(n=10)
    
    print(f'Top 10 TF-IDF features:')
    for i, (name, weight) in enumerate(top_features, 1):
        print(f'  {i}. {name}: {weight:.4f}')
    
    # Verify ordering (descending by weight)
    weights = [w for _, w in top_features]
    assert weights == sorted(weights, reverse=True)
    
    # Verify count
    assert len(top_features) <= 10
    
    print('Top TF-IDF features test passed!')


if __name__ == '__main__':
    test_feature_extractor_fit_transform()
    test_feature_extractor_serialization()
    test_feature_extractor_analyze_features()
    test_feature_extractor_top_tfidf_features()
    print('\n=== All FeatureExtractor tests passed! ===')
