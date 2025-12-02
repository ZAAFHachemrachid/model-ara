"""Visualization utilities for feature extraction analysis.

This module provides functions for visualizing feature distributions
to compare fake vs real news articles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    figsize: tuple[int, int] = (15, 10),
    bins: int = 30,
) -> Figure:
    """Plot feature distributions comparing fake vs real news.
    
    Generates histograms for each feature, showing the distribution
    for fake (label=0) and real (label=1) news articles side by side.
    
    Args:
        features: Dense feature matrix of shape (n_samples, n_features).
        labels: Array of labels (0=fake, 1=real) of shape (n_samples,).
        feature_names: List of feature names corresponding to columns.
        figsize: Figure size as (width, height) tuple.
        bins: Number of histogram bins.
        
    Returns:
        Matplotlib Figure object containing the distribution plots.
        
    Raises:
        ValueError: If features and labels have mismatched lengths.
    """
    if features.shape[0] != len(labels):
        raise ValueError(
            f"Features and labels must have same length: "
            f"{features.shape[0]} vs {len(labels)}"
        )
    
    n_features = len(feature_names)
    labels_array = np.asarray(labels)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Separate features by label
    fake_mask = labels_array == 0
    real_mask = labels_array == 1
    
    for idx, (ax, name) in enumerate(zip(axes[:n_features], feature_names)):
        fake_values = features[fake_mask, idx]
        real_values = features[real_mask, idx]
        
        # Plot overlapping histograms
        ax.hist(fake_values, bins=bins, alpha=0.6, label='Fake', color='red', density=True)
        ax.hist(real_values, bins=bins, alpha=0.6, label='Real', color='blue', density=True)
        
        ax.set_title(name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Hide unused subplots
    for ax in axes[n_features:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    return fig



def plot_linguistic_features(
    linguistic_features: np.ndarray,
    labels: np.ndarray,
    figsize: tuple[int, int] = (15, 10),
    bins: int = 30,
) -> Figure:
    """Plot histograms for linguistic features comparing fake vs real.
    
    Generates distribution plots for all 7 linguistic features:
    text_length, word_count, uppercase_ratio, exclamation_count,
    question_count, punctuation_ratio, fake_keyword_count.
    
    Args:
        linguistic_features: Dense array of shape (n_samples, 7).
        labels: Array of labels (0=fake, 1=real).
        figsize: Figure size as (width, height) tuple.
        bins: Number of histogram bins.
        
    Returns:
        Matplotlib Figure object with linguistic feature distributions.
    """
    linguistic_names = [
        'text_length',
        'word_count', 
        'uppercase_ratio',
        'exclamation_count',
        'question_count',
        'punctuation_ratio',
        'fake_keyword_count'
    ]
    
    return plot_feature_distributions(
        features=linguistic_features,
        labels=labels,
        feature_names=linguistic_names,
        figsize=figsize,
        bins=bins,
    )


def plot_sentiment_features(
    sentiment_features: np.ndarray,
    labels: np.ndarray,
    figsize: tuple[int, int] = (10, 5),
    bins: int = 30,
) -> Figure:
    """Plot histograms for sentiment features comparing fake vs real.
    
    Generates distribution plots for polarity and subjectivity.
    
    Args:
        sentiment_features: Dense array of shape (n_samples, 2).
        labels: Array of labels (0=fake, 1=real).
        figsize: Figure size as (width, height) tuple.
        bins: Number of histogram bins.
        
    Returns:
        Matplotlib Figure object with sentiment feature distributions.
    """
    sentiment_names = ['polarity', 'subjectivity']
    
    return plot_feature_distributions(
        features=sentiment_features,
        labels=labels,
        feature_names=sentiment_names,
        figsize=figsize,
        bins=bins,
    )


def save_feature_plots(
    fig: Figure,
    path: str,
    dpi: int = 150,
) -> None:
    """Save a figure to disk.
    
    Args:
        fig: Matplotlib Figure to save.
        path: Output file path (e.g., 'plots/features.png').
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
