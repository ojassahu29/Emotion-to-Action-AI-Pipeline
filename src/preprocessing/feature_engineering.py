"""
Feature engineering: Advanced TF-IDF (word + char n-grams), text statistics,
encoding for structured features, and interaction terms.
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import (
    STRUCTURED_NUMERIC, STRUCTURED_CATEGORICAL
)


def build_word_tfidf_vectorizer(max_features=2000):
    """Create a Word TF-IDF vectorizer (unigrams and bigrams)."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        strip_accents='unicode'
    )


def build_char_tfidf_vectorizer(max_features=3000):
    """Create a Character TF-IDF vectorizer to capture subword patterns."""
    return TfidfVectorizer(
        max_features=max_features,
        analyzer='char_wb',
        ngram_range=(3, 5),
        strip_accents='unicode'
    )


def extract_text_stats(texts):
    """Extract basic statistical features from text."""
    stats = pd.DataFrame(index=texts.index)
    stats['text_len'] = texts.str.len().fillna(0)
    stats['word_count'] = texts.str.split().str.len().fillna(0)
    
    # Normalize
    for col in stats.columns:
        mean = stats[col].mean()
        std = stats[col].std() if stats[col].std() > 0 else 1.0
        stats[col] = (stats[col] - mean) / std
        
    return stats


def encode_structured(df):
    """
    Encode structured features and create interaction terms.
    Returns a DataFrame with all encoded features.
    """
    result = pd.DataFrame(index=df.index)

    # Numeric features — fill NaN
    for col in STRUCTURED_NUMERIC:
        if col in df.columns:
            result[col] = df[col].fillna(df[col].median())

    # Interaction terms
    if 'stress_level' in result.columns and 'energy_level' in result.columns:
        result['stress_x_energy'] = result['stress_level'] * result['energy_level']
        # High stress, low energy indicator
        result['burnout_index'] = result['stress_level'] / (result['energy_level'] + 1.0)

    # Normalize numeric features
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = result[col].mean()
        std = result[col].std() if result[col].std() > 0 else 1.0
        result[col] = (result[col] - mean) / std

    # Categorical features — one-hot encode
    for col in STRUCTURED_CATEGORICAL:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna("unknown"), prefix=col)
            result = pd.concat([result, dummies.astype(float)], axis=1)

    return result


def combine_features(word_matrix, char_matrix, text_stats_df, structured_df):
    """
    Combine all sparse and dense features into a single sparse matrix.
    """
    stats_sparse = csr_matrix(text_stats_df.values.astype(np.float64))
    struct_sparse = csr_matrix(structured_df.values.astype(np.float64))
    
    combined = hstack([word_matrix, char_matrix, stats_sparse, struct_sparse])
    return combined
