"""
Data loading: read Excel files and split features/targets.
"""
import pandas as pd
from src.utils.config import (
    TRAIN_FILE, TEST_FILE, TEXT_COL,
    TARGET_EMOTION, TARGET_INTENSITY, STRUCTURED_COLS
)


def load_train_data():
    """Load training dataset from Excel."""
    df = pd.read_excel(TRAIN_FILE)
    print(f"  Loaded training data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_test_data():
    """Load test dataset from Excel."""
    df = pd.read_excel(TEST_FILE)
    print(f"  Loaded test data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def split_features_targets(df):
    """
    Split DataFrame into text, structured features, and targets.
    Returns: text_series, structured_df, y_emotion, y_intensity
    """
    text = df[TEXT_COL].fillna("")
    structured = df[STRUCTURED_COLS].copy()
    y_emotion = df[TARGET_EMOTION]
    y_intensity = df[TARGET_INTENSITY]
    return text, structured, y_emotion, y_intensity


def get_features(df):
    """
    Extract text and structured features (no targets) for test data.
    Returns: text_series, structured_df
    """
    text = df[TEXT_COL].fillna("")
    structured = df[STRUCTURED_COLS].copy()
    return text, structured
