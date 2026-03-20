"""
Text cleaning: lowercase, remove special characters, strip whitespace.
"""
import re


def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_column(df, col="journal_text"):
    """Apply text cleaning to a DataFrame column. Returns modified df."""
    df = df.copy()
    df[col] = df[col].fillna("").apply(clean_text)
    return df
