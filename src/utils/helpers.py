"""
Reusable helper functions.
"""
import os
import pickle


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, filepath):
    """Save object to pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {filepath}")


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_csv(df, filepath):
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
