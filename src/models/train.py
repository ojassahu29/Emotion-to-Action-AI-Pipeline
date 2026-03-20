"""
Training pipeline: train emotion and intensity models and save to disk.
"""
import os
from sklearn.model_selection import train_test_split

from src.models.emotion_model import create_emotion_model
from src.models.intensity_model import create_intensity_model
from src.utils.config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.utils.helpers import save_pickle, ensure_dir


def train_emotion_model(X, y):
    """
    Train the emotion classifier.
    Returns: trained model, X_val, y_val (for evaluation).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = create_emotion_model()
    model.fit(X_train, y_train)

    # Save model
    ensure_dir(MODELS_DIR)
    save_pickle(model, os.path.join(MODELS_DIR, "emotion_model.pkl"))

    return model, X_val, y_val


def train_intensity_model(X, y):
    """
    Train the intensity classifier.
    Returns: trained model, X_val, y_val (for evaluation).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = create_intensity_model()
    model.fit(X_train, y_train)

    # Save model
    ensure_dir(MODELS_DIR)
    save_pickle(model, os.path.join(MODELS_DIR, "intensity_model.pkl"))

    return model, X_val, y_val
