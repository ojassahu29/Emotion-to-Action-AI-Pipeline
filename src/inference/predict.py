"""
Prediction: load saved models and generate predictions.
"""
import os
from src.utils.config import MODELS_DIR
from src.utils.helpers import load_pickle


def load_models():
    """Load trained emotion and intensity models from disk."""
    emotion_model = load_pickle(os.path.join(MODELS_DIR, "emotion_model.pkl"))
    intensity_model = load_pickle(os.path.join(MODELS_DIR, "intensity_model.pkl"))
    return emotion_model, intensity_model


def predict_emotion(model, X):
    """Predict emotional_state labels."""
    return model.predict(X)


def predict_intensity(model, X):
    """Predict intensity labels."""
    return model.predict(X)
