"""
Emotion classifier: LinearSVC for strong text classification accuracy,
wrapped in CalibratedClassifierCV to provide `predict_proba` for uncertainty estimation.
"""
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.utils.config import RANDOM_STATE


def create_emotion_model():
    """Create and return an optimized, calibrated emotion classification model."""
    base_svc = LinearSVC(
        C=1.0, 
        class_weight="balanced", 
        dual=False,
        random_state=RANDOM_STATE
    )
    
    # Calibrate to extract probabilities (required for uncertainty.py)
    return CalibratedClassifierCV(base_svc, method='sigmoid', cv=5)
