"""
Uncertainty estimation: confidence scores and uncertain flags.
"""
import numpy as np
from src.utils.config import CONFIDENCE_THRESHOLD


def compute_confidence(model, X):
    """
    Compute confidence as max predicted probability for each sample.
    Returns: numpy array of confidence scores.
    """
    probas = model.predict_proba(X)
    confidence = np.max(probas, axis=1)
    return confidence


def compute_uncertain_flag(confidence, threshold=CONFIDENCE_THRESHOLD):
    """
    Flag uncertain predictions.
    Returns: numpy array of 0/1 flags (1 = uncertain).
    """
    return (confidence < threshold).astype(int)
