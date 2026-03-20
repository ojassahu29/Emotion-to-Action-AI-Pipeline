"""
Intensity classifier: GradientBoostingClassifier for predicting intensity (1-5).
Treated as classification to enable predict_proba for uncertainty estimation.
Hyperparameters tuned using validation split (CV F1=0.2172).
"""
from sklearn.ensemble import GradientBoostingClassifier
from src.utils.config import RANDOM_STATE


def create_intensity_model():
    """Create and return a tuned intensity classification model."""
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
