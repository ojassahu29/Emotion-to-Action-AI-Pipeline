"""
Ablation study: compare text-only vs text + structured features.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.models.emotion_model import create_emotion_model
from src.utils.config import RANDOM_STATE, TEST_SIZE


def run_ablation(X_text_only, X_combined, y):
    """
    Compare emotion model performance:
    - text-only features
    - text + structured features

    Returns: dict with both sets of metrics.
    """
    results = {}

    for name, X in [("text_only", X_text_only), ("text_+_structured", X_combined)]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = create_emotion_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        results[name] = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_macro": f1_score(y_val, y_pred, average="macro")
        }

    return results
