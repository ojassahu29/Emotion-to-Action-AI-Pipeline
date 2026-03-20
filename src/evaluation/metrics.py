"""
Evaluation metrics: accuracy and F1 score.
"""
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_emotion(y_true, y_pred):
    """Evaluate emotion classifier. Returns dict with accuracy and F1."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1, "report": report}


def evaluate_intensity(y_true, y_pred):
    """Evaluate intensity classifier. Returns dict with accuracy and F1."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1, "report": report}
