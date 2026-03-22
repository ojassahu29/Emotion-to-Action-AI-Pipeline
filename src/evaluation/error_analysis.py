"""
Error analysis: identify and describe failure cases.
Also provides label-noise / low-confidence detection.
"""
import os
import pandas as pd
from src.utils.helpers import ensure_dir


def analyze_errors(y_true, y_pred, original_df, model_name="Emotion", n=10):
    """
    Find misclassified samples and return analysis.

    Args:
        y_true: true labels (Series or array)
        y_pred: predicted labels (array)
        original_df: original DataFrame (for context columns)
        model_name: name for display
        n: number of failure cases to return

    Returns:
        DataFrame of failure cases with context.
    """
    y_true = y_true.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    errors = pd.DataFrame({
        "index": range(len(y_true)),
        "true_label": y_true,
        "predicted": y_pred,
    })

    # Keep only misclassifications
    errors = errors[errors["true_label"] != errors["predicted"]].copy()

    if errors.empty:
        print(f"  No errors found for {model_name} model!")
        return pd.DataFrame()

    # Add context columns
    context_cols = ["journal_text", "stress_level", "energy_level",
                    "sleep_hours", "ambience_type"]
    for col in context_cols:
        if col in original_df.columns:
            errors[col] = original_df.loc[errors["index"], col].values

    # Take top n errors
    errors = errors.head(n).reset_index(drop=True)

    print(f"\n  {model_name} Model — {len(errors)} failure cases (of {(y_true.values != y_pred).sum()} total):")
    for i, row in errors.iterrows():
        text_snippet = str(row.get("journal_text", ""))[:80]
        print(f"    [{i+1}] True: {row['true_label']} | Pred: {row['predicted']}")
        print(f"        Text: \"{text_snippet}...\"")
        print(f"        Stress={row.get('stress_level','?')} Energy={row.get('energy_level','?')} Sleep={row.get('sleep_hours','?')}")

    return errors


def detect_low_confidence_predictions(
    y_pred,
    confidence,
    threshold=0.6,
    log_path=None,
    model_name="Model",
):
    """
    Detect and log low-confidence (potentially noisy) predictions.

    Strategy:
      - Flag samples whose confidence score is below `threshold`
      - These predictions are unreliable and likely affected by label noise
        or ambiguous features
      - Flagged samples are logged to CSV for manual review / down-weighting

    Args:
        y_pred      (array-like): predicted labels
        confidence  (array-like): per-sample confidence scores
        threshold   (float)     : confidence cutoff (default 0.6)
        log_path    (str|None)  : CSV file path for saving flagged samples;
                                  if None, saves to outputs/low_confidence_log.csv
        model_name  (str)       : label for display

    Returns:
        pd.DataFrame: flagged low-confidence samples with index, prediction, and confidence
    """
    import numpy as np
    confidence = np.array(confidence)
    y_pred     = list(y_pred)

    low_conf_mask = confidence < threshold
    flagged_indices = list(np.where(low_conf_mask)[0])

    flagged_df = pd.DataFrame({
        "sample_index" : flagged_indices,
        "predicted"    : [y_pred[i] for i in flagged_indices],
        "confidence"   : confidence[flagged_indices],
        "model"        : model_name,
    })

    n_flagged = len(flagged_df)
    n_total   = len(confidence)
    print(
        f"  [{model_name}] Low-confidence predictions: {n_flagged}/{n_total} "
        f"(below threshold={threshold})"
    )

    # ── Log to CSV ────────────────────────────────────────────────────────
    if log_path is None:
        # Default: outputs/low_confidence_log.csv (relative to project root)
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))),
            "outputs", "low_confidence_log.csv"
        )

    ensure_dir(os.path.dirname(log_path))

    # Append if file exists so both models write to the same log
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, flagged_df], ignore_index=True)
        combined.to_csv(log_path, index=False)
    else:
        flagged_df.to_csv(log_path, index=False)

    print(f"    Logged to: {log_path}")
    return flagged_df
