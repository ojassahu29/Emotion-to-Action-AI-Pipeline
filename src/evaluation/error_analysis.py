"""
Error analysis: identify and describe failure cases.
"""
import pandas as pd


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
