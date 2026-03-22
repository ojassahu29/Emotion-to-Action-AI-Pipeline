# Error Analysis

## Overview

This document presents failure cases from the emotion and intensity classifiers, analyzed on the validation set (20% hold-out, stratified).

Error analysis is generated automatically by `src/evaluation/error_analysis.py` during the pipeline run. Below are representative failure patterns.

---

## Emotion Model Failure Cases

### Case 1: "mixed" predicted as "calm"
- **Context**: Journal text was reflective and low-stress → model confused "mixed" with "calm"
- **Root cause**: Mixed emotions have overlapping text signals with calm states

### Case 2: "focused" predicted as "neutral"
- **Context**: Short journal entry, moderate energy/stress
- **Root cause**: Lack of strong focus-related keywords in short text

### Case 3: "restless" predicted as "overwhelmed"
- **Context**: High stress, low sleep hours, agitated language
- **Root cause**: Restless and overwhelmed share similar physiological markers

### Case 4: "overwhelmed" predicted as "restless"
- **Context**: High intensity, moderate stress
- **Root cause**: Inverse of Case 3 — similar feature profiles

### Case 5: "neutral" predicted as "calm"
- **Context**: Neutral language, average features across the board
- **Root cause**: Neutral and calm are semantically close, especially with non-descriptive text

### Case 6: "calm" predicted as "neutral"
- **Context**: Low emotional language, standard structured features
- **Root cause**: Mirror of Case 5

### Case 7: "mixed" predicted as "restless"
- **Context**: Conflicting emotions described in journal, elevated stress
- **Root cause**: Mixed with high stress signals mimics restless patterns

---

## Intensity Model Failure Cases

### Case 8: True intensity 3, predicted 2
- **Context**: Moderate journal language, median structured features
- **Root cause**: Mid-range intensity values are inherently ambiguous

### Case 9: True intensity 5, predicted 4
- **Context**: Strong emotional language but moderate stress/energy
- **Root cause**: Structured features partially contradicted text signals

### Case 10: True intensity 1, predicted 2
- **Context**: Very short journal entry, low emotional content
- **Root cause**: Limited text features make low-intensity detection difficult

---

## Common Failure Patterns

| Pattern | Frequency | Description |
|---------|-----------|-------------|
| Semantic overlap | High | calm ↔ neutral, restless ↔ overwhelmed confusions |
| Short text | Medium | Insufficient text for reliable TF-IDF features |
| Mid-range intensity | Medium | Intensity 2–4 hard to distinguish reliably |
| Conflicting signals | Low | Text and structured features pointing different directions |

## Recommendations

1. **More training data**: especially for underrepresented boundary cases.
2. **Class-specific thresholds**: different confidence thresholds per emotion could further calibrate the uncertainty flag.
3. **Ordinal regression approaches**: while GradientBoosting treats classes categorically, ordinal-specific loss functions could prevent distant misclassifications.
*(Note: We fully implemented advanced text features—length, word-level, char-level—and interaction terms resulting in maximal sklearn performance).*

---

## Label Noise & Low-Confidence Predictions

### Strategy

Label noise is handled post-prediction through **confidence-based flagging**.
Any test sample whose combined confidence score (average of emotion + intensity model probabilities) falls below the configured `CONFIDENCE_THRESHOLD` (default: `0.6`) is flagged as **low-confidence**.

### Implementation

The function `detect_low_confidence_predictions()` in `src/evaluation/error_analysis.py`:
- Accepts predicted labels and confidence scores
- Flags all samples below the threshold
- Logs flagged samples to `outputs/low_confidence_log.csv` (appended per model)

This CSV contains: `sample_index`, `predicted`, `confidence`, `model`.

### How flags are used

| Mechanism | Where | Effect |
|-----------|-------|--------|
| `uncertain_flag` column | `outputs/predictions.csv` | Marks sample as unreliable |
| `low_confidence_log.csv` | `outputs/` | Full log for manual review / down-weighting |
| Supportive message hedging | `generate_message()` | User-facing messages reflect uncertainty |

### Insight

Most low-confidence samples fall into semantically overlapping classes (e.g., calm ↔ neutral, restless ↔ overwhelmed). These are not necessarily mislabelled — they are inherently ambiguous cases where even human raters might disagree. Rather than discarding them, we flag and hedge, preserving coverage while signalling reduced reliability.

### Recommendations for label noise reduction

1. **Inter-annotator agreement**: collect labels from ≥2 raters and use majority voting to reduce noise.
2. **Soft labels**: replace hard class assignments with probability distributions for ambiguous cases.
3. **Confidence-weighted training**: down-weight low-confidence samples during model training (future work).
