# Emotion-to-Action AI Pipeline

## Approach

This system predicts a user's emotional state and intensity from **journal text** and **structured behavioral features** (sleep, stress, energy, etc.), then recommends an **action** (rest, breathing, journaling, focus work) and **timing** (now, later, tomorrow) using rule-based logic.

### Pipeline Overview

1. **Text cleaning** — lowercase, remove special chars, strip whitespace
2. **Feature engineering** — Advanced TF-IDF (word unigrams/bigrams & character n-grams 3-5), normalized text stats (length/word count), and structured feature interactions (e.g. `stress_x_energy`).
3. **Classification** — `LinearSVC` (calibrated) for emotion (6 classes), `GradientBoostingClassifier` for intensity (5 classes).
4. **Uncertainty** — confidence = mean of max probabilities from both models; `uncertain_flag = 1 if confidence < 0.6`.
5. **Decision engine** — rule-based action/timing assignment using predicted emotion, intensity, stress, and energy levels.

## Model Choices

| Model | Task | Algorithm | Reason |
|-------|------|-----------|--------|
| Emotion | 6-class classification | Calibrated LinearSVC | Best performance on high-dimensional sparse text data, lightweight, provides `predict_proba` via calibration. |
| Intensity | 5-class classification | GradientBoostingClassifier | Handles complex non-linear splits and mixed feature types well for ordinal intensity classification. |

The models are tuned for maximizing accuracy without deep learning dependencies, using only `sklearn`.

## Intensity Modeling Justification

**Intensity is modeled as classification (not regression):**

1. **Discrete ordinal values** — intensity is 1–5 with no fractional values, making it naturally categorical
2. **Uniform distribution** — ~230 samples per class, no class imbalance concerns
3. **Uncertainty estimation** — classification provides `predict_proba()`, essential for computing confidence scores and `uncertain_flag`; regression models only give point estimates without natural probability distributions
4. **Bounded output** — classification guarantees predictions in {1,2,3,4,5}; regression could predict values outside this range
5. **Consistent pipeline** — both models use the same architecture, simplifying the codebase

## Feature Engineering

- **Text Features**: Word-level TF-IDF (1-2 n-grams) & Character-level TF-IDF (3-5 n-grams) capturing rich semantic & morphological signals. Also extracted text statistics.
- **Structured Features**: Imputed numeric inputs with calculated interactions (`stress_x_energy` and `burnout_index`). One-hot encoding for categorical variables.
- **Combination**: Sparse horizontal concatenation of all dense strings, text statistics, and one-hot structured data to create a high-fidelity unified feature space.

## Decision Engine Rules

### Action Rules
| Condition | Action |
|-----------|--------|
| overwhelmed OR (stress ≥ 4 AND energy ≤ 2) | rest |
| restless OR (stress ≥ 3 AND intensity ≥ 4) | breathing |
| focused OR (energy ≥ 4 AND stress ≤ 2) | focus_work |
| calm / neutral / mixed | journaling |

### Timing Rules
| Condition | Timing |
|-----------|--------|
| overwhelmed OR (stress ≥ 4 AND intensity ≥ 4) | now |
| calm AND stress ≤ 2 | tomorrow |
| Default | later |
