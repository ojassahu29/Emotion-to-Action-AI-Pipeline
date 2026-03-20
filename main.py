"""
main.py — Full end-to-end ML pipeline.

Pipeline:
  1. Load data
  2. Clean text
  3. Feature engineering (TF-IDF + structured)
  4. Train models (emotion + intensity)
  5. Evaluate on validation set
  6. Ablation study
  7. Error analysis
  8. Predict on test data
  9. Uncertainty estimation
  10. Decision engine (action + timing)
  11. Save outputs/predictions.csv
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ── Imports ──────────────────────────────────────────────────────────────────
from src.preprocessing.data_loader import (
    load_train_data, load_test_data, split_features_targets, get_features
)
from src.preprocessing.clean_text import clean_text_column
from src.preprocessing.feature_engineering import (
    build_word_tfidf_vectorizer, build_char_tfidf_vectorizer, 
    extract_text_stats, encode_structured, combine_features
)
from src.models.train import train_emotion_model, train_intensity_model
from src.inference.predict import predict_emotion, predict_intensity
from src.inference.uncertainty import compute_confidence, compute_uncertain_flag
from src.decision_engine.recommendation import generate_recommendations
from src.evaluation.metrics import evaluate_emotion, evaluate_intensity
from src.evaluation.ablation import run_ablation
from src.evaluation.error_analysis import analyze_errors
from src.utils.config import PREDICTIONS_FILE, OUTPUTS_DIR, MODELS_DIR
from src.utils.helpers import save_csv, save_pickle, ensure_dir


def main():
    print("=" * 60)
    print("  EMOTION-TO-ACTION AI PIPELINE")
    print("=" * 60)

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────
    print("\n[1/11] Loading data...")
    train_df = load_train_data()
    test_df = load_test_data()

    # ── 2. CLEAN TEXT ────────────────────────────────────────────────────
    print("\n[2/11] Cleaning text...")
    train_df = clean_text_column(train_df)
    test_df = clean_text_column(test_df)
    print("  Text cleaning complete.")

    # ── 3. FEATURE ENGINEERING ───────────────────────────────────────────
    print("\n[3/11] Feature engineering...")
    text_train, struct_train, y_emotion, y_intensity = split_features_targets(train_df)
    text_test, struct_test = get_features(test_df)

    # Text Stats
    stats_train = extract_text_stats(text_train)
    stats_test = extract_text_stats(text_test)

    # TF-IDF Word & Char
    word_tfidf = build_word_tfidf_vectorizer()
    char_tfidf = build_char_tfidf_vectorizer()
    
    X_word_train = word_tfidf.fit_transform(text_train)
    X_word_test = word_tfidf.transform(text_test)
    
    X_char_train = char_tfidf.fit_transform(text_train)
    X_char_test = char_tfidf.transform(text_test)

    # Encode structured features — fit on train, align test
    struct_train_encoded = encode_structured(struct_train)
    struct_test_encoded = encode_structured(struct_test)

    # Align columns (test may have different one-hot columns)
    struct_test_encoded = struct_test_encoded.reindex(
        columns=struct_train_encoded.columns, fill_value=0
    )

    # Combine all sparse and dense features
    X_train = combine_features(X_word_train, X_char_train, stats_train, struct_train_encoded)
    X_test = combine_features(X_word_test, X_char_test, stats_test, struct_test_encoded)

    print(f"  Train features: {X_train.shape}")
    print(f"  Test features:  {X_test.shape}")

    # Save vectorizers and feature columns for reproducibility
    ensure_dir(MODELS_DIR)
    save_pickle(word_tfidf, os.path.join(MODELS_DIR, "word_tfidf.pkl"))
    save_pickle(char_tfidf, os.path.join(MODELS_DIR, "char_tfidf.pkl"))
    save_pickle(list(struct_train_encoded.columns),
                os.path.join(MODELS_DIR, "structured_columns.pkl"))

    # ── 4. TRAIN MODELS ─────────────────────────────────────────────────
    print("\n[4/11] Training emotion model...")
    emotion_model, X_val_em, y_val_em = train_emotion_model(X_train, y_emotion)

    print("\n[5/11] Training intensity model...")
    intensity_model, X_val_in, y_val_in = train_intensity_model(X_train, y_intensity)

    # ── 5. EVALUATE ──────────────────────────────────────────────────────
    print("\n[6/11] Evaluating emotion model...")
    y_pred_em_val = predict_emotion(emotion_model, X_val_em)
    em_metrics = evaluate_emotion(y_val_em, y_pred_em_val)
    print(f"  Accuracy: {em_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {em_metrics['f1_macro']:.4f}")
    print(f"\n{em_metrics['report']}")

    print("\n[7/11] Evaluating intensity model...")
    y_pred_in_val = predict_intensity(intensity_model, X_val_in)
    in_metrics = evaluate_intensity(y_val_in, y_pred_in_val)
    print(f"  Accuracy: {in_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {in_metrics['f1_macro']:.4f}")
    print(f"\n{in_metrics['report']}")

    # ── 6. ABLATION ──────────────────────────────────────────────────────
    print("\n[8/11] Running simple ablation (Text vs All)...")
    # Combining text features only for ablation
    X_text_only_train = hstack([X_word_train, X_char_train, csr_matrix(stats_train.values)])
    ablation_results = run_ablation(X_text_only_train, X_train, y_emotion)
    for variant, metrics in ablation_results.items():
        print(f"  {variant}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")

    # ── 7. ERROR ANALYSIS ────────────────────────────────────────────────
    print("\n[9/11] Running error analysis...")
    emotion_errors = analyze_errors(
        y_val_em, y_pred_em_val, train_df.iloc[y_val_em.index],
        model_name="Emotion", n=10
    )
    intensity_errors = analyze_errors(
        y_val_in, y_pred_in_val, train_df.iloc[y_val_in.index],
        model_name="Intensity", n=10
    )

    # ── 8. PREDICT ON TEST DATA ──────────────────────────────────────────
    print("\n[10/11] Predicting on test data...")
    test_emotions = predict_emotion(emotion_model, X_test)
    test_intensity = predict_intensity(intensity_model, X_test)

    # ── 9. UNCERTAINTY ───────────────────────────────────────────────────
    emotion_confidence = compute_confidence(emotion_model, X_test)
    intensity_confidence = compute_confidence(intensity_model, X_test)

    # Overall confidence = average of both models
    confidence = (emotion_confidence + intensity_confidence) / 2
    uncertain_flag = compute_uncertain_flag(confidence)

    print(f"  Mean confidence: {confidence.mean():.4f}")
    print(f"  Uncertain samples: {uncertain_flag.sum()} / {len(uncertain_flag)}")

    # ── 10. DECISION ENGINE ──────────────────────────────────────────────
    print("\n[11/11] Generating recommendations...")
    results_df = pd.DataFrame({
        "emotional_state": test_emotions,
        "intensity": test_intensity,
        "stress_level": test_df["stress_level"].values,
        "energy_level": test_df["energy_level"].values,
    })
    results_df = generate_recommendations(results_df)

    # ── 11. SAVE OUTPUT ──────────────────────────────────────────────────
    output_df = pd.DataFrame({
        "emotional_state": results_df["emotional_state"],
        "intensity": results_df["intensity"],
        "action": results_df["action"],
        "timing": results_df["timing"],
        "confidence": np.round(confidence, 4),
        "uncertain_flag": uncertain_flag,
    })

    save_csv(output_df, PREDICTIONS_FILE)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Predictions saved to: {PREDICTIONS_FILE}")
    print(f"  Total test samples: {len(output_df)}")
    print("=" * 60)

    # Print summary stats
    print("\n  Output Summary:")
    print(f"    Emotion distribution: {dict(output_df['emotional_state'].value_counts())}")
    print(f"    Intensity distribution: {dict(output_df['intensity'].value_counts())}")
    print(f"    Action distribution: {dict(output_df['action'].value_counts())}")
    print(f"    Timing distribution: {dict(output_df['timing'].value_counts())}")
    print(f"    Uncertain predictions: {output_df['uncertain_flag'].sum()}")


if __name__ == "__main__":
    main()
