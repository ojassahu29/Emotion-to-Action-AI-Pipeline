"""
src/api/app.py — Lightweight FastAPI endpoint for the Emotion-to-Action pipeline.

Endpoint:
  POST /predict
  Input  JSON: journal_text, sleep_hours, stress_level, energy_level,
               duration_min, ambience_type, time_of_day, previous_day_mood,
               face_emotion_hint, reflection_quality
  Output JSON: emotion, intensity, action, timing, confidence, message

Run:
  cd ml-emotion-assistant
  uvicorn src.api.app:app --reload --port 8000

Test:
  curl -X POST http://127.0.0.1:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"journal_text":"I feel overwhelmed today","sleep_hours":5,
         "stress_level":5,"energy_level":1,"duration_min":10,
         "ambience_type":"quiet","time_of_day":"morning",
         "previous_day_mood":"bad","face_emotion_hint":"stressed",
         "reflection_quality":"low"}'
"""
import os
import sys

# Ensure project root is on sys.path so `src.*` imports resolve correctly
_HERE = os.path.dirname(os.path.abspath(__file__))          # src/api/
_ROOT = os.path.dirname(os.path.dirname(_HERE))             # ml-emotion-assistant/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Pipeline imports ───────────────────────────────────────────────────────────
from src.inference.predict import load_models, predict_emotion, predict_intensity
from src.inference.uncertainty import compute_confidence, compute_uncertain_flag
from src.preprocessing.clean_text import clean_text_column
from src.preprocessing.feature_engineering import (
    extract_text_stats, combine_features,
)
from src.decision_engine.rules import get_action
from src.decision_engine.scheduler import get_timing
from src.decision_engine.recommendation import generate_message
from src.utils.helpers import load_pickle
from src.utils.config import MODELS_DIR, CONFIDENCE_THRESHOLD

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion-to-Action API",
    description="Predict emotional state, intensity, and recommended action from journal text.",
    version="1.0.0",
)


# ── Load models + vectorizers once at startup ─────────────────────────────────
@app.on_event("startup")
def load_resources():
    """Load all trained artifacts into module-level globals at startup."""
    global _emotion_model, _intensity_model
    global _word_tfidf, _char_tfidf, _struct_columns

    _emotion_model, _intensity_model = load_models()
    _word_tfidf    = load_pickle(os.path.join(MODELS_DIR, "word_tfidf.pkl"))
    _char_tfidf    = load_pickle(os.path.join(MODELS_DIR, "char_tfidf.pkl"))
    _struct_columns = load_pickle(os.path.join(MODELS_DIR, "structured_columns.pkl"))


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    journal_text:       str   = Field(..., description="User's journal / reflection text")
    sleep_hours:        float = Field(7.0, ge=0, le=24, description="Hours slept")
    stress_level:       int   = Field(3,   ge=1, le=5,  description="Self-reported stress (1–5)")
    energy_level:       int   = Field(3,   ge=1, le=5,  description="Self-reported energy (1–5)")
    duration_min:       float = Field(10.0, ge=0,       description="Reflection duration in minutes")
    ambience_type:      str   = Field("quiet",   description="E.g. quiet, noisy, nature")
    time_of_day:        str   = Field("morning", description="morning / afternoon / evening / night")
    previous_day_mood:  str   = Field("neutral", description="E.g. good, bad, neutral")
    face_emotion_hint:  str   = Field("neutral", description="E.g. calm, stressed, happy")
    reflection_quality: str   = Field("medium",  description="low / medium / high")


class PredictResponse(BaseModel):
    emotion:       str
    intensity:     int
    action:        str
    timing:        str
    confidence:    float
    uncertain:     bool
    message:       str


# ── Helper: build feature matrix from a single request ────────────────────────
def _build_features(req: PredictRequest) -> csr_matrix:
    """
    Replicate the training feature pipeline for a single input sample.
    Returns a (1, n_features) sparse matrix compatible with saved models.
    """
    # ── Text features ──────────────────────────────────────────────────────
    text_series = pd.Series([req.journal_text], name="journal_text")

    # Clean text the same way training did
    tmp_df = pd.DataFrame({"journal_text": text_series})
    tmp_df = clean_text_column(tmp_df)
    clean_text = tmp_df["journal_text"]

    X_word = _word_tfidf.transform(clean_text)
    X_char = _char_tfidf.transform(clean_text)
    stats  = extract_text_stats(clean_text)

    # ── Structured features ────────────────────────────────────────────────
    struct_raw = {
        "duration_min":       [req.duration_min],
        "sleep_hours":        [req.sleep_hours],
        "energy_level":       [req.energy_level],
        "stress_level":       [req.stress_level],
        "ambience_type":      [req.ambience_type],
        "time_of_day":        [req.time_of_day],
        "previous_day_mood":  [req.previous_day_mood],
        "face_emotion_hint":  [req.face_emotion_hint],
        "reflection_quality": [req.reflection_quality],
    }
    struct_df = pd.DataFrame(struct_raw)

    # Numeric features + interaction terms (mirror feature_engineering.py)
    numeric_cols = ["duration_min", "sleep_hours", "energy_level", "stress_level"]
    struct_encoded = pd.DataFrame()
    for col in numeric_cols:
        struct_encoded[col] = struct_df[col]

    stress  = struct_encoded["stress_level"].values[0]
    energy  = struct_encoded["energy_level"].values[0]
    struct_encoded["stress_x_energy"] = stress * energy
    struct_encoded["burnout_index"]    = stress / (energy + 1.0)

    # One-hot encode categorical columns
    cat_cols = ["ambience_type", "time_of_day", "previous_day_mood",
                "face_emotion_hint", "reflection_quality"]
    for col in cat_cols:
        dummies = pd.get_dummies(struct_df[col].fillna("unknown"), prefix=col)
        struct_encoded = pd.concat([struct_encoded, dummies.astype(float)], axis=1)

    # Align to training columns (fill missing OHE columns with 0)
    struct_encoded = struct_encoded.reindex(columns=_struct_columns, fill_value=0)

    # Combine all features
    X = combine_features(X_word, X_char, stats, struct_encoded)
    return X


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, summary="Predict emotion and recommend action")
def predict(req: PredictRequest):
    """
    Full Emotion-to-Action pipeline for a single journal entry.

    Returns predicted emotion, intensity, recommended action & timing,
    confidence score, uncertainty flag, and a supportive message.
    """
    try:
        X = _build_features(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    try:
        # ── Predictions ────────────────────────────────────────────────────
        emotion   = predict_emotion(_emotion_model, X)[0]
        intensity = int(predict_intensity(_intensity_model, X)[0])

        # ── Confidence & uncertainty ───────────────────────────────────────
        em_conf  = float(compute_confidence(_emotion_model, X)[0])
        int_conf = float(compute_confidence(_intensity_model, X)[0])
        confidence    = round((em_conf + int_conf) / 2, 4)
        uncertain_val = int(compute_uncertain_flag(
            np.array([confidence]), threshold=CONFIDENCE_THRESHOLD
        )[0])

        # ── Decision engine ────────────────────────────────────────────────
        action = get_action(emotion, intensity, req.stress_level, req.energy_level)
        timing = get_timing(emotion, intensity, req.stress_level, req.energy_level)

        # ── Supportive message ─────────────────────────────────────────────
        message = generate_message(
            emotion, intensity, action, timing, uncertain_flag=uncertain_val
        )

        return PredictResponse(
            emotion=emotion,
            intensity=intensity,
            action=action,
            timing=timing,
            confidence=confidence,
            uncertain=bool(uncertain_val),
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok", "models_loaded": True}
