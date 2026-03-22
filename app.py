"""
app.py — Streamlit UI demo for the Emotion-to-Action AI Pipeline.

Run:
  cd ml-emotion-assistant
  streamlit run app.py

No external HTTP calls — calls pipeline functions directly for zero-latency inference.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Ensure project root on path so src.* imports resolve
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion-to-Action AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #3ECFCF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #888;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 28px;
}

/* Result cards */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-top: 20px;
}
.result-card {
    background: linear-gradient(145deg, #1E1E2E, #252540);
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid rgba(108,99,255,0.25);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.result-card:hover { transform: translateY(-3px); }
.card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9B8FFF;
    font-weight: 600;
    margin-bottom: 4px;
}
.card-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #EAEAF5;
}
.card-value-small {
    font-size: 1.1rem;
    font-weight: 600;
    color: #EAEAF5;
}

/* Message box */
.message-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-left: 4px solid #6C63FF;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 22px;
    font-size: 1.05rem;
    color: #D0D0E8;
    line-height: 1.65;
    font-style: italic;
    box-shadow: 0 4px 24px rgba(108,99,255,0.15);
}

/* Uncertainty badge */
.badge-uncertain {
    display: inline-block;
    background: rgba(255,165,0,0.15);
    border: 1px solid #FFA500;
    color: #FFA500;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-confident {
    display: inline-block;
    background: rgba(62,207,107,0.15);
    border: 1px solid #3ECF6B;
    color: #3ECF6B;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 24px 0;
}

/* Analyze button */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #3ECFCF) !important;
    color: white !important;
    border: none !important;
    padding: 12px 36px !important;
    border-radius: 30px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(108,99,255,0.6) !important;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    emotion_model, intensity_model = load_models()
    word_tfidf     = load_pickle(os.path.join(MODELS_DIR, "word_tfidf.pkl"))
    char_tfidf     = load_pickle(os.path.join(MODELS_DIR, "char_tfidf.pkl"))
    struct_columns = load_pickle(os.path.join(MODELS_DIR, "structured_columns.pkl"))
    return emotion_model, intensity_model, word_tfidf, char_tfidf, struct_columns


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER (single sample)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(text, sleep, stress, energy, duration,
                   ambience, time_of_day, prev_mood, face_hint, ref_quality,
                   word_tfidf, char_tfidf, struct_columns):
    # Text
    text_series = pd.Series([text], name="journal_text")
    tmp_df = pd.DataFrame({"journal_text": text_series})
    tmp_df = clean_text_column(tmp_df)
    clean  = tmp_df["journal_text"]

    X_word = word_tfidf.transform(clean)
    X_char = char_tfidf.transform(clean)
    stats  = extract_text_stats(clean)

    # Structured
    numeric = pd.DataFrame({
        "duration_min": [duration],
        "sleep_hours":  [sleep],
        "energy_level": [energy],
        "stress_level": [stress],
    })
    numeric["stress_x_energy"] = stress * energy
    numeric["burnout_index"]   = stress / (energy + 1.0)

    cat_df = pd.DataFrame({
        "ambience_type":      [ambience],
        "time_of_day":        [time_of_day],
        "previous_day_mood":  [prev_mood],
        "face_emotion_hint":  [face_hint],
        "reflection_quality": [ref_quality],
    })
    for col in cat_df.columns:
        dummies = pd.get_dummies(cat_df[col].fillna("unknown"), prefix=col)
        numeric = pd.concat([numeric, dummies.astype(float)], axis=1)

    numeric = numeric.reindex(columns=struct_columns, fill_value=0)
    return combine_features(X_word, X_char, stats, numeric)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Emotion-to-Action AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Reflect. Understand. Act. — Get personalised emotional insights from your journal.</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("📓 Journal Entry")
    journal_text = st.text_area(
        "What's on your mind today?",
        height=140,
        placeholder="I've been feeling overwhelmed with work lately. Sleep has been poor and I can't seem to focus...",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("⚙️ Context")

    col1, col2, col3 = st.columns(3)
    with col1:
        sleep_hours  = st.slider("😴 Sleep (hours)",   0.0, 12.0, 7.0, 0.5)
        stress_level = st.slider("😰 Stress level",    1,   5,    3)
    with col2:
        energy_level = st.slider("⚡ Energy level",    1,   5,    3)
        duration_min = st.slider("⏱ Reflection (min)", 0,   60,   10)
    with col3:
        ambience_type = st.selectbox(
            "🌿 Ambience", ["quiet", "noisy", "nature", "home", "office", "café"]
        )
        time_of_day = st.selectbox(
            "🕐 Time of day", ["morning", "afternoon", "evening", "night"]
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        prev_mood = st.selectbox(
            "📅 Yesterday's mood", ["good", "neutral", "bad", "mixed"]
        )
    with col5:
        face_hint = st.selectbox(
            "😊 Face emotion hint",
            ["neutral", "calm", "stressed", "happy", "sad", "anxious", "angry"]
        )
    with col6:
        ref_quality = st.selectbox(
            "📝 Reflection quality", ["low", "medium", "high"]
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Analyze")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    if not journal_text.strip():
        st.warning("Please enter some journal text before analyzing.")
    else:
        with st.spinner("Reading your reflection..."):
            emotion_model, intensity_model, word_tfidf, char_tfidf, struct_columns = get_models()

            X = build_features(
                journal_text, sleep_hours, stress_level, energy_level, duration_min,
                ambience_type, time_of_day, prev_mood, face_hint, ref_quality,
                word_tfidf, char_tfidf, struct_columns
            )

            emotion   = predict_emotion(emotion_model, X)[0]
            intensity = int(predict_intensity(intensity_model, X)[0])

            em_conf   = float(compute_confidence(emotion_model, X)[0])
            int_conf  = float(compute_confidence(intensity_model, X)[0])
            confidence = round((em_conf + int_conf) / 2, 4)
            uncertain  = int(compute_uncertain_flag(
                np.array([confidence]), threshold=CONFIDENCE_THRESHOLD
            )[0])

            action = get_action(emotion, intensity, stress_level, energy_level)
            timing = get_timing(emotion, intensity, stress_level, energy_level)
            message = generate_message(emotion, intensity, action, timing, uncertain_flag=uncertain)

        # ── Display results ────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Emoji maps
        emotion_emoji = {
            "overwhelmed": "😮‍💨", "restless": "😤", "mixed": "😕",
            "calm": "😌", "neutral": "😐", "focused": "🎯",
        }
        action_emoji = {
            "rest": "💤", "breathing": "🌬️", "journaling": "✍️", "focus_work": "💡"
        }
        timing_emoji = {"now": "⚡", "later": "🕑", "tomorrow": "🌅"}

        e_emoji = emotion_emoji.get(emotion, "🙂")
        a_emoji = action_emoji.get(action, "✅")
        t_emoji = timing_emoji.get(timing, "🕒")

        st.markdown("### 📊 Analysis Results")

        # Result cards via columns
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("🧠 Emotion",   f"{e_emoji} {emotion.capitalize()}")
        r2.metric("📈 Intensity", f"{'🔥' * intensity}  {intensity}/5")
        r3.metric("🎯 Action",    f"{a_emoji} {action.replace('_', ' ').capitalize()}")
        r4.metric("⏰ Timing",    f"{t_emoji} {timing.capitalize()}")
        r5.metric("📊 Confidence", f"{confidence:.0%}")

        # Confidence bar
        conf_pct = int(confidence * 100)
        conf_color = "#3ECF6B" if confidence >= CONFIDENCE_THRESHOLD else "#FFA500"
        st.markdown(f"""
        <div style="margin-top:-8px; margin-bottom:16px;">
          <div style="font-size:0.72rem; color:#888; margin-bottom:4px;">Confidence: {conf_pct}%</div>
          <div style="background:rgba(255,255,255,0.08); border-radius:8px; height:8px;">
            <div style="width:{conf_pct}%; background:{conf_color}; border-radius:8px; height:8px;
                        transition:width 0.5s ease;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if uncertain:
            st.markdown(
                '<span class="badge-uncertain">⚠️ Low confidence — interpretation may vary</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge-confident">✅ High confidence</span>',
                unsafe_allow_html=True
            )

        # Supportive message
        st.markdown(
            f'<div class="message-box">💬 &nbsp;{message}</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>Emotion-to-Action AI Pipeline · Powered by scikit-learn · No external APIs</p>",
    unsafe_allow_html=True
)
