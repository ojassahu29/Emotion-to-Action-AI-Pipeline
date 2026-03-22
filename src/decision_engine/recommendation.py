"""
Recommendation engine: combines rules, scheduling, and message generation.

generate_recommendations() — batch: adds action, timing, and message columns.
generate_message()         — single-sample: returns a supportive message string.
"""
import random
import pandas as pd
from src.decision_engine.rules import get_action
from src.decision_engine.scheduler import get_timing
from src.decision_engine.message_templates import (
    TEMPLATES, FALLBACK_TEMPLATES,
    INTENSITY_WORDS, TIMING_PHRASES, UNCERTAINTY_HEDGES,
)


def generate_message(emotion, intensity, action, timing, uncertain_flag=0):
    """
    Generate a human-like, supportive message for a single prediction.

    Args:
        emotion       (str): predicted emotional state
        intensity     (int): predicted intensity (1–5)
        action        (str): recommended action (e.g. 'breathing', 'rest')
        timing        (str): recommended timing ('now', 'later', 'tomorrow')
        uncertain_flag (int): 1 = model was uncertain, 0 = confident

    Returns:
        str: a supportive, contextualised message with slight variation.
    """
    # ── Resolve descriptive words ──────────────────────────────────────────
    intensity = int(intensity)
    intensity_word = INTENSITY_WORDS.get(intensity, "moderately")

    # Pick a random timing phrase for naturalistic variation
    timing_options = TIMING_PHRASES.get(timing, ["when you can"])
    timing_phrase = random.choice(timing_options)

    # ── Select base template ───────────────────────────────────────────────
    key = (emotion, action)
    if key in TEMPLATES:
        template = random.choice(TEMPLATES[key])
    else:
        # Fallback: generic template for unseen (emotion, action) pairs
        template = random.choice(FALLBACK_TEMPLATES)

    # ── Fill placeholders ──────────────────────────────────────────────────
    message = template.format(
        intensity_word=intensity_word,
        timing_phrase=timing_phrase,
        emotion=emotion,
        action=action,
    )

    # ── Prepend uncertainty hedge if model was unsure ──────────────────────
    if uncertain_flag == 1:
        hedge = random.choice(UNCERTAINTY_HEDGES)
        # Lowercase the start of the original message before attaching
        message = f"{hedge} you're feeling {intensity_word} {emotion}. {message}"

    return message


def generate_recommendations(df, uncertain_flags=None):
    """
    Generate action, timing, and message recommendations for each row.

    Args:
        df              (pd.DataFrame): must have columns:
                            emotional_state, intensity, stress_level, energy_level
        uncertain_flags (array-like, optional): uncertainty flags (0/1) per row.
                            If None, all messages are treated as confident.

    Returns:
        df with added 'action', 'timing', and 'message' columns.
    """
    df = df.copy()

    actions  = []
    timings  = []
    messages = []

    for i, (_, row) in enumerate(df.iterrows()):
        emotion  = row["emotional_state"]
        intensity = int(row["intensity"])
        stress   = int(row["stress_level"])
        energy   = int(row["energy_level"])

        action = get_action(emotion, intensity, stress, energy)
        timing = get_timing(emotion, intensity, stress, energy)

        # Resolve uncertain flag for this row
        flag = 0
        if uncertain_flags is not None and i < len(uncertain_flags):
            flag = int(uncertain_flags[i])

        msg = generate_message(emotion, intensity, action, timing, uncertain_flag=flag)

        actions.append(action)
        timings.append(timing)
        messages.append(msg)

    df["action"]  = actions
    df["timing"]  = timings
    df["message"] = messages

    return df
