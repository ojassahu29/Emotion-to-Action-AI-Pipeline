"""
Recommendation engine: combines rules and scheduling for all samples.
"""
import pandas as pd
from src.decision_engine.rules import get_action
from src.decision_engine.scheduler import get_timing


def generate_recommendations(df):
    """
    Generate action and timing recommendations for each row.

    Expects df to have columns:
      emotional_state, intensity, stress_level, energy_level

    Returns: df with added 'action' and 'timing' columns.
    """
    df = df.copy()

    actions = []
    timings = []

    for _, row in df.iterrows():
        emotion = row["emotional_state"]
        intensity = int(row["intensity"])
        stress = int(row["stress_level"])
        energy = int(row["energy_level"])

        actions.append(get_action(emotion, intensity, stress, energy))
        timings.append(get_timing(emotion, intensity, stress, energy))

    df["action"] = actions
    df["timing"] = timings

    return df
