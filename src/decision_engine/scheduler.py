"""
Scheduling logic: determines timing for recommended action.
Timing options: now, later, tomorrow
"""


def get_timing(emotion, intensity, stress_level, energy_level):
    """
    Determine when the action should be taken.

    Rules:
    - High urgency (overwhelmed, high stress, high intensity) → now
    - Moderate state → later
    - Low urgency (calm, low stress, low intensity) → tomorrow
    """
    # Urgent: act now
    if emotion == "overwhelmed":
        return "now"
    if stress_level >= 4 and intensity >= 4:
        return "now"
    if emotion == "restless" and intensity >= 3:
        return "now"

    # Low urgency: can wait
    if emotion == "calm" and stress_level <= 2:
        return "tomorrow"
    if emotion == "neutral" and intensity <= 2 and stress_level <= 2:
        return "tomorrow"

    # Default: later
    return "later"
