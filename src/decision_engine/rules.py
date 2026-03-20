"""
Rule-based action assignment.
Maps predicted emotion, intensity, stress, and energy to an action.
Actions: rest, breathing, journaling, focus_work
"""


def get_action(emotion, intensity, stress_level, energy_level):
    """
    Determine recommended action based on predicted state.

    Rules:
    - overwhelmed OR (stress >= 4 AND energy <= 2) → rest
    - restless OR (stress >= 3 AND intensity >= 4) → breathing
    - calm OR neutral OR mixed (low energy) → journaling
    - focused OR (energy >= 4 AND stress <= 2) → focus_work
    - Default fallback → journaling
    """
    # High stress / overwhelmed → rest
    if emotion == "overwhelmed":
        return "rest"
    if stress_level >= 4 and energy_level <= 2:
        return "rest"

    # Restless / agitated → breathing
    if emotion == "restless":
        return "breathing"
    if stress_level >= 3 and intensity >= 4:
        return "breathing"

    # Focused / high energy → focus work
    if emotion == "focused":
        return "focus_work"
    if energy_level >= 4 and stress_level <= 2:
        return "focus_work"

    # Calm / neutral / mixed → journaling
    if emotion in ("calm", "neutral", "mixed"):
        return "journaling"

    # Fallback
    return "journaling"
