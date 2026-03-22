"""
message_templates.py — Template pool for supportive conversational messages.

Each key is a tuple of (emotion, action).
Each value is a list of message templates (strings).
Placeholders:
  {intensity_word} — e.g. "slightly", "moderately", "very"
  {timing_phrase}  — e.g. "right now", "a bit later", "tomorrow"

Usage: randomly pick one from the list to add variation.
"""

# ── Intensity → descriptive adverb ────────────────────────────────────────────
INTENSITY_WORDS = {
    1: "slightly",
    2: "a bit",
    3: "moderately",
    4: "quite",
    5: "very",
}

# ── Timing → human phrase ─────────────────────────────────────────────────────
TIMING_PHRASES = {
    "now":      ["right now", "immediately", "as soon as possible"],
    "later":    ["a bit later today", "when you get a chance", "in a little while"],
    "tomorrow": ["tomorrow", "when you're ready", "at your own pace"],
}

# ── Uncertainty hedge prefixes ─────────────────────────────────────────────────
UNCERTAINTY_HEDGES = [
    "I might be mistaken, but it seems like",
    "I'm not entirely sure, but it looks like",
    "This is my best guess, but it appears that",
    "I could be off, but it seems like",
]

# ── Message templates by (emotion, action) ────────────────────────────────────
# Each entry: list of 3-4 variants. Use {intensity_word} and {timing_phrase}.
TEMPLATES = {

    # ── REST ─────────────────────────────────────────────────────────────────
    ("overwhelmed", "rest"): [
        "You seem {intensity_word} overwhelmed right now. Your mind and body need a break. "
        "Try to rest {timing_phrase} — even a short pause can make a big difference.",
        "It looks like you're carrying {intensity_word} too much right now. "
        "Give yourself permission to rest {timing_phrase}. You deserve it.",
        "Feeling overwhelmed is a signal your system needs recovery. "
        "Step away and rest {timing_phrase} — come back refreshed.",
        "It seems like things are {intensity_word} heavy for you right now. "
        "A proper rest {timing_phrase} will help you reset.",
    ],
    ("restless", "rest"): [
        "You're feeling {intensity_word} restless. "
        "Paradoxically, resting {timing_phrase} can calm that inner agitation.",
        "When restlessness peaks, rest is often the best medicine. "
        "Take a break {timing_phrase} and let your body settle.",
        "It seems like your energy is {intensity_word} scattered right now. "
        "Some quiet rest {timing_phrase} could help ground you.",
    ],
    ("neutral", "rest"): [
        "Your body might benefit from a gentle rest {timing_phrase}. "
        "Even when things feel neutral, recharging is always a good idea.",
        "A short rest {timing_phrase} can help maintain your balance. Nothing urgent — just a small reset.",
    ],
    ("mixed", "rest"): [
        "With mixed emotions swirling around, rest can be grounding. "
        "Give yourself some quiet time {timing_phrase}.",
        "It's okay to feel a mix of things. Resting {timing_phrase} gives your mind space to settle.",
    ],
    ("calm", "rest"): [
        "You're feeling calm — a perfect time for a gentle, restorative rest {timing_phrase}.",
        "A calm state pairs beautifully with quality rest. Take that time {timing_phrase}.",
    ],
    ("focused", "rest"): [
        "Even focused minds need breaks! Schedule a short rest {timing_phrase} to stay sharp.",
        "You're focused, which is great — just don't forget to rest {timing_phrase} so you sustain it.",
    ],

    # ── BREATHING ────────────────────────────────────────────────────────────
    ("overwhelmed", "breathing"): [
        "You seem {intensity_word} overwhelmed. A guided breathing exercise, done {timing_phrase}, "
        "can help bring your nervous system back to balance.",
        "When overwhelm hits, breathing is your quickest reset tool. "
        "Try a few deep breaths {timing_phrase}.",
        "Your stress signals suggest a breathing exercise would help {timing_phrase}. "
        "Even 5 slow breaths can shift your state.",
    ],
    ("restless", "breathing"): [
        "You're feeling {intensity_word} restless. A short breathing exercise {timing_phrase} "
        "can help slow things down and ease that tension.",
        "Restlessness often responds well to controlled breathing. "
        "Try box breathing {timing_phrase}: inhale 4s, hold 4s, exhale 4s.",
        "It looks like your mind is {intensity_word} unsettled. "
        "A breathing exercise {timing_phrase} can act as an anchor.",
        "A few intentional breaths {timing_phrase} could do wonders for the restlessness you're feeling.",
    ],
    ("mixed", "breathing"): [
        "Mixed emotions can create inner noise. "
        "A breathing exercise {timing_phrase} can help you find clarity.",
        "When feelings are tangled, breathing is a simple way to create space. "
        "Try it {timing_phrase}.",
    ],
    ("neutral", "breathing"): [
        "A short mindful breathing session {timing_phrase} can sharpen your focus and awareness.",
        "Breathing exercises aren't just for stress — they also deepen calm. Try one {timing_phrase}.",
    ],
    ("calm", "breathing"): [
        "You're already calm — a breathing exercise {timing_phrase} can deepen that stillness even further.",
        "Reinforce your calm state with a brief breathing practice {timing_phrase}.",
    ],
    ("focused", "breathing"): [
        "Use a breathing technique {timing_phrase} to lock in that focus and keep distractions at bay.",
        "A quick breath reset {timing_phrase} will help sustain your concentration.",
    ],

    # ── JOURNALING ───────────────────────────────────────────────────────────
    ("overwhelmed", "journaling"): [
        "You seem {intensity_word} overwhelmed. Writing down your thoughts {timing_phrase} "
        "can help you process what you're carrying.",
        "Journaling {timing_phrase} can externalise the weight you're feeling — "
        "getting it on paper makes it more manageable.",
        "When overwhelm is {intensity_word}, journaling {timing_phrase} gives your emotions a safe outlet.",
    ],
    ("restless", "journaling"): [
        "You seem {intensity_word} restless. Journaling {timing_phrase} can help you "
        "untangle what's driving that unease.",
        "Writing {timing_phrase} is a great way to channel restless energy productively.",
        "Your restlessness might have something to say — a journal entry {timing_phrase} can help you listen.",
    ],
    ("mixed", "journaling"): [
        "Mixed feelings are often best explored through writing. "
        "Try journaling {timing_phrase} to understand what you're experiencing.",
        "When emotions are {intensity_word} complex, a journal entry {timing_phrase} "
        "can bring surprising clarity.",
        "Journaling {timing_phrase} is a gentle way to make sense of a complicated emotional state.",
    ],
    ("neutral", "journaling"): [
        "A neutral state is a great time to reflect. Try journaling {timing_phrase} "
        "to capture your current thoughts.",
        "Use this stable moment to journal {timing_phrase} — it's good for insight and self-awareness.",
    ],
    ("calm", "journaling"): [
        "Your calm state is perfect for thoughtful reflection. Journal {timing_phrase} "
        "to capture your insights.",
        "Writing from a place of calm {timing_phrase} often yields your clearest insights.",
    ],
    ("focused", "journaling"): [
        "Channel your focus into a journal entry {timing_phrase} — document your current goals or ideas.",
        "A journal session {timing_phrase} can help you organise and prioritise your focused thoughts.",
    ],

    # ── FOCUS WORK ───────────────────────────────────────────────────────────
    ("focused", "focus_work"): [
        "You're in a great state of focus! This is your moment — dive into deep work {timing_phrase}.",
        "Your energy and clarity are aligned. Use this window for focused, distraction-free work {timing_phrase}.",
        "You're {intensity_word} focused right now — lean into that and get your most important task done {timing_phrase}.",
        "Perfect conditions for deep work. Block off time and get into flow {timing_phrase}.",
    ],
    ("calm", "focus_work"): [
        "Your calm state is a productivity asset. "
        "Try some light but meaningful focused work {timing_phrase}.",
        "Calm focus is powerful. Use this window for thoughtful, quality work {timing_phrase}.",
    ],
    ("neutral", "focus_work"): [
        "A neutral state is actually ideal for clear-headed, focused work. "
        "Dive in {timing_phrase}.",
        "With a steady baseline, you're set up well for a focused work session {timing_phrase}.",
    ],
    ("mixed", "focus_work"): [
        "Despite mixed feelings, you have enough energy to do some focused work {timing_phrase}. "
        "Start small.",
        "Even with complex emotions, a short focused work session {timing_phrase} "
        "can create a sense of accomplishment.",
    ],
    ("overwhelmed", "focus_work"): [
        "Even when overwhelmed, small tasks can help restore a sense of control. "
        "Try a gentle work session {timing_phrase}.",
    ],
    ("restless", "focus_work"): [
        "Channelling restless energy into focused work {timing_phrase} can be very effective. "
        "Pick one task and go.",
    ],
}

# ── Fallback template (when no exact match) ───────────────────────────────────
FALLBACK_TEMPLATES = [
    "Based on how you're feeling ({intensity_word} {emotion}), "
    "a {action} session {timing_phrase} could be really beneficial.",
    "You seem {intensity_word} {emotion} right now. "
    "Trying a {action} activity {timing_phrase} is a good step forward.",
    "Your current state suggests that {action} {timing_phrase} would support your wellbeing.",
]
