"""
Configuration: paths, constants, and column definitions.
"""
import os

# ── Project root (ml-emotion-assistant/) ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Data paths ──
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")

TRAIN_FILE = os.path.join(DATA_RAW, "Sample_arvyax_reflective_dataset.xlsx")
TEST_FILE = os.path.join(DATA_RAW, "arvyax_test_inputs_120.xlsx")

# ── Output paths ──
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, "predictions.csv")

# ── Reports ──
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ── Column definitions ──
TEXT_COL = "journal_text"
TARGET_EMOTION = "emotional_state"
TARGET_INTENSITY = "intensity"

STRUCTURED_NUMERIC = ["duration_min", "sleep_hours", "energy_level", "stress_level"]
STRUCTURED_CATEGORICAL = [
    "ambience_type", "time_of_day", "previous_day_mood",
    "face_emotion_hint", "reflection_quality"
]
STRUCTURED_COLS = STRUCTURED_NUMERIC + STRUCTURED_CATEGORICAL

# ── Model constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 2000
CONFIDENCE_THRESHOLD = 0.6
N_ESTIMATORS = 100
