# config.py

# ---------------------------------------------------------
# MODES
# ---------------------------------------------------------
# Set to True for extra visual debug overlays (HOG box, fingertip, centroid, SVM score,
# exclusion band). Set to False for a cleaner game-only view.
DEBUG_MODE = True

# ---------------------------------------------------------
# HOG + SVM CONFIG
# ---------------------------------------------------------
HOG_WIN_SIZE = (64, 64)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9

SVM_MODEL_PATH = "models/hog_finger_svm.joblib"

SLIDE_STEP = 32  # pixels between windows on each scaled frame

# Base scale for detection; we'll build a small pyramid around this
DETECT_SCALE = 0.5

# Multi-scale pyramid factors (relative to full-res)
PYRAMID_SCALES = [1.0, 0.75, DETECT_SCALE]  # e.g. full, 3/4, and 1/2 size

# SVM decision-function threshold.
SCORE_THRESHOLD = 0.3

# How often to run the full HOG detection (every N frames)
DETECT_EVERY_N = 3

# Exclusion zone (middle vertical band in the FULL-RES frame)
EXCLUDE_X1_FRAC = 0.30  # left edge of middle band (30% of width)
EXCLUDE_X2_FRAC = 0.70  # right edge of middle band (70% of width)
EXCLUDE_Y1_FRAC = 0.0
EXCLUDE_Y2_FRAC = 1.0

# Fingertip motion sanity check
# Max physically plausible fingertip speed (pixels per second).
# At 30 FPS and 1500 px/s, max per-frame jump ~ 50 px.
MAX_FINGER_SPEED = 1500.0

# ---------------------------------------------------------
# GAME CONFIG
# ---------------------------------------------------------
# Paddle config
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 120
PADDLE_X_OFFSET = 40  # distance from each side wall
PADDLE_ALPHA = 0.4    # smoothing factor for player paddle movement

# AI paddle config
AI_PADDLE_ALPHA = 0.2  # smoothing factor for AI paddle following ball

# Ball config
BALL_RADIUS = 12
BALL_SPEED_X = 350.0   # px/sec (horizontal)
BALL_SPEED_Y = 250.0   # px/sec (vertical)
