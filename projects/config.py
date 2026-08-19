
# =============================================================================
# Configuración global del proyecto TFM Peg Transfer
# =============================================================================

# --- Detector ---
DETECTOR_IMGSZ     = 1280
DETECTOR_CONF      = 0.4
DETECTOR_IOU       = 0.5
STABILIZE_MAX_FRAMES = 30

# --- Stabilizer ---
CLASS_CONF = {
    'ring'    : 0.40,
    'TFM'     : 0.65,
    'peg'     : 0.50,
    'platform': 0.50,
}
CLASS_MAX = {
    'ring'    : 6,
    'TFM'     : 2,
    'peg'     : 12,
    'platform': 1,
}
N_PEGS        = 12
OUTLIER_SIGMA = 2.0

# --- Tracker — Kalman filter (ajustado para instrumentos quirúrgicos) ---
# Valores por defecto de ultralytics (calibrados para peatones):
#   _std_weight_position = 1/20  (0.05)
#   _std_weight_velocity = 1/160 (0.00625)
# Valores ajustados (paper two-point tracking):
KALMAN_STD_WEIGHT_POSITION = 1. / 10   # 0.10 — mayor tolerancia posicional
KALMAN_STD_WEIGHT_VELOCITY = 1. / 40   # 0.025 — mayor tolerancia de velocidad

# --- Tracker — boxmot ByteTrack ---
TRACK_THRESH   = 0.60   # umbral alto de confianza (track_thresh)
MIN_CONF       = 0.10   # umbral bajo de confianza (min_conf)
TRACK_BUFFER   = 60     # frames que mantiene un track perdido antes de eliminarlo
MATCH_THRESH   = 0.80   # umbral de costo max (costo=1-IoU); 0.80 → IoU>=0.20
FRAME_RATE     = 30     # fps del video de entrada

# --- Tracker — mini-bbox sintético ---
MINI_BBOX_FRACTION = 0.60   # fracción del bbox original; ajustar empíricamente


