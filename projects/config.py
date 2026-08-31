
# =============================================================================
# Configuración global del proyecto TFM Peg Transfer
# =============================================================================

# --- Detector ---
DETECTOR_IMGSZ     = 1280
DETECTOR_CONF      = 0.4
DETECTOR_IOU       = 0.5

# --- Ventana de inicialización (objetos estáticos) ---
# Expresada en SEGUNDOS, no en frames: el dataset mezcla videos de 30 y 60 fps,
# y un valor fijo en frames significaba 1.0 s en unos videos y 0.5 s en otros.
# Los frames se derivan del fps real de cada video.
#
# La estabilización NO usa una ventana fija: acumula solo "frames completos"
# (los 12 pegs y los 6 rings detectados a la vez) y se detiene al juntar
# INIT_COMPLETE_FRAMES de ellos. Motivo: K-means con k=12 sobre frames donde
# solo hay 11 pegs visibles no falla — parte un peg real en dos, produciendo
# errores de hasta 279 px. Filtrando por completitud ese modo de fallo
# desaparece (error < 1.2 px con 8 frames completos, medido sobre P01 y P07).
INIT_MAX_SECONDS     = 2.0   # tope de búsqueda; si no se juntan, se avisa por log
INIT_COMPLETE_FRAMES = 8     # frames completos suficientes (~1 px de error)

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

# Punto de apoyo del peg (donde realmente descansa el ring): centro del
# BASE_MASK_PCT% inferior de la máscara del peg. El centroide de la máscara
# cae a media altura del poste y queda ~58 px del ring en reposo; esta base
# lo baja a ~18 px. Ver Stabilizer._compute_anclaje.
BASE_MASK_PCT = 10

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
TRACK_BUFFER   = 60     # frames de vida de un track perdido a 30 fps; ByteTrack
                        # lo escala con frame_rate para mantener ~2 s reales
MATCH_THRESH   = 0.80   # umbral de costo max (costo=1-IoU); 0.80 → IoU>=0.20
FRAME_RATE     = 30     # fps de RESPALDO si el stable JSON no trae fps real

# Asignación track→ring: radio máximo para aceptar un track como propio.
# Sin límite, un ring cuyo track murió (oclusión) recibía el id del track
# más cercano que existiera — aunque estuviera a 400 px, sobre OTRO ring
# (medido: 175 teleports y 2.8% de frames con id duplicado en P01 Trial1-2).
# Referencia: mini-bbox de ring ~75 px de ancho, desplazamiento p99 = 28 px/frame.
RING_TRACK_MAX_DIST = 80   # px

# --- Tracker — mini-bbox sintético ---
MINI_BBOX_FRACTION = 0.60   # fracción del bbox original; ajustar empíricamente


