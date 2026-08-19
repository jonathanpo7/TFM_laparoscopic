import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import boxmot.trackers.bbox.bytetrack as _bt_module
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH

import config

logger = logging.getLogger(__name__)

TRACKED_CLASSES = {'ring', 'TFM'}

_DUMMY_IMG = np.zeros((2, 2, 3), dtype=np.uint8)


# =============================================================================
# Kalman ajustado para instrumentos quirúrgicos (paper two-point tracking)
# Reemplaza KalmanFilterXYAH en el namespace de boxmot ByteTrack para que
# los 3 lugares donde se instancia (init, _update_impl, reset) usen el nuestro.
# =============================================================================

class KalmanAjustado(KalmanFilterXYAH):
    def __init__(self, ndim: int = 4):
        super().__init__(ndim)
        self._std_weight_position = config.KALMAN_STD_WEIGHT_POSITION
        self._std_weight_velocity = config.KALMAN_STD_WEIGHT_VELOCITY

_bt_module.KalmanFilterXYAH = KalmanAjustado


# =============================================================================
# Clase principal
# =============================================================================

class Tracker:
    """
    Consume el JSON estable producido por Stabilizer y agrega track_id
    a cada deteccion dinamica (ring, TFM) usando ByteTrack.

    Tecnica two-point tracking (paper Sec. III-C):
      - ring : un mini-bbox centrado en el centroide de la mascara
      - TFM  : dos mini-bboxes en los extremos del eje dominante;
               reconciliados en un ID logico mediante logica de 3 niveles

    La asignacion de tracks a extremos se hace consumiendo el pool desde
    las posiciones de INPUT (no desde el pool compartido), garantizando que
    cada track_id se asigna a exactamente un extremo de un grasper.

    Estado mantenido entre frames:
      track_to_grasper  : {track_id: (grasper_idx, extreme_idx)}
      grasper_logical_id: {grasper_idx: logical_id}
    """

    def __init__(self, fraction=config.MINI_BBOX_FRACTION):
        self.fraction = fraction
        self._tracker = ByteTrack(
            track_thresh  = config.TRACK_THRESH,
            min_conf      = config.MIN_CONF,
            track_buffer  = config.TRACK_BUFFER,
            match_thresh  = config.MATCH_THRESH,
            frame_rate    = config.FRAME_RATE,
        )

    # ------------------------------------------------------------------
    # publico
    # ------------------------------------------------------------------

    def process(self, stable_path):
        stable_path = Path(stable_path)
        with open(stable_path) as f:
            stable = json.load(f)

        tracked_frames = []
        # track_to_grasper : {track_id -> (grasper_idx, extreme_idx)}
        # grasper_logical_id: {grasper_idx -> logical_id estable en el JSON de salida}
        track_to_grasper   = {}
        grasper_logical_id = {}

        for frame in stable['frames']:
            frame_idx = frame['frame_idx']
            dets_ring = []
            dets_tfm  = []

            for det in frame['detections']:
                cn = det['class_name']
                if cn not in TRACKED_CLASSES:
                    continue
                if cn == 'ring':
                    dets_ring.append(det)
                else:
                    dets_tfm.append(det)

            # --- Construir input en orden conocido ---
            # [rings..., TFM_0_ext1, TFM_0_ext2, TFM_1_ext1, TFM_1_ext2, ...]
            # Guardamos tambien los mini-bboxes de TFMs para la asignacion posterior.
            tracker_input  = []
            tfm_mini_bboxes = []  # [(g_idx, extreme_idx, bbox), ...]

            for det in dets_ring:
                mb = self._mini_bbox_centroid(det['mask_polygon'], det['bbox'])
                tracker_input.append([*mb, det['confidence'], det['class_id']])

            tfm_bboxes_by_gidx = {}
            for g_idx, det in enumerate(dets_tfm):
                bb1, bb2 = self._mini_bboxes_two_point(det['mask_polygon'], det['bbox'])
                tracker_input.append([*bb1, det['confidence'], det['class_id']])
                tracker_input.append([*bb2, det['confidence'], det['class_id']])
                tfm_mini_bboxes.append((g_idx, 0, bb1))
                tfm_mini_bboxes.append((g_idx, 1, bb2))
                tfm_bboxes_by_gidx[g_idx] = (bb1, bb2)

            tracked_dets = []
            if tracker_input:
                arr    = np.array(tracker_input, dtype=np.float32)
                tracks = self._tracker.update(arr, _DUMMY_IMG)

                ring_class_id = dets_ring[0]['class_id'] if dets_ring else -1
                tfm_class_id  = dets_tfm[0]['class_id']  if dets_tfm  else -1
                ring_tracks   = [t for t in tracks if int(t[6]) == ring_class_id]
                tfm_tracks    = [t for t in tracks if int(t[6]) == tfm_class_id]

                # --- rings: nearest por centroide (sin cambios) ---
                for det in dets_ring:
                    mb   = self._mini_bbox_centroid(det['mask_polygon'], det['bbox'])
                    cx   = (mb[0] + mb[2]) / 2
                    cy   = (mb[1] + mb[3]) / 2
                    tid  = self._nearest_track_id(cx, cy, ring_tracks)
                    entry = {**det, 'track_id': tid}
                    entry['centroid'] = self._compute_centroid(det['mask_polygon'], det['bbox'])
                    tracked_dets.append(entry)

                # --- TFMs: asignacion consumiendo pool desde posiciones de INPUT ---
                # Procesamos los mini-bboxes en el orden en que los construimos,
                # retirando del pool cada track asignado. Asi cada track_id va
                # exactamente a un extremo de un grasper.
                pool     = list(tfm_tracks)
                assigned = {}   # {(g_idx, extreme_idx): track_id}

                for g_idx, ext_idx, bbox in tfm_mini_bboxes:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    pool_i, tid = self._nearest_from_pool(cx, cy, pool)
                    assigned[(g_idx, ext_idx)] = tid
                    if pool_i >= 0:
                        pool.pop(pool_i)

                # --- Reconciliacion 3 niveles por grasper (paper Sec. III-C) ---
                new_track_to_grasper = {}

                for g_idx, det in enumerate(dets_tfm):
                    id_ext1 = assigned.get((g_idx, 0), -1)
                    id_ext2 = assigned.get((g_idx, 1), -1)

                    logical_id = self._reconcile_3level(
                        id_ext1, id_ext2, g_idx,
                        track_to_grasper, grasper_logical_id,
                    )

                    # Actualizar memoria solo para extremos confirmados
                    # como pertenecientes a este grasper (evita contaminar
                    # la memoria en casos de conflicto Nivel 3).
                    g_of_1 = track_to_grasper.get(id_ext1, (-1,))[0] if id_ext1 != -1 else -1
                    g_of_2 = track_to_grasper.get(id_ext2, (-1,))[0] if id_ext2 != -1 else -1
                    prev_logical = grasper_logical_id.get(g_idx, -1)

                    if id_ext1 != -1 and (g_of_1 == g_idx or g_of_1 == -1 or prev_logical == -1):
                        new_track_to_grasper[id_ext1] = (g_idx, 0)
                    if id_ext2 != -1 and (g_of_2 == g_idx or g_of_2 == -1 or prev_logical == -1):
                        new_track_to_grasper[id_ext2] = (g_idx, 1)

                    grasper_logical_id[g_idx] = logical_id

                    entry = {**det, 'track_id': logical_id}
                    bb1, bb2 = tfm_bboxes_by_gidx[g_idx]
                    tip1 = [float((bb1[0] + bb1[2]) / 2), float((bb1[1] + bb1[3]) / 2)]
                    tip2 = [float((bb2[0] + bb2[2]) / 2), float((bb2[1] + bb2[3]) / 2)]
                    entry['tips'] = [tip1, tip2]
                    tracked_dets.append(entry)

                track_to_grasper = new_track_to_grasper

            tracked_frames.append({'frame_idx': frame_idx, 'detections': tracked_dets})
            logger.debug('Frame %d: %d tracks', frame_idx, len(tracked_dets))

        return {
            'video'         : stable['video'],
            'imgsz'         : stable['imgsz'],
            'static_objects': stable['static_objects'],
            'frames'        : tracked_frames,
        }

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info('JSON tracked guardado en: %s', output_path)

    # ------------------------------------------------------------------
    # internos
    # ------------------------------------------------------------------

    def _reconcile_3level(self, id_ext1, id_ext2, g_idx,
                          track_to_grasper, grasper_logical_id):
        """
        Reconciliacion de 3 niveles (paper Sec. III-C).

        Nivel 1 — ambos extremos reconocidos como de este grasper:
            maxima confianza, se mantiene el ID logico.
        Nivel 2 — solo un extremo reconocido:
            el extremo no ambiguo preserva el ID logico.
        Nivel 3 — conflicto o grasper nuevo:
            bootstrap en frame 1, o se mantiene ID logico previo
            usando el extremo no conflictivo como ancla.
        """
        prev_logical = grasper_logical_id.get(g_idx, -1)

        g_of_1 = track_to_grasper.get(id_ext1, (-1,))[0] if id_ext1 != -1 else -1
        g_of_2 = track_to_grasper.get(id_ext2, (-1,))[0] if id_ext2 != -1 else -1

        # Nivel 1: ambos extremos reconocidos como de este grasper
        if g_of_1 == g_idx and g_of_2 == g_idx:
            return prev_logical

        # Nivel 2: uno de los extremos reconocido como de este grasper
        if g_of_1 == g_idx or g_of_2 == g_idx:
            return prev_logical

        # Nivel 3: ninguno matchea — conflicto o frame 1 (bootstrap)
        if prev_logical == -1:
            # Frame 1: no hay historial, usamos ext1 como ID logico del grasper
            return id_ext1 if id_ext1 != -1 else id_ext2

        # Frames siguientes: al menos un extremo es nuevo (track recien creado)
        # — usamos el ID logico previo para mantener identidad del grasper.
        # Si ambos son de otro grasper (cruce severo), tambien mantenemos
        # el ID previo; el proximo frame tendra mas informacion.
        return prev_logical

    def _nearest_from_pool(self, cx, cy, pool):
        """
        Busca en el pool el track mas cercano a (cx, cy).
        Devuelve (pool_index, track_id). Devuelve (-1, -1) si el pool esta vacio.
        El llamador debe hacer pool.pop(pool_index) para consumir el track.
        """
        if not pool:
            return -1, -1
        best_i    = -1
        best_id   = -1
        best_dist = float('inf')
        for i, t in enumerate(pool):
            tcx = (t[0] + t[2]) / 2
            tcy = (t[1] + t[3]) / 2
            d   = (tcx - cx) ** 2 + (tcy - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_i    = i
                best_id   = int(t[4])
        return best_i, best_id

    def _nearest_track_id(self, cx, cy, tracks):
        """Devuelve el track_id mas cercano a (cx, cy). Para rings."""
        if not tracks:
            return -1
        best_id   = -1
        best_dist = float('inf')
        for t in tracks:
            tcx = (t[0] + t[2]) / 2
            tcy = (t[1] + t[3]) / 2
            d   = (tcx - cx) ** 2 + (tcy - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_id   = int(t[4])
        return best_id

    def _compute_centroid(self, mask_polygon, bbox):
        if mask_polygon:
            pts = np.array(mask_polygon)
            return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def _mini_bbox_centroid(self, mask_polygon, bbox):
        """Mini-bbox centrado en el centroide real de la mascara (ring)."""
        if mask_polygon:
            pts = np.array(mask_polygon)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
        else:
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
        half_w = ((x_max - x_min) * self.fraction) / 2
        half_h = ((y_max - y_min) * self.fraction) / 2
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    def _mini_bboxes_two_point(self, mask_polygon, bbox):
        """
        Dos mini-bboxes en los extremos del eje dominante del grasper (TFM).
        Devuelve (bbox_extremo_1, bbox_extremo_2).
        """
        if mask_polygon:
            pts = np.array(mask_polygon, dtype=np.float32)
        else:
            x1, y1, x2, y2 = bbox
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()

        if w >= h:
            p1 = pts[np.argmin(pts[:, 0])]   # extremo izquierdo
            p2 = pts[np.argmax(pts[:, 0])]   # extremo derecho
        else:
            p1 = pts[np.argmin(pts[:, 1])]   # extremo superior
            p2 = pts[np.argmax(pts[:, 1])]   # extremo inferior

        def _bbox(p):
            cx, cy = float(p[0]), float(p[1])
            hw = (w * self.fraction) / 2
            hh = (h * self.fraction) / 2
            return [cx - hw, cy - hh, cx + hw, cy + hh]

        return _bbox(p1), _bbox(p2)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    ROOT         = Path(__file__).resolve().parent.parent
    STABLE_PATH  = ROOT / 'outputs' / 'stable'  / 'Video corto_stable.json'
    TRACKED_PATH = ROOT / 'outputs' / 'tracked' / 'Video corto_tracked.json'

    tracker = Tracker()
    data    = tracker.process(STABLE_PATH)
    tracker.save(data, TRACKED_PATH)


if __name__ == '__main__':
    main()
