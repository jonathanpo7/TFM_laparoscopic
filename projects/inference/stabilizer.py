import sys
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logger = logging.getLogger(__name__)


class Stabilizer:
    """
    Stabilizer

    Consume el JSON crudo producido por Detector y genera un JSON limpio con:
      - static_objects: posición estable de cada peg (K-means k=12 + StandardScaler)
                        y de la platform (promedio directo).
      - frames: solo detecciones dinámicas (ring, TFM) filtradas por CLASS_CONF
                y acotadas por CLASS_MAX (top-N por confianza).

    Parametros:
        n_pegs (int): número de pegs esperados en escena (default 12).
        stabilize_frames (int): frames usados para estabilizar objetos estáticos.
        class_conf (dict): umbral de confianza mínimo por clase.
        class_max (dict): máximo de instancias por clase por frame.
        outlier_sigma (float): factor σ para filtrar outliers dentro de cada cluster.

    Ejemplo:
        stabilizer = Stabilizer()
        data = stabilizer.process('projects/outputs/raw/video_raw.json')
        stabilizer.save(data, 'projects/outputs/stable/video_stable.json')
    """

    def __init__(self, n_pegs=config.N_PEGS, stabilize_frames=config.STABILIZE_MAX_FRAMES,
                 class_conf=None, class_max=None, outlier_sigma=config.OUTLIER_SIGMA):
        self.n_pegs           = n_pegs
        self.stabilize_frames = stabilize_frames
        self.class_conf       = class_conf or config.CLASS_CONF
        self.class_max        = class_max  or config.CLASS_MAX
        self.outlier_sigma    = outlier_sigma

    def process(self, raw_path):
        raw_path = Path(raw_path)
        with open(raw_path) as f:
            raw = json.load(f)

        peg_accum      = []
        platform_accum = []
        clean_frames   = []

        for frame in raw['frames']:
            frame_idx    = frame['frame_idx']
            dynamic_dets = []

            for det in frame['detections']:
                cn   = det['class_name']
                conf = det['confidence']

                if cn == 'peg':
                    if frame_idx < self.stabilize_frames and conf >= self.class_conf['peg']:
                        peg_accum.append(det)
                elif cn == 'platform':
                    if frame_idx < self.stabilize_frames and conf >= self.class_conf['platform']:
                        platform_accum.append(det)
                elif conf >= self.class_conf.get(cn, 0.5):
                    dynamic_dets.append(det)

            # agrupar por clase y aplicar CLASS_MAX (top-N por confianza)
            by_class = {}
            for det in dynamic_dets:
                by_class.setdefault(det['class_name'], []).append(det)

            filtered = []
            for cn, dets in by_class.items():
                max_n = self.class_max.get(cn, len(dets))
                filtered.extend(
                    sorted(dets, key=lambda d: d['confidence'], reverse=True)[:max_n]
                )

            clean_frames.append({'frame_idx': frame_idx, 'detections': filtered})

        static_objects = self._stabilize(peg_accum, platform_accum)

        return {
            'video'         : raw['video'],
            'imgsz'         : raw['imgsz'],
            'static_objects': static_objects,
            'frames'        : clean_frames,
        }

    # ------------------------------------------------------------------
    # internos
    # ------------------------------------------------------------------

    def _compute_centroid(self, det):
        polygon = det.get('mask_polygon', [])
        if polygon:
            pts = np.array(polygon)
            return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        bbox = det['bbox']
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def _stabilize(self, peg_accum, platform_accum):
        result = {'pegs': [], 'platform': None}

        if platform_accum:
            result['platform'] = self._average_cluster(platform_accum)
            logger.info('Platform estabilizada con %d detecciones', len(platform_accum))

        if len(peg_accum) < self.n_pegs:
            logger.warning(
                'Insuficientes detecciones de peg para K-means: %d (se esperan >= %d)',
                len(peg_accum), self.n_pegs,
            )
            return result

        centroids = np.array([self._compute_centroid(d) for d in peg_accum])

        scaler           = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids)

        kmeans = KMeans(n_clusters=self.n_pegs, random_state=42, n_init=10)
        labels = kmeans.fit_predict(centroids_scaled)

        for cluster_id in range(self.n_pegs):
            cluster_dets = [d for d, lbl in zip(peg_accum, labels) if lbl == cluster_id]
            cluster_pts  = centroids_scaled[labels == cluster_id]
            center       = kmeans.cluster_centers_[cluster_id]

            dists  = np.linalg.norm(cluster_pts - center, axis=1)
            cutoff = dists.mean() + self.outlier_sigma * dists.std()
            clean_dets = [d for d, dist in zip(cluster_dets, dists) if dist <= cutoff]

            if not clean_dets:
                clean_dets = cluster_dets

            entry = self._average_cluster(clean_dets)
            entry['peg_id']       = cluster_id
            entry['n_detections'] = len(clean_dets)
            result['pegs'].append(entry)

        logger.info('Pegs estabilizados: %d/%d', len(result['pegs']), self.n_pegs)
        return result

    def _average_cluster(self, dets):
        bboxes    = np.array([d['bbox'] for d in dets])
        centroids = np.array([self._compute_centroid(d) for d in dets])
        best      = max(dets, key=lambda d: d['confidence'])
        return {
            'class_id'    : dets[0]['class_id'],
            'class_name'  : dets[0]['class_name'],
            'centroid'    : centroids.mean(axis=0).tolist(),
            'bbox'        : bboxes.mean(axis=0).tolist(),
            'mask_polygon': best['mask_polygon'],
            'confidence'  : float(np.mean([d['confidence'] for d in dets])),
        }

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info('JSON estabilizado guardado en: %s', output_path)


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    ROOT        = Path(__file__).resolve().parent.parent
    RAW_PATH    = ROOT / 'outputs' / 'raw'    / 'Video corto_raw.json'
    STABLE_PATH = ROOT / 'outputs' / 'stable' / 'Video corto_stable.json'

    stabilizer  = Stabilizer()
    data        = stabilizer.process(RAW_PATH)
    stabilizer.save(data, STABLE_PATH)


if __name__ == '__main__':
    main()
