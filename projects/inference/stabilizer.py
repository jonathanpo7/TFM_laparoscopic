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

    La ventana de inicialización es ADAPTATIVA, no fija: se recorren los frames
    hasta `max_seconds` y se acumulan solo los "completos" (los 12 pegs y los 6
    rings detectados a la vez), cortando al juntar `complete_frames`. Si no se
    juntan, se emite un warning y el resultado marca `init_confiable: False`.

    Parametros:
        n_pegs (int): número de pegs esperados en escena (default 12).
        max_seconds (float): tope de búsqueda de frames completos, en segundos.
        complete_frames (int): cuántos frames completos bastan para estabilizar.
        class_conf (dict): umbral de confianza mínimo por clase.
        class_max (dict): máximo de instancias por clase por frame.
        outlier_sigma (float): factor σ para filtrar outliers dentro de cada cluster.

    Ejemplo:
        stabilizer = Stabilizer()
        data = stabilizer.process('projects/outputs/raw/video_raw.json')
        stabilizer.save(data, 'projects/outputs/stable/video_stable.json')
    """

    def __init__(self, n_pegs=config.N_PEGS, max_seconds=config.INIT_MAX_SECONDS,
                 complete_frames=config.INIT_COMPLETE_FRAMES,
                 class_conf=None, class_max=None, outlier_sigma=config.OUTLIER_SIGMA):
        self.n_pegs          = n_pegs
        self.max_seconds     = max_seconds
        self.complete_frames = complete_frames
        self.class_conf      = class_conf or config.CLASS_CONF
        self.class_max       = class_max  or config.CLASS_MAX
        self.outlier_sigma   = outlier_sigma

    def process(self, raw_path_or_data):
        if isinstance(raw_path_or_data, dict):
            raw = raw_path_or_data
        else:
            with open(Path(raw_path_or_data)) as f:
                raw = json.load(f)

        fps        = raw.get('fps') or 30.0
        max_frames = int(self.max_seconds * fps)

        # Frames completos: los que tienen los 12 pegs y los 6 rings a la vez.
        # Solo esos alimentan al K-means (ver docstring de la clase).
        init_idx = self._find_complete_frames(raw['frames'], max_frames)

        peg_accum      = []
        platform_accum = []
        clean_frames   = []

        for frame in raw['frames']:
            frame_idx    = frame['frame_idx']
            es_init      = frame_idx in init_idx
            dynamic_dets = []

            for det in frame['detections']:
                cn   = det['class_name']
                conf = det['confidence']

                if cn == 'peg':
                    if es_init and conf >= self.class_conf['peg']:
                        peg_accum.append(det)
                elif cn == 'platform':
                    if es_init and conf >= self.class_conf['platform']:
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
            'video'          : raw['video'],
            'fps'            : fps,
            'imgsz'          : raw['imgsz'],
            'init_frames'    : sorted(init_idx),
            'init_confiable' : len(init_idx) >= self.complete_frames,
            'static_objects' : static_objects,
            'frames'         : clean_frames,
        }

    def _find_complete_frames(self, frames, max_frames):
        """Selecciona los frames 'completos' para estabilizar objetos estáticos.

        Un frame es completo si tiene los N_PEGS pegs y los 6 rings detectados
        a la vez (por encima de su umbral de confianza). Se recorren los frames
        hasta `max_frames` y se corta al juntar `complete_frames` completos.

        Motivo: K-means con k=12 sobre frames donde solo hay 11 pegs visibles no
        falla — inventa la partición de un peg real en dos clusters, con errores
        medidos de hasta 279 px. Alimentándolo solo con frames completos ese modo
        de fallo desaparece (error < 1.2 px con 8 frames).

        Los rings solo se exigen VISIBLES, no posados en su peg: se aceptó el
        riesgo de videos que arrancan con un ring ya levantado.
        """
        elegidos = []
        for frame in frames[:max_frames]:
            n_pegs  = sum(1 for d in frame['detections']
                          if d['class_name'] == 'peg'
                          and d['confidence'] >= self.class_conf['peg'])
            n_rings = sum(1 for d in frame['detections']
                          if d['class_name'] == 'ring'
                          and d['confidence'] >= self.class_conf['ring'])
            if n_pegs == self.n_pegs and n_rings == self.class_max['ring']:
                elegidos.append(frame['frame_idx'])
                if len(elegidos) >= self.complete_frames:
                    break

        if len(elegidos) >= self.complete_frames:
            logger.info('Inicialización: %d frames completos (hasta el frame %d de %d explorados)',
                        len(elegidos), elegidos[-1], max_frames)
        else:
            logger.warning(
                'Inicialización NO confiable: solo %d frames completos de los %d requeridos '
                '(explorados %d frames = %.1f s). El video no tiene una ventana inicial con '
                'los %d pegs y %d rings visibles a la vez — las métricas de este video '
                'pueden degradarse.',
                len(elegidos), self.complete_frames, max_frames, self.max_seconds,
                self.n_pegs, self.class_max['ring'])
            if not elegidos:
                # Sin ningún frame completo: caer a la ventana fija para no quedar sin pegs.
                elegidos = [f['frame_idx'] for f in frames[:max_frames]]
                logger.warning('Sin frames completos — usando los %d frames iniciales como respaldo',
                               len(elegidos))

        return set(elegidos)

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

    def _compute_anclaje(self, det, pct=config.BASE_MASK_PCT):
        """Punto de apoyo del peg: centro del `pct`% inferior de su máscara.

        El peg se segmenta como un poste alto (90-175 px), así que su centroide
        cae a media altura — pero el ring descansa en la BASE. Medido contra la
        posición de reposo real de los rings:

            centroide de máscara : 58.4 px de distancia media (peor caso 66.8)
            base del bbox        : 18.0 px                    (peor caso 32.7)
            base de máscara (10%): 17.9 px                    (peor caso 25.2)

        Se usa una franja inferior en vez del punto más bajo porque un único
        vértice del polígono es ruidoso; promediar el 10% inferior conserva la
        precisión y recupera estabilidad.
        """
        polygon = det.get('mask_polygon', [])
        if not polygon:
            bbox = det['bbox']
            return [(bbox[0] + bbox[2]) / 2, bbox[3]]
        pts   = np.array(polygon)
        y_max = pts[:, 1].max()
        y_min = pts[:, 1].min()
        corte = y_max - (y_max - y_min) * pct / 100.0
        bajos = pts[pts[:, 1] >= corte]
        return [float(bajos[:, 0].mean()), float(bajos[:, 1].mean())]

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
        anclajes  = np.array([self._compute_anclaje(d)  for d in dets])
        best      = max(dets, key=lambda d: d['confidence'])
        return {
            'class_id'    : dets[0]['class_id'],
            'class_name'  : dets[0]['class_name'],
            'centroid'    : centroids.mean(axis=0).tolist(),
            # Punto de apoyo (base de la máscara) — es el que deben usar las
            # etapas siguientes para medir distancia ring↔peg. El centroide se
            # conserva porque es lo que agrupa el K-means.
            'anclaje'     : anclajes.mean(axis=0).tolist(),
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
    RAW_PATH    = ROOT / 'outputs' / 'raw'    / '20230925162944 Trial1-2_raw.json'
    STABLE_PATH = ROOT / 'outputs' / 'stable' / '20230925162944 Trial1-2_stable.json'

    stabilizer  = Stabilizer()
    data        = stabilizer.process(RAW_PATH)
    stabilizer.save(data, STABLE_PATH)


if __name__ == '__main__':
    main()
