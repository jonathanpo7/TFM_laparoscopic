import json
import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detector:
    """
    Detector

    Ejecuta inferencia cruda (sin tracking) de un modelo YOLO de segmentación
    sobre un video completo, frame a frame, y devuelve/guarda las detecciones
    (clase, confianza, bbox, polígono de máscara) en una estructura serializable
    a JSON.

    Parametros:
        model_path (str | Path): ruta al archivo de pesos del modelo (ej. 'projects/model/pruebam1280.pt').
        imgsz (int): resolución de inferencia, debe coincidir con la usada en entrenamiento (default 1280).

    Ejemplo:
        detector = Detector(model_path='projects/model/pruebam1280.pt', imgsz=1280)
        data = detector.run(video_path='ruta/al/video.mp4')
        detector.save(data, output_path='projects/outputs/raw/video1_raw.json')
    """

    def __init__(self, model_path, imgsz=1280):
        self.model = YOLO(model_path)
        self.imgsz = imgsz

    def run(self, video_path):
        video_path = Path(video_path)
        frames = []

        results_generator = self.model.predict(
            source=str(video_path), stream=True, imgsz=self.imgsz
        )

        for frame_idx, result in enumerate(results_generator):
            detections = []
            boxes = result.boxes
            masks = result.masks

            if boxes is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    detection = {
                        'class_id': class_id,
                        'class_name': self.model.names[class_id],
                        'confidence': float(boxes.conf[i]),
                        'bbox': boxes.xyxy[i].tolist(),
                        'mask_polygon': masks.xy[i].tolist() if masks is not None else [],
                    }
                    detections.append(detection)

            frames.append({'frame_idx': frame_idx, 'detections': detections})
            logger.debug(f'Frame {frame_idx}: {len(detections)} detecciones')

        return {
            'video': video_path.name,
            'imgsz': self.imgsz,
            'frames': frames,
        }

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        logger.info(f'Inferencia cruda guardada en: {output_path}')
