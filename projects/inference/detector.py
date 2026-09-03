import sys
import json
import logging
from pathlib import Path
import torch
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logger = logging.getLogger(__name__)

STATIC_CLASSES = {'platform', 'peg'}


class Detector:
    """
    Detector

    Ejecuta inferencia cruda (sin tracking ni filtrado por clase) de un modelo
    YOLO de segmentación sobre un video completo y serializa las detecciones a JSON.

    Las clases estáticas (peg, platform) solo se registran durante los primeros
    `stabilize_max_frames` frames; el Stabilizer las procesará después con K-means.
    Las clases dinámicas (ring, TFM) se registran en todos los frames sin filtrar.

    Parametros:
        model_path (str | Path): ruta al archivo de pesos del modelo.
        imgsz (int): resolución de inferencia (debe coincidir con entrenamiento).

    Ejemplo:
        detector = Detector(model_path='projects/model/pruebam1280.pt', imgsz=1280)
        data = detector.run(video_path='video.mp4', confianza=0.4)
        detector.save(data, output_path='projects/outputs/raw/video_raw.json')
    """

    def __init__(self, model_path, imgsz=1280):
        self.model = YOLO(model_path)
        self.model.model.end2end = False
        self.imgsz = imgsz

    def run(self, video_path, show=False, stabilize_max_frames=None, confianza=0.4, supresion=0.45):
        video_path = Path(video_path)
        dispositivo = 0 if torch.cuda.is_available() else 'cpu'

        # El fps se lee ANTES del recorrido: la ventana de grabación de objetos
        # estáticos está definida en segundos (config.INIT_MAX_SECONDS) y sus
        # frames dependen del fps real del video.
        fps = self._read_fps(video_path)
        if stabilize_max_frames is None:
            stabilize_max_frames = int(config.INIT_MAX_SECONDS * fps)
        logger.info('FPS %.2f — grabando estáticos hasta el frame %d (%.1f s)',
                    fps, stabilize_max_frames, config.INIT_MAX_SECONDS)

        frames = []
        results_generator = self.model.predict(
            source=str(video_path), verbose=False, stream=True,
            imgsz=self.imgsz, conf=confianza, iou=supresion,
            device=dispositivo,
        )

        if show:
            cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug', 800, 450)

        for frame_idx, result in enumerate(results_generator):
            detections = []
            boxes = result.boxes
            masks = result.masks

            if boxes is not None:
                for i in range(len(boxes)):
                    class_id   = int(boxes.cls[i])
                    class_name = self.model.names[class_id]

                    if class_name in STATIC_CLASSES and frame_idx >= stabilize_max_frames:
                        continue

                    detections.append({
                        'class_id'    : class_id,
                        'class_name'  : class_name,
                        'confidence'  : float(boxes.conf[i]),
                        'bbox'        : boxes.xyxy[i].tolist(),
                        'mask_polygon': masks.xy[i].tolist() if masks is not None else [],
                    })

            if show:
                cv2.imshow('debug', result.plot())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frames.append({'frame_idx': frame_idx, 'detections': detections})
            logger.debug('Frame %d: %d detecciones', frame_idx, len(detections))

        if show:
            cv2.destroyAllWindows()

        return {
            'video'                : video_path.name,
            'fps'                  : fps,
            'imgsz'               : self.imgsz,
            'conf'                : confianza,
            'stabilize_max_frames': stabilize_max_frames,
            'frames'              : frames,
        }

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info('Inferencia cruda guardada en: %s', output_path)

    def _read_fps(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            logger.warning('No se pudo leer FPS del video, usando 30.0')
            fps = 30.0
        return fps


def main():
    ROOT        = Path(__file__).resolve().parent.parent
    MODEL_PATH  = ROOT / 'model' / 'xl1280-1.pt'
    VIDEO_PATH  = Path(r"C:\Users\Jonathan Piedrahita\Desktop\Maestria en IA\Trabajo Fin de Master (TFM)\Datasets\Pruebas_personas\P07_FLS Task A_1\20230925165330 Trial3-1.mp4")
    OUTPUT_PATH = ROOT / 'outputs' / 'raw'
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    detector = Detector(model_path=MODEL_PATH, imgsz=1280)
    data = detector.run(video_path=VIDEO_PATH, show=True, confianza=0.55)
    detector.save(data, output_path=OUTPUT_PATH / f'{VIDEO_PATH.stem}_raw.json')


if __name__ == '__main__':
    main()
