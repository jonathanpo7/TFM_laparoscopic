import logging
from pathlib import Path

from inference.detector import Detector
from inference.stabilizer import Stabilizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_PATH  = Path(__file__).resolve().parent.parent / 'model' / 'pruebam1280.pt'
VIDEO_PATH  = Path('RUTA_AL_VIDEO_AQUI.mp4')
RAW_PATH    = Path(__file__).resolve().parent.parent / 'outputs' / 'raw'    / f'{VIDEO_PATH.stem}_raw.json'
STABLE_PATH = Path(__file__).resolve().parent.parent / 'outputs' / 'stable' / f'{VIDEO_PATH.stem}_stable.json'


def main():
    # Etapa A — inferencia cruda
    detector = Detector(model_path=MODEL_PATH, imgsz=1280)
    raw_data = detector.run(video_path=VIDEO_PATH, confianza=0.4)
    detector.save(raw_data, output_path=RAW_PATH)

    # Etapa B — estabilización de objetos estáticos + filtrado por clase
    stabilizer  = Stabilizer()
    stable_data = stabilizer.process(RAW_PATH)
    stabilizer.save(stable_data, output_path=STABLE_PATH)


if __name__ == '__main__':
    main()
