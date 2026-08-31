import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.detector   import Detector
from inference.stabilizer import Stabilizer
from tracking.tracker     import Tracker
from states.state_builder import StateBuilder
from metrics.metrics      import Metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / 'model' / 'xl1280-1.pt'


def run_pipeline(video_path, output_path, confianza=0.4):
    video_path  = Path(video_path)
    output_path = Path(output_path)
    sesion_id   = video_path.stem

    logger.info('=== PIPELINE: %s ===', sesion_id)

    logger.info('[1/4] Detección e inferencia...')
    detector = Detector(model_path=MODEL_PATH, imgsz=1280)
    raw      = detector.run(video_path, confianza=confianza)

    logger.info('[2/4] Estabilización de objetos estáticos...')
    stabilizer = Stabilizer()
    stable     = stabilizer.process(raw)

    logger.info('[3/4] Tracking (ByteTrack)...')
    tracker = Tracker()
    tracked = tracker.process(stable)

    ROOT         = Path(__file__).resolve().parent.parent
    TRACKED_PATH = ROOT / 'outputs' / 'tracked' / f'{sesion_id}_tracked.json'
    tracker.save(tracked, TRACKED_PATH)

    logger.info('[4/4] States y métricas...')
    builder = StateBuilder()
    states  = builder.build(tracked)

    m       = Metrics()
    result  = m.compute(states, fps=raw.get('fps', 30.0))
    result['sesion_id'] = sesion_id

    m.save(result, output_path)
    logger.info('=== FIN: %s ===', sesion_id)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline Peg Transfer — TFM')
    parser.add_argument('video',   help='Ruta al video (.mp4)')
    parser.add_argument('--output', '-o', default=None,
                        help='Ruta del JSON de salida (default: outputs/metrics/<stem>_metrics.json)')
    parser.add_argument('--conf',  type=float, default=0.4,
                        help='Umbral de confianza del detector (default: 0.4)')
    args = parser.parse_args()

    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        ROOT        = Path(__file__).resolve().parent.parent
        output_path = ROOT / 'outputs' / 'metrics' / f'{video_path.stem}_metrics.json'

    run_pipeline(video_path, output_path, confianza=args.conf)
