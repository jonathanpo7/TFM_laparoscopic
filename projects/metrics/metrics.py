import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class Metrics:
    """
    Calcula métricas del ejercicio Peg Transfer a partir del states JSON.

    Métodos disponibles (cada uno retorna un fragmento del resultado):
      arrival_at_valid_dest  — si cada ring llegó a un peg destino válido
      transit_time           — frames y segundos que tardó cada ring en moverse

    Uso:
        m = Metrics()
        data = m.compute(states_path, video_path)
        m.save(data, output_path)
    """

    def compute(self, states_path, video_path=None, fps=None, tracked_path=None):
        with open(Path(states_path)) as f:
            states = json.load(f)

        tracked = None
        if tracked_path is not None:
            with open(Path(tracked_path)) as f:
                tracked = json.load(f)

        if fps is None and video_path is not None:
            fps = self._read_fps(video_path)
        if fps is None:
            fps = 30.0
            logger.warning('FPS no disponible, usando %.1f por defecto', fps)

        logger.info('FPS: %.2f', fps)

        arrival  = self._arrival_at_valid_dest(states)
        transit  = self._transit_time(states, arrival, fps)
        economy  = self._movement_economy(states, transit, arrival)

        rings_completed = sum(1 for v in arrival.values() if v['arrived'])
        rings_total     = len(arrival)

        # Tiempo total del ejercicio: desde la primera salida hasta la última llegada
        departures = [transit[r]['departure_frame'] for r in transit if transit[r]['departure_frame'] is not None]
        arrivals   = [transit[r]['arrival_frame']   for r in transit if transit[r]['arrival_frame']   is not None]
        first_departure = min(departures) if departures else None
        last_arrival    = max(arrivals)   if arrivals   else None
        total_frames    = (last_arrival - first_departure) if (first_departure is not None and last_arrival is not None) else None
        total_time_s    = round(total_frames / fps, 2) if total_frames is not None else None

        rings_out = {}
        for rid in sorted(arrival, key=int):
            a = arrival[rid]
            t = transit[rid]
            e = economy.get(rid, {})
            rings_out[rid] = {
                'home_peg':              states['physical_rings'][rid]['home_peg'],
                'arrived_at_valid_dest': a['arrived'],
                'dest_peg':              a['dest_peg'],
                'arrival_window':        a['arrival_window'],
                'departure_frame':       t['departure_frame'],
                'arrival_frame':         t['arrival_frame'],
                'transit_frames':        t['transit_frames'],
                'transit_time_s':        t['transit_time_s'],
                'direct_dist_px':        e.get('direct_dist_px'),
                'accumulated_dist_px':   e.get('accumulated_dist_px'),
                'economy_ratio':         e.get('economy_ratio'),
            }

        result = {
            'video':   states['video'],
            'fps':     fps,
            'rings':   rings_out,
            'summary': {
                'rings_completed':       rings_completed,
                'rings_total':           rings_total,
                'completion_rate':       round(rings_completed / rings_total, 3),
                'first_departure_frame': first_departure,
                'last_arrival_frame':    last_arrival,
                'total_exercise_frames': total_frames,
                'total_exercise_time_s': total_time_s,
            },
        }

        self._log_summary(result)
        return result

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info('Métricas guardadas: %s', output_path)

    # ------------------------------------------------------------------
    # métricas individuales
    # ------------------------------------------------------------------

    def _arrival_at_valid_dest(self, states):
        """Determina si cada ring llegó a un peg destino válido."""
        dest_pegs = set(states['dest_pegs'])
        result    = {}

        for rid in states['physical_rings']:
            arrived_peg    = None
            arrival_window = None

            for wi, w in enumerate(states['windows']):
                fact = w['facts'].get(rid, {})
                if fact.get('arrived_peg') is not None:
                    arrived_peg    = fact['arrived_peg']
                    arrival_window = wi
                    break

            result[rid] = {
                'arrived':        arrived_peg is not None and arrived_peg in dest_pegs,
                'dest_peg':       arrived_peg,
                'arrival_window': arrival_window,
            }

        return result

    def _transit_time(self, states, arrival_info, fps):
        """
        Calcula el tiempo de tránsito por ring.

        departure_frame: primer frame de la primera ventana donde
                         near_peg_id != home_peg (ring ya no está en origen).
        arrival_frame:   primer frame de la ventana de confirmación de llegada.
        """
        phy   = states['physical_rings']
        ws    = states['window_size']
        result = {}

        for rid in phy:
            home_peg = phy[rid]['home_peg']
            a        = arrival_info[rid]

            # Frame de llegada
            arrival_frame = None
            if a['arrival_window'] is not None:
                arrival_frame = states['windows'][a['arrival_window']]['window'][0]

            # Frame de salida: última ventana donde near_peg_id == home_peg,
            # la siguiente ventana es cuando el ring se va.
            departure_frame = None
            last_home_window = None
            for wi, w in enumerate(states['windows']):
                fact = w['facts'].get(rid, {})
                if fact.get('near_peg_id') == home_peg and fact.get('detected', False):
                    last_home_window = wi

            if last_home_window is not None:
                # Departure = inicio de la ventana siguiente a la última en home.
                # Solo contamos ventanas donde detected=True (excluye carry-forward):
                # si el ring ya no se veía pero el carry-forward mantuvo near_peg_id=home,
                # ese no es un momento real en casa.
                next_wi = last_home_window + 1
                if next_wi < len(states['windows']):
                    departure_frame = states['windows'][next_wi]['window'][0]

            transit_frames = None
            transit_time_s = None
            if departure_frame is not None and arrival_frame is not None:
                transit_frames = arrival_frame - departure_frame
                transit_time_s = round(transit_frames / fps, 2)

            result[rid] = {
                'departure_frame': departure_frame,
                'arrival_frame':   arrival_frame,
                'transit_frames':  transit_frames,
                'transit_time_s':  transit_time_s,
            }

        return result

    def _movement_economy(self, states, transit_info, arrival_info):
        """
        Economía del movimiento usando los centroides por ventana del state_builder.

        El centroide de cada ventana ya está correctamente asignado al ring físico,
        por lo que la trayectoria acumulada es la suma de distancias entre centroides
        consecutivos durante el tránsito (ventana a ventana).

        direct_dist:      distancia euclidiana peg origen → peg destino
        accumulated_dist: suma de distancias entre centroides de ventanas consecutivas
        economy_ratio:    direct_dist / accumulated_dist
                          (1.0 = camino recto; <1.0 = más desvíos)
        """
        pegs    = states.get('pegs', {})
        windows = states['windows']
        ws      = states['window_size']
        phy     = states['physical_rings']

        if not pegs:
            logger.warning('States JSON no tiene pegs — re-corre state_builder para generarlos')
            return {}

        result = {}
        for rid in phy:
            home_peg = phy[rid]['home_peg']
            a        = arrival_info[rid]
            t        = transit_info[rid]
            dest_peg = a['dest_peg']
            arr_w    = a['arrival_window']
            dep_f    = t['departure_frame']

            # Distancia directa peg origen → peg destino
            direct_dist = (self._dist(pegs[str(home_peg)], pegs[str(dest_peg)])
                           if dest_peg is not None
                              and str(home_peg) in pegs
                              and str(dest_peg) in pegs
                           else None)

            # Ventana de inicio del tránsito
            dep_w = dep_f // ws if dep_f is not None else None

            if dep_w is not None and arr_w is not None and direct_dist is not None:
                # Centroides ventana a ventana (solo las que tienen centroide válido)
                centroids = []
                for wi in range(dep_w, arr_w + 1):
                    if wi >= len(windows):
                        break
                    c = windows[wi]['facts'].get(rid, {}).get('centroid')
                    if c is not None:
                        centroids.append(c)

                if len(centroids) >= 2:
                    accumulated   = sum(self._dist(centroids[i], centroids[i + 1])
                                        for i in range(len(centroids) - 1))
                    economy_ratio = round(direct_dist / accumulated, 3) if accumulated > 0 else None
                else:
                    accumulated   = None
                    economy_ratio = None
            else:
                accumulated   = None
                economy_ratio = None

            result[rid] = {
                'direct_dist_px':      round(direct_dist, 1) if direct_dist is not None else None,
                'accumulated_dist_px': round(accumulated, 1) if accumulated is not None else None,
                'economy_ratio':       economy_ratio,
            }

        return result

    # ------------------------------------------------------------------
    # utilidades
    # ------------------------------------------------------------------

    def _dist(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _read_fps(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            logger.warning('No se pudo leer FPS del video, usando 30.0')
            fps = 30.0
        return fps

    def _log_summary(self, data):
        s = data['summary']
        logger.info('=== MÉTRICAS ===')
        logger.info('Rings completados: %d / %d  (%.0f%%)',
                    s['rings_completed'], s['rings_total'],
                    s['completion_rate'] * 100)
        if s['total_exercise_time_s'] is not None:
            logger.info('Tiempo total ejercicio: %.2f s (%d frames)',
                        s['total_exercise_time_s'], s['total_exercise_frames'])
        logger.info('--- por ring ---')
        for rid, r in data['rings'].items():
            status = 'OK' if r['arrived_at_valid_dest'] else 'NO LLEGÓ'
            if r['transit_time_s'] is not None:
                logger.info('  Ring %s → peg %s  [%s]  tránsito: %.2f s (%d frames)',
                            rid, r['dest_peg'], status,
                            r['transit_time_s'], r['transit_frames'])
            else:
                logger.info('  Ring %s → %s  [%s]', rid, r['dest_peg'], status)
            if r.get('economy_ratio') is not None:
                logger.info('    economía: %.3f  (directo=%.0f px  acumulado=%.0f px)',
                            r['economy_ratio'],
                            r['direct_dist_px'],
                            r['accumulated_dist_px'])


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    ROOT         = Path(__file__).resolve().parent.parent
    STATES_PATH  = ROOT / 'outputs' / 'states'  / 'Video corto_states.json'
    TRACKED_PATH = ROOT / 'outputs' / 'tracked' / 'Video corto_tracked.json'
    VIDEO_PATH   = Path(r"C:\Users\Jonathan Piedrahita\Desktop\UAO-Inhealth\Script\videos_pruebas\Video corto.mp4")
    OUTPUT_PATH  = ROOT / 'outputs' / 'metrics' / 'Video corto_metrics.json'

    m    = Metrics()
    data = m.compute(STATES_PATH, video_path=VIDEO_PATH, tracked_path=TRACKED_PATH)
    m.save(data, OUTPUT_PATH)


if __name__ == '__main__':
    main()
