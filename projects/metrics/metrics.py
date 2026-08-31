import sys
import json
import logging
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from states.state_builder import ARRIVAL_WINDOWS

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

    def compute(self, states_path_or_data, video_path=None, fps=None, tracked_path=None):
        if isinstance(states_path_or_data, dict):
            states = states_path_or_data
        else:
            with open(Path(states_path_or_data)) as f:
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

        arrival      = self._arrival_at_valid_dest(states)
        transit      = self._transit_time(states, arrival, fps)
        economy      = self._movement_economy(states, transit, arrival)
        bimanual     = self._bimanual_coordination(states, transit, arrival)
        drops_fuera  = self._ring_drop_errors(states, transit, arrival)
        drops_dentro = self._ring_loose_errors(states, transit, arrival)
        intentos     = self._intentos(drops_fuera, drops_dentro)

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
            e  = economy.get(rid, {})
            b  = bimanual.get(rid, {})
            df = drops_fuera.get(rid, {})
            dd = drops_dentro.get(rid, {})
            i  = intentos.get(rid, {})
            rings_out[rid] = {
                'home_peg':              states['physical_rings'][rid]['home_peg'],
                'arrived_at_valid_dest': a['arrived'],
                'dest_peg':              a['dest_peg'],
                'arrival_window':        a['arrival_window'],
                'perdido':               a['perdido'],
                'departure_frame':       t['departure_frame'],
                'arrival_frame':         t['arrival_frame'],
                'transit_frames':        t['transit_frames'],
                'transit_time_s':        t['transit_time_s'],
                'direct_dist_px':        e.get('direct_dist_px'),
                'accumulated_dist_px':   e.get('accumulated_dist_px'),
                'economy_ratio':         e.get('economy_ratio'),
                'bimanual':              b,
                'drop_errors':           df,
                'loose_errors':          dd,
                'intentos':              i,
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
        dest_pegs  = set(states['dest_pegs'])
        lost_rings = set(str(rid) for rid in states.get('lost_rings', []))
        result     = {}

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
                'perdido':        rid in lost_rings,
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
            # La búsqueda se acota a arrival_window para evitar que detecciones
            # fantasma tardías (después de la llegada) produzcan tiempos negativos.
            departure_frame = None
            last_home_window = None
            search_limit = a['arrival_window'] if a['arrival_window'] is not None else len(states['windows'])
            for wi, w in enumerate(states['windows']):
                if wi >= search_limit:
                    break
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
                    economy_ratio = round(min(direct_dist / accumulated, 1.0), 3) if accumulated > 0 else None
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

    def _ring_drop_errors(self, states, transit_info, arrival_info, drop_windows=3):
        """
        Detecta caídas de ring fuera de la plataforma DURANTE EL TRÁNSITO.

        Solo evalúa ventanas de tránsito (dep_w → arr_w) — un ring en su peg
        de origen no puede caerse, lo que elimina falsos positivos por pegs del
        borde que quedan fuera del área detectada.

        Usa el bbox estable de la plataforma (más tolerante que el polígono,
        compensa el ángulo de cámara que hace salir pegs superiores del contorno).

        Una caída se confirma con ≥ drop_windows ventanas consecutivas con
        centroide válido fuera del bbox.
        """
        platform = states.get('platform')
        if platform is None:
            logger.warning('Sin datos de plataforma — omitiendo detección de caídas')
            return {rid: {'drops': [], 'n_drops': 0, 'error': 'sin plataforma'}
                    for rid in states['physical_rings']}

        bbox = platform.get('bbox', [])
        if len(bbox) != 4:
            logger.warning('Bbox de plataforma no válido: %s', bbox)
            return {rid: {'drops': [], 'n_drops': 0, 'error': 'bbox invalido'}
                    for rid in states['physical_rings']}

        x1, y1, x2, y2 = bbox

        def in_platform(pt):
            return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

        windows = states['windows']
        ws      = states['window_size']
        result  = {}

        for rid in states['physical_rings']:
            t     = transit_info[rid]
            a     = arrival_info[rid]
            dep_f = t['departure_frame']
            arr_w = a['arrival_window']
            dep_w = dep_f // ws if dep_f is not None else None

            if dep_w is None or arr_w is None:
                result[rid] = {'drops': [], 'n_drops': 0}
                continue

            drops       = []
            outside_run = 0
            run_start   = None

            for wi in range(dep_w, min(arr_w + 1, len(windows))):
                c = windows[wi]['facts'].get(rid, {}).get('centroid')
                if c is None:
                    continue

                if not in_platform(c):
                    if outside_run == 0:
                        run_start = wi
                    outside_run += 1
                else:
                    if outside_run >= drop_windows:
                        drops.append({
                            'start_window': run_start,
                            'end_window':   wi - 1,
                            'n_windows':    outside_run,
                        })
                    outside_run = 0
                    run_start   = None

            if outside_run >= drop_windows:
                drops.append({
                    'start_window': run_start,
                    'end_window':   arr_w,
                    'n_windows':    outside_run,
                })

            result[rid] = {'drops': drops, 'n_drops': len(drops)}

        return result

    def _ring_loose_errors(self, states, transit_info, arrival_info, loose_windows=3):
        """
        Detecta caídas DENTRO de la plataforma (Métrica 3 del diseño original,
        ver memoria project_metrics_etapa_c): el ring queda suelto — sin
        ninguna pinza cerca y sin estar sobre ningún peg — en vez de fuera del
        área de trabajo (eso lo cubre `_ring_drop_errors`, la otra mitad de la
        Métrica 3: "salida del campo").

        Condición por ventana:
          detected      True  — se ve el ring
          near_peg_id   None  — ni sobre un peg ni claramente cerca de uno
                                solo. Cubre tanto "lejos de todo" como "justo
                                en medio de dos pegs": ambos casos ya colapsan
                                a None en state_builder vía RING_PEG_THRESH
                                (distancia) y RATIO_THRESH (ambigüedad entre
                                el peg más cercano y el segundo, ≥0.85 →
                                ratio_ok=False → near_peg=None).
          graspers_any  []    — ninguna pinza estuvo cerca en toda la ventana
          centroide           dentro del bbox de plataforma

        Se confirma con ≥ loose_windows ventanas consecutivas — mismo criterio
        de racha que `_ring_drop_errors`, para no confundir un instante de
        detección inestable con una caída real.
        """
        platform = states.get('platform')
        if platform is None:
            logger.warning('Sin datos de plataforma — omitiendo detección de caídas sueltas')
            return {rid: {'drops': [], 'n_drops': 0, 'error': 'sin plataforma'}
                    for rid in states['physical_rings']}

        bbox = platform.get('bbox', [])
        if len(bbox) != 4:
            logger.warning('Bbox de plataforma no válido: %s', bbox)
            return {rid: {'drops': [], 'n_drops': 0, 'error': 'bbox invalido'}
                    for rid in states['physical_rings']}

        x1, y1, x2, y2 = bbox

        def in_platform(pt):
            return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

        windows = states['windows']
        ws      = states['window_size']
        result  = {}

        for rid in states['physical_rings']:
            t     = transit_info[rid]
            a     = arrival_info[rid]
            dep_f = t['departure_frame']
            arr_w = a['arrival_window']
            dep_w = dep_f // ws if dep_f is not None else None

            if dep_w is None or arr_w is None:
                result[rid] = {'drops': [], 'n_drops': 0}
                continue

            drops     = []
            loose_run = 0
            run_start = None

            for wi in range(dep_w, min(arr_w + 1, len(windows))):
                f = windows[wi]['facts'].get(rid, {})
                c = f.get('centroid')
                suelto = (f.get('detected', False)
                          and f.get('near_peg_id') is None
                          and not f.get('graspers_any')
                          and c is not None
                          and in_platform(c))

                if suelto:
                    if loose_run == 0:
                        run_start = wi
                    loose_run += 1
                else:
                    if loose_run >= loose_windows:
                        drops.append({
                            'start_window': run_start,
                            'end_window':   wi - 1,
                            'n_windows':    loose_run,
                        })
                    loose_run = 0
                    run_start = None

            if loose_run >= loose_windows:
                drops.append({
                    'start_window': run_start,
                    'end_window':   arr_w,
                    'n_windows':    loose_run,
                })

            result[rid] = {'drops': drops, 'n_drops': len(drops)}

        return result

    def _bimanual_coordination(self, states, transit_info, arrival_info):
        """
        Mide la participación de cada pinza (grasper) por ring durante el tránsito.

        Por cada ventana de tránsito se revisa graspers_any (presencia en al menos
        1 frame) — más flexible que graspers_near que exige mayoría de frames.

        Resultado por ring:
          transit_windows   — total de ventanas de tránsito
          graspers          — {track_id: {windows_present, fraction}}
          bimanual          — True si ≥ 2 graspers distintos participaron
        """
        windows = states['windows']
        ws      = states['window_size']
        phy     = states['physical_rings']

        result = {}
        for rid in phy:
            t     = transit_info[rid]
            a     = arrival_info[rid]
            dep_f = t['departure_frame']
            arr_w = a['arrival_window']
            dep_w = dep_f // ws if dep_f is not None else None

            if dep_w is None or arr_w is None:
                result[rid] = {'transit_windows': None, 'graspers': {}, 'bimanual': None}
                continue

            transit_windows = arr_w - dep_w + 1
            grasper_counts  = {}

            for wi in range(dep_w, min(arr_w + 1, len(windows))):
                for gid in windows[wi]['facts'].get(rid, {}).get('graspers_any', []):
                    grasper_counts[gid] = grasper_counts.get(gid, 0) + 1

            graspers_out = {
                str(gid): {
                    'windows_present': count,
                    'fraction':        round(count / transit_windows, 3),
                }
                for gid, count in sorted(grasper_counts.items(), key=lambda x: -x[1])
            }

            result[rid] = {
                'transit_windows': transit_windows,
                'graspers':        graspers_out,
                'bimanual':        len(grasper_counts) >= 2,
            }

        return result

    def _intentos(self, drops_fuera, drops_dentro):
        """
        Total de errores por ring — los dos tipos de la Métrica 3 original
        (ver memoria project_metrics_etapa_c), nada más:

          caidas_fuera_campo:   ring salió del polígono de la plataforma
                                 (`_ring_drop_errors`)
          caidas_dentro_suelta: ring suelto dentro de la plataforma, sin
                                 pinza y sin peg (`_ring_loose_errors`)

        Se descarta el criterio anterior de "contactos_fallidos" (contacto
        con un peg destino equivocado) — no formaba parte del diseño
        original y se coló sin corresponder a ninguna de las dos métricas
        de error definidas.
        """
        result = {}
        for rid in set(drops_fuera) | set(drops_dentro):
            fuera  = drops_fuera.get(rid, {}).get('n_drops', 0)
            dentro = drops_dentro.get(rid, {}).get('n_drops', 0)
            result[rid] = {
                'caidas_fuera_campo':   fuera,
                'caidas_dentro_suelta': dentro,
                'total':                fuera + dentro,
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
            status = 'OK' if r['arrived_at_valid_dest'] else ('PERDIDO' if r['perdido'] else 'NO LLEGÓ')
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
            i = r.get('intentos', {})
            if i.get('total', 0) > 0:
                logger.info('    errores: %d total  (fuera del campo=%d  sueltas dentro=%d)',
                            i['total'], i['caidas_fuera_campo'], i['caidas_dentro_suelta'])
            elif i:
                logger.info('    errores: 0')
            d = r.get('drop_errors', {})
            if d.get('n_drops', 0) > 0:
                logger.info('    caídas fuera del campo: %d episodio(s)', d['n_drops'])
                for ep in d['drops']:
                    logger.info('      ventanas %d–%d (%d ventanas fuera)',
                                ep['start_window'], ep['end_window'], ep['n_windows'])
            dl = r.get('loose_errors', {})
            if dl.get('n_drops', 0) > 0:
                logger.info('    caídas sueltas dentro de plataforma: %d episodio(s)', dl['n_drops'])
                for ep in dl['drops']:
                    logger.info('      ventanas %d–%d (%d ventanas sueltas)',
                                ep['start_window'], ep['end_window'], ep['n_windows'])
            b = r.get('bimanual', {})
            if b.get('transit_windows'):
                tag = 'BIMANUAL' if b['bimanual'] else 'UNIMANUAL'
                logger.info('    coordinación [%s] — %d ventanas tránsito', tag, b['transit_windows'])
                for gid, g in b['graspers'].items():
                    logger.info('      grasper %s: %d/%d ventanas (%.0f%%)',
                                gid, g['windows_present'], b['transit_windows'],
                                g['fraction'] * 100)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    ROOT         = Path(__file__).resolve().parent.parent
    STATES_PATH  = ROOT / 'outputs' / 'states'  / '20230925162944 Trial1-2_states.json'
    TRACKED_PATH = ROOT / 'outputs' / 'tracked' / '20230925162944 Trial1-2_tracked.json'
    VIDEO_PATH   = Path(r"C:\Users\Jonathan Piedrahita\Desktop\Maestria en IA\Trabajo Fin de Master (TFM)\Datasets\Pruebas_personas\P07_FLS Task A_1\20230925162944 Trial1-2.mp4")
    OUTPUT_PATH  = ROOT / 'outputs' / 'metrics' / '20230925162944 Trial1-2_metrics.json'

    m    = Metrics()
    data = m.compute(STATES_PATH, video_path=VIDEO_PATH, tracked_path=TRACKED_PATH)
    m.save(data, OUTPUT_PATH)


if __name__ == '__main__':
    main()
