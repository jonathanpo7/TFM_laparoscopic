import sys
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

RING_PEG_THRESH = 80    # px — ring "sobre peg" si dist < este valor
RATIO_THRESH    = 0.65  # ratio dist_nearest/dist_second — ring entre pegs si ratio ≥ este valor
RING_TIP_THRESH = 50    # px — ring "con pinza" si dist < este valor
PICKUP_THRESH   = 150   # px — punta de pinza cerca del peg del ring → pickup inferido
INIT_FRAMES     = 30    # frames para inicializar rings físicos
WINDOW_SIZE     = 10    # frames por ventana
ARRIVAL_WINDOWS = 3     # ventanas consecutivas para confirmar llegada a base


class StateBuilder:
    """
    Lee el tracked JSON y genera hechos espaciales por ventana de 10 frames.

    Por cada ventana y ring físico se registran tres hechos:
      detected      bool      — detectado en ≥50% de los frames de la ventana
      near_peg_id   int|None  — peg más cercano si dist < 120px; None si en tránsito
      graspers_near [int]     — IDs de pinzas presentes en mayoría de frames

    near_peg_id refleja la posición real del ring. La presencia de un grasper no
    anula near_peg_id: si el ring sigue sobre un peg mientras la pinza lo toca,
    near_peg_id permanece asignado.

    Asignación por exclusión (regla un-ring-a-la-vez):
      1. Rings con peg conocido se asignan por proximidad a su peg esperado.
      2. El sobrante va al único ring cuyo prev_peg es None (está en tránsito).
      3. Si hay 2+ rings en tránsito simultáneamente → ambigüedad, no asignar.
    """

    def build(self, tracked_path_or_data):
        if isinstance(tracked_path_or_data, dict):
            data = tracked_path_or_data
        else:
            with open(Path(tracked_path_or_data)) as f:
                data = json.load(f)

        pegs   = {p['peg_id']: p['centroid'] for p in data['static_objects']['pegs']}
        frames = data['frames']

        physical_rings = self._init_rings(frames[:INIT_FRAMES], pegs)
        source_pegs    = {r['home_peg'] for r in physical_rings.values()}
        dest_pegs      = sorted(pid for pid in pegs if pid not in source_pegs)

        logger.info('Rings físicos inicializados: %d', len(physical_rings))
        for rid, r in physical_rings.items():
            logger.info('  Ring %d → peg %s  (track_id inicial: %s)',
                        rid, r['home_peg'], r['init_track_id'])
        logger.info('Pegs de origen:  %s', sorted(source_pegs))
        logger.info('Pegs de destino: %s', dest_pegs)

        # Contexto por ring — persiste entre ventanas
        ctx = {
            rid: {
                'prev_peg':      r['home_peg'],
                'prev_graspers': [],
                'prev_detected': True,
                'arrival_buf':   {},   # {peg_id: n_ventanas_consecutivas_cerca}
                'arrived_peg':   None, # primer peg destino confirmado (para no re-loguear)
            }
            for rid, r in physical_rings.items()
        }

        windows         = []
        n_windows       = (len(frames) + WINDOW_SIZE - 1) // WINDOW_SIZE
        confirmed_dests = set()   # pegs destino ya confirmados — no se reasignan

        for wi in range(n_windows):
            f0       = wi * WINDOW_SIZE
            f1       = min(f0 + WINDOW_SIZE, len(frames))
            w_frames = frames[f0:f1]

            facts = self._process_window(w_frames, physical_rings, pegs, ctx)

            # Actualizar contexto y detectar llegadas
            for rid, f in facts.items():
                ctx[rid]['prev_peg']      = f['near_peg_id']
                ctx[rid]['prev_graspers'] = f['graspers_near']
                ctx[rid]['prev_detected'] = f['detected']

                pid = f['near_peg_id']
                if ctx[rid]['arrived_peg'] is not None:
                    # Ring ya confirmó — solo carry-forward, no toca confirmed_dests
                    f['arrived_peg'] = ctx[rid]['arrived_peg']
                    ctx[rid]['arrival_buf'] = {}
                elif pid in dest_pegs and pid not in confirmed_dests:
                    buf      = ctx[rid]['arrival_buf']
                    buf[pid] = buf.get(pid, 0) + 1
                    if buf[pid] >= ARRIVAL_WINDOWS:
                        f['arrived_peg'] = pid
                        confirmed_dests.add(pid)
                        ctx[rid]['arrived_peg'] = pid
                        logger.info('Ring %d llegó a peg destino %s (ventana %d, frame ~%d)',
                                    rid, pid, wi, f0)
                else:
                    ctx[rid]['arrival_buf'] = {}

            windows.append({
                'window': [f0, f1 - 1],
                'facts':  {str(k): v for k, v in facts.items()},
            })

        raw_platform = data['static_objects'].get('platform')
        if raw_platform is None:
            logger.warning('No se encontró plataforma en static_objects — métrica de caídas no disponible')
            platform = None
        else:
            # Solo bbox y centroide — mask_polygon no se usa en métricas
            platform = {
                'bbox':       raw_platform.get('bbox'),
                'centroid':   raw_platform.get('centroid'),
                'confidence': raw_platform.get('confidence'),
            }

        return {
            'video':          data['video'],
            'window_size':    WINDOW_SIZE,
            'init_frames':    INIT_FRAMES,
            'physical_rings': {str(k): v for k, v in physical_rings.items()},
            'source_pegs':    sorted(source_pegs),
            'dest_pegs':      dest_pegs,
            'pegs':           {str(pid): centroid for pid, centroid in pegs.items()},
            'platform':       platform,
            'windows':        windows,
        }

    def save(self, data, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info('States JSON guardado: %s', output_path)

    # ------------------------------------------------------------------
    # privados
    # ------------------------------------------------------------------

    def _init_rings(self, init_frames, pegs):
        """Detecta rings físicos en los primeros INIT_FRAMES frames.

        Cada track_id vota por el peg más cercano (solo distancia, sin ratio,
        porque al inicio los rings pueden estar apilados y el ratio sería demasiado
        estricto). Si el mejor peg ya está asignado a otro ring, se intenta el
        siguiente mejor peg del mismo track_id (fallback ordenado por votos).
        """
        votes = defaultdict(lambda: defaultdict(int))
        for frame in init_frames:
            for det in frame['detections']:
                if det['class_name'] != 'ring':
                    continue
                cx, cy      = det['centroid']
                pid, dst, _ = self._nearest_peg(cx, cy, pegs)
                if dst < RING_PEG_THRESH:
                    votes[det['track_id']][pid] += 1

        physical, used_pegs = {}, set()
        for rid, (tid, peg_votes) in enumerate(sorted(votes.items())):
            # Intentar pegs en orden de votos (mejor → peor)
            for best_peg in sorted(peg_votes, key=peg_votes.get, reverse=True):
                if best_peg not in used_pegs:
                    physical[rid] = {'home_peg': best_peg, 'init_track_id': tid}
                    used_pegs.add(best_peg)
                    break
        return physical

    def _process_window(self, w_frames, physical_rings, pegs, ctx):
        """Calcula hechos por ring para una ventana de WINDOW_SIZE frames."""
        per_frame = {rid: [] for rid in physical_rings}

        for frame in w_frames:
            ring_dets = [d for d in frame['detections'] if d['class_name'] == 'ring']
            tfm_dets  = [d for d in frame['detections'] if d['class_name'] == 'TFM']
            assign    = self._assign(ring_dets, physical_rings, pegs, ctx)

            for rid in physical_rings:
                det = assign.get(rid)
                if det is None:
                    # Si el ring tenía peg conocido, verificar si hay pinza cerca de ese peg.
                    # Punta de pinza dentro de PICKUP_THRESH del peg → ring fue tomado aunque
                    # no lo veamos — registrar como frame "con pinza" en vez de invisible.
                    inferred_grasper = None
                    expected_peg = ctx[rid]['prev_peg']
                    if expected_peg is not None:
                        peg_pos = pegs[expected_peg]
                        for tfm in tfm_dets:
                            for tip in tfm.get('tips', []):
                                if self._dist(tip, peg_pos) < PICKUP_THRESH:
                                    inferred_grasper = tfm['track_id']
                                    break
                            if inferred_grasper is not None:
                                break
                    # Condición de seguridad: un ring a la vez.
                    # Si ese grasper ya transporta otro ring, no puede iniciar otro pickup.
                    if inferred_grasper is not None:
                        claimed_by_other = any(
                            inferred_grasper in ctx[r]['prev_graspers']
                            for r in physical_rings if r != rid
                        )
                        if claimed_by_other:
                            inferred_grasper = None

                    if inferred_grasper is not None:
                        per_frame[rid].append({'near_peg': None, 'graspers': [inferred_grasper], 'centroid': None})
                    else:
                        per_frame[rid].append(None)
                    continue

                cx, cy             = det['centroid']
                pid, dst, second_d = self._nearest_peg(cx, cy, pegs)
                ratio_ok           = dst / second_d < RATIO_THRESH if second_d > 0 else True
                near_peg           = pid if dst < RING_PEG_THRESH and ratio_ok else None
                graspers = self._graspers_near(cx, cy, tfm_dets)

                # near_peg refleja posición real del ring.
                # Si hay grasper pero el ring sigue sobre un peg, near_peg se conserva.

                per_frame[rid].append({'near_peg': near_peg, 'graspers': graspers, 'centroid': [cx, cy]})

        return {rid: self._aggregate(per_frame[rid], ctx[rid])
                for rid in physical_rings}

    def _assign(self, ring_dets, physical_rings, pegs, ctx):
        """Asigna detecciones a rings físicos.

        Paso 1 — todos los rings buscan detección cerca de su peg esperado.
        Paso 2 — exclusión pura: si exactamente 1 ring quedó sin emparejar
                 Y exactamente 1 detección quedó sobrante → son el mismo objeto.
                 Con la regla de un-ring-a-la-vez esto es determinístico.
                 Si hay 2+ rings sin emparejar → ambigüedad, no asignar.
        """
        assign, used = {}, set()

        # Paso 1: cada ring busca detección cerca de su peg esperado.
        # Si ya confirmó llegada, se ancla al dest confirmado para evitar
        # que un switch de track_id lo desvíe de vuelta a su home peg.
        for rid, ring in physical_rings.items():
            if ctx[rid]['arrived_peg'] is not None:
                expected = ctx[rid]['arrived_peg']
            else:
                expected = ctx[rid]['prev_peg'] or ring['home_peg']
            if expected is None:
                continue
            peg_pos = pegs[expected]

            best_i, best_d = None, float('inf')
            for i, det in enumerate(ring_dets):
                if i in used:
                    continue
                d = self._dist(det['centroid'], peg_pos)
                if d < RING_PEG_THRESH and d < best_d:
                    best_d, best_i = d, i

            if best_i is not None:
                assign[rid] = ring_dets[best_i]
                used.add(best_i)

        # Paso 2: el sobrante va al ring que ya tiene pinza confirmada (en tránsito).
        # Esto es determinístico: la pinza que tomó el ring lo acompaña hasta el destino.
        leftover = [det for i, det in enumerate(ring_dets) if i not in used]
        if leftover:
            # En tránsito: sin peg conocido, o no detectado (carry-forward).
            in_transit = [rid for rid in physical_rings
                              if rid not in assign and
                              (ctx[rid]['prev_peg'] is None or not ctx[rid]['prev_detected'])]
            # Matching espacial: cada detección sobrante va al ring en tránsito
            # cuyo home_peg es geográficamente más cercano a esa detección.
            for det in leftover:
                if not in_transit:
                    break
                cx, cy = det['centroid']
                best_rid = min(in_transit, key=lambda r: self._dist(
                    [cx, cy], pegs[physical_rings[r]['home_peg']]))
                assign[best_rid] = det
                in_transit.remove(best_rid)

        return assign

    def _aggregate(self, signals, ctx):
        """Aplica moda sobre los frames de la ventana para obtener el hecho."""
        detected  = [s for s in signals if s is not None]
        rate      = len(detected) / max(len(signals), 1)

        if not detected:
            return {
                'detected':      False,
                'near_peg_id':   ctx['prev_peg'],
                'graspers_near': ctx['prev_graspers'],
                'graspers_any':  [],
                'centroid':      None,
            }

        # Moda del peg (incluye None si es el valor más frecuente)
        peg_mode = Counter(s['near_peg'] for s in detected).most_common(1)[0][0]

        # Conteo de apariciones por pinza en frames detectados
        g_counts      = Counter(g for s in detected for g in s['graspers'])
        threshold     = max(1, len(detected) // 2)
        graspers_near = [g for g, c in g_counts.items() if c >= threshold]  # mayoría
        graspers_any  = list(g_counts.keys())                                # cualquier frame

        # Centroide promedio de los frames donde el ring fue detectado
        cs = [s['centroid'] for s in detected if s.get('centroid') is not None]
        avg_centroid = [round(sum(x) / len(cs), 1) for x in zip(*cs)] if cs else None

        return {
            'detected':      rate >= 0.5,
            'near_peg_id':   peg_mode,
            'graspers_near': graspers_near,
            'graspers_any':  graspers_any,
            'centroid':      avg_centroid,
        }

    def _graspers_near(self, cx, cy, tfm_dets):
        near = []
        for tfm in tfm_dets:
            for tip in tfm.get('tips', []):
                if self._dist([cx, cy], tip) < RING_TIP_THRESH:
                    near.append(tfm['track_id'])
                    break
        return near

    def _nearest_peg(self, cx, cy, pegs):
        distances = sorted(
            (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5, pid)
            for pid, (px, py) in pegs.items()
        )
        best_d,   best_pid = distances[0] if distances else (float('inf'), None)
        second_d            = distances[1][0] if len(distances) > 1 else float('inf')
        return best_pid, best_d, second_d

    def _dist(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    ROOT         = Path(__file__).resolve().parent.parent
    TRACKED_PATH = ROOT / 'outputs' / 'tracked' / '20230911125148 Trial1-2_tracked.json'
    STATES_PATH  = ROOT / 'outputs' / 'states'  / '20230911125148 Trial1-2_states.json'

    builder = StateBuilder()
    data    = builder.build(TRACKED_PATH)
    builder.save(data, STATES_PATH)


if __name__ == '__main__':
    main()
