import sys
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# --- Umbrales espaciales (px) ---
# RING_PEG_THRESH responde solo a "¿cerca de cuál peg está?" — una pregunta que
# debe ser LAXA. El rigor para decidir una llegada NO vive aquí sino en dos
# reglas independientes de la cámara: ARRIVAL_WINDOWS ventanas consecutivas
# (descarta el sobrevuelo: el ring que pasa por encima resetea el contador) y
# la exclusividad de pegs destino (confirmed_dests).
#
# Calibrarlo estricto fue un error medido: la distancia de reposo NO es
# uniforme en el tablero — por la perspectiva oblicua, en la fila de origen el
# ring queda a ~18 px del anclaje pero en la de destino a ~49 px. Con 45 px
# solo el 25% de las llegadas a destino llegaba a registrar peg, así que las
# reglas temporales nunca tenían oportunidad de actuar.
#
# 60 px es el codo empírico (P01 / P07): cobertura destino 87% / 88% con solo
# 3.4% / 5.5% de frames ambiguos. A 70 px la cobertura sube 3 puntos pero la
# ambigüedad salta a 15-19% (empieza a caber más de un peg en el radio).
#
# Ajustado a 65 px (2026-08-31): en Trial1-1, R2 (ring0) se asienta genuinamente
# a 60.7-61.2 px de su peg destino — 0.7-1.2 px por encima del corte de 60,
# suficiente para que nunca acumule la racha de confirmación y su tránsito se
# infle +41s. Verificado que no es ambigüedad con un peg vecino (el más cercano
# a peg0 está a 134.9 px, muy por debajo del margen de riesgo).
RING_PEG_THRESH = 65    # px — ring "cerca de" un peg si dist al anclaje < este valor
# Mismo criterio que RING_PEG_THRESH: la pregunta espacial debe ser LAXA.
# Con 0.65 se exigía que el peg ganador estuviera 35% más cerca que el segundo,
# y en la fila de destino —donde la perspectiva junta los pegs— eso rechazaba
# colocaciones válidas: un ring a 49 px de su peg con el siguiente a 67 px
# (ratio 0.73) quedaba como "en ningún peg". Medido: subirlo de 0.65 a 0.75
# lleva P01 de 3/6 a 5/6 y P07 de 5/6 a 6/6, y de ahí en adelante es meseta.
RATIO_THRESH    = 0.85  # ratio dist_nearest/dist_second — ring entre pegs si ratio ≥ este valor
RING_TIP_THRESH = 50    # px — ring "con pinza" si dist < este valor
PICKUP_THRESH   = 150   # px — punta de pinza cerca del peg del ring → pickup inferido
# Salto máximo del viajero entre frames consecutivos cuando su track_id cambió
# (respaldo por continuidad). Medido: p99 = 28 px/frame, máximos legítimos ~50.
TRANSIT_MAX_JUMP = 100  # px

# --- Tiempos (segundos) ---
# El dataset mezcla 30 y 60 fps: todo se define en tiempo real y los frames se
# derivan del fps que viene en el tracked JSON. Con ventanas de duración fija,
# los conteos de "ventanas consecutivas" significan lo mismo en todos los videos.
WINDOW_SECONDS  = 1 / 3   # duración de una ventana (10 frames a 30 fps, 20 a 60)
INIT_SECONDS    = 1.0     # respaldo para inicializar rings si el JSON no trae init_frames

# --- Conteos de ventanas (ventanas ya normalizadas en tiempo → ~1 s cada racha) ---
# Con 3 ventanas (~1 s) una maniobra sobre un peg se confirmaba como llegada:
# medido en P01, el ring 4 rondó 4 ventanas el peg 11, lo confirmó y lo bloqueó,
# cuando en realidad se posó en el peg 7 (247 ventanas) — y de paso dejó sin
# destino al ring 2, que sí había llegado al 11. Con 6 (~2 s) el ring 4 toma su
# peg correcto y P07 sube de 4/6 a 5/6.
ARRIVAL_WINDOWS   = 6   # ventanas consecutivas para confirmar llegada a base
LOST_WINDOWS      = 3   # ventanas consecutivas sin detectar al mover activo → sospecha de pérdida
DEPARTURE_WINDOWS = 3   # ventanas consecutivas lejos de casa para confirmar nuevo mover (filtra ruido de threshold)
FINAL_VOTE_WINDOWS = 10 # ventanas del cierre para el reparto final por exclusividad (estable entre 8 y 15)
# Tope de la búsqueda de distancia cruda, en segundos DESDE LA SALIDA real del
# ring (no desde el inicio del video). El tránsito genuino más largo confirmado
# contra GT es 38s (con una caída); 55s da margen sin permitir que enganche
# coincidencias a 60-75s de la salida, como pasó sin acotar (Trial2-3).
DISTANCE_SEARCH_MAX_S = 55


class StateBuilder:
    """
    Lee el tracked JSON y genera hechos espaciales por ventana de tiempo.
    La ventana dura WINDOW_SECONDS reales; sus frames se derivan del fps del
    video, así los umbrales de "ventanas consecutivas" significan el mismo
    tiempo en videos de 30 y de 60 fps.

    Por cada ventana y ring físico se registran tres hechos:
      detected      bool      — detectado en ≥50% de los frames de la ventana
      near_peg_id   int|None  — peg más cercano si dist al ANCLAJE (base de la
                                máscara del peg) < RING_PEG_THRESH; None si en tránsito
      graspers_near [int]     — IDs de pinzas presentes en mayoría de frames

    near_peg_id refleja la posición real del ring. La presencia de un grasper no
    anula near_peg_id: si el ring sigue sobre un peg mientras la pinza lo toca,
    near_peg_id permanece asignado.

    Regla de exclusividad — "mover" único global:
      Solo puede haber UN ring en movimiento a la vez (regla del ejercicio: las
      transferencias son unimanuales, un ring por vez). En vez de que cada ring
      tenga su propia bandera individual de "en tránsito" (lo que permitía que
      2+ rings compitieran ambiguamente por una detección sobrante y se
      intercambiaran identidades), el estado "moviéndose" es una variable
      global (`active_mover`) con un único dueño posible:
        - Mientras `active_mover` esté definido, es el único candidato a
          reclamar detecciones sobrantes (Paso 2 de _assign).
        - Un ring quieto que se ve momentáneamente sin match en Paso 1
          (ruido, oclusión) NO se reasigna por cercanía-adivinada; se deja
          sin resolver esa ventana y hereda su última posición conocida.
        - La promoción a nuevo `active_mover` exige que el ring se detecte
          lejos de su peg de origen durante DEPARTURE_WINDOWS ventanas
          SEGUIDAS (racha `away_streak`), no una sola ventana de cambio —
          esto filtra el parpadeo de un ring quieto justo en el borde del
          umbral de detección (ruido de píxeles, no movimiento real). Si al
          cumplirse la racha hay más de un candidato simultáneo, es
          ambigüedad genuina y no se promueve a nadie esa ventana.
        - Al confirmar llegada, el ring libera el cupo de `active_mover`.
        - Si el mover activo deja de detectarse ≥ LOST_WINDOWS ventanas
          seguidas, se libera el cupo y el ring queda marcado
          `suspected_lost`. Si al cerrar el video nunca confirmó llegada a
          un peg destino, se reporta en `lost_rings` (ring perdido/caído,
          por eliminación contra los pegs destino disponibles).
    """

    def build(self, tracked_path_or_data):
        if isinstance(tracked_path_or_data, dict):
            data = tracked_path_or_data
        else:
            with open(Path(tracked_path_or_data)) as f:
                data = json.load(f)

        # Anclaje del peg = base de su máscara (donde el ring descansa).
        # Fallback a centroid para tracked JSON generados antes del cambio.
        pegs   = {p['peg_id']: (p.get('anclaje') or p['centroid'])
                  for p in data['static_objects']['pegs']}
        frames = data['frames']

        fps         = data.get('fps') or 30.0
        window_size = max(1, round(WINDOW_SECONDS * fps))
        logger.info('FPS %.1f → ventana de %d frames (%.2f s)', fps, window_size, window_size / fps)

        # Nacimiento de rings sobre los MISMOS frames completos que usó el
        # stabilizer (12 pegs + 6 rings visibles a la vez): la votación queda
        # alineada con la referencia de pegs. Respaldo: primer INIT_SECONDS.
        init_idx = data.get('init_frames')
        if init_idx:
            init_sel = [frames[i] for i in init_idx if i < len(frames)]
        else:
            init_sel = frames[:max(1, round(INIT_SECONDS * fps))]

        physical_rings = self._init_rings(init_sel, pegs)
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
                'prev_peg':       r['home_peg'],
                'prev_graspers':  [],
                'prev_detected':  True,
                'last_centroid':  None,
                'arrival_buf':    {},   # {peg_id: n_ventanas_consecutivas_cerca}
                'arrived_peg':    None, # primer peg destino confirmado (para no re-loguear)
                'lost_streak':    0,    # ventanas consecutivas sin detectar mientras era el mover
                'suspected_lost': False,
                'away_streak':    0,    # ventanas consecutivas detectado lejos de casa (candidato a mover)
                'ids':            {r['init_track_id']},  # alias: nombres con los que se le ha visto
                'unseen':         0,    # ventanas seguidas sin verlo (amplía el radio de continuidad)
            }
            for rid, r in physical_rings.items()
        }

        # Estado global — único "mover" autorizado a la vez (regla unimanual).
        # 'huerfano': ring señalado por contradicción (ver _assign) — obliga a
        # cerrar un tránsito estancado y dar el cupo a quien sí se está moviendo.
        state = {'active_mover': None, 'huerfano': None}

        windows         = []
        n_windows       = (len(frames) + window_size - 1) // window_size
        confirmed_dests = set()   # pegs destino ya confirmados — no se reasignan

        for wi in range(n_windows):
            f0       = wi * window_size
            f1       = min(f0 + window_size, len(frames))
            w_frames = frames[f0:f1]

            facts = self._process_window(w_frames, physical_rings, pegs, ctx, state)

            # Actualizar contexto y detectar llegadas
            for rid, f in facts.items():
                ctx[rid]['prev_peg']      = f['near_peg_id']
                ctx[rid]['prev_graspers'] = f['graspers_near']
                ctx[rid]['prev_detected'] = f['detected']
                if f.get('centroid') is not None:
                    ctx[rid]['last_centroid'] = f['centroid']
                    ctx[rid]['unseen'] = 0
                else:
                    ctx[rid]['unseen'] += 1

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

                # Racha de "detectado lejos de casa" — señal de posible
                # desplazamiento real. Requiere sostenerse DEPARTURE_WINDOWS
                # ventanas seguidas para filtrar el parpadeo de un ring
                # quieto justo en el borde del umbral de detección (ruido
                # de píxeles, no movimiento real).
                if rid != state['active_mover'] and ctx[rid]['arrived_peg'] is None:
                    home_peg_r = physical_rings[rid]['home_peg']
                    if f['detected'] and pid != home_peg_r:
                        ctx[rid]['away_streak'] += 1
                    else:
                        ctx[rid]['away_streak'] = 0

            # --- Gestión del "mover" activo (un solo dueño a la vez) ---
            mover = state['active_mover']
            if mover is not None:
                if not facts[mover]['detected']:
                    ctx[mover]['lost_streak'] += 1
                else:
                    ctx[mover]['lost_streak'] = 0

                if ctx[mover]['arrived_peg'] is not None:
                    # Llegó a destino — libera el cupo, vuelve a "quieto".
                    state['active_mover'] = None
                    mover = None
                elif ctx[mover]['lost_streak'] >= LOST_WINDOWS:
                    # Desapareció mientras se movía — sospecha de caída/pérdida.
                    ctx[mover]['suspected_lost'] = True
                    logger.info('Ring %d sin detectar %d ventanas (mover activo) — '
                                'sospecha de pérdida (ventana %d)', mover, LOST_WINDOWS, wi)
                    state['active_mover'] = None
                    mover = None

            if state['active_mover'] is None:
                # Candidato a nuevo mover: único ring aún sin llegar cuya
                # racha de "lejos de casa" ya se sostuvo DEPARTURE_WINDOWS
                # ventanas seguidas (desplazamiento real confirmado, no
                # parpadeo de un ring quieto en el borde del umbral).
                candidates = [
                    rid for rid in physical_rings
                    if ctx[rid]['away_streak'] >= DEPARTURE_WINDOWS
                ]
                if len(candidates) == 1:
                    new_mover = candidates[0]
                    state['active_mover']         = new_mover
                    ctx[new_mover]['lost_streak'] = 0
                    ctx[new_mover]['away_streak'] = 0
                    logger.info('Ring %d pasa a ser el mover activo (ventana %d)', new_mover, wi)
                elif len(candidates) >= 2:
                    logger.debug('ventana %d: promoción bloqueada — candidatos simultáneos %s (away: %s)',
                                 wi, candidates,
                                 {r: ctx[r]['away_streak'] for r in candidates})

            windows.append({
                'window': [f0, f1 - 1],
                'facts':  {str(k): v for k, v in facts.items()},
            })

        # ---- Cierre: reparto final por exclusividad ----
        # La confirmación incremental puede equivocarse cuando un ring RONDA un
        # peg sin lograr colocarlo y termina en otro: medido en P01, el ring 1
        # estuvo 22 ventanas sobre el peg 11 (intento fallido), lo confirmó y lo
        # bloqueó, para acabar posándose en el peg 0 — dejando sin destino al
        # ring 2, que sí se posó en el 11.
        #
        # Al cerrar el video la ambigüedad desaparece: cada ring está en un peg
        # distinto y solo hay que repartir. Se vota con las últimas ventanas y
        # se adjudica por mayor evidencia, respetando la exclusividad (un peg,
        # un ring). Esto corrige la adjudicación sin tocar los tiempos de
        # tránsito, que salen de la confirmación incremental.
        if windows:
            ultimas = windows[-FINAL_VOTE_WINDOWS:]
            votos = []
            for rid in physical_rings:
                c = Counter(w['facts'][str(rid)]['near_peg_id'] for w in ultimas
                            if w['facts'][str(rid)]['detected']
                            and w['facts'][str(rid)]['near_peg_id'] in dest_pegs)
                for peg, n in c.items():
                    votos.append((n, rid, peg))

            # Bug corregido: un ring SIN evidencia fresca en el cierre pero con
            # una confirmación en vivo previa quedaba invisible para este
            # reparto — otro ring podía ganarle su peg por voto fresco sin que
            # nadie se enterara, y los dos terminaban con el mismo destino
            # (confirmado con datos reales: Trial2-1, dos rings → peg 11).
            #
            # Se agrega esa confirmación previa como UN VOTO MÁS, con peso
            # máximo (equivale a dominar toda la ventana final). Así nunca es
            # invisible — compite en la misma lista, con la misma regla de
            # exclusividad. Si el ring SÍ tiene evidencia fresca propia
            # (aunque sea ambigua, apuntando a otro peg), esa evidencia manda
            # y la confirmación vieja no se protege — necesario para el caso
            # ya validado en P01 (un ring mal confirmado por un intento
            # fallido debe poder corregirse).
            con_evidencia_fresca = {rid for _, rid, _ in votos}
            for rid in physical_rings:
                previo = ctx[rid]['arrived_peg']
                if previo is not None and rid not in con_evidencia_fresca:
                    votos.append((FINAL_VOTE_WINDOWS, rid, previo))

            final_asig, pegs_usados = {}, set()
            for n, rid, peg in sorted(votos, reverse=True):
                if rid in final_asig or peg in pegs_usados:
                    continue
                final_asig[rid], _ = peg, pegs_usados.add(peg)

            for rid in physical_rings:
                peg    = final_asig.get(rid)
                previo = ctx[rid]['arrived_peg']

                # Bug corregido: si un ring quedó marcado `suspected_lost`
                # (fue mover, se perdió de vista) y MÁS ADELANTE se
                # reconfirma por la vía normal en el mismo peg que ya vota
                # el cierre, `previo == peg` entraba directo al `continue` de
                # abajo y NUNCA limpiaba `suspected_lost` — el ring terminaba
                # con `arrived_at_valid_dest=True` Y `perdido=True` a la vez
                # (confirmado con datos reales: Trial2-2, ring0). Cualquier
                # ring con un peg confirmado, sea nuevo o el mismo de antes,
                # deja de estar "perdido" — se limpia ANTES del atajo.
                if peg is not None:
                    ctx[rid]['suspected_lost'] = False

                if previo == peg:
                    continue

                if peg is None:
                    # Tenía una confirmación previa pero perdió la competencia
                    # por ese peg (otro ring tenía mejor evidencia) y no le
                    # quedó ningún otro — se desconfirma en vez de dejarlo
                    # duplicado.
                    #
                    # Bug corregido: el `break` original solo limpiaba la
                    # PRIMERA ventana con `arrived_peg` marcado. El carry-forward
                    # (línea ~200) escribe ese mismo valor en TODAS las ventanas
                    # posteriores a la confirmación original — con `break`,
                    # todas esas quedaban con el peg viejo intacto, y el escaneo
                    # de metrics.py (que busca la primera ventana no-None)
                    # encontraba la siguiente y reportaba al ring como llegado
                    # de todos modos (confirmado con datos reales: Trial2-2,
                    # ring 0 y ring 5 — el log mostraba "pierde su confirmación"
                    # pero las métricas seguían marcando llegada al peg viejo).
                    # Hay que limpiar TODAS las ventanas con ese valor, no solo
                    # la primera.
                    ctx[rid]['arrived_peg'] = None
                    for w in windows:
                        if w['facts'][str(rid)].get('arrived_peg') == previo:
                            w['facts'][str(rid)]['arrived_peg'] = None
                    logger.info('Reparto final: ring %d pierde su confirmación en peg %s '
                                '(otro ring tenía mejor evidencia)', rid, previo)
                    continue

                ctx[rid]['arrived_peg']    = peg
                ctx[rid]['suspected_lost'] = False

                # Intento principal: CUÁNDO llegó de verdad, por distancia
                # cruda al peg ya confirmado (sin el filtro de ratio). El
                # filtro de ratio decide A QUÉ peg pertenece, pero puede
                # ocultar el momento del asentamiento cuando dos pegs quedan
                # muy cerca — medido en P01: near_peg_id nunca marcó "peg 0"
                # para un ring sentado ahí 20+ s porque el peg 11 vecino
                # mantenía el ratio ambiguo, pero la distancia cruda mostraba
                # el asentamiento clarísimo (racha estable de 42-50 px desde
                # el segundo exacto del GT). Exige racha COMPLETA (no mayoría
                # — probado y descartado: la mayoría mejoraba un ring pero
                # empeoraba otro que ya estaba bien). Si no hay racha limpia
                # en todo el video, no se toca nada — se cae al respaldo de
                # siempre.
                arrival_wi = self._find_arrival_by_distance(windows, pegs, physical_rings, str(rid), peg)

                if arrival_wi is None:
                    for w in windows:
                        if w['facts'][str(rid)].get('arrived_peg') is not None:
                            w['facts'][str(rid)]['arrived_peg'] = peg
                            break
                    else:
                        for w in ultimas:
                            if w['facts'][str(rid)]['near_peg_id'] == peg:
                                w['facts'][str(rid)]['arrived_peg'] = peg
                                break
                else:
                    for w in windows:
                        w['facts'][str(rid)].pop('arrived_peg', None)
                    windows[arrival_wi]['facts'][str(rid)]['arrived_peg'] = peg

                logger.info('Reparto final: ring %d → peg %s (antes: %s)', rid, peg, previo)

        # Perdido por eliminación: fue mover, se dio por perdido, y nunca
        # confirmó llegada a ningún peg destino antes de terminar el video.
        lost_rings = sorted(
            rid for rid in physical_rings
            if ctx[rid]['suspected_lost'] and ctx[rid]['arrived_peg'] is None
        )
        if lost_rings:
            logger.warning('Rings marcados como perdidos (no confirmaron llegada): %s', lost_rings)

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
            'fps':            fps,
            'window_size':    window_size,
            'window_seconds': round(window_size / fps, 4),
            'init_frames':    sorted(init_idx) if init_idx else len(init_sel),
            'init_confiable': data.get('init_confiable'),
            'physical_rings': {str(k): v for k, v in physical_rings.items()},
            'source_pegs':    sorted(source_pegs),
            'dest_pegs':      dest_pegs,
            'pegs':           {str(pid): centroid for pid, centroid in pegs.items()},
            'platform':       platform,
            'windows':        windows,
            'lost_rings':     lost_rings,
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
        """Detecta rings físicos en los frames de inicialización recibidos.

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

    def _consolidar(self, w_frames):
        """Agrupa las detecciones de la ventana en OBJETOS, uno por track_id.

        Primer paso del diseño por ventana: antes de decidir identidades se
        estabiliza qué hay en la ventana. Cada track_id presente produce un
        objeto con su posición media y en cuántos frames apareció.

        Un ring que sufre un switch aparece como DOS objetos (su id viejo y el
        nuevo); ambos se le asignarán y el de más frames — la moda — será su
        representante. Medido en el switch 310→332 de P01: v356 tiene 310 con
        9/10 frames y 332 con 1/10; v357 invierte a 332 con 7/10 y 310 con 3/10.
        Nunca hay empate, así que la moda resuelve el relevo sin ambigüedad.

        `-1` (sin track) se ignora: no es una identidad.
        """
        por_id = defaultdict(list)
        for frame in w_frames:
            for d in frame['detections']:
                if d['class_name'] == 'ring' and d['track_id'] != -1:
                    por_id[d['track_id']].append(d)

        objetos = []
        for tid, dets in por_id.items():
            cs = [d['centroid'] for d in dets]
            objetos.append({
                'tid':      tid,
                'centroid': [sum(c[0] for c in cs) / len(cs),
                             sum(c[1] for c in cs) / len(cs)],
                'n':        len(dets),
                'dets':     dets,
            })
        return objetos

    def _resolver_ventana(self, objetos, physical_rings, pegs, ctx, state):
        """Asigna OBJETOS de la ventana a rings físicos — una sola decisión.

        Jerarquía (evidencia más fuerte primero):

          1. ALIAS — el objeto trae un track_id que ya es alias de un ring.
             Es el nombre por el que lo conocemos; nadie más compite por él.
             Esto impide el robo que se observó: un ring "zombi" anclado en un
             peg viejo reclamaba por cercanía al viajero que pasaba a 54 px, y
             de paso le arrebataba el id.

          2. ANCLA — un ring todavía sin objeto reclama el que esté sobre su
             peg (casa o destino confirmado), siempre que ese objeto no tenga
             ya dueño por alias.

          3. CONTINUIDAD — para el que quedó sin objeto y sin alias (switch
             recién nacido): el objeto más cercano a su última posición
             conocida. El radio crece con las ventanas que lleva sin verse,
             porque el ring físico siguió moviéndose mientras no lo veíamos.

        Devuelve {rid: [objetos]} — un ring puede recibir dos objetos durante
        un switch (id viejo + id nuevo).
        """
        asign  = defaultdict(list)
        dueno  = {}

        # --- 1. por alias ---
        for o in objetos:
            for rid in physical_rings:
                if o['tid'] in ctx[rid]['ids']:
                    asign[rid].append(o)
                    dueno[o['tid']] = rid
                    break

        # --- 2. por ancla (peg propio) ---
        for rid in physical_rings:
            if asign[rid]:
                continue
            peg = ctx[rid]['arrived_peg']
            if peg is None and rid != state['active_mover']:
                peg = ctx[rid]['prev_peg'] or physical_rings[rid]['home_peg']
            if peg is None:
                continue
            libres = [o for o in objetos if o['tid'] not in dueno]
            mejor, dmin = None, float('inf')
            for o in libres:
                d = self._dist(o['centroid'], pegs[peg])
                if d < RING_PEG_THRESH and d < dmin:
                    dmin, mejor = d, o
            if mejor is not None:
                asign[rid].append(mejor)
                dueno[mejor['tid']] = rid

        # --- 3. por continuidad de posición ---
        pendientes = [r for r in physical_rings if not asign[r]]
        pares = []
        for rid in pendientes:
            ref = ctx[rid]['last_centroid'] or pegs[physical_rings[rid]['home_peg']]
            # el radio crece con el tiempo sin ver al ring
            radio = TRANSIT_MAX_JUMP * (1 + ctx[rid]['unseen'])
            for o in objetos:
                if o['tid'] in dueno:
                    continue
                d = self._dist(o['centroid'], ref)
                if d < radio:
                    pares.append((d, rid, o))
        for d, rid, o in sorted(pares, key=lambda x: x[0]):
            if asign[rid] or o['tid'] in dueno:
                continue
            asign[rid].append(o)
            dueno[o['tid']] = rid

        return asign

    def _process_window(self, w_frames, physical_rings, pegs, ctx, state):
        """Calcula hechos por ring para una ventana de frames.

        La identidad ring↔detección se resuelve UNA vez por ventana sobre los
        objetos consolidados (`_consolidar` + `_resolver_ventana`), no frame a
        frame: decidir en cada frame hacía que un switch se disputara diez
        veces dentro de la misma ventana y contaminaba los alias entre rings.
        """
        objetos = self._consolidar(w_frames)
        asign_w = self._resolver_ventana(objetos, physical_rings, pegs, ctx, state)

        # Alias: el ring aprende los nombres con los que se le vio en esta
        # ventana. Los conjuntos no se le quitan a nadie — cada objeto tuvo un
        # único dueño en la resolución, así que no hay competencia posible.
        for rid, objs in asign_w.items():
            for o in objs:
                ctx[rid]['ids'].add(o['tid'])

        # Detecciones que le corresponden a cada ring en esta ventana
        mias = {rid: {id(d) for o in asign_w[rid] for d in o['dets']}
                for rid in physical_rings}

        per_frame = {rid: [] for rid in physical_rings}

        for frame in w_frames:
            ring_dets = [d for d in frame['detections'] if d['class_name'] == 'ring']
            tfm_dets  = [d for d in frame['detections'] if d['class_name'] == 'TFM']
            assign    = {rid: next((d for d in ring_dets if id(d) in mias[rid]), None)
                         for rid in physical_rings}
            assign    = {k: v for k, v in assign.items() if v is not None}

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

    def _find_arrival_by_distance(self, windows, pegs, physical_rings, key, peg,
                                   thresh=RING_PEG_THRESH, req=ARRIVAL_WINDOWS,
                                   max_ahead_s=DISTANCE_SEARCH_MAX_S):
        """Primera racha de `req` ventanas SEGUIDAS con el centroide del ring
        a menos de `thresh` px del peg — por DISTANCIA CRUDA, no por
        `near_peg_id` (que ya pasó por el filtro de ratio y puede no marcar
        nunca ese peg si otro queda cerca). Ver comentario en el reparto
        final para el caso real que motivó esto.

        ACOTADO (2026-08-31): sin límite, si la racha real es sucia, la
        búsqueda seguía de largo y enganchaba una racha limpia muy posterior
        por pura coincidencia — medido en Trial2-3: dos rings distintos
        "llegaban" casi al mismo segundo (74.7s y 75.7s) sin relación con su
        salida real (1.3s y 11.3s). Se acota el inicio a la salida real del
        ring (última ventana detectada en su peg de casa) y el rango a
        `max_ahead_s` segundos desde ahí — más que el tránsito genuino más
        largo confirmado contra GT (38s, con una caída). Si no hay racha
        limpia dentro de ese rango, devuelve None — el llamador cae al
        respaldo, nunca busca más lejos a ciegas.
        """
        home_peg = physical_rings[int(key)]['home_peg']
        dep_wi = 0
        for wi, w in enumerate(windows):
            f = w['facts'][key]
            if f.get('detected') and f.get('near_peg_id') == home_peg:
                dep_wi = wi

        max_ahead_w = int(max_ahead_s / WINDOW_SECONDS)
        end = min(len(windows), dep_wi + max_ahead_w)

        target = pegs[peg]
        streak = 0
        for wi in range(dep_wi, end):
            w = windows[wi]
            f = w['facts'][key]
            c = f.get('centroid')
            if f.get('detected') and c is not None and self._dist(c, target) < thresh:
                streak += 1
                if streak >= req:
                    return wi
            else:
                streak = 0
        return None


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
