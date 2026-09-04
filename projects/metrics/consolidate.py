"""
Consolida todos los *_metrics.json de outputs/metrics/<participante>/
en un único CSV con una fila por trial.
"""
import csv
import json
from pathlib import Path

METRICS_DIR   = Path(__file__).resolve().parent.parent / 'outputs' / 'metrics'
OUTPUT_CSV    = Path(__file__).resolve().parent.parent / 'outputs' / 'consolidado.csv'
IGNORAR_DIRS  = {'metrics_viejas'}

PARTICIPANTE_POR_FECHA = {
    '20230911': 'P01',
    '20230925': 'P07',
    '20231109': 'P10',
}

COLUMNS = [
    'participante', 'nombre_video', 'trial',
    'rings_completados', 'rings_total', 'tasa_completacion',
    'tiempo_ejercicio_s',
    'transit_time_promedio_s', 'transit_time_min_s', 'transit_time_max_s',
    'economy_ratio_promedio',
    'caidas_fuera_campo_total', 'caidas_dentro_suelta_total', 'intentos_total',
    'rings_bimanual', 'rings_perdidos',
]


def _participante(jf):
    if jf.parent != METRICS_DIR:
        return jf.parent.name.split('_')[0]
    fecha = jf.stem.split(' ')[0]
    return PARTICIPANTE_POR_FECHA.get(fecha)


def _trial(stem):
    # "20230911124552 Trial1-1_metrics" → "Trial1-1"
    parts = stem.replace('_metrics', '').split(' ')
    return parts[1] if len(parts) > 1 else stem


def _row(participante, nombre_video, trial, data):
    summary = data.get('summary', {})
    rings   = data.get('rings', {})

    completed = [r for r in rings.values() if r.get('arrived_at_valid_dest')]

    transit_times  = [r['transit_time_s']  for r in completed if r.get('transit_time_s')  is not None]
    economy_ratios = [r['economy_ratio']    for r in completed if r.get('economy_ratio')   is not None]
    caidas_fuera   = sum(r.get('drop_errors',  {}).get('n_drops', 0) for r in rings.values())
    caidas_dentro  = sum(r.get('loose_errors', {}).get('n_drops', 0) for r in rings.values())
    intentos       = sum(r.get('intentos',     {}).get('total', 0)   for r in rings.values())
    bimanual       = sum(1 for r in completed if r.get('bimanual', {}).get('bimanual', False))
    perdidos       = sum(1 for r in rings.values() if r.get('perdido'))

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else None

    return {
        'participante':             participante,
        'nombre_video':             nombre_video,
        'trial':                    trial,
        'rings_completados':        summary.get('rings_completed'),
        'rings_total':              summary.get('rings_total'),
        'tasa_completacion':        summary.get('completion_rate'),
        'tiempo_ejercicio_s':       summary.get('total_exercise_time_s'),
        'transit_time_promedio_s':  avg(transit_times),
        'transit_time_min_s':       round(min(transit_times), 3) if transit_times else None,
        'transit_time_max_s':       round(max(transit_times), 3) if transit_times else None,
        'economy_ratio_promedio':   avg(economy_ratios),
        'caidas_fuera_campo_total':   caidas_fuera,
        'caidas_dentro_suelta_total': caidas_dentro,
        'intentos_total':             intentos,
        'rings_bimanual':           bimanual,
        'rings_perdidos':           perdidos,
    }


def main():
    rows = []

    for jf in sorted(METRICS_DIR.rglob('*_metrics.json')):
        if IGNORAR_DIRS & set(jf.relative_to(METRICS_DIR).parts[:-1]):
            continue
        stem = jf.stem
        participante = _participante(jf)
        if participante is None:
            print(f'  (omitido, sin participante reconocido: {jf})')
            continue
        with open(jf) as f:
            data = json.load(f)
        nombre_video = stem.replace('_metrics', '')
        trial        = _trial(stem)
        rows.append(_row(participante, nombre_video, trial, data))

    rows.sort(key=lambda r: (r['participante'], r['trial']))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'CSV generado: {OUTPUT_CSV}')
    print(f'Filas: {len(rows)}')
    for r in rows:
        print(f"  {r['participante']:4s}  {r['trial']:10s}  "
              f"{r['rings_completados']}/{r['rings_total']}  "
              f"{r['tiempo_ejercicio_s']}s")


if __name__ == '__main__':
    main()
