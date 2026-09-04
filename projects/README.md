# TFM Peg Transfer — Sistema de métricas y retroalimentación IA

Sistema de visión por computador para extraer métricas objetivas de desempeño en el ejercicio quirúrgico **Peg Transfer** (FLS — Fundamentals of Laparoscopic Surgery) a partir de video, como Trabajo de Fin de Máster.

El ejercicio consiste en mover 6 argollas (`ring`), una a la vez, desde 6 clavijas de origen a 6 clavijas de destino distintas, usando dos pinzas laparoscópicas. El pipeline procesa el video completo y produce un JSON de métricas por argolla: tiempo de tránsito, economía de movimiento, coordinación bimanual y errores (caídas / salidas de plataforma).

## Estado actual

Pipeline completo y validado contra ground truth manual. Dataset de 26 videos (3 participantes: P01, P07, P10) procesado íntegramente con el diseño final. Ver `outputs/consolidado.csv` para los resultados agregados.

## Arquitectura — 5 etapas secuenciales

```
video.mp4
   │
   ▼
[1] Detector (inference/detector.py)
   │  YOLO26-seg (Ultralytics) sobre cada frame, sin tracking.
   │  4 clases: ring, TFM (pinza — nombre heredado del dataset), peg, platform.
   ▼  → outputs/raw/<video>_raw.json
[2] Stabilizer (inference/stabilizer.py)
   │  Fija la posición de los 12 pegs y la plataforma (K-means sobre los
   │  primeros frames "completos" de la ventana de init) y filtra ruido
   │  de detecciones dinámicas por confianza/clase.
   ▼  → outputs/stable/<video>_stable.json
[3] Tracker (tracking/tracker.py)
   │  ByteTrack (boxmot) con Kalman ajustado y mini-bbox sintético centrado
   │  en el centroide de la máscara (evita ID-switches por solape de bbox).
   ▼  → outputs/tracked/<video>_tracked.json
[4] StateBuilder (states/state_builder.py)
   │  Agrega por ventana temporal (1/3 s), resuelve identidad con cascada
   │  alias→ancla→continuidad, determina llegada a peg por contención
   │  bbox-ring/anclaje-peg, y hace un reparto final por exclusividad al
   │  cierre del video.
   ▼  → outputs/states/<video>_states.json
[5] Metrics (metrics/metrics.py)
   │  Calcula economía de movimiento, coordinación bimanual, errores
   │  (caída fuera de campo / suelta dentro de plataforma) y tiempos.
   ▼  → outputs/metrics/<video>_metrics.json
```

`metrics/consolidate.py` agrega todos los `*_metrics.json` de `outputs/metrics/` en un único `outputs/consolidado.csv` (una fila por video/trial).

## Estructura del repositorio

```
projects/
├── config.py              # constantes globales de las 5 etapas (única fuente de verdad)
├── model/                 # pesos entrenados (.pt) — xl1280-1.pt es el usado en main.py
├── inference/
│   ├── detector.py        # etapa 1
│   └── stabilizer.py      # etapa 2
├── tracking/
│   └── tracker.py         # etapa 3
├── states/
│   └── state_builder.py   # etapa 4
├── metrics/
│   ├── metrics.py         # etapa 5
│   └── consolidate.py     # agregación final a CSV
├── main/
│   └── main.py            # orquesta las 5 etapas para un video
└── outputs/
    ├── raw/ stable/ tracked/ states/ metrics/   # JSON intermedios por etapa
    └── consolidado.csv                           # tabla final consolidada
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

- **Con GPU NVIDIA/CUDA:** instalar `torch` antes, con el índice de PyTorch correspondiente a tu versión de CUDA (ver comentario al inicio de `requirements.txt`).
- **Sin GPU NVIDIA (CPU, ej. Intel Core Ultra):** no requiere nada especial, `pip install -r requirements.txt` ya instala la build CPU.

## Uso

Procesar un video completo (las 5 etapas, guarda solo el JSON de métricas final):

```bash
python -m projects.main.main "ruta/al/video.mp4"
```

Opciones: `--output`/`-o` (ruta del JSON de salida, default `outputs/metrics/<video>_metrics.json`), `--conf` (umbral de confianza del detector, default 0.4).

Consolidar todos los resultados procesados en un CSV:

```bash
python -m projects.metrics.consolidate
```

## Validación

Cada etapa fue validada de forma independiente contra ground truth observado manualmente y datos crudos del pipeline (no solo contra la salida final):

- **Detección:** 86.9% de frames con el conteo correcto de 6 rings; 0.4% de duplicados reales.
- **Tracking:** cero IDs duplicados/robados tras corrección (antes: 125 duplicados y 175 "teleports" en un solo video).
- **StateBuilder:** 6/6 rings correctos en videos de referencia con ground truth manual; asignación ring→peg por geometría (no por umbral de distancia calibrado), para generalizar entre datasets con ángulos de cámara distintos.
- **Metrics:** dos bugs de contradicción/carrera de tiempo encontrados y corregidos vía comparación contra ground truth real.
- **Dataset completo (26 videos):** comparación cuantitativa antes/después del rediseño muestra mejora sistémica — economía de movimiento +15%, caídas/errores −34%, coordinación bimanual detectada +27%.

## Efecto del ángulo de cámara

Comparando el dataset principal (P01/P07/P10, cámara oblicua ~45°, encuadre laparoscópico típico) contra un video de prueba con cámara más cenital (`Mov_Exi_1`, dataset previo del investigador), el ángulo de cámara resultó ser un factor real de precisión, no solo estético:

- **Cámara oblicua (dataset principal):** la perspectiva distorsiona las distancias de forma NO uniforme sobre el tablero. En la fila de pegs de ORIGEN el ring en reposo queda a ~18px de su anclaje, pero en la fila de DESTINO a ~49px — casi el triple. Un umbral de distancia fijo no puede servir para ambas filas a la vez, y calibrar uno por fila sobreajusta a ese video/cámara específico.
- **Cámara cenital (Mov_Exi_1):** un ring en reposo genuino quedó a 86px de su anclaje — muy por encima de cualquier umbral razonable calibrado en el dataset oblicuo. Confirmó que un umbral fijo en píxeles no generaliza entre cámaras.
- Esta observación fue la que forzó a abandonar el umbral de distancia calibrado y migrar la asignación ring→peg a **contención geométrica** (bbox del ring vs. punto de anclaje del peg), un criterio invariante al ángulo de cámara. Con ese cambio, `Mov_Exi_1` pasó a 6/6 (incluyendo un ring que antes nunca confirmaba llegada) sin degradar el dataset principal.
- **Conclusión:** una cámara más cenital reduce la distorsión de perspectiva y hace la asignación ring→peg geométricamente más simple y estable. Recomendación para futuras capturas del ejercicio: un encuadre más cenital da métricas más precisas con menos necesidad de heurísticas de compensación por perspectiva.

## Limitaciones documentadas (no son bugs)

- **Ventana de inicialización inestable:** si el video arranca con una pinza ya sobre los pegs o un ring ya en movimiento, el conteo físico inicial de rings puede quedar por debajo de 6. Es una limitación de captura del dataset, no del pipeline.
- **Videos recortados sin margen final:** algunos videos del dataset fueron recortados por el investigador (originalmente grabaciones de ida y vuelta) muy cerca del último movimiento válido, lo que puede impedir confirmar esa llegada a tiempo.
- **Ángulo de cámara oblicuo:** el detector de "salida de plataforma" es débil en encuadres muy oblicuos donde la plataforma cubre casi todo el frame.
