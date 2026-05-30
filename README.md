# qwen2-vl-7b-parallel-test

Procesamiento paralelo de imágenes con **Qwen2-VL-7B** via LM Studio. Cada worker recibe un rango de imágenes y un pack de prompts, ejecuta el modelo visión sobre cada imagen y escribe los resultados en un fichero JSON.

## Cómo encaja en la plataforma

```
Plataforma de paralelización
  │
  ├── lee config.toml
  │     ├── inputs.directory → inputs/images/  (detecta imágenes automáticamente)
  │     └── runner.command   → python run_vision.py
  │
  └── lanza por cada worker:
        python run_vision.py --start 0   --end 49  --pack general --output outputs/results_0_49.json
        python run_vision.py --start 50  --end 99  --pack general --output outputs/results_50_99.json
        ...
```

`run_vision.py` **no lee `config.toml`**. Recibe todo por argumentos de línea de comandos y escribe su output donde se le indica.

## Estructura

```
qwen2-vl-7b-parallel-test/
├── config.toml          # Solo para la plataforma (inputs, runner, outputs)
├── model_config.toml    # Solo para run_vision.py (modelo, timeouts, reintentos)
├── run_vision.py        # Script principal — recibe args, llama al modelo, escribe JSON
├── Makefile             # Atajos para setup, desarrollo y pruebas
├── inputs/images/       # ← pon aquí las imágenes a procesar
├── prompts/             # Packs de prompts seleccionables
│   ├── general.json
│   ├── accesibilidad.json
│   ├── documentos.json
│   ├── escenas.json
│   ├── inspeccion.json
│   ├── marketing.json
│   ├── personas.json
│   └── productos.json
└── outputs/             # Un JSON por worker (definido por la plataforma)
    └── results_0_49.json
```

## Requisitos

- Python 3.11+ (usa `tomllib` nativo; en 3.10 instala `tomli`)
- [LM Studio](https://lmstudio.ai) con el modelo `Qwen2-VL-7B-Instruct` cargado
- `pip install lmstudio`

## Setup

```bash
make setup        # instala lms CLI + descarga Qwen2-VL-7B (~4.5 GB) + instala librería Python
```

El setup realiza los siguientes pasos:
1. Descarga e instala la CLI de LM Studio (`lms`).
2. Descarga el modelo `Qwen2-VL-7B-Instruct-GGUF@Q4_K_M`.
3. Instala la librería Python `lmstudio`.

## Uso manual

```bash
make list                               # muestra las imágenes disponibles con sus índices

make run START=0 END=9                  # procesa imágenes 0-9 con el pack 'general'
make run START=0 END=9 PACK=marketing   # procesa imágenes 0-9 con el pack 'marketing'

make test                               # prueba rápida: solo la imagen 0, pack 'general'

make clean                              # borra todos los outputs
```

## Argumentos de `run_vision.py`

| Argumento | Requerido | Por defecto | Descripción |
|---|---|---|---|
| `--start` | ✓ | — | Índice inicial (inclusivo) |
| `--end` | ✓ | — | Índice final (inclusivo) |
| `--output` | ✓ | — | Ruta del fichero JSON de salida |
| `--pack` | — | `general` | Pack de prompts a usar |
| `--input-dir` | — | `inputs/images` | Directorio de imágenes |
| `--model-config` | — | `model_config.toml` | Config interna del modelo |

## Packs de prompts

| Pack | Prompts | Enfoque |
|---|---|---|
| `general` | G01–G05 | Descripción general, objetos, colores, texto visible, resumen |
| `accesibilidad` | A01–A05 | Alt text WCAG, descripción larga, contenido sensible, keywords |
| `documentos` | D01–D06 | Tipo de documento, extracción de texto, datos clave, estructura |
| `escenas` | E01–E06 | Tipo de escena, entorno, iluminación, clima, geolocalización, época |
| `inspeccion` | Q01–Q05 | Calidad de imagen, anomalías, comparación con estándar, medidas |
| `marketing` | M01–M05 | Captions para Instagram, LinkedIn, X/Twitter, anuncios, engagement |
| `personas` | H01–H05 | Número de personas, actividad, vestimenta, emociones, contexto social |
| `productos` | P01–P06 | Identificación, descripción e-commerce, estado, atributos, tags SEO |

Para añadir un pack personalizado, crea un fichero `prompts/mi_pack.json` con la misma estructura:

```json
{
  "prompts": [
    { "id": "X01", "label": "Nombre del prompt", "prompt": "Texto del prompt..." }
  ]
}
```

## Configuración del modelo

Edita `model_config.toml` para ajustar el comportamiento del modelo:

```toml
[model]
id             = "qwen2-vl-7b-instruct@Q4_K_M"
temperature    = 0.7
max_tokens     = 1024
context_length = 4096
timeout_sec    = 180
max_retries    = 3
retry_delay    = 5.0
```

## Formato de salida

Cada worker genera un fichero JSON con esta estructura:

```json
{
  "_meta": {
    "model": "qwen2-vl-7b-instruct@Q4_K_M",
    "pack": "general",
    "start": 0,
    "end": 1,
    "total_items": 2,
    "prompts_per_item": 5,
    "total_ops": 10,
    "successes": 10,
    "failures": 0,
    "started_at": "2026-03-03T16:00:36.159050",
    "finished_at": "2026-03-03T16:01:41.255589"
  },
  "results": {
    "0": {
      "index": 0,
      "filename": "img_001.jpg",
      "path": "inputs/images/img_001.jpg",
      "size_bytes": 23814,
      "prompts": {
        "G01": {
          "label": "Descripción general",
          "prompt": "Describe detalladamente...",
          "timestamp": "2026-03-03T16:00:49.614087",
          "success": true,
          "response": "La imagen muestra...",
          "elapsed_sec": 10.6,
          "attempt": 1,
          "error": null
        }
      }
    }
  }
}
```

## Comandos del Makefile

- `make setup`: Instala lms CLI, descarga el modelo y la librería Python.
- `make list`: Muestra las imágenes disponibles con sus índices.
- `make run START=<n> END=<n> [PACK=<pack>]`: Procesa el rango especificado.
- `make test`: Procesa solo la imagen 0 con el pack `general`.
- `make clean`: Elimina todos los ficheros de `outputs/`.
