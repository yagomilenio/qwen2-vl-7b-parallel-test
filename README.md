# LM Studio Vision Runner

Procesamiento paralelo de imágenes con **Qwen2-VL-7B** via LM Studio CLI.

## Cómo encaja en la plataforma

```
Servidor de paralelización
  │
  ├── lee config.toml          → sabe qué imágenes hay y cuántas
  │                            → sabe cómo trocear el trabajo
  │                            → sabe dónde escribir los outputs
  │
  └── lanza por cada worker:
        python run_vision.py --start 0 --end 9 --pack general --output outputs/results_0_9.json
        python run_vision.py --start 10 --end 19 --pack general --output outputs/results_10_19.json
        ...
```

`run_vision.py` **no lee `config.toml`**. Recibe todo por argumentos y escribe su output donde se le indica.

## Estructura

```
├── config.toml          # Solo para el servidor (inputs · runner · outputs)
├── model_config.toml    # Solo para run_vision.py (modelo, timeouts...)
├── run_vision.py        # Script principal — recibe args, escribe output
├── Makefile             # Uso manual / desarrollo
├── inputs/images/       # ← pon aquí tus imágenes
├── prompts/             # Packs de prompts (el script los busca aquí)
│   ├── general.json
│   ├── ocr.json
│   ├── technical.json
│   ├── art_critique.json
│   └── accessibility.json
└── outputs/             # El servidor define dónde escribe cada worker
```

## Setup

```bash
make setup        # instala lms CLI + descarga Qwen2-VL (~4.5 GB)
```

## Uso manual

```bash
make list         # ver índices disponibles

make run START=0 END=9 PACK=general
make run START=0 END=9 PACK=ocr
make test         # prueba rápida: imagen 0
```

## Argumentos de run_vision.py

| Argumento | Requerido | Descripción |
|---|---|---|
| `--start` | Índice inicial (inclusivo) |
| `--end` | Índice final (inclusivo) |
| `--output` | Ruta del JSON de salida |
| `--pack` | — | Pack de prompts (defecto: `general`) |
| `--input-dir` | — | Directorio de imágenes (defecto: `inputs/images`) |
| `--model-config` | — | Config del modelo (defecto: `model_config.toml`) |

## Packs de prompts

| Pack | Enfoque |
|---|---|
| `general` | Descripción, sujeto, mood, contexto |
| `ocr` | Extracción y análisis de texto |
| `technical` | Composición, luz, calidad, tipo de imagen |
| `art_critique` | Estilo, simbolismo, narrativa, impacto |
| `accessibility` | Alt-text WCAG, screen reader, contraste |
