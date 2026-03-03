#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lmstudio as lms


# ══════════════════════════════════════════════════════════════════
#  CONFIG INTERNA DEL MODELO
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    id:             str   = "qwen2-vl-7b-instruct@Q4_K_M"
    temperature:    float = 0.7
    max_tokens:     int   = 1024
    context_length: int   = 4096
    timeout_sec:    int   = 180
    max_retries:    int   = 3
    retry_delay:    float = 5.0


def load_model_config(path: str = "model_config.toml") -> ModelConfig:
    p = Path(path)
    if not p.exists():
        return ModelConfig()
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    return ModelConfig(**raw.get("model", {}))


# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════

def setup_logging(output_path: Path) -> None:
    log_path = output_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]
    )


# ══════════════════════════════════════════════════════════════════
#  PROMPT PACK
# ══════════════════════════════════════════════════════════════════

def load_prompt_pack(pack_name: str) -> list[dict]:
    pack_file = Path("prompts") / f"{pack_name}.json"
    if not pack_file.exists():
        available = [p.stem for p in Path("prompts").glob("*.json")]
        print(f"ERROR: Pack '{pack_name}' no encontrado. Disponibles: {available}", file=sys.stderr)
        sys.exit(1)
    with open(pack_file, encoding="utf-8") as f:
        return json.load(f)["prompts"]


# ══════════════════════════════════════════════════════════════════
#  LLAMADA AL MODELO
# ══════════════════════════════════════════════════════════════════


def get_model(client, model_cfg):
    for attempt in range(1, model_cfg.max_retries + 1):
        try:
            return client.llm.model(model_cfg.id)
        except Exception as e:
            logging.warning(f"Error cargando modelo {model_cfg.id} (intento {attempt}): {e}")
            time.sleep(model_cfg.retry_delay)
    raise RuntimeError(f"No se pudo inicializar el modelo {model_cfg.id} tras {model_cfg.max_retries} intentos")

def run_prompt(
    client: lms.Client,
    model: object,
    image_path: Path,
    prompt_text: str,
    model_cfg: ModelConfig,
) -> dict:
    """
    Envía una imagen + prompt al modelo usando el SDK oficial de LM Studio.
    El cliente y el modelo se reciben ya inicializados para reutilizarlos
    entre llamadas y evitar reconexiones innecesarias.
    """
    for attempt in range(1, model_cfg.max_retries + 1):
        try:
            t0 = time.time()

            # Preparar la imagen (el SDK la sube al servidor y devuelve un handle)
            image = client.files.prepare_image(str(image_path))

            image = None
            for attempt_img in range(3):
                try:
                    image = client.files.prepare_image(str(image_path))
                    if image is not None:
                        break
                except Exception as e:
                    logging.warning(f"Intento {attempt_img+1} para subir {image_path} falló: {e}")
                    time.sleep(1)

            if image is None:
                # Si no conseguimos subir la imagen, retornamos fallo inmediato
                elapsed = 0
                return {
                    "success": False,
                    "response": None,
                    "elapsed_sec": elapsed,
                    "attempt": 0,
                    "error": f"No se pudo preparar la imagen {image_path}"
                }
            chat = lms.Chat()


            chat.add_user_message(prompt_text, images=[image])

            # Llamada multimodal: el contenido del mensaje incluye texto + imagen
            response = model.respond(chat,
                config={"temperature": model_cfg.temperature, "maxTokens": model_cfg.max_tokens},
            )

            elapsed = round(time.time() - t0, 2)
            return {
                "success":     True,
                "response":    response.content.strip(),
                "elapsed_sec": elapsed,
                "attempt":     attempt,
                "error":       None,
            }

        except Exception as e:
            logging.warning(f"    Error en intento {attempt}: {e}")
            if attempt < model_cfg.max_retries:
                time.sleep(model_cfg.retry_delay)

    return {
        "success":     False,
        "response":    None,
        "elapsed_sec": 0,
        "attempt":     model_cfg.max_retries,
        "error":       f"Falló tras {model_cfg.max_retries} intentos",
    }


# ══════════════════════════════════════════════════════════════════
#  PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════

def process(
    start: int,
    end: int,
    images_dir: Path,
    pack_name: str,
    output_path: Path,
    model_cfg: ModelConfig,
) -> None:

    # Resolver imágenes del rango
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in extensions],
        key=lambda p: p.name,
    )
    total = len(all_images)

    if start < 0 or end >= total or start > end:
        logging.error(
            f"Rango [{start}..{end}] inválido. "
            f"Imágenes disponibles: 0–{total - 1} ({total} total)"
        )
        sys.exit(1)

    images  = all_images[start : end + 1]
    prompts = load_prompt_pack(pack_name)

    logging.info("=" * 60)
    logging.info(f"  Modelo   : {model_cfg.id}")
    logging.info(f"  Pack     : {pack_name}  ({len(prompts)} prompts)")
    logging.info(f"  Rango    : [{start}..{end}]  →  {len(images)} imágenes")
    logging.info(f"  Output   : {output_path}")
    logging.info(f"  Ops total: {len(images) * len(prompts)}")
    logging.info("=" * 60)

    results    = {}
    started_at = datetime.now().isoformat()

    # Una sola conexión y un solo handle de modelo para todo el proceso
    with lms.Client() as client:
        model = get_model(client, model_cfg)

        for offset, image_path in enumerate(images):
            global_idx = start + offset
            logging.info(f"\n[{offset + 1}/{len(images)}] {image_path.name}")

            item_results = {}
            for p in prompts:
                pid, label, text = p["id"], p["label"], p["prompt"]
                logging.info(f"  → [{pid}] {label}")
                result = run_prompt(client, model, image_path, text, model_cfg)
                status = "✓" if result["success"] else "✗"
                logging.info(f"    {status} {result['elapsed_sec']}s")
                item_results[pid] = {
                    "label":     label,
                    "prompt":    text,
                    "timestamp": datetime.now().isoformat(),
                    **result,
                }

            results[str(global_idx)] = {
                "index":      global_idx,
                "filename":   image_path.name,
                "path":       str(image_path),
                "size_bytes": image_path.stat().st_size,
                "prompts":    item_results,
            }

    n_ok  = sum(1 for it in results.values() for r in it["prompts"].values() if r["success"])
    n_err = sum(1 for it in results.values() for r in it["prompts"].values() if not r["success"])

    output = {
        "_meta": {
            "model":            model_cfg.id,
            "pack":             pack_name,
            "start":            start,
            "end":              end,
            "total_items":      len(images),
            "prompts_per_item": len(prompts),
            "total_ops":        len(images) * len(prompts),
            "successes":        n_ok,
            "failures":         n_err,
            "started_at":       started_at,
            "finished_at":      datetime.now().isoformat(),
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logging.info(f"\n {output_path}  ({n_ok}/{len(images) * len(prompts)} ok)")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2-VL vision runner — procesa un rango de imágenes"
    )
    parser.add_argument("--start",        type=int, required=True,              help="Índice inicial (inclusivo)")
    parser.add_argument("--end",          type=int, required=True,              help="Índice final (inclusivo)")
    parser.add_argument("--pack",         type=str, default="general",          help="Pack de prompts a usar")
    parser.add_argument("--input-dir",    type=str, default="inputs/images",    help="Directorio de imágenes")
    parser.add_argument("--output",       type=str, required=True,              help="Ruta del fichero de output JSON")
    parser.add_argument("--model-config", type=str, default="model_config.toml", help="Config interna del modelo")
    args = parser.parse_args()

    output_path = Path(args.output)
    setup_logging(output_path)

    model_cfg = load_model_config(args.model_config)
    process(args.start, args.end, Path(args.input_dir), args.pack, output_path, model_cfg)


if __name__ == "__main__":
    main()
