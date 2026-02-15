"""API router for model discovery."""

import json
import os
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["models"])

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_DIRS = [os.path.expanduser("~/workspace/llama.cpp/models")]


def _load_model_dirs() -> list[str]:
    """Load model directories from config.json, with tilde expansion."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return [os.path.expanduser(d) for d in cfg.get("model_dirs", DEFAULT_DIRS)]
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_DIRS


def _get_model_info(path: Path) -> dict:
    """Extract model metadata from a .gguf file path."""
    stat = path.stat()
    size_gb = round(stat.st_size / (1024**3), 2)
    name = path.stem  # filename without extension

    # Try to extract useful info from the filename
    # Common patterns: ModelName-Size-Quant.gguf or ModelName.Quant.gguf
    parts = name.replace(".", "-").split("-")
    quant = ""
    for part in reversed(parts):
        # Match common quant patterns like Q4_K_M, q4_0, f16, etc.
        if (part.upper().startswith("Q") and len(part) > 1 and part[1].isdigit()) or \
           part.lower() in ("f16", "f32", "bf16", "iq4_nl"):
            quant = part
            break

    return {
        "name": name,
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_gb": size_gb,
        "quantization": quant,
        "modified": stat.st_mtime,
        "default_settings": _get_model_defaults(name, str(path)),
    }


# Hardcoded defaults for specific models
MODEL_DEFAULTS = {
    "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q8_0": {
        "ctx_size": 8192,
        "n_gpu_layers": 28,
        "max_tokens": 2048,
        "temperature": 0.7,
        "repeat_penalty": 1.10,
        "top_p": 0.9
    }
}

def _get_model_defaults(name: str, path: str) -> dict | None:
    """Return specific defaults if defined for this model."""
    # Check for exact name match or if key is part of filename
    for key, settings in MODEL_DEFAULTS.items():
        if key in name or key in path:
            return settings
    return None


@router.get("")
async def list_models():
    """List all available .gguf models from all configured directories."""
    all_models = []
    
    for directory in _load_model_dirs():
        models_path = Path(directory)
        if not models_path.exists():
            continue

        for gguf_file in sorted(models_path.glob("*.gguf")):
            # Skip vocabulary-only test files
            if gguf_file.name.startswith("ggml-vocab-"):
                continue
            all_models.append(_get_model_info(gguf_file))

    # Sort by name
    all_models.sort(key=lambda x: x["name"])

    return {"models": all_models, "model_dirs": _load_model_dirs()}
