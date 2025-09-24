import json, os
from typing import Any, Dict

ARTIF_BASE = "models/lstm"

def ensure_dirs():
    os.makedirs(ARTIF_BASE, exist_ok=True)
    os.makedirs(os.path.join(ARTIF_BASE, "latest"), exist_ok=True)

def latest_dir() -> str:
    ensure_dirs()
    return os.path.join(ARTIF_BASE, "latest")

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
