from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
