from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def project_root() -> Path:
    # fyp_trading/ -> project root
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_outputs_dir() -> Dict[str, Path]:
    root = project_root()
    outputs = ensure_dir(root / "outputs")
    return {
        "outputs": outputs,
        "models": ensure_dir(outputs / "models"),
        "plots": ensure_dir(outputs / "plots"),
        "reports": ensure_dir(outputs / "reports"),
    }


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_torch_device() -> Any:
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def save_json(path: Path, payload: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, default=to_jsonable)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float):
            return x
        return float(x)
    except Exception:
        return None


def as_float_dict(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {k: safe_float(v) for k, v in d.items()}


