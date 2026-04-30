from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = PROJECT_ROOT / "Hybrid_grape" / "configuration.json"


@lru_cache(maxsize=1)
def load_configuration(path: Path = CONFIGURATION_PATH) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def khz_to_rad_per_us(value_khz: float) -> float:
    return 2.0 * math.pi * value_khz / 1000.0


def chi_rad_per_us(path: Path = CONFIGURATION_PATH) -> float:
    config = load_configuration(path)
    return khz_to_rad_per_us(config["chi_kHz"])


def self_kerr_rad_per_us(path: Path = CONFIGURATION_PATH) -> float:
    config = load_configuration(path)
    return khz_to_rad_per_us(config["self_Kerr_kHz"])
