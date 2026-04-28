"""Load antenna designs from YAML / JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..designs.base import AntennaDesign
from .specs import spec_from_dict


def load_design(path: str | Path) -> AntennaDesign:
    """Load a design from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data: Any = yaml.safe_load(text) if p.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a single design dict.")
    return spec_from_dict(data)
