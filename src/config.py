from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def random_state(self) -> int:
        return int(self.raw.get("project", {}).get("random_state", 42))

    @property
    def target_name(self) -> str:
        return str(self.raw.get("target", {}).get("name", "target_return_next_1d"))

    @property
    def horizon(self) -> int:
        return int(self.raw.get("target", {}).get("horizon", 1))


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            raw = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML config requires PyYAML. Install requirements.txt or use a .json config file."
                ) from exc
            raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return ExperimentConfig(raw=raw, path=config_path)


def ensure_output_dirs(cfg: ExperimentConfig) -> dict[str, Path]:
    outputs = cfg.raw.get("outputs", {})
    dirs = {
        "processed": Path(outputs.get("processed_dir", "data/processed")),
        "models": Path(outputs.get("model_dir", "models")),
        "reports": Path(outputs.get("report_dir", "reports")),
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs
