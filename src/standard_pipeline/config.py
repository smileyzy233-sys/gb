from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


DEFAULT_CONFIG_PATH = Path("configs/pipeline.toml")


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data: dict[str, Any]

    def section(self, *names: str) -> dict[str, Any]:
        current: Any = self.data
        for name in names:
            current = current.get(name, {})
        if not isinstance(current, dict):
            return {}
        return current

    def path(self, key: str) -> Path:
        value = self.section("paths").get(key)
        if not value:
            raise KeyError(f"Missing paths.{key} in pipeline config")
        return resolve_path(self.project_root, value)


def load_config(config_path: str | Path | None = None, project_root: str | Path | None = None) -> PipelineConfig:
    raw_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not raw_config_path.is_absolute():
        raw_config_path = Path.cwd() / raw_config_path
    raw_config_path = raw_config_path.resolve()

    with raw_config_path.open("rb") as file:
        data = tomllib.load(file)

    root = Path(project_root).resolve() if project_root else raw_config_path.parent.parent.resolve()
    return PipelineConfig(project_root=root, data=data)


def resolve_path(project_root: Path, value: str | Path | None) -> Path:
    if value is None:
        raise ValueError("Path value cannot be None")
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

