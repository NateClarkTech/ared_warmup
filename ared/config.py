"""Load one JSON config by file name, or by a number read from stdin."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from comparison_distance import STRATEGIES as COMPARISON_DISTANCE
from neighborhood_merge import STRATEGIES as NEIGHBORHOOD_MERGE
from singleton_merge import STRATEGIES as SINGLETON_MERGE
from smart_forgetting import STRATEGIES as SMART_FORGETTING

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(Exception):
    """The requested config was not one of the choices, so nothing ran."""


@dataclass(frozen=True)
class Config:
    name: str
    warmup: int
    kappa: float
    buffer_size: int
    k_neighbors: int
    comparison_distance: str
    smart_forgetting: str
    neighborhood_merge: str
    singleton_merge: str
    stream_seed: int
    stream_n: int
    stream_dim: int


def available_configs() -> list[Path]:
    return sorted(path for path in CONFIG_DIR.glob("*.json") if path.is_file())


def load_config(name: str | None, stdin=None) -> Config:
    """Load ``name``, or print the numbered configs and read one number."""
    if name is None or str(name).strip() == "":
        path = _prompt(sys.stdin if stdin is None else stdin)
    else:
        path = resolve_config_path(str(name))
    return read_config(path)


def resolve_config_path(name: str) -> Path:
    raw = Path(name)
    options = [raw]
    if raw.suffix.lower() != ".json":
        options.append(raw.with_suffix(".json"))

    has_directory = raw.parent != Path(".")
    candidates: list[Path] = []
    for item in options:
        if item.is_absolute():
            candidates.append(item)
        elif has_directory:
            candidates.append(Path.cwd() / item)
            candidates.append(CONFIG_DIR.parent / item)
        else:
            candidates.append(CONFIG_DIR / item.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"Unknown config {name!r}. Choose a JSON file from {CONFIG_DIR}."
    )


def read_config(path: Path) -> Config:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a JSON object")

    warmup = _integer(data, "warmup", path.name)
    stream_n = _integer(data, "stream_n", path.name)
    if warmup < 1:
        raise ConfigError(f"warmup in {path.name} must be at least 1")
    if stream_n < warmup:
        raise ConfigError(
            f"stream_n ({stream_n}) is shorter than warmup ({warmup}) in {path.name}"
        )
    buffer_size = _integer(data, "buffer_size", path.name)
    k_neighbors = _integer(data, "k_neighbors", path.name)
    stream_dim = _integer(data, "stream_dim", path.name)
    if buffer_size < 1 or k_neighbors < 1 or stream_dim < 1:
        raise ConfigError(f"{path.name} has a non-positive buffer, k, or dimension")

    return Config(
        name=path.name,
        warmup=warmup,
        kappa=_number(data, "kappa", path.name),
        buffer_size=buffer_size,
        k_neighbors=k_neighbors,
        comparison_distance=_choice(data, "comparison_distance", COMPARISON_DISTANCE, path.name),
        smart_forgetting=_choice(data, "smart_forgetting", SMART_FORGETTING, path.name),
        neighborhood_merge=_choice(data, "neighborhood_merge", NEIGHBORHOOD_MERGE, path.name),
        singleton_merge=_choice(data, "singleton_merge", SINGLETON_MERGE, path.name),
        stream_seed=_integer(data, "stream_seed", path.name),
        stream_n=stream_n,
        stream_dim=stream_dim,
    )


def _prompt(stdin) -> Path:
    files = available_configs()
    if not files:
        raise ConfigError(f"No JSON configs in {CONFIG_DIR}")
    print("Available configs:")
    for number, path in enumerate(files, start=1):
        print(f"{number}. {path.name}")
    print("Select a config number:")
    line = stdin.readline()
    if line == "":
        raise ConfigError("No config number was entered.")
    text = line.strip()
    if not text.isdigit():
        raise ConfigError(
            f"{text!r} is not a config number. Expected a number from 1 to {len(files)}."
        )
    number = int(text)
    if number < 1 or number > len(files):
        raise ConfigError(
            f"Config number {number} is out of range. Expected a number from 1 to {len(files)}."
        )
    return files[number - 1]


def _integer(data: dict, key: str, filename: str) -> int:
    if key not in data:
        raise ConfigError(f"{filename} is missing {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} in {filename} must be an integer")
    return value


def _number(data: dict, key: str, filename: str) -> float:
    if key not in data:
        raise ConfigError(f"{filename} is missing {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} in {filename} must be a number")
    return float(value)


def _choice(data: dict, key: str, strategies: dict, filename: str) -> str:
    if key not in data:
        raise ConfigError(f"{filename} is missing {key}")
    value = data[key]
    if not isinstance(value, str) or value not in strategies:
        known = ", ".join(sorted(strategies))
        raise ConfigError(f"Unknown {key} {value!r} in {filename}. Expected one of: {known}")
    return value
