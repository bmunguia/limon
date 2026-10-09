"""Marker and solution-field names that travel with GMF files.

GMF files store integer boundary references and unnamed fields. The loaders return
the names in memory (``markers``, ``labels``); GMF writers also save them in a
``<stem>.names.json`` sidecar next to the file, and GMF loaders read it back when
the caller supplies no names.
"""

import json
from pathlib import Path


class LimonIOError(RuntimeError):
    """A mesh or solution file could not be read or written."""


def sidecar_path(path: Path | str) -> Path:
    return Path(path).with_suffix('.names.json')


def read_names(path: Path | str) -> dict:
    """The sidecar of ``path`` as ``{'markers': {int: str}, 'labels': [str]}``, empty if absent."""
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        return {}
    data = json.loads(sidecar.read_text())
    if 'markers' in data:
        data['markers'] = {int(k): v for k, v in data['markers'].items()}
    return data


def update_names(path: Path | str, **entries) -> None:
    """Merge ``markers`` / ``labels`` into the sidecar of ``path``."""
    sidecar = sidecar_path(path)
    data = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    for key, value in entries.items():
        data[key] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else list(value)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(data, indent=2) + '\n')
