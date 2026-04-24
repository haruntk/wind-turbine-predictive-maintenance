"""YAML configuration loading with environment variable interpolation.

Loads the pipeline configuration from a YAML file, resolves ``${VAR}``
placeholders from environment variables, and provides validated access
to all configuration sections.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from edge.utils.logging import get_logger

_LOG = get_logger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


# ------------------------------------------------------------------
# YAML loading helpers
# ------------------------------------------------------------------

def _resolve_env_vars(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with env values."""
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                _LOG.warning(
                    "Environment variable '%s' is not set; "
                    "leaving placeholder unresolved.",
                    var_name,
                )
                return match.group(0)
            return env_val
        return _ENV_VAR_PATTERN.sub(_replacer, value)

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively.  All other values in *override*
    replace those in *base*.  The ``extends`` key is skipped.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file with ``extends`` inheritance and env vars.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Fully resolved configuration dictionary.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_yaml(path)

    # Resolve parent configs via `extends` key
    parents = raw.get("extends", [])
    if isinstance(parents, (str, Path)):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / Path(parent)).resolve()
        parent_config = load_config(parent_path)
        merged = deep_merge(merged, parent_config)

    merged = deep_merge(merged, raw)

    # Resolve environment variables
    merged = _resolve_env_vars(merged)

    merged["_config_path"] = str(path)
    _LOG.info("Loaded configuration from %s", path)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate that all required configuration sections are present.

    Raises
    ------
    ValueError
        If a required section or field is missing.
    """
    required_sections = [
        "turbine",
        "database",
        "sensor_groups",
        "windowing",
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: '{section}'"
            )

    # Validate database section
    db = config["database"]
    for field in ("host", "port", "dbname", "user", "password"):
        if field not in db:
            raise ValueError(
                f"Missing required database field: 'database.{field}'"
            )

    # Validate sensor groups
    groups = config["sensor_groups"]
    if not groups:
        raise ValueError("At least one sensor_group must be defined.")
    for name, group_cfg in groups.items():
        for field in ("sampling_rate_hz", "channels", "representation"):
            if field not in group_cfg:
                raise ValueError(
                    f"Sensor group '{name}' missing required field: '{field}'"
                )

    _LOG.debug("Configuration validation passed.")
