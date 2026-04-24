"""Batch and single-file data loading utilities.

Provides helpers to iterate over directories of ``.mat`` files or load
a single file, yielding ``(path, signals)`` pairs for the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from edge.data_ingestion.mat_loader import load_mat_file
from edge.utils.logging import get_logger

_LOG = get_logger(__name__)


def load_single(path: str | Path) -> dict[str, np.ndarray]:
    """Load a single ``.mat`` file and return its channel signals.

    Parameters
    ----------
    path:
        Path to the ``.mat`` file.

    Returns
    -------
    dict[str, np.ndarray]
        Channel name → 1-D float32 array.
    """
    return load_mat_file(path)


def load_batch(
    directory: str | Path,
    pattern: str = "*.mat",
) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    """Iterate over all ``.mat`` files in *directory*, yielding signals.

    Files that cannot be loaded are logged as warnings and skipped —
    a single corrupt file never crashes the batch.

    Parameters
    ----------
    directory:
        Root directory to scan.  Searched recursively.
    pattern:
        Glob pattern for file matching (default ``"*.mat"``).

    Yields
    ------
    tuple[Path, dict[str, np.ndarray]]
        ``(file_path, channel_signals)`` for each successfully loaded file.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    mat_files = sorted(root.rglob(pattern))
    if not mat_files:
        _LOG.warning("No files matching '%s' found in %s", pattern, root)
        return

    _LOG.info(
        "Found %d file(s) matching '%s' in %s",
        len(mat_files),
        pattern,
        root,
    )

    for file_path in mat_files:
        try:
            signals = load_mat_file(file_path)
            _LOG.debug(
                "Loaded %d channels from %s",
                len(signals),
                file_path.name,
            )
            yield file_path, signals
        except Exception:
            _LOG.warning(
                "Skipping file %s — failed to load.",
                file_path,
                exc_info=True,
            )
