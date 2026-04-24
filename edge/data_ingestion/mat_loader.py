"""MATLAB file loading and channel discovery.

Supports both classic MATLAB v5 files (via ``scipy.io.loadmat``) and
MATLAB v7.3 / HDF5 files (via ``h5py``).  All numeric signal arrays
are discovered automatically, sanitised for non-finite values, and
returned as a flat ``dict[str, np.ndarray]``.

Refactored from ``wind_turbine/src/fraunhofer_ad/data/mat_loader.py``
with training-specific code removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from edge.utils.logging import get_logger

_LOG = get_logger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _is_numeric_array(value: Any) -> bool:
    """Check whether *value* is a numeric NumPy array."""
    return isinstance(value, np.ndarray) and np.issubdtype(
        value.dtype, np.number
    )


def _to_float_array(value: np.ndarray, channel_name: str) -> np.ndarray:
    """Safely convert a numeric array to float32, replacing non-finites."""
    with np.errstate(invalid="ignore", over="ignore"):
        converted = np.array(value, dtype=np.float32, copy=True)

    if np.isfinite(converted).all():
        return converted

    nan_count = int(np.isnan(converted).sum())
    posinf = int(np.isposinf(converted).sum())
    neginf = int(np.isneginf(converted).sum())
    _LOG.warning(
        "Non-finite values in channel '%s' "
        "(nan=%d, +inf=%d, -inf=%d) — replacing with channel mean.",
        channel_name,
        nan_count,
        posinf,
        neginf,
    )
    channel_mean = float(np.nanmean(converted))
    if not np.isfinite(channel_mean):
        channel_mean = 0.0
    return np.where(np.isfinite(converted), converted, channel_mean)


def _extract_numeric_arrays(
    value: Any,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Recursively extract numeric arrays from nested MATLAB structures."""
    arrays: dict[str, np.ndarray] = {}

    if _is_numeric_array(value):
        array = np.asarray(value).squeeze()
        if array.ndim == 1:
            arrays[prefix] = _to_float_array(array, prefix)
        elif array.ndim == 2:
            # Split multi-column arrays into per-channel 1-D signals
            if array.shape[0] <= array.shape[1]:
                for idx in range(array.shape[0]):
                    ch = f"{prefix}_{idx}"
                    arrays[ch] = _to_float_array(array[idx, :], ch)
            else:
                for idx in range(array.shape[1]):
                    ch = f"{prefix}_{idx}"
                    arrays[ch] = _to_float_array(array[:, idx], ch)
        return arrays

    # Recurse into MATLAB struct fields
    fieldnames = getattr(value, "_fieldnames", None)
    if fieldnames:
        for field_name in fieldnames:
            field_value = getattr(value, field_name)
            arrays.update(
                _extract_numeric_arrays(field_value, f"{prefix}.{field_name}")
            )
    return arrays


def _load_mat_v73(resolved: Path) -> dict[str, np.ndarray]:
    """Load a MATLAB v7.3 (HDF5-based) ``.mat`` file using ``h5py``."""
    import h5py  # optional dependency — only needed for HDF5 files

    signals: dict[str, np.ndarray] = {}

    def _visit(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            arr = obj[()]
            if not np.issubdtype(arr.dtype, np.number):
                return
            arr = np.array(arr, dtype=np.float64)
            # HDF5/MATLAB v7.3 stores arrays in Fortran order — transpose
            if arr.ndim == 2:
                arr = arr.T
            arr = arr.squeeze()
            if arr.ndim == 1:
                signals[name] = _to_float_array(arr, name)
            elif arr.ndim == 2:
                if arr.shape[0] <= arr.shape[1]:
                    for idx in range(arr.shape[0]):
                        ch = f"{name}_{idx}"
                        signals[ch] = _to_float_array(arr[idx, :], ch)
                else:
                    for idx in range(arr.shape[1]):
                        ch = f"{name}_{idx}"
                        signals[ch] = _to_float_array(arr[:, idx], ch)
        except Exception:
            pass

    with h5py.File(resolved, "r") as f:
        f.visititems(_visit)
    return signals


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_mat_file(path: str | Path) -> dict[str, np.ndarray]:
    """Load a ``.mat`` file and return all discovered numeric channels.

    Parameters
    ----------
    path:
        Path to the ``.mat`` file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of channel name → 1-D float32 signal array.

    Raises
    ------
    RuntimeError
        If the file cannot be read.
    ValueError
        If no numeric signals are discovered.
    """
    resolved = Path(path).resolve()

    # Try scipy.io first (MATLAB v5)
    try:
        raw = loadmat(str(resolved), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # MATLAB v7.3 — fall back to h5py
        _LOG.info("Detected MATLAB v7.3 (HDF5) file, using h5py: %s", resolved)
        try:
            signals = _load_mat_v73(resolved)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot read v7.3 file {resolved}: h5py not installed. "
                "Install it with: pip install h5py"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MATLAB v7.3 file via h5py: {resolved}"
            ) from exc

        if not signals:
            raise ValueError(
                f"No numeric signals found in v7.3 file: {resolved}"
            )
        _LOG.debug(
            "Discovered %d channels in %s (v7.3)", len(signals), resolved
        )
        return signals
    except Exception as exc:
        raise RuntimeError(f"Failed to load MATLAB file: {resolved}") from exc

    # Extract signals from v5 file
    signals: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        signals.update(_extract_numeric_arrays(value, key))

    if not signals:
        raise ValueError(
            f"No numeric signals found in MATLAB file: {resolved}"
        )

    _LOG.debug("Discovered %d channels in %s", len(signals), resolved)
    return signals


def discover_channels(path: str | Path) -> list[str]:
    """Load a ``.mat`` file and return discovered channel names."""
    return list(load_mat_file(path).keys())
