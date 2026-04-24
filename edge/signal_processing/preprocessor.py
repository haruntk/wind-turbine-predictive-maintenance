"""Signal preprocessing: DC removal and optional bandstop filtering.

Applied per-channel before windowing and FFT.  Refactored from
``wind_turbine/src/fraunhofer_ad/data/signal_preprocessing.py`` with the
broken ``resample_placeholder`` removed.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from edge.utils.logging import get_logger

_LOG = get_logger(__name__)


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset from a signal via mean subtraction.

    Parameters
    ----------
    signal:
        1-D raw signal array.

    Returns
    -------
    np.ndarray
        Signal with zero mean.
    """
    return signal - np.mean(signal)


def bandstop_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-stop (notch) filter.

    Removes frequency content in ``[low_hz, high_hz]``.  Useful for
    eliminating power-line interference (e.g. 50 Hz) or known electrical
    noise (e.g. the 50 kHz noise documented in Fraunhofer LBF data).

    Parameters
    ----------
    signal:
        1-D input signal.
    low_hz:
        Lower cut-off frequency in Hz.
    high_hz:
        Upper cut-off frequency in Hz.
    sampling_rate_hz:
        Sampling frequency of the signal in Hz.
    order:
        Butterworth filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * sampling_rate_hz
    low_norm = float(np.clip(low_hz / nyq, 1e-6, 1.0 - 1e-6))
    high_norm = float(np.clip(high_hz / nyq, 1e-6, 1.0 - 1e-6))

    if low_norm >= high_norm:
        raise ValueError(
            f"low_hz ({low_hz}) must be < high_hz ({high_hz}) "
            f"for sampling_rate_hz={sampling_rate_hz}."
        )

    sos = butter(order, [low_norm, high_norm], btype="bandstop", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    config: dict[str, Any],
    sampling_rate_hz: float,
) -> np.ndarray:
    """Apply the configured preprocessing pipeline to a 1-D signal.

    The pipeline is:
      1. Detrend (DC removal) — if enabled
      2. Bandstop filter — if enabled

    Parameters
    ----------
    signal:
        Raw 1-D signal array.
    config:
        Full pipeline configuration dict (must contain ``preprocessing``).
    sampling_rate_hz:
        Native sampling rate of this channel.

    Returns
    -------
    np.ndarray
        Preprocessed signal (float32).
    """
    processed = np.asarray(signal, dtype=np.float32).copy()
    pp_cfg = config.get("preprocessing", {})

    # Step 1: Detrend
    if pp_cfg.get("detrend", False):
        processed = detrend_signal(processed)

    # Step 2: Bandstop filter
    bs_cfg = pp_cfg.get("bandstop", {})
    if bs_cfg.get("enabled", False):
        low_hz = bs_cfg.get("low_hz")
        high_hz = bs_cfg.get("high_hz")
        if low_hz is not None and high_hz is not None:
            processed = bandstop_filter(
                processed,
                low_hz=float(low_hz),
                high_hz=float(high_hz),
                sampling_rate_hz=sampling_rate_hz,
            )
        else:
            warnings.warn(
                "Bandstop filter enabled but low_hz/high_hz not set; skipping.",
                stacklevel=2,
            )

    return processed
