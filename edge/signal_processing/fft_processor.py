"""FFT computation with Hann windowing.

Provides the frequency-domain transformation stage of the pipeline:
Hann window → amplitude-corrected FFT → frequency / amplitude / power arrays.

Separated from feature calculation so the same FFT output can feed both
spectral statistics and band-energy computations.
"""

from __future__ import annotations

import numpy as np


def apply_hann_window(signal: np.ndarray) -> np.ndarray:
    """Apply a Hann window with amplitude correction.

    The correction factor ``2 / sum(w)`` preserves amplitude-spectrum
    magnitudes for peak and harmonic features (ISO 13373-3).

    Parameters
    ----------
    signal:
        1-D time-domain signal.

    Returns
    -------
    np.ndarray
        Windowed signal (same length as input).
    """
    hann = np.hanning(len(signal))
    hann_sum = np.sum(hann)
    correction = 2.0 / hann_sum if hann_sum > 0 else 1.0
    return signal * hann * correction


def compute_fft(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the one-sided FFT of a (Hann-windowed) signal.

    Parameters
    ----------
    signal:
        1-D signal — should already be Hann-windowed via
        :func:`apply_hann_window`.
    sampling_rate_hz:
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins (Hz), length ``N // 2 + 1``.
    amplitudes : np.ndarray
        Amplitude spectrum ``|FFT|``.
    power : np.ndarray
        Power spectrum ``|FFT|²``.

    Raises
    ------
    ValueError
        If the input signal is empty.
    """
    if len(signal) == 0:
        raise ValueError("Cannot compute FFT of an empty signal.")

    fft_values = np.fft.rfft(signal)
    amplitudes = np.abs(fft_values)
    power = np.square(amplitudes)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate_hz)

    return freqs, amplitudes, power
