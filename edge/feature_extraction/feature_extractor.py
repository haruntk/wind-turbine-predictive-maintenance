"""Spectral and time-domain feature extraction.

Produces the per-channel feature vectors that are concatenated into the
426-dimensional output.

**FFT groups** (bearing, nacelle, tower):
    9 spectral statistics + N band energies per channel.

**Time group** (slow):
    15 time-domain statistics per channel.

Refactored from ``wind_turbine/src/fraunhofer_ad/features/``.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.stats import kurtosis, skew

from edge.utils.math_helpers import safe_divide


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _stable_spectral_moments(
    amplitudes: np.ndarray,
) -> tuple[float, float]:
    """Return stable skewness and excess kurtosis for a spectrum.

    For nearly constant spectra (flat noise floor), both values are
    returned as 0.0 to avoid numerical noise in ``scipy.stats``.
    Excess kurtosis (``fisher=True``) is the vibration-analysis standard:
    Gaussian baseline → 0, impulsive peak → large positive.
    """
    if np.allclose(amplitudes, amplitudes[0]):
        return 0.0, 0.0
    s = float(skew(amplitudes, bias=False))
    k = float(kurtosis(amplitudes, fisher=True, bias=False))
    return float(np.nan_to_num(s)), float(np.nan_to_num(k))


def _band_energies(
    freqs: np.ndarray,
    power: np.ndarray,
    band_edges_hz: list[float] | tuple[float, ...],
) -> OrderedDict[str, float]:
    """Compute normalised energy in each frequency band.

    Parameters
    ----------
    freqs:
        Frequency bins (Hz).
    power:
        Power spectrum (|FFT|²).
    band_edges_hz:
        Monotonically increasing list of band boundaries.
        ``len(band_edges) - 1`` energy values are produced.

    Returns
    -------
    OrderedDict[str, float]
        ``{"band_energy_<start>_<end>": ratio, ...}``
    """
    total = float(np.sum(power))
    energies: OrderedDict[str, float] = OrderedDict()
    for start, end in zip(band_edges_hz[:-1], band_edges_hz[1:]):
        mask = (freqs >= start) & (freqs < end)
        band_total = float(np.sum(power[mask]))
        key = f"band_energy_{int(start)}_{int(end)}"
        energies[key] = band_total / total if total > 0 else 0.0
    return energies


# ------------------------------------------------------------------
# Public API — Spectral features
# ------------------------------------------------------------------

def extract_spectral_features(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    power: np.ndarray,
    band_edges_hz: list[float] | tuple[float, ...],
    sampling_rate_hz: float,
    n_signal_samples: int,
) -> OrderedDict[str, float]:
    """Extract spectral (FFT-domain) features from pre-computed FFT output.

    Produces 9 spectral statistics plus ``len(band_edges) - 1`` normalised
    band energies.

    Parameters
    ----------
    freqs:
        Frequency bins array from :func:`compute_fft`.
    amplitudes:
        Amplitude spectrum from :func:`compute_fft`.
    power:
        Power spectrum from :func:`compute_fft`.
    band_edges_hz:
        Frequency band boundaries.
    sampling_rate_hz:
        Sampling rate (used for harmonic ratio Nyquist check).
    n_signal_samples:
        Original signal length (for Parseval normalisation).

    Returns
    -------
    OrderedDict[str, float]
        Feature name → value mapping with deterministic key ordering.
    """
    if len(freqs) == 0:
        raise ValueError("FFT arrays are empty — cannot extract features.")

    # Dominant frequency (skip DC bin at index 0)
    if len(amplitudes) > 1:
        dominant_index = int(np.argmax(amplitudes[1:])) + 1
    else:
        dominant_index = 0
    dominant_frequency = float(freqs[dominant_index])

    # Total and mean spectral power
    total_energy = float(np.sum(power))
    mean_spectral_power = (
        total_energy / n_signal_samples if n_signal_samples > 0 else 0.0
    )

    # Spectral centroid and bandwidth
    amp_sum = float(np.sum(amplitudes))
    spectral_centroid = safe_divide(
        float(np.sum(freqs * amplitudes)), amp_sum
    )
    spectral_bandwidth = float(
        np.sqrt(
            safe_divide(
                float(np.sum(((freqs - spectral_centroid) ** 2) * amplitudes)),
                amp_sum,
            )
        )
    )

    # Normalised spectral entropy [0, 1]
    normalised_power = (
        power / total_energy if total_energy > 0 else np.zeros_like(power)
    )
    raw_entropy = float(
        -np.sum(
            normalised_power
            * np.log2(np.clip(normalised_power, 1e-12, None))
        )
    )
    n_bins = len(power)
    max_entropy = float(np.log2(n_bins)) if n_bins > 1 else 1.0
    spectral_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    # Harmonic ratio (2nd harmonic / fundamental)
    harmonic_freq = dominant_frequency * 2.0
    if harmonic_freq > (sampling_rate_hz / 2.0):
        harmonic_ratio = 0.0
    else:
        harmonic_index = int(np.argmin(np.abs(freqs - harmonic_freq)))
        harmonic_ratio = safe_divide(
            float(amplitudes[harmonic_index]),
            float(amplitudes[dominant_index]),
        )

    # Spectral moments
    spectral_skewness, spectral_kurtosis = _stable_spectral_moments(
        amplitudes
    )

    # --- Assemble ordered feature dict (9 spectral stats) ---
    features: OrderedDict[str, float] = OrderedDict()
    features["dominant_frequency"] = dominant_frequency
    features["spectral_centroid"] = spectral_centroid
    features["spectral_bandwidth"] = spectral_bandwidth
    features["spectral_entropy"] = spectral_entropy
    features["spectral_kurtosis"] = spectral_kurtosis
    features["spectral_skewness"] = spectral_skewness
    features["mean_spectral_power"] = mean_spectral_power
    features["peak_frequency_amplitude"] = float(amplitudes[dominant_index])
    features["harmonic_ratio"] = harmonic_ratio

    # --- Band energies ---
    features.update(_band_energies(freqs, power, band_edges_hz))

    return features


# ------------------------------------------------------------------
# Public API — Time-domain features
# ------------------------------------------------------------------

def extract_time_features(window: np.ndarray) -> OrderedDict[str, float]:
    """Extract 15 time-domain statistical features from a signal window.

    Used for the *slow* sensor group (temperature, wind speed, wind vane)
    where FFT carries no meaningful information.

    Parameters
    ----------
    window:
        1-D signal window array.

    Returns
    -------
    OrderedDict[str, float]
        15 features with deterministic key ordering.
    """
    window = np.asarray(window, dtype=np.float64)
    absolute = np.abs(window)
    rms = float(np.sqrt(np.mean(np.square(window))))
    peak = float(np.max(absolute))
    mean_abs = float(np.mean(absolute))
    sqrt_abs_mean = float(
        np.mean(np.sqrt(np.clip(absolute, a_min=0.0, a_max=None)))
    )

    # Stable moments
    if np.allclose(window, window[0]):
        skewness, kurt = 0.0, 0.0
    else:
        skewness = float(np.nan_to_num(skew(window, bias=False)))
        kurt = float(
            np.nan_to_num(kurtosis(window, fisher=True, bias=False))
        )

    features: OrderedDict[str, float] = OrderedDict()
    features["mean"] = float(np.mean(window))
    features["std"] = float(np.std(window))
    features["variance"] = float(np.var(window))
    features["rms"] = rms
    features["peak"] = peak
    features["peak_to_peak"] = float(np.ptp(window))
    features["skewness"] = skewness
    features["kurtosis"] = kurt
    features["crest_factor"] = safe_divide(peak, rms + 1e-12)
    features["shape_factor"] = safe_divide(rms, mean_abs + 1e-12)
    features["impulse_factor"] = safe_divide(peak, mean_abs + 1e-12)
    features["clearance_factor"] = safe_divide(
        peak, sqrt_abs_mean**2 + 1e-12
    )
    features["median"] = float(np.median(window))
    features["iqr"] = float(
        np.percentile(window, 75) - np.percentile(window, 25)
    )
    features["zero_crossing_rate"] = float(
        np.mean(np.abs(np.diff(np.signbit(window))))
    )

    return features
