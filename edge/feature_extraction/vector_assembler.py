"""426-dimensional feature vector assembly with multi-rate alignment.

Orchestrates the full per-file processing flow:
  1. Preprocess each channel (detrend, optional bandstop)
  2. Window each group at its native rate with 50 % overlap
  3. Extract features (FFT spectral or time-domain)
  4. Align groups to a common 1 Hz timeline via zero-order hold
  5. Downsample to produce exactly 1 vector per second

The output is a sequence of ``np.ndarray`` of shape ``(426,)``
in deterministic column order:
  ``[bearing_96 | nacelle_45 | tower_195 | slow_90]``

Per project report §A.4.2.2:
  "5 saniyelik pencerelerden elde edilen özellikler ise yeni bir pencere
   oluşana kadar ara zaman adımlarında korunarak ortak sekans yapısına
   hizalanmıştır."
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

from edge.config.channel_groups import ChannelGroup
from edge.feature_extraction.feature_extractor import (
    extract_spectral_features,
    extract_time_features,
)
from edge.signal_processing.fft_processor import apply_hann_window, compute_fft
from edge.signal_processing.preprocessor import preprocess_signal
from edge.signal_processing.windowing import WindowSegment, sliding_window
from edge.utils.logging import get_logger

_LOG = get_logger(__name__)

# Expected total feature count
EXPECTED_FEATURE_DIM = 426


def _extract_windows_for_group(
    signals: dict[str, np.ndarray],
    group: ChannelGroup,
    config: dict[str, Any],
) -> list[list[WindowSegment]]:
    """Preprocess and window every channel in *group*.

    Returns a list (per channel) of WindowSegment lists.  Channels that
    are missing from *signals* are skipped with a warning.
    """
    overlap_ratio: float = config.get("windowing", {}).get(
        "overlap_ratio", 0.5
    )
    drop_last: bool = config.get("windowing", {}).get("drop_last", True)

    per_channel_windows: list[list[WindowSegment]] = []

    for channel_name in group.channels:
        raw = signals.get(channel_name)
        if raw is None:
            _LOG.warning(
                "Channel '%s' not found in file — skipping.",
                channel_name,
            )
            per_channel_windows.append([])
            continue

        processed = preprocess_signal(
            raw, config, sampling_rate_hz=group.sampling_rate_hz
        )
        windows = sliding_window(
            processed,
            window_size_samples=group.window_size_samples,
            overlap_ratio=overlap_ratio,
            drop_last=drop_last,
        )
        per_channel_windows.append(windows)

    return per_channel_windows


def _features_for_window(
    window: WindowSegment,
    group: ChannelGroup,
) -> OrderedDict[str, float]:
    """Extract features from a single window for a given group type."""
    if group.representation == "fft":
        windowed = apply_hann_window(window.values)
        freqs, amplitudes, power = compute_fft(
            windowed, group.sampling_rate_hz
        )
        return extract_spectral_features(
            freqs=freqs,
            amplitudes=amplitudes,
            power=power,
            band_edges_hz=list(group.band_energy_edges_hz),
            sampling_rate_hz=group.sampling_rate_hz,
            n_signal_samples=len(window.values),
        )
    else:
        return extract_time_features(window.values)


def _compute_group_feature_table(
    signals: dict[str, np.ndarray],
    group: ChannelGroup,
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Compute the feature table for an entire group.

    Returns a list of 1-D arrays, one per window position.  Each array
    has ``group.total_features`` elements (channels × features_per_ch).
    """
    per_channel_windows = _extract_windows_for_group(signals, group, config)

    # Determine number of window positions from channels that have data
    n_windows = 0
    for ch_windows in per_channel_windows:
        if ch_windows:
            n_windows = max(n_windows, len(ch_windows))

    if n_windows == 0:
        _LOG.warning(
            "Group '%s' produced no windows — returning zeros.",
            group.name,
        )
        return []

    n_feats = group.features_per_channel
    n_channels = len(group.channels)

    table: list[np.ndarray] = []
    for wid in range(n_windows):
        row = np.zeros(n_channels * n_feats, dtype=np.float64)
        for ch_idx, ch_windows in enumerate(per_channel_windows):
            if wid < len(ch_windows):
                feats = _features_for_window(ch_windows[wid], group)
                values = list(feats.values())
                offset = ch_idx * n_feats
                row[offset : offset + len(values)] = values
        table.append(row)

    return table


def _window_time_seconds(
    window_id: int,
    group: ChannelGroup,
    overlap_ratio: float,
) -> float:
    """Compute the centre time (in seconds) of window *window_id*."""
    step_samples = int(
        round(group.window_size_samples * (1.0 - overlap_ratio))
    )
    start_sample = window_id * step_samples
    centre_sample = start_sample + group.window_size_samples / 2.0
    return centre_sample / group.sampling_rate_hz


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def assemble_feature_vectors(
    signals: dict[str, np.ndarray],
    groups: list[ChannelGroup],
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Assemble 426-dim feature vectors at 1 Hz from raw channel signals.

    Processing steps:
      1. For each sensor group: preprocess → window (50 % overlap) →
         extract features → produce per-window feature rows.
      2. Compute a time axis in seconds for each group's windows.
      3. Determine full integer-second timeline from the shortest group.
      4. For each integer second *t*:
         - Fast groups (1 s window): take the window closest to *t*.
         - Slow groups (5 s window): zero-order hold — use the most recent
           window that starts at or before *t*.
      5. Concatenate group features → 426-dim vector.

    Parameters
    ----------
    signals:
        Channel-name → 1-D signal array mapping from a single .mat file.
    groups:
        Ordered list of :class:`ChannelGroup` objects.
    config:
        Full pipeline configuration dict.

    Returns
    -------
    list[np.ndarray]
        List of 1-D float64 arrays, each of length 426.
        One vector per output second.
    """
    overlap_ratio: float = config.get("windowing", {}).get(
        "overlap_ratio", 0.5
    )

    # --- Step 1: Compute per-group feature tables ---
    group_tables: list[list[np.ndarray]] = []
    group_times: list[list[float]] = []

    for group in groups:
        table = _compute_group_feature_table(signals, group, config)
        times = [
            _window_time_seconds(wid, group, overlap_ratio)
            for wid in range(len(table))
        ]
        group_tables.append(table)
        group_times.append(times)

        _LOG.debug(
            "Group '%s': %d windows, %d features/window, "
            "total span ≈ %.1f s",
            group.name,
            len(table),
            group.total_features if table else 0,
            times[-1] if times else 0.0,
        )

    # --- Step 2: Determine output timeline (integer seconds) ---
    if not any(group_tables):
        _LOG.error("No features computed for any group.")
        return []

    # Use the shortest signal duration to avoid extrapolation
    max_times = []
    for gt in group_times:
        if gt:
            max_times.append(gt[-1])
    if not max_times:
        return []

    duration_seconds = int(min(max_times))
    if duration_seconds <= 0:
        _LOG.warning("Signal too short for 1 Hz output.")
        return []

    output_seconds = list(range(duration_seconds))
    _LOG.info(
        "Assembling %d feature vectors at 1 Hz (%.1f s signal).",
        len(output_seconds),
        duration_seconds,
    )

    # --- Step 3: For each second, pick the appropriate window ---
    vectors: list[np.ndarray] = []
    # Cache last-used index per group for zero-order-hold efficiency
    last_idx = [0] * len(groups)

    for t in output_seconds:
        parts: list[np.ndarray] = []

        for g_idx, (group, table, times) in enumerate(
            zip(groups, group_tables, group_times)
        ):
            if not table:
                # Group has no data — fill with zeros
                parts.append(
                    np.zeros(group.total_features, dtype=np.float64)
                )
                continue

            # Find the window whose centre time is closest to (and ≤) t
            # This implements zero-order hold for slow groups and nearest
            # selection for fast groups.
            best_idx = last_idx[g_idx]
            for idx in range(best_idx, len(times)):
                if times[idx] <= t + 0.5:
                    best_idx = idx
                else:
                    break
            last_idx[g_idx] = best_idx
            parts.append(table[best_idx])

        vector = np.concatenate(parts)

        if len(vector) != EXPECTED_FEATURE_DIM:
            _LOG.error(
                "Feature vector dimension mismatch at t=%d: "
                "expected %d, got %d.",
                t,
                EXPECTED_FEATURE_DIM,
                len(vector),
            )

        vectors.append(vector)

    return vectors
