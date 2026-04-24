"""Sliding-window segmentation for 1-D signals.

All sensor groups use 50 % overlap as required by the project report.
The caller computes ``window_size_samples`` from the group's
``window_size_seconds × sampling_rate_hz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowSegment:
    """A single signal window and its sample-level boundaries.

    Attributes
    ----------
    values:
        1-D array of signal samples within the window.
    start:
        Start sample index (inclusive) relative to the full signal.
    end:
        End sample index (exclusive) relative to the full signal.
    window_id:
        Sequential window counter (0-based).
    """

    values: np.ndarray
    start: int
    end: int
    window_id: int


def sliding_window(
    signal: np.ndarray,
    window_size_samples: int,
    overlap_ratio: float = 0.5,
    drop_last: bool = True,
) -> list[WindowSegment]:
    """Create sliding windows over a 1-D signal.

    Parameters
    ----------
    signal:
        1-D input signal array.
    window_size_samples:
        Number of samples per window.
    overlap_ratio:
        Fraction of window overlap (0.0 = no overlap, 0.5 = 50 %).
        All groups use 0.5 per project report §A.4.2.
    drop_last:
        If ``True``, discard the last incomplete window.  If ``False``,
        zero-pad it to ``window_size_samples``.

    Returns
    -------
    list[WindowSegment]
        Ordered list of signal windows.

    Raises
    ------
    ValueError
        If signal is not 1-D or window size is non-positive.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got shape {signal.shape}")
    if window_size_samples <= 0:
        raise ValueError("window_size_samples must be positive.")

    step = max(1, int(round(window_size_samples * (1.0 - overlap_ratio))))
    windows: list[WindowSegment] = []

    # Signal shorter than a single window
    if len(signal) < window_size_samples:
        if drop_last:
            return []
        padded = np.pad(signal, (0, window_size_samples - len(signal)))
        return [
            WindowSegment(
                values=padded,
                start=0,
                end=window_size_samples,
                window_id=0,
            )
        ]

    # Main sliding-window loop
    for wid, start in enumerate(
        range(0, len(signal) - window_size_samples + 1, step)
    ):
        end = start + window_size_samples
        windows.append(
            WindowSegment(
                values=signal[start:end],
                start=start,
                end=end,
                window_id=wid,
            )
        )

    # Optional tail window (zero-padded if needed)
    if not drop_last and windows:
        last_end = windows[-1].end
        if last_end < len(signal):
            tail = signal[-window_size_samples:]
            windows.append(
                WindowSegment(
                    values=tail,
                    start=max(0, len(signal) - window_size_samples),
                    end=len(signal),
                    window_id=len(windows),
                )
            )

    return windows
