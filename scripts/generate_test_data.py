"""Generate a synthetic .mat file for local pipeline testing.

Creates a small test file with all 28 channels at correct sampling rates,
allowing the pipeline to be validated without the real Fraunhofer LBF dataset.

Usage::

    python scripts/generate_test_data.py
    python scripts/generate_test_data.py --output data/raw/Healthy/test_001.mat --duration 10

The generated signals contain:
  - Sinusoidal components with known frequencies (for FFT verification)
  - Gaussian noise (realistic baseline)
  - All 28 channel names matching the Fraunhofer LBF dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


# Channel definitions: (name, sampling_rate_hz)
CHANNELS = {
    # Bearing @ 74 kHz
    "brng_f_x": 74000, "brng_f_y": 74000, "brng_f_z": 74000,
    "brng_r_x": 74000, "brng_r_y": 74000, "brng_r_z": 74000,
    # Nacelle @ 37 kHz
    "Nacl_x": 37000, "Nacl_y": 37000, "Nacl_z": 37000,
    # Tower / Tach @ 2960 Hz
    "tach": 2960,
    "bot_f_x": 2960, "bot_f_y": 2960, "bot_f_z": 2960,
    "bot_r_x": 2960, "bot_r_y": 2960, "bot_r_z": 2960,
    "top_l_x": 2960, "top_l_y": 2960, "top_l_z": 2960,
    "top_r_x": 2960, "top_r_y": 2960, "top_r_z": 2960,
    # Slow @ 1480 Hz
    "tmp_amb": 1480, "tmp_brng_f": 1480, "tmp_brng_r": 1480,
    "anm_mst": 1480, "anm_roof": 1480, "van": 1480,
}


def generate_synthetic_signal(
    fs: float,
    duration: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic vibration-like signal.

    Combines:
    - 10 Hz fundamental (turbine rotation)
    - 120 Hz component (bearing signature)
    - Gaussian noise
    """
    rng = np.random.RandomState(seed)
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    signal = (
        1.0 * np.sin(2 * np.pi * 10.0 * t)        # 10 Hz fundamental
        + 0.3 * np.sin(2 * np.pi * 120.0 * t)      # Bearing harmonic
        + 0.5 * np.sin(2 * np.pi * 0.5 * t)        # Low-freq modulation
        + 0.1 * rng.randn(n_samples)                # Noise
    )
    return signal.astype(np.float32)


def generate_test_mat(
    output_path: str | Path,
    duration: float = 30.0,
) -> None:
    """Generate a test .mat file with all 28 channels.

    Parameters
    ----------
    output_path:
        Where to save the .mat file.
    duration:
        Signal duration in seconds (default 30 s).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for idx, (name, fs) in enumerate(CHANNELS.items()):
        data[name] = generate_synthetic_signal(
            fs=fs, duration=duration, seed=42 + idx
        )

    savemat(str(output), data)
    total_samples = sum(len(v) for v in data.values())
    print(
        f"Generated {output} — "
        f"{len(data)} channels, {duration:.0f}s, "
        f"{total_samples:,} total samples"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic .mat test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/Healthy/test_synthetic.mat",
        help="Output path for the .mat file",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Signal duration in seconds (default: 30)",
    )
    args = parser.parse_args()
    generate_test_mat(args.output, args.duration)


if __name__ == "__main__":
    main()
