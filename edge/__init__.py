"""Wind Turbine Edge Processing Pipeline.

Processes raw .mat sensor data through FFT and feature extraction,
producing 426-dimensional feature vectors at 1 Hz for TimescaleDB ingestion.
"""

__version__ = "0.1.0"
