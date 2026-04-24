# Local Testing Guide

How to test the edge processing pipeline on Windows or Linux before
deploying to the Jetson device.

---

## 1. Prerequisites

- Python 3.8+
- Git
- (Optional) PostgreSQL + TimescaleDB for full integration testing

---

## 2. Setup

```bash
cd wind-turbine-predictive-maintenance

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r edge/requirements.txt
```

---

## 3. Generate Test Data

```bash
python scripts/generate_test_data.py
# Creates: data/raw/Healthy/test_synthetic.mat (28 channels, 30s)

# Custom duration
python scripts/generate_test_data.py --duration 60 --output data/raw/Healthy/test_long.mat
```

---

## 4. Dry Run (No Database)

```bash
python -m edge.main --config edge/config/config.yaml --dry-run
```

Expected output:

```
... | INFO     | edge.main | ============================================================
... | INFO     | edge.main | Wind Turbine Edge Processing Pipeline
... | INFO     | edge.main | ============================================================
... | INFO     | edge.main | Mode: batch | Dry run: True
... | INFO     | edge.main | Sensor groups: ['bearing', 'nacelle', 'tower_tach', 'slow'] → 426 total features
... | INFO     | edge.main | Processing test_synthetic.mat (28 channels, scenario=healthy)
... | INFO     | edge.main | Produced 25 × 426 feature vectors for test_synthetic.mat.
... | INFO     | edge.main | [DRY RUN] Skipping database insert.
... | INFO     | edge.main | Pipeline complete.
... | INFO     | edge.main | Files: 1 | Vectors: 25 | Time: 2.3 s
```

---

## 5. Full Integration Test (with TimescaleDB)

### 5.1 Start TimescaleDB (Docker)

```bash
docker run -d --name timescaledb \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=password \
    timescale/timescaledb:latest-pg16
```

### 5.2 Create Database and Schema

```bash
docker exec -i timescaledb psql -U postgres -c "CREATE DATABASE wind_turbine;"
docker exec -i timescaledb psql -U postgres -d wind_turbine < scripts/init_db.sql
```

### 5.3 Update Config

Edit `edge/config/config.yaml`:

```yaml
database:
  host: "${DB_HOST}"
  port: 5432
  dbname: "${DB_NAME}"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
```

Set environment variables before running:

```bash
export DB_HOST=localhost
export DB_NAME=edge_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### 5.4 Run Pipeline

```bash
python -m edge.main --config edge/config/config.yaml --mode batch
```

### 5.5 Verify

```bash
docker exec -i timescaledb psql -U postgres -d wind_turbine \
    -c "SELECT count(*), array_length(features, 1) FROM feature_vectors;"
```

Expected: row count > 0, array_length = 426.

---

## 6. Running Unit Checks

Quick validation of individual modules:

```python
# Test feature dimension
from edge.config.channel_groups import load_groups_from_config
from edge.config.settings import load_config

config = load_config("edge/config/config.yaml")
groups = load_groups_from_config(config)
total = sum(g.total_features for g in groups)
assert total == 426, f"Expected 426, got {total}"
print(f"✓ Feature dimension: {total}")

# Test windowing
from edge.signal_processing.windowing import sliding_window
import numpy as np
sig = np.random.randn(74000)  # 1 second @ 74 kHz
windows = sliding_window(sig, window_size_samples=74000, overlap_ratio=0.5)
print(f"✓ Windows: {len(windows)} (should be 1 for 1s signal)")

# Test FFT processor
from edge.signal_processing.fft_processor import apply_hann_window, compute_fft
windowed = apply_hann_window(sig)
freqs, amps, power = compute_fft(windowed, 74000)
print(f"✓ FFT bins: {len(freqs)}")
```

---

## 7. Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: edge` | Run from the repo root: `python -m edge.main ...` |
| `FileNotFoundError: config` | Use absolute path or run from repo root |
| `No files matching *.mat` | Generate test data first: `python scripts/generate_test_data.py` |
| `psycopg2` not installed | `pip install psycopg2-binary` |
| DB connection refused | Check TimescaleDB is running and config host/port are correct |
