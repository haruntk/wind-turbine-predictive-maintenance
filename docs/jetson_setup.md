# Jetson Deployment Guide

Complete step-by-step guide for deploying the edge processing pipeline
on an NVIDIA Jetson device.

---

## 1. Hardware Prerequisites

- **NVIDIA Jetson** (Nano / Xavier NX / Orin Nano)
- MicroSD card (≥ 32 GB) or NVMe SSD
- Ethernet cable or Wi-Fi adapter
- Power supply (5V/4A for Nano, USB-C for Orin)
- Host PC with SD card reader

---

## 2. OS Setup (JetPack)

1. Download **JetPack SDK** from [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)
2. Flash the SD card using **Balena Etcher** or NVIDIA SDK Manager
3. Boot the Jetson and complete initial setup (user, timezone, etc.)
4. Update packages:

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 3. Python Environment

JetPack ships with Python 3.8+. Create a virtual environment:

```bash
sudo apt install python3-venv python3-pip -y
python3 -m venv ~/edge-env
source ~/edge-env/bin/activate
pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
# NumPy and SciPy (ARM64 wheels available via pip)
pip install numpy>=1.26 scipy>=1.11

# YAML configuration
pip install PyYAML>=6.0

# PostgreSQL client
pip install psycopg2-binary>=2.9

# HDF5 support for MATLAB v7.3 files
pip install h5py>=3.9

# Progress bars
pip install tqdm>=4.66
```

> **Note**: If `psycopg2-binary` fails on ARM64, install the build
> dependencies and compile from source:
> ```bash
> sudo apt install libpq-dev python3-dev -y
> pip install psycopg2
> ```

---

## 5. Transfer Project Files

From the host PC, transfer the edge pipeline:

```bash
# Option A: rsync (recommended)
rsync -avz --exclude='.git' --exclude='data/raw' \
    ./wind-turbine-predictive-maintenance/ \
    jetson@<JETSON_IP>:~/wind-turbine/

# Option B: scp
scp -r ./wind-turbine-predictive-maintenance/ \
    jetson@<JETSON_IP>:~/wind-turbine/
```

Transfer `.mat` data files separately:

```bash
rsync -avz ./data/raw/ jetson@<JETSON_IP>:~/wind-turbine/data/raw/
```

---

## 6. Network Configuration

The Jetson must reach the PC running TimescaleDB.

### Static IP (recommended)

Edit `/etc/netplan/01-netcfg.yaml`:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      addresses: [192.168.1.200/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8]
```

Apply: `sudo netplan apply`

### Verify connectivity

```bash
ping 192.168.1.100  # PC running TimescaleDB
```

---

## 7. Configure the Pipeline

Edit `edge/config/config.yaml` on the Jetson:

```yaml
database:
  host: "192.168.1.100"   # PC IP
  port: 5432
  dbname: "wind_turbine"
  user: "jetson"
  password: "${DB_PASSWORD}"
```

Set the database password:

```bash
export DB_PASSWORD="your_secure_password"
```

---

## 8. Initialize the Database

On the **PC** (not Jetson), create the database and run the schema:

```bash
# Create database
createdb -U postgres wind_turbine

# Run schema
psql -U postgres -d wind_turbine -f scripts/init_db.sql
```

Create the Jetson user:

```sql
CREATE USER jetson WITH PASSWORD 'your_secure_password';
GRANT INSERT ON feature_vectors TO jetson;
GRANT USAGE ON SCHEMA public TO jetson;
```

---

## 9. Run the Pipeline

```bash
cd ~/wind-turbine
source ~/edge-env/bin/activate

# Dry run (no DB insert — verify feature extraction)
python -m edge.main --config edge/config/config.yaml --dry-run

# Full run (insert into TimescaleDB)
python -m edge.main --config edge/config/config.yaml --mode batch
```

---

## 10. Auto-Start on Boot (systemd)

Create `/etc/systemd/system/edge-pipeline.service`:

```ini
[Unit]
Description=Wind Turbine Edge Processing Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/wind-turbine
Environment="DB_PASSWORD=your_secure_password"
ExecStart=/home/jetson/edge-env/bin/python -m edge.main \
    --config edge/config/config.yaml --mode batch
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable edge-pipeline
sudo systemctl start edge-pipeline
sudo journalctl -u edge-pipeline -f  # View logs
```

---

## 11. Monitoring

Check pipeline logs:

```bash
tail -f ~/wind-turbine/logs/edge_pipeline.log
```

Verify data in TimescaleDB (on PC):

```sql
SELECT count(*), min(time), max(time)
FROM feature_vectors
WHERE turbine_id = 'WT-001';

SELECT array_length(features, 1) FROM feature_vectors LIMIT 1;
-- Should return 426
```

---

## 12. Troubleshooting

| Issue | Solution |
|---|---|
| `psycopg2` install fails | `sudo apt install libpq-dev python3-dev` then `pip install psycopg2` |
| Cannot reach TimescaleDB | Check PC firewall, PostgreSQL `listen_addresses = '*'`, `pg_hba.conf` allows Jetson IP |
| Out of memory on Jetson | Reduce `batch_size` in config or process fewer files |
| Channel not found warning | Verify `.mat` file channel names match config |
