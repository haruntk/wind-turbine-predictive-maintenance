#!/bin/bash
# =============================================================
# Wind Turbine Edge Pipeline — Jetson Deployment Script
# =============================================================
# Usage:
#   1. Copy this repo to the Jetson
#   2. Copy .env.example to .env and fill in your values
#   3. SSH into Jetson
#   4. Run: bash scripts/deploy_jetson.sh
#
# Prerequisites:
#   - Python 3.8+ installed on Jetson
#   - Network access to the host running TimescaleDB
#   - .mat data files placed in ./data/ directory
#   - .env file with DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
# =============================================================

set -e

echo "============================================================"
echo " Wind Turbine Edge Pipeline — Jetson Setup"
echo "============================================================"

# Load environment variables from .env if present
if [ -f .env ]; then
    echo "[*] Loading environment from .env ..."
    set -a
    source .env
    set +a
else
    echo "⚠  No .env file found. Make sure DB_HOST, DB_NAME,"
    echo "   DB_USER, DB_PASSWORD are set in your environment."
fi

# Validate required environment variables
for var in DB_HOST DB_NAME DB_USER DB_PASSWORD; do
    if [ -z "${!var}" ]; then
        echo "✗ Required environment variable $var is not set!"
        exit 1
    fi
done
echo "✓ Environment variables loaded"

# 1. Install Python dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip3 install --user numpy scipy PyYAML psycopg2-binary h5py tqdm 2>/dev/null || \
pip3 install --user numpy scipy PyYAML h5py tqdm && \
    echo "Note: psycopg2-binary failed, trying source build..." && \
    sudo apt-get install -y libpq-dev python3-dev && \
    pip3 install --user psycopg2

echo "✓ Dependencies installed"

# 2. Test network connectivity to DB host
echo ""
echo "[2/4] Testing network connectivity to $DB_HOST ..."
if ping -c 1 -W 2 "$DB_HOST" > /dev/null 2>&1; then
    echo "✓ Host $DB_HOST is reachable"
else
    echo "✗ Cannot reach $DB_HOST — check your network connection!"
    exit 1
fi

# 3. Test database connection
echo ""
echo "[3/4] Testing TimescaleDB connection..."
python3 -c "
import os, psycopg2
try:
    conn = psycopg2.connect(
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', '5432')),
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
    )
    cur = conn.cursor()
    cur.execute('SELECT count(*) FROM feature_vectors')
    rows = cur.fetchone()[0]
    conn.close()
    print(f'✓ Connected to TimescaleDB — current rows: {rows}')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    exit(1)
"

# 4. Check data directory
echo ""
echo "[4/4] Checking data directory..."
MAT_COUNT=$(find ./data -name "*.mat" 2>/dev/null | wc -l)
if [ "$MAT_COUNT" -gt 0 ]; then
    echo "✓ Found $MAT_COUNT .mat file(s) in ./data/"
else
    echo "✗ No .mat files found in ./data/ — copy your data first!"
    exit 1
fi

echo ""
echo "============================================================"
echo " ✓ All checks passed! Ready to run the pipeline."
echo "============================================================"
echo ""
echo "Commands:"
echo "  Dry run (no DB):    python3 -m edge.main --dry-run"
echo "  Full run:           python3 -m edge.main"
echo ""
