"""MQTT → TimescaleDB bridge service.

Runs on Google Cloud (Cloud Run or Compute Engine).
Subscribes to all turbine feature topics and writes each vector into
the TimescaleDB (Cloud SQL for PostgreSQL) ``feature_vectors`` hypertable.

Topic pattern:  turbines/+/features
Payload schema:
    {
        "turbine_id":     str,
        "timestamp":      ISO-8601 UTC,
        "scenario_label": str,
        "features":       list[float]   (length 426)
    }

Environment variables (set in Cloud Run / .env):
    MQTT_HOST           GCE broker hostname or IP
    MQTT_PORT           default 8883
    MQTT_USER
    MQTT_PASSWORD
    MQTT_CA_CERT_PATH   path to CA cert (optional for plain connections)
    MQTT_TLS            "true" | "false"  (default true)

    DB_HOST             Cloud SQL public IP or Unix socket path
    DB_PORT             default 5432
    DB_NAME
    DB_USER
    DB_PASSWORD

Usage:
    python mqtt_bridge.py
"""

from __future__ import annotations

import msgpack
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOG = logging.getLogger("mqtt_bridge")

TOPIC = "turbines/+/features"
EXPECTED_DIM = 426


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

def _db_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    conn.autocommit = False
    _LOG.info("Connected to TimescaleDB at %s/%s", os.environ["DB_HOST"], os.environ["DB_NAME"])
    return conn


def _ensure_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                time           TIMESTAMPTZ        NOT NULL,
                turbine_id     TEXT               NOT NULL,
                scenario_label TEXT               NOT NULL DEFAULT 'unknown',
                features       DOUBLE PRECISION[] NOT NULL,
                created_at     TIMESTAMPTZ        DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Make it a TimescaleDB hypertable (idempotent)
        cur.execute("""
            SELECT create_hypertable(
                'feature_vectors', 'time',
                if_not_exists => TRUE
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_fv_turbine_time
                ON feature_vectors (turbine_id, time DESC);
        """)
    conn.commit()
    _LOG.info("Schema verified / created.")


def _insert(
    conn: psycopg2.extensions.connection,
    turbine_id: str,
    timestamp: datetime,
    scenario_label: str,
    features: list[float],
) -> None:
    sql = """
        INSERT INTO feature_vectors (time, turbine_id, scenario_label, features)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (timestamp, turbine_id, scenario_label, features))
    conn.commit()


# ------------------------------------------------------------------
# MQTT callbacks
# ------------------------------------------------------------------

class Bridge:
    def __init__(self) -> None:
        self._conn: psycopg2.extensions.connection | None = None
        self._client: mqtt.Client | None = None
        self._running = True

    def start(self) -> None:
        self._conn = _db_connect()
        _ensure_schema(self._conn)

        host = os.environ["MQTT_HOST"]
        port = int(os.environ.get("MQTT_PORT", 8883))
        tls = os.environ.get("MQTT_TLS", "true").lower() == "true"
        ca_cert = os.environ.get("MQTT_CA_CERT_PATH")

        self._client = mqtt.Client(client_id="bridge-01", clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        user = os.environ.get("MQTT_USER")
        password = os.environ.get("MQTT_PASSWORD")
        if user:
            self._client.username_pw_set(user, password)

        if tls:
            self._client.tls_set(ca_certs=ca_cert)

        self._client.connect(host, port, keepalive=60)
        _LOG.info("Connecting to MQTT broker %s:%d …", host, port)
        self._client.loop_start()

        try:
            while self._running:
                time.sleep(1)
        finally:
            self._client.loop_stop()
            self._client.disconnect()
            if self._conn:
                self._conn.close()
            _LOG.info("Bridge stopped.")

    def stop(self, *_) -> None:
        _LOG.info("Shutdown signal received.")
        self._running = False

    def _on_connect(self, client: mqtt.Client, userdata, flags, rc: int) -> None:
        if rc != 0:
            _LOG.error("MQTT connect failed, rc=%d", rc)
            return
        client.subscribe(TOPIC, qos=1)
        _LOG.info("Subscribed to %s", TOPIC)

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        if rc != 0:
            _LOG.warning("Unexpected disconnect (rc=%d). Paho will retry.", rc)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            data = msgpack.unpackb(msg.payload, raw=False)
        except Exception as e:
            _LOG.warning("Non-MsgPack payload on %s — ignored (%s).", msg.topic, e)
            return

        turbine_id: str = data.get("turbine_id", "unknown")
        scenario_label: str = data.get("scenario_label", "unknown")
        features: list[float] = data.get("features", [])

        if len(features) != EXPECTED_DIM:
            _LOG.warning(
                "Bad feature dim %d (expected %d) from %s — dropped.",
                len(features), EXPECTED_DIM, turbine_id,
            )
            return

        raw_ts: str = data.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(raw_ts).astimezone(timezone.utc)
        except ValueError:
            timestamp = datetime.now(tz=timezone.utc)

        try:
            _insert(self._conn, turbine_id, timestamp, scenario_label, features)
            _LOG.debug("Inserted vector from %s at %s.", turbine_id, timestamp.isoformat())
        except Exception:
            _LOG.error("DB insert failed — attempting reconnect.", exc_info=True)
            try:
                self._conn = _db_connect()
                _insert(self._conn, turbine_id, timestamp, scenario_label, features)
            except Exception:
                _LOG.error("Reconnect failed. Vector dropped.", exc_info=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    bridge = Bridge()
    signal.signal(signal.SIGINT, bridge.stop)
    signal.signal(signal.SIGTERM, bridge.stop)
    bridge.start()
