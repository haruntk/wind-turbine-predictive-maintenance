"""MQTT publisher for feature vector streaming to the cloud broker.

Replaces direct TimescaleDB insertion on the Jetson edge device.
Feature vectors are serialised as JSON and published with QoS=1 so the
broker guarantees at-least-once delivery even across network interruptions.

Topic layout:
    turbines/{turbine_id}/features

Payload schema:
    {
        "turbine_id":      str,
        "timestamp":       ISO-8601 UTC string,
        "scenario_label":  str,
        "features":        list[float]   (length 426)
    }
"""

from __future__ import annotations

import msgpack
import threading
from datetime import datetime
from typing import Any

import numpy as np

from edge.utils.logging import get_logger

_LOG = get_logger(__name__)


class MQTTPublisher:
    """Publishes 426-dim feature vectors to a remote MQTT broker.

    Parameters
    ----------
    host:
        MQTT broker hostname or IP (Google Cloud Compute Engine).
    port:
        Broker port — 8883 for TLS, 1883 for plain.
    topic_prefix:
        Root of the topic tree (default ``"turbines"``).
    turbine_id:
        Identifies this edge device; used in the topic and payload.
    qos:
        MQTT QoS level (0, 1, or 2).  Use 1 for reliable delivery.
    username / password:
        Broker credentials.
    tls:
        Enable TLS.  Requires *ca_cert*.
    ca_cert:
        Path to the CA certificate file on the Jetson.
    """

    def __init__(
        self,
        host: str,
        port: int = 8883,
        topic_prefix: str = "turbines",
        turbine_id: str = "WT-001",
        qos: int = 1,
        username: str | None = None,
        password: str | None = None,
        tls: bool = True,
        ca_cert: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._topic = f"{topic_prefix}/{turbine_id}/features"
        self._turbine_id = turbine_id
        self._qos = qos
        self._username = username
        self._password = password
        self._tls = tls
        self._ca_cert = ca_cert
        self._client: Any = None
        self._connected = threading.Event()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the MQTT broker and start the network loop."""
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(client_id=self._turbine_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        if self._username:
            self._client.username_pw_set(self._username, self._password)

        if self._tls:
            self._client.tls_set(ca_certs=self._ca_cert)

        self._client.connect(self._host, self._port, keepalive=60)
        self._client.loop_start()

        if not self._connected.wait(timeout=10):
            raise ConnectionError(
                f"MQTT broker did not respond within 10 s ({self._host}:{self._port})"
            )

    def disconnect(self) -> None:
        """Stop the network loop and disconnect."""
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected.clear()
            _LOG.info("MQTT disconnected.")

    def __enter__(self) -> MQTTPublisher:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            self._connected.set()
            _LOG.info("MQTT connected to %s:%d (topic: %s)", self._host, self._port, self._topic)
        else:
            _LOG.error("MQTT connection refused, rc=%d", rc)

    def _on_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        self._connected.clear()
        if rc != 0:
            _LOG.warning("Unexpected MQTT disconnect (rc=%d). Paho will retry.", rc)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_feature_vectors_batch(
        self,
        records: list[tuple[datetime, np.ndarray, str]],
    ) -> None:
        """Publish a batch of feature vectors.

        Parameters
        ----------
        records:
            List of ``(timestamp, features, scenario_label)`` tuples.
        """
        if not records:
            return

        for ts, features, scenario_label in records:
            payload = msgpack.packb(
                {
                    "turbine_id": self._turbine_id,
                    "timestamp": ts.isoformat(),
                    "scenario_label": scenario_label,
                    "features": features.tolist(),
                },
                use_bin_type=True
            )
            self._client.publish(self._topic, payload, qos=self._qos)

        _LOG.debug("Published %d vector(s) to %s.", len(records), self._topic)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any], turbine_id: str = "WT-001") -> MQTTPublisher:
        """Build from the pipeline config dict (``mqtt`` section)."""
        cfg = config["mqtt"]
        return cls(
            host=str(cfg["host"]),
            port=int(cfg.get("port", 8883)),
            topic_prefix=str(cfg.get("topic_prefix", "turbines")),
            turbine_id=turbine_id,
            qos=int(cfg.get("qos", 1)),
            username=cfg.get("username") or None,
            password=cfg.get("password") or None,
            tls=bool(cfg.get("tls", True)),
            ca_cert=cfg.get("ca_cert") or None,
        )
