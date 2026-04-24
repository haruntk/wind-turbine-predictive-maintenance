"""TimescaleDB client for feature vector ingestion.

Connects to a remote TimescaleDB instance (Docker on main PC)
and inserts 426-dimensional feature vectors as ``DOUBLE PRECISION[]``
arrays.  The database is the single source of truth — all downstream
processing (LSTM Autoencoder, Dashboard) reads from it.

Direct psycopg2 insertion — no MQTT required for this test setup.

Table schema (existing in Docker):
    time       TIMESTAMPTZ NOT NULL
    features   DOUBLE PRECISION[] NOT NULL  (CHECK array_length = 426)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from edge.utils.logging import get_logger

_LOG = get_logger(__name__)


class TimescaleDBClient:
    """Client for inserting feature vectors into TimescaleDB.

    Parameters
    ----------
    host:
        Database server hostname or IP.
    port:
        Database server port.
    dbname:
        Database name.
    user:
        Database user.
    password:
        Database password.
    """

    def __init__(
        self,
        host: str,
        port: int = 5432,
        dbname: str = "edge_db",
        user: str = "postgres",
        password: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._conn: Any = None

    # ----------------------------------------------------------
    # Connection lifecycle
    # ----------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to TimescaleDB."""
        import psycopg2

        self._conn = psycopg2.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
        )
        self._conn.autocommit = False
        _LOG.info(
            "Connected to TimescaleDB at %s:%d/%s",
            self._host,
            self._port,
            self._dbname,
        )

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            _LOG.info("Disconnected from TimescaleDB.")

    def __enter__(self) -> TimescaleDBClient:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        self.disconnect()

    # ----------------------------------------------------------
    # Schema management
    # ----------------------------------------------------------

    def ensure_schema(self) -> None:
        """Verify the feature_vectors table exists and is accessible.

        Does NOT create the table — assumes it was already created in
        the Docker TimescaleDB container with the correct schema:
            time       TIMESTAMPTZ NOT NULL
            features   DOUBLE PRECISION[] NOT NULL
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM feature_vectors LIMIT 1;"
                )
                count = cur.fetchone()[0]
            self._conn.commit()
            _LOG.info(
                "feature_vectors table verified (current rows: %d).", count
            )
        except Exception as exc:
            _LOG.error(
                "feature_vectors table not accessible: %s", exc
            )
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    # ----------------------------------------------------------
    # Data insertion
    # ----------------------------------------------------------

    def insert_feature_vector(
        self,
        timestamp: datetime,
        features: np.ndarray,
    ) -> None:
        """Insert a single 426-dim feature vector.

        Parameters
        ----------
        timestamp:
            UTC timestamp for this vector.
        features:
            1-D float array of length 426.
        """
        self._insert_rows([(timestamp, features)])

    def insert_feature_vectors_batch(
        self,
        records: list[tuple[datetime, np.ndarray]],
    ) -> None:
        """Batch-insert multiple feature vectors.

        Parameters
        ----------
        records:
            List of ``(timestamp, features)`` tuples.
        """
        if not records:
            return
        self._insert_rows(records)

    def _insert_rows(
        self,
        records: list[tuple[datetime, np.ndarray]],
    ) -> None:
        """Low-level batch insert using ``executemany``.

        Matches the existing table schema:
            INSERT INTO feature_vectors (time, features) VALUES (%s, %s)
            -- created_at auto-fills via DEFAULT CURRENT_TIMESTAMP
        """
        sql = (
            "INSERT INTO feature_vectors (time, features) "
            "VALUES (%s, %s)"
        )
        rows = [
            (ts, features.tolist())
            for ts, features in records
        ]

        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
        self._conn.commit()
        _LOG.debug("Inserted %d feature vector(s).", len(rows))

    # ----------------------------------------------------------
    # Factory
    # ----------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TimescaleDBClient:
        """Create a client instance from a pipeline config dict.

        Parameters
        ----------
        config:
            Full pipeline config with a ``database`` section.

        Returns
        -------
        TimescaleDBClient
        """
        db_cfg = config["database"]
        return cls(
            host=str(db_cfg["host"]),
            port=int(db_cfg.get("port", 5432)),
            dbname=str(db_cfg.get("dbname", "edge_db")),
            user=str(db_cfg.get("user", "postgres")),
            password=str(db_cfg.get("password", "")),
        )
