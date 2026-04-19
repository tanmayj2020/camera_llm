"""Persistence layer — PostgreSQL with pgvector for production, SQLite fallback for dev.

PostgreSQL provides:
  - Concurrent writes (vs SQLite single-writer lock)
  - pgvector for embedding similarity search
  - Connection pooling for multi-tenant workloads
  - Full-text search via tsvector
"""

import json
import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_PATH = os.environ.get("VISIONBRAIN_DB", "./visionbrain.db")


class PersistenceStore:
    """Multi-backend persistence: PostgreSQL (production) or SQLite (dev).

    Automatically selects backend based on DATABASE_URL environment variable.
    """

    def __init__(self, db_path: str = DB_PATH, database_url: str = DATABASE_URL):
        self._pg_pool = None
        self._sqlite_conn = None
        self._backend = "sqlite"

        if database_url and database_url.startswith("postgres"):
            try:
                import psycopg_pool
                self._pg_pool = psycopg_pool.ConnectionPool(
                    database_url, min_size=2, max_size=10
                )
                self._backend = "postgresql"
                self._ensure_tables_pg()
                logger.info("PersistenceStore: PostgreSQL + pgvector")
                return
            except Exception as e:
                logger.warning("PostgreSQL init failed (%s), falling back to SQLite", e)

        self._sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._sqlite_conn.row_factory = sqlite3.Row
        self._ensure_tables_sqlite()
        logger.info("PersistenceStore: SQLite at %s", db_path)

    def _ensure_tables_pg(self):
        with self._pg_pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    event_id TEXT PRIMARY KEY,
                    timestamp DOUBLE PRECISION,
                    camera_id TEXT,
                    severity TEXT,
                    anomaly_type TEXT,
                    description TEXT,
                    causal_explanation TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acked_by TEXT,
                    acked_at DOUBLE PRECISION,
                    embedding vector(1152)
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
                CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS investigations (
                    investigation_id TEXT PRIMARY KEY,
                    entity_id TEXT,
                    timestamp DOUBLE PRECISION,
                    report_json JSONB
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT,
                    rtsp_url TEXT,
                    site_x DOUBLE PRECISION DEFAULT 0,
                    site_y DOUBLE PRECISION DEFAULT 0,
                    rotation_deg DOUBLE PRECISION DEFAULT 0,
                    fov_deg DOUBLE PRECISION DEFAULT 60,
                    enabled BOOLEAN DEFAULT TRUE,
                    tenant_id TEXT,
                    tags JSONB DEFAULT '[]',
                    created_at DOUBLE PRECISION
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_configs (
                    webhook_id TEXT PRIMARY KEY,
                    config_json JSONB
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    user_id TEXT,
                    access_level TEXT,
                    cameras_json JSONB,
                    created_at DOUBLE PRECISION
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS report_schedules (
                    schedule_id TEXT PRIMARY KEY,
                    config_json JSONB
                );
            """)
            conn.commit()

    def _ensure_tables_sqlite(self):
        c = self._sqlite_conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                event_id TEXT PRIMARY KEY,
                timestamp REAL,
                camera_id TEXT,
                severity TEXT,
                anomaly_type TEXT,
                description TEXT,
                causal_explanation TEXT,
                acknowledged INTEGER DEFAULT 0,
                acked_by TEXT,
                acked_at REAL
            );
            CREATE TABLE IF NOT EXISTS investigations (
                investigation_id TEXT PRIMARY KEY,
                entity_id TEXT,
                timestamp REAL,
                report_json TEXT
            );
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                name TEXT,
                rtsp_url TEXT,
                site_x REAL,
                site_y REAL,
                rotation_deg REAL DEFAULT 0,
                fov_deg REAL DEFAULT 60,
                enabled INTEGER DEFAULT 1,
                tenant_id TEXT,
                tags TEXT,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS webhook_configs (
                webhook_id TEXT PRIMARY KEY,
                config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                tenant_id TEXT,
                user_id TEXT,
                access_level TEXT,
                cameras_json TEXT,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS report_schedules (
                schedule_id TEXT PRIMARY KEY,
                config_json TEXT
            );
        """)
        self._sqlite_conn.commit()

    # ---- Unified query helpers ----

    def _execute(self, query: str, params: tuple = ()):
        if self._backend == "postgresql":
            # Convert ? placeholders to %s for psycopg
            pg_query = query.replace("?", "%s")
            with self._pg_pool.connection() as conn:
                conn.execute(pg_query, params)
                conn.commit()
        else:
            self._sqlite_conn.execute(query, params)
            self._sqlite_conn.commit()

    def _fetchall(self, query: str, params: tuple = ()) -> list[dict]:
        if self._backend == "postgresql":
            pg_query = query.replace("?", "%s")
            with self._pg_pool.connection() as conn:
                cur = conn.execute(pg_query, params)
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        else:
            return [dict(r) for r in self._sqlite_conn.execute(query, params).fetchall()]

    def _fetchone(self, query: str, params: tuple = ()) -> dict | None:
        if self._backend == "postgresql":
            pg_query = query.replace("?", "%s")
            with self._pg_pool.connection() as conn:
                cur = conn.execute(pg_query, params)
                row = cur.fetchone()
                if not row:
                    return None
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
        else:
            row = self._sqlite_conn.execute(query, params).fetchone()
            return dict(row) if row else None

    # --- Alerts ---

    def save_alert(self, alert: dict):
        self._execute(
            "INSERT INTO alerts (event_id, timestamp, camera_id, severity, anomaly_type, description, causal_explanation) VALUES (?,?,?,?,?,?,?) ON CONFLICT (event_id) DO UPDATE SET timestamp=EXCLUDED.timestamp, severity=EXCLUDED.severity, description=EXCLUDED.description",
            (alert["event_id"], alert.get("timestamp", time.time()), alert.get("camera_id"),
             alert.get("severity"), alert.get("anomaly_type"), alert.get("description"),
             alert.get("causal_explanation")),
        )

    def get_alerts(self, camera_id: str | None = None, severity: str | None = None, limit: int = 100) -> list[dict]:
        q, params = "SELECT * FROM alerts WHERE 1=1", []
        if camera_id:
            q += " AND camera_id=?"
            params.append(camera_id)
        if severity:
            q += " AND severity=?"
            params.append(severity)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(q, tuple(params))

    def ack_alert(self, event_id: str, user_id: str):
        self._execute(
            "UPDATE alerts SET acknowledged=1, acked_by=?, acked_at=? WHERE event_id=?",
            (user_id, time.time(), event_id),
        )

    # --- Investigations ---

    def save_investigation(self, investigation_id: str, entity_id: str, report: dict):
        self._execute(
            "INSERT INTO investigations (investigation_id, entity_id, timestamp, report_json) VALUES (?,?,?,?) ON CONFLICT (investigation_id) DO UPDATE SET report_json=EXCLUDED.report_json",
            (investigation_id, entity_id, time.time(), json.dumps(report, default=str)),
        )

    def get_investigations(self, limit: int = 20) -> list[dict]:
        rows = self._fetchall("SELECT * FROM investigations ORDER BY timestamp DESC LIMIT ?", (limit,))
        for d in rows:
            d["report"] = json.loads(d.pop("report_json", "{}"))
        return rows

    # --- Cameras ---

    def upsert_camera(self, cam: dict):
        self._execute(
            "INSERT INTO cameras (camera_id, name, rtsp_url, site_x, site_y, rotation_deg, fov_deg, enabled, tenant_id, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT (camera_id) DO UPDATE SET name=EXCLUDED.name, rtsp_url=EXCLUDED.rtsp_url, enabled=EXCLUDED.enabled",
            (cam["camera_id"], cam.get("name"), cam.get("rtsp_url"),
             cam.get("site_x", 0), cam.get("site_y", 0),
             cam.get("rotation_deg", 0), cam.get("fov_deg", 60),
             cam.get("enabled", 1), cam.get("tenant_id"),
             json.dumps(cam.get("tags", []), default=str),
             cam.get("created_at", time.time())),
        )

    def get_cameras(self, tenant_id: str | None = None) -> list[dict]:
        if tenant_id:
            return self._fetchall("SELECT * FROM cameras WHERE tenant_id=?", (tenant_id,))
        return self._fetchall("SELECT * FROM cameras")

    def delete_camera(self, camera_id: str):
        self._execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))

    # --- Webhooks ---

    def save_webhook(self, config) -> None:
        if hasattr(config, "__dataclass_fields__"):
            data = {k: getattr(config, k) for k in config.__dataclass_fields__}
            data["webhook_type"] = str(data.get("webhook_type", ""))
            wid = config.webhook_id
        else:
            data = config
            wid = config["webhook_id"]
        self._execute(
            "INSERT INTO webhook_configs (webhook_id, config_json) VALUES (?,?) ON CONFLICT (webhook_id) DO UPDATE SET config_json=EXCLUDED.config_json",
            (wid, json.dumps(data, default=str)),
        )

    def get_webhooks(self) -> list[dict]:
        rows = self._fetchall("SELECT config_json FROM webhook_configs")
        return [json.loads(r["config_json"]) for r in rows]

    def delete_webhook(self, webhook_id: str):
        self._execute("DELETE FROM webhook_configs WHERE webhook_id=?", (webhook_id,))

    # --- API Keys ---

    def save_api_key(self, key_hash: str, tenant_id: str, user_id: str, access_level: str, cameras: list[str]):
        self._execute(
            "INSERT INTO api_keys (key_hash, tenant_id, user_id, access_level, cameras_json, created_at) VALUES (?,?,?,?,?,?) ON CONFLICT (key_hash) DO UPDATE SET access_level=EXCLUDED.access_level",
            (key_hash, tenant_id, user_id, access_level, json.dumps(cameras), time.time()),
        )

    def verify_api_key(self, key_hash: str) -> dict | None:
        d = self._fetchone("SELECT * FROM api_keys WHERE key_hash=?", (key_hash,))
        if not d:
            return None
        d["cameras"] = json.loads(d.pop("cameras_json", "[]"))
        return d

    # --- Report Schedules ---

    def save_report_schedule(self, schedule: dict):
        self._execute(
            "INSERT INTO report_schedules (schedule_id, config_json) VALUES (?,?) ON CONFLICT (schedule_id) DO UPDATE SET config_json=EXCLUDED.config_json",
            (schedule["schedule_id"], json.dumps(schedule, default=str)),
        )

    def get_report_schedules(self) -> list[dict]:
        rows = self._fetchall("SELECT config_json FROM report_schedules")
        return [json.loads(r["config_json"]) for r in rows]
