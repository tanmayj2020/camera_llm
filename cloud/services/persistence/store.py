"""SQLite persistence for VisionBrain — zero external dependencies."""

import json
import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("VISIONBRAIN_DB", "./visionbrain.db")


class PersistenceStore:
    def __init__(self, db_path: str = DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()
        logger.info("PersistenceStore initialized at %s", db_path)

    def _ensure_tables(self):
        c = self._conn.cursor()
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
        self._conn.commit()

    # --- Alerts ---

    def save_alert(self, alert: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO alerts (event_id, timestamp, camera_id, severity, anomaly_type, description, causal_explanation) VALUES (?,?,?,?,?,?,?)",
            (alert["event_id"], alert.get("timestamp", time.time()), alert.get("camera_id"),
             alert.get("severity"), alert.get("anomaly_type"), alert.get("description"),
             alert.get("causal_explanation")),
        )
        self._conn.commit()

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
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def ack_alert(self, event_id: str, user_id: str):
        self._conn.execute(
            "UPDATE alerts SET acknowledged=1, acked_by=?, acked_at=? WHERE event_id=?",
            (user_id, time.time(), event_id),
        )
        self._conn.commit()

    # --- Investigations ---

    def save_investigation(self, investigation_id: str, entity_id: str, report: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO investigations (investigation_id, entity_id, timestamp, report_json) VALUES (?,?,?,?)",
            (investigation_id, entity_id, time.time(), json.dumps(report, default=str)),
        )
        self._conn.commit()

    def get_investigations(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM investigations ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["report"] = json.loads(d.pop("report_json", "{}"))
            results.append(d)
        return results

    # --- Cameras ---

    def upsert_camera(self, cam: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO cameras (camera_id, name, rtsp_url, site_x, site_y, rotation_deg, fov_deg, enabled, tenant_id, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cam["camera_id"], cam.get("name"), cam.get("rtsp_url"),
             cam.get("site_x", 0), cam.get("site_y", 0),
             cam.get("rotation_deg", 0), cam.get("fov_deg", 60),
             cam.get("enabled", 1), cam.get("tenant_id"),
             json.dumps(cam.get("tags", []), default=str),
             cam.get("created_at", time.time())),
        )
        self._conn.commit()

    def get_cameras(self, tenant_id: str | None = None) -> list[dict]:
        if tenant_id:
            rows = self._conn.execute("SELECT * FROM cameras WHERE tenant_id=?", (tenant_id,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM cameras").fetchall()
        return [dict(r) for r in rows]

    def delete_camera(self, camera_id: str):
        self._conn.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
        self._conn.commit()

    # --- Webhooks ---

    def save_webhook(self, config) -> None:
        """Accept a WebhookConfig dataclass or dict."""
        if hasattr(config, "__dataclass_fields__"):
            from cloud.services.webhooks.manager import WebhookConfig
            data = {k: getattr(config, k) for k in config.__dataclass_fields__}
            data["webhook_type"] = str(data.get("webhook_type", ""))
            wid = config.webhook_id
        else:
            data = config
            wid = config["webhook_id"]
        self._conn.execute(
            "INSERT OR REPLACE INTO webhook_configs (webhook_id, config_json) VALUES (?,?)",
            (wid, json.dumps(data, default=str)),
        )
        self._conn.commit()

    def get_webhooks(self) -> list[dict]:
        rows = self._conn.execute("SELECT config_json FROM webhook_configs").fetchall()
        return [json.loads(r["config_json"]) for r in rows]

    def delete_webhook(self, webhook_id: str):
        self._conn.execute("DELETE FROM webhook_configs WHERE webhook_id=?", (webhook_id,))
        self._conn.commit()

    # --- API Keys ---

    def save_api_key(self, key_hash: str, tenant_id: str, user_id: str, access_level: str, cameras: list[str]):
        self._conn.execute(
            "INSERT OR REPLACE INTO api_keys (key_hash, tenant_id, user_id, access_level, cameras_json, created_at) VALUES (?,?,?,?,?,?)",
            (key_hash, tenant_id, user_id, access_level, json.dumps(cameras), time.time()),
        )
        self._conn.commit()

    def verify_api_key(self, key_hash: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM api_keys WHERE key_hash=?", (key_hash,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["cameras"] = json.loads(d.pop("cameras_json", "[]"))
        return d

    # --- Report Schedules ---

    def save_report_schedule(self, schedule: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO report_schedules (schedule_id, config_json) VALUES (?,?)",
            (schedule["schedule_id"], json.dumps(schedule, default=str)),
        )
        self._conn.commit()

    def get_report_schedules(self) -> list[dict]:
        rows = self._conn.execute("SELECT config_json FROM report_schedules").fetchall()
        return [json.loads(r["config_json"]) for r in rows]
