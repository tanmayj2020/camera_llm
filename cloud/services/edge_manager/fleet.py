"""Edge fleet management — heartbeat tracking and device status."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EdgeDevice:
    device_id: str
    camera_ids: list[str] = field(default_factory=list)
    ip_address: str = ""
    last_heartbeat: float = 0.0
    model_version: str = ""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    fps: float = 0.0
    status: str = "unknown"


class FleetManager:
    def __init__(self):
        self._devices: dict[str, EdgeDevice] = {}

    def register_heartbeat(self, device_id: str, stats: dict):
        dev = self._devices.get(device_id)
        if not dev:
            dev = EdgeDevice(device_id=device_id)
            self._devices[device_id] = dev
        dev.last_heartbeat = time.time()
        dev.camera_ids = stats.get("camera_ids", dev.camera_ids)
        dev.ip_address = stats.get("ip_address", dev.ip_address)
        dev.model_version = stats.get("model_version", dev.model_version)
        dev.cpu_usage = stats.get("cpu_usage", dev.cpu_usage)
        dev.memory_usage = stats.get("memory_usage", dev.memory_usage)
        dev.fps = stats.get("fps", dev.fps)
        dev.status = "online"

    def get_fleet_status(self) -> list[dict]:
        now = time.time()
        result = []
        for dev in self._devices.values():
            d = dev.__dict__.copy()
            d["status"] = "online" if (now - dev.last_heartbeat) < 300 else "offline"
            result.append(d)
        return result

    def get_offline_devices(self, timeout_s: float = 300) -> list[str]:
        now = time.time()
        return [d.device_id for d in self._devices.values()
                if (now - d.last_heartbeat) > timeout_s]

    def push_config(self, device_id: str, config: dict) -> bool:
        if device_id not in self._devices:
            return False
        logger.info("Config pushed to %s: %s", device_id, list(config.keys()))
        return True
