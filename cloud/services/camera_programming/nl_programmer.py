"""Natural Language Camera Programming."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CameraProgram:
    description: str
    camera_id: str
    schedule: dict = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    enabled: bool = True
    program_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)


class CameraProgrammer:
    def __init__(self, vlm_client=None, reasoner=None):
        self._vlm_client = vlm_client
        self._reasoner = reasoner
        self._programs: dict[str, CameraProgram] = {}

    def parse_instruction(self, text: str) -> CameraProgram:
        """Parse natural language instruction into a CameraProgram via VLM."""
        prompt = (
            "Parse this camera programming instruction into JSON with keys: "
            "description, camera_id, schedule (start_hour, end_hour, days), "
            "actions (list of {type, params}), conditions (list of {type, params}).\n"
            f"Instruction: {text}"
        )
        try:
            if self._vlm_client:
                client = self._vlm_client._get_client()
                if client != "stub":
                    result = client.generate(prompt)
                    data = json.loads(result)
                    return CameraProgram(
                        description=data.get("description", text),
                        camera_id=data.get("camera_id", ""),
                        schedule=data.get("schedule", {}),
                        actions=data.get("actions", []),
                        conditions=data.get("conditions", []),
                    )
        except Exception as e:
            logger.warning("VLM parse failed, using fallback: %s", e)

        logger.info("Using fallback template for instruction")
        return CameraProgram(description=text, camera_id="", schedule={}, actions=[])

    def add_program(self, program: CameraProgram) -> None:
        self._programs[program.program_id] = program
        logger.info("Added program %s", program.program_id)

    def remove_program(self, program_id: str) -> None:
        self._programs.pop(program_id, None)
        logger.info("Removed program %s", program_id)

    def list_programs(self) -> list[CameraProgram]:
        return list(self._programs.values())

    def evaluate(self, scene_state: dict, current_hour: int) -> list[dict]:
        """Return actions from all matching enabled programs."""
        import datetime
        day = datetime.datetime.now().strftime("%A")
        results = []
        for prog in self._programs.values():
            if not prog.enabled:
                continue
            if self._check_schedule(prog, current_hour, day) and self._check_conditions(prog, scene_state):
                results.extend(prog.actions)
        return results

    def _check_schedule(self, program: CameraProgram, hour: int, day: str) -> bool:
        s = program.schedule
        if not s:
            return True
        if "start_hour" in s and "end_hour" in s:
            if not (s["start_hour"] <= hour < s["end_hour"]):
                return False
        if "days" in s and day not in s["days"]:
            return False
        return True

    def _check_conditions(self, program: CameraProgram, scene_state: dict) -> bool:
        for cond in program.conditions:
            ctype = cond.get("type", "")
            params = cond.get("params", {})
            if ctype == "threshold":
                key = params.get("key", "")
                if scene_state.get(key, 0) < params.get("min", 0):
                    return False
            elif ctype == "presence":
                key = params.get("key", "")
                if key not in scene_state:
                    return False
        return True
