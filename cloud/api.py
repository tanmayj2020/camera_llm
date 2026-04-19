"""VisionBrain Cloud API — FastAPI service orchestrating all intelligence layers."""

import json
import logging
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionbrain.cloud")

app = FastAPI(title="VisionBrain Cloud API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_services = {}


def get_kg():
    if "kg" not in _services:
        from services.knowledge_graph.temporal_kg import TemporalKnowledgeGraph
        _services["kg"] = TemporalKnowledgeGraph(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
    return _services["kg"]


def get_baseline():
    if "baseline" not in _services:
        from services.baseline.learner import BaselineLearner
        _services["baseline"] = BaselineLearner()
    return _services["baseline"]


def get_spatial():
    if "spatial" not in _services:
        from services.spatial.memory import SpatialMemory
        _services["spatial"] = SpatialMemory()
    return _services["spatial"]


def get_reasoner():
    if "reasoner" not in _services:
        from services.reasoning.engine import ReasoningEngine
        engine = ReasoningEngine()
        engine.add_default_rules()
        _services["reasoner"] = engine
    return _services["reasoner"]


def get_causal():
    if "causal" not in _services:
        from services.causal.understander import CausalUnderstander
        _services["causal"] = CausalUnderstander()
    return _services["causal"]


def get_world_model():
    if "world_model" not in _services:
        from services.world_model.predictor import WorldModel
        _services["world_model"] = WorldModel()
    return _services["world_model"]


def get_action_engine():
    if "action" not in _services:
        from services.action_engine.engine import ActionEngine
        _services["action"] = ActionEngine()
    return _services["action"]


def get_query_engine():
    if "query" not in _services:
        from services.query_engine.engine import MultiAgentQueryEngine
        _services["query"] = MultiAgentQueryEngine(
            kg=get_kg(), spatial=get_spatial(), vlm_client=get_causal(),
            rag=get_hypergraph_rag(),
        )
    return _services["query"]


def get_kpi_engine():
    if "kpi" not in _services:
        from services.kpi_engine.engine import KPIEngine
        _services["kpi"] = KPIEngine(kg=get_kg(), vlm_client=get_causal())
    return _services["kpi"]


def get_tracker_engine():
    if "tracker" not in _services:
        from services.custom_tracker.tracker import CustomTrackerEngine
        _services["tracker"] = CustomTrackerEngine(reasoner=get_reasoner())
        _services["tracker"].set_vlm_client(get_causal())
    return _services["tracker"]


def get_story_builder():
    if "story" not in _services:
        from services.query_engine.story_builder import StoryBuilder
        _services["story"] = StoryBuilder(kg=get_kg())
        _services["story"].set_vlm_client(get_causal())
    return _services["story"]


def get_entity_profiler():
    if "profiler" not in _services:
        from services.knowledge_graph.entity_profiles import EntityProfiler
        _services["profiler"] = EntityProfiler(kg=get_kg())
        _services["profiler"].set_vlm_client(get_causal())
    return _services["profiler"]


def get_live_agent():
    if "live_agent" not in _services:
        from services.conversational.live_agent import LiveConversationalAgent
        _services["live_agent"] = LiveConversationalAgent(
            kg=get_kg(), spatial=get_spatial(),
            query_engine=get_query_engine(), profiler=get_entity_profiler(),
        )
        _services["live_agent"].set_vlm_client(get_causal())
    return _services["live_agent"]


def get_investigator():
    if "investigator" not in _services:
        from services.investigation.auto_investigator import AutoInvestigator
        _services["investigator"] = AutoInvestigator(
            kg=get_kg(), profiler=get_entity_profiler(),
            story_builder=get_story_builder(), spatial=get_spatial(),
        )
        _services["investigator"].set_vlm_client(get_causal())
    return _services["investigator"]


def get_synopsis_engine():
    if "synopsis" not in _services:
        from services.video_synopsis.synopsis import SynopsisEngine
        _services["synopsis"] = SynopsisEngine(kg=get_kg())
        _services["synopsis"].set_vlm_client(get_causal())
    return _services["synopsis"]


def get_forensic_search():
    if "forensic" not in _services:
        from services.forensic_search.privacy_search import ForensicSearchEngine
        _services["forensic"] = ForensicSearchEngine(kg=get_kg())
        _services["forensic"].set_vlm_client(get_causal())
    return _services["forensic"]


def get_scene_predictor():
    if "scene_predictor" not in _services:
        from services.anomaly.video_predictor import SceneAnomalyPredictor
        _services["scene_predictor"] = SceneAnomalyPredictor(
            world_model=get_world_model(), baseline=get_baseline(),
        )
    return _services["scene_predictor"]


def get_persistence():
    if "persistence" not in _services:
        from services.persistence.store import PersistenceStore
        _services["persistence"] = PersistenceStore()
    return _services["persistence"]


def get_webhook_manager():
    if "webhook_mgr" not in _services:
        from services.webhooks.manager import WebhookManager
        _services["webhook_mgr"] = WebhookManager()
    return _services["webhook_mgr"]


def get_camera_manager():
    if "camera_mgr" not in _services:
        from services.cameras.manager import CameraManager
        _services["camera_mgr"] = CameraManager(
            persistence=get_persistence(), floor_plan=get_floor_plan(),
        )
    return _services["camera_mgr"]


def get_occupancy():
    if "occupancy" not in _services:
        from services.occupancy.counter import OccupancyCounter
        _services["occupancy"] = OccupancyCounter(
            spatial=get_spatial(), vlm_client=get_causal(),
        )
    return _services["occupancy"]


def get_safety_monitor():
    if "safety" not in _services:
        from services.compliance.safety_monitor import SafetyMonitor
        _services["safety"] = SafetyMonitor(
            spatial=get_spatial(), reasoner=get_reasoner(),
        )
    return _services["safety"]


def get_lpr():
    if "lpr" not in _services:
        from services.lpr.recognizer import LPREngine
        _services["lpr"] = LPREngine(vlm_client=get_causal())
    return _services["lpr"]


def get_report_scheduler():
    if "report_sched" not in _services:
        from services.reports.scheduler import ReportScheduler
        _services["report_sched"] = ReportScheduler(
            kpi_engine=get_kpi_engine(), persistence=get_persistence(),
            vlm_client=get_causal(),
        )
    return _services["report_sched"]


def get_alert_digest():
    if "alert_digest" not in _services:
        from services.reports.alert_digest import AlertDigest
        _services["alert_digest"] = AlertDigest(
            persistence=get_persistence(), vlm_client=get_causal(),
            continual_learner=get_continual_learner(),
            action_engine=get_action_engine(),
        )
    return _services["alert_digest"]


def get_explainer():
    if "explainer" not in _services:
        from services.reasoning.explainer import AlertExplainer
        _services["explainer"] = AlertExplainer(
            baseline=get_baseline(), kg=get_kg(), vlm_client=get_causal(),
        )
    return _services["explainer"]


def get_continual_learner():
    if "continual" not in _services:
        from services.continual_learning.learner import ContinualLearner
        _services["continual"] = ContinualLearner()
    return _services["continual"]


def get_auth_provider():
    if "auth" not in _services:
        from services.auth.middleware import AuthProvider, set_auth_provider
        _services["auth"] = AuthProvider(get_persistence())
        set_auth_provider(_services["auth"])
    return _services["auth"]


def get_floor_plan():
    if "floor_plan" not in _services:
        from services.digital_twin.floor_plan import DigitalTwinFloorPlan
        fp = DigitalTwinFloorPlan()
        try:
            cam_mgr = _services.get("camera_mgr")
            if cam_mgr:
                cam_ids = cam_mgr.get_camera_ids()
                if cam_ids:
                    for cam_id in cam_ids:
                        cfg = cam_mgr.get_camera(cam_id)
                        fp.register_camera(cam_id, site_x=cfg.site_x,
                                           site_y=cfg.site_y, rotation_deg=cfg.rotation_deg)
                    fp.register_spatial_memory(cam_ids[0], get_spatial())
                    _services["floor_plan"] = fp
                    return fp
        except Exception:
            pass
        for i, cam_id in enumerate(["cam-0", "cam-1", "cam-2", "cam-3"]):
            fp.register_camera(cam_id, site_x=10 + (i % 2) * 25,
                               site_y=10 + (i // 2) * 25, rotation_deg=i * 90)
        fp.register_spatial_memory("cam-0", get_spatial())
        _services["floor_plan"] = fp
    return _services["floor_plan"]


def get_crowd_analyzer():
    if "crowd" not in _services:
        from services.crowd.social_force import analyze_crowd
        _services["crowd"] = analyze_crowd
    return _services["crowd"]


def get_scene_graph():
    if "scene_graph" not in _services:
        from services.scene_graph.dynamic_graph import DynamicSceneGraph
        _services["scene_graph"] = DynamicSceneGraph(spatial=get_spatial())
    return _services["scene_graph"]


def get_contextual_normality():
    if "ctx_norm" not in _services:
        from services.anomaly.contextual_normality import ContextualNormalityModel
        _services["ctx_norm"] = ContextualNormalityModel()
    return _services["ctx_norm"]


def get_counterfactual_explainer():
    if "counterfactual" not in _services:
        from services.reasoning.counterfactual_explainer import CounterfactualExplainer
        _services["counterfactual"] = CounterfactualExplainer(reasoner=get_reasoner())
    return _services["counterfactual"]


def get_collective_detector():
    if "collective" not in _services:
        from services.anomaly.collective_anomaly import CollectiveAnomalyDetector
        _services["collective"] = CollectiveAnomalyDetector()
    return _services["collective"]


def get_what_if():
    if "what_if" not in _services:
        from services.simulation.what_if import WhatIfSimulator
        _services["what_if"] = WhatIfSimulator(
            world_model=get_world_model(), spatial=get_spatial(),
        )
    return _services["what_if"]


def get_self_evaluator():
    if "self_eval" not in _services:
        from services.meta.self_evaluator import SelfEvaluator
        _services["self_eval"] = SelfEvaluator(
            persistence=get_persistence(), continual_learner=get_continual_learner(),
            action_engine=get_action_engine(), vlm_client=get_causal(),
        )
    return _services["self_eval"]


def get_semantic_dedup():
    if "dedup" not in _services:
        from services.nlp.alert_dedup_semantic import SemanticDeduplicator
        _services["dedup"] = SemanticDeduplicator()
    return _services["dedup"]


def get_environment_detector():
    if "env_detect" not in _services:
        from services.detection.environment_detector import EnvironmentDetector
        _services["env_detect"] = EnvironmentDetector(vlm_client=get_causal())
    return _services["env_detect"]


def get_tailgate_detector():
    if "tailgate" not in _services:
        from services.access.tailgate_detector import TailgateDetector
        _services["tailgate"] = TailgateDetector(spatial=get_spatial())
    return _services["tailgate"]


def get_dwell_analyzer():
    if "dwell" not in _services:
        from services.analytics.dwell_time import DwellTimeAnalyzer
        _services["dwell"] = DwellTimeAnalyzer(spatial=get_spatial())
    return _services["dwell"]


def get_heatmap_engine():
    if "heatmap" not in _services:
        from services.analytics.heatmap_engine import HeatmapEngine
        _services["heatmap"] = HeatmapEngine()
    return _services["heatmap"]


def get_fleet_manager():
    if "fleet" not in _services:
        from services.edge_manager.fleet import FleetManager
        _services["fleet"] = FleetManager()
    return _services["fleet"]


def get_proactive_agent():
    if "proactive" not in _services:
        from services.streaming.proactive_agent import ProactiveAgent
        _services["proactive"] = ProactiveAgent(
            spatial=get_spatial(), kg=get_kg(), vlm_client=get_causal(),
        )
    return _services["proactive"]


def get_av_fusion():
    if "av_fusion" not in _services:
        from services.audio_visual.fusion import AudioVisualFusion
        _services["av_fusion"] = AudioVisualFusion(spatial=get_spatial())
    return _services["av_fusion"]


def get_consent_manager():
    if "consent" not in _services:
        from services.privacy.consent_manager import ConsentManager
        _services["consent"] = ConsentManager(persistence=get_persistence())
    return _services["consent"]


def get_evidence_packager():
    if "evidence" not in _services:
        from services.evidence.packager import EvidencePackager
        _services["evidence"] = EvidencePackager(
            persistence=get_persistence(), kg=get_kg(), vlm_client=get_causal(),
        )
    return _services["evidence"]


def get_shift_agent():
    if "shift" not in _services:
        from services.shift.handover import ShiftHandoverAgent
        _services["shift"] = ShiftHandoverAgent(
            persistence=get_persistence(), kg=get_kg(),
            vlm_client=get_causal(), action_engine=get_action_engine(),
        )
    return _services["shift"]


def get_camera_programmer():
    if "cam_prog" not in _services:
        from services.camera_programming.nl_programmer import CameraProgrammer
        _services["cam_prog"] = CameraProgrammer(
            vlm_client=get_causal(), reasoner=get_reasoner(),
        )
    return _services["cam_prog"]


def get_adversarial_monitor():
    if "adversarial" not in _services:
        from services.adversarial.robustness import AdversarialMonitor
        _services["adversarial"] = AdversarialMonitor()
    return _services["adversarial"]


def get_journey_visualizer():
    if "journey" not in _services:
        from services.entity_journey.visualizer import JourneyVisualizer
        _services["journey"] = JourneyVisualizer(spatial=get_spatial(), kg=get_kg())
    return _services["journey"]


def get_retail_analytics():
    if "retail" not in _services:
        from services.retail.analytics import RetailAnalytics
        _services["retail"] = RetailAnalytics(
            spatial=get_spatial(), occupancy=get_occupancy(),
            dwell_analyzer=get_dwell_analyzer(), heatmap=get_heatmap_engine(),
        )
    return _services["retail"]


def get_event_bus():
    if "event_bus" not in _services:
        from services.integration.event_bus import EventBus
        _services["event_bus"] = EventBus()
    return _services["event_bus"]


def get_visual_grounder():
    if "grounder" not in _services:
        from services.visual_grounding.grounder import VisualGrounder
        _services["grounder"] = VisualGrounder(vlm_client=get_causal(), spatial=get_spatial())
    return _services["grounder"]


def get_resource_optimizer():
    if "resource_opt" not in _services:
        from services.resource_optimizer.staffing import ResourceOptimizer
        _services["resource_opt"] = ResourceOptimizer(
            occupancy=get_occupancy(), contextual_normality=get_contextual_normality(),
            kpi_engine=get_kpi_engine(), vlm_client=get_causal())
    return _services["resource_opt"]


def get_stress_detector():
    if "stress" not in _services:
        from services.behavioral.stress_detector import StressDetector
        _services["stress"] = StressDetector(spatial=get_spatial())
    return _services["stress"]


def get_federated_intel():
    if "federation" not in _services:
        from services.federation.intelligence import FederatedIntelligence
        _services["federation"] = FederatedIntelligence(
            webhook_manager=get_webhook_manager(), profiler=get_entity_profiler())
    return _services["federation"]


def get_reid_embedder():
    if "reid" not in _services:
        from services.cross_camera.reid_embedder import ReIDEmbedder
        _services["reid"] = ReIDEmbedder()
    return _services["reid"]


def get_nl_rule_compiler():
    if "nl_compiler" not in _services:
        from services.nl_rule_compiler.compiler import NLRuleCompiler
        _services["nl_compiler"] = NLRuleCompiler(
            vlm_client=get_causal(), reasoner=get_reasoner(),
        )
    return _services["nl_compiler"]


def get_ambient_score():
    if "ambient" not in _services:
        from services.ambient_score.engine import AmbientIntelligenceEngine
        _services["ambient"] = AmbientIntelligenceEngine(
            spatial=get_spatial(), baseline=get_baseline(),
            dwell_analyzer=get_dwell_analyzer(),
            contextual_normality=get_contextual_normality(),
            scene_graph=get_scene_graph(),
            world_model=get_world_model(),
            stress_detector=get_stress_detector(),
        )
    return _services["ambient"]


def get_predictive_interceptor():
    if "interceptor" not in _services:
        from services.predictive_interception.interceptor import PredictiveInterceptor
        _services["interceptor"] = PredictiveInterceptor(
            world_model=get_world_model(), spatial=get_spatial(),
            floor_plan=get_floor_plan(),
        )
    return _services["interceptor"]


def get_gait_dna():
    if "gait" not in _services:
        from services.gait_dna.engine import GaitDNAEngine
        _services["gait"] = GaitDNAEngine()
    return _services["gait"]


def get_contagion_network():
    if "contagion" not in _services:
        from services.contagion.network import AnomalyContagionNetwork
        _services["contagion"] = AnomalyContagionNetwork()
    return _services["contagion"]


def get_deja_vu():
    if "deja_vu" not in _services:
        from services.deja_vu.engine import DejaVuEngine
        _services["deja_vu"] = DejaVuEngine()
    return _services["deja_vu"]


def get_cross_camera_tracker():
    if "cross_cam" not in _services:
        from services.cross_camera.tracker import CrossCameraTracker
        _services["cross_cam"] = CrossCameraTracker(kg=get_kg())
    return _services["cross_cam"]


def get_hypergraph_rag():
    if "hgrag" not in _services:
        from services.knowledge_graph.hypergraph_rag import HyperGraphRAG
        _services["hgrag"] = HyperGraphRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "visionbrain-keyframes"),
            neo4j_driver=get_kg()._driver if get_kg() else None,
        )
    return _services["hgrag"]


# --- Auth dependency ---
try:
    from services.auth.middleware import get_current_tenant
except ImportError:
    def get_current_tenant():
        return "default"

auth_dep = Depends(get_current_tenant)


# --- Models ---

class EventPayload(BaseModel):
    event_id: str
    timestamp: float
    camera_id: str
    event_type: str = "detection"
    scene_activity: float = 0.0
    objects: list = []
    audio_events: list = []
    keyframe_b64: str | None = None


class QueryRequest(BaseModel):
    query: str
    camera_id: str | None = None


class RulePayload(BaseModel):
    rule_id: str
    name: str
    severity: str = "medium"
    conditions: list[dict]
    action: str


class TrackerCreateRequest(BaseModel):
    description: str


class TrackerUpdateRequest(BaseModel):
    status: str


class ConverseRequest(BaseModel):
    message: str
    session_id: str | None = None
    camera_id: str | None = None


class ForensicSearchRequest(BaseModel):
    query: str
    camera_id: str | None = None
    user_id: str = "anonymous"
    access_level: str = "anonymous"
    time_range_hours: int = 24


class UnlockRequest(BaseModel):
    result_id: str
    user_id: str
    access_level: str
    justification: str = ""


class CameraPayload(BaseModel):
    camera_id: str
    name: str
    rtsp_url: str = ""
    site_x: float = 0
    site_y: float = 0
    rotation_deg: float = 0
    fov_deg: float = 60
    tenant_id: str = "default"
    tags: list[str] = []


class WebhookPayload(BaseModel):
    name: str
    url: str
    webhook_type: str = "GENERIC"
    min_severity: str = "medium"
    camera_filter: list[str] = []
    event_types: list[str] = []


class ZonePayload(BaseModel):
    zone_id: str
    name: str
    polygon: list
    zone_type: str = "monitored"
    capacity: int = 0


class SOPRulePayload(BaseModel):
    rule_id: str
    name: str
    required_steps: list[dict]
    zone_id: str = ""
    severity: str = "medium"
    cooldown_s: float = 300


class WatchlistPayload(BaseModel):
    plate: str
    reason: str
    severity: str = "high"


class ReportSchedulePayload(BaseModel):
    name: str
    schedule_type: str
    report_type: str
    camera_ids: list[str] = []
    recipients: list[str] = []


class ApiKeyPayload(BaseModel):
    tenant_id: str
    user_id: str
    access_level: str = "operator"
    cameras: list[str] = []


class SimulatePayload(BaseModel):
    scenario_type: str
    parameters: dict = {}
    description: str = ""


class HeartbeatPayload(BaseModel):
    device_id: str
    camera_ids: list[str] = []
    ip_address: str = ""
    model_version: str = ""
    cpu_usage: float = 0
    memory_usage: float = 0
    fps: float = 0


class PrioritiesPayload(BaseModel):
    priorities: list[str]


class WatchEntityPayload(BaseModel):
    entity_id: str


class ConsentPayload(BaseModel):
    entity_id: str
    consent_type: str = "processing"
    granted: bool = True
    purpose: str = ""
    legal_basis: str = "consent"
    duration_days: int = 365


class EvidenceRequest(BaseModel):
    incident_id: str
    camera_ids: list[str] = []
    time_range_hours: float = 4
    created_by: str = "system"


class CameraProgramRequest(BaseModel):
    instruction: str


class RetailConfigPayload(BaseModel):
    entry_zones: list[str] = []
    checkout_zones: list[str] = []
    queue_zones: list[str] = []


class EventSubscriptionPayload(BaseModel):
    subscriber_name: str
    event_types: list[str]
    filters: dict = {}
    callback_url: str


class FederationPeerPayload(BaseModel):
    site_id: str
    callback_url: str


class FederationAlertPayload(BaseModel):
    entity_id: str
    severity: str = "medium"
    description: str = ""


# --- WebSocket connections ---
ws_connections: list[WebSocket] = []


async def broadcast_alert(alert: dict):
    for ws in ws_connections[:]:
        try:
            await ws.send_json(alert)
        except Exception:
            ws_connections.remove(ws)


# --- Core event pipeline ---

@app.post("/api/events")
async def ingest_event(payload: EventPayload):
    """Ingest a vision event from edge — runs full intelligence pipeline."""
    event = payload.model_dump()

    # 1. Knowledge graph ingestion
    from services.knowledge_graph.temporal_kg import ingest_vision_event
    try:
        ingest_vision_event(get_kg(), event)
    except Exception as e:
        logger.error("KG ingestion failed: %s", e)

    # 2. Baseline learning / anomaly scoring
    baseline = get_baseline()
    baseline.ingest_event(event)
    anomaly_score = baseline.compute_anomaly_score(event)

    # 3. Spatial update
    spatial = get_spatial()
    for obj in event.get("objects", []):
        if obj.get("bbox"):
            spatial.update(obj["track_id"], obj["class_name"], obj["bbox"], event["timestamp"])

    # 3b. Occupancy update
    try:
        get_occupancy().update(event["camera_id"], spatial)
    except Exception:
        pass

    # 3b-ii. Re-ID embeddings + cross-camera tracking
    if event.get("keyframe_b64"):
        try:
            reid = get_reid_embedder()
            xtracker = get_cross_camera_tracker()
            for obj in event.get("objects", []):
                if obj.get("class_name") == "person" and obj.get("bbox"):
                    emb = reid.extract(event["keyframe_b64"], obj["bbox"])
                    if emb is not None:
                        import numpy as _np
                        xtracker.match_across_cameras(
                            str(obj["track_id"]), event["camera_id"], _np.array(emb))
        except Exception:
            pass

    # 3b-iii. HyperGraphRAG event indexing
    if event.get("keyframe_b64"):
        try:
            entity_ids = [str(o.get("track_id", "")) for o in event.get("objects", []) if o.get("track_id", -1) >= 0]
            get_hypergraph_rag().index_event(
                event["event_id"], event["camera_id"], event["timestamp"],
                event["keyframe_b64"],
                f"{len(event.get('objects', []))} objects detected",
                entity_ids)
        except Exception:
            pass

    # 3c. Safety evaluation
    scene_state = {
        "objects": event.get("objects", []),
        "audio_events": event.get("audio_events", []),
        "spatial": spatial,
        "timestamp": event["timestamp"],
        "camera_id": event["camera_id"],
        "anomaly_score": anomaly_score,
        "keyframe_b64": event.get("keyframe_b64"),
    }
    try:
        safety_alerts = get_safety_monitor().evaluate(scene_state)
        for sa in (safety_alerts or []):
            await broadcast_alert({"type": "safety_alert", "alert": sa})
            try:
                get_event_bus().publish("safety_alert", sa)
            except Exception:
                pass
    except Exception:
        pass

    # 3d. LPR for vehicle objects
    for obj in event.get("objects", []):
        if obj.get("class_name", "").lower() in ("car", "truck", "bus", "vehicle"):
            try:
                from services.lpr.recognizer import crop_vehicle_from_keyframe
                crop_b64 = crop_vehicle_from_keyframe(
                    event.get("keyframe_b64", ""), obj.get("bbox", [])) if event.get("keyframe_b64") else None
                if crop_b64:
                    get_lpr().recognize(crop_b64, event["camera_id"],
                                        str(obj.get("track_id", "")), event["timestamp"])
            except Exception:
                pass

    # 3e. Contextual normality scoring
    ctx_anomaly = 0.0
    try:
        ctx_norm = get_contextual_normality()
        person_count = sum(1 for o in event.get("objects", []) if o.get("class_name") == "person")
        ctx_norm.ingest(event["camera_id"], person_count, event["timestamp"])
        ctx_anomaly = ctx_norm.compute_contextual_anomaly_score(
            event["camera_id"], person_count, event["timestamp"])
    except Exception:
        pass

    # 3f. Scene graph update
    try:
        get_scene_graph().update(spatial, event["timestamp"])
    except Exception:
        pass

    # 3g. Collective anomaly ingestion
    try:
        get_collective_detector().ingest(event)
    except Exception:
        pass

    # 3h. Environment detection when keyframe present
    if event.get("keyframe_b64"):
        try:
            env_alerts = get_environment_detector().check_environment(
                event["keyframe_b64"], event["camera_id"], spatial)
            for ea in env_alerts:
                await broadcast_alert({"type": "environment_alert", "alert": ea.__dict__})
        except Exception:
            pass

    # 3i. Tailgate detection
    try:
        tailgate_events = get_tailgate_detector().evaluate(scene_state)
        for te in tailgate_events:
            await broadcast_alert({"type": "tailgate_alert", "alert": te.__dict__})
            try:
                get_event_bus().publish("tailgate_alert", te.__dict__)
            except Exception:
                pass
    except Exception:
        pass

    # 3j. Dwell time + heatmap updates
    try:
        dwell = get_dwell_analyzer()
        heatmap = get_heatmap_engine()
        from datetime import datetime
        hour = datetime.fromtimestamp(event["timestamp"]).hour
        for eid, ent in spatial._entities.items():
            pos = ent.position
            heatmap.update(float(pos[0]), float(pos[1]), 1.0, hour)
            for zid in getattr(spatial, "_zones", {}):
                try:
                    in_zone = spatial.entities_in_zone(zid)
                    if any(getattr(e, "track_id", None) == eid for e in in_zone):
                        dwell.update(eid, zid, event["timestamp"])
                except Exception:
                    pass
    except Exception:
        pass

    # 3k. Proactive agent evaluation
    try:
        proactive_insights = get_proactive_agent().evaluate(scene_state)
        for pi in proactive_insights:
            await broadcast_alert({"type": "proactive_insight", "insight": pi.__dict__})
            try:
                get_event_bus().publish("proactive_insight", pi.__dict__)
            except Exception:
                pass
    except Exception:
        pass

    # 3l. Audio-visual fusion when both audio and visual present
    if event.get("audio_events") and event.get("objects"):
        try:
            av_results = get_av_fusion().fuse(event["audio_events"], scene_state, spatial)
            for avr in av_results:
                if avr.correlation_type == "contradicted":
                    logger.info("AV fusion contradicted: %s", avr.description)
                elif avr.correlation_type == "confirmed":
                    await broadcast_alert({"type": "av_confirmed", "correlation": avr.__dict__})
                    try:
                        get_event_bus().publish("av_confirmed", avr.__dict__)
                    except Exception:
                        pass
        except Exception:
            pass

    # 3m. Adversarial robustness monitoring
    try:
        det_count = len(event.get("objects", []))
        frame_hash = hash(event.get("keyframe_b64", "")[:100]) if event.get("keyframe_b64") else None
        deg_alerts = get_adversarial_monitor().evaluate(event["camera_id"], det_count, frame_hash)
        for da in deg_alerts:
            await broadcast_alert({"type": "degradation_alert", "alert": da.__dict__})
            try:
                get_event_bus().publish("degradation_alert", da.__dict__)
            except Exception:
                pass
    except Exception:
        pass

    # 3n. Entity journey recording
    try:
        jv = get_journey_visualizer()
        for eid, ent in spatial._entities.items():
            jv.record_sighting(eid, event["camera_id"], ent.position.tolist(), event["timestamp"])
    except Exception:
        pass

    # 3o. Behavioral stress detection
    # 3o. Behavioral stress detection
    try:
        stress_results = get_stress_detector().evaluate(scene_state)
        for sa in stress_results:
            await broadcast_alert({"type": "stress_assessment", "assessment": sa.__dict__})
            try:
                get_event_bus().publish("stress_assessment", sa.__dict__)
            except Exception:
                pass
    except Exception:
        pass

    # 3p. Gait DNA observation
    try:
        gait = get_gait_dna()
        for eid, ent in spatial._entities.items():
            pos = ent.position
            vel = getattr(ent, "velocity", [0, 0])
            pose_kps = None
            for obj in event.get("objects", []):
                if str(obj.get("track_id")) == str(eid) and obj.get("keypoints"):
                    pose_kps = obj["keypoints"]
            gait.observe(str(eid), float(pos[0]), float(pos[1]),
                         float(vel[0]) if len(vel) > 0 else 0.0,
                         float(vel[1]) if len(vel) > 1 else 0.0,
                         event["timestamp"], pose_kps)
            # Check for cross-session matches
            match = gait.identify_across_sessions(str(eid))
            if match and match.similarity > 0.85:
                await broadcast_alert({
                    "type": "gait_reidentification",
                    "current_entity": str(eid),
                    "matched_entity": match.matched_entity_id,
                    "similarity": match.similarity,
                })
    except Exception:
        pass

    # 3q. Predictive path interception
    try:
        intercept_alerts = get_predictive_interceptor().evaluate(scene_state)
        for ia in intercept_alerts:
            await broadcast_alert({
                "type": "predictive_interception",
                "entity": ia.entity_id,
                "target_zone": ia.target_zone_name,
                "eta_seconds": ia.estimated_arrival_s,
                "confidence": ia.confidence,
                "recommendation": ia.recommendation,
            })
            try:
                get_event_bus().publish("predictive_interception", {
                    "entity": ia.entity_id, "zone": ia.target_zone_id,
                    "eta": ia.estimated_arrival_s, "confidence": ia.confidence})
            except Exception:
                pass
    except Exception:
        pass

    # 3r. Scene déjà vu encoding
    try:
        get_deja_vu().encode_and_store(
            scene_state, event_type=event.get("event_type", ""),
            zone_id=event.get("zone_id", ""))
    except Exception:
        pass

    # 4. Reasoning engine
    rule_results = get_reasoner().evaluate(scene_state)

    # 4b. Custom tracker evaluation
    tracker_alerts = get_tracker_engine().evaluate(scene_state)
    for ta in tracker_alerts:
        await broadcast_alert({"type": "tracker_alert", "alert": ta})

    # 5. World model predictions
    wm = get_world_model()
    for obj in event.get("objects", []):
        ent = spatial._entities.get(obj["track_id"])
        if ent is not None:
            wm.update_trajectory(obj["track_id"], ent.position, event["timestamp"])
    predictions = wm.predict_collision(spatial)

    # 5b. Scene prediction divergence scoring
    divergence = get_scene_predictor().predict_and_score(
        event["camera_id"], event, spatial)
    if divergence.overall > 0.6:
        await broadcast_alert({
            "type": "prediction_divergence",
            "score": divergence.overall,
            "camera_id": event["camera_id"],
            "details": divergence.details,
        })
        try:
            get_event_bus().publish("prediction_divergence", {
                "score": divergence.overall, "camera_id": event["camera_id"],
                "details": divergence.details})
        except Exception:
            pass

    # 6. Process triggered rules → causal analysis → action engine
    alerts_sent = []
    for rr in rule_results:
        causal = None
        if rr.severity.value in ("high", "critical"):
            keyframes = [event["keyframe_b64"]] if event.get("keyframe_b64") else []
            graph_ctx = get_kg().get_recent_events(event["camera_id"], limit=5)
            causal = get_causal().analyze(
                {"anomaly_type": rr.rule_name, "severity": rr.severity.value,
                 "description": " → ".join(rr.explanation_chain)},
                keyframes, graph_ctx, event.get("audio_events"),
            )

        anomaly = {
            "event_id": event["event_id"],
            "camera_id": event["camera_id"],
            "severity": rr.severity.value,
            "anomaly_type": rr.rule_name,
            "description": " → ".join(rr.explanation_chain),
            "rule_id": rr.rule_id,
        }

        # Persist and deliver alert
        try:
            if get_semantic_dedup().is_duplicate(anomaly):
                continue
        except Exception:
            pass
        try:
            get_persistence().save_alert(anomaly)
        except Exception:
            pass
        try:
            get_webhook_manager().deliver_alert(anomaly)
        except Exception:
            pass

        # Explain alert
        try:
            explanation = get_explainer().explain(anomaly, event["camera_id"])
            anomaly["explanation"] = explanation
        except Exception:
            pass

        # Visual grounding
        try:
            grounded = get_visual_grounder().ground_anomaly(anomaly, event.get("keyframe_b64"), spatial)
            anomaly["grounded_boxes"] = grounded.bounding_boxes
        except Exception:
            pass

        notifications = get_action_engine().process_anomaly(anomaly, causal)
        alerts_sent.extend([n.subject for n in notifications])

        try:
            get_floor_plan().add_alert(
                event["event_id"], event["camera_id"],
                rr.severity.value, " → ".join(rr.explanation_chain))
        except Exception:
            pass

        await broadcast_alert({
            "type": "alert", "anomaly": anomaly,
            "causal": causal.__dict__ if causal else None,
        })

        # Publish to event bus
        try:
            get_event_bus().publish("alert", anomaly)
        except Exception:
            pass

    # 7. Broadcast predictions
    for pred in predictions:
        await broadcast_alert({"type": "prediction", "prediction": pred.__dict__})

    # 8. Ambient Intelligence Score
    ambient_score = None
    try:
        zone_id = event.get("zone_id", event["camera_id"])
        ambient = get_ambient_score().compute(zone_id, scene_state)
        ambient_score = ambient.score
        if ambient.level in ("warning", "critical"):
            await broadcast_alert({
                "type": "ambient_score",
                "zone_id": zone_id,
                "score": ambient.score,
                "level": ambient.level,
                "top_driver": ambient.top_driver,
            })
    except Exception:
        pass

    # 9. Anomaly contagion prediction (when alerts fired)
    if rule_results:
        try:
            contagion = get_contagion_network()
            zone_id = event.get("zone_id", event["camera_id"])
            for rr in rule_results:
                contagion.record_event(zone_id, rr.rule_name, event["timestamp"])
            warnings = contagion.predict_contagion(zone_id, rule_results[0].rule_name)
            for w in warnings[:3]:
                await broadcast_alert({
                    "type": "contagion_warning",
                    "at_risk_zone": w.zone_id,
                    "probability": w.risk_probability,
                    "expected_delay_s": w.expected_delay_s,
                    "source_zone": w.source_zone,
                    "recommendation": w.recommendation,
                })
        except Exception:
            pass

    # 10. Scene déjà vu detection (when anomaly detected)
    deja_vu_matches = []
    if rule_results:
        try:
            matches = get_deja_vu().find_similar(scene_state, top_k=3)
            for m in matches:
                deja_vu_matches.append({
                    "similarity": m.similarity,
                    "historical_time": m.historical_timestamp,
                    "narrative": m.narrative,
                })
                if m.similarity > 0.85:
                    await broadcast_alert({
                        "type": "deja_vu",
                        "similarity": m.similarity,
                        "narrative": m.narrative,
                        "historical_event": m.historical_event_type,
                    })
        except Exception:
            pass

    return {
        "status": "ok",
        "anomaly_score": anomaly_score,
        "ambient_score": ambient_score,
        "prediction_divergence": divergence.overall,
        "rules_triggered": len(rule_results),
        "tracker_alerts": len(tracker_alerts),
        "predictions": len(predictions),
        "alerts_sent": len(alerts_sent),
        "deja_vu_matches": len(deja_vu_matches),
    }


# --- Query & Chat ---

@app.post("/api/query")
async def query_video(req: QueryRequest):
    result = get_query_engine().query(req.query, req.camera_id)
    return {
        "answer": result.answer, "evidence": result.evidence,
        "confidence": result.confidence, "complexity": result.complexity.value,
        "latency_ms": result.latency_ms, "grounded": result.grounded,
    }


@app.post("/api/converse")
async def converse(req: ConverseRequest):
    """Live conversational camera agent with session memory."""
    resp = get_live_agent().converse(req.message, req.session_id, req.camera_id)
    return {
        "answer": resp.answer, "session_id": resp.session_id,
        "confidence": resp.confidence, "evidence": resp.evidence,
        "live_snapshot": {
            "entity_count": resp.live_snapshot.get("entity_count", 0),
            "zone_count": resp.live_snapshot.get("zone_count", 0),
        },
    }


# --- KPI ---

@app.get("/api/kpi/{camera_id}")
async def get_kpis(camera_id: str):
    return get_kpi_engine().compute_daily_kpis(camera_id)


@app.get("/api/kpi/{camera_id}/summary")
async def get_kpi_summary(camera_id: str):
    engine = get_kpi_engine()
    kpis = engine.compute_daily_kpis(camera_id)
    trends = engine.compute_trends(kpis, kpis)
    report = engine.generate_summary(kpis, trends, [], [])
    return {
        "narrative": report.narrative, "metrics": report.metrics,
        "trends": report.trends, "recommendations": report.recommendations,
    }


# --- Rules ---

@app.post("/api/rules")
async def add_rule(payload: RulePayload):
    from services.reasoning.engine import Rule, Severity
    rule = Rule(
        rule_id=payload.rule_id, name=payload.name,
        severity=Severity(payload.severity),
        conditions=payload.conditions, action=payload.action,
    )
    get_reasoner().add_rule(rule)
    return {"status": "ok", "rule_id": rule.rule_id}


# --- Spatial ---

@app.get("/api/spatial/{camera_id}")
async def get_spatial_state(camera_id: str):
    spatial = get_spatial()
    entities = [
        {"track_id": e.track_id, "class_name": e.class_name,
         "position": e.position.tolist(), "velocity": e.velocity.tolist()}
        for e in spatial._entities.values()
    ]
    zones = [
        {"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon, "type": z.zone_type}
        for z in spatial._zones.values()
    ]
    return {"entities": entities, "zones": zones}


# --- Alerts ---

@app.post("/api/alerts/{event_id}/ack")
async def acknowledge_alert(event_id: str):
    get_action_engine().acknowledge(event_id)
    return {"status": "acknowledged"}


@app.get("/api/graph/{camera_id}/recent")
async def get_recent_graph(camera_id: str, limit: int = 50):
    return get_kg().get_recent_events(camera_id, limit)


@app.get("/api/graph/journey/{track_id}")
async def get_entity_journey(track_id: str, since_hours: int = 24):
    """Temporal path: trace entity's full journey across cameras with transit times."""
    return get_kg().get_entity_journey(track_id, since_hours)


@app.get("/api/graph/path/{track_id}")
async def get_temporal_path(track_id: str, start: float = 0, end: float = 0):
    """Query: where was entity X between time A and time B?"""
    if end == 0:
        end = time.time()
    if start == 0:
        start = end - 3600
    return get_kg().get_temporal_path(track_id, start, end)


@app.get("/api/graph/causal/{event_id}")
async def get_causal_chain(event_id: str, depth: int = 5):
    """Follow causal/temporal edges to build event chain."""
    return get_kg().get_causal_chain(event_id, depth)


# --- Custom Trackers ---

@app.post("/api/trackers")
async def create_tracker(req: TrackerCreateRequest):
    tracker = get_tracker_engine().create_from_text(req.description)
    return {
        "tracker_id": tracker.tracker_id, "name": tracker.name,
        "description": tracker.description, "severity": tracker.severity,
        "status": tracker.status.value, "detection_classes": tracker.detection_classes,
    }


@app.get("/api/trackers")
async def list_trackers():
    return [
        {"tracker_id": t.tracker_id, "name": t.name, "description": t.description,
         "severity": t.severity, "status": t.status.value,
         "trigger_count": t.trigger_count, "created_at": t.created_at,
         "last_triggered": t.last_triggered}
        for t in get_tracker_engine().list_trackers()
    ]


@app.patch("/api/trackers/{tracker_id}")
async def update_tracker(tracker_id: str, req: TrackerUpdateRequest):
    engine = get_tracker_engine()
    if req.status == "paused":
        engine.pause_tracker(tracker_id)
    elif req.status == "active":
        engine.resume_tracker(tracker_id)
    return {"status": "ok"}


@app.delete("/api/trackers/{tracker_id}")
async def delete_tracker(tracker_id: str):
    get_tracker_engine().delete_tracker(tracker_id)
    return {"status": "deleted"}


# --- Story / Profile ---

@app.get("/api/story/{entity_id}")
async def get_story(entity_id: str, since_hours: int = 24):
    story = get_story_builder().build_story(entity_id, since_hours)
    return get_story_builder().export_report(story)


@app.get("/api/profile/{entity_id}")
async def get_profile(entity_id: str, since_hours: int = 168):
    profiler = get_entity_profiler()
    profile = profiler.get_profile(entity_id, since_hours)
    return profiler.to_dict(profile)


# --- Investigation ---

@app.post("/api/investigate/{entity_id}")
async def investigate_entity(entity_id: str, since_hours: int = 48):
    """Run autonomous investigation on an entity."""
    report = get_investigator().investigate(entity_id, since_hours)
    return get_investigator().export_report(report)


# --- Video Synopsis ---

@app.get("/api/synopsis/{camera_id}")
async def get_synopsis(camera_id: str, hours: float = 4.0):
    """Generate forensic video synopsis for a camera."""
    synopsis = get_synopsis_engine().generate(camera_id, hours=hours)
    return get_synopsis_engine().export(synopsis)


# --- Digital Twin Floor Plan ---

@app.get("/api/floorplan")
async def get_floorplan():
    """Get real-time 2D digital twin floor plan snapshot."""
    fp = get_floor_plan()
    snapshot = fp.get_snapshot()
    return fp.export_snapshot(snapshot)


# --- Forensic Search ---

@app.post("/api/search")
async def forensic_search(req: ForensicSearchRequest):
    """Privacy-first forensic search across all footage."""
    engine = get_forensic_search()
    response = engine.search(
        req.query, req.user_id, req.access_level,
        req.camera_id, req.time_range_hours,
    )
    return engine.export_response(response)


@app.post("/api/search/unlock")
async def unlock_search_result(req: UnlockRequest):
    """Unlock a specific search result — requires supervisor+ access."""
    engine = get_forensic_search()
    result = engine.unlock_result(
        req.result_id, req.user_id, req.access_level, req.justification)
    if not result:
        return {"status": "denied", "message": "Insufficient access level or result not found"}
    return {
        "status": "unlocked", "result_id": result.result_id,
        "entity_id": result.entity_id, "keyframe_ref": result.keyframe_ref,
        "description": result.description,
    }


@app.get("/api/search/audit")
async def get_search_audit(limit: int = 100):
    """Get forensic search audit log."""
    return get_forensic_search().get_audit_log(limit)


# --- Cameras CRUD ---

@app.get("/api/cameras")
async def list_cameras():
    from dataclasses import asdict
    return [asdict(cam) for cam in get_camera_manager().list_cameras()]


@app.post("/api/cameras")
async def add_camera(payload: CameraPayload):
    from services.cameras.manager import CameraConfig
    cfg = CameraConfig(
        camera_id=payload.camera_id, name=payload.name, rtsp_url=payload.rtsp_url,
        site_x=payload.site_x, site_y=payload.site_y,
        rotation_deg=payload.rotation_deg, fov_deg=payload.fov_deg,
        tenant_id=payload.tenant_id, tags=payload.tags,
    )
    get_camera_manager().add_camera(cfg)
    return {"status": "ok", "camera_id": payload.camera_id}


@app.patch("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, body: dict):
    get_camera_manager().update_camera(camera_id, body)
    return {"status": "ok", "camera_id": camera_id}


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    get_camera_manager().remove_camera(camera_id)
    return {"status": "deleted"}


# --- Webhooks CRUD ---

@app.get("/api/webhooks")
async def list_webhooks():
    return get_webhook_manager().list_webhooks()


@app.post("/api/webhooks")
async def register_webhook(payload: WebhookPayload):
    wh = get_webhook_manager().register(payload.model_dump())
    return {"status": "ok", "webhook": wh}


@app.patch("/api/webhooks/{webhook_id}")
async def update_webhook(webhook_id: str, body: dict):
    get_webhook_manager().update(webhook_id, body)
    return {"status": "ok", "webhook_id": webhook_id}


@app.delete("/api/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    get_webhook_manager().unregister(webhook_id)
    return {"status": "deleted"}


@app.get("/api/webhooks/{webhook_id}/logs")
async def get_webhook_logs(webhook_id: str):
    return get_webhook_manager().get_delivery_log(webhook_id)


# --- Zones ---

@app.post("/api/zones")
async def add_zone(payload: ZonePayload):
    spatial = get_spatial()
    spatial.add_zone(payload.zone_id, payload.name, payload.polygon, payload.zone_type)
    if payload.capacity > 0:
        get_occupancy().set_zone_capacity(payload.zone_id, payload.capacity)
    return {"status": "ok", "zone_id": payload.zone_id}


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str):
    get_spatial()._zones.pop(zone_id, None)
    return {"status": "deleted"}


# --- Occupancy ---

@app.get("/api/occupancy/{camera_id}")
async def get_camera_occupancy(camera_id: str):
    result = get_occupancy().get_current(camera_id)
    return dict(result) if hasattr(result, "__dict__") else result


@app.get("/api/occupancy")
async def get_site_occupancy():
    return get_occupancy().get_site_summary()


# --- Compliance ---

@app.get("/api/compliance/{camera_id}")
async def get_compliance(camera_id: str):
    spatial = get_spatial()
    scene_state = {
        "objects": [], "audio_events": [], "spatial": spatial,
        "timestamp": time.time(), "camera_id": camera_id,
        "anomaly_score": 0, "keyframe_b64": None,
    }
    return get_safety_monitor().evaluate(scene_state)


@app.post("/api/compliance/sop")
async def add_sop_rule(payload: SOPRulePayload):
    get_safety_monitor().add_sop_rule(payload.model_dump())
    return {"status": "ok", "rule_id": payload.rule_id}


# --- LPR ---

@app.get("/api/lpr/log/{camera_id}")
async def get_lpr_log(camera_id: str):
    return get_lpr().get_log(camera_id)


@app.post("/api/lpr/watchlist")
async def add_to_watchlist(payload: WatchlistPayload):
    get_lpr().add_to_watchlist(payload.plate, payload.reason, payload.severity)
    return {"status": "ok", "plate": payload.plate}


@app.get("/api/lpr/watchlist")
async def get_watchlist():
    return list(get_lpr().watchlist.items())


@app.delete("/api/lpr/watchlist/{plate}")
async def remove_from_watchlist(plate: str):
    get_lpr().remove_from_watchlist(plate)
    return {"status": "deleted"}


# --- Reports ---

@app.get("/api/reports/schedules")
async def list_report_schedules():
    return get_report_scheduler().list_schedules()


@app.post("/api/reports/schedules")
async def add_report_schedule(payload: ReportSchedulePayload):
    sched = get_report_scheduler().add_schedule(payload.model_dump())
    return {"status": "ok", "schedule": sched}


@app.delete("/api/reports/schedules/{schedule_id}")
async def remove_report_schedule(schedule_id: str):
    get_report_scheduler().remove_schedule(schedule_id)
    return {"status": "deleted"}


@app.post("/api/reports/generate/{schedule_id}")
async def generate_report(schedule_id: str):
    report = get_report_scheduler().generate_report(schedule_id)
    return {"status": "ok", "report": report}


# --- Digest ---

@app.get("/api/digest")
async def get_digest(hours: int = 24, camera_id: str | None = None):
    return get_alert_digest().generate(hours=hours, camera_id=camera_id)


# --- Auth ---

@app.post("/api/auth/keys")
async def create_api_key(payload: ApiKeyPayload):
    key = get_auth_provider().create_api_key(
        tenant_id=payload.tenant_id, user_id=payload.user_id,
        access_level=payload.access_level, cameras=payload.cameras,
    )
    return {"status": "ok", "key": key}


@app.get("/api/auth/keys")
async def list_api_keys():
    try:
        return get_persistence().get_api_keys()
    except Exception:
        return []


@app.delete("/api/auth/keys/{key_hash}")
async def delete_api_key(key_hash: str):
    get_persistence().delete_api_key(key_hash)
    return {"status": "deleted"}


# --- Crowd Analytics ---

@app.get("/api/crowd/{camera_id}")
async def get_crowd_analysis(camera_id: str):
    return get_crowd_analyzer()(get_spatial())


# --- Scene Graph ---

@app.get("/api/scene-graph/{camera_id}")
async def get_scene_graph_state(camera_id: str):
    sg = get_scene_graph()
    return {"edges": sg.get_edges(), "transitions": sg.get_transitions()}


# --- Contextual Normality ---

@app.get("/api/normality/{camera_id}")
async def get_normality(camera_id: str):
    return get_contextual_normality().get_expected(camera_id, time.time())


# --- Counterfactual Explanation ---

@app.post("/api/explain/counterfactual/{event_id}")
async def get_counterfactual(event_id: str):
    spatial = get_spatial()
    scene_state = {
        "objects": [], "audio_events": [], "spatial": spatial,
        "timestamp": time.time(), "camera_id": "unknown",
        "anomaly_score": 0, "keyframe_b64": None,
    }
    cfe = get_counterfactual_explainer()
    results = cfe.explain({"event_id": event_id}, scene_state)
    return {"counterfactuals": [r.__dict__ if hasattr(r, "__dict__") else r for r in results],
            "explanation": cfe.format_explanation(results)}


# --- Collective Anomaly ---

@app.get("/api/anomaly/collective/{camera_id}")
async def get_collective_anomalies(camera_id: str):
    results = get_collective_detector().detect(camera_id)
    return [r.__dict__ if hasattr(r, "__dict__") else r for r in results]


# --- What-If Simulation ---

@app.post("/api/simulate")
async def run_simulation(payload: SimulatePayload):
    from services.simulation.what_if import WhatIfScenario
    scenario = WhatIfScenario(
        scenario_type=payload.scenario_type,
        parameters=payload.parameters,
        description=payload.description,
    )
    result = get_what_if().simulate(scenario)
    return result.__dict__ if hasattr(result, "__dict__") else result


# --- System Health ---

@app.get("/api/system/health")
async def get_system_health():
    evaluator = get_self_evaluator()
    return {
        "metrics": evaluator.get_metrics(),
        "report": evaluator.generate_health_report(),
    }


# --- Dwell Analytics ---

@app.get("/api/analytics/dwell/{zone_id}")
async def get_dwell_analytics(zone_id: str):
    return get_dwell_analyzer().get_zone_analytics(zone_id)


# --- Heatmap ---

@app.get("/api/heatmap/{camera_id}")
async def get_heatmap(camera_id: str, type: str = "movement", hour: int | None = None):
    engine = get_heatmap_engine()
    return {
        "grid": engine.generate(type, hour),
        "hot_spots": engine.get_hot_spots(),
        "grid_size": engine._grid_size,
    }


# --- Fleet ---

@app.post("/api/fleet/heartbeat")
async def fleet_heartbeat(payload: HeartbeatPayload):
    get_fleet_manager().register_heartbeat(payload.device_id, payload.model_dump())
    return {"status": "ok", "device_id": payload.device_id}


@app.get("/api/fleet/status")
async def get_fleet_status():
    return get_fleet_manager().get_fleet_status()


# --- Proactive Agent ---

@app.get("/api/proactive/priorities")
async def get_proactive_priorities():
    return {"priorities": get_proactive_agent()._priorities,
            "watch_entities": get_proactive_agent()._watch_entities}


@app.post("/api/proactive/priorities")
async def set_proactive_priorities(payload: PrioritiesPayload):
    get_proactive_agent().set_priorities(payload.priorities)
    return {"status": "ok", "priorities": payload.priorities}


@app.post("/api/proactive/watch")
async def add_watch_entity(payload: WatchEntityPayload):
    agent = get_proactive_agent()
    if payload.entity_id not in agent._watch_entities:
        agent._watch_entities.append(payload.entity_id)
    return {"status": "ok", "watch_entities": agent._watch_entities}


# --- Consent & GDPR ---

@app.get("/api/consent/{entity_id}")
async def get_consent(entity_id: str):
    return get_consent_manager().get_data_subject_report(entity_id)


@app.post("/api/consent")
async def record_consent(payload: ConsentPayload):
    rec = get_consent_manager().record_consent(
        payload.entity_id, payload.consent_type, payload.granted,
        payload.purpose, payload.legal_basis, payload.duration_days,
    )
    return {"status": "ok", "record": rec.__dict__}


@app.get("/api/consent/dpia-report")
async def get_dpia_report():
    return get_consent_manager().generate_dpia_report()


@app.post("/api/retention/enforce")
async def enforce_retention():
    return get_consent_manager().enforce_retention()


# --- Evidence Package ---

@app.post("/api/evidence/build")
async def build_evidence(req: EvidenceRequest):
    pkg = get_evidence_packager().build_package(
        req.incident_id, req.camera_ids, req.time_range_hours, req.created_by)
    return get_evidence_packager().export_json(pkg)


@app.get("/api/evidence/{package_id}/html")
async def get_evidence_html(package_id: str):
    from fastapi.responses import HTMLResponse
    pkg = get_evidence_packager().build_package(package_id, [])
    return HTMLResponse(get_evidence_packager().export_html(pkg))


# --- Shift Handover ---

@app.get("/api/shift/briefing")
async def get_shift_briefing(hours: float = 8):
    briefing = get_shift_agent().generate_briefing(hours)
    return get_shift_agent().export_briefing(briefing)


# --- Camera Programs ---

@app.post("/api/camera-programs")
async def create_camera_program(req: CameraProgramRequest):
    prog = get_camera_programmer().parse_instruction(req.instruction)
    get_camera_programmer().add_program(prog)
    return {"status": "ok", "program": prog.__dict__}


@app.get("/api/camera-programs")
async def list_camera_programs():
    return [p.__dict__ for p in get_camera_programmer().list_programs()]


@app.delete("/api/camera-programs/{program_id}")
async def delete_camera_program(program_id: str):
    get_camera_programmer().remove_program(program_id)
    return {"status": "deleted"}


# --- Adversarial Robustness ---

@app.get("/api/adversarial/health")
async def get_adversarial_health():
    return get_adversarial_monitor().get_camera_health()


@app.get("/api/adversarial/{camera_id}")
async def get_adversarial_camera(camera_id: str):
    health = get_adversarial_monitor().get_camera_health()
    return health.get(camera_id, {"status": "no_data"})


# --- Entity Journey ---

@app.get("/api/journey/active")
async def get_active_journeys():
    return get_journey_visualizer().get_active_journeys()


@app.get("/api/journey/{entity_id}")
async def get_entity_journey(entity_id: str, since_hours: int = 24):
    journey = get_journey_visualizer().build_journey(entity_id, since_hours)
    return journey.__dict__ if hasattr(journey, "__dict__") else journey


@app.get("/api/journey/{entity_id}/path")
async def get_journey_path(entity_id: str):
    return get_journey_visualizer().get_floor_plan_path(entity_id)


# --- Retail Analytics ---

@app.get("/api/retail/metrics")
async def get_retail_metrics(store_id: str = "default"):
    return get_retail_analytics().get_metrics(store_id).__dict__


@app.post("/api/retail/configure")
async def configure_retail(payload: RetailConfigPayload):
    get_retail_analytics().configure(
        payload.entry_zones, payload.checkout_zones, payload.queue_zones)
    return {"status": "ok"}


@app.get("/api/retail/funnel")
async def get_retail_funnel():
    return get_retail_analytics().get_funnel()


@app.get("/api/retail/attention")
async def get_retail_attention():
    return get_retail_analytics().get_attention_heatmap()


# --- Event Bus ---

@app.post("/api/events/subscribe")
async def subscribe_events(payload: EventSubscriptionPayload):
    sub = get_event_bus().subscribe(
        payload.subscriber_name, payload.event_types,
        payload.filters, payload.callback_url)
    return {"status": "ok", "subscription": sub.__dict__}


@app.get("/api/events/subscriptions")
async def list_event_subscriptions():
    return [s.__dict__ for s in get_event_bus().list_subscriptions()]


@app.delete("/api/events/subscriptions/{subscription_id}")
async def unsubscribe_events(subscription_id: str):
    get_event_bus().unsubscribe(subscription_id)
    return {"status": "deleted"}


@app.get("/api/events/log")
async def get_event_log(event_type: str | None = None, limit: int = 100):
    return get_event_bus().get_event_log(event_type, limit)


# --- Visual Grounding ---

@app.get("/api/grounding/{camera_id}")
async def get_grounded_entities(camera_id: str):
    return get_visual_grounder().ground_entities(get_spatial(), camera_id)


# --- Resource Optimizer ---

@app.get("/api/staffing/predict")
async def predict_demand(camera_id: str = "all", hours: int = 24):
    return get_resource_optimizer().predict_demand(camera_id, hours)


@app.get("/api/staffing/recommend")
async def recommend_staffing():
    return [r.__dict__ for r in get_resource_optimizer().recommend_staffing()]


@app.get("/api/staffing/plan")
async def get_resource_plan():
    return get_resource_optimizer().get_resource_plan()


# --- Behavioral Stress ---

@app.get("/api/behavioral/stress")
async def get_stress_assessments():
    return {k: v.__dict__ for k, v in get_stress_detector()._assessments.items()
            if v.stress_level > 0.3}


# --- Federation ---

@app.post("/api/federation/peers")
async def register_federation_peer(payload: FederationPeerPayload):
    get_federated_intel().register_peer(payload.site_id, payload.callback_url)
    return {"status": "ok"}


@app.get("/api/federation/peers")
async def list_federation_peers():
    return get_federated_intel().list_peers()


@app.post("/api/federation/share")
async def share_federation_alert(payload: FederationAlertPayload):
    alert = get_federated_intel().share_entity_alert(
        payload.entity_id, payload.severity, payload.description)
    return {"status": "ok", "alert": alert.__dict__}


@app.get("/api/federation/alerts")
async def get_federation_alerts(since_hours: int = 24):
    return get_federated_intel().get_shared_alerts(since_hours)


@app.get("/api/federation/check/{entity_id}")
async def check_federation_entity(entity_id: str):
    return get_federated_intel().check_entity(entity_id)


@app.get("/api/federation/status")
async def get_federation_status():
    return get_federated_intel().get_federation_status()


# --- Natural Language Rule Compiler ---

class NLRuleRequest(BaseModel):
    instruction: str


class RuleToggleRequest(BaseModel):
    active: bool


@app.post("/api/nl-rules/compile")
async def compile_nl_rule(req: NLRuleRequest):
    """Compile a natural language instruction into a live detection rule."""
    compiled = get_nl_rule_compiler().compile(req.instruction)
    return {
        "rule_id": compiled.rule_id,
        "name": compiled.name,
        "severity": compiled.severity,
        "conditions": compiled.conditions,
        "description": compiled.description,
        "source_text": compiled.source_text,
    }


@app.get("/api/nl-rules")
async def list_nl_rules():
    return get_nl_rule_compiler().list_rules()


@app.delete("/api/nl-rules/{rule_id}")
async def delete_nl_rule(rule_id: str):
    ok = get_nl_rule_compiler().delete_rule(rule_id)
    if not ok:
        raise HTTPException(404, "Rule not found")
    return {"status": "deleted"}


@app.patch("/api/nl-rules/{rule_id}/toggle")
async def toggle_nl_rule(rule_id: str, req: RuleToggleRequest):
    ok = get_nl_rule_compiler().toggle_rule(rule_id, req.active)
    if not ok:
        raise HTTPException(404, "Rule not found")
    return {"status": "ok", "active": req.active}


@app.get("/api/nl-rules/{rule_id}/explain")
async def explain_nl_rule(rule_id: str):
    return {"explanation": get_nl_rule_compiler().explain_rule(rule_id)}


# --- Ambient Intelligence Score ---

@app.get("/api/ambient/{zone_id}")
async def get_ambient_zone_score(zone_id: str):
    """Get the fused multi-modal safety score for a zone."""
    spatial = get_spatial()
    scene_state = {
        "objects": [{"class_name": e.class_name, "bbox": e.bbox.tolist() if hasattr(e.bbox, 'tolist') else e.bbox,
                     "track_id": eid}
                    for eid, e in spatial._entities.items()],
        "audio_events": [],
        "timestamp": time.time(),
        "camera_id": zone_id,
    }
    result = get_ambient_score().compute(zone_id, scene_state)
    return {
        "zone_id": result.zone_id,
        "score": result.score,
        "level": result.level,
        "top_driver": result.top_driver,
        "contributions": [
            {"signal": c.signal_name, "value": round(c.raw_value, 3),
             "weight": round(c.weight, 3), "description": c.description}
            for c in result.contributions
        ],
    }


@app.get("/api/ambient/{zone_id}/trend")
async def get_ambient_trend(zone_id: str, minutes: int = 30):
    return get_ambient_score().get_trend(zone_id, minutes)


@app.get("/api/ambient")
async def get_all_ambient_scores():
    """Get ambient scores for all active zones."""
    spatial = get_spatial()
    zones = list(getattr(spatial, "_zones", {}).keys())
    if not zones:
        zones = ["default"]
    results = []
    for zone_id in zones:
        try:
            scene_state = {
                "objects": [{"class_name": e.class_name, "bbox": [],
                             "track_id": eid}
                            for eid, e in spatial._entities.items()],
                "audio_events": [], "timestamp": time.time(),
                "camera_id": zone_id,
            }
            result = get_ambient_score().compute(zone_id, scene_state)
            results.append({
                "zone_id": result.zone_id, "score": result.score,
                "level": result.level, "top_driver": result.top_driver,
            })
        except Exception:
            pass
    return results


# --- Predictive Path Interception ---

@app.get("/api/interception/active")
async def get_active_interceptions():
    """Get current predictive interception alerts."""
    scene_state = {
        "objects": [], "audio_events": [],
        "timestamp": time.time(), "camera_id": "all",
    }
    alerts = get_predictive_interceptor().evaluate(scene_state)
    return [
        {
            "entity_id": a.entity_id,
            "target_zone": a.target_zone_name,
            "eta_seconds": a.estimated_arrival_s,
            "confidence": a.confidence,
            "recommendation": a.recommendation,
            "current_position": a.current_position,
            "predicted_entry": a.predicted_entry_point,
        }
        for a in alerts
    ]


@app.get("/api/interception/trajectories")
async def get_predicted_trajectories():
    """Get all predicted trajectories for floor plan overlay."""
    return get_predictive_interceptor().get_all_trajectories()


# --- Gait DNA (Behavioral Biometrics) ---

@app.get("/api/gait/gallery")
async def get_gait_gallery():
    """List all stored gait fingerprints."""
    return get_gait_dna().get_gallery_summary()


@app.get("/api/gait/{entity_id}")
async def get_gait_fingerprint(entity_id: str):
    fp = get_gait_dna().get_fingerprint(entity_id)
    if not fp:
        raise HTTPException(404, "No fingerprint for this entity")
    return {
        "entity_id": fp.entity_id,
        "confidence": round(fp.confidence, 2),
        "observations": fp.observation_count,
        "first_seen": fp.first_seen,
        "last_updated": fp.last_updated,
    }


@app.get("/api/gait/{entity_id}/match")
async def match_gait(entity_id: str):
    """Find entities with similar movement patterns (cross-session re-ID)."""
    matches = get_gait_dna().match(entity_id)
    return [
        {
            "matched_entity": m.matched_entity_id,
            "similarity": m.similarity,
            "match_type": m.match_type,
        }
        for m in matches
    ]


# --- Anomaly Contagion Network ---

@app.get("/api/contagion/graph")
async def get_contagion_graph():
    """Get the full zone-to-zone anomaly propagation graph."""
    return get_contagion_network().get_graph()


@app.get("/api/contagion/{zone_id}/predict")
async def predict_zone_contagion(zone_id: str):
    """Predict which zones are at risk after an event in this zone."""
    warnings = get_contagion_network().predict_contagion(zone_id)
    return [
        {
            "zone_id": w.zone_id,
            "zone_name": w.zone_name,
            "probability": w.risk_probability,
            "expected_delay_s": w.expected_delay_s,
            "recommendation": w.recommendation,
        }
        for w in warnings
    ]


@app.get("/api/contagion/{zone_id}/profile")
async def get_contagion_profile(zone_id: str):
    """Get incoming and outgoing risk profile for a zone."""
    return get_contagion_network().get_zone_risk_profile(zone_id)


# --- Scene Déjà Vu ---

@app.get("/api/deja-vu/{camera_id}")
async def find_deja_vu(camera_id: str, min_similarity: float = 0.7,
                       top_k: int = 5, same_camera: bool = False):
    """Find historical scenes similar to the current one."""
    spatial = get_spatial()
    scene_state = {
        "objects": [{"class_name": e.class_name, "bbox": e.bbox.tolist() if hasattr(e.bbox, 'tolist') else e.bbox,
                     "track_id": eid}
                    for eid, e in spatial._entities.items()],
        "audio_events": [], "timestamp": time.time(),
        "camera_id": camera_id,
    }
    matches = get_deja_vu().find_similar(
        scene_state, top_k=top_k, same_camera_only=same_camera,
        min_similarity=min_similarity)
    return [
        {
            "similarity": m.similarity,
            "historical_camera": m.historical_camera_id,
            "historical_time": m.historical_timestamp,
            "historical_event": m.historical_event_type,
            "narrative": m.narrative,
            "time_ago_hours": m.time_ago_hours,
        }
        for m in matches
    ]


@app.get("/api/deja-vu/stats")
async def get_deja_vu_stats():
    return get_deja_vu().get_stats()


# --- WebSocket ---

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    ws_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections.remove(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": "0.5.0"}
