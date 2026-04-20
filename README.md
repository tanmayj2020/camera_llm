# VisionBrain — Intelligent CCTV Analytics Platform

An industry-agnostic platform that gives every CCTV camera a "brain." With **55+ service modules**, **98+ API endpoints**, and **64 unit tests**, VisionBrain combines YOLO-World v2 open-vocabulary detection, Qwen2.5-VL edge vision, Depth Anything V2 spatial understanding, CLIP-ReID cross-camera tracking, GTE-Qwen2 embeddings, SigLIP visual encoding, BEATs/AST audio classification, temporal knowledge hypergraphs, neuro-symbolic reasoning, agentic action engines, Gemini 2.5 Pro causal analysis, social force crowd modeling, dynamic scene graphs, contextual normality baselines, counterfactual explanations, visual grounding, behavioral stress detection, predictive staffing, cross-site intelligence federation, and GDPR-native consent management into a single coherent intelligence layer. Novel features include **Natural Language Rule Compilation**, **Ambient Intelligence Scoring** (9-signal Bayesian fusion), **Predictive Interception**, **Gait DNA Biometrics**, **Anomaly Contagion Networks**, and **Déjà Vu scene matching** — capabilities not available in any existing CCTV analytics platform.

---

## Features

- **Edge AI Pipeline** — Adaptive RTSP capture, open-vocab detection (YOLOv8/YOLOE), pose estimation, audio classification, privacy engine (face/plate blur), edge VLM (InternVL-2B)
- **Temporal Knowledge Graph** — Neo4j-backed bi-temporal KG with Graphiti-style edges, HyperGraphRAG (Qdrant + Neo4j), entity resolution via embeddings
- **Neuro-Symbolic Reasoning** — Rule engine with temporal/spatial/audio conditions, behavioral intent recognition (casing, pacing, evasive, aggressive), causal VLM analysis with hallucination guardrails
- **World Model & Prediction** — Trajectory prediction, crowd dynamics, collision detection, scene anomaly via prediction divergence
- **Social Force Crowd Modeling** — Pairwise repulsive/attractive forces, automatic group detection from trajectory correlation, crowd pressure monitoring with stampede risk scoring
- **Dynamic Scene Graph** — Real-time relationship tracking (near, approaching, following, blocking, handing_object_to), suspicious transition detection when relationships change
- **Context-Aware Normality** — Separate baselines per (hour, day-of-week) bucket with holiday/weekend tolerance multipliers — reduces false positives vs flat baselines
- **Counterfactual Explanations** — "Would NOT trigger if: duration < 3 min, OR time were 8 am–6 pm, OR zone were lobby" — systematic factor modification to explain why alerts fired
- **Visual Grounding** — GLaMM-inspired bounding-box explanations that highlight exactly which pixels drove each alert, linking natural-language rationale to spatial regions
- **Predictive Staffing** — Demand forecasting from historical occupancy patterns plus guard-count recommendations, shift-level staffing plans with coverage scoring
- **Behavioral Stress Detection** — Fidgeting, pacing, erratic movement analysis via pose-sequence features; per-entity stress scoring with temporal trend tracking
- **Cross-Site Intelligence Federation** — Privacy-preserving entity sharing across sites using embedding similarity; federated alert propagation without raw PII exchange
- **Collective Anomaly Detection** — Coordinated movement, systematic coverage of exits, distraction patterns, temporal clustering — patterns only visible across multiple individually-normal events
- **What-If Simulation** — Close a zone, add a camera, change a threshold, add a barrier — simulate forward impact on crowd flow, alert rates, and coverage
- **System Self-Evaluation** — FP rate analysis per rule, alert-to-ack time, coverage gaps, model drift — LLM-generated health reports with recommendations
- **Semantic Alert Deduplication** — Embedding-based similarity (sentence-transformers with keyword fallback), suppresses duplicate alerts within time/location window
- **Agentic Action Engine** — Reflexion self-critique, counterfactual simulation, multi-step playbooks, outcome learning, intelligent voice deterrence (4 escalation levels)
- **Autonomous Investigation** — Auto-traces entities across cameras, finds associates, builds evidence timeline, generates investigation report with risk assessment
- **Live Conversational Agent** — Session-persistent chat with cameras, answers "what's happening right now?", cross-references entity profiles
- **Proactive Agent** — Priority-driven watch system that monitors entities/zones and pushes alerts before operators ask
- **Forensic Video Synopsis** — BriefCam-style compression of hours into segments, LLM-narrated
- **Privacy-First Forensic Search** — 4 access levels, PII anonymization, unlock with audit trail
- **Workplace Safety & Compliance** — PPE detection, slip-and-fall, SOP sequence monitoring
- **License Plate Recognition** — VLM-based OCR, vehicle fingerprinting, watchlist alerts
- **Digital Twin Floor Plan** — 2D bird's-eye site map, real-time entity positions, zone occupancy, heatmap overlay
- **Occupancy Analytics** — Real-time people counting, capacity alerts, historical charts
- **Custom Trackers** — Natural language rule creation ("alert when someone enters parking after 10 pm")
- **Webhook Integrations** — Slack, PagerDuty, Teams, SIEM, generic HTTP with HMAC signing
- **Scheduled Reports & Digests** — Daily/weekly KPI summaries, alert digests with LLM analysis
- **Multi-Tenant Auth** — API key + JWT, camera-level isolation
- **Continual Learning** — Concept drift detection, federated learning, synthetic data flywheel
- **Three-Layer Agent Memory** — Episodic → semantic → procedural auto-promotion
- **Evidence Package Builder** — Court-ready HTML evidence bundles with chain-of-custody metadata
- **Shift Handover Briefing** — LLM-generated end-of-shift summaries with pending items and risk outlook
- **NL Camera Programming** — Define camera behaviors in plain English; compiled to executable rule sets
- **Adversarial Robustness** — Patch-attack detection, input perturbation scoring, per-camera robustness health
- **Entity Journey Mapping** — Cross-camera path visualization with dwell times, zone transitions, and animated replay
- **Retail Analytics** — Footfall funnels, shelf attention heatmaps, conversion metrics, zone dwell analysis
- **Event Bus** — Pub/Sub-style internal event subscriptions with filtering and delivery logs
- **Environment Detection** — Lighting, weather, and scene-condition classification for adaptive thresholds
- **Tailgate Detection** — Access-point monitoring for unauthorized piggyback entry
- **Dwell Analytics** — Per-zone dwell-time distributions with anomaly flagging
- **Heatmap Engine** — Spatial activity heatmaps per camera with configurable time windows
- **Fleet Management** — Edge device heartbeat monitoring, health dashboard, remote diagnostics
- **Audio-Visual Fusion** — Joint reasoning over audio events and visual detections for richer context
- **GDPR Consent Management** — Per-entity consent records, DPIA report generation, automated retention enforcement

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EDGE LAYER  (per site · Python)                                         │
│                                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐ │
│  │ Adaptive  │  │ YOLOE     │  │ Pose      │  │ Audio    │  │Privacy │ │
│  │ RTSP      │─▶│ Detection │─▶│ Estimator │  │ Classify │  │Engine  │ │
│  │ Capture   │  │ + Edge VLM│  │           │  │          │  │(blur)  │ │
│  └───────────┘  └─────┬─────┘  └───────────┘  └──────────┘  └────────┘ │
│                       ▼                                                  │
│              Event Emitter ──▶ GCP Pub/Sub                               │
└──────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CLOUD LAYER  (FastAPI + Go · 49 service modules)                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Event Processor (Go) ──▶ fan-out to all downstream services       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ── Core Intelligence ──────────────────────────────────────────────── │
│  │ Temporal KG │ Spatial Memory │ Baseline Learner │ World Model     │  │
│  │ HyperGraph  │ Scene Graph    │ Crowd SFM        │ Causal VLM      │  │
│                                                                          │
│  ── Reasoning & Anomaly ────────────────────────────────────────────── │
│  │ Neuro-Symbolic │ Counterfactual │ Contextual Normality             │  │
│  │ Collective     │ Visual Ground. │ Behavioral Stress │ Environment  │  │
│                                                                          │
│  ── Agents & Actions ───────────────────────────────────────────────── │
│  │ Action Engine  │ Investigation │ Conversational │ Proactive Agent  │  │
│  │ Query Engine   │ Story Builder │ Camera Program │ Simulation       │  │
│                                                                          │
│  ── Analytics & Operations ─────────────────────────────────────────── │
│  │ Occupancy  │ Dwell Time │ Heatmaps  │ Retail Analytics │ LPR      │  │
│  │ Compliance │ Tailgate   │ Fleet Mgr │ Predictive Staff │ KPI      │  │
│                                                                          │
│  ── Platform Services ──────────────────────────────────────────────── │
│  │ Persistence │ Auth     │ Cameras   │ Webhooks  │ Reports          │  │
│  │ Event Bus   │ Evidence │ Shift     │ Consent   │ Retention        │  │
│  │ Continual   │ Dedup    │ Self-Eval │ Adversar. │ Federation       │  │
│  │ Audio-Visual│ Synopsis │ Forensic  │ Digital Twin │ Entity Journey│  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD LAYER  (Next.js 14 · 17 pages)                               │
│                                                                          │
│  Home │ Cameras │ Video Wall │ Alerts │ Chat │ KPIs │ Occupancy │ Graph │
│  Trackers │ Investigate │ Floor Plan │ Zones │ Timeline │ Search        │
│  Webhooks │ Fleet │ Journey                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Edge Runtime | Python 3.12, OpenCV, PyAudio |
| Detection | YOLO-World v2 / YOLO11 (open-vocabulary), ViTPose++ (pose), Qwen2.5-VL-3B (edge VLM) |
| Face/Plate | InsightFace SCRFD (face detection), YOLO plate detector |
| Audio | BEATs / AST (audio classification) |
| Cloud API | Python / FastAPI (async, 98+ endpoints, rate-limited, structured JSON logging) |
| Event Processor | Go 1.22 (batch BQ inserts, exponential retry, structured slog logging) |
| Knowledge Graph | Neo4j 5 (bi-temporal, Graphiti-style edges) |
| Vector Store | Qdrant (SigLIP + GTE-Qwen2 embeddings) |
| Relational Store | PostgreSQL + pgvector (config, trackers, webhooks, audit) |
| Depth Estimation | Depth Anything V2 (spatial memory) |
| Re-ID | CLIP-ReID (cross-camera tracking) |
| Vision LLM | Gemini 2.5 Pro (causal analysis, synopsis, grounding) |
| Embeddings | GTE-Qwen2 (semantic dedup, entity resolution) |
| Dashboard | Next.js 14, TypeScript, React 18, Tailwind CSS, TanStack Query, HLS.js, Zustand, Sonner |
| Observability | OpenTelemetry (traces + metrics) |
| Infrastructure | GCP (Pub/Sub, GCS, BigQuery, GKE Autopilot), Terraform, Docker |
| CI/CD | GitHub Actions (Python tests + lint, Dashboard build, Go vet, Docker builds) |
| Testing | pytest (64 unit tests across 7 test files) |

---

## Quick Start

```bash
# 1 — Edge (per-camera device)
cd edge && pip install -r requirements.txt && python main.py

# 2 — Cloud services
cd cloud && pip install -r requirements.txt && uvicorn api:app --reload --port 8000

# 3 — Dashboard
cd dashboard && npm install && npm run dev
```

> The cloud layer lazy-initializes all 56 service instances on first request — no external dependencies required for local development beyond Neo4j and Qdrant.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Detailed system architecture and data flow diagrams |
| [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md) | End-to-end event lifecycle from RTSP frame to dashboard alert |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Step-by-step setup guide with prerequisites and configuration |

---

## API Reference

### Core Pipeline

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/events` | Ingest vision event from edge (full pipeline) |
| `POST` | `/api/query` | Natural language video query |
| `POST` | `/api/converse` | Live conversational camera agent |
| `WS` | `/ws/alerts` | Real-time WebSocket alert stream |
| `GET` | `/health` | Health check |

### KPIs & Analytics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/kpi/{camera_id}` | Daily KPI metrics |
| `GET` | `/api/kpi/{camera_id}/summary` | KPI summary with LLM narrative |
| `GET` | `/api/analytics/dwell/{zone_id}` | Zone dwell-time distribution |
| `GET` | `/api/heatmap/{camera_id}` | Spatial activity heatmap |
| `GET` | `/api/retail/metrics` | Retail footfall & conversion metrics |
| `POST` | `/api/retail/configure` | Configure retail zones |
| `GET` | `/api/retail/funnel` | Retail conversion funnel |
| `GET` | `/api/retail/attention` | Shelf attention heatmap |

### Knowledge Graph & Spatial

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graph/{camera_id}/recent` | Recent KG events |
| `GET` | `/api/spatial/{camera_id}` | Spatial state (entities + zones) |
| `GET` | `/api/scene-graph/{camera_id}` | Dynamic scene graph edges + transitions |
| `GET` | `/api/story/{entity_id}` | Entity story timeline |
| `GET` | `/api/profile/{entity_id}` | Entity profile |

### Reasoning & Anomaly

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rules` | Add neuro-symbolic reasoning rule |
| `POST` | `/api/alerts/{event_id}/ack` | Acknowledge alert |
| `GET` | `/api/normality/{camera_id}` | Contextual normality expected values |
| `POST` | `/api/explain/counterfactual/{event_id}` | Counterfactual alert explanation |
| `GET` | `/api/anomaly/collective/{camera_id}` | Collective anomaly patterns |
| `GET` | `/api/crowd/{camera_id}` | Crowd dynamics (forces, groups, pressure) |
| `GET` | `/api/grounding/{camera_id}` | Visual grounding bbox explanations |
| `GET` | `/api/behavioral/stress` | Behavioral stress scores |
| `GET` | `/api/adversarial/health` | Adversarial robustness health |
| `GET` | `/api/adversarial/{camera_id}` | Per-camera robustness score |

### Agents & Investigation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/investigate/{entity_id}` | Autonomous investigation |
| `GET` | `/api/synopsis/{camera_id}` | Forensic video synopsis |
| `POST` | `/api/simulate` | What-if scenario simulation |
| `GET` | `/api/system/health` | System self-evaluation + health report |
| `GET` | `/api/proactive/priorities` | Get proactive watch priorities |
| `POST` | `/api/proactive/priorities` | Set proactive watch priorities |
| `POST` | `/api/proactive/watch` | Add entity/zone to proactive watch |

### Search & Forensics

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/search` | Privacy-first forensic search |
| `POST` | `/api/search/unlock` | Unlock search result (audit-logged) |
| `GET` | `/api/search/audit` | Search audit log |
| `POST` | `/api/evidence/build` | Build court-ready evidence package |
| `GET` | `/api/evidence/{package_id}/html` | Render evidence package as HTML |

### Entity Journey

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/journey/active` | List active entity journeys |
| `GET` | `/api/journey/{entity_id}` | Full entity journey with zone transitions |
| `GET` | `/api/journey/{entity_id}/path` | Animated path coordinates |

### Cameras & Fleet

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/cameras` | List cameras |
| `POST` | `/api/cameras` | Add camera |
| `PATCH` | `/api/cameras/{id}` | Update camera |
| `DELETE` | `/api/cameras/{id}` | Delete camera |
| `POST` | `/api/fleet/heartbeat` | Edge device heartbeat |
| `GET` | `/api/fleet/status` | Fleet health dashboard |

### Custom Trackers

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/trackers` | Create custom tracker (NL) |
| `GET` | `/api/trackers` | List custom trackers |
| `PATCH` | `/api/trackers/{id}` | Pause/resume tracker |
| `DELETE` | `/api/trackers/{id}` | Delete tracker |

### Camera Programming

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/camera-programs` | Create NL camera program |
| `GET` | `/api/camera-programs` | List camera programs |
| `DELETE` | `/api/camera-programs/{id}` | Delete camera program |

### Occupancy & Compliance

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/occupancy/{camera_id}` | Camera occupancy snapshot |
| `GET` | `/api/occupancy` | Site-wide occupancy summary |
| `GET` | `/api/compliance/{camera_id}` | Safety compliance check |
| `POST` | `/api/compliance/sop` | Add SOP rule |

### License Plate Recognition

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/lpr/log/{camera_id}` | LPR recognition log |
| `POST` | `/api/lpr/watchlist` | Add plate to watchlist |
| `GET` | `/api/lpr/watchlist` | Get watchlist |
| `DELETE` | `/api/lpr/watchlist/{plate}` | Remove from watchlist |

### Zones

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/zones` | Add spatial zone |
| `DELETE` | `/api/zones/{zone_id}` | Delete zone |

### Webhooks

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/webhooks` | List webhooks |
| `POST` | `/api/webhooks` | Register webhook |
| `PATCH` | `/api/webhooks/{id}` | Update webhook |
| `DELETE` | `/api/webhooks/{id}` | Delete webhook |
| `GET` | `/api/webhooks/{id}/logs` | Webhook delivery logs |

### Reports & Digests

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/reports/schedules` | List report schedules |
| `POST` | `/api/reports/schedules` | Create report schedule |
| `DELETE` | `/api/reports/schedules/{id}` | Delete schedule |
| `POST` | `/api/reports/generate/{id}` | Generate report manually |
| `GET` | `/api/digest` | Alert digest with LLM summary |
| `GET` | `/api/shift/briefing` | Shift handover briefing |

### Staffing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/staffing/predict` | Demand forecast |
| `GET` | `/api/staffing/recommend` | Guard count recommendation |
| `GET` | `/api/staffing/plan` | Full shift staffing plan |

### Auth

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/auth/keys` | Create API key |
| `GET` | `/api/auth/keys` | List API keys |
| `DELETE` | `/api/auth/keys/{hash}` | Revoke API key |

### GDPR & Consent

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/consent/{entity_id}` | Get entity consent record |
| `POST` | `/api/consent` | Record consent |
| `GET` | `/api/consent/dpia-report` | Generate DPIA report |
| `POST` | `/api/retention/enforce` | Enforce data retention policy |

### Event Bus

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/events/subscribe` | Subscribe to event types |
| `GET` | `/api/events/subscriptions` | List subscriptions |
| `DELETE` | `/api/events/subscriptions/{id}` | Remove subscription |
| `GET` | `/api/events/log` | Event delivery log |

### Cross-Site Federation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/federation/peers` | Register federation peer |
| `GET` | `/api/federation/peers` | List federation peers |
| `POST` | `/api/federation/share` | Share entity across sites |
| `GET` | `/api/federation/alerts` | Federated alert feed |
| `GET` | `/api/federation/check/{entity_id}` | Check entity across federation |
| `GET` | `/api/federation/status` | Federation health status |

### Floor Plan

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/floorplan` | Digital twin floor plan |

---

## Project Structure

```
cctv_llm_project_twp/
├── edge/                                # Edge AI pipeline (per-camera)
│   ├── main.py                          # Pipeline orchestrator
│   ├── capture/
│   │   ├── frame_extractor.py           # Adaptive RTSP frame capture
│   │   └── mock_rtsp.py                 # Development mock stream
│   ├── detection/
│   │   ├── detector.py                  # YOLOv8/YOLOE open-vocab detection
│   │   └── edge_vlm.py                  # Edge VLM (InternVL-2B)
│   ├── audio/
│   │   └── processor.py                 # Audio capture + classification
│   ├── privacy/
│   │   └── engine.py                    # Face/plate blur engine
│   └── emitter/
│       └── event_emitter.py             # GCP Pub/Sub event emitter
│
├── cloud/                               # Cloud intelligence layer
│   ├── api.py                           # FastAPI app (98 endpoints)
│   ├── event_processor/
│   │   └── main.go                      # Go event fan-out processor
│   └── services/
│       ├── knowledge_graph/
│       │   ├── temporal_kg.py           # Neo4j bi-temporal KG
│       │   ├── hypergraph_rag.py        # HyperGraphRAG (Qdrant + Neo4j)
│       │   ├── entity_profiles.py       # Entity profiling
│       │   └── agent_memory.py          # 3-layer agent memory
│       ├── spatial/memory.py            # 4D spatial memory
│       ├── baseline/learner.py          # Hourly baseline + anomaly scoring
│       ├── reasoning/
│       │   ├── engine.py               # Neuro-symbolic rule engine
│       │   ├── intent_recognizer.py    # Behavioral intent recognition
│       │   ├── explainer.py            # Plain English alert explainer
│       │   └── counterfactual_explainer.py
│       ├── causal/understander.py       # VLM causal analysis + guardrails
│       ├── world_model/predictor.py     # Trajectory + crowd prediction
│       ├── anomaly/
│       │   ├── video_predictor.py       # Scene prediction divergence
│       │   ├── contextual_normality.py  # Context-aware normality baselines
│       │   └── collective_anomaly.py    # Multi-event collective anomalies
│       ├── crowd/social_force.py        # Social force model + groups
│       ├── scene_graph/dynamic_graph.py # Dynamic scene relationship graph
│       ├── simulation/what_if.py        # What-if scenario simulator
│       ├── meta/self_evaluator.py       # System self-evaluation + health
│       ├── nlp/alert_dedup_semantic.py  # Semantic alert deduplication
│       ├── action_engine/
│       │   ├── engine.py               # Reflexion action engine
│       │   └── deterrence.py           # Voice deterrence (4 levels)
│       ├── investigation/auto_investigator.py
│       ├── conversational/live_agent.py # Session-persistent chat agent
│       ├── query_engine/
│       │   ├── engine.py               # Multi-agent query engine
│       │   └── story_builder.py        # Entity story builder
│       ├── video_synopsis/synopsis.py   # Forensic video synopsis
│       ├── forensic_search/privacy_search.py
│       ├── digital_twin/floor_plan.py   # 2D digital twin floor plan
│       ├── custom_tracker/tracker.py    # NL custom tracker engine
│       ├── kpi_engine/engine.py         # KPI computation + summary
│       ├── occupancy/counter.py         # Real-time occupancy analytics
│       ├── compliance/safety_monitor.py # PPE, fall, SOP compliance
│       ├── lpr/recognizer.py            # License plate recognition
│       ├── webhooks/manager.py          # Webhook delivery (Slack/PD/Teams)
│       ├── reports/
│       │   ├── scheduler.py            # Scheduled report generation
│       │   └── alert_digest.py         # Daily alert digest + LLM summary
│       ├── persistence/store.py         # SQLite persistence layer
│       ├── auth/middleware.py           # API key + JWT multi-tenant auth
│       ├── cameras/manager.py           # Dynamic camera management
│       ├── continual_learning/
│       │   ├── learner.py              # Drift detection + FP tracking
│       │   └── synthetic_flywheel.py   # Synthetic data generation
│       ├── cross_camera/tracker.py      # Cross-camera entity tracking
│       ├── federated/manager.py         # Federated learning manager
│       ├── visual_grounding/grounder.py # GLaMM-inspired bbox explanations
│       ├── resource_optimizer/staffing.py # Predictive staffing
│       ├── behavioral/stress_detector.py  # Stress detection
│       ├── federation/intelligence.py   # Cross-site federation
│       ├── evidence/packager.py         # Evidence package builder
│       ├── shift/handover.py            # Shift handover briefing
│       ├── entity_journey/visualizer.py # Entity journey mapping
│       ├── retail/analytics.py          # Retail analytics
│       ├── integration/event_bus.py     # Internal event bus
│       ├── adversarial/robustness.py    # Adversarial robustness
│       ├── camera_programming/nl_programmer.py
│       ├── edge_manager/fleet.py        # Fleet management
│       ├── analytics/
│       │   ├── dwell_time.py           # Dwell analytics
│       │   └── heatmap_engine.py       # Heatmap generation
│       ├── streaming/proactive_agent.py # Proactive agent
│       ├── access/tailgate_detector.py  # Tailgate detection
│       ├── detection/environment_detector.py
│       ├── privacy/consent_manager.py   # GDPR consent management
│       └── audio_visual/fusion.py       # Audio-visual fusion
│
├── dashboard/                           # Next.js 14 frontend
│   ├── app/
│   │   ├── layout.tsx                   # Root layout with navigation
│   │   ├── page.tsx                     # Home / overview
│   │   ├── cameras/page.tsx             # Camera management
│   │   ├── videowall/page.tsx           # Multi-camera video wall
│   │   ├── alerts/page.tsx              # Real-time alert feed
│   │   ├── chat/page.tsx                # Conversational agent
│   │   ├── kpi/page.tsx                 # KPI dashboard
│   │   ├── occupancy/page.tsx           # Occupancy analytics
│   │   ├── graph/page.tsx               # Knowledge graph explorer
│   │   ├── trackers/page.tsx            # Custom tracker management
│   │   ├── investigate/page.tsx         # Autonomous investigation
│   │   ├── floorplan/page.tsx           # Digital twin floor plan
│   │   ├── zones/page.tsx               # Zone management
│   │   ├── timeline/page.tsx            # Incident timeline viewer
│   │   ├── search/page.tsx              # Forensic AI search
│   │   ├── webhooks/page.tsx            # Webhook management
│   │   ├── fleet/page.tsx               # Fleet health dashboard
│   │   └── journey/page.tsx             # Entity journey explorer
│   ├── lib/api.ts                       # API client
│   └── globals.css                      # Tailwind + CSS variables
│
├── shared/                              # Shared utilities
│   ├── config.py                        # Configuration constants
│   └── schemas.py                       # Shared data schemas
│
└── infra/
    └── main.tf                          # Terraform GCP infrastructure
```

---

## Dashboard Pages

| # | Page | Route | Description |
|---|------|-------|-------------|
| 1 | Home | `/` | System overview with camera grid and live stats |
| 2 | Cameras | `/cameras` | Add, edit, delete cameras with RTSP configuration |
| 3 | Video Wall | `/videowall` | 2×2 live camera grid with entity counts |
| 4 | Alerts | `/alerts` | Real-time WebSocket alert feed with causal explanations |
| 5 | Chat | `/chat` | Conversational agent — ask cameras questions in plain English |
| 6 | KPIs | `/kpi` | Daily metrics with LLM-generated narrative summaries |
| 7 | Occupancy | `/occupancy` | Real-time people counting, zone capacity, 24 h charts |
| 8 | Graph | `/graph` | Knowledge graph event explorer with relationship visualization |
| 9 | Trackers | `/trackers` | Create NL rules ("alert when someone enters parking after 10 pm") |
| 10 | Investigate | `/investigate` | Autonomous entity investigation with evidence timeline |
| 11 | Floor Plan | `/floorplan` | 2D digital twin with live entity positions and zone heatmaps |
| 12 | Zones | `/zones` | Create and manage spatial zones with capacity thresholds |
| 13 | Timeline | `/timeline` | SVG incident timeline with severity markers |
| 14 | Search | `/search` | Privacy-first forensic AI search with 4 access levels |
| 15 | Webhooks | `/webhooks` | Manage Slack / PagerDuty / Teams integrations |
| 16 | Fleet | `/fleet` | Edge device health monitoring and remote diagnostics |
| 17 | Journey | `/journey` | Cross-camera entity journey explorer with animated path replay |

---

## Novel Features (Industry Firsts)

| Feature | Module | Description |
|---------|--------|-------------|
| **Natural Language Rule Compiler** | `nl_rule_compiler/compiler.py` | Write detection rules in plain English → compiled to executable rule sets with confidence, cooldown, scheduling |
| **Ambient Intelligence Score** | `ambient_score/engine.py` | 9-signal Bayesian safety fusion: crowd, anomaly, audio, dwell, lighting, compliance, velocity, loitering, historical → single 0–1 safety score per zone |
| **Predictive Interception** | `predictive_interception/interceptor.py` | Trajectory extrapolation + restricted-zone geometry → pre-emptive alerts before entry occurs |
| **Gait DNA Biometrics** | `gait_dna/engine.py` | 32-dimensional movement fingerprint from pose sequences — re-identifies people without faces or clothing features |
| **Anomaly Contagion Network** | `contagion/network.py` | Zone-to-zone anomaly propagation graph with Bayesian probability flow — detects cascade patterns |
| **Déjà Vu Engine** | `deja_vu/engine.py` | Historical scene similarity matching via embedding cosine similarity — recognizes recurring suspicious patterns |

---

## Production Hardening

| Area | Implementation |
|------|---------------|
| **Error Handling** | All 43 silent `except: pass` blocks replaced with structured `logger.warning()` calls |
| **CORS** | Locked to `CORS_ORIGINS` env var (default: `http://localhost:3000`) — no more `allow_origins=["*"]` |
| **Rate Limiting** | Token-bucket middleware at `RATE_LIMIT_RPM` per IP (default: 300 req/min) |
| **Structured Logging** | JSON log formatter via `LOG_FORMAT=json` env var; Go processor uses `slog.NewJSONHandler` |
| **Startup Health Checks** | `_check_connectivity()` verifies Neo4j, Qdrant, PostgreSQL on boot |
| **Graceful Shutdown** | Lifespan manager closes DB pools, Neo4j driver, WebSocket connections |
| **Go Batch Inserts** | BigQuery batch inserter (50 rows / 2s flush), exponential backoff retry (3 attempts) |
| **Go Metrics** | Atomic counters for processed/errors/bq_rows/gcs_writes, logged every 30s |
| **Dashboard State** | All 8 data-fetching pages migrated from `useState+useEffect` to TanStack Query hooks |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_reasoning_engine.py -v
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_reasoning_engine.py` | 9 | Crowd triggers, audio triggers, cooldown, custom rules, loitering duration, explanation chains |
| `test_nl_rule_compiler.py` | 11 | Loitering, time constraints, running detection, crowd count, gunshot severity, list/delete/toggle/explain |
| `test_ambient_score.py` | 10 | Empty scene, crowd pressure, audio alerts, anomaly propagation, signal contributions, trend detection |
| `test_gait_dna.py` | 8 | Minimum observations, fingerprint computation, similarity matching, gallery management |
| `test_contagion_network.py` | 8 | Empty network, propagation recording, probability increase, graph structure, risk profiles |
| `test_deja_vu.py` | 10 | Encode/store, scene matching, camera filtering, incident confirmation, stats, top-k limits |
| `test_predictive_interception.py` | 8 | Point-in-polygon (square, triangle, edge), interceptor initialization |

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push to `main`/`develop` and PRs to `main`:

| Job | Steps |
|-----|-------|
| **python-tests** | Install deps → `pytest tests/ -v` → `ruff check` (Python 3.11 + 3.12 matrix) |
| **dashboard** | `npm ci` → `tsc --noEmit` → `npm run build` |
| **go-build** | `go build` → `go vet` |
| **docker** | Build edge + cloud images (main branch only, after all checks pass) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed CORS origins |
| `RATE_LIMIT_RPM` | `300` | Max requests per minute per IP |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logging |
| `LOG_LEVEL` | `info` | Go processor log level (`info` / `debug`) |

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `QDRANT_HOST` | `localhost` | Qdrant vector store host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `visionbrain-keyframes` | Qdrant collection name |
| `VISIONBRAIN_DB` | `./visionbrain.db` | SQLite database path |
| `AUTH_ENABLED` | `false` | Enable API key / JWT auth |
| `JWT_SECRET` | — | JWT signing secret (required when auth enabled) |
| `PUBSUB_TOPIC` | `visionbrain-events` | GCP Pub/Sub topic |
| `PUBSUB_SUBSCRIPTION` | `visionbrain-events-sub` | GCP Pub/Sub subscription |
| `GCS_KEYFRAME_BUCKET` | `visionbrain-keyframes` | GCS bucket for keyframes |
| `BIGQUERY_DATASET` | `visionbrain` | BigQuery dataset |
| `BIGQUERY_EVENTS_TABLE` | `events` | BigQuery events table |
| `DEFAULT_RTSP_URL` | `rtsp://localhost:8554/stream` | Default RTSP stream URL |
| `DEFAULT_DETECTION_CONFIDENCE` | `0.3` | Minimum detection confidence |
| `DEFAULT_DETECTION_CLASSES` | `["person"]` | Default detection classes |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Dashboard → API base URL |

---

## License

Proprietary. All rights reserved.
