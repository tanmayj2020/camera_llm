# Getting Started with VisionBrain

Everything you need to get the system running locally for development.

---

## Prerequisites

- **Python 3.12+**
- **Node.js 18+** (for dashboard)
- **Neo4j 5.x** (optional — system works without it)
- pip packages: see `cloud/requirements.txt`

---

## Quick Start

### 1. Cloud API (required)

```bash
cd cloud
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

API available at [http://localhost:8000](http://localhost:8000). Health check: `GET /health`

### 2. Dashboard (optional)

```bash
cd dashboard
npm install
npm run dev
```

Dashboard at [http://localhost:3000](http://localhost:3000)

### 3. Edge Pipeline (optional)

```bash
cd edge
pip install -r requirements.txt
python main.py
```

---

## Running Without External Services

VisionBrain is designed to work with **ZERO external dependencies**:

- **No Neo4j?** KG operations log warnings and return empty results
- **No Qdrant?** RAG falls back to keyword search
- **No Gemini API key?** All VLM calls return stub/template responses
- **No GCP?** Event emitter logs locally
- **SQLite** is the only storage — auto-created at `./visionbrain.db`
- **Auth disabled** by default (`AUTH_ENABLED=false`) — all endpoints accessible

---

## Testing the Pipeline

Send a test event:

```bash
curl -X POST http://localhost:8000/api/events \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "test-001",
    "timestamp": 1713500000,
    "camera_id": "cam-0",
    "event_type": "detection",
    "scene_activity": 0.5,
    "objects": [
      {"track_id": "p1", "class_name": "person", "bbox": [100,200,150,400], "confidence": 0.9}
    ],
    "audio_events": [],
    "keyframe_b64": null
  }'
```

Expected response:

```json
{"status": "ok", "anomaly_score": 0.0, "rules_triggered": 0, ...}
```

### Verify Sub-systems

After sending the test event, confirm the pipeline processed it:

- **Alerts:** `GET /api/alerts` — should return an empty list (no rules triggered for normal event)
- **Spatial:** `GET /api/spatial/entities` — should show entity `p1` at camera `cam-0`
- **Health:** `GET /health` — returns system status and module availability
- **WebSocket:** Connect to `ws://localhost:8000/ws` to receive real-time events

---

## Adding a New Service Module

1. Create package: `cloud/services/my_service/__init__.py` (empty)
2. Create implementation: `cloud/services/my_service/engine.py`
   - Use dataclasses for data structures
   - Accept dependencies via constructor
   - Use `_get_client()` pattern for optional VLM
3. Add lazy initializer to `cloud/api.py`:
   ```python
   def get_my_service():
       if "my_svc" not in _services:
           from services.my_service.engine import MyService
           _services["my_svc"] = MyService(spatial=get_spatial())
       return _services["my_svc"]
   ```
4. Add Pydantic model if needed
5. Add endpoint(s)
6. Add to `dashboard/lib/api.ts`
7. Wire into event pipeline if real-time processing needed

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `VISIONBRAIN_DB` | `./visionbrain.db` | SQLite path |
| `AUTH_ENABLED` | `false` | Enable API auth |
| `JWT_SECRET` | *(none)* | JWT signing key |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Dashboard API URL |

---

## Project Structure

```
cctv_llm_project_twp/
├── cloud/              # FastAPI backend + all service modules
│   ├── api.py          # Main application entry point
│   ├── services/       # Service modules (spatial, reasoning, etc.)
│   └── requirements.txt
├── dashboard/          # Next.js frontend
│   ├── app/            # App router pages
│   ├── components/     # React components
│   └── lib/api.ts      # API client
├── edge/               # Edge processing pipeline
│   ├── main.py         # Edge entry point
│   └── requirements.txt
└── docs/               # Documentation
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` from the correct directory |
| Neo4j connection refused | Expected if Neo4j isn't running — system continues without it |
| Dashboard can't reach API | Check `NEXT_PUBLIC_API_URL` matches your API port |
| No alerts appearing | Send events with higher `scene_activity` or add custom rules |
| WebSocket disconnects | Ensure you're connecting to `/ws` not `/api/ws` |
