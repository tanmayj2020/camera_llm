# VisionBrain — Architecture Diagrams

All diagrams are in [Mermaid](https://mermaid.js.org/) format (`.mmd` files) and can be rendered to high-resolution PNG/SVG.

## Diagrams

| # | Diagram | File | Description |
|---|---------|------|-------------|
| 1 | **System Architecture** | `01-system-architecture.mmd` | Full 4-layer overview: Edge → Transport → Cloud → Dashboard |
| 2 | **Edge AI Pipeline** | `02-edge-pipeline.mmd` | RTSP capture → YOLO-World v2 → ViTPose++ → Qwen2.5-VL → Privacy → Pub/Sub |
| 3 | **Cloud Event Pipeline** | `03-cloud-event-pipeline.mmd` | Go processor + 20-step FastAPI intelligence chain |
| 4 | **Reasoning & Anomaly** | `04-reasoning-anomaly-pipeline.mmd` | 4 parallel reasoning engines → dedup → severity → dispatch |
| 5 | **Knowledge Graph & Memory** | `05-knowledge-graph-memory.mmd` | Neo4j KG, HyperGraphRAG, Entity Profiles, 3-Layer Memory, Scene Graph |
| 6 | **Alert & Action Engine** | `06-alert-action-engine.mmd` | Reflexion engine, 4-level deterrence, multi-channel dispatch, investigation |
| 7 | **Novel Features** | `07-novel-features.mmd` | 6 industry-first capabilities (NL Rules, Ambient Score, Gait DNA, etc.) |
| 8 | **Dashboard Architecture** | `08-dashboard-architecture.mmd` | Next.js 14, TanStack Query, 17 pages, component hierarchy |
| 9 | **CI/CD Pipeline** | `09-cicd-pipeline.mmd` | GitHub Actions: Python tests, Dashboard build, Go vet, Docker |
| 10 | **Data Storage** | `10-data-storage-architecture.mmd` | 5 store types: Neo4j, Qdrant, PostgreSQL, GCS, BigQuery |
| 11 | **Security & Auth** | `11-security-auth-flow.mmd` | CORS, rate limiting, auth middleware, 4 access levels, privacy controls |
| 12 | **Deployment** | `12-deployment-architecture.mmd` | GKE Autopilot, managed data services, observability stack |

## Rendering to High-Resolution Images

### Using Mermaid CLI (recommended)

```bash
# Install
npm install -g @mermaid-js/mermaid-cli

# Render all diagrams to PNG (2x resolution)
for f in diagrams/*.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.png" -s 2 -b transparent
done

# Render all diagrams to SVG (infinite resolution)
for f in diagrams/*.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.svg" -b transparent
done
```

### Using Mermaid Live Editor

1. Go to [mermaid.live](https://mermaid.live)
2. Paste the contents of any `.mmd` file
3. Click "Download PNG" or "Download SVG"

### Using VS Code

Install the "Mermaid Preview" extension and open any `.mmd` file to see a live preview.
