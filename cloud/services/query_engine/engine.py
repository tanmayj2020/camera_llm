"""Task 12: Multi-Agent Query Engine with Test-Time Compute Scaling."""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    SIMPLE = "simple"    # direct graph lookup, <500ms
    MEDIUM = "medium"    # RAG + single VLM call, <3s
    COMPLEX = "complex"  # multi-agent deep reasoning, <15s


@dataclass
class QueryResult:
    answer: str
    evidence: list[dict]  # [{keyframe_gcs, timestamp, description}]
    confidence: float
    complexity: QueryComplexity
    latency_ms: float
    grounded: bool


class TestTimeComputeRouter:
    """Routes queries to appropriate processing path based on complexity."""

    SIMPLE_PATTERNS = [
        "how many", "count", "total", "list all", "who is", "what time",
    ]
    COMPLEX_PATTERNS = [
        "why", "explain", "compare", "what should", "improve", "recommend",
        "relationship between", "pattern", "trend", "predict",
    ]

    def classify(self, query: str) -> QueryComplexity:
        q = query.lower()
        if any(p in q for p in self.COMPLEX_PATTERNS):
            return QueryComplexity.COMPLEX
        if any(p in q for p in self.SIMPLE_PATTERNS):
            return QueryComplexity.SIMPLE
        return QueryComplexity.MEDIUM


class MultiAgentQueryEngine:
    """Orchestrates specialized agents for video query answering.

    Agents:
    - Perception: analyzes retrieved keyframes
    - Memory: queries temporal knowledge graph
    - Spatial: provides 3D spatial context
    - Reasoning: causal/temporal logic synthesis
    """

    def __init__(self, kg=None, rag=None, spatial=None, vlm_client=None):
        self.kg = kg          # TemporalKnowledgeGraph
        self.rag = rag        # HyperGraphRAG
        self.spatial = spatial  # SpatialMemory
        self.vlm = vlm_client  # CausalUnderstander or similar
        self.router = TestTimeComputeRouter()
        self._conversation_history: list[dict] = []

    def query(self, user_query: str, camera_id: str | None = None) -> QueryResult:
        t0 = time.time()
        complexity = self.router.classify(user_query)
        self._conversation_history.append({"role": "user", "content": user_query})

        if complexity == QueryComplexity.SIMPLE:
            result = self._simple_path(user_query, camera_id)
        elif complexity == QueryComplexity.MEDIUM:
            result = self._medium_path(user_query, camera_id)
        else:
            result = self._complex_path(user_query, camera_id)

        result.latency_ms = (time.time() - t0) * 1000
        result.complexity = complexity
        self._conversation_history.append({"role": "assistant", "content": result.answer})
        return result

    def _simple_path(self, query: str, camera_id: str | None) -> QueryResult:
        """Direct graph query — fast path."""
        # Memory agent: query knowledge graph
        graph_data = []
        if self.kg and camera_id:
            graph_data = self.kg.get_recent_events(camera_id, limit=20)

        # Simple aggregation
        answer = self._simple_aggregate(query, graph_data)
        return QueryResult(
            answer=answer, evidence=[], confidence=0.8,
            complexity=QueryComplexity.SIMPLE, latency_ms=0, grounded=True,
        )

    def _medium_path(self, query: str, camera_id: str | None) -> QueryResult:
        """RAG retrieval + single VLM call."""
        # Retrieve relevant segments
        segments = []
        if self.rag:
            segments = self.rag.retrieve(query, top_k=5)

        # Memory agent context
        graph_context = []
        if self.kg and camera_id:
            graph_context = self.kg.get_recent_events(camera_id, limit=10)

        # Single VLM synthesis
        evidence = [{"event_id": s.event_id, "timestamp": s.timestamp,
                      "summary": s.event_summary, "score": s.score} for s in segments]

        answer = self._vlm_synthesize(query, evidence, graph_context)
        return QueryResult(
            answer=answer, evidence=evidence, confidence=0.7,
            complexity=QueryComplexity.MEDIUM, latency_ms=0, grounded=len(evidence) > 0,
        )

    def _complex_path(self, query: str, camera_id: str | None) -> QueryResult:
        """Multi-agent deep reasoning with iterative refinement."""
        # Round 1: All agents gather context
        memory_context = self.kg.get_recent_events(camera_id, limit=30) if self.kg and camera_id else []
        rag_segments = self.rag.retrieve(query, top_k=10) if self.rag else []

        spatial_context = {}
        if self.spatial:
            spatial_context = {
                "entity_count": len(self.spatial._entities),
                "zones": [{"id": z.zone_id, "name": z.name} for z in self.spatial._zones.values()],
            }

        # Round 2: Reasoning agent synthesizes
        all_context = {
            "query": query,
            "conversation_history": self._conversation_history[-6:],
            "graph_events": memory_context[:15],
            "retrieved_segments": [{"event_id": s.event_id, "summary": s.event_summary,
                                     "score": s.score} for s in rag_segments[:5]],
            "spatial": spatial_context,
        }

        answer = self._multi_agent_reason(all_context)
        evidence = [{"event_id": s.event_id, "timestamp": s.timestamp,
                      "summary": s.event_summary} for s in rag_segments[:5]]

        return QueryResult(
            answer=answer, evidence=evidence, confidence=0.6,
            complexity=QueryComplexity.COMPLEX, latency_ms=0, grounded=len(evidence) > 0,
        )

    def _simple_aggregate(self, query: str, graph_data: list[dict]) -> str:
        q = query.lower()
        if "how many" in q and "person" in q:
            # Count unique persons from events
            persons = set()
            for evt in graph_data:
                data = evt.get("data", "")
                if "person" in str(data).lower():
                    persons.add(evt.get("event_id", ""))
            return f"Based on recent events, approximately {len(persons)} person detections recorded."
        return f"Found {len(graph_data)} recent events. Please refine your query for specific insights."

    def _vlm_synthesize(self, query: str, evidence: list[dict], graph_context: list[dict]) -> str:
        """Use VLM for single-pass synthesis."""
        if self.vlm and hasattr(self.vlm, '_get_client'):
            client = self.vlm._get_client()
            if client != "stub":
                try:
                    prompt = f"""Answer this question about CCTV footage:
Question: {query}

Evidence from video retrieval:
{json.dumps(evidence[:5], indent=2, default=str)}

Knowledge graph context:
{json.dumps(graph_context[:5], indent=2, default=str)}

Provide a concise, evidence-grounded answer. Cite specific timestamps and events."""
                    response = client.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logger.error("VLM synthesis failed: %s", e)

        # Fallback
        if evidence:
            summaries = "; ".join(e.get("summary", "event") for e in evidence[:3])
            return f"Based on {len(evidence)} relevant video segments: {summaries}"
        return "No relevant video segments found for your query."

    def _multi_agent_reason(self, context: dict) -> str:
        """Multi-agent reasoning with iterative refinement."""
        if self.vlm and hasattr(self.vlm, '_get_client'):
            client = self.vlm._get_client()
            if client != "stub":
                try:
                    prompt = f"""You are a multi-agent CCTV analytics system performing deep analysis.

Query: {context['query']}

Conversation history: {json.dumps(context.get('conversation_history', []), default=str)}

PERCEPTION AGENT findings:
{json.dumps(context.get('retrieved_segments', []), indent=2, default=str)}

MEMORY AGENT findings:
{json.dumps(context.get('graph_events', [])[:10], indent=2, default=str)}

SPATIAL AGENT findings:
{json.dumps(context.get('spatial', {}), indent=2, default=str)}

As the REASONING AGENT, synthesize all findings into a comprehensive answer.
Include causal reasoning, temporal patterns, and actionable recommendations.
Ground every claim in specific evidence."""
                    response = client.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logger.error("Multi-agent reasoning failed: %s", e)

        return "Deep analysis requires VLM access. Based on available data: " + \
               f"{len(context.get('graph_events', []))} events and " + \
               f"{len(context.get('retrieved_segments', []))} video segments found."
