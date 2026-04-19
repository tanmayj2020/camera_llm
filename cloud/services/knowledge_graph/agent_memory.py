"""Three-layer agent memory: episodic + semantic + procedural.

Auto-promotes: episodic → semantic after N similar episodes,
               episodic → procedural after N successful action sequences.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """Raw event memory — what happened, when, where."""
    memory_id: str
    timestamp: float
    camera_id: str
    event_type: str
    description: str
    entities: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SemanticMemory:
    """Distilled general knowledge — patterns, norms, facts."""
    memory_id: str
    category: str
    knowledge: str
    confidence: float
    source_episodes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class ProceduralMemory:
    """Learned action sequences — what to do in specific situations."""
    memory_id: str
    trigger_pattern: str
    action_sequence: list[dict]
    success_rate: float
    source_episodes: list[str] = field(default_factory=list)
    execution_count: int = 0
    created_at: float = field(default_factory=time.time)


class AgentMemory:
    """Three-layer memory system with automatic promotion.

    Episodic: raw events (bounded, FIFO eviction)
    Semantic: distilled knowledge (promoted from episodic after N similar episodes)
    Procedural: action sequences (promoted from episodic after N successful action sequences)
    """

    EPISODIC_CAPACITY = 10000
    SEMANTIC_PROMOTION_THRESHOLD = 5   # N similar episodes → semantic
    PROCEDURAL_PROMOTION_THRESHOLD = 3  # N successful action sequences → procedural
    SIMILARITY_THRESHOLD = 0.8

    def __init__(self):
        self._episodic: list[EpisodicMemory] = []
        self._semantic: dict[str, SemanticMemory] = {}
        self._procedural: dict[str, ProceduralMemory] = {}
        self._episode_clusters: dict[str, list[str]] = defaultdict(list)  # event_type → memory_ids
        self._action_sequences: dict[str, list[dict]] = defaultdict(list)  # event_type → [{actions, success}]
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    # --- Episodic layer ---

    def store_episode(self, memory_id: str, camera_id: str, event_type: str,
                      description: str, entities: list[str] = None,
                      data: dict = None) -> EpisodicMemory:
        ep = EpisodicMemory(
            memory_id=memory_id, timestamp=time.time(), camera_id=camera_id,
            event_type=event_type, description=description,
            entities=entities or [], data=data or {},
        )
        self._episodic.append(ep)
        self._episode_clusters[event_type].append(memory_id)

        # Evict oldest if over capacity
        if len(self._episodic) > self.EPISODIC_CAPACITY:
            self._episodic = self._episodic[-self.EPISODIC_CAPACITY:]

        # Check for promotion
        self._check_semantic_promotion(event_type)
        return ep

    def store_action_outcome(self, event_type: str, actions: list[dict], success: bool):
        """Record an action sequence outcome for procedural promotion."""
        self._action_sequences[event_type].append({"actions": actions, "success": success})
        self._check_procedural_promotion(event_type)

    def recall_episodic(self, event_type: str = None, camera_id: str = None,
                        limit: int = 20) -> list[EpisodicMemory]:
        results = self._episodic
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if camera_id:
            results = [e for e in results if e.camera_id == camera_id]
        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    # --- Semantic layer ---

    def recall_semantic(self, category: str = None) -> list[SemanticMemory]:
        results = list(self._semantic.values())
        if category:
            results = [s for s in results if s.category == category]
        for s in results:
            s.access_count += 1
        return sorted(results, key=lambda s: s.confidence, reverse=True)

    def get_semantic(self, memory_id: str) -> SemanticMemory | None:
        return self._semantic.get(memory_id)

    # --- Procedural layer ---

    def recall_procedural(self, trigger_pattern: str = None) -> list[ProceduralMemory]:
        results = list(self._procedural.values())
        if trigger_pattern:
            results = [p for p in results if trigger_pattern in p.trigger_pattern]
        return sorted(results, key=lambda p: p.success_rate, reverse=True)

    def get_best_procedure(self, event_type: str) -> ProceduralMemory | None:
        """Get the best action sequence for a given event type."""
        candidates = [p for p in self._procedural.values() if event_type in p.trigger_pattern]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.success_rate)

    # --- Auto-promotion ---

    def _check_semantic_promotion(self, event_type: str):
        cluster = self._episode_clusters.get(event_type, [])
        if len(cluster) < self.SEMANTIC_PROMOTION_THRESHOLD:
            return

        # Promote: distill episodes into semantic knowledge
        episodes = [e for e in self._episodic if e.memory_id in cluster[-self.SEMANTIC_PROMOTION_THRESHOLD:]]
        if not episodes:
            return

        sem_id = f"sem_{event_type}_{len(self._semantic)}"
        if any(s.category == event_type for s in self._semantic.values()):
            # Update existing semantic memory
            for s in self._semantic.values():
                if s.category == event_type:
                    s.source_episodes.extend([e.memory_id for e in episodes])
                    s.confidence = min(1.0, s.confidence + 0.05)
                    return

        knowledge = self._distill_knowledge(episodes, event_type)
        self._semantic[sem_id] = SemanticMemory(
            memory_id=sem_id, category=event_type, knowledge=knowledge,
            confidence=0.6, source_episodes=[e.memory_id for e in episodes],
        )
        logger.info("Promoted episodic → semantic: %s (%d episodes)", sem_id, len(episodes))

    def _check_procedural_promotion(self, event_type: str):
        sequences = self._action_sequences.get(event_type, [])
        successful = [s for s in sequences if s["success"]]
        if len(successful) < self.PROCEDURAL_PROMOTION_THRESHOLD:
            return

        proc_id = f"proc_{event_type}_{len(self._procedural)}"
        if any(event_type in p.trigger_pattern for p in self._procedural.values()):
            # Update existing
            for p in self._procedural.values():
                if event_type in p.trigger_pattern:
                    p.execution_count += 1
                    p.success_rate = len(successful) / len(sequences)
                    return

        # Use most common successful action sequence
        best_actions = successful[-1]["actions"]
        self._procedural[proc_id] = ProceduralMemory(
            memory_id=proc_id, trigger_pattern=event_type,
            action_sequence=best_actions,
            success_rate=len(successful) / len(sequences),
            source_episodes=[],
        )
        logger.info("Promoted episodic → procedural: %s (success rate %.0f%%)",
                    proc_id, len(successful) / len(sequences) * 100)

    def _distill_knowledge(self, episodes: list[EpisodicMemory], event_type: str) -> str:
        """Distill episodes into a semantic knowledge statement."""
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    descriptions = [e.description for e in episodes[:10]]
                    prompt = (f"Summarize these {len(descriptions)} similar '{event_type}' events "
                              f"into a general knowledge statement (1-2 sentences): {descriptions}")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception:
                    pass

        # Fallback: simple template
        cameras = set(e.camera_id for e in episodes)
        return (f"'{event_type}' events occur frequently "
                f"(observed {len(episodes)} times across cameras: {', '.join(cameras)})")

    # --- Summary ---

    def get_stats(self) -> dict:
        return {
            "episodic_count": len(self._episodic),
            "semantic_count": len(self._semantic),
            "procedural_count": len(self._procedural),
            "cluster_types": list(self._episode_clusters.keys()),
        }
