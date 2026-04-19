"""Task 11: HyperGraphRAG — video retrieval + multi-hop reasoning."""

import logging
import uuid
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedSegment:
    event_id: str
    timestamp: float
    camera_id: str
    score: float
    keyframe_gcs: str
    event_summary: str
    entities: list[str]


class HyperGraphRAG:
    """Combines vector retrieval (Qdrant) with graph traversal (Neo4j) for multi-hop reasoning.

    Hypergraph: events are hyperedges connecting multiple entities (n-ary relations).
    Dual index: Qdrant for semantic similarity, Neo4j for structural graph queries.
    """

    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 collection: str = "visionbrain-keyframes", neo4j_driver=None):
        self._qdrant = None
        self._collection = collection
        self._neo4j = neo4j_driver
        self._embedding_model = None

        try:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            self._ensure_collection()
        except Exception as e:
            logger.warning("Qdrant init failed: %s", e)

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self._collection not in collections:
            self._qdrant.create_collection(
                self._collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
                self._embedding_model = (model, preprocess, open_clip.get_tokenizer("ViT-B-32"))
            except Exception as e:
                logger.warning("CLIP load failed: %s", e)
        return self._embedding_model

    def embed_image(self, image_b64: str) -> np.ndarray | None:
        """Generate embedding for a keyframe image."""
        model_tuple = self._get_embedding_model()
        if model_tuple is None:
            return None

        import base64
        import io
        import torch
        from PIL import Image

        model, preprocess, _ = model_tuple
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def embed_text(self, text: str) -> np.ndarray | None:
        model_tuple = self._get_embedding_model()
        if model_tuple is None:
            return None

        import torch
        model, _, tokenizer = model_tuple
        tokens = tokenizer([text])
        with torch.no_grad():
            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def index_event(self, event_id: str, camera_id: str, timestamp: float,
                    keyframe_b64: str, event_summary: str, entities: list[str],
                    keyframe_gcs: str = ""):
        """Index a video event with both vector and graph representations."""
        # Vector index
        embedding = self.embed_image(keyframe_b64)
        if embedding is not None and self._qdrant:
            from qdrant_client.models import PointStruct
            self._qdrant.upsert(self._collection, [PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, event_id)),
                vector=embedding.tolist(),
                payload={
                    "event_id": event_id, "camera_id": camera_id,
                    "timestamp": timestamp, "event_summary": event_summary,
                    "entities": entities, "keyframe_gcs": keyframe_gcs,
                },
            )])

        # Graph index: create hyperedge connecting all entities
        if self._neo4j:
            with self._neo4j.session() as s:
                s.run("""
                    CREATE (he:HyperEdge {event_id: $eid, timestamp: datetime($ts),
                            summary: $summary, keyframe_gcs: $gcs})
                """, eid=event_id, ts=timestamp, summary=event_summary, gcs=keyframe_gcs)
                for entity_id in entities:
                    s.run("""
                        MATCH (he:HyperEdge {event_id: $eid})
                        MERGE (e {track_id: $tid})
                        CREATE (e)-[:PART_OF]->(he)
                    """, eid=event_id, tid=entity_id)

    def retrieve(self, query: str, top_k: int = 10, time_filter: dict | None = None,
                 entity_filter: list[str] | None = None) -> list[RetrievedSegment]:
        """Hybrid retrieval: vector similarity + graph filtering."""
        results = []

        # Vector search
        query_emb = self.embed_text(query)
        if query_emb is not None and self._qdrant:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

            filters = []
            if time_filter:
                if "start" in time_filter:
                    filters.append(FieldCondition(key="timestamp", range=Range(gte=time_filter["start"])))
                if "end" in time_filter:
                    filters.append(FieldCondition(key="timestamp", range=Range(lte=time_filter["end"])))
            if entity_filter:
                filters.append(FieldCondition(key="entities", match=MatchAny(any=entity_filter)))

            search_filter = Filter(must=filters) if filters else None
            hits = self._qdrant.search(
                self._collection, query_vector=query_emb.tolist(),
                limit=top_k, query_filter=search_filter,
            )
            for hit in hits:
                p = hit.payload
                results.append(RetrievedSegment(
                    event_id=p["event_id"], timestamp=p["timestamp"],
                    camera_id=p["camera_id"], score=hit.score,
                    keyframe_gcs=p.get("keyframe_gcs", ""),
                    event_summary=p.get("event_summary", ""),
                    entities=p.get("entities", []),
                ))

        # Graph-based multi-hop (if entity filter provided)
        if entity_filter and self._neo4j:
            graph_results = self._graph_multi_hop(entity_filter, top_k)
            # Merge with vector results, dedup by event_id
            seen = {r.event_id for r in results}
            for gr in graph_results:
                if gr.event_id not in seen:
                    results.append(gr)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _graph_multi_hop(self, entity_ids: list[str], limit: int) -> list[RetrievedSegment]:
        """Multi-hop graph traversal: find events connecting specified entities."""
        if not self._neo4j:
            return []
        with self._neo4j.session() as s:
            result = s.run("""
                UNWIND $eids AS eid
                MATCH (e {track_id: eid})-[:PART_OF]->(he:HyperEdge)
                WITH he, count(DISTINCT eid) AS matched
                WHERE matched >= $min_match
                RETURN he.event_id AS event_id, he.timestamp AS timestamp,
                       he.summary AS summary, he.keyframe_gcs AS gcs, matched
                ORDER BY matched DESC, he.timestamp DESC
                LIMIT $lim
            """, eids=entity_ids, min_match=max(1, len(entity_ids) // 2), lim=limit)
            return [
                RetrievedSegment(
                    event_id=r["event_id"], timestamp=0, camera_id="",
                    score=r["matched"] / len(entity_ids),
                    keyframe_gcs=r.get("gcs", ""), event_summary=r.get("summary", ""),
                    entities=entity_ids,
                )
                for r in result
            ]
