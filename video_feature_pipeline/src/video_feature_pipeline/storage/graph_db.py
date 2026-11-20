from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)

try:
    from neo4j import GraphDatabase as Neo4jDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    Neo4jDriver = None


class GraphDatabase:
    def __init__(self):
        self.settings = get_settings()
        self.driver = None
        self._connect()

    def _connect(self) -> None:
        if not NEO4J_AVAILABLE:
            logger.warning("neo4j_not_installed", message="pip install neo4j to enable graph features")
            self.driver = None
            return

        try:
            uri = getattr(self.settings, "neo4j_uri", "bolt://localhost:7687")
            user = getattr(self.settings, "neo4j_user", "neo4j")
            password = getattr(self.settings, "neo4j_password", "password")

            self.driver = Neo4jDriver.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("neo4j_connected", uri=uri)
        except Exception as e:
            logger.warning("neo4j_connection_failed", error=str(e))
            self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self.driver = None

    def is_available(self) -> bool:
        return self.driver is not None

    def sync_identity(self, identity_id: str, video_id: str, cluster_id: int) -> None:
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run(
                "MERGE (i:Identity {id: $identity_id}) "
                "SET i.video_id = $video_id, i.cluster_id = $cluster_id",
                identity_id=identity_id, video_id=video_id, cluster_id=cluster_id
            )

    def sync_co_occurrence(self, identity_id_a: str, identity_id_b: str,
                          video_id: str, frame_idx: int, timestamp: float) -> None:
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Identity {id: $id_a}) "
                "MERGE (b:Identity {id: $id_b}) "
                "MERGE (a)-[r:CO_OCCURS_WITH {video_id: $video_id, "
                "frame_idx: $frame_idx}]->(b) "
                "SET r.timestamp = $timestamp, "
                "r.weight = coalesce(r.weight, 0) + 1",
                id_a=identity_id_a, id_b=identity_id_b,
                video_id=video_id, frame_idx=frame_idx, timestamp=timestamp
            )

    def get_partners(self, identity_id: str, min_weight: int = 1) -> list[dict[str, Any]]:
        if not self.driver:
            return []
        with self.driver.session() as session:
            result = session.run(
                "MATCH (a:Identity {id: $id})-[r:CO_OCCURS_WITH]-(b:Identity) "
                "WHERE r.weight >= $min_weight "
                "RETURN b.id as partner_id, r.weight as weight, "
                "collect(r.video_id) as videos, collect(r.frame_idx) as frames "
                "ORDER BY r.weight DESC",
                id=identity_id, min_weight=min_weight
            )
            return [dict(record) for record in result]

    def get_network(self, identity_id: str, depth: int = 2) -> dict[str, Any]:
        if not self.driver:
            return {"nodes": [], "edges": []}
        return self._get_network_simple(identity_id, depth)

    def _get_network_simple(self, identity_id: str, depth: int) -> dict[str, Any]:
        if not self.driver:
            return {"nodes": [], "edges": []}
        with self.driver.session() as session:
            nodes = {identity_id}
            edges = []
            current = {identity_id}

            for _ in range(depth):
                if not current:
                    break
                result = session.run(
                    "MATCH (a:Identity)-[r:CO_OCCURS_WITH]-(b:Identity) "
                    "WHERE a.id IN $current "
                    "RETURN a.id as source, b.id as target, r.weight as weight",
                    current=list(current)
                )
                next_level = set()
                for record in result:
                    edges.append(dict(record))
                    next_level.add(record["target"])
                nodes.update(next_level)
                current = next_level

            return {"nodes": list(nodes), "edges": edges}

    def get_stats(self) -> dict[str, int]:
        if not self.driver:
            return {"identities": 0, "relationships": 0}
        with self.driver.session() as session:
            identities = session.run("MATCH (i:Identity) RETURN count(i) as count").single()["count"]
            rels = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) as count").single()["count"]
            return {"identities": identities, "relationships": rels}

    def export_graphml(self, path: str) -> None:
        if not self.driver:
            return

        graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        graph = ET.SubElement(graphml, "graph", id="G", edgedefault="undirected")

        with self.driver.session() as session:
            for record in session.run("MATCH (i:Identity) RETURN i.id as id, i.cluster_id as cluster"):
                node = ET.SubElement(graph, "node", id=record["id"])
                ET.SubElement(node, "data", key="cluster_id").text = str(record["cluster"])

            for record in session.run(
                "MATCH (a)-[r:CO_OCCURS_WITH]->(b) "
                "RETURN a.id as source, b.id as target, r.weight as weight"
            ):
                edge = ET.SubElement(graph, "edge", source=record["source"], target=record["target"])
                ET.SubElement(edge, "data", key="weight").text = str(record["weight"])

        tree = ET.ElementTree(graphml)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        logger.info("graphml_exported", path=path)


_graph_db_instance: GraphDatabase | None = None


def get_graph_db() -> GraphDatabase:
    global _graph_db_instance
    if _graph_db_instance is None:
        _graph_db_instance = GraphDatabase()
    return _graph_db_instance