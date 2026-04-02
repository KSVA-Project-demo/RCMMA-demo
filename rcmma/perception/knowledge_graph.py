"""Dynamic knowledge graph builder.

Uses networkx to maintain a graph of detection nodes and simple relations.

Nodes have attributes: label, bbox, confidence, timestamp
Edges represent relations like 'co_occurs' and 'near'.
"""
from typing import Dict, Tuple, List
import networkx as nx
import math


class DynamicKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_detection(self, det: Dict):
        """Add a detection node. det must include 'id', 'label', 'bbox', 'confidence', 'timestamp'."""
        node_id = det["id"]
        attrs = {
            "label": det.get("label"),
            "bbox": det.get("bbox"),
            "confidence": det.get("confidence"),
            "timestamp": det.get("timestamp"),
        }
        self.graph.add_node(node_id, **attrs)

        # add co-occurrence edges to other nodes with close timestamps (same frame)
        for other in list(self.graph.nodes()):
            if other == node_id:
                continue
            other_ts = self.graph.nodes[other].get("timestamp")
            if other_ts is None:
                continue
            if abs(other_ts - attrs["timestamp"]) < 0.5:  # same frame window
                self.graph.add_edge(node_id, other, relation="co_occurs")

        # add 'near' relations based on bbox proximity
        for other in list(self.graph.nodes()):
            if other == node_id:
                continue
            other_bbox = self.graph.nodes[other].get("bbox")
            if other_bbox is None or attrs["bbox"] is None:
                continue
            if self._bboxes_near(attrs["bbox"], other_bbox):
                # prioritize adding 'near' relation (could be multiple edges; store as list)
                self.graph.add_edge(node_id, other, relation="near")

    def _bboxes_near(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], thresh: float = 0.2) -> bool:
        # compute center distance relative to diagonal of frame-size-approx
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        dx = acx - bcx
        dy = acy - bcy
        dist = math.hypot(dx, dy)
        # approximate frame diagonal using max bbox size to avoid needing frame dims
        diag = math.hypot(max(aw, bw), max(ah, bh))
        if diag == 0:
            return False
        return dist / diag < (1.0 + thresh)

    def nodes(self) -> List:
        return list(self.graph.nodes(data=True))

    def edges(self) -> List:
        return list(self.graph.edges(data=True))

    def to_dict(self) -> Dict:
        return nx.node_link_data(self.graph)
