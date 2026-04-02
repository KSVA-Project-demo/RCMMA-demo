import time
from rcmma.perception.knowledge_graph import DynamicKnowledgeGraph


def test_add_detection_and_relations():
    g = DynamicKnowledgeGraph()
    ts = time.time()
    det1 = {"id": "a1", "label": "object", "bbox": (10, 10, 20, 20), "confidence": 0.9, "timestamp": ts}
    det2 = {"id": "a2", "label": "object", "bbox": (15, 15, 20, 20), "confidence": 0.8, "timestamp": ts}
    g.add_detection(det1)
    g.add_detection(det2)

    nodes = g.nodes()
    assert any(n[0] == "a1" for n in nodes)
    assert any(n[0] == "a2" for n in nodes)

    edges = g.edges()
    # Expect at least one relation (co_occurs or near)
    assert len(edges) >= 1
