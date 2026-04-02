import time
from rcmma.perception.knowledge_graph import DynamicKnowledgeGraph
from rcmma.logic_analysis.analyzer import find_logical_chains, map_bbox_to_nodes


def test_map_bbox_to_nodes_and_chains():
    g = DynamicKnowledgeGraph()
    ts = time.time()
    # two overlapping detections in same frame
    det1 = {"id": "n1", "label": "object", "bbox": (10, 10, 30, 30), "confidence": 0.9, "timestamp": ts}
    det2 = {"id": "n2", "label": "object", "bbox": (20, 20, 30, 30), "confidence": 0.8, "timestamp": ts}
    # later detection connected to n2
    det3 = {"id": "n3", "label": "object", "bbox": (200, 200, 30, 30), "confidence": 0.7, "timestamp": ts + 0.5}
    g.add_detection(det1)
    g.add_detection(det2)
    g.add_detection(det3)

    # mapping: bbox overlapping n2 should match n2 and maybe n1
    matches = map_bbox_to_nodes(g.graph, (22, 22, 10, 10), iou_threshold=0.01)
    assert any(m[0] == "n2" for m in matches)

    chains = find_logical_chains(g.graph, min_length=2, time_gap=1.0)
    # expect at least one chain (n1-n2 or n2-n3 depending on edges)
    assert len(chains) >= 1
