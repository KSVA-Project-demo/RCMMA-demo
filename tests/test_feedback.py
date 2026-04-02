import time
from rcmma.perception.knowledge_graph import DynamicKnowledgeGraph
from rcmma.logic_analysis.analyzer import find_logical_chains
from rcmma.feedback.controller import FeedbackController


def test_feedback_triggers():
    g = DynamicKnowledgeGraph()
    ts = time.time()
    det1 = {"id": "n1", "label": "object", "bbox": (10, 10, 30, 30), "confidence": 0.9, "timestamp": ts}
    det2 = {"id": "n2", "label": "object", "bbox": (50, 10, 30, 30), "confidence": 0.85, "timestamp": ts + 0.2}
    g.add_detection(det1)
    g.add_detection(det2)

    chains = find_logical_chains(g.graph)
    fb = FeedbackController(threshold=0.01)
    action = fb.evaluate_and_act(chains, g.graph, frame=1)
    assert action is not None
    assert action["action"] == "alert_chain_detected"
