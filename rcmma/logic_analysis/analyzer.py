"""Logic analyzer for extracting chains and mapping bboxes to graph nodes."""
from typing import List, Dict, Tuple, Any
import networkx as nx
import math


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    ix1 = max(a_x1, b_x1)
    iy1 = max(a_y1, b_y1)
    ix2 = min(a_x2, b_x2)
    iy2 = min(a_y2, b_y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def map_bbox_to_nodes(g: nx.Graph, bbox: Tuple[int, int, int, int], iou_threshold: float = 0.1) -> List[Tuple[str, float]]:
    """Map a bbox to graph nodes by IoU.

    Returns list of (node_id, iou) sorted by descending IoU for nodes with IoU >= threshold.
    """
    matches: List[Tuple[str, float]] = []
    for node, data in g.nodes(data=True):
        node_bbox = data.get("bbox")
        if node_bbox is None:
            continue
        score = _iou(bbox, node_bbox)
        if score >= iou_threshold:
            matches.append((node, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def find_logical_chains(g: nx.Graph, min_length: int = 2, max_length: int = 6, time_gap: float = 2.0, max_chains: int = 100) -> List[Dict[str, Any]]:
    """Find candidate logical chains (paths) in the graph.

    A chain is a simple path where timestamps are non-decreasing and consecutive node timestamp gaps
    are <= time_gap. Returns a list of dicts: {"nodes": [ids], "score": float, "times": [ts]}
    Chains are scored by average confidence multiplied by a temporal closeness factor.
    """
    chains = []

    # iterate components and examine simple paths up to max_length
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        nodes = list(sub.nodes())
        # consider all simple paths between pairs (bounded length)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    for path in nx.all_simple_paths(sub, source=nodes[i], target=nodes[j], cutoff=max_length):
                        if len(path) < min_length:
                            continue
                        # collect timestamps and confidences
                        times = [sub.nodes[n].get("timestamp") for n in path]
                        if any(t is None for t in times):
                            continue
                        # check non-decreasing and time gaps
                        ok = True
                        for a, b in zip(times, times[1:]):
                            if b < a or (b - a) > time_gap:
                                ok = False
                                break
                        if not ok:
                            continue
                        confs = [sub.nodes[n].get("confidence", 0.0) or 0.0 for n in path]
                        avg_conf = sum(confs) / len(confs) if confs else 0.0
                        # temporal closeness factor: shorter overall span -> higher score
                        span = times[-1] - times[0] if times else 0.0
                        temporal_factor = 1.0 / (1.0 + span)
                        score = avg_conf * temporal_factor
                        chains.append({"nodes": path, "score": float(score), "times": times})
                except nx.NetworkXNoPath:
                    continue

    # sort chains by score desc and limit
    chains.sort(key=lambda x: x["score"], reverse=True)
    return chains[:max_chains]
