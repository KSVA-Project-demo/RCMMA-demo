"""Simple runner for the perception pipeline.

Runs a VideoSource -> MockDetector -> DynamicKnowledgeGraph loop and prints
summary information periodically. Designed as a starting point for integration.
"""
import argparse
import time
import json
import cv2

from .capture import VideoSource
from .detector import MockDetector
from .knowledge_graph import DynamicKnowledgeGraph
from .visualize import GraphVisualizer
from rcmma.logic_analysis.analyzer import find_logical_chains
import os
from rcmma.feedback.controller import FeedbackController


def run(source=0, max_frames: int = 500, save_graph: str = None, analyze: bool = False, chains_interval: int = 30, save_chains: str = None, feedback: bool = False, feedback_threshold: float = 0.2, feedback_log: str = None):
    src = VideoSource(source)
    if not src.open():
        print(f"Unable to open source: {source}")
        return

    detector = MockDetector()
    graph = DynamicKnowledgeGraph()
    visualizer = None
    visualize_enabled = False
    fb = None
    if feedback:
        fb = FeedbackController(threshold=feedback_threshold, feedback_log=feedback_log)
    frame_count = 0

    try:
        for ts, frame in src.frames():
            detections = detector.detect(frame)
            for d in detections:
                d["timestamp"] = ts
                graph.add_detection(d)

            # logic analysis: periodically extract top chains
            if analyze and frame_count % chains_interval == 0:
                chains = find_logical_chains(graph.graph)
                if save_chains:
                    # append a JSON line with snapshot info
                    os.makedirs(os.path.dirname(save_chains), exist_ok=True)
                    with open(save_chains, "a", encoding="utf-8") as cf:
                        json.dump({"frame": frame_count, "time": ts, "chains": chains}, cf)
                        cf.write("\n")
                # print top chains summary
                if chains:
                    top = chains[0]
                    print(f"Top chain (frame {frame_count}): nodes={top['nodes']} score={top['score']:.3f}")
                    # feedback
                    if fb is not None:
                        action = fb.evaluate_and_act(chains, graph.graph, frame=frame_count)
                        if action:
                            # optionally annotate frame with brief overlay (best-effort)
                            try:
                                cv2.putText(frame, f"ACTION: {action['action']} {action['score']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            except Exception:
                                pass

            frame_count += 1

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames, nodes={len(graph.graph.nodes())}, edges={len(graph.graph.edges())}")

            # update visualizer if enabled
            if visualize_enabled and visualizer is not None:
                try:
                    visualizer.update(graph.graph)
                except Exception as e:
                    print(f"Visualizer error: {e}")

            # show simple overlay for debugging
            for det in detections:
                x, y, w, h = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, det["id"], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("RCMMA Perception (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if 0 < max_frames <= frame_count:
                break

    finally:
        src.release()
        cv2.destroyAllWindows()

    if save_graph:
        with open(save_graph, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"Saved graph to {save_graph}")


def run_with_visualization(source=0, max_frames: int = 500, save_graph: str = None, analyze: bool = False, chains_interval: int = 10, save_chains: str = None):
    """Run the pipeline with a live graph visualizer."""
    vis = GraphVisualizer()
    # delegate to run but enable visualization by re-using much of the logic
    src = VideoSource(source)
    if not src.open():
        print(f"Unable to open source: {source}")
        return

    detector = MockDetector()
    graph = DynamicKnowledgeGraph()
    fb = None
    if analyze and save_chains is not None:
        # create a feedback controller if analysis is enabled and feedback will be used via CLI later
        fb = None
    frame_count = 0

    try:
        for ts, frame in src.frames():
            detections = detector.detect(frame)
            for d in detections:
                d["timestamp"] = ts
                graph.add_detection(d)

            frame_count += 1

            # update visual graph every 10 frames
            if frame_count % 10 == 0:
                vis.update(graph.graph)

            top_chains = []
            # logic analysis and overlay chains (if enabled)
            if analyze and frame_count % chains_interval == 0:
                chains = find_logical_chains(graph.graph)
                top_chains = chains[:3] if chains else []
                if save_chains:
                    os.makedirs(os.path.dirname(save_chains), exist_ok=True)
                    with open(save_chains, "a", encoding="utf-8") as cf:
                        json.dump({"frame": frame_count, "time": ts, "chains": chains}, cf)
                        cf.write("\n")
                if top_chains:
                    top = top_chains[0]
                    nodes = top.get("nodes", [])
                # draw lines between consecutive nodes using bbox centers
                centers = []
                for n in nodes:
                    nb = graph.graph.nodes[n].get("bbox")
                    if nb:
                        x, y, w, h = nb
                        centers.append((int(x + w / 2), int(y + h / 2)))
                for a, b in zip(centers, centers[1:]):
                    cv2.line(frame, a, b, (0, 0, 255), 2)
                # annotate chain id/score
                if centers:
                    cx, cy = centers[0]
                    cv2.putText(frame, f"chain:{top.get('score',0):.2f}", (cx, max(10, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for det in detections:
                x, y, w, h = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, det["id"], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("RCMMA Perception (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if 0 < max_frames <= frame_count:
                break

    finally:
        src.release()
        cv2.destroyAllWindows()

    if save_graph:
        with open(save_graph, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"Saved graph to {save_graph}")


def main():
    parser = argparse.ArgumentParser(description="Run RCMMA perception pipeline (mock)")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam or path to file)")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to process")
    parser.add_argument("--save-graph", help="Optional path to save graph JSON")
    parser.add_argument("--visualize", action="store_true", help="Show live knowledge-graph visualization")
    parser.add_argument("--analyze", action="store_true", help="Enable logic analysis (find logical chains)")
    parser.add_argument("--chains-interval", type=int, default=10, help="Frames between logic-analysis runs when visualizing")
    parser.add_argument("--save-chains", help="Optional path to append chain snapshots as JSONL")
    args = parser.parse_args()

    # try convert numeric
    try:
        src = int(args.source)
    except Exception:
        src = args.source

    if args.visualize:
        run_with_visualization(src, args.max_frames, args.save_graph, analyze=args.analyze, chains_interval=args.chains_interval, save_chains=args.save_chains)
    else:
        run(src, args.max_frames, args.save_graph, analyze=args.analyze, chains_interval=args.chains_interval, save_chains=args.save_chains)


if __name__ == "__main__":
    main()
