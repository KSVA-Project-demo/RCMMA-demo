"""Minimal Flask frontend to display perception -> logic -> feedback in real-time.

Usage:
    python -m rcmma.frontend.server --source 0 --port 5000

The server exposes:
  /            - simple HTML page that auto-refreshes frames and graph images
  /frame.jpg   - latest video frame (JPEG)
  /graph.png   - latest graph visualization (PNG)
  /actions     - JSON list of recent feedback actions

This is intentionally minimal and designed for local demo only.
"""
import argparse
import threading
import time
from io import BytesIO
import json

from flask import Flask, Response, send_file, jsonify
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cv2

from rcmma.perception.capture import VideoSource
from rcmma.perception.detector import MockDetector
from rcmma.perception.knowledge_graph import DynamicKnowledgeGraph
from rcmma.logic_analysis.analyzer import find_logical_chains
from rcmma.feedback.controller import FeedbackController


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None  # bytes (JPEG)
        self.latest_graph = None  # bytes (PNG)
        self.actions = []  # list of action dicts
        self.chains = []


def render_graph_image(g: nx.Graph, width: int = 640, height: int = 480) -> bytes:
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Knowledge Graph")
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(no nodes)", horizontalalignment="center", verticalalignment="center")
    else:
        try:
            pos = nx.spring_layout(g, k=0.5, iterations=30)
        except Exception:
            pos = nx.circular_layout(g)
        labels = nx.get_node_attributes(g, "label")
        node_colors = ["#1f78b4" if labels.get(n, "") == "object" else "#33a02c" for n in g.nodes()]
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.6)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=200)
        nx.draw_networkx_labels(g, pos, {n: n for n in g.nodes()}, ax=ax, font_size=8)
    ax.set_axis_off()
    buf = BytesIO()
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def encode_frame_jpeg(frame) -> bytes:
    if frame is None:
        return None
    try:
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buf.tobytes()
    except Exception:
        return None


def pipeline_loop(state: SharedState, source, analyze: bool, chains_interval: int, feedback_enabled: bool, feedback_threshold: float, feedback_log: str, stop_event: threading.Event):
    vs = VideoSource(source)
    if not vs.open():
        print(f"Unable to open source: {source}")
        return
    detector = MockDetector()
    graph = DynamicKnowledgeGraph()
    fb = FeedbackController(threshold=feedback_threshold, feedback_log=feedback_log) if feedback_enabled else None

    frame_count = 0
    try:
        for ts, frame in vs.frames():
            if stop_event.is_set():
                break
            detections = detector.detect(frame)
            for d in detections:
                d['timestamp'] = ts
                graph.add_detection(d)

            # encode latest frame with overlays
            vis_frame = frame.copy()
            for det in detections:
                x, y, w, h = det['bbox']
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_frame, det['id'], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # analysis and feedback periodically
            if analyze and frame_count % chains_interval == 0:
                chains = find_logical_chains(graph.graph)
                action = None
                if fb is not None and chains:
                    action = fb.evaluate_and_act(chains, graph.graph, frame=frame_count)
                with state.lock:
                    state.chains = chains
                    if action:
                        state.actions.append(action)

            # update shared images
            frame_jpg = encode_frame_jpeg(vis_frame)
            try:
                graph_png = render_graph_image(graph.graph)
            except Exception:
                graph_png = None

            with state.lock:
                if frame_jpg is not None:
                    state.latest_frame = frame_jpg
                if graph_png is not None:
                    state.latest_graph = graph_png

            frame_count += 1
            # small sleep to avoid pegging CPU for very fast inputs
            time.sleep(0.005)

    finally:
        vs.release()


def create_app(state: SharedState):
    app = Flask(__name__)

    @app.route('/')
    def index():
        # simple page that refreshes images every second
        html = '''
        <!doctype html>
        <html>
        <head>
          <title>RCMMA Live</title>
          <style> img { max-width: 640px; display:block; } body{ font-family: Arial, sans-serif; margin: 20px; }</style>
        </head>
        <body>
          <h2>RCMMA Perception — Live</h2>
          <div>
            <h3>Video</h3>
            <img id="frame" src="/frame.jpg?ts=0" />
          </div>
          <div>
            <h3>Knowledge Graph</h3>
            <img id="graph" src="/graph.png?ts=0" />
          </div>
          <div>
            <h3>Actions</h3>
            <pre id="actions">[]</pre>
          </div>
          <script>
            async function refresh(){
              document.getElementById('frame').src = '/frame.jpg?ts=' + Date.now();
              document.getElementById('graph').src = '/graph.png?ts=' + Date.now();
              try{
                const r = await fetch('/actions');
                const j = await r.json();
                document.getElementById('actions').textContent = JSON.stringify(j, null, 2);
              }catch(e){ console.log(e); }
            }
            setInterval(refresh, 1000);
            refresh();
          </script>
        </body>
        </html>
        '''
        return html

    @app.route('/frame.jpg')
    def frame_jpg():
        with state.lock:
            data = state.latest_frame
        if data is None:
            return Response(status=204)
        return Response(data, mimetype='image/jpeg')

    @app.route('/graph.png')
    def graph_png():
        with state.lock:
            data = state.latest_graph
        if data is None:
            return Response(status=204)
        return Response(data, mimetype='image/png')

    @app.route('/actions')
    def actions():
        with state.lock:
            return jsonify(state.actions[-50:])

    return app


def main():
    parser = argparse.ArgumentParser(description='Run RCMMA frontend server')
    parser.add_argument('--source', default=0, help='Video source (0 or path)')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--chains-interval', type=int, default=10)
    parser.add_argument('--feedback', action='store_true')
    parser.add_argument('--feedback-threshold', type=float, default=0.2)
    parser.add_argument('--feedback-log', help='Optional path to feedback log')
    args = parser.parse_args()

    try:
        src = int(args.source)
    except Exception:
        src = args.source

    state = SharedState()
    stop_event = threading.Event()
    t = threading.Thread(target=pipeline_loop, args=(state, src, args.analyze, args.chains_interval, args.feedback, args.feedback_threshold, args.feedback_log, stop_event), daemon=True)
    t.start()

    app = create_app(state)
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        stop_event.set()
        t.join(timeout=1.0)


if __name__ == '__main__':
    main()
