![](docs/images/rcmma-banner.png)

# RCMMA — Perception • Logic • Feedback

RCMMA implements an end-to-end research prototype for a perception → logic-analysis → feedback loop.
This repository contains a lightweight Python prototype that demonstrates each stage and a tiny demo frontend to visualize the full closed loop.

This README is intentionally comprehensive: it explains the architecture, how to run the demo locally, how to capture the dynamic generation of the knowledge graph from a video stream, and how to extend the system.

Table of contents
- Features
- Architecture diagram
- Quick start (Windows PowerShell)
- Perception: how video becomes a dynamic knowledge graph
- Logic analysis: extracting logical chains and mapping bboxes
- Feedback: turning chains into actions
- Frontend demo (live web UI)
- Saving, replay and export options
- Development notes and extension points
- Troubleshooting
- Tests
- Contributing

## Features

- Live video capture from webcam or a video file
- A lightweight MockDetector for fast prototyping (background subtraction + contours)
- A DynamicKnowledgeGraph (networkx) that records detections as nodes and simple relations (co-occurrence, proximity)
- Logic-analysis utilities that map image bboxes to graph nodes and extract candidate logical chains (ranked)
- A FeedbackController that turns high-scoring chains into simulated actions (alerts / log entries)
- A minimal Flask frontend to view: current video frame, rendered knowledge graph image, and recent feedback actions

## Architecture

At a high level the system is organized as:

- `rcmma/perception/` — video capture, detector, dynamic knowledge graph, visualization, CLI runner
- `rcmma/logic_analysis/` — algorithms that map bboxes to graph nodes and extract logical chains
- `rcmma/feedback/` — a simple controller that converts chains into actions
- `rcmma/frontend/` — a minimal Flask app that runs the pipeline in a background thread and serves live frames/graph/actions

Architecture diagram (conceptual)

```
[Video source] -> [Perception: Detector] -> [DynamicKnowledgeGraph]
																				 |-> [Visualizer window]
																				 |-> [Logic Analysis] -> [Feedback Controller] -> [Actions]
																				 |-> [Frontend Server (frame.jpg, graph.png, actions)]
```

You can create more detailed diagrams by exporting graph snapshots and combining them with screenshots of the video window. See the "Saving, replay and export options" section.

## Quick start (Windows PowerShell)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run unit tests to confirm everything is working:

```powershell
# Make sure the repository root is on PYTHONPATH so tests can import `rcmma`
$env:PYTHONPATH = 'D:\projectspycharm\RCMMA'; pytest -q
```

3. Run the perception CLI (webcam):

```powershell
python -m rcmma.perception.app --source 0 --max-frames 500 --save-graph out_graph.json
```

4. Run the frontend demo (web UI) and open your browser:

```powershell
python -m rcmma.frontend.server --source 0 --port 5000 --analyze --chains-interval 10 --feedback --feedback-log out/feedback.jsonl
# then open http://localhost:5000 in your browser
```

If you prefer using a video file instead of a webcam, pass the file path to `--source`:

```powershell
python -m rcmma.frontend.server --source "D:\path\to\video.mp4" --analyze --feedback
```

## Perception: how video becomes a dynamic knowledge graph

Files of interest:

- `rcmma/perception/capture.py` — `VideoSource` yields `(timestamp, frame)` tuples from webcam or file.
- `rcmma/perception/detector.py` — `MockDetector.detect(frame)` returns a list of detections: `{'id','label','bbox','confidence'}`. The mock detector uses OpenCV background subtraction + contour extraction for speed. Replace it with any model that returns the same detection schema.
- `rcmma/perception/knowledge_graph.py` — `DynamicKnowledgeGraph` stores detections as `networkx` nodes with attributes `label, bbox, confidence, timestamp`. It also attaches simple edges:
	- `co_occurs` for nodes detected within a short time window (default 0.5s)
	- `near` for nodes with bbox centers close to each other (heuristic)

The CLI runner (`rcmma/perception/app.py`) ties these pieces together to produce a growing graph as the video is processed.

Example (conceptual):

- Frame 1: two moving objects detected -> two nodes added and `co_occurs` edge created.
- Frame 10: one object moves near the other -> `near` edge created.

## Logic analysis: extracting logical chains and mapping bboxes

The logic-analysis module (`rcmma/logic_analysis/analyzer.py`) includes two primary functions:

- `map_bbox_to_nodes(g, bbox, iou_threshold)` — computes IoU between an input bbox and all node bboxes stored in graph `g`, returning matches ordered by IoU.
- `find_logical_chains(g, min_length=2, max_length=6, time_gap=2.0)` — enumerates simple paths within connected components of `g`, filters them to preserve non-decreasing timestamps with bounded gaps, and scores them by average detection confidence scaled by temporal compactness. It returns ranked candidate chains.

These outputs are intended to bridge image-level detections and symbolic reasoning: a chain is a short temporal-spatial sequence of detections that might represent an event or sequence of interest.

## Feedback: turning chains into actions

The `rcmma/feedback/controller.py` implements a simple `FeedbackController` that:

- Evaluates ranked chains and checks whether the top chain's score exceeds a threshold.
- If so, emits a simulated action: writes a JSON line to a feedback log file (if specified) and prints a concise alert to stdout.

The design is intentionally minimal so you can plug in a real actuator interface (MQTT, REST, ROS, etc.) or a more advanced decision/policy layer.

## Frontend demo (live web UI)

The demo server in `rcmma/frontend/server.py` runs the perception pipeline in a background thread and exposes a tiny web UI with three things:

- live video frame (`/frame.jpg`)
- current graph visualization (`/graph.png`)
- a JSON list of recent actions (`/actions`)

Open http://localhost:5000 after starting the server (see Quick start). The page polls every second and displays the latest frame, graph image, and action log.

Screenshot placeholders

If you want to include screenshots in presentations, the repository suggests the following paths for generated images (they are not checked in):

- `out_graph.json` — final graph exported by `--save-graph`
- `out/chains.jsonl` — per-interval chain snapshots (if you enable `--save-chains` or use the app hooks)
- `out/feedback.jsonl` — feedback event log

You can create presentation images by running the frontend and taking screenshots of the browser, or by exporting the graph image programmatically (the frontend saves PNG bytes when rendering).

## Saving, replay and export options

Several export options are available or easy to add:

1. Save the final graph as node-link JSON (already implemented): pass `--save-graph out_graph.json` to the CLI runner.
2. Save per-interval chain snapshots to a JSONL file: use the `--save-chains` option or the frontend's `--feedback-log` in combination with `--analyze`.
3. Record the graph visualization as a video: you can capture each PNG produced by the visualizer and write them to `cv2.VideoWriter` frames to produce an MP4. (If you want, I can add a `--record-graph out_graph.mp4` flag.)

Example: save a final graph JSON

```powershell
python -m rcmma.perception.app --source "D:\path\to\video.mp4" --max-frames 500 --save-graph out_graph.json
```

## Development notes and extension points

Where to plug in a real detector
- Replace `MockDetector` in `rcmma/perception/detector.py` with a class that implements `detect(frame)` and returns the same detection schema. Popular choices:
	- YOLOv5/YOLOv8 (PyTorch)
	- ONNX runtime for lighter-weight cross-platform deployment
	- Detectron2 for richer instance segmentation

Graph schema improvements
- Add event nodes, attributes (object color, speed), and explicit edge types (interacts_with, follows, contains). Consider using a property graph DB (Neo4j) for persistence and richer queries.

Logic/chain improvements
- Improve scoring with spatial consistency, semantic labels, or learned ranking models (train on annotated sequences).

Feedback and actuation
- Replace the print/log actions with a message bus (MQTT), HTTP POST to an actuator, or ROS topic publishing for robotic integration. Add safe confirmation and authentication for real actuators.

Frontend improvements
- Stream MJPEG for smooth video, or use WebRTC for low-latency streaming.
- Add interactive controls (threshold sliders, step-frame, play/pause).

## Troubleshooting

- If imports fail during tests, make sure the project root is on `PYTHONPATH` as the test commands above show.
- If the Flask server does not display images, ensure matplotlib backend is configured (the frontend sets `matplotlib.use('Agg')` for headless rendering) and OpenCV can encode JPEG frames.
- On headless machines (no GUI), do not use the `--visualize` flag which opens matplotlib interactive windows. Use the frontend server instead for remote viewing.

## Tests

Run the small test suite:

```powershell
$env:PYTHONPATH = 'D:\projectspycharm\RCMMA'; pytest -q
```

The repository includes unit tests for the knowledge graph, logic analysis, and feedback controller.

## Example workflows (end-to-end)

1) Rapid local demo (webcam):

```powershell
python -m rcmma.frontend.server --source 0 --port 5000 --analyze --chains-interval 10 --feedback --feedback-log out/feedback.jsonl
# open http://localhost:5000 and watch the loop
```

2) Offline processing + export graph:

```powershell
python -m rcmma.perception.app --source "D:\videos\demo.mp4" --max-frames 2000 --save-graph demo_out_graph.json
```

3) Record visual graph movie (manual flow)

- Run the frontend or the visualizer and save PNG snapshots for each interval (or I can add a helper to write MP4s).

## Contributing

Contributions are welcome. Good first issues:

- Integrate an ONNX-based object detector and add an installation instruction for model weights.
- Add a `--record-graph` option to the app to save an MP4 of the dynamic graph.
- Add authentication and a small REST API for publishing feedback actions to external systems.

Please open an issue describing your plan before implementing large changes.

## License

This prototype is provided under the MIT license. See `LICENSE` for details.

---

