"""Simple feedback controller.

This module implements a FeedbackController that inspects logical chains and
decides whether to emit an action. Actions are currently simulated by:
 - printing a short alert to stdout
 - appending a JSON line to a feedback log file (if provided)

The decision rule is simple and intended as a placeholder: if the top chain's
score exceeds a threshold, emit an action that references the node ids and timestamps.
"""
from typing import Optional, Dict, Any, List
import json
import time
import os


class FeedbackController:
    def __init__(self, threshold: float = 0.2, feedback_log: Optional[str] = None):
        self.threshold = float(threshold)
        self.feedback_log = feedback_log

    def evaluate_and_act(self, chains: List[Dict[str, Any]], graph: Any, frame: int = 0) -> Optional[Dict[str, Any]]:
        """Evaluate chains and optionally produce an action dict.

        Returns the action dict if acted, otherwise None.
        """
        if not chains:
            return None

        top = chains[0]
        score = float(top.get("score", 0.0))
        if score < self.threshold:
            return None

        action = {
            "time": time.time(),
            "frame": frame,
            "action": "alert_chain_detected",
            "score": score,
            "nodes": top.get("nodes", []),
            "times": top.get("times", []),
        }

        # print a concise alert
        print(f"FEEDBACK: action={action['action']} score={action['score']:.3f} nodes={action['nodes']}")

        # append to log if requested
        if self.feedback_log:
            os.makedirs(os.path.dirname(self.feedback_log), exist_ok=True)
            with open(self.feedback_log, "a", encoding="utf-8") as f:
                json.dump(action, f)
                f.write("\n")

        return action
