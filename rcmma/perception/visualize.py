"""Graph visualization utilities for RCMMA perception.

This module provides a simple live visualizer for the DynamicKnowledgeGraph
using matplotlib + networkx. It's lightweight and intended for debugging and
demonstration. For richer web-based visualization consider `pyvis` or D3.
"""
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:
    def __init__(self, figsize=(6, 6), title: str = "RCMMA Knowledge Graph"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.pos = None

    def update(self, nx_graph: nx.Graph):
        """Redraw the provided networkx graph.

        This recomputes the spring layout each call which is fine for small/medium graphs.
        """
        self.ax.clear()
        self.ax.set_title(self.ax.get_title())
        if nx_graph.number_of_nodes() == 0:
            self.ax.text(0.5, 0.5, "(no nodes)", horizontalalignment="center", verticalalignment="center")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return

        try:
            # compute layout
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
        except Exception:
            pos = nx.circular_layout(nx_graph)

        # node colors by label if available
        labels = nx.get_node_attributes(nx_graph, "label")
        node_colors = []
        node_sizes = []
        for n in nx_graph.nodes():
            lbl = labels.get(n, "")
            if lbl == "object":
                node_colors.append("#1f78b4")
            else:
                node_colors.append("#33a02c")
            conf = nx_graph.nodes[n].get("confidence", 0.5)
            node_sizes.append(300 + float(conf) * 700)

        nx.draw_networkx_edges(nx_graph, pos, ax=self.ax, alpha=0.6)
        nx.draw_networkx_nodes(nx_graph, pos, ax=self.ax, node_color=node_colors, node_size=node_sizes)

        # draw labels as node ids
        nx.draw_networkx_labels(nx_graph, pos, {n: n for n in nx_graph.nodes()}, ax=self.ax, font_size=8)

        self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
