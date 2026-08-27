"""Figures: network topologies and training curves.

Everything here takes an optional ``ax`` so the same drawing code serves both
standalone figures and the multi-panel summary grids.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import config as cfg
from .genome import (
    ACT_NAMES,
    BIAS,
    HIDDEN,
    INPUT,
    OUTPUT,
    UNUSED,
    n_connections,
    n_hidden,
    node_layers,
)

NODE_COLORS = {
    INPUT: "#6db1ff",
    BIAS: "#7ed07e",
    HIDDEN: "#ffb14e",
    OUTPUT: "#fa7268",
}


def draw_genome(
    ind,
    ax=None,
    title="genome",
    save_path=None,
    label_activations=False,
    input_labels=None,
    figsize=(9, 5),
):
    """Draw one genome as a layered graph: inputs on the left, outputs right.

    Horizontal position is the node's depth, so the drawing reflects the actual
    order of computation. Edge width tracks weight magnitude and red edges are
    negative weights. With ``label_activations`` the hidden nodes are labelled
    with their activation function instead of their slot index -- readable for
    the small classification networks, cluttered for the big ones.
    """
    node_type = np.asarray(ind.node_type)
    node_act = np.asarray(ind.node_act)
    conn_in = np.asarray(ind.conn_in)
    conn_out = np.asarray(ind.conn_out)
    conn_w = np.asarray(ind.conn_w)
    conn_on = np.asarray(ind.conn_on, dtype=bool)
    layer = node_layers(ind)

    graph = nx.DiGraph()
    graph.add_nodes_from(int(i) for i in np.where(node_type != UNUSED)[0])
    for k in np.where(conn_on)[0]:
        graph.add_edge(int(conn_in[k]), int(conn_out[k]), weight=float(conn_w[k]))

    # x = depth, y = spread evenly within the layer
    pos = {}
    for depth in sorted({int(v) for v in layer if v >= 0}):
        in_layer = [int(i) for i in np.where(layer == depth)[0]]
        for j, i in enumerate(in_layer):
            pos[i] = (depth, j - (len(in_layer) - 1) / 2.0)

    node_colors = [NODE_COLORS[int(node_type[i])] for i in graph.nodes]
    edge_colors = [
        "#d1495b" if graph[u][v]["weight"] < 0 else "#2e2e2e" for u, v in graph.edges
    ]
    edge_widths = [
        0.5 + 2.0 * min(abs(graph[u][v]["weight"]), cfg.W_CAP) / cfg.W_CAP
        for u, v in graph.edges
    ]

    if label_activations:
        labels, node_size, font_size = {}, 620, 6.5
        for i in graph.nodes:
            kind = int(node_type[i])
            if kind == INPUT:
                labels[i] = (
                    input_labels[i] if input_labels and i < len(input_labels) else f"in{i}"
                )
            elif kind == BIAS:
                labels[i] = "b"
            elif kind == OUTPUT:
                labels[i] = "out"
            else:
                labels[i] = ACT_NAMES[int(node_act[i])][:4]
    else:
        labels, node_size, font_size = None, 350, 7

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=figsize)

    nx.draw(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        labels=labels,
        with_labels=True,
        node_size=node_size,
        font_size=font_size,
        arrows=True,
        arrowsize=8,
    )
    ax.set_title(
        f"{title}  ({int(n_hidden(ind))} hidden nodes, "
        f"{int(n_connections(ind))} connections)",
        fontsize=10,
    )
    ax.axis("off")

    if save_path and own_figure:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if own_figure:
        plt.close(ax.figure)
    return ax


def plot_fitness(history, title="NEAT", ylabel="best episode reward", save_path=None):
    """Best-of-population fitness against generation."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(history, lw=1.6, color="#2c5f8a")
    ax.axhline(0, ls="--", lw=1, c="gray")  # break-even against the built-in AI
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_fitness_and_complexity(
    history, hidden_hist, title="Backprop NEAT", save_path=None
):
    """Fitness and mean hidden-node count on twin axes.

    Putting them together shows the thing worth seeing: fitness sits on a
    plateau -- the best a linear model can do -- until the population grows its
    first hidden node, and the jump follows the complexification.
    """
    h = np.asarray(history, dtype=float)
    ah = np.asarray(hidden_hist, dtype=float)
    gens = np.arange(len(h))

    plateau = h[0]
    jumped = np.where(h > plateau + 0.05)[0]
    jump = int(jumped[0]) if len(jumped) else None

    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(gens, h, lw=2, color="#2c5f8a", label="best fitness")
    ax1.set_xlabel("generation")
    ax1.set_ylabel(r"best fitness  ($-$loss $-$ penalty)", color="#2c5f8a")
    ax1.tick_params(axis="y", labelcolor="#2c5f8a")
    ax1.axhline(plateau, ls=":", lw=1, color="#888")
    # Offsets are in points, not data units: a run whose fitness barely moves
    # has a near-zero y-range, and a data-unit offset would then place this
    # label far off-canvas, which bbox_inches="tight" would try to include.
    ax1.annotate(
        "generation-0 plateau",
        xy=(0.5, plateau),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#666",
    )

    ax2 = ax1.twinx()
    ax2.plot(gens, ah, lw=2, color="#c1553b", label="mean hidden nodes")
    ax2.set_ylabel("population mean hidden nodes", color="#c1553b")
    ax2.tick_params(axis="y", labelcolor="#c1553b")

    if jump is not None:
        ax1.axvline(jump, ls="--", lw=1.2, color="#444", zorder=0)
        ax1.annotate(
            f"gen {jump}: fitness leaves\nthe plateau",
            xy=(jump, h[jump]),
            xytext=(34, -34),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1),
        )

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9)
    ax1.set_title(title, fontsize=11)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return jump
