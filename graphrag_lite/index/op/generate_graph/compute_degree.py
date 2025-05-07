
"""A module containing create_graph definition."""

import networkx as nx
import pandas as pd


def compute_degree(graph: nx.Graph) -> pd.DataFrame:
    """Create a new DataFrame with the degree of each node in the graph."""
    return pd.DataFrame([
        {"alias": node, "degree": int(degree)}
        for node, degree in graph.degree  # type: ignore
    ])
