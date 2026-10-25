import pandas as pd
import numpy as np
from causal_swarm.causal.scm import *
from causal_swarm.causal.graph import CausalGraph

def node_score(scm: SCM, data: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for node, p in scm.params.items():
        residual = scm.residual(node, data).abs()
        if node in LOGISTIC_NODE:
            prob = scm.predict(node, data).clip(1e-6, 1-1e-6) # avoid divided by 0
            scores[node] = residual / np.sqrt(prob * (1 - prob))
        else:
            scores[node] = residual / p["sigma"]
    return pd.DataFrame(scores).dropna() # drop first row for each drone as they contain empty value

def centrality_weights(graph: CausalGraph) -> dict:
    scored = [n for n in graph.graph.nodes if n not in EXTERNAL_INPUT_NODES]
    weights = {}
    for node in scored:
        related = set(graph.get_ancestors(node)) | set(graph.get_descendants(node))
        count = len([n for n in related if n not in EXTERNAL_INPUT_NODES])
        weights[node] = count / (2 * len(scored))
    return weights

def mcai(scores: pd.DataFrame, weights: dict) -> pd.Series:
    return sum(weights[node] * scores[node] for node in scores.columns)