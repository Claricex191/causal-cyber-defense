"""Tests for the causal anomaly scores (eq. 9), centrality weights (eq. 11)
and MCAI (eq. 10). The config, data and scm fixtures live in conftest.py."""

import pandas as pd
import pytest

from causal_swarm.causal.anomaly import centrality_weights, mcai, node_score
from causal_swarm.causal.graph import CausalGraph
from causal_swarm.causal.scm import EXTERNAL_INPUT_NODES, LOGISTIC_NODE


@pytest.fixture(scope="module")
def weights():
    return centrality_weights(CausalGraph())


@pytest.fixture(scope="module")
def benign_scores(scm, data):
    return node_score(scm, data)


@pytest.mark.parametrize("node, expected", [
    ("pos_x", 0.300),            # 12 scored ancestors/descendants / (2 * 20 scored nodes)
    ("formation_error", 0.200),  # 8 / 40
    ("signal_strength", 0.175),  # 7 / 40
    ("battery_voltage", 0.125),  # 5 / 40
    ("mag_heading", 0.025),      # 1 / 40 (heading_delta is an input node, so it isn't counted)
    ("accel_z", 0.0),            # no edges, so it never contributes
])
def test_weights_match_hand_counts(weights, node, expected):
    assert weights[node] == pytest.approx(expected)


def test_weights_cover_exactly_the_scored_nodes(scm, weights):
    assert set(weights) == set(scm.params)
    assert not set(weights) & set(EXTERNAL_INPUT_NODES)


def test_scores_drop_each_drones_first_row(scm, data, benign_scores):
    assert len(benign_scores) == len(data) - data["drone_id"].nunique()
    assert not benign_scores.isna().any().any()
    assert set(benign_scores.columns) == set(scm.params)


def test_benign_continuous_scores_average_about_point_eight(benign_scores):
    # For normally distributed errors the average of |error| / sigma is about 0.8.
    continuous = benign_scores.drop(columns=LOGISTIC_NODE)
    assert continuous.mean().between(0.75, 0.85).all()


def test_link_active_score_is_small_when_up_and_large_when_down(scm, data):
    scores = {}
    for state in (True, False):
        row = data.iloc[[10]].copy()   # not a drone's first row, so no empty values
        row["link_active"] = state
        scores[state] = node_score(scm, row)["link_active"].iloc[0]
    assert scores[True] < 0.5
    assert scores[False] > 5


def test_shifted_value_raises_its_own_score_and_the_mcai(scm, data, weights, benign_scores):
    attacked = data.copy()
    rows = attacked.index[(attacked["drone_id"] == 1) & (attacked["timestep"] >= 1000)]
    attacked.loc[rows, "pos_y"] += 5.0

    attacked_scores = node_score(scm, attacked)
    hit = attacked_scores.index.intersection(rows)

    assert attacked_scores.loc[hit, "pos_y"].mean() > 1.5 * benign_scores.loc[hit, "pos_y"].mean()
    assert mcai(attacked_scores, weights).loc[hit].mean() > mcai(benign_scores, weights).loc[hit].mean()


def test_mcai_is_the_weighted_sum_per_row():
    scores = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 0.0]}, index=[7, 9])
    result = mcai(scores, {"a": 0.5, "b": 0.1})
    assert result.tolist() == pytest.approx([0.8, 1.0])
    assert result.index.tolist() == [7, 9]


def test_zero_weight_nodes_do_not_change_the_mcai(benign_scores, weights):
    inflated = benign_scores.copy()
    inflated["accel_z"] += 100.0
    assert mcai(inflated, weights).tolist() == pytest.approx(mcai(benign_scores, weights).tolist())
