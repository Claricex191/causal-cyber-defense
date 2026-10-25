"""Tests for the causal graph and SCM fitting.

The constants in swarm_env.py are the ground truth, so fitting the SCM on
simulated benign data should recover them.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_swarm.causal.graph import CausalGraph
from causal_swarm.causal.scm import EXTERNAL_INPUT_NODES, SCM

# The config, data and scm fixtures live in conftest.py.


def test_graph_is_acyclic():
    assert nx.is_directed_acyclic_graph(CausalGraph().graph)


def test_every_node_is_fit_except_the_inputs(scm):
    graph_nodes = set(scm.causal_graph.graph.nodes)
    assert set(scm.params) == graph_nodes - set(EXTERNAL_INPUT_NODES)


def test_previous_value_columns_do_not_cross_drones(data):
    for _, drone in data.groupby("drone_id"):
        assert drone["battery_voltage_lag1"].isna().sum() == 1
        assert drone["heading_delta"].isna().sum() == 1
        np.testing.assert_allclose(
            drone["battery_voltage_lag1"].to_numpy()[1:],
            drone["battery_voltage"].to_numpy()[:-1],
        )


@pytest.mark.parametrize("node, parent, true_value, tol", [
    ("relative_velocity", "formation_error", 0.3, 0.01),
    ("accel_magnitude", "formation_error", 0.2, 0.01),
    ("signal_strength", "inter_drone_distance", -0.5, 0.05),
    ("gyro_z", "heading_delta", 1.0, 0.02),
    ("battery_voltage", "battery_voltage_lag1", 1.0, 0.005),
    ("battery_voltage", "current_draw", -0.002, 0.0005),
    ("pos_x", "slot_x", 1.0, 0.05),
    ("pos_y", "slot_y", 1.0, 0.05),
])
def test_coefficients_recover_ground_truth(scm, node, parent, true_value, tol):
    assert scm.params[node]["betas"][parent] == pytest.approx(true_value, abs=tol)


@pytest.mark.parametrize("node, true_sigma", [
    ("relative_velocity", 0.05),
    ("accel_magnitude", 0.05),
    ("gyro_z", 0.5),
    ("battery_voltage", 0.01),
    ("signal_strength", 2.0),
    ("packet_rate", 2.0),
    ("latency", 2.0),
    ("motor_rpm", 20.0),
    ("satellite_count", 1.0),
])
def test_sigma_recovers_true_noise(scm, node, true_sigma):
    assert scm.params[node]["sigma"] == pytest.approx(true_sigma, rel=0.1)


def test_formation_offsets_do_not_inflate_pos_sigma_or_bias_drones(scm, data):
    # Without slot_x/slot_y the followers' different offsets inflated sigma
    # (3.1 / 4.6) and left each drone with a constant residual of up to 5 m.
    for node in ["pos_x", "pos_y"]:
        assert 1.7 < scm.params[node]["sigma"] < 2.6
        residual = scm.residual(node, data)
        for _, drone_residual in residual.groupby(data["drone_id"]):
            assert abs(drone_residual.mean()) < 1.0


def test_residuals_are_centered_and_match_sigma(scm, data):
    for node, p in scm.params.items():
        if "sigma" not in p:
            continue
        r = scm.residual(node, data).dropna()
        assert abs(r.mean()) < 0.05 * p["sigma"], node
        assert r.std() == pytest.approx(p["sigma"], rel=0.02), node


def test_circular_residual_takes_the_shortest_angle(scm):
    circular_scm = SCM(scm.causal_graph, {"mag_heading": {"alpha": 1.0, "sigma": 1.0}})
    observed = pd.DataFrame({"mag_heading": [359.0, 1.0, 180.0]})
    assert circular_scm.residual("mag_heading", observed).tolist() == [-2.0, 0.0, 179.0]


def test_logistic_prediction_is_a_probability(scm, data):
    p = scm.predict("link_active", data)
    assert ((p > 0) & (p < 1)).all()


def test_predict_on_an_input_node_raises(scm, data):
    with pytest.raises(KeyError):
        scm.predict("slot_x", data)


def test_predict_is_missing_only_on_each_drones_first_row(scm, data):
    for node in ["battery_voltage", "gyro_z"]:
        assert scm.predict(node, data).isna().sum() == data["drone_id"].nunique()
