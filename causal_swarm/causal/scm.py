from causal_swarm.causal.graph import CausalGraph
from causal_swarm.simulation.swarm_env import SimulatorConfig
import pandas as pd
import numpy as np
import statsmodels.api as sm

# helper:
def build_fitting_data(raw_df: pd.DataFrame, config: SimulatorConfig) -> pd.DataFrame:
    leader_df = raw_df[raw_df.drone_id==config.leader_id][["timestep", "pos_x", "pos_y"]]
    leader_df = leader_df.rename(columns={"pos_x": "leader_pos_x", "pos_y": "leader_pos_y"})

    frames = []
    for drone_id, neighbor_id in config.neighbor_pairs.items():
        drone_df = raw_df[raw_df.drone_id==drone_id]
        neighbor_df = raw_df[raw_df.drone_id==neighbor_id][["timestep", "pos_x", "pos_y"]]
        neighbor_df = neighbor_df.rename(columns={"pos_x": "neighbor_pos_x", "pos_y": "neighbor_pos_y"})

        merged = drone_df.merge(leader_df, on="timestep").merge(neighbor_df, on="timestep")
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)

    # accel_x/accel_y mix a magnitude that scales with formation_error together
    # with a direction that's a nonlinear function of position, fitting them
    # directly produces a heteroscedastic residual. accel_magnitude alone is
    # linear in formation_error by construction (see swarm_env.py), so that's
    # what the causal graph fits instead; accel_x/accel_y stay as raw telemetry.
    result["accel_magnitude"] = np.hypot(result["accel_x"], result["accel_y"])
    return result

CIRCULAR_ROOT_NODE = ["mag_heading"]
EXTERNAL_INPUT_NODES = ["leader_pos_x", "leader_pos_y", "neighbor_pos_x", "neighbor_pos_y"]

class SCM:
    def __init__(self, causal_graph: CausalGraph, params: dict):
        self.causal_graph = causal_graph
        self.params = params

    @classmethod
    def fit(cls, causal_graph: CausalGraph, data: pd.DataFrame) -> "SCM":
        params = {}
        for node in causal_graph.topological_order():
            parents = causal_graph.get_parents(node)
            if node in EXTERNAL_INPUT_NODES: # leader_pos_x etc. they aren't fit, they're given
                continue
            elif node in CIRCULAR_ROOT_NODE:
                params[node] = cls._fit_circular_root(data, node)
            elif not parents:
                params[node] = cls._fit_root(data, node) # root nodes do not have parents
            elif node == "link_active":
                params[node] = cls._fit_logistic(data, node, parents) # for binary node
            elif node == "battery_voltage":
                params[node] = cls._fit_with_lag(data, node, parents)
            else:
                params[node] = cls._fit_linear(data, node, parents) # for continuous node
        return cls(causal_graph, params)


    @staticmethod
    def _fit_root(data: pd.DataFrame, node: str) -> dict:
        return {"alpha": data[node].mean(), "sigma": data[node].std()}

    @staticmethod
    def _fit_circular_root(data: pd.DataFrame, node: str) -> dict:
        radians = np.deg2rad(data[node])
        mean_angle = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360
        R = np.hypot(np.cos(radians).mean(), np.sin(radians).mean())
        circular_std = np.rad2deg(np.sqrt(-2 * np.log(R)))
        return {
            "alpha": mean_angle, 
            "sigma": circular_std,
        }

    @staticmethod
    def _fit_logistic(data: pd.DataFrame, node: str, parents: list) -> dict:
        X = sm.add_constant(data[parents])
        y = data[node].astype(int)
        model = sm.Logit(y, X).fit()
        return {
            "alpha": model.params["const"],
            "betas": {p: model.params[p] for p in parents},
        }

    @staticmethod
    def _fit_linear(data: pd.DataFrame, node: str, parents: list) -> dict:
        X = sm.add_constant(data[parents])
        y = data[node]
        model = sm.OLS(y, X).fit()
        return {
            "alpha": model.params["const"],
            "betas": {p: model.params[p] for p in parents},
            "sigma": model.resid.std(),
        }

    @staticmethod
    def _fit_with_lag(data: pd.DataFrame, node: str, parents: list) -> dict:
        df = data.copy()
        lag_col = f"{node}_lag1"
        df[lag_col] = df.groupby("drone_id")[node].shift(1)
        df = df.dropna(subset=[lag_col])
        regressors = parents + [lag_col]
        X = sm.add_constant(df[regressors])
        y = df[node]
        model = sm.OLS(y, X).fit()
        return {
            "alpha": model.params["const"],
            "betas": {p: model.params[p] for p in parents},
            "lag_beta": model.params[lag_col],
            "sigma": model.resid.std(),
        }