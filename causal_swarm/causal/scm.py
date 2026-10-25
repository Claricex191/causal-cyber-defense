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

        # Assigned formation slot: where this drone is supposed to be. Without
        # it, one shared intercept can't absorb each drone's different offset,
        # which inflates sigma for pos_x/pos_y and biases every drone's residual.
        offset_x, offset_y = config.formation_offsets[drone_id]
        merged["slot_x"] = merged["leader_pos_x"] + offset_x
        merged["slot_y"] = merged["leader_pos_y"] + offset_y
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["drone_id", "timestep"]).reset_index(drop=True)

    # Previous-timestep features, computed per drone so a shift never reaches
    # across drone boundaries. The first row of each drone is NaN.
    by_drone = result.groupby("drone_id")
    result["battery_voltage_lag1"] = by_drone["battery_voltage"].shift(1)
    # gyro_z measures turning rate: shortest signed heading change per timestep
    # (handles the 359 -> 1 degree wraparound), same formula as swarm_env.py.
    heading_diff = result["mag_heading"] - by_drone["mag_heading"].shift(1)
    result["heading_delta"] = ((heading_diff + 180) % 360 - 180) / config.dt

    # The causal graph fits accel_magnitude (one node, linear in formation_error by
    # construction, see swarm_env.py) instead of accel_x and accel_y. Those would
    # also fit linearly given slot_x / slot_y (docs/experiments.md, experiment 4);
    # they stay as raw telemetry columns.
    result["accel_magnitude"] = np.hypot(result["accel_x"], result["accel_y"])
    return result

CIRCULAR_ROOT_NODE = ["mag_heading"]
LOGISTIC_NODE = ["link_active"]
# Given or derived in build_fitting_data, so they are inputs, not fit.
EXTERNAL_INPUT_NODES = ["slot_x", "slot_y", "neighbor_pos_x", "neighbor_pos_y",
                        "heading_delta", "battery_voltage_lag1"]

class SCM:
    def __init__(self, causal_graph: CausalGraph, params: dict):
        self.causal_graph = causal_graph
        self.params = params

    @classmethod
    def fit(cls, causal_graph: CausalGraph, data: pd.DataFrame) -> "SCM":
        params = {}
        for node in causal_graph.topological_order():
            parents = causal_graph.get_parents(node)
            if node in EXTERNAL_INPUT_NODES: # slot_x etc. they aren't fit, they're given
                continue
            elif node in CIRCULAR_ROOT_NODE:
                params[node] = cls._fit_circular_root(data, node)
            elif not parents:
                params[node] = cls._fit_root(data, node) # root nodes do not have parents
            elif node == "link_active":
                params[node] = cls._fit_logistic(data, node, parents) # for binary node
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
        df = data[[node] + parents].dropna() # drops each drone's first row (lag/delta columns are NaN there)
        X = sm.add_constant(df[parents])
        y = df[node]
        model = sm.OLS(y, X).fit()
        return {
            "alpha": model.params["const"],
            "betas": {p: model.params[p] for p in parents},
            "sigma": model.resid.std(),
        }

    def predict(self, node: str, data: pd.DataFrame) -> pd.Series:
        p = self.params[node]                       # input nodes have no params, so this raises KeyError
        linear = pd.Series(p["alpha"], index=data.index)
        for col, beta in p.get("betas", {}).items():
            linear = linear + beta * data[col]
        if node in LOGISTIC_NODE:
            return 1 / (1 + np.exp(-linear))        # probability that link_active = 1
        return linear

    def residual(self, node: str, data: pd.DataFrame) -> pd.Series:
        diff = data[node].astype(float) - self.predict(node, data)
        if node in CIRCULAR_ROOT_NODE:
            diff = (diff + 180) % 360 - 180         # shortest signed angle
        return diff