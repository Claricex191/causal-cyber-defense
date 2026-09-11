from graph import CausalGraph
from simulation.swarm_env import SimulatorConfig
import pandas as pd

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
    return pd.concat(frames, ignore_index=True)

EXTERNAL_INPUT_NODES = ["satellite_count", "altitude", "accel_z", "gyro_x", "gyro_y"]
CIRCULAR_ROOT_NODE = ["mag_heading"]

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

def _fit_logistic(data: pd.DataFrame, node: str, parents: )


