"""Causal graph for follower nodes only"""

import networkx as nx
# import matplotlib.pyplot as plt

class CausalGraph:
    def __init__(self):
        self.graph = self.build()

    def build(self):
        g = nx.DiGraph()
        g.add_node("accel_z")
        g.add_node("gyro_x")
        g.add_node("gyro_y")
        g.add_node("altitude")
        g.add_edges_from([
            ("leader_pos_x", "pos_x"),
            ("satellite_count", "pos_x"),
            ("leader_pos_y", "pos_y"),
            ("satellite_count", "pos_y"),
            ("pos_x", "formation_error"),
            ("pos_y", "formation_error"),
            ("leader_pos_x", "formation_error"),
            ("leader_pos_y", "formation_error"),
            ("pos_x", "inter_drone_distance"),
            ("pos_y", "inter_drone_distance"),
            ("neighbor_pos_x", "inter_drone_distance"),
            ("neighbor_pos_y", "inter_drone_distance"),
            ("formation_error", "relative_velocity"),
            ("formation_error", "accel_magnitude"),
            ("mag_heading", "gyro_z"),
            ("inter_drone_distance", "signal_strength"),
            ("formation_error", "current_draw"),
            ("signal_strength", "packet_rate"),
            ("signal_strength", "latency"),
            ("signal_strength", "link_active"),
            ("current_draw", "motor_rpm"),
            ("current_draw", "battery_voltage")
        ])
        return g

    def get_parents(self, node: str) -> str:
        return list(self.graph.predecessors(node))

    def topological_order(self) -> list:
        return list(nx.topological_sort(self.graph))

# sanity check:

# g = CausalGraph().graph
# assert nx.is_directed_acyclic_graph(g)
# print(list(nx.topological_sort(g)))
# nx.draw(g, with_labels=False, node_color='skyblue')
# plt.show()