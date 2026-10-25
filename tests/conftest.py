import pytest

from causal_swarm.causal.graph import CausalGraph
from causal_swarm.causal.scm import SCM, build_fitting_data
from causal_swarm.simulation.swarm_env import Simulator, SimulatorConfig


@pytest.fixture(scope="session")
def config():
    return SimulatorConfig(num_drone=4, num_timesteps=2000, dt=1.0, leader_id=0, seed=42)


@pytest.fixture(scope="session")
def data(config):
    # Shared across test files, so tests must copy it before changing anything.
    return build_fitting_data(Simulator(config).run(), config)


@pytest.fixture(scope="session")
def scm(data):
    return SCM.fit(CausalGraph(), data)
