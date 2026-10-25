"""Tests for the attack definitions and how the simulator applies them."""

import pytest

from causal_swarm.simulation.attacks import (
    DoSFlood,
    GPSSpoof,
    active_attack,
    attack_label,
)
from causal_swarm.simulation.swarm_env import Simulator, SimulatorConfig

CONFIG = {"num_drone": 4, "num_timesteps": 1200, "dt": 1.0, "leader_id": 0, "seed": 7}
SPOOF = GPSSpoof(drone_id=3, start=400, end=700, drift_per_step=0.1)
DOS = DoSFlood(drone_id=2, start=800, end=900)


def run(attacks=()):
    return Simulator(SimulatorConfig(**CONFIG, attacks=list(attacks))).run()


def telemetry(df):
    return df.drop(columns="attack_type")


def rows(df, drone, start, end):
    """One drone's rows for timesteps start <= t < end."""
    return df[(df["drone_id"] == drone) & df["timestep"].between(start, end - 1)]


@pytest.fixture(scope="module")
def benign():
    return run()


@pytest.fixture(scope="module")
def attacked():
    return run([SPOOF, DOS])


# ---- the attack definitions ----------------------------------------------

def test_attack_window_end_is_exclusive():
    assert SPOOF.is_active(400) and SPOOF.is_active(699)
    assert not SPOOF.is_active(399) and not SPOOF.is_active(700)


def test_spoof_offset_grows_linearly_along_the_normalized_direction():
    spoof = GPSSpoof(drone_id=1, start=10, end=100, drift_per_step=0.1, direction=(1.0, 1.0))
    assert spoof.offset(10) == (0.0, 0.0)
    assert spoof.offset(60) == pytest.approx((5 / 2 ** 0.5, 5 / 2 ** 0.5))   # 5 m in total
    assert spoof.offset(100) == (0.0, 0.0)                                  # the window has ended


def test_dos_helpers():
    assert DOS.degrade(50.0, 20.0) == pytest.approx((5.0, 220.0))
    assert DOS.link_probability(0.98) == pytest.approx(0.098)


def test_active_attack_filters_by_drone_time_and_kind():
    attacks = [SPOOF, DOS]
    assert active_attack(attacks, 3, 500) is SPOOF
    assert active_attack(attacks, 3, 500, DoSFlood) is None
    assert active_attack(attacks, 1, 500) is None
    assert active_attack(attacks, 3, 700) is None
    assert attack_label(attacks, 2, 850) == "dos"
    assert attack_label(attacks, 1, 850) == "none"


# ---- how the simulator applies them --------------------------------------

def test_benign_run_is_labeled_none(benign):
    assert benign["attack_type"].unique().tolist() == ["none"]


def test_attacks_scheduled_after_the_run_change_nothing(benign):
    late = run([GPSSpoof(drone_id=3, start=5000, end=6000), DoSFlood(drone_id=2, start=5000, end=6000)])
    assert late.equals(benign)


def test_run_is_identical_to_benign_before_the_first_attack(benign, attacked):
    before_benign = benign[benign["timestep"] < 400]
    before_attacked = attacked[attacked["timestep"] < 400]
    assert telemetry(before_benign).equals(telemetry(before_attacked))


def test_gps_spoof_pushes_the_drone_by_the_drift(benign, attacked):
    t = 500   # 100 steps after the start, so 100 * 0.1 = 10 m
    shift = rows(attacked, 3, t, t + 1)["pos_x"].iloc[0] - rows(benign, 3, t, t + 1)["pos_x"].iloc[0]
    assert shift == pytest.approx(10.0)


def test_gps_spoof_raises_formation_error_and_current_draw(benign, attacked):
    for column in ("formation_error", "current_draw"):
        spoofed = rows(attacked, 3, 600, 700)[column].mean()
        normal = rows(benign, 3, 600, 700)[column].mean()
        assert spoofed > 2 * normal, column


def test_other_drones_are_untouched_during_the_spoof(benign, attacked):
    for drone in (0, 1, 2):
        assert telemetry(rows(benign, drone, 400, 700)).equals(telemetry(rows(attacked, drone, 400, 700)))


def test_dos_degrades_the_attacked_link(benign, attacked):
    hit = rows(attacked, 2, 800, 900)
    normal = rows(benign, 2, 800, 900)
    assert hit["packet_rate"].mean() < 0.2 * normal["packet_rate"].mean()
    assert hit["latency"].mean() > normal["latency"].mean() + 150
    assert hit["link_active"].mean() < 0.3


def test_labels_follow_the_attack_windows(attacked):
    drone3 = attacked[attacked["drone_id"] == 3].set_index("timestep")["attack_type"]
    assert drone3[399] == "none" and drone3[400] == "gps_spoof"
    assert drone3[699] == "gps_spoof" and drone3[700] == "none"

    counts = attacked.groupby(["drone_id", "attack_type"]).size()
    assert counts[(3, "gps_spoof")] == 300
    assert counts[(2, "dos")] == 100
    assert (attacked[~attacked["drone_id"].isin([2, 3])]["attack_type"] == "none").all()


def test_attacking_the_leader_raises():
    with pytest.raises(ValueError):
        run([GPSSpoof(drone_id=0, start=1, end=2)])
