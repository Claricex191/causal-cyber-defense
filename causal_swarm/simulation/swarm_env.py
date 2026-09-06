"""Multi-drone swarm telemetry simulator.

Forward-samples a structural causal model to generate synthetic swarm
telemetry: one leader drone follows an exogenous random-walk mission
trajectory, and follower drones track a fixed formation slot relative to
the leader. Parameters here are hand-picked ground truth, not fitted, see
causal_swarm/causal/scm.py for the MLE-fitted version run on the resulting
data. See README.md for the full data schema and causal graph.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DroneState:
    drone_id: int
    timestep: int

    # GPS/nav
    pos_x: float
    pos_y: float
    altitude: float
    satellite_count: float

    # IMU
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_heading: float

    # Pairwise, relative to a fixed neighbor
    packet_rate: float
    signal_strength: float
    latency: float
    link_active: bool

    # Formation
    inter_drone_distance: float
    relative_velocity: float
    formation_error: float

    # Power
    battery_voltage: float
    current_draw: float
    motor_rpm: float


@dataclass
class NoiseState:
    # Autocorrelated (AR(1)) noise terms, carried over between timesteps
    pos_noise_x: float = 0.0
    pos_noise_y: float = 0.0
    heading_noise: float = 0.0


@dataclass
class SimulatorConfig:
    num_drone: int
    num_timesteps: int
    dt: float
    leader_id: int
    formation_offsets: dict = field(default_factory=lambda: {
        0: (0.0, 0.0),     # leader itself, offset from itself is 0.0
        1: (-5.0, -5.0),   # follower 1: 5m behind, 5m left of leader
        2: (-5.0, 5.0),    # follower 2: 5m behind, 5m right of leader
        3: (-10.0, 0.0),   # follower 3: 10m directly behind leader
    })  # drone_id -> (offset_x, offset_y) from leader
    neighbor_pairs: dict = field(default_factory=lambda: {
        1: 2,   # drone 1's comms neighbor is drone 2
        2: 1,
        3: 1,   # drone 3's comms neighbor is drone 1
    })  # drone_id -> neighbor drone_id, leader excluded
    seed: Optional[int] = None
    leader_start_pos: tuple = (0.0, 0.0)


class Simulator:
    """Generates benign (or attack-injected) multi-drone telemetry."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.noise_states: dict[int, NoiseState] = {
            i: NoiseState() for i in range(config.num_drone)
        }

        # Root/exogenous variable params
        self.heading_baseline = 0.0
        self.heading_noise_sigma = 5.0 # degrees per step
        self.satellite_baseline = 10.0
        self.satellite_noise_sigma = 1.0
        self.target_altitude = 50.0
        self.altitude_noise_sigma = 0.5
        self.leader_step_sigma = 1.0 # meters per step

        # Follower position tracking (base sigma at satellite_baseline;
        # scaled by satellite_count, fewer satellites -> noisier tracking)
        self.pos_noise_sigma = 0.5

        # IMU
        self.gyro_noise_sigma = 0.5

        # Formation dynamics
        self.relvel_gain = 0.3
        self.relvel_noise_sigma = 0.05
        self.accel_gain = 0.2
        self.accel_noise_sigma = 0.05

        # Comms (signal_strength in dBm; more negative = weaker)
        self.signal_baseline = -30.0
        self.signal_decay_per_meter = 0.5
        self.signal_noise_sigma = 2.0
        self.packet_rate_baseline = 50.0
        self.packet_rate_gain = 0.3 # packets/sec per dB
        self.packet_rate_noise_sigma = 2.0
        self.latency_baseline = 20.0
        self.latency_gain = 0.4 # ms per dB of weakness
        self.latency_noise_sigma = 2.0
        self.link_active_alpha = 6.0 # eq. 7 logistic intercept
        self.link_active_beta = 0.06 # eq. 7 logistic slope on signal_strength

        # Power
        self.current_baseline = 2.0
        self.current_accel_gain = 0.5
        self.current_formation_gain = 0.3
        self.current_noise_sigma = 0.1
        self.motor_rpm_gain = 400.0
        self.motor_rpm_noise_sigma = 20.0
        self.discharge_rate = 0.002
        self.battery_start = 100.0
        self.battery_noise_sigma = 0.01

    # ================= HELPER FUNCTIONS =================

    def _ar1_step(self, prev_noise: float, rho: float, sigma: float) -> float:
        # One step of an AR(1) process: noise[t] = rho * noise[t-1] + eps[t]
        eps = self.rng.normal(0, sigma)
        return rho * prev_noise + eps

    def _generate_exogenous_roots(self, drone_id: int) -> dict:
        # Generate values for root nodes/exogenous nodes (mag_heading, satellite_count, altitude), independent per drone, every timestep
        noise_state = self.noise_states[drone_id]
        noise_state.heading_noise = self._ar1_step(
            noise_state.heading_noise, rho=0.97, sigma=self.heading_noise_sigma
        )
        mag_heading = (self.heading_baseline + noise_state.heading_noise) % 360

        sat_noise = self.rng.normal(0, self.satellite_noise_sigma)
        satellite_count = float(np.clip(self.satellite_baseline + sat_noise, 4, 16))

        alt_noise = self.rng.normal(0, self.altitude_noise_sigma)
        altitude = self.target_altitude + alt_noise

        return {
            "mag_heading": mag_heading,
            "satellite_count": satellite_count,
            "altitude": altitude,
        }

    def _gyro_from_heading(self, prev_state: Optional[DroneState], mag_heading: float) -> tuple:
        """gyro_z from heading rate of change (shortest signed angular
        difference, handles the 359 -> 1 degree wraparound). gyro_x/y are
        plain noise since roll/pitch aren't modeled in this 2D simulation."""
        if prev_state is None:
            heading_rate = 0.0
        else:
            diff = (mag_heading - prev_state.mag_heading + 180) % 360 - 180
            heading_rate = diff / self.config.dt
        gyro_z = heading_rate + self.rng.normal(0, self.gyro_noise_sigma)
        gyro_x = self.rng.normal(0, self.gyro_noise_sigma)
        gyro_y = self.rng.normal(0, self.gyro_noise_sigma)
        return gyro_x, gyro_y, gyro_z

    def _follower_ids(self) -> list:
        return [i for i in range(self.config.num_drone) if i != self.config.leader_id]

    # ---- per-drone generation (pass 1) ---------------------------------

    def step_leader(self, t: int, prev_state: Optional[DroneState]) -> DroneState:
        # Advance the leader's exogenous mission trajectory by one timestep.
        if prev_state is None:
            pos_x, pos_y = self.config.leader_start_pos
        else:
            pos_x = prev_state.pos_x + self.rng.normal(0, self.leader_step_sigma)
            pos_y = prev_state.pos_y + self.rng.normal(0, self.leader_step_sigma)

        roots = self._generate_exogenous_roots(self.config.leader_id)
        gyro_x, gyro_y, gyro_z = self._gyro_from_heading(prev_state, roots["mag_heading"])

        # Leader has no formation to track (formation_error = 0 by definition), so accel/current_draw don't follow the same
        # correction-effort chain followers use below.
        accel_x = self.rng.normal(0, self.accel_noise_sigma)
        accel_y = self.rng.normal(0, self.accel_noise_sigma)
        accel_z = self.rng.normal(0, self.accel_noise_sigma)
        current_draw = max(0.0, self.current_baseline + self.rng.normal(0, self.current_noise_sigma))
        motor_rpm = max(0.0, self.motor_rpm_gain * current_draw + self.rng.normal(0, self.motor_rpm_noise_sigma))
        prev_voltage = prev_state.battery_voltage if prev_state is not None else self.battery_start
        battery_voltage = (prev_voltage - self.discharge_rate * current_draw
                            + self.rng.normal(0, self.battery_noise_sigma))

        return DroneState(
            drone_id=self.config.leader_id,
            timestep=t,
            pos_x=pos_x, pos_y=pos_y,
            altitude=roots["altitude"],
            satellite_count=roots["satellite_count"],
            accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            mag_heading=roots["mag_heading"],
            packet_rate=self.packet_rate_baseline,
            signal_strength=self.signal_baseline,
            latency=self.latency_baseline,
            link_active=True,
            inter_drone_distance=0.0,   # leader excluded from neighbor_pairs
            relative_velocity=0.0,
            formation_error=0.0,        # leader defines the formation frame
            battery_voltage=battery_voltage,
            current_draw=current_draw,
            motor_rpm=motor_rpm,
        )

    def step_follower(self, t: int, drone_id: int, prev_state: Optional[DroneState],
                       leader_state: DroneState) -> DroneState:
        # Advance one follower by one timestep: position tracking, formation error, and everything that only depends on this drone's own history
        # plus the leader (not other followers, that's pass 2 below).
        noise_state = self.noise_states[drone_id]
        roots = self._generate_exogenous_roots(drone_id)

        # Fewer satellites => noisier position tracking
        sat_ratio = self.satellite_baseline / max(roots["satellite_count"], 1.0)
        pos_sigma = self.pos_noise_sigma * sat_ratio
        noise_state.pos_noise_x = self._ar1_step(noise_state.pos_noise_x, rho=0.97, sigma=pos_sigma)
        noise_state.pos_noise_y = self._ar1_step(noise_state.pos_noise_y, rho=0.97, sigma=pos_sigma)

        offset_x, offset_y = self.config.formation_offsets[drone_id]
        pos_x = leader_state.pos_x + offset_x + noise_state.pos_noise_x
        pos_y = leader_state.pos_y + offset_y + noise_state.pos_noise_y

        # formation_error is just the magnitude of the deviation we already generated
        formation_error = float(np.hypot(noise_state.pos_noise_x, noise_state.pos_noise_y))

        gyro_x, gyro_y, gyro_z = self._gyro_from_heading(prev_state, roots["mag_heading"])

        relative_velocity = max(0.0, self.relvel_gain * formation_error
                                 + self.rng.normal(0, self.relvel_noise_sigma))

        # Correction acceleration points from the current deviation back toward the ideal slot
        accel_mag = self.accel_gain * formation_error + self.rng.normal(0, self.accel_noise_sigma)
        angle = np.arctan2(-noise_state.pos_noise_y, -noise_state.pos_noise_x)
        accel_x = accel_mag * np.cos(angle)
        accel_y = accel_mag * np.sin(angle)
        accel_z = self.rng.normal(0, self.accel_noise_sigma)

        current_draw = max(0.0, self.current_baseline
                            + self.current_accel_gain * abs(accel_mag)
                            + self.current_formation_gain * formation_error
                            + self.rng.normal(0, self.current_noise_sigma))
        motor_rpm = max(0.0, self.motor_rpm_gain * current_draw + self.rng.normal(0, self.motor_rpm_noise_sigma))
        prev_voltage = prev_state.battery_voltage if prev_state is not None else self.battery_start
        battery_voltage = (prev_voltage - self.discharge_rate * current_draw
                            + self.rng.normal(0, self.battery_noise_sigma))

        return DroneState(
            drone_id=drone_id,
            timestep=t,
            pos_x=pos_x, pos_y=pos_y,
            altitude=roots["altitude"],
            satellite_count=roots["satellite_count"],
            accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            mag_heading=roots["mag_heading"],
            packet_rate=0.0,           
            signal_strength=0.0,       
            latency=0.0,
            link_active=True,
            inter_drone_distance=0.0,
            relative_velocity=relative_velocity,
            formation_error=formation_error,
            battery_voltage=battery_voltage,
            current_draw=current_draw,
            motor_rpm=motor_rpm,
        )

    # ---- cross-drone generation (pass 2) -------------------------------

    def _compute_relational_fields(self, states: dict) -> None:
        # Mutates inter_drone_distance/signal_strength/packet_rate/latency/link_active in place
        for drone_id, neighbor_id in self.config.neighbor_pairs.items():
            drone = states[drone_id]
            neighbor = states[neighbor_id]

            distance = float(np.hypot(drone.pos_x - neighbor.pos_x, drone.pos_y - neighbor.pos_y))
            drone.inter_drone_distance = distance

            signal_strength = (self.signal_baseline - self.signal_decay_per_meter * distance
                                + self.rng.normal(0, self.signal_noise_sigma))
            drone.signal_strength = signal_strength

            drone.packet_rate = max(0.0, self.packet_rate_baseline
                                     + self.packet_rate_gain * signal_strength
                                     + self.rng.normal(0, self.packet_rate_noise_sigma))
            drone.latency = max(0.0, self.latency_baseline
                                 - self.latency_gain * signal_strength
                                 + self.rng.normal(0, self.latency_noise_sigma))

            # eq. 7 logistic form, the one binary variable in the schema
            logit = self.link_active_alpha + self.link_active_beta * signal_strength
            p_active = 1.0 / (1.0 + np.exp(-logit))
            drone.link_active = bool(self.rng.random() < p_active)

    # ---- orchestration --------------------------------------------------

    def step_timestep(self, t: int, prev_states: dict) -> dict:
        # Advance every drone by one timestep
        leader_prev = prev_states.get(self.config.leader_id)
        leader_state = self.step_leader(t, leader_prev)

        new_states = {self.config.leader_id: leader_state}
        for drone_id in self._follower_ids():
            follower_prev = prev_states.get(drone_id)
            new_states[drone_id] = self.step_follower(t, drone_id, follower_prev, leader_state)

        self._compute_relational_fields(new_states)
        return new_states

    def run(self) -> pd.DataFrame:
        # Run the full simulation:
        prev_states: dict = {}
        rows = []
        for t in range(self.config.num_timesteps):
            prev_states = self.step_timestep(t, prev_states)
            rows.extend(asdict(state) for state in prev_states.values())
        return pd.DataFrame(rows)
