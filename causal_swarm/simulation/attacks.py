"""Attack definitions for the swarm simulator.

Each attack says which drone is hit, when, and how strongly. The helper
methods say what the attack does at a given timestep, so the simulator only
asks "what is the effect right now?" and holds no attack logic itself. The
helpers use no random numbers, so adding an attack never shifts the
simulator's random sequence, and untouched parts of a run stay identical to
the benign run with the same seed.
"""

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Attack:
    drone_id: int
    start: int   # first attacked timestep
    end: int     # first timestep after the attack (end is exclusive)
    attack_type: ClassVar[str] = "none"

    def is_active(self, t: int) -> bool:
        return self.start <= t < self.end


@dataclass
class GPSSpoof(Attack):
    """Pushes a drone off its formation slot by an offset that grows each timestep."""
    drift_per_step: float = 0.1            # meters per step
    direction: tuple = (1.0, 0.0)          # any nonzero (x, y); normalized when used
    attack_type: ClassVar[str] = "gps_spoof"

    def offset(self, t: int) -> tuple:
        """(x, y) offset in meters at timestep t; (0, 0) outside the attack window."""
        if not self.is_active(t):
            return (0.0, 0.0)
        distance = self.drift_per_step * (t - self.start)
        norm = math.hypot(*self.direction)
        return (distance * self.direction[0] / norm, distance * self.direction[1] / norm)


@dataclass
class DoSFlood(Attack):
    """Floods a drone's comms link: fewer packets, higher latency, link often down."""
    packet_loss: float = 0.9               # fraction of packets lost
    added_latency: float = 200.0           # ms
    link_drop: float = 0.9                 # fraction of the link's up-probability removed
    attack_type: ClassVar[str] = "dos"

    def degrade(self, packet_rate: float, latency: float) -> tuple:
        return packet_rate * (1.0 - self.packet_loss), latency + self.added_latency

    def link_probability(self, p_active: float) -> float:
        """Reduced probability that the link is up, for the simulator's existing random draw."""
        return p_active * (1.0 - self.link_drop)


def active_attack(attacks: list, drone_id: int, t: int, kind: type = Attack):
    """The first attack of the given kind hitting this drone at timestep t, or None."""
    for attack in attacks:
        if isinstance(attack, kind) and attack.drone_id == drone_id and attack.is_active(t):
            return attack
    return None


def attack_label(attacks: list, drone_id: int, t: int) -> str:
    """Ground-truth label for one drone at one timestep: an attack_type, or "none"."""
    attack = active_attack(attacks, drone_id, t)
    return attack.attack_type if attack else "none"
