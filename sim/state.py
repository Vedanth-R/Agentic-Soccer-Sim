"""Plain state objects shared by the simulator, future agents, and viewer."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

Vector = Tuple[float, float]


class ScenarioStatus(str, Enum):
    RUNNING = "running"
    GOAL = "goal"
    PROGRESSED = "progressed"
    TIMEOUT = "timeout"


@dataclass
class PlayerState:
    player_id: int
    team: int
    position: Vector
    velocity: Vector = (0.0, 0.0)
    has_ball: bool = False


@dataclass
class BallState:
    position: Vector
    velocity: Vector = (0.0, 0.0)
    owner_id: Optional[int] = None


@dataclass
class WorldState:
    tick: int
    players: Dict[int, PlayerState]
    ball: BallState
    status: ScenarioStatus = ScenarioStatus.RUNNING

