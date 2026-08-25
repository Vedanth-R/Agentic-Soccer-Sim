"""Actions describe intent; only the simulator mutates world state."""

from dataclasses import dataclass
from typing import Tuple, Union

Vector = Tuple[float, float]


@dataclass(frozen=True)
class MoveAction:
    """Move in a direction. Magnitude is capped at one by the world."""

    direction: Vector


@dataclass(frozen=True)
class PassAction:
    """Pass toward a point in pitch coordinates."""

    target: Vector


Action = Union[MoveAction, PassAction]

