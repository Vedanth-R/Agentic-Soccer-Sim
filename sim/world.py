"""Deterministic fixed-timestep rules for the 2D soccer world."""

from dataclasses import dataclass
from math import hypot
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .actions import Action, MoveAction, PassAction
from .state import BallState, PlayerState, ScenarioStatus, WorldState

Vector = Tuple[float, float]
InitialStateFactory = Callable[[np.random.Generator, "WorldConfig"], WorldState]


@dataclass(frozen=True)
class WorldConfig:
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    ticks_per_second: int = 10
    player_max_speed: float = 7.0
    pass_speed: float = 18.0
    ball_friction: float = 0.97
    control_radius: float = 1.0
    goal_width: float = 7.32
    max_ticks: int = 600
    progression_x: Optional[float] = None

    @property
    def dt(self) -> float:
        return 1.0 / self.ticks_per_second


class SoccerWorld:
    """Owns state and applies all movement and ball rules."""

    def __init__(
        self,
        config: Optional[WorldConfig] = None,
        initial_state_factory: Optional[InitialStateFactory] = None,
    ) -> None:
        self.config = config or WorldConfig()
        self._initial_state_factory = initial_state_factory
        self._rng = np.random.default_rng()
        self.state = self._initial_state()

    def reset(self, seed: Optional[int] = None) -> WorldState:
        """Restore the initial 1v1 setup and seed future randomness."""
        self._rng = np.random.default_rng(seed)
        self.state = self._initial_state()
        return self.state

    def step(self, actions: Dict[int, Action]) -> WorldState:
        """Advance exactly one tick, ignoring actions after the episode ends."""
        if self.state.status is not ScenarioStatus.RUNNING:
            return self.state

        self._apply_player_movement(actions)
        self._apply_passes(actions)
        self._advance_ball()
        self._resolve_loose_ball()
        self.state.tick += 1
        self._update_status()
        return self.state

    def _initial_state(self) -> WorldState:
        if self._initial_state_factory is not None:
            return self._initial_state_factory(self._rng, self.config)
        y = self.config.pitch_width / 2.0
        players = {
            1: PlayerState(1, team=0, position=(20.0, y), has_ball=True),
            2: PlayerState(2, team=1, position=(70.0, y)),
        }
        return WorldState(0, players, BallState((20.0, y), owner_id=1))

    def _apply_player_movement(self, actions: Dict[int, Action]) -> None:
        for player_id in sorted(self.state.players):
            player = self.state.players[player_id]
            action = actions.get(player_id)
            direction = action.direction if isinstance(action, MoveAction) else (0.0, 0.0)
            unit = _limit_magnitude(direction, 1.0)
            player.velocity = _scale(unit, self.config.player_max_speed)
            player.position = self._inside_pitch(
                _add(player.position, _scale(player.velocity, self.config.dt))
            )
        if self.state.ball.owner_id is not None:
            self.state.ball.position = self.state.players[self.state.ball.owner_id].position
            self.state.ball.velocity = (0.0, 0.0)

    def _apply_passes(self, actions: Dict[int, Action]) -> None:
        owner_id = self.state.ball.owner_id
        if owner_id is None:
            return
        action = actions.get(owner_id)
        if not isinstance(action, PassAction):
            return
        owner = self.state.players[owner_id]
        direction = _subtract(action.target, self.state.ball.position)
        unit = _limit_magnitude(direction, 1.0)
        if unit == (0.0, 0.0):
            return
        owner.has_ball = False
        self.state.ball.owner_id = None
        self.state.ball.velocity = _scale(unit, self.config.pass_speed)

    def _advance_ball(self) -> None:
        ball = self.state.ball
        if ball.owner_id is not None:
            return
        ball.position = _add(ball.position, _scale(ball.velocity, self.config.dt))
        ball.velocity = _scale(ball.velocity, self.config.ball_friction)
        if hypot(*ball.velocity) < 0.05:
            ball.velocity = (0.0, 0.0)

    def _resolve_loose_ball(self) -> None:
        ball = self.state.ball
        if ball.owner_id is not None:
            return
        candidates = [
            player
            for player in self.state.players.values()
            if _distance(player.position, ball.position) <= self.config.control_radius
        ]
        if not candidates:
            return
        # Distance then id gives an explicit, deterministic tie-break.
        owner = min(candidates, key=lambda p: (_distance(p.position, ball.position), p.player_id))
        owner.has_ball = True
        ball.owner_id = owner.player_id
        ball.position = owner.position
        ball.velocity = (0.0, 0.0)

    def _update_status(self) -> None:
        x, y = self.state.ball.position
        goal_half_width = self.config.goal_width / 2.0
        goal_center = self.config.pitch_width / 2.0
        if x >= self.config.pitch_length and abs(y - goal_center) <= goal_half_width:
            self.state.status = ScenarioStatus.GOAL
        elif self.config.progression_x is not None and x >= self.config.progression_x:
            self.state.status = ScenarioStatus.PROGRESSED
        elif self.state.tick >= self.config.max_ticks:
            self.state.status = ScenarioStatus.TIMEOUT

    def _inside_pitch(self, position: Vector) -> Vector:
        return (
            min(max(position[0], 0.0), self.config.pitch_length),
            min(max(position[1], 0.0), self.config.pitch_width),
        )


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1])


def _subtract(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1])


def _scale(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor)


def _distance(a: Vector, b: Vector) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _limit_magnitude(vector: Vector, maximum: float) -> Vector:
    magnitude = hypot(*vector)
    if magnitude == 0.0:
        return (0.0, 0.0)
    scale = min(magnitude, maximum) / magnitude
    return _scale(vector, scale)
