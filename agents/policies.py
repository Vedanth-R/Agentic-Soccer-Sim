"""Simple hand-written policies used to validate the simulator."""

from math import hypot
from typing import Protocol, Tuple

import numpy as np

from sim import Action, MoveAction, PassAction

from .observation import ObservedPlayer, PlayerObservation

Vector = Tuple[float, float]


class Policy(Protocol):
    """Anything that turns one player's observation into an intention."""

    def act(self, observation: PlayerObservation) -> Action:
        ...


class RandomPolicy:
    """Reproducible baseline that samples movement and occasional passes."""

    DIRECTIONS = (
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    )

    def __init__(self, seed: int = 0, pass_probability: float = 0.1) -> None:
        self._rng = np.random.default_rng(seed)
        self.pass_probability = pass_probability

    def act(self, observation: PlayerObservation) -> Action:
        if observation.self_player.has_ball and self._rng.random() < self.pass_probability:
            target = (
                float(self._rng.uniform(0.0, observation.attacking_goal[0] or 105.0)),
                float(self._rng.uniform(0.0, observation.attacking_goal[1] * 2.0)),
            )
            return PassAction(target)
        index = int(self._rng.integers(0, len(self.DIRECTIONS)))
        return MoveAction(self.DIRECTIONS[index])


class PressBallPolicy:
    """Move directly toward the current ball position."""

    def act(self, observation: PlayerObservation) -> Action:
        return MoveAction(_toward(observation.self_player.position, observation.ball_position))


class AttackGoalPolicy:
    """Dribble toward goal and shoot once close enough."""

    def __init__(self, shoot_distance: float = 24.0) -> None:
        self.shoot_distance = shoot_distance

    def act(self, observation: PlayerObservation) -> Action:
        player = observation.self_player
        if player.has_ball and _distance(player.position, observation.attacking_goal) <= self.shoot_distance:
            return PassAction(observation.attacking_goal)
        if player.has_ball:
            return MoveAction(_toward(player.position, observation.attacking_goal))
        return MoveAction(_toward(player.position, observation.ball_position))


class PassOrAdvancePolicy:
    """Advance, pass to a forward teammate under pressure, or shoot."""

    def __init__(self, pressure_distance: float = 9.0, shoot_distance: float = 24.0) -> None:
        self.pressure_distance = pressure_distance
        self.shoot_distance = shoot_distance

    def act(self, observation: PlayerObservation) -> Action:
        player = observation.self_player
        if not player.has_ball:
            if observation.ball_owner_id is None:
                return MoveAction(_toward(player.position, observation.ball_position))
            return MoveAction(_toward(player.position, self._support_target(observation)))
        if _distance(player.position, observation.attacking_goal) <= self.shoot_distance:
            return PassAction(observation.attacking_goal)
        nearest_opponent = min(
            (_distance(player.position, opponent.position) for opponent in observation.opponents),
            default=float("inf"),
        )
        forward_teammates = [
            teammate
            for teammate in observation.teammates
            if _is_ahead(player, teammate, observation.attacking_goal)
        ]
        if nearest_opponent <= self.pressure_distance and forward_teammates:
            teammate = min(
                forward_teammates,
                key=lambda item: _distance(player.position, item.position),
            )
            return PassAction(teammate.position)
        return MoveAction(_toward(player.position, observation.attacking_goal))

    @staticmethod
    def _support_target(observation: PlayerObservation) -> Vector:
        x_direction = 1.0 if observation.attacking_goal[0] > observation.self_player.position[0] else -1.0
        lane_y = observation.self_player.position[1]
        return (observation.ball_position[0] + 10.0 * x_direction, lane_y)


class HoldPositionPolicy:
    """Return toward a fixed tactical position."""

    def __init__(self, position: Vector, tolerance: float = 0.5) -> None:
        self.position = position
        self.tolerance = tolerance

    def act(self, observation: PlayerObservation) -> Action:
        if _distance(observation.self_player.position, self.position) <= self.tolerance:
            return MoveAction((0.0, 0.0))
        return MoveAction(_toward(observation.self_player.position, self.position))


def _toward(origin: Vector, target: Vector) -> Vector:
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    length = hypot(dx, dy)
    return (0.0, 0.0) if length == 0.0 else (dx / length, dy / length)


def _distance(a: Vector, b: Vector) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _is_ahead(player: ObservedPlayer, teammate: ObservedPlayer, goal: Vector) -> bool:
    if goal[0] > player.position[0]:
        return teammate.position[0] > player.position[0]
    return teammate.position[0] < player.position[0]
