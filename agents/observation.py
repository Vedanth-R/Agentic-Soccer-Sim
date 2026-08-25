"""Agent-facing snapshots derived from the simulator's world state."""

from dataclasses import dataclass
from typing import Optional, Tuple

from sim.state import Vector, WorldState


@dataclass(frozen=True)
class ObservedPlayer:
    player_id: int
    team: int
    position: Vector
    velocity: Vector
    has_ball: bool


@dataclass(frozen=True)
class PlayerObservation:
    """Perfect-information observation for one player.

    Keeping this separate from ``WorldState`` lets us restrict perception later
    without changing the policy interface.
    """

    tick: int
    self_player: ObservedPlayer
    teammates: Tuple[ObservedPlayer, ...]
    opponents: Tuple[ObservedPlayer, ...]
    ball_position: Vector
    ball_velocity: Vector
    ball_owner_id: Optional[int]
    attacking_goal: Vector


def build_observation(
    state: WorldState,
    player_id: int,
    pitch_length: float,
    pitch_width: float,
) -> PlayerObservation:
    player = state.players[player_id]
    observed = {
        item.player_id: ObservedPlayer(
            item.player_id,
            item.team,
            item.position,
            item.velocity,
            item.has_ball,
        )
        for item in state.players.values()
    }
    teammates = tuple(
        observed[item.player_id]
        for item in sorted(state.players.values(), key=lambda value: value.player_id)
        if item.team == player.team and item.player_id != player_id
    )
    opponents = tuple(
        observed[item.player_id]
        for item in sorted(state.players.values(), key=lambda value: value.player_id)
        if item.team != player.team
    )
    # Team 0 attacks right; team 1 attacks left.
    attacking_goal = (pitch_length, pitch_width / 2.0) if player.team == 0 else (0.0, pitch_width / 2.0)
    return PlayerObservation(
        tick=state.tick,
        self_player=observed[player_id],
        teammates=teammates,
        opponents=opponents,
        ball_position=state.ball.position,
        ball_velocity=state.ball.velocity,
        ball_owner_id=state.ball.owner_id,
        attacking_goal=attacking_goal,
    )

