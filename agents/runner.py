"""Collect simultaneous actions from independently acting policies."""

from typing import Dict, Mapping

from sim import Action, SoccerWorld, WorldState

from .observation import build_observation
from .policies import Policy


class PolicyRunner:
    def __init__(self, world: SoccerWorld, policies: Mapping[int, Policy]) -> None:
        self.world = world
        self.policies = dict(policies)

    def actions_for(self, state: WorldState) -> Dict[int, Action]:
        """Ask every policy using the same pre-step state snapshot."""
        return {
            player_id: policy.act(
                build_observation(
                    state,
                    player_id,
                    self.world.config.pitch_length,
                    self.world.config.pitch_width,
                )
            )
            for player_id, policy in sorted(self.policies.items())
        }

    def __call__(self, state: WorldState) -> Dict[int, Action]:
        return self.actions_for(state)

