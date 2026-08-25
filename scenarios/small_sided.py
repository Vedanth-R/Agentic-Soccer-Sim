"""Initial state factories for the first scripted-agent experiments."""

from sim import BallState, PlayerState, SoccerWorld, WorldConfig, WorldState


def one_vs_one() -> SoccerWorld:
    """One attacker in possession against one defender."""

    def initial_state(rng, config):
        center_y = config.pitch_width / 2.0
        players = {
            1: PlayerState(1, team=0, position=(22.0, center_y), has_ball=True),
            2: PlayerState(2, team=1, position=(62.0, center_y + 8.0)),
        }
        return WorldState(0, players, BallState(players[1].position, owner_id=1))

    return SoccerWorld(WorldConfig(max_ticks=500), initial_state_factory=initial_state)


def two_vs_one() -> SoccerWorld:
    """Two attackers in separate lanes against one defender."""

    def initial_state(rng, config):
        center_y = config.pitch_width / 2.0
        players = {
            1: PlayerState(1, team=0, position=(22.0, center_y + 8.0), has_ball=True),
            2: PlayerState(2, team=0, position=(34.0, center_y - 10.0)),
            3: PlayerState(3, team=1, position=(60.0, center_y + 3.0)),
        }
        return WorldState(0, players, BallState(players[1].position, owner_id=1))

    return SoccerWorld(WorldConfig(max_ticks=500), initial_state_factory=initial_state)

