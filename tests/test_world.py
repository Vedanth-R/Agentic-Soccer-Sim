from copy import deepcopy

import pytest

from sim import MoveAction, PassAction, ScenarioStatus, SoccerWorld, WorldConfig


def snapshot(world: SoccerWorld):
    state = world.state
    return (
        state.tick,
        state.status,
        tuple(
            (p.player_id, p.position, p.velocity, p.has_ball)
            for p in state.players.values()
        ),
        state.ball.position,
        state.ball.velocity,
        state.ball.owner_id,
    )


def test_player_speed_is_capped_and_ball_follows_owner():
    world = SoccerWorld()
    world.reset(seed=1)

    state = world.step({1: MoveAction((100.0, 0.0))})

    assert state.players[1].position == pytest.approx((20.7, 34.0))
    assert state.ball.position == state.players[1].position
    assert state.ball.owner_id == 1


def test_pass_becomes_a_loose_moving_ball():
    world = SoccerWorld()
    world.reset(seed=1)

    state = world.step({1: PassAction((40.0, 34.0))})

    assert state.players[1].has_ball is False
    assert state.ball.owner_id is None
    assert state.ball.position == pytest.approx((21.8, 34.0))
    assert state.ball.velocity == pytest.approx((17.46, 0.0))


def test_nearby_player_intercepts_a_pass():
    world = SoccerWorld()
    world.reset(seed=1)
    world.state.players[2].position = (21.8, 34.0)

    state = world.step({1: PassAction((40.0, 34.0))})

    assert state.ball.owner_id == 2
    assert state.players[2].has_ball is True
    assert state.ball.velocity == (0.0, 0.0)


def test_progression_line_finishes_scenario():
    world = SoccerWorld(WorldConfig(progression_x=21.0))
    world.reset(seed=1)

    state = world.step({1: MoveAction((1.0, 0.0))})
    assert state.status is ScenarioStatus.RUNNING
    state = world.step({1: MoveAction((1.0, 0.0))})
    assert state.status is ScenarioStatus.PROGRESSED


def test_same_seed_and_actions_produce_identical_state():
    first = SoccerWorld()
    second = SoccerWorld()
    first.reset(seed=42)
    second.reset(seed=42)
    actions = [
        {1: MoveAction((1.0, 0.25)), 2: MoveAction((-1.0, 0.0))},
        {1: PassAction((60.0, 30.0)), 2: MoveAction((-1.0, 0.0))},
        {1: MoveAction((0.0, 1.0)), 2: MoveAction((-1.0, 0.0))},
    ]

    for tick_actions in actions:
        first.step(deepcopy(tick_actions))
        second.step(deepcopy(tick_actions))

    assert snapshot(first) == snapshot(second)

