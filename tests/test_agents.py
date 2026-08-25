from agents import (
    AttackGoalPolicy,
    HoldPositionPolicy,
    PassOrAdvancePolicy,
    PolicyRunner,
    PressBallPolicy,
    build_observation,
)
from scenarios import one_vs_one, two_vs_one
from sim import MoveAction, PassAction, ScenarioStatus


def observation(world, player_id):
    return build_observation(
        world.state,
        player_id,
        world.config.pitch_length,
        world.config.pitch_width,
    )


def test_press_policy_moves_toward_ball():
    world = one_vs_one()
    world.reset(seed=3)
    action = PressBallPolicy().act(observation(world, 2))
    assert isinstance(action, MoveAction)
    assert action.direction[0] < 0.0
    assert action.direction[1] < 0.0


def test_attacker_shoots_when_close_to_goal():
    world = one_vs_one()
    world.reset(seed=3)
    world.state.players[1].position = (90.0, 34.0)
    world.state.ball.position = (90.0, 34.0)
    action = AttackGoalPolicy().act(observation(world, 1))
    assert action == PassAction((105.0, 34.0))


def test_pass_policy_uses_forward_teammate_when_pressed():
    world = two_vs_one()
    world.reset(seed=3)
    world.state.players[3].position = (25.0, 42.0)
    action = PassOrAdvancePolicy().act(observation(world, 1))
    assert action == PassAction(world.state.players[2].position)


def test_hold_policy_stops_at_target():
    world = one_vs_one()
    world.reset(seed=3)
    target = world.state.players[2].position
    assert HoldPositionPolicy(target).act(observation(world, 2)) == MoveAction((0.0, 0.0))


def test_runner_returns_one_action_per_policy():
    world = one_vs_one()
    world.reset(seed=3)
    runner = PolicyRunner(world, {1: AttackGoalPolicy(), 2: PressBallPolicy()})
    assert set(runner.actions_for(world.state)) == {1, 2}


def test_scripted_two_vs_one_episode_finishes():
    world = two_vs_one()
    runner = PolicyRunner(
        world,
        {1: PassOrAdvancePolicy(), 2: PassOrAdvancePolicy(), 3: PressBallPolicy()},
    )
    world.reset(seed=11)
    while world.state.status is ScenarioStatus.RUNNING:
        world.step(runner(world.state))
    assert world.state.status is ScenarioStatus.GOAL
    assert world.state.tick <= world.config.max_ticks
