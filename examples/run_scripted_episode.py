"""Run a scripted 2v1 episode without opening a window."""

from agents import PassOrAdvancePolicy, PolicyRunner, PressBallPolicy
from scenarios import two_vs_one
from sim import ScenarioStatus


def main() -> None:
    world = two_vs_one()
    runner = PolicyRunner(
        world,
        {1: PassOrAdvancePolicy(), 2: PassOrAdvancePolicy(), 3: PressBallPolicy()},
    )
    world.reset(seed=11)
    while world.state.status is ScenarioStatus.RUNNING:
        world.step(runner(world.state))
    seconds = world.state.tick / world.config.ticks_per_second
    print(f"status={world.state.status.value} ticks={world.state.tick} time={seconds:.1f}s")


if __name__ == "__main__":
    main()

