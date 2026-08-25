"""Watch two scripted attackers play against a pressing defender."""

from agents import PassOrAdvancePolicy, PolicyRunner, PressBallPolicy
from scenarios import two_vs_one
from viewer import PygameViewer


def main() -> None:
    world = two_vs_one()
    runner = PolicyRunner(
        world,
        {
            1: PassOrAdvancePolicy(),
            2: PassOrAdvancePolicy(),
            3: PressBallPolicy(),
        },
    )
    PygameViewer(world, runner, seed=11).run()


if __name__ == "__main__":
    main()

