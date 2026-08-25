"""Watch a deterministic scripted 1v1 sequence in the Pygame viewer."""

from typing import Dict

from sim import Action, MoveAction, PassAction, SoccerWorld, WorldState
from viewer import PygameViewer


def scripted_actions(state: WorldState) -> Dict[int, Action]:
    """Move both players, then send a pass through the defender's path."""
    if state.tick < 25:
        return {
            1: MoveAction((1.0, -0.2)),
            2: MoveAction((-0.5, -0.1)),
        }
    if state.tick == 25 and state.ball.owner_id == 1:
        return {
            1: PassAction((95.0, 24.0)),
            2: MoveAction((-0.5, -0.1)),
        }
    return {
        1: MoveAction((1.0, 0.0)),
        2: MoveAction((-1.0, 0.0)),
    }


def main() -> None:
    PygameViewer(SoccerWorld(), scripted_actions, seed=7).run()


if __name__ == "__main__":
    main()

