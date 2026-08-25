"""Run a tiny headless sequence: move, then pass toward the opponent."""

from sim import MoveAction, PassAction, SoccerWorld


def main() -> None:
    world = SoccerWorld()
    state = world.reset(seed=7)
    print(f"tick={state.tick:02d} ball={state.ball.position} owner={state.ball.owner_id}")

    for _ in range(5):
        state = world.step({1: MoveAction((1.0, 0.0))})
        print(f"tick={state.tick:02d} ball={state.ball.position} owner={state.ball.owner_id}")

    state = world.step({1: PassAction(state.players[2].position)})
    print(f"tick={state.tick:02d} ball={state.ball.position} owner={state.ball.owner_id}")


if __name__ == "__main__":
    main()

