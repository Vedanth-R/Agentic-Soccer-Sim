# PitchLab

I made this project to learn more about swarm agents and ML training in general. Project is not yet finished.

## Run it

```bash
source .venv/bin/activate
python -m pytest
python -m examples.scripted_pass
python -m examples.watch_world
python -m examples.run_scripted_episode
python -m examples.watch_scripted_agents
```

The world runs at 10 ticks per simulated second. Agents submit intentions
(`MoveAction` or `PassAction`), and `SoccerWorld.step()` applies the rules. This is not yet finished though.

## Layout

- `sim/`: world state, actions, and deterministic simulation rules
- `agents/`: policy interface, observations, scripted baselines, and policy runner
- `scenarios/`: reusable 1v1 and 2v1 starting setups
- `examples/`: small runnable scripts
- `tests/`: behavior and determinism tests
- `viewer/`: read-only Pygame rendering and playback controls
- `training/`, `metrics/`: placeholders for later additions

## Viewer controls

- `Space`: pause or resume
- `R`: reset with the same seed
- `-` / `+`: slow down or speed up playback
- `Esc`: close the viewer
