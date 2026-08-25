# PitchLab

I made this project to learn more about swarm agents and ML training in general. Project is not yet finished.

## Run it

```bash
source .venv/bin/activate
python -m pytest
python -m examples.scripted_pass
```

The world runs at 10 ticks per simulated second. Agents submit intentions
(`MoveAction` or `PassAction`), and `SoccerWorld.step()` applies the rules. This is not yet finished though.

## Layout

- `sim/`: world state, actions, and deterministic simulation rules
- `examples/`: small runnable scripts
- `tests/`: behavior and determinism tests
- `agents/`, `scenarios/`, `training/`, `viewer/`, `metrics/`: placeholders for later additions. 
