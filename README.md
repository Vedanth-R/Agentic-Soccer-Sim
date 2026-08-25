# PitchLab

PitchLab is a deliberately small, deterministic 2D soccer simulation. This
first milestone contains only the headless world; agents, training, and the
Pygame viewer will be added as separate layers.

## Run it

```bash
source .venv/bin/activate
python -m pytest
python -m examples.scripted_pass
```

The world runs at 10 ticks per simulated second. Agents submit intentions
(`MoveAction` or `PassAction`), and `SoccerWorld.step()` applies the rules.

## Layout

- `sim/`: world state, actions, and deterministic simulation rules
- `examples/`: small runnable scripts
- `tests/`: behavior and determinism tests
- `agents/`, `scenarios/`, `training/`, `viewer/`, `metrics/`: placeholders for later milestones
