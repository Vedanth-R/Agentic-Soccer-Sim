# PitchLab: Soccer Swarm

PitchLab trains three attacking soccer agents that share one PyTorch neural
network. They play against two scripted defenders and must score in the goal.

The nearest defender presses the ball while the second defender marks a
forward attacker. The attackers make separate decisions because each receives
its own view of the players and ball.


## Run it

```bash
source .venv/bin/activate
python pygame_visualizer.py
```

Viewer controls:

- `Space`: pause or resume
- `R`: replay the same layout
- `N`: use the next seeded layout
- `E`: edit the current starting setup
- `-` / `+`: change playback speed
- `Esc`: quit

In edit mode:

- Click and drag any attacker or defender.
- Press `A` to add an attacker at the mouse position.
- Press `D` to add a defender at the mouse position.
- Select a player and press `Delete` or `Backspace` to remove it.
- Select an attacker and press `B` to give it the ball.
- Press `S` to save the layout to `scenarios/custom.json`.
- Press `L` to load that saved layout.
- Press `Enter` to save that setup and run the model.
- Press `R` afterward to replay the exact custom setup.

You can also open the editor with a generated roster:

```bash
python pygame_visualizer.py --attackers 5 --defenders 4
python pygame_visualizer.py --attackers 6 --defenders 4
```

The simulator, scripted defenders, editor, JSON scenarios, and PPO data
collection now support different team sizes. The current trained checkpoint
still expects the 16 inputs produced by a 3v2 layout. Larger layouts therefore
remain paused in the editor with a compatibility warning. A variable-roster
observation model is the next phase; until then the UI can create and save its
test scenarios without claiming the 3v2 model can run them.

Custom 3v2 layouts can run immediately and test positions the model did not
encounter during training. Poor performance is useful evidence about the
limits of its generalization.

```bash
python pygame_visualizer.py --seed 20025
python pygame_visualizer.py --jitter 12
```

The trained model used the default 8-metre vertical variation. Larger jitter
values test layouts outside its normal training distribution.

## Evaluate

```bash
python swarm_3v2.py
```

Evaluation uses 200 held-out layouts and reports:

- The complete learned policy
- The same policy with both off-ball attackers forced to stand still
- A direct run-and-shoot strategy

Current checkpoint results:

| Strategy | Goals | Turnovers | Completed passes per episode |
|---|---:|---:|---:|
| Learned policy | 33.0% | 66.5% | 0.00 |
| Off-ball players frozen | 31.5% | 68.0% | 0.00 |
| Direct run and shoot | 35.5% | 64.5% | 0.00 |


## Train

```bash
python swarm_3v2.py --train
```

Training uses a three-stage curriculum:

1. Learn to approach and shoot near goal with stationary defenders.
2. Start farther from goal against moderately fast defenders.
3. Train on the complete 3v2 with broader layouts and stronger defenders.

The final stage uses 700,000 simulation steps by default. Change only that
stage's budget with, for example:

```bash
python swarm_3v2.py --train --steps 1000000
```

The resulting model is saved to `artifacts/goal_swarm_3v2.pt`.

## Rewards

All three attackers receive the same team reward:

- `+20` for scoring
- `-10` for losing possession, putting the ball out, or timing out
- `+0.6` to `+1.0` for each of the first three completed forward passes
- Small changes for forward ball progress and creating safe passing options
- A small time cost and penalty for trying to shoot too early
