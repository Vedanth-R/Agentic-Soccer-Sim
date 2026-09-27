# PitchLab: Soccer Swarm

PitchLab trains soccer agents that share one PyTorch neural network. The
current checkpoint is trained in 3v2, then evaluated without further training
in 5v4 and 6v4 to measure zero-shot transfer.

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

The simulator, scripted defenders, editor, JSON scenarios, PPO data collection,
and trained model support two to six attackers and one to six defenders. Each
agent always receives 48 inputs: its own situation plus five teammate slots
and six defender slots. Players are ordered by distance; missing slots contain
zeros and an existence mask. Observations also include ball velocity, whether
the ball is loose, whether the agent is the intended receiver, distance from
the ball path and nearest defender, and whether an open pass exists. This fixed
format lets the same feed-forward network run different roster sizes without
changing PPO or the network layers.

The policy has 13 discrete actions: seven movement/hold actions, shoot, and
five pass-target actions corresponding to the five teammate slots. Action
masking prevents off-ball shooting or passing, early shooting, and passes to
empty teammate slots.

The checkpoint has only trained in 3v2. Larger layouts and custom positions are
therefore tests of generalization, not situations it has already learned.

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
- Zero-shot performance in 5v4 and 6v4

For each strategy it measures goals, outcome types, pass attempts, completion
rate, forward passes, pass-caused turnovers, goals after a pass, and receiver
movement toward the ball.

Current checkpoint results:

| Strategy | Goals | Turnovers | Completed passes per episode |
|---|---:|---:|---:|
| Learned policy, 3v2 | 88.5% | 1.5% | 0.77 |
| Off-ball players frozen, 3v2 | 96.5% | 3.5% | 0.96 |
| Direct run and shoot | 35.5% | 64.5% | 0.00 |
| Learned policy, 5v4 zero-shot | 39.0% | 25.0% | 7.46 |
| Direct run and shoot, 5v4 | 12.5% | 87.5% | 0.00 |
| Learned policy, 6v4 zero-shot | 30.0% | 32.0% | 8.41 |
| Direct run and shoot, 6v4 | 13.0% | 87.0% | 0.00 |

The learned policy completes 86.9% of attempted passes in 3v2, and 59.5% of
episodes end with a goal after a pass. Larger-roster results are zero-shot: the
checkpoint never trained in 5v4 or 6v4.

The frozen-player ablation exposes the next weakness. Stationary off-ball
players currently score more often than the full policy, so passing has been
learned but active off-ball movement is not consistently useful yet.


## Train

```bash
python swarm_3v2.py --train
```

Training uses a five-stage curriculum:

1. Learn passing and receiving without pressure.
2. Receive passes against a slow defender.
3. Learn to approach and shoot near goal with stationary defenders.
4. Start farther from goal against moderately fast defenders.
5. Train on the complete 3v2 with broader layouts and stronger defenders.

The final stage uses 700,000 simulation steps by default. Change only that
stage's budget with, for example:

```bash
python swarm_3v2.py --train --steps 1000000
```

Each curriculum stage is tested on held-out full 3v2 layouts. The best stage is
kept so a later stage cannot replace it after a training collapse. The
resulting model is saved to `artifacts/goal_swarm_3v2.pt`.

## Rewards

All three attackers receive the same team reward:

- `+20` for scoring
- `-10` for losing possession, putting the ball out, or timing out
- `+0.6` to `+1.0` for each of the first three completed forward passes
- Small changes for forward ball progress and creating safe passing options
- A small time cost and penalty for trying to shoot too early
