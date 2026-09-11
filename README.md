# PitchLab: 3v2 Swarm

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
- `-` / `+`: change playback speed
- `Esc`: quit

Choose a repeatable layout or adjust the starting variation:

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

Current results:

| Strategy | Goals | Turnovers | Completed passes per episode |
|---|---:|---:|---:|
| Learned policy | 99.0% | 1.0% | 1.45 |
| Off-ball players frozen | 68.5% | 31.5% | 2.37 |
| Direct run and shoot | 35.5% | 64.5% | 0.00 |

The off-ball test is important: the large performance drop when those players
are frozen shows that their movement contributes to the trained policy.

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
- Up to `+0.4` for each of the first three completed forward passes
- Small changes for forward ball progress and creating safe passing options
- A small time cost and penalty for trying to shoot too early

Scoring is worth much more than the shaping rewards. Kicking the ball beyond
the end line outside the goal is a failure, so the old strategy of booting the
ball toward a progression line no longer works.

## Code

Read the project in this order:

1. `swarm_3v2.py` — world, physics, observations, rewards, training stages, and evaluation
2. `neural_network.py` — shared actor-critic network and PPO training
3. `pygame_visualizer.py` — visual replay of the saved policy

The result measures performance in this small simulation and should not be
treated as evidence about real-world soccer tactics.

