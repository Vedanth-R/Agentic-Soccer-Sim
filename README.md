# PitchLab: Soccer Swarm

PitchLab trains soccer agents with Multi-Agent Proximal Policy Optimization
(MAPPO). The current checkpoint is trained in 3v2, then evaluated without
further training in 5v4 and 6v4 to measure zero-shot transfer.

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

Watch the two passing scenarios used during mixed training:

```bash
python pygame_visualizer.py --mode easy-pass
python pygame_visualizer.py --mode pressured-pass
```

The easy drill places the defenders away from the passing lane. The pressured
drill adds a slow pressing defender. A drill ends successfully when the agents
complete a forward pass. Press `N` for another seeded layout or `E` to edit the
drill positions before running it.

The simulator, scripted defenders, editor, JSON scenarios, PPO data collection,
and trained model support two to six attackers and one to six defenders. Each
agent always receives 51 inputs: its own situation plus five teammate slots
and six defender slots. Players are ordered by distance; missing slots contain
zeros and an existence mask. Observations also include ball velocity, whether
the ball is loose, whether the agent is the intended receiver, distance from
the ball path and nearest defender, and whether an open pass exists. This fixed
format also includes a three-value training-scenario indicator and lets the
same feed-forward network run different roster sizes.

The policy has 13 discrete actions: seven movement/hold actions, shoot, and
five pass-target actions corresponding to the five teammate slots. Action
masking prevents off-ball shooting or passing, early shooting, and passes to
empty teammate slots.

The checkpoint has only trained in 3v2. Larger layouts and custom positions are
therefore tests of generalization, not situations it has already learned.

During training, every agent's shared actor sees only its own 51-value local
observation. A separate centralized critic receives the complete padded world
state plus that agent's local observation (132 values total). The Pygame
simulation uses only the decentralized actor. The earlier shared-PPO model is
preserved at `artifacts/shared_ppo_baseline.pt`, and the earlier sequential
MAPPO model is preserved at `artifacts/mappo_sequential.pt`.

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
- The intended receiver frozen
- Non-receiving support players frozen
- Random off-ball movement
- A direct run-and-shoot strategy
- Zero-shot performance in 5v4 and 6v4

For each strategy it measures goals, outcome types, pass attempts, completion
rate, forward passes, pass-caused turnovers, goals after a pass, and receiver
movement toward the ball.

Current MAPPO checkpoint results:

| Strategy | Goals | Turnovers | Completed passes per episode |
|---|---:|---:|---:|
| Learned policy, 3v2 | 97.0% | 1.5% | 0.74 |
| Off-ball players frozen, 3v2 | 96.5% | 3.5% | 0.81 |
| Intended receiver frozen, 3v2 | 96.5% | 3.5% | 0.82 |
| Support players frozen, 3v2 | 96.5% | 2.0% | 0.74 |
| Random off-ball movement, 3v2 | 48.5% | 2.0% | 0.24 |
| Direct run and shoot | 35.5% | 64.5% | 0.00 |
| Learned policy, 5v4 zero-shot | 43.0% | 32.0% | 25.20 |
| Direct run and shoot, 5v4 | 12.5% | 87.5% | 0.00 |
| Learned policy, 6v4 zero-shot | 42.5% | 33.5% | 37.00 |
| Direct run and shoot, 6v4 | 13.0% | 87.0% | 0.00 |

The selected policy completes 98.0% of 3v2 pass attempts, and 65.5% of episodes
end with a multi-attacker goal after a pass. Freezing learned off-ball movement
has little effect, but replacing it with random movement drops scoring to
48.5%, showing that uncontrolled movement is strongly harmful. The much higher
pass counts in larger rosters also reveal a remaining tendency to over-pass.

Run the preserved baseline through the same evaluator with:

```bash
python swarm_3v2.py --model artifacts/shared_ppo_baseline.pt
```


## Train

```bash
python swarm_3v2.py --train
```

Training starts from the passing-capable shared-PPO actor and gives it a new
centralized MAPPO critic. It then samples three 3v2 scenarios throughout
training:

1. An unpressured passing drill.
2. A pressured passing drill.
3. The complete 3v2 game.

The mixture changes from 40/30/30 early, to 20/30/50 in the middle, and
10/20/70 late. Passing practice therefore never disappears. The same Adam
optimizer is preserved across all phases.

The reported experiment used three independent seeds and a 400,000-step late
phase:

```bash
python swarm_3v2.py --train --steps 400000 --seeds 3
```

Every 100,000 steps, training measures goals, passing, and off-ball value on
held-out layouts. Each seed saves separate best-goals, best-teamwork, and
best-overall checkpoints. A best-overall checkpoint must meet minimum scoring,
passing, goal-after-pass, and off-ball requirements. The selected model is
saved to `artifacts/goal_swarm_3v2.pt`.

Across the three reported seeds, average 3v2 scoring was 91.8% with a 3.7
percentage-point standard deviation.

## Rewards

All three attackers receive the same team reward:

- `+20` for scoring
- `-10` for losing possession, putting the ball out, or timing out
- `+0.6` to `+1.0` for each of the first three completed forward passes
- Small changes for forward ball progress and creating safe passing options
- A small time cost and penalty for trying to shoot too early
