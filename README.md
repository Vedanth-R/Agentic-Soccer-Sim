# PitchLab: Simple 3v2 Swarm

This project has one experiment: three attacking agents share one neural
network and try to move the ball past two scripted defenders.

## Run it

Activate the existing environment:

```bash
source .venv/bin/activate
```

Watch the trained agents:

```bash
python pygame_visualizer.py
```

Press `N` to generate a new seeded starting layout and `R` to replay the
current layout. Increase the amount of position variation with:

```bash
python pygame_visualizer.py --jitter 6
```

You can also start from a specific repeatable layout:

```bash
python pygame_visualizer.py --seed 20025
```

To change the formation itself, edit the clearly labeled
`STARTING_POSITIONS` dictionary near the top of `swarm_3v2.py`.

Evaluate the trained model over 100 repeatable starting positions:

```bash
python swarm_3v2.py
```

Train a new model and then evaluate it:

```bash
python swarm_3v2.py --train
```

The default training run uses 500,000 simulation steps and saves the result to
`artifacts/ppo_swarm_3v2.pt`.

## Read the code in this order

1. `swarm_3v2.py`
   - Defines players, ball, actions, physics, defenders, observations, rewards,
     and the 3v2 episode.
   - The three attackers each receive a different 16-number observation.
   - All three use the same neural network.

2. `neural_network.py`
   - Defines the small actor-critic neural network.
   - Collects experience from all three attackers.
   - Updates the network with PPO and saves the weights.

3. `pygame_visualizer.py`
   - Loads the saved model.
   - Asks the model for three actions every simulation tick.
   - Draws the resulting world without performing any training.

## What the agents learn

The attackers share this team reward:

- `+10` for crossing the yellow progression line
- `-10` when the defenders take the ball
- `-10` for running out of time
- Small rewards for forward ball movement and completed passes

The defenders are not trained. The nearest defender presses the ball and the
other protects the middle.

Current evaluation over seeds `20000` through `20099` gives the trained swarm
99% success and 1% turnovers. This is a small learning simulation, not a claim
about real soccer tactics.

## Viewer controls

- `Space`: pause or resume
- `R`: replay the same starting seed
- `N`: advance to a new seed and new starting positions
- `-` / `+`: change playback speed
- `Esc`: quit
