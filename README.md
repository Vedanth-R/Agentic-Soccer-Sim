# PitchLab: 3v2 Swarm

This project currently runs a 3v2 sim with 3 attacking agents sharing one neural network trained on shared data, and 2 scripted defenders. Success is defined as moving the ball past a certain point on the field without losing it to the defenders. Current bugs include the attackers simply kicking the ball downfield to the point on the field.

## Run it

Activate the existing environment:

```bash
source .venv/bin/activate
```

Watch the trained agents in the Pygame visualization:

```bash
python pygame_visualizer.py
```

Press `N` to generate a new seeded starting layout and `R` to replay the
current layout. Increase the amount of position variation with:

```bash
python pygame_visualizer.py --jitter 6
```

The `--jitter` option only changes the layouts shown in the viewer. The saved
network was trained with default 2-metre variations among the players, so larger values test
how well it generalizes.

To change the formation itself, edit the `STARTING_POSITIONS` dictionary near the top of `swarm_3v2.py`.

Evaluate the trained model over 100 repeatable layouts from its familiar
formation range:

```bash
python swarm_3v2.py
```

Train a new model and then evaluate it:

```bash
python swarm_3v2.py --train
```

The default training run uses 500,000 simulation steps and saves the result to
`artifacts/ppo_swarm_3v2.pt`. Each training episode uses the base formation with
up to 2 metres of random movement per player.


## What the agents learn

The attackers share this team reward:

- `+10` for crossing the yellow progression line
- `-10` when the defenders take the ball
- `-10` for running out of time
- Small rewards for forward ball movement and completed passes

I am currently working on fixing this system so they do not just kick the ball towards the line. I am also working to add better defending actions. 

The defenders are not trained. The nearest defender presses the ball and the
other protects the middle.

