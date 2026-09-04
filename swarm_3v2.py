"""The complete 3v2 soccer task: world, physics, reward, and commands.

Three attackers share one neural network. The nearest defender presses the
ball; the other defender covers the center. The attackers succeed when the
ball crosses the yellow progression line at x=88 metres.
"""

import argparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch

from neural_network import load_model, train


class Action(IntEnum):
    HOLD = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SEND_FORWARD = 5
    UP_RIGHT = 6
    DOWN_RIGHT = 7
    PASS = 8


MOVEMENT = {
    Action.HOLD: (0.0, 0.0),
    Action.UP: (0.0, -1.0),
    Action.DOWN: (0.0, 1.0),
    Action.LEFT: (-1.0, 0.0),
    Action.RIGHT: (1.0, 0.0),
    Action.UP_RIGHT: (1.0, -1.0),
    Action.DOWN_RIGHT: (1.0, 1.0),
}

# Edit these five coordinates to change the base formation. Coordinates are
# measured in metres from the pitch's top-left corner.
STARTING_POSITIONS = {
    1: (34.0, 34.0),  # Attacker with the ball
    2: (43.0, 21.0),  # Upper attacker
    3: (43.0, 47.0),  # Lower attacker
    4: (61.0, 28.0),  # Upper defender
    5: (63.0, 41.0),  # Lower defender
}


@dataclass
class Player:
    number: int
    team: int
    position: np.ndarray
    velocity: np.ndarray
    has_ball: bool = False


@dataclass
class Ball:
    position: np.ndarray
    velocity: np.ndarray
    owner: Optional[int] = None
    possession_ticks: int = 0
    target_player: Optional[int] = None


class Swarm3v2:
    """Small fixed-timestep soccer world and learning environment."""

    width = 105.0
    height = 68.0
    progression_x = 88.0
    ticks_per_second = 10
    max_ticks = 300
    observation_size = 16
    action_count = 9
    attacker_ids = (1, 2, 3)
    defender_ids = (4, 5)

    def __init__(self, starting_jitter=2.0):
        self.starting_jitter = starting_jitter
        self.players = {}
        self.ball = None
        self.tick = 0
        self.result = "running"
        self.passer = None
        self.reset(0)

    def reset(self, seed=0):
        """Start a repeatable episode with slightly randomized positions."""

        rng = np.random.default_rng(seed)
        self.players = {}
        for number, start in STARTING_POSITIONS.items():
            position = np.array(
                [
                    start[0] + rng.uniform(-self.starting_jitter, self.starting_jitter),
                    start[1] + rng.uniform(-self.starting_jitter, self.starting_jitter),
                ],
                dtype=np.float64,
            )
            position[:] = np.clip(position, (0, 0), (self.width, self.height))
            self.players[number] = Player(
                number=number,
                team=0 if number <= 3 else 1,
                position=position,
                velocity=np.zeros(2),
                has_ball=number == 1,
            )
        self.ball = Ball(self.players[1].position.copy(), np.zeros(2), owner=1)
        self.tick = 0
        self.result = "running"
        self.passer = None
        return self.observations()

    def step(self, action_ids):
        """Advance one tenth of a second and return the shared team reward."""

        if len(action_ids) != 3:
            raise ValueError("The three attackers each need one action")
        if self.result != "running":
            return self.observations(), 0.0, True, self.result

        action_ids = [Action(int(value)) for value in action_ids]
        previous_x = self.ball.position[0]
        previous_owner = self.ball.owner

        directions = {}
        for player_id, action in zip(self.attacker_ids, action_ids):
            # A player without the ball treats SEND_FORWARD as running forward.
            directions[player_id] = MOVEMENT.get(
                action,
                (1.0, 0.0) if action == Action.SEND_FORWARD else (0.0, 0.0),
            )
        directions.update(self._defender_directions())
        self._move_players(directions)
        self._tackle_if_close()
        self._kick_ball(action_ids)
        self._move_ball()
        self._collect_loose_ball()

        self.tick += 1
        if self.ball.owner is not None:
            self.ball.possession_ticks += 1

        reward = 5.0 * (self.ball.position[0] - previous_x) / self.width - 0.002

        # A pass is complete when a different attacker receives it.
        completed_pass = False
        if self.passer is not None and self.ball.owner in self.attacker_ids:
            completed_pass = self.ball.owner != self.passer
            reward += 0.2 if completed_pass else 0.0
            self.passer = None
        if previous_owner in self.attacker_ids and self.ball.owner is None:
            owner_action = action_ids[self.attacker_ids.index(previous_owner)]
            if owner_action == Action.PASS:
                self.passer = previous_owner

        if self.ball.position[0] >= self.progression_x:
            self.result = "success"
            reward += 10.0
        elif self.ball.owner in self.defender_ids:
            self.result = "turnover"
            reward -= 10.0
        elif self.tick >= self.max_ticks:
            self.result = "timeout"
            reward -= 10.0

        self.completed_pass = completed_pass
        return self.observations(), float(reward), self.result != "running", self.result

    def observations(self):
        """Return one 16-number, player-centered view per attacker."""

        return np.stack([self._observation(number) for number in self.attacker_ids])

    def _observation(self, number):
        player = self.players[number]
        teammates = [self.players[n] for n in self.attacker_ids if n != number]
        defenders = [self.players[n] for n in self.defender_ids]
        values = [
            2 * player.position[0] / self.width - 1,
            2 * player.position[1] / self.height - 1,
            (self.ball.position[0] - player.position[0]) / self.width,
            (self.ball.position[1] - player.position[1]) / self.height,
            (self.progression_x - player.position[0]) / self.width,
            (self.height / 2 - player.position[1]) / self.height,
            1.0 if self.ball.owner == number else -1.0,
            1.0 if self.ball.owner in self.attacker_ids else -1.0,
        ]
        for other in teammates + defenders:
            values.extend(
                [
                    (other.position[0] - player.position[0]) / self.width,
                    (other.position[1] - player.position[1]) / self.height,
                ]
            )
        return np.clip(np.array(values, np.float32), -1, 1)

    def _move_players(self, directions):
        for number, player in self.players.items():
            direction = np.array(directions[number], dtype=np.float64)
            length = np.linalg.norm(direction)
            if length > 1:
                direction /= length
            player.velocity = direction * 7.0
            player.position += player.velocity / self.ticks_per_second
            player.position[:] = np.clip(player.position, (0, 0), (self.width, self.height))
        if self.ball.owner is not None:
            self.ball.position = self.players[self.ball.owner].position.copy()
            self.ball.velocity[:] = 0

    def _tackle_if_close(self):
        if self.ball.owner is None or self.ball.possession_ticks < 3:
            return
        owner = self.players[self.ball.owner]
        opponents = [
            player
            for player in self.players.values()
            if player.team != owner.team and distance(player.position, owner.position) <= 1.75
        ]
        if opponents:
            winner = min(opponents, key=lambda player: (distance(player.position, owner.position), player.number))
            owner.has_ball = False
            winner.has_ball = True
            self.ball.owner = winner.number
            self.ball.position = winner.position.copy()
            self.ball.velocity[:] = 0
            self.ball.possession_ticks = 0

    def _kick_ball(self, actions):
        owner = self.ball.owner
        if owner not in self.attacker_ids:
            return
        action = actions[self.attacker_ids.index(owner)]
        if action == Action.PASS:
            teammates = [self.players[n] for n in self.attacker_ids if n != owner]
            ahead = [player for player in teammates if player.position[0] > self.players[owner].position[0]]
            receiver = max(ahead or teammates, key=lambda player: player.position[0])
            self._start_kick(receiver.position, receiver.number)
        elif action == Action.SEND_FORWARD:
            self._start_kick(np.array([self.progression_x + 2, self.height / 2]), None)

    def _start_kick(self, target, receiver):
        owner = self.players[self.ball.owner]
        direction = target - self.ball.position
        length = np.linalg.norm(direction)
        if length == 0:
            return
        owner.has_ball = False
        self.ball.owner = None
        self.ball.possession_ticks = 0
        self.ball.target_player = receiver
        self.ball.velocity = direction / length * 18.0

    def _move_ball(self):
        if self.ball.owner is not None:
            return
        if self.ball.target_player in self.players:
            target = self.players[self.ball.target_player].position
            direction = target - self.ball.position
            length = np.linalg.norm(direction)
            if length <= 2.8:  # One tick of travel plus the 1 m control radius.
                self.ball.position = target.copy()
                self.ball.velocity[:] = 0
                self.ball.target_player = None
                return
            self.ball.velocity = direction / length * 18.0
        self.ball.position += self.ball.velocity / self.ticks_per_second
        if self.ball.target_player is None:
            self.ball.velocity *= 0.97
        if np.linalg.norm(self.ball.velocity) < 0.05:
            self.ball.velocity[:] = 0

    def _collect_loose_ball(self):
        if self.ball.owner is not None:
            return
        nearby = [
            player
            for player in self.players.values()
            if distance(player.position, self.ball.position) <= 1.0
        ]
        if nearby:
            winner = min(nearby, key=lambda player: (distance(player.position, self.ball.position), player.number))
            winner.has_ball = True
            self.ball.owner = winner.number
            self.ball.position = winner.position.copy()
            self.ball.velocity[:] = 0
            self.ball.possession_ticks = 0
            self.ball.target_player = None

    def _defender_directions(self):
        pressing = min(
            self.defender_ids,
            key=lambda number: distance(self.players[number].position, self.ball.position),
        )
        result = {}
        for number in self.defender_ids:
            target = self.ball.position
            if number != pressing:
                target = np.array(
                    [min(self.ball.position[0] + 12, self.progression_x - 3), self.height / 2]
                )
            result[number] = direction_to(self.players[number].position, target, 0.78)
        return result


def distance(a, b):
    return float(np.linalg.norm(a - b))


def direction_to(start, target, speed=1.0):
    difference = target - start
    length = np.linalg.norm(difference)
    return (0.0, 0.0) if length == 0 else tuple(difference / length * speed)


def model_actions(model, observations):
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observations))
    return torch.argmax(logits, dim=-1).numpy()


def evaluate(model, episodes=100, first_seed=20_000):
    successes = turnovers = passes = 0
    for seed in range(first_seed, first_seed + episodes):
        env = Swarm3v2()
        observations = env.reset(seed)
        while env.result == "running":
            observations, _, _, _ = env.step(model_actions(model, observations))
            passes += env.completed_pass
        successes += env.result == "success"
        turnovers += env.result == "turnover"
    print(
        f"evaluation episodes={episodes} success={successes / episodes:.1%} "
        f"turnovers={turnovers / episodes:.1%} passes_per_episode={passes / episodes:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train a new model")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--model", default="artifacts/ppo_swarm_3v2.pt")
    args = parser.parse_args()
    env = Swarm3v2()
    model = train(env, args.steps, filename=args.model) if args.train else load_model(args.model)
    evaluate(model)


if __name__ == "__main__":
    main()
