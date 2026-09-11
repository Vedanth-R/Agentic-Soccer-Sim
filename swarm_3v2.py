"""A small 3v2 soccer world used for training and evaluation.

Three attackers share one neural network. One defender presses the ball while
the other marks a forward attacker. The episode is won only by scoring a goal.
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
    SHOOT = 5
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

# Edit these base positions to change the formation. Training adds random
# variation and randomly chooses which attacker starts with the ball.
STARTING_POSITIONS = {
    1: (42.0, 34.0),
    2: (50.0, 18.0),
    3: (50.0, 50.0),
    4: (67.0, 27.0),
    5: (69.0, 41.0),
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
    width = 105.0
    height = 68.0
    goal_center_y = 34.0
    goal_width = 7.32
    shooting_x = 72.0
    ticks_per_second = 10
    max_ticks = 400
    observation_size = 16
    action_count = 9
    attacker_ids = (1, 2, 3)
    defender_ids = (4, 5)

    def __init__(
        self,
        starting_jitter=8.0,
        defender_speed=0.90,
        attacker_x_offset=0.0,
        max_ticks=400,
    ):
        self.starting_jitter = starting_jitter
        self.defender_speed = defender_speed
        self.attacker_x_offset = attacker_x_offset
        self.max_ticks = max_ticks
        self.players = {}
        self.ball = None
        self.reset(0)

    def reset(self, seed=0):
        """Create a repeatable but meaningfully varied starting layout."""

        rng = np.random.default_rng(seed)
        self.players = {}
        for number, start in STARTING_POSITIONS.items():
            # Vertical variation is wider than horizontal variation.
            position = np.array(
                [
                    start[0] + rng.uniform(-0.6, 0.6) * self.starting_jitter,
                    start[1] + rng.uniform(-1.0, 1.0) * self.starting_jitter,
                ]
            )
            if number in self.attacker_ids:
                position[0] += self.attacker_x_offset
            position[:] = np.clip(position, (3, 3), (self.width - 3, self.height - 3))
            self.players[number] = Player(
                number,
                0 if number in self.attacker_ids else 1,
                position,
                np.zeros(2),
            )

        owner = int(rng.choice(self.attacker_ids))
        self.players[owner].has_ball = True
        self.ball = Ball(self.players[owner].position.copy(), np.zeros(2), owner)
        self.tick = 0
        self.result = "running"
        self.passer = None
        self.pass_start_x = 0.0
        self.rewarded_passes = 0
        self.completed_pass = False
        return self.observations()

    def step(self, action_ids):
        """Advance one tenth of a second and return one shared team reward."""

        if len(action_ids) != 3:
            raise ValueError("The three attackers each need one action")
        if self.result != "running":
            return self.observations(), 0.0, True, self.result

        actions = [Action(int(value)) for value in action_ids]
        old_ball_x = self.ball.position[0]
        old_owner = self.ball.owner
        old_support = self._support_score()

        directions = {
            number: MOVEMENT.get(action, (0.0, 0.0))
            for number, action in zip(self.attacker_ids, actions)
        }
        directions.update(self._defender_directions())
        self._move_players(directions)
        self._tackle_if_close()

        premature_shot = False
        if self.ball.owner in self.attacker_ids:
            owner_action = actions[self.attacker_ids.index(self.ball.owner)]
            premature_shot = owner_action == Action.SHOOT and self.ball.position[0] < self.shooting_x
        self._kick_ball(actions)
        self._move_ball()
        self._collect_loose_ball()

        self.tick += 1
        if self.ball.owner is not None:
            self.ball.possession_ticks += 1

        # Small shaping rewards help learning, but scoring is worth much more.
        progress = float(np.clip(self.ball.position[0] - old_ball_x, -1.0, 1.0))
        reward = 0.02 * progress - 0.003
        reward += 0.15 * (self._support_score() - old_support)
        reward -= 0.03 if premature_shot else 0.0

        self.completed_pass = False
        if self.passer is not None and self.ball.owner in self.attacker_ids:
            self.completed_pass = self.ball.owner != self.passer
            forward_pass = self.players[self.ball.owner].position[0] > self.pass_start_x + 2.0
            if self.completed_pass and forward_pass and self.rewarded_passes < 3:
                reward += 0.4
                self.rewarded_passes += 1
            self.passer = None

        if old_owner in self.attacker_ids and self.ball.owner is None:
            old_action = actions[self.attacker_ids.index(old_owner)]
            if old_action == Action.PASS:
                self.passer = old_owner
                self.pass_start_x = self.players[old_owner].position[0]

        if self._is_goal():
            self.result = "success"
            reward += 20.0
        elif self.ball.owner in self.defender_ids:
            self.result = "turnover"
            reward -= 10.0
        elif self._ball_is_out():
            self.result = "out"
            reward -= 10.0
        elif self.tick >= self.max_ticks:
            self.result = "timeout"
            reward -= 10.0

        return self.observations(), float(reward), self.result != "running", self.result

    def observations(self):
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
            (self.width - player.position[0]) / self.width,
            (self.goal_center_y - player.position[1]) / self.height,
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

    def _support_score(self):
        """Measure safe forward passing options; changes matter, not its raw value."""

        if self.ball.owner not in self.attacker_ids:
            return 0.0
        owner = self.players[self.ball.owner]
        scores = []
        for number in self.attacker_ids:
            if number == owner.number:
                continue
            option = self.players[number]
            forward_distance = option.position[0] - owner.position[0]
            useful_distance = 1.0 if -3.0 <= forward_distance <= 30.0 else 0.0
            defender_space = min(distance(option.position, self.players[d].position) for d in self.defender_ids)
            safety = float(np.clip(defender_space / 10.0, 0.0, 1.0))
            lane_open = min(
                point_to_segment(self.players[d].position, owner.position, option.position)
                for d in self.defender_ids
            ) >= 2.5
            scores.append(0.35 * useful_distance + 0.35 * safety + 0.30 * lane_open)
        return sum(scores) / len(scores)

    def _move_players(self, directions):
        for number, player in self.players.items():
            direction = np.array(directions[number], dtype=float)
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
            player for player in self.players.values()
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
        elif action == Action.SHOOT and self.players[owner].position[0] >= self.shooting_x:
            self._start_kick(np.array([self.width + 1, self.goal_center_y]), None)

    def _start_kick(self, target, receiver):
        owner = self.players[self.ball.owner]
        difference = target - self.ball.position
        length = np.linalg.norm(difference)
        if length == 0:
            return
        owner.has_ball = False
        self.ball.owner = None
        self.ball.possession_ticks = 0
        self.ball.target_player = receiver
        self.ball.velocity = difference / length * 18.0

    def _move_ball(self):
        if self.ball.owner is not None:
            return
        if self.ball.target_player in self.players:
            target = self.players[self.ball.target_player].position
            difference = target - self.ball.position
            length = np.linalg.norm(difference)
            if length <= 2.8:
                self.ball.position = target.copy()
                self.ball.velocity[:] = 0
                self.ball.target_player = None
                return
            self.ball.velocity = difference / length * 18.0
        self.ball.position += self.ball.velocity / self.ticks_per_second
        if self.ball.target_player is None:
            self.ball.velocity *= 0.97

    def _collect_loose_ball(self):
        if self.ball.owner is not None:
            return
        nearby = [
            player for player in self.players.values()
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
        directions = {}
        for number in self.defender_ids:
            if number == pressing:
                target = self.ball.position
            else:
                options = [self.players[n] for n in self.attacker_ids]
                marked = max(options, key=lambda player: player.position[0])
                target = np.array([min(marked.position[0] + 3.0, 94.0), marked.position[1]])
            directions[number] = direction_to(
                self.players[number].position,
                target,
                self.defender_speed,
            )
        return directions

    def _is_goal(self):
        return self.ball.position[0] >= self.width and abs(self.ball.position[1] - self.goal_center_y) <= self.goal_width / 2

    def _ball_is_out(self):
        x, y = self.ball.position
        return y < 0 or y > self.height or x < 0 or x >= self.width


def distance(a, b):
    return float(np.linalg.norm(a - b))


def direction_to(start, target, speed=1.0):
    difference = target - start
    length = np.linalg.norm(difference)
    return (0.0, 0.0) if length == 0 else tuple(difference / length * speed)


def point_to_segment(point, start, end):
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0:
        return distance(point, start)
    amount = float(np.clip(np.dot(point - start, segment) / length_squared, 0, 1))
    return distance(point, start + amount * segment)


def model_actions(model, observations):
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observations))
    return torch.argmax(logits, dim=-1).numpy()


def _evaluate_actions(name, choose_actions, episodes=200, first_seed=20_000):
    outcomes = {"success": 0, "turnover": 0, "out": 0, "timeout": 0}
    passes = 0
    for seed in range(first_seed, first_seed + episodes):
        env = Swarm3v2()
        observations = env.reset(seed)
        while env.result == "running":
            actions = choose_actions(env, observations)
            observations, _, _, _ = env.step(actions)
            passes += env.completed_pass
        outcomes[env.result] += 1
    print(
        f"{name}: goals={outcomes['success'] / episodes:.1%} "
        f"turnovers={outcomes['turnover'] / episodes:.1%} "
        f"out={outcomes['out'] / episodes:.1%} timeouts={outcomes['timeout'] / episodes:.1%} "
        f"passes_per_episode={passes / episodes:.2f}"
    )


def evaluate(model, episodes=200, first_seed=20_000):
    """Compare the policy with its off-ball movement removed and direct play."""

    _evaluate_actions(
        "learned policy",
        lambda env, observations: model_actions(model, observations),
        episodes,
        first_seed,
    )

    def freeze_off_ball(env, observations):
        actions = model_actions(model, observations)
        for index, number in enumerate(env.attacker_ids):
            if number != env.ball.owner:
                actions[index] = Action.HOLD
        return actions

    _evaluate_actions("off-ball players frozen", freeze_off_ball, episodes, first_seed)

    def direct_play(env, observations):
        actions = np.full(3, Action.RIGHT)
        if env.ball.owner in env.attacker_ids:
            owner = env.players[env.ball.owner]
            if owner.position[0] >= env.shooting_x:
                actions[env.attacker_ids.index(env.ball.owner)] = Action.SHOOT
        return actions

    _evaluate_actions("direct run and shoot", direct_play, episodes, first_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=700_000,
        help="steps in the final and hardest curriculum stage",
    )
    parser.add_argument("--model", default="artifacts/goal_swarm_3v2.pt")
    args = parser.parse_args()
    if args.train:
        print("stage 1/3: learn to approach and shoot")
        model = train(
            Swarm3v2(starting_jitter=4, defender_speed=0, attacker_x_offset=25, max_ticks=200),
            150_000,
            seed=4,
            filename=args.model,
        )
        print("stage 2/3: add distance and moderate pressure")
        model = train(
            Swarm3v2(starting_jitter=6, defender_speed=0.65, attacker_x_offset=12, max_ticks=300),
            300_000,
            seed=10_000,
            filename=args.model,
            model=model,
        )
        print("stage 3/3: train the complete 3v2")
        model = train(
            Swarm3v2(),
            args.steps,
            seed=20_000,
            filename=args.model,
            model=model,
        )
    else:
        model = load_model(args.model)
    evaluate(model)


if __name__ == "__main__":
    main()
