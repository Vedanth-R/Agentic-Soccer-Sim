"""A configurable soccer world used for training and evaluation.

Attackers share one neural network. One defender presses the ball while the
others mark forward attackers. The episode is won only by scoring a goal.
"""

import argparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch

from neural_network import load_model, save_model, train


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

# The original formation is preserved for the default 3v2 experiment.
THREE_V_TWO_POSITIONS = {
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
    intended_receiver: Optional[int] = None


class SwarmSoccer:
    width = 105.0
    height = 68.0
    goal_center_y = 34.0
    goal_width = 7.32
    shooting_x = 72.0
    ticks_per_second = 10
    max_ticks = 400
    action_count = 9
    max_attackers = 6
    max_defenders = 6
    observation_size = 41

    def __init__(
        self,
        num_attackers=3,
        num_defenders=2,
        starting_jitter=8.0,
        defender_speed=0.90,
        attacker_x_offset=0.0,
        max_ticks=400,
    ):
        self._validate_roster(num_attackers, num_defenders)
        self.num_attackers = num_attackers
        self.num_defenders = num_defenders
        self.attacker_ids = tuple(range(1, num_attackers + 1))
        self.defender_ids = tuple(
            range(num_attackers + 1, num_attackers + num_defenders + 1)
        )
        self.starting_positions = self._starting_positions()
        self.starting_jitter = starting_jitter
        self.defender_speed = defender_speed
        self.attacker_x_offset = attacker_x_offset
        self.max_ticks = max_ticks
        self.players = {}
        self.ball = None
        self.reset(0)

    @classmethod
    def _validate_roster(cls, num_attackers, num_defenders):
        if num_attackers < 2 or num_defenders < 1:
            raise ValueError("A scenario needs at least two attackers and one defender")
        if num_attackers > cls.max_attackers or num_defenders > cls.max_defenders:
            raise ValueError(
                f"The shared observation supports at most {cls.max_attackers} attackers "
                f"and {cls.max_defenders} defenders"
            )

    def _starting_positions(self):
        if self.num_attackers == 3 and self.num_defenders == 2:
            return dict(THREE_V_TWO_POSITIONS)

        positions = {1: (42.0, 34.0)}
        other_y = np.linspace(10.0, 58.0, self.num_attackers)[1:]
        for index, y in enumerate(other_y, start=2):
            positions[index] = (48.0, float(y))
        defender_y = np.linspace(12.0, 56.0, self.num_defenders)
        for index, y in enumerate(defender_y, start=self.num_attackers + 1):
            positions[index] = (68.0, float(y))
        return positions

    def reset(self, seed=0):
        """Create a repeatable but meaningfully varied starting layout."""

        rng = np.random.default_rng(seed)
        self.players = {}
        for number, start in self.starting_positions.items():
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
        return self._begin_episode(owner)

    def set_setup(self, players, owner=None):
        """Replace the roster and positions with an editor or saved scenario."""

        entries = list(players)
        attacker_ids = tuple(sorted(int(p["number"]) for p in entries if int(p["team"]) == 0))
        defender_ids = tuple(sorted(int(p["number"]) for p in entries if int(p["team"]) == 1))
        self._validate_roster(len(attacker_ids), len(defender_ids))
        numbers = attacker_ids + defender_ids
        if len(numbers) != len(set(numbers)):
            raise ValueError("Player numbers must be unique")

        self.attacker_ids = attacker_ids
        self.defender_ids = defender_ids
        self.num_attackers = len(attacker_ids)
        self.num_defenders = len(defender_ids)
        self.starting_positions = {
            int(entry["number"]): tuple(float(value) for value in entry["position"])
            for entry in entries
        }
        self.players = {}
        for entry in entries:
            number = int(entry["number"])
            position = np.clip(np.asarray(entry["position"], dtype=float), (0, 0), (self.width, self.height))
            self.players[number] = Player(
                number=number,
                team=int(entry["team"]),
                position=position,
                velocity=np.zeros(2),
            )
        if owner not in self.attacker_ids:
            owner = self.attacker_ids[0]
        return self._begin_episode(owner)

    def _begin_episode(self, owner):
        for player in self.players.values():
            player.has_ball = player.number == owner
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

        if len(action_ids) != self.num_attackers:
            raise ValueError(f"All {self.num_attackers} attackers need one action")
        if self.result != "running":
            return self.observations(), 0.0, True, self.result

        actions = [Action(int(value)) for value in action_ids]
        old_ball_x = self.ball.position[0]
        old_owner = self.ball.owner
        old_support = self._support_score()

        # If a pass is already travelling, measure whether its receiver moves
        # toward where the ball will be next. This isolates the receiver's
        # movement instead of rewarding the ball simply moving toward them.
        receiver_move = 0.0
        receiver = self.ball.intended_receiver
        if self.ball.owner is None and receiver in self.attacker_ids:
            next_ball_position = (
                self.ball.position + self.ball.velocity / self.ticks_per_second
            )
            old_receiver_distance = distance(
                self.players[receiver].position, next_ball_position
            )

        directions = {
            number: MOVEMENT.get(action, (0.0, 0.0))
            for number, action in zip(self.attacker_ids, actions)
        }
        directions.update(self._defender_directions())
        self._move_players(directions)
        if self.ball.owner is None and receiver in self.attacker_ids:
            new_receiver_distance = distance(
                self.players[receiver].position, next_ball_position
            )
            receiver_move = float(
                np.clip(old_receiver_distance - new_receiver_distance, 0.0, 0.7)
            )
        self._tackle_if_close()

        premature_shot = False
        if self.ball.owner in self.attacker_ids:
            owner_action = actions[self.attacker_ids.index(self.ball.owner)]
            premature_shot = owner_action == Action.SHOOT and self.ball.position[0] < self.shooting_x
        useful_pass_started = self._kick_ball(actions)
        self._move_ball()
        self._collect_loose_ball()

        self.tick += 1
        if self.ball.owner is not None:
            self.ball.possession_ticks += 1

        # Small shaping rewards help learning, but scoring is worth much more.
        progress = float(np.clip(self.ball.position[0] - old_ball_x, -1.0, 1.0))
        reward = 0.02 * progress - 0.003
        # Compare support positions only while the same attacker keeps the
        # ball. A kick makes support_score zero, which should not penalize the
        # act of passing.
        if old_owner in self.attacker_ids and self.ball.owner == old_owner:
            reward += 0.15 * (self._support_score() - old_support)
        reward += 0.03 * receiver_move
        reward += 0.10 if useful_pass_started else 0.0
        reward -= 0.03 if premature_shot else 0.0

        self.completed_pass = False
        if self.passer is not None and self.ball.owner in self.attacker_ids:
            self.completed_pass = self.ball.owner != self.passer
            forward_pass = self.players[self.ball.owner].position[0] > self.pass_start_x + 2.0
            if self.completed_pass and forward_pass and self.rewarded_passes < 3:
                forward_metres = self.players[self.ball.owner].position[0] - self.pass_start_x
                reward += 0.6 + min(0.02 * forward_metres, 0.4)
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
        teammates = sorted(
            (self.players[n] for n in self.attacker_ids if n != number),
            key=lambda other: (distance(player.position, other.position), other.number),
        )
        defenders = sorted(
            (self.players[n] for n in self.defender_ids),
            key=lambda other: (distance(player.position, other.position), other.number),
        )
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
        for other in teammates:
            values.extend(
                [
                    (other.position[0] - player.position[0]) / self.width,
                    (other.position[1] - player.position[1]) / self.height,
                    1.0,
                ]
            )
        values.extend([0.0, 0.0, 0.0] * (self.max_attackers - 1 - len(teammates)))
        for other in defenders:
            values.extend(
                [
                    (other.position[0] - player.position[0]) / self.width,
                    (other.position[1] - player.position[1]) / self.height,
                    1.0,
                ]
            )
        values.extend([0.0, 0.0, 0.0] * (self.max_defenders - len(defenders)))
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
            return False
        action = actions[self.attacker_ids.index(owner)]
        if action == Action.PASS:
            teammates = [self.players[n] for n in self.attacker_ids if n != owner]
            ahead = [player for player in teammates if player.position[0] > self.players[owner].position[0]]
            receiver = max(ahead or teammates, key=lambda player: player.position[0])
            # Aim once at where the receiver is currently running. The ball
            # keeps this direction after the kick; it never homes or curves.
            travel_time = distance(self.ball.position, receiver.position) / 20.0
            lead_time = min(travel_time, 1.5)
            target = receiver.position + receiver.velocity * lead_time
            target = np.clip(target, (1, 1), (self.width - 1, self.height - 1))
            forward_pass = receiver.position[0] > self.players[owner].position[0] + 2.0
            open_lane = min(
                point_to_segment(self.players[d].position, self.ball.position, target)
                for d in self.defender_ids
            ) >= 2.5
            self._start_kick(target, receiver.number, speed=24.0)
            return forward_pass and open_lane
        elif action == Action.SHOOT and self.players[owner].position[0] >= self.shooting_x:
            self._start_kick(np.array([self.width + 1, self.goal_center_y]), speed=22.0)
        return False

    def _start_kick(self, target, intended_receiver=None, speed=24.0):
        owner = self.players[self.ball.owner]
        difference = target - self.ball.position
        length = np.linalg.norm(difference)
        if length == 0:
            return
        owner.has_ball = False
        self.ball.owner = None
        self.ball.possession_ticks = 0
        self.ball.intended_receiver = intended_receiver
        self.ball.velocity = difference / length * speed

    def _move_ball(self):
        if self.ball.owner is not None:
            return
        self.ball.position += self.ball.velocity / self.ticks_per_second
        self.ball.velocity *= 0.97

    def _collect_loose_ball(self):
        if self.ball.owner is not None:
            return
        nearby = [
            player for player in self.players.values()
            if distance(player.position, self.ball.position)
            <= (2.0 if player.number == self.ball.intended_receiver else 1.0)
        ]
        if nearby:
            winner = min(nearby, key=lambda player: (distance(player.position, self.ball.position), player.number))
            winner.has_ball = True
            self.ball.owner = winner.number
            self.ball.position = winner.position.copy()
            self.ball.velocity[:] = 0
            self.ball.possession_ticks = 0
            self.ball.intended_receiver = None

    def _defender_directions(self):
        pressing = min(
            self.defender_ids,
            key=lambda number: distance(self.players[number].position, self.ball.position),
        )
        directions = {}
        markers = [number for number in self.defender_ids if number != pressing]
        attacking_options = sorted(
            (self.players[n] for n in self.attacker_ids),
            key=lambda player: player.position[0],
            reverse=True,
        )
        for number in self.defender_ids:
            if number == pressing:
                target = self.ball.position
            else:
                marked = attacking_options[markers.index(number) % len(attacking_options)]
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


class Swarm3v2(SwarmSoccer):
    """The original training environment and checkpoint-compatible roster."""

    def __init__(self, **kwargs):
        super().__init__(num_attackers=3, num_defenders=2, **kwargs)


def model_actions(model, observations):
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observations))
    return torch.argmax(logits, dim=-1).numpy()


def goal_rate(model, env_factory=Swarm3v2, episodes=100, first_seed=30_000):
    """Return deterministic goal rate on held-out layouts without printing."""

    goals = 0
    for seed in range(first_seed, first_seed + episodes):
        env = env_factory()
        observations = env.reset(seed)
        while env.result == "running":
            observations, _, _, result = env.step(model_actions(model, observations))
        goals += result == "success"
    return goals / episodes


def _evaluate_actions(
    name,
    choose_actions,
    episodes=200,
    first_seed=20_000,
    env_factory=Swarm3v2,
):
    outcomes = {"success": 0, "turnover": 0, "out": 0, "timeout": 0}
    passes = 0
    for seed in range(first_seed, first_seed + episodes):
        env = env_factory()
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
        actions = np.full(len(env.attacker_ids), Action.RIGHT)
        if env.ball.owner in env.attacker_ids:
            owner = env.players[env.ball.owner]
            if owner.position[0] >= env.shooting_x:
                actions[env.attacker_ids.index(env.ball.owner)] = Action.SHOOT
        return actions

    _evaluate_actions("direct run and shoot", direct_play, episodes, first_seed)

    for attackers, defenders in ((5, 4), (6, 4)):
        make_env = lambda a=attackers, d=defenders: SwarmSoccer(a, d)
        _evaluate_actions(
            f"learned policy {attackers}v{defenders}",
            lambda env, observations: model_actions(model, observations),
            episodes,
            first_seed,
            make_env,
        )
        _evaluate_actions(
            f"direct baseline {attackers}v{defenders}",
            direct_play,
            episodes,
            first_seed,
            make_env,
        )


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
        candidates = []

        def remember_stage(stage_name, trained_model):
            rate = goal_rate(trained_model)
            state = {name: value.detach().clone() for name, value in trained_model.state_dict().items()}
            candidates.append((rate, stage_name, state))
            print(f"held-out full-3v2 goal rate after {stage_name}: {rate:.1%}")

        print("stage 1/3: learn to approach and shoot")
        model = train(
            Swarm3v2(starting_jitter=4, defender_speed=0, attacker_x_offset=25, max_ticks=200),
            150_000,
            seed=4,
            filename=args.model,
        )
        remember_stage("stage 1", model)
        print("stage 2/3: add distance and moderate pressure")
        model = train(
            Swarm3v2(starting_jitter=6, defender_speed=0.65, attacker_x_offset=12, max_ticks=300),
            300_000,
            seed=10_000,
            filename=args.model,
            model=model,
        )
        remember_stage("stage 2", model)
        print("stage 3/3: train the complete 3v2")
        model = train(
            Swarm3v2(),
            args.steps,
            seed=20_000,
            filename=args.model,
            model=model,
        )
        remember_stage("stage 3", model)
        best_rate, best_stage, best_state = max(candidates, key=lambda candidate: candidate[0])
        model.load_state_dict(best_state)
        save_model(model, args.model)
        print(f"kept {best_stage} checkpoint ({best_rate:.1%} held-out goals)")
    else:
        model = load_model(args.model)
    evaluate(model)


if __name__ == "__main__":
    main()
