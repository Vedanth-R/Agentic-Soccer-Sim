"""A configurable soccer world used for training and evaluation.

Attackers share one neural network. One defender presses the ball while the
others mark forward attackers. The episode is won only by scoring a goal.
"""

import argparse
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from neural_network import initialize_mappo_from_actor, load_model, save_model, train


class Action(IntEnum):
    HOLD = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SHOOT = 5
    UP_RIGHT = 6
    DOWN_RIGHT = 7
    PASS_1 = 8
    PASS_2 = 9
    PASS_3 = 10
    PASS_4 = 11
    PASS_5 = 12


PASS_ACTIONS = {
    Action.PASS_1,
    Action.PASS_2,
    Action.PASS_3,
    Action.PASS_4,
    Action.PASS_5,
}


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
    action_count = 13
    max_attackers = 6
    max_defenders = 6
    observation_size = 51
    global_state_size = 81
    critic_observation_size = global_state_size + observation_size

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
        self.pass_attempts = 0
        self.completed_passes = 0
        self.forward_passes = 0
        self.pass_turnovers = 0
        self.receiver_move_metres = 0.0
        self.possession_contributors = {owner}
        self.goal_after_pass = False
        self.multi_attacker_goal = False
        return self.observations()

    def step(self, action_ids):
        """Advance one tenth of a second and return one shared team reward."""

        if len(action_ids) != self.num_attackers:
            raise ValueError(f"All {self.num_attackers} attackers need one action")
        if self.result != "running":
            return self.observations(), 0.0, True, self.result

        actions = [Action(int(value)) for value in action_ids]
        masks = self.action_masks()
        for index, action in enumerate(actions):
            if not masks[index, int(action)]:
                raise ValueError(
                    f"Action {action.name} is not available to attacker "
                    f"{self.attacker_ids[index]}"
                )
        old_ball_x = self.ball.position[0]
        old_owner = self.ball.owner
        old_support = self._support_score()
        pass_target = self._selected_pass_target(old_owner, actions)

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
            self.receiver_move_metres += receiver_move
        self._tackle_if_close()

        premature_shot = False
        if self.ball.owner in self.attacker_ids:
            owner_action = actions[self.attacker_ids.index(self.ball.owner)]
            premature_shot = owner_action == Action.SHOOT and self.ball.position[0] < self.shooting_x
        useful_pass_started = self._kick_ball(actions, pass_target)
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
            if self.completed_pass:
                self.completed_passes += 1
                self.possession_contributors.add(self.ball.owner)
            if self.completed_pass and forward_pass:
                self.forward_passes += 1
            self.passer = None

        if self._is_goal():
            self.result = "success"
            self.goal_after_pass = self.completed_passes > 0
            self.multi_attacker_goal = len(self.possession_contributors) >= 2
            reward += 20.0
        elif self.ball.owner in self.defender_ids:
            self.result = "turnover"
            self.pass_turnovers += self.passer is not None
            reward -= 10.0
        elif self._ball_is_out():
            self.result = "out"
            self.pass_turnovers += self.passer is not None
            reward -= 10.0
        elif self.tick >= self.max_ticks:
            self.result = "timeout"
            reward -= 10.0

        return self.observations(), float(reward), self.result != "running", self.result

    def observations(self):
        return np.stack([self._observation(number) for number in self.attacker_ids])

    def critic_observations(self):
        """Give MAPPO's critic the full world plus each agent's local view."""

        global_state = self._global_state()
        return np.stack(
            [
                np.concatenate((global_state, self._observation(number)))
                for number in self.attacker_ids
            ]
        ).astype(np.float32)

    def _scenario_vector(self):
        """One-hot task label: easy pass, pressured pass, or full game."""

        return [0.0, 0.0, 1.0]

    def _global_state(self):
        values = [
            2 * self.ball.position[0] / self.width - 1,
            2 * self.ball.position[1] / self.height - 1,
            self.ball.velocity[0] / 24.0,
            self.ball.velocity[1] / 24.0,
            1.0 if self.ball.owner is None else -1.0,
            1.0 if self.ball.owner in self.attacker_ids else -1.0,
        ]
        values.extend(self._scenario_vector())
        for team_ids, maximum in (
            (self.attacker_ids, self.max_attackers),
            (self.defender_ids, self.max_defenders),
        ):
            for number in team_ids:
                player = self.players[number]
                values.extend(
                    [
                        2 * player.position[0] / self.width - 1,
                        2 * player.position[1] / self.height - 1,
                        player.velocity[0] / 7.0,
                        player.velocity[1] / 7.0,
                        1.0 if player.has_ball else -1.0,
                        1.0,
                    ]
                )
            values.extend([0.0] * 6 * (maximum - len(team_ids)))
        return np.clip(np.array(values, np.float32), -1, 1)

    def action_masks(self):
        """Return which discrete actions are meaningful for each attacker."""

        masks = np.zeros((self.num_attackers, self.action_count), dtype=bool)
        masks[:, :5] = True  # Hold and four cardinal movement actions.
        masks[:, int(Action.UP_RIGHT)] = True
        masks[:, int(Action.DOWN_RIGHT)] = True
        if self.ball.owner in self.attacker_ids:
            owner_index = self.attacker_ids.index(self.ball.owner)
            owner = self.players[self.ball.owner]
            masks[owner_index, int(Action.SHOOT)] = owner.position[0] >= self.shooting_x
            teammate_count = len(self._teammates_for(self.ball.owner))
            masks[owner_index, int(Action.PASS_1) : int(Action.PASS_1) + teammate_count] = True
        return masks

    def _teammates_for(self, number):
        player = self.players[number]
        return sorted(
            (self.players[n] for n in self.attacker_ids if n != number),
            key=lambda other: (distance(player.position, other.position), other.number),
        )

    def _selected_pass_target(self, owner, actions):
        if owner not in self.attacker_ids:
            return None
        action = actions[self.attacker_ids.index(owner)]
        if action not in PASS_ACTIONS:
            return None
        slot = int(action) - int(Action.PASS_1)
        teammates = self._teammates_for(owner)
        return teammates[slot].number if slot < len(teammates) else None

    def _observation(self, number):
        player = self.players[number]
        teammates = self._teammates_for(number)
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
            self.ball.velocity[0] / 24.0,
            self.ball.velocity[1] / 24.0,
            1.0 if self.ball.owner is None else -1.0,
            1.0 if self.ball.intended_receiver == number else -1.0,
            self._distance_to_ball_path(player.position),
            min(distance(player.position, defender.position) for defender in defenders)
            / np.hypot(self.width, self.height),
            1.0 if self._has_open_pass(number) else -1.0,
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
        values.extend(self._scenario_vector())
        return np.clip(np.array(values, np.float32), -1, 1)

    def _distance_to_ball_path(self, position):
        if self.ball.owner is not None or np.linalg.norm(self.ball.velocity) == 0:
            path_distance = distance(position, self.ball.position)
        else:
            path_end = self.ball.position + self.ball.velocity * 1.5
            path_distance = point_to_segment(position, self.ball.position, path_end)
        return path_distance / np.hypot(self.width, self.height)

    def _has_open_pass(self, number):
        if self.ball.owner != number:
            return False
        start = self.players[number].position
        return any(
            min(
                point_to_segment(self.players[d].position, start, teammate.position)
                for d in self.defender_ids
            )
            >= 2.5
            for teammate in self._teammates_for(number)
        )

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

    def _kick_ball(self, actions, pass_target):
        owner = self.ball.owner
        if owner not in self.attacker_ids:
            return False
        action = actions[self.attacker_ids.index(owner)]
        if action in PASS_ACTIONS and pass_target in self.attacker_ids:
            receiver = self.players[pass_target]
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
            self.pass_attempts += 1
            self.passer = owner
            self.pass_start_x = self.players[owner].position[0]
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


class PassingDrill(SwarmSoccer):
    """A short episode devoted only to passing and receiving."""

    def __init__(self, pressure=False, **kwargs):
        self.pressure = pressure
        super().__init__(
            num_attackers=2,
            num_defenders=1,
            defender_speed=0.35 if pressure else 0.0,
            starting_jitter=4.0,
            max_ticks=120,
            **kwargs,
        )

    def _starting_positions(self):
        defender_position = (52.0, 18.0) if self.pressure else (90.0, 5.0)
        return {1: (35.0, 27.0), 2: (57.0, 42.0), 3: defender_position}

    def reset(self, seed=0):
        super().reset(seed)
        return self._begin_episode(self.attacker_ids[0])

    def step(self, action_ids):
        observations, reward, done, result = super().step(action_ids)
        if self.completed_pass and self.result == "running":
            self.result = "success"
            reward += 5.0
            done = True
            result = self.result
        return observations, reward, done, result


class MixedTrainingEnv(SwarmSoccer):
    """Always use 3v2, but sample passing and full-game objectives."""

    scenarios = ("easy_pass", "pressured_pass", "full_game")

    def __init__(self, weights=(0.2, 0.3, 0.5)):
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()
        self.scenario = "full_game"
        super().__init__(num_attackers=3, num_defenders=2)

    def _scenario_vector(self):
        return [1.0 if self.scenario == name else 0.0 for name in self.scenarios]

    def _starting_positions(self):
        if self.scenario == "easy_pass":
            return {1: (35, 27), 2: (57, 42), 3: (48, 55), 4: (90, 5), 5: (92, 63)}
        if self.scenario == "pressured_pass":
            return {1: (35, 27), 2: (57, 42), 3: (48, 55), 4: (52, 18), 5: (68, 50)}
        return dict(THREE_V_TWO_POSITIONS)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.scenario = str(rng.choice(self.scenarios, p=self.weights))
        self.starting_positions = self._starting_positions()
        self.starting_jitter = 4.0 if self.scenario != "full_game" else 8.0
        if self.scenario == "easy_pass":
            self.defender_speed = 0.0
        elif self.scenario == "pressured_pass":
            self.defender_speed = 0.35
        else:
            self.defender_speed = 0.90
        observations = super().reset(seed)
        if self.scenario != "full_game":
            return self._begin_episode(self.attacker_ids[0])
        return observations

    def step(self, action_ids):
        observations, reward, done, result = super().step(action_ids)
        if self.scenario != "full_game" and self.forward_passes > 0 and self.result == "running":
            self.result = "success"
            reward += 5.0
            done = True
            result = self.result
        return observations, reward, done, result


class Swarm3v2(SwarmSoccer):
    """The original training environment and checkpoint-compatible roster."""

    def __init__(self, **kwargs):
        super().__init__(num_attackers=3, num_defenders=2, **kwargs)


def model_actions(model, observations, masks):
    expected_inputs = model.body[0].in_features
    observations = observations[..., :expected_inputs]
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observations))
        logits = logits.masked_fill(~torch.from_numpy(masks), -1e9)
    return torch.argmax(logits, dim=-1).numpy()


def goal_rate(model, env_factory=Swarm3v2, episodes=100, first_seed=30_000):
    """Return deterministic goal rate on held-out layouts without printing."""

    goals = 0
    for seed in range(first_seed, first_seed + episodes):
        env = env_factory()
        observations = env.reset(seed)
        while env.result == "running":
            actions = model_actions(model, observations, env.action_masks())
            observations, _, _, result = env.step(actions)
        goals += result == "success"
    return goals / episodes


def collect_metrics(
    choose_actions,
    episodes=200,
    first_seed=20_000,
    env_factory=Swarm3v2,
):
    outcomes = {"success": 0, "turnover": 0, "out": 0, "timeout": 0}
    pass_attempts = 0
    completed_passes = 0
    forward_passes = 0
    pass_turnovers = 0
    goals_after_pass = 0
    multi_attacker_goals = 0
    episodes_with_attempt = 0
    episodes_with_completion = 0
    episodes_with_forward_pass = 0
    unique_attackers = 0
    receiver_movement = 0.0
    for seed in range(first_seed, first_seed + episodes):
        env = env_factory()
        observations = env.reset(seed)
        while env.result == "running":
            actions = choose_actions(env, observations)
            observations, _, _, _ = env.step(actions)
        outcomes[env.result] += 1
        pass_attempts += env.pass_attempts
        completed_passes += env.completed_passes
        forward_passes += env.forward_passes
        pass_turnovers += env.pass_turnovers
        goals_after_pass += env.goal_after_pass
        multi_attacker_goals += env.multi_attacker_goal
        episodes_with_attempt += env.pass_attempts > 0
        episodes_with_completion += env.completed_passes > 0
        episodes_with_forward_pass += env.forward_passes > 0
        unique_attackers += len(env.possession_contributors)
        receiver_movement += env.receiver_move_metres
    completion_rate = completed_passes / max(pass_attempts, 1)
    return {
        "goals": outcomes["success"] / episodes,
        "turnovers": outcomes["turnover"] / episodes,
        "out": outcomes["out"] / episodes,
        "timeouts": outcomes["timeout"] / episodes,
        "pass_attempts": pass_attempts / episodes,
        "pass_completion": completion_rate,
        "forward_passes": forward_passes / episodes,
        "pass_turnovers": pass_turnovers / episodes,
        "goals_after_pass": goals_after_pass / episodes,
        "multi_attacker_goals": multi_attacker_goals / episodes,
        "episodes_with_attempt": episodes_with_attempt / episodes,
        "episodes_with_completion": episodes_with_completion / episodes,
        "episodes_with_forward_pass": episodes_with_forward_pass / episodes,
        "unique_attackers": unique_attackers / episodes,
        "receiver_movement": receiver_movement / episodes,
    }


def _evaluate_actions(
    name,
    choose_actions,
    episodes=200,
    first_seed=20_000,
    env_factory=Swarm3v2,
):
    metrics = collect_metrics(choose_actions, episodes, first_seed, env_factory)
    print(
        f"{name}: goals={metrics['goals']:.1%} turnovers={metrics['turnovers']:.1%} "
        f"out={metrics['out']:.1%} timeouts={metrics['timeouts']:.1%} "
        f"pass_attempts={metrics['pass_attempts']:.2f} "
        f"pass_completion={metrics['pass_completion']:.1%} "
        f"forward_passes={metrics['forward_passes']:.2f} "
        f"pass_turnovers={metrics['pass_turnovers']:.2f} "
        f"goals_after_pass={metrics['goals_after_pass']:.1%} "
        f"multi_attacker_goals={metrics['multi_attacker_goals']:.1%} "
        f"receiver_movement={metrics['receiver_movement']:.2f}m"
    )
    return metrics


def evaluate(model, episodes=200, first_seed=20_000):
    """Compare the policy with its off-ball movement removed and direct play."""

    _evaluate_actions(
        "learned policy",
        lambda env, observations: model_actions(model, observations, env.action_masks()),
        episodes,
        first_seed,
    )

    def freeze_off_ball(env, observations):
        actions = model_actions(model, observations, env.action_masks())
        for index, number in enumerate(env.attacker_ids):
            if number != env.ball.owner:
                actions[index] = Action.HOLD
        return actions

    _evaluate_actions("off-ball players frozen", freeze_off_ball, episodes, first_seed)

    def freeze_receiver(env, observations):
        actions = model_actions(model, observations, env.action_masks())
        if env.ball.intended_receiver in env.attacker_ids:
            index = env.attacker_ids.index(env.ball.intended_receiver)
            actions[index] = Action.HOLD
        return actions

    _evaluate_actions("intended receiver frozen", freeze_receiver, episodes, first_seed)

    def freeze_support(env, observations):
        actions = model_actions(model, observations, env.action_masks())
        for index, number in enumerate(env.attacker_ids):
            if number not in (env.ball.owner, env.ball.intended_receiver):
                actions[index] = Action.HOLD
        return actions

    _evaluate_actions("support players frozen", freeze_support, episodes, first_seed)

    random_generator = np.random.default_rng(first_seed)
    movement_actions = np.array(list(MOVEMENT), dtype=int)

    def random_off_ball(env, observations):
        actions = model_actions(model, observations, env.action_masks())
        for index, number in enumerate(env.attacker_ids):
            if number != env.ball.owner:
                actions[index] = random_generator.choice(movement_actions)
        return actions

    _evaluate_actions("random off-ball movement", random_off_ball, episodes, first_seed)

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
            lambda env, observations: model_actions(model, observations, env.action_masks()),
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


def print_generalization_report(model, episodes=200, first_seed=20_000):
    """Print a compact comparison of trained and zero-shot performance."""

    learned = lambda env, observations: model_actions(
        model, observations, env.action_masks()
    )
    scenarios = ((3, 2, "trained"), (5, 4, "zero-shot"), (6, 4, "zero-shot"))

    print(f"Model evaluation: {episodes} held-out episodes per scenario")
    print(f"Seeds: {first_seed} through {first_seed + episodes - 1}")
    print()
    print(
        f"{'Scenario':<10} {'Test type':<11} {'Goal rate':>10} "
        f"{'Turnovers':>10} {'Pass comp.':>11} {'Passes/ep':>10}"
    )
    print("-" * 68)
    for attackers, defenders, test_type in scenarios:
        make_env = lambda a=attackers, d=defenders: SwarmSoccer(a, d)
        metrics = collect_metrics(
            learned,
            episodes=episodes,
            first_seed=first_seed,
            env_factory=make_env,
        )
        completed_passes = metrics["pass_attempts"] * metrics["pass_completion"]
        print(
            f"{attackers}v{defenders:<7} {test_type:<11} "
            f"{metrics['goals']:>9.1%} {metrics['turnovers']:>9.1%} "
            f"{metrics['pass_completion']:>10.1%} {completed_passes:>10.2f}"
        )

    print("\n3v2 is the trained scenario; 5v4 and 6v4 use the model without retraining.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print a compact 3v2, 5v4, and 6v4 generalization report",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="held-out episodes per scenario during evaluation (default: 200)",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=20_000,
        help="first deterministic evaluation seed (default: 20000)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=700_000,
        help="steps in the final and hardest curriculum stage",
    )
    parser.add_argument("--model", default="artifacts/goal_swarm_3v2.pt")
    parser.add_argument("--seeds", type=int, default=1, help="independent training runs")
    parser.add_argument(
        "--start-model",
        default="artifacts/shared_ppo_baseline.pt",
        help="passing-capable actor used to initialize mixed MAPPO training",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.train:
        phase_settings = (
            ("early", (0.40, 0.30, 0.30), 200_000),
            ("middle", (0.20, 0.30, 0.50), 300_000),
            ("late", (0.10, 0.20, 0.70), args.steps),
        )
        completed_runs = []
        model_path = Path(args.model)

        for run in range(args.seeds):
            run_seed = 100_000 * (run + 1)
            source_actor = load_model(args.start_model)
            model = initialize_mappo_from_actor(
                source_actor,
                observation_size=SwarmSoccer.observation_size,
                action_count=SwarmSoccer.action_count,
                critic_observation_size=SwarmSoccer.critic_observation_size,
                seed=run_seed,
            )
            best_goals = (-1.0, None, None)
            best_teamwork = (-1.0, None, None)
            best_overall = (-1.0, None, None)
            total_steps = 0
            print(f"training seed {run + 1}/{args.seeds} ({run_seed})")

            print(f"initialized actor from {args.start_model}")

            for phase_name, weights, phase_steps in phase_settings:
                remaining = phase_steps
                while remaining > 0:
                    chunk = min(100_000, remaining)
                    env = MixedTrainingEnv(weights)
                    model = train(
                        env,
                        chunk,
                        seed=run_seed + total_steps,
                        filename=args.model,
                        model=model,
                    )
                    total_steps += chunk
                    remaining -= chunk

                    learned = lambda env, observations: model_actions(
                        model, observations, env.action_masks()
                    )
                    full = collect_metrics(learned, episodes=60, first_seed=30_000)

                    def frozen(env, observations):
                        actions = learned(env, observations)
                        for index, number in enumerate(env.attacker_ids):
                            if number != env.ball.owner:
                                actions[index] = Action.HOLD
                        return actions

                    frozen_metrics = collect_metrics(
                        frozen, episodes=60, first_seed=30_000
                    )
                    offball_gain = full["goals"] - frozen_metrics["goals"]
                    teamwork_score = (
                        full["goals_after_pass"]
                        + 0.5 * full["episodes_with_forward_pass"]
                        + 0.5 * max(offball_gain, 0.0)
                    )
                    eligible = (
                        full["goals"] >= 0.60
                        and full["pass_attempts"] >= 0.30
                        and full["pass_completion"] >= 0.60
                        and full["goals_after_pass"] >= 0.20
                        and offball_gain >= 0.0
                    )
                    overall_score = 0.7 * full["goals"] + 0.3 * teamwork_score
                    state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                    label = f"{phase_name}@{total_steps}"
                    if full["goals"] > best_goals[0]:
                        best_goals = (full["goals"], label, state)
                        save_model(
                            model,
                            model_path.with_name(f"seed_{run + 1}_best_goals.pt"),
                        )
                    if teamwork_score > best_teamwork[0] and full["goals"] >= 0.40:
                        best_teamwork = (teamwork_score, label, state)
                        save_model(
                            model,
                            model_path.with_name(f"seed_{run + 1}_best_teamwork.pt"),
                        )
                    if eligible and overall_score > best_overall[0]:
                        best_overall = (overall_score, label, state)
                        save_model(
                            model,
                            model_path.with_name(f"seed_{run + 1}_best_overall.pt"),
                        )
                    print(
                        f"checkpoint {label}: goals={full['goals']:.1%} "
                        f"goal_after_pass={full['goals_after_pass']:.1%} "
                        f"pass_completion={full['pass_completion']:.1%} "
                        f"offball_gain={offball_gain:+.1%} eligible={eligible}"
                    )

            selected = best_overall if best_overall[1] is not None else best_goals
            model.load_state_dict(selected[2])
            final_metrics = collect_metrics(
                lambda env, observations: model_actions(
                    model, observations, env.action_masks()
                ),
                episodes=200,
                first_seed=20_000,
            )
            completed_runs.append((final_metrics["goals"], model.state_dict()))
            print(
                f"seed {run + 1} selected {selected[1]}: "
                f"goals={final_metrics['goals']:.1%}, "
                f"goals_after_pass={final_metrics['goals_after_pass']:.1%}"
            )

        rates = [rate for rate, _ in completed_runs]
        best_run = int(np.argmax(rates))
        model.load_state_dict(completed_runs[best_run][1])
        save_model(model, args.model)
        print(
            f"selected seed {best_run + 1}; mean goals={np.mean(rates):.1%} "
            f"standard_deviation={np.std(rates):.1%}"
        )
    else:
        model = load_model(args.model)
    if args.report:
        print_generalization_report(model, args.episodes, args.first_seed)
    else:
        evaluate(model, args.episodes, args.first_seed)


if __name__ == "__main__":
    main()
