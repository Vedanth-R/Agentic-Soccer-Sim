"""The shared neural network and its PPO training loop.

All three attackers use this one network. Each attacker gives it a different
observation, so they can still choose different actions.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class SharedPolicy(nn.Module):
    """Two small hidden layers followed by an actor and a critic."""

    def __init__(self, observation_size=16, action_count=9):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_count)  # Which action should I take?
        self.critic = nn.Linear(64, 1)  # How promising is this situation?

    def forward(self, observations):
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def choose(self, observations, actions=None):
        logits, values = self(observations)
        choices = Categorical(logits=logits)
        actions = choices.sample() if actions is None else actions
        return actions, choices.log_prob(actions), choices.entropy(), values


def save_model(model, filename="artifacts/swarm.pt"):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation_size": int(model.body[0].in_features),
            "action_size": int(model.actor.out_features),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(filename="artifacts/swarm.pt"):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
    model = SharedPolicy(checkpoint["observation_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def train(
    env,
    total_steps=500_000,
    seed=4,
    filename="artifacts/swarm.pt",
    model=None,
):
    """Train one policy from experience collected by all three attackers."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if model is None:
        model = SharedPolicy(env.observation_size, env.action_count)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    observations = env.reset(seed)
    episode_number = 0
    successes = 0

    # PPO learns from one rollout at a time.
    for rollout_start in range(0, total_steps, 1024):
        count = min(1024, total_steps - rollout_start)
        saved_observations = np.zeros((count, 3, env.observation_size), np.float32)
        saved_actions = np.zeros((count, 3), np.int64)
        saved_log_probs = np.zeros((count, 3), np.float32)
        saved_values = np.zeros((count, 3), np.float32)
        saved_rewards = np.zeros(count, np.float32)
        saved_dones = np.zeros(count, np.float32)

        for step in range(count):
            saved_observations[step] = observations
            with torch.no_grad():
                actions, log_probs, _, values = model.choose(torch.from_numpy(observations))

            observations, reward, done, result = env.step(actions.numpy())
            saved_actions[step] = actions.numpy()
            saved_log_probs[step] = log_probs.numpy()
            saved_values[step] = values.numpy()
            saved_rewards[step] = reward
            saved_dones[step] = done

            if done:
                successes += result == "success"
                episode_number += 1
                observations = env.reset(seed + episode_number)

        with torch.no_grad():
            _, next_values = model(torch.from_numpy(observations))

        advantages = _advantages(
            saved_rewards, saved_values, saved_dones, next_values.numpy()
        )
        returns = advantages + saved_values
        _ppo_update(
            model,
            optimizer,
            saved_observations.reshape(-1, env.observation_size),
            saved_actions.reshape(-1),
            saved_log_probs.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
        )

    save_model(model, filename)
    rate = successes / max(episode_number, 1)
    print(f"training steps={total_steps} episodes={episode_number} success_rate={rate:.1%}")
    print(f"saved {filename}")
    return model


def _advantages(rewards, values, dones, next_values):
    """Estimate how much better each action was than the critic expected."""

    result = np.zeros_like(values)
    advantage = np.zeros(3, np.float32)
    for step in reversed(range(len(rewards))):
        continuing = 1.0 - dones[step]
        following_value = next_values if step == len(rewards) - 1 else values[step + 1]
        surprise = rewards[step] + 0.99 * following_value * continuing - values[step]
        advantage = surprise + 0.99 * 0.95 * continuing * advantage
        result[step] = advantage
    return result


def _ppo_update(model, optimizer, observations, actions, old_logs, advantages, returns):
    observations = torch.from_numpy(observations)
    actions = torch.from_numpy(actions)
    old_logs = torch.from_numpy(old_logs)
    advantages = torch.from_numpy(advantages)
    returns = torch.from_numpy(returns)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = np.arange(len(actions))

    for _ in range(8):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            _, new_logs, entropy, values = model.choose(observations[batch], actions[batch])
            probability_change = (new_logs - old_logs[batch]).exp()
            normal_gain = probability_change * advantages[batch]
            clipped_gain = torch.clamp(probability_change, 0.8, 1.2) * advantages[batch]
            actor_loss = -torch.min(normal_gain, clipped_gain).mean()
            critic_loss = 0.5 * (values - returns[batch]).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.015 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
