"""The shared neural network and its PPO training loop.

All attackers use this one network. Each attacker gives it a different
observation, so they can still choose different actions.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class SharedPolicy(nn.Module):
    """A decentralized actor and a centralized MAPPO critic."""

    def __init__(self, observation_size=48, action_count=13, critic_observation_size=126):
        super().__init__()
        self.critic_observation_size = critic_observation_size
        self.body = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_count)  # Which action should I take?
        self.critic_body = nn.Sequential(
            nn.Linear(critic_observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.critic = nn.Linear(64, 1)  # How promising is the full situation?

    def forward(self, observations, critic_observations=None):
        features = self.body(observations)
        logits = self.actor(features)
        if critic_observations is None:
            values = torch.zeros(observations.shape[:-1], device=observations.device)
        else:
            critic_features = self.critic_body(critic_observations)
            values = self.critic(critic_features).squeeze(-1)
        return logits, values

    def choose(self, observations, actions=None, masks=None, critic_observations=None):
        logits, values = self(observations, critic_observations)
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e9)
        choices = Categorical(logits=logits)
        actions = choices.sample() if actions is None else actions
        return actions, choices.log_prob(actions), choices.entropy(), values


class LegacySharedPolicy(nn.Module):
    """Load the saved pre-MAPPO baseline for comparison only."""

    def __init__(self, observation_size, action_count):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_count)
        self.critic = nn.Linear(64, 1)

    def forward(self, observations):
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


def save_model(model, filename="artifacts/swarm.pt"):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation_size": int(model.body[0].in_features),
            "action_size": int(model.actor.out_features),
            "critic_observation_size": model.critic_observation_size,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(filename="artifacts/swarm.pt"):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
    if "critic_observation_size" in checkpoint:
        model = SharedPolicy(
            checkpoint["observation_size"],
            checkpoint["action_size"],
            checkpoint["critic_observation_size"],
        )
    else:
        model = LegacySharedPolicy(checkpoint["observation_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def train(
    env,
    total_steps=500_000,
    seed=4,
    filename="artifacts/swarm.pt",
    model=None,
):
    """Train one policy from experience collected by every attacker."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if model is None:
        model = SharedPolicy(
            env.observation_size,
            env.action_count,
            env.critic_observation_size,
        )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    observations = env.reset(seed)
    agent_count = len(env.attacker_ids)
    episode_number = 0
    successes = 0

    # PPO learns from one rollout at a time.
    for rollout_start in range(0, total_steps, 1024):
        count = min(1024, total_steps - rollout_start)
        saved_observations = np.zeros((count, agent_count, env.observation_size), np.float32)
        saved_critic_observations = np.zeros(
            (count, agent_count, env.critic_observation_size), np.float32
        )
        saved_actions = np.zeros((count, agent_count), np.int64)
        saved_log_probs = np.zeros((count, agent_count), np.float32)
        saved_values = np.zeros((count, agent_count), np.float32)
        saved_masks = np.zeros((count, agent_count, env.action_count), bool)
        saved_rewards = np.zeros(count, np.float32)
        saved_dones = np.zeros(count, np.float32)

        for step in range(count):
            saved_observations[step] = observations
            critic_observations = env.critic_observations()
            saved_critic_observations[step] = critic_observations
            masks = env.action_masks()
            with torch.no_grad():
                actions, log_probs, _, values = model.choose(
                    torch.from_numpy(observations),
                    masks=torch.from_numpy(masks),
                    critic_observations=torch.from_numpy(critic_observations),
                )

            observations, reward, done, result = env.step(actions.numpy())
            saved_actions[step] = actions.numpy()
            saved_log_probs[step] = log_probs.numpy()
            saved_values[step] = values.numpy()
            saved_masks[step] = masks
            saved_rewards[step] = reward
            saved_dones[step] = done

            if done:
                successes += result == "success"
                episode_number += 1
                observations = env.reset(seed + episode_number)

        with torch.no_grad():
            _, next_values = model(
                torch.from_numpy(observations),
                torch.from_numpy(env.critic_observations()),
            )

        advantages = _advantages(
            saved_rewards, saved_values, saved_dones, next_values.numpy()
        )
        returns = advantages + saved_values
        _ppo_update(
            model,
            optimizer,
            saved_observations.reshape(-1, env.observation_size),
            saved_critic_observations.reshape(-1, env.critic_observation_size),
            saved_actions.reshape(-1),
            saved_log_probs.reshape(-1),
            saved_masks.reshape(-1, env.action_count),
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
    advantage = np.zeros(values.shape[1], np.float32)
    for step in reversed(range(len(rewards))):
        continuing = 1.0 - dones[step]
        following_value = next_values if step == len(rewards) - 1 else values[step + 1]
        surprise = rewards[step] + 0.99 * following_value * continuing - values[step]
        advantage = surprise + 0.99 * 0.95 * continuing * advantage
        result[step] = advantage
    return result


def _ppo_update(
    model,
    optimizer,
    observations,
    critic_observations,
    actions,
    old_logs,
    masks,
    advantages,
    returns,
):
    observations = torch.from_numpy(observations)
    critic_observations = torch.from_numpy(critic_observations)
    actions = torch.from_numpy(actions)
    old_logs = torch.from_numpy(old_logs)
    masks = torch.from_numpy(masks)
    advantages = torch.from_numpy(advantages)
    returns = torch.from_numpy(returns)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = np.arange(len(actions))

    for _ in range(8):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            _, new_logs, entropy, values = model.choose(
                observations[batch],
                actions[batch],
                masks[batch],
                critic_observations[batch],
            )
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
