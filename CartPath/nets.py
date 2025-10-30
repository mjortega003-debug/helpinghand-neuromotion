# nets.py
import math
from typing import Tuple, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], activation=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [layer_init(nn.Linear(last, h)), activation()]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Minimal discrete-action actor–critic:
      - Shared MLP body
      - Policy head -> logits for Categorical
      - Value head  -> scalar state-value
    Designed for CartPole (obs_dim=4, act_dim=2).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        self.body = MLP(obs_dim, hidden, activation=nn.Tanh)
        self.pi = layer_init(nn.Linear(self.body.out_dim, act_dim), std=0.01)   # small init keeps early logits sane
        self.v  = layer_init(nn.Linear(self.body.out_dim, 1),      std=1.0)

    def _dist(self, obs: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions for discrete control.
        """
        z = self.body(obs)
        logits = self.pi(z)
        return Categorical(logits=logits)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Convenience forward: (action_distribution, state_value)
        """
        z = self.body(obs)
        dist = Categorical(logits=self.pi(z))
        value = self.v(z).squeeze(-1)  # (batch,)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action given a batch of observations.
        Returns: action, log_prob, value
        Shapes:
          obs: (batch, obs_dim)
          action: (batch,)
          log_prob: (batch,)
          value: (batch,)
        """
        dist, value = self.forward(obs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during PPO updates:
          - log probs of provided actions
          - entropy bonus
          - state values
        """
        dist, value = self.forward(obs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value
