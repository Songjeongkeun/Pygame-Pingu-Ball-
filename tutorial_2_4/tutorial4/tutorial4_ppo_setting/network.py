from __future__ import annotations
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
        )
        self.actor = nn.Linear(width, action_dim)
        self.critic = nn.Linear(width, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def action_distribution(logits, *, greedy_mix=0.05):
    """Behavior policy: (1-greedy_mix)*pi + greedy_mix*one_hot(argmax(pi)).

    This exactly represents stochastic choice 95% and greedy 5% on EACH
    decision. Use the same distribution for collection AND PPO ratios.
    """
    if not (0.0 <= greedy_mix <= 1.0):
        raise ValueError('greedy_mix must be in [0, 1]')
    probs = torch.softmax(logits, dim=-1)
    if greedy_mix:
        best = torch.argmax(logits, dim=-1)
        one_hot = torch.nn.functional.one_hot(best, num_classes=logits.shape[-1]).to(probs.dtype)
        probs = (1. - greedy_mix) * probs + greedy_mix * one_hot
    return torch.distributions.Categorical(probs=probs)


def action_sample(net, obs, *, mode='mixed', greedy_mix=0.05):
    with torch.no_grad():
        logits, value = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if mode == 'mixed':
            dist = action_distribution(logits, greedy_mix=greedy_mix)
            action = dist.sample()
        elif mode == 'stochastic':
            dist = action_distribution(logits, greedy_mix=0.0)
            action = dist.sample()
        elif mode == 'greedy':
            dist = action_distribution(logits, greedy_mix=0.0)
            action = torch.argmax(logits, dim=-1)
        else:
            raise ValueError('unknown action mode')
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())
