"""Same architecture for A2C, PPO, and the distilled five-map student."""
import numpy as np
import torch
from torch import nn
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .five_env import GLOBALS, CAPACITY, FIELDS, LOCAL_COUNT


class FiveExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=192)
        self.global_net = nn.Sequential(nn.Linear(GLOBALS, 64), nn.Tanh())
        self.entity_net = nn.Sequential(nn.Linear(FIELDS, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.full_net = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.local_net = nn.Sequential(nn.Linear(LOCAL_COUNT*FIELDS, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())

    def forward(self, obs):
        end = GLOBALS + CAPACITY*FIELDS
        entities = obs[:, GLOBALS:end].reshape(-1, CAPACITY, FIELDS)
        mask = entities[:, :, 17:18]
        encoded = self.entity_net(entities)
        mean = (encoded*mask).sum(1)/mask.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(mask == 0, -1e6).max(1).values
        maximum = torch.where(mask.sum(1)>0, maximum, torch.zeros_like(maximum))
        local = obs[:, end:].reshape(-1, LOCAL_COUNT, FIELDS)
        local = local*local[:, :, 17:18]
        return torch.cat([self.global_net(obs[:, :GLOBALS]),
                          self.full_net(torch.cat([mean, maximum], 1)),
                          self.local_net(local.flatten(1))], 1)


class UpdateHooks:
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['after_update', 'teachers']

    def train(self):
        super().train()
        # Optional, explicitly labelled PPO + teacher-KL auxiliary update.
        if getattr(self, 'teachers', None):
            obs = torch.as_tensor(self.rollout_buffer.observations.reshape(-1, self.observation_space.shape[0]), device=self.device)
            stages = obs[:, 24:29].argmax(1)
            losses = []
            for s, teacher in self.teachers.items():
                ids = torch.where(stages == s-1)[0]
                if not len(ids):
                    raise RuntimeError('Joint rollout must contain every map')
                ids = ids[torch.randperm(len(ids), device=self.device)[:64]]
                x = obs[ids]
                with torch.no_grad():
                    target = teacher.policy.get_distribution(x).distribution.probs
                logp = self.policy.get_distribution(x).distribution.logits
                losses.append((target*(target.clamp_min(1e-8).log()-logp)).sum(1).mean())
            fraction = min(1., self.num_timesteps/self.joint_budget)
            beta = .1*(1-fraction)+.01*fraction
            loss = beta*torch.stack(losses).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
            self.policy.optimizer.step()
            self.logger.record('train/teacher_kl', float(loss.detach())/beta)
            self.logger.record('train/teacher_kl_weight', beta)
        if getattr(self, 'after_update', None):
            self.after_update(self)


class FiveA2C(UpdateHooks, A2C):
    pass


class FivePPO(UpdateHooks, PPO):
    pass


ALGORITHMS = {'a2c': FiveA2C, 'ppo': FivePPO}


def action_probs(model, observations):
    with torch.no_grad():
        return model.policy.get_distribution(torch.as_tensor(observations, device=model.device)).distribution.probs.cpu().numpy()
