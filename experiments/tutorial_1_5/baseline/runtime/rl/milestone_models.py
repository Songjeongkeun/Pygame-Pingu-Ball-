"""Same object encoder as previous experiment, with explicit milestone state."""
import torch
from torch import nn
from .five_models import FiveExtractor, FiveA2C, FivePPO
from .milestone_env import GLOBALS, CAPACITY, FIELDS, LOCAL_COUNT


class MilestoneExtractor(FiveExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self.global_net = nn.Sequential(nn.Linear(GLOBALS, 64), nn.Tanh())

    def forward(self, obs):
        end = GLOBALS+CAPACITY*FIELDS
        entities = obs[:, GLOBALS:end].reshape(-1, CAPACITY, FIELDS)
        mask = entities[:, :, 17:18]
        encoded = self.entity_net(entities)
        mean = (encoded*mask).sum(1)/mask.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(mask == 0, -1e6).max(1).values
        maximum = torch.where(mask.sum(1)>0, maximum, torch.zeros_like(maximum))
        local = obs[:, end:].reshape(-1, LOCAL_COUNT, FIELDS)
        local = local*local[:, :, 17:18]
        return torch.cat([self.global_net(obs[:, :GLOBALS]), self.full_net(torch.cat([mean, maximum], 1)),
                          self.local_net(local.flatten(1))], 1)


class MilestoneA2C(FiveA2C):
    pass


class MilestonePPO(FivePPO):
    pass


ALGORITHMS = {'a2c':MilestoneA2C, 'ppo':MilestonePPO}
