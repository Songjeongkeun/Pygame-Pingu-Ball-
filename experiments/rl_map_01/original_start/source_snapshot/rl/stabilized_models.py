"""기존 맵 인코더에 보상 추적 상태 인코더를 연결한 9행동 PPO."""
import torch
from torch import nn
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .milestone_models import MilestoneExtractor, MilestonePPO
from .stabilized_env import BASE_SIZE, EXTRA_SIZE


class StabilizedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=224)
        self.base = MilestoneExtractor(spaces.Box(-8., 8., (BASE_SIZE,), dtype=np.float32))
        self.extra_net = nn.Sequential(nn.Linear(EXTRA_SIZE, 32), nn.Tanh())

    def forward(self, observations):
        return torch.cat([self.base(observations[:, :BASE_SIZE]),
                          self.extra_net(observations[:, BASE_SIZE:])], dim=1)


class StabilizedPPO(MilestonePPO):
    pass


class FixedRate:
    """실행 진입점과 무관하게 복원할 수 있는 수치 학습률 함수."""
    def __init__(self, value): self.value = float(value)
    def __call__(self, remaining): return self.value
