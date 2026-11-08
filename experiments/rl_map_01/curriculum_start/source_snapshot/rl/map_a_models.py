"""A와 같은 특징 추출·정책 구조를 확장된 맵 입력에 연결한다."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .five_models import FivePPO
from .map_a_env import GLOBALS, CAPACITY, FIELDS, LOCAL_COUNT, EXTRA_SIZE


class MapAExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space,features_dim=224)
        self.global_net = nn.Sequential(nn.Linear(GLOBALS,64),nn.Tanh())
        self.entity_net = nn.Sequential(nn.Linear(FIELDS,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.full_net = nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.local_net = nn.Sequential(nn.Linear(LOCAL_COUNT*FIELDS,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.extra_net = nn.Sequential(nn.Linear(EXTRA_SIZE,32),nn.Tanh())

    def forward(self, obs):
        end = GLOBALS+CAPACITY*FIELDS
        entities = obs[:,GLOBALS:end].reshape(-1,CAPACITY,FIELDS)
        mask = entities[:,:,17:18]
        encoded = self.entity_net(entities)
        mean = (encoded*mask).sum(1)/mask.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(mask==0,-1e6).max(1).values
        maximum = torch.where(mask.sum(1)>0,maximum,torch.zeros_like(maximum))
        local_end = end+LOCAL_COUNT*FIELDS
        local = obs[:,end:local_end].reshape(-1,LOCAL_COUNT,FIELDS)
        local = local*local[:,:,17:18]
        return torch.cat([self.global_net(obs[:,:GLOBALS]),self.full_net(torch.cat([mean,maximum],1)),
                          self.local_net(local.flatten(1)),self.extra_net(obs[:,local_end:])],1)


class MapAPPO(FivePPO):
    pass


def make_model(vec, seed):
    model = MapAPPO('MlpPolicy',vec,seed=seed,device='cpu',verbose=0,
        gamma=1.,gae_lambda=.98,n_steps=128,batch_size=256,learning_rate=.0003,
        n_epochs=5,clip_range=.2,target_kl=.02,ent_coef=.02,vf_coef=.5,max_grad_norm=.5,
        policy_kwargs=dict(features_extractor_class=MapAExtractor,net_arch=dict(pi=[128,128],vf=[128,128])))
    with torch.no_grad():
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    return model


def load_policy(path):
    return MapAPPO.load(path,device='cpu',custom_objects={'lr_schedule':None})
