"""재학습한 2·4번 정책을 실제 상태를 보며 재생한다. 시간표 입력은 사용하지 않는다."""
import argparse
from pathlib import Path
from .five_train import read
from .milestone_env import MilestoneEnv
from .milestone_models import ALGORITHMS
from .recovery_train import sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--stage',type=int,choices=(2,4),required=True)
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--gif',type=Path)
    a=p.parse_args()
    if a.checkpoint is None:
        if a.root is None:p.error('--root 또는 --checkpoint가 필요합니다.')
        a.checkpoint=Path(read(a.root/'deployment.json')['maps'][str(a.stage)]['checkpoint'])
    config=read(a.checkpoint.parent/'config.json')
    repeat=config['settings']['repeat']
    from . import five_watch
    five_watch.FiveEnv=lambda stage,reward_mode:MilestoneEnv(stage=stage,repeat=repeat)
    five_watch.ALGORITHMS,five_watch.sources=ALGORITHMS,sources
    five_watch.watch(a.checkpoint,a.stage,a.gif,sampled=not a.greedy,seed=a.seed)


if __name__=='__main__':main()
