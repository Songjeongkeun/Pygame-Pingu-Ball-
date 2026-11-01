"""Render the user-reward policy against live native game state."""
import argparse
from pathlib import Path
from .milestone_train import install, sources
from .milestone_env import MilestoneEnv
from .milestone_models import ALGORITHMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--stage',type=int,choices=range(1,6),required=True)
    p.add_argument('--gif',type=Path)
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--seed',type=int,default=20261030)
    a=p.parse_args()
    install()
    from . import five_watch
    five_watch.FiveEnv, five_watch.ALGORITHMS, five_watch.sources=MilestoneEnv,ALGORITHMS,sources
    five_watch.watch(a.checkpoint,a.stage,a.gif,sampled=not a.greedy,seed=a.seed)


if __name__=='__main__': main()
