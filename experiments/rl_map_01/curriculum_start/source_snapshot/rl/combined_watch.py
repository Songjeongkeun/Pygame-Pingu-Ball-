"""통합 보상으로 학습한 맵별 정책을 실제 게임 화면 또는 GIF로 확인한다."""
import argparse
from pathlib import Path
from .combined_train import install, sources
from .combined_env import CombinedEnv
from .milestone_models import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=range(1,6), required=True)
    parser.add_argument('--training-seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--gif', type=Path)
    args = parser.parse_args()
    checkpoint = args.root/'experts'/'ppo'/f'seed_{args.training_seed}'/f'stage_{args.stage:02d}'/'best.zip'
    install()
    from . import five_watch
    five_watch.FiveEnv, five_watch.ALGORITHMS, five_watch.sources = CombinedEnv, ALGORITHMS, sources
    five_watch.watch(checkpoint,args.stage,args.gif,sampled=not args.greedy,seed=args.seed)


if __name__ == '__main__':
    main()
