"""Play the final selected shared policy, or explicitly selected map experts."""
import argparse
from pathlib import Path
import secrets
from .milestone_train import read, sources, install
from .milestone_env import MilestoneEnv
from .milestone_models import ALGORITHMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',type=int,choices=range(1,6),required=True)
    p.add_argument('--seed',type=int)
    a=p.parse_args()
    descriptor=a.root/'deployment.json'
    if not descriptor.exists():
        raise SystemExit('Evaluation is not finished yet. See training_console.txt for progress.')
    selected=read(descriptor)
    print('Policy kind:',selected['policy_kind'],flush=True)
    print('Execution: sampling learned action probabilities; this can fail.',flush=True)
    install()
    from . import five_watch
    five_watch.FiveEnv,five_watch.ALGORITHMS,five_watch.sources=MilestoneEnv,ALGORITHMS,sources
    seed=secrets.randbits(32) if a.seed is None else a.seed
    print('Action sampling seed:',seed,flush=True)
    five_watch.watch(selected['maps'][str(a.stage)]['checkpoint'],a.stage,sampled=True,seed=seed)


if __name__=='__main__': main()
