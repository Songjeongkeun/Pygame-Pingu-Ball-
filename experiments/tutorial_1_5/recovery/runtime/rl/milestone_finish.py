"""고정된 학습이 완료된 뒤 남아 있는 보고서와 검증 작업을 마무리한다."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import torch
import numpy as np
from .milestone_train import install, read, write, sources
from .milestone_models import ALGORITHMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    root=a.root.resolve()
    if a.wait:
        started=time.monotonic()
        while not (root/'completion.json').exists():
            if time.monotonic()-started>21600:
                raise TimeoutError('Training did not complete in six hours; outputs retained')
            console=root.parent/'training_console.txt'
            if console.exists() and 'Traceback (most recent call last)' in console.read_text(encoding='utf8',errors='replace'):
                raise RuntimeError('Training failed. Inspect training_console.txt; no reports labelled final.')
            time.sleep(10)
    completed=read(root/'completion.json')
    install()
    assert completed['complete'] and completed['sources']==sources()
    from . import milestone_report
    sys.argv=['milestone_report','--root',str(root),'--verify','--workers','3']
    milestone_report.main()
    torch.set_num_threads(1)
    plan=read(root/'plan.json')
    parity=[]
    for seed in plan['seeds']:
        for stage in range(1,6):
            models=[ALGORITHMS[alg].load(root/'experts'/alg/f'seed_{seed}'/f'stage_{stage:02d}'/'initial.zip',device='cpu')
                    for alg in ('a2c','ppo')]
            same=all(torch.equal(x,y) for x,y in zip(models[0].policy.parameters(),models[1].policy.parameters()))
            assert same
            for m in models:
                assert m.num_timesteps==0 and m.gamma==1. and m.learning_rate==.0003
                assert torch.equal(m.policy.action_net.weight,torch.zeros_like(m.policy.action_net.weight))
                assert torch.equal(m.policy.action_net.bias,torch.zeros_like(m.policy.action_net.bias))
            parity.append(dict(stage=stage,seed=seed,identical_initial_parameters=True,uniform_actions=True))
    write(root/'initialization_audit.json',dict(complete=True,pairs=parity))
    folder=root/'consolidated'/f'seed_{plan["seeds"][0]}'
    status=read(folder/'completion.json')
    if status['status']=='completed':
        mappings={str(s):dict(algorithm='ppo',checkpoint=str(folder/'best.zip')) for s in range(1,6)}
        kind='single_shared_policy'
    else:
        mappings=read(folder/'teachers.json')
        kind='separate_map_experts_shared_policy_not_trained'
    write(root/'deployment.json',dict(policy_kind=kind,maps=mappings,execution_mode='sampled',
        selected_from_validation=True,final_test_used_for_selection=False,consolidation_status=status))
    audit=read(root/'native_verification.json')
    assert audit['complete'] and audit['max_ball_error']==0.
    summaries=list((root/'experts').rglob('summary.json'))
    assert len(summaries)==10*len(plan['seeds'])
    for path in summaries:
        d=read(path)
        assert d['config']['sources']==d['sources_after']==sources()
        assert d['config']['from_scratch'] and not d['config']['previous_models_loaded']
    write(root/'finalization.json',dict(complete=True,scope=[1,2,3,4,5],expert_models=len(summaries),
        expert_action_decisions=sum(read(p)['decisions'] for p in summaries),gamma=1.,learning_rate=.0003,
        original_physics_max_error=0.,sources=sources(),consolidation_status=status,
        shared_policy_available=kind=='single_shared_policy',learning_seed_count=len(plan['seeds']),
        validation_selection_not_test=True,git_actions=False))
    manifest=read(root.parent/'runtime_manifest.json')
    manifest['analysis_additions']={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path('rl').glob('milestone_*.py') if str(p).replace('\\','/') not in plan['sources']}
    write(root.parent/'runtime_manifest.json',manifest)
    print('FINALIZATION COMPLETE',kind,flush=True)


if __name__=='__main__':main()
