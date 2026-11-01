"""Overlap identical held-out teacher validation with pending final tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import torch
from .milestone_train import install, old, read, write, sources
from .milestone_models import ALGORITHMS


def candidate(payload):
    folder,cache=map(Path,payload)
    install();torch.set_num_threads(1)
    config=read(folder/'config.json')
    live=read(folder/'live.json')
    assert config['sources']==sources() and live['decisions']>=config['decisions_budget']
    checkpoint=folder/'best.zip'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out=cache/f"{config['algorithm']}_stage_{config['stage']:02d}.json"
    if out.exists():
        result=read(out)
        assert result['checkpoint_sha256']==digest
        return result
    model=ALGORITHMS[config['algorithm']].load(checkpoint,device='cpu')
    evaluation=old.evaluate(model,[config['stage']],50,3510000,'milestone')
    assert digest==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result=dict(algorithm=config['algorithm'],stage=config['stage'],seed=config['seed'],
        checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,validation=evaluation,
        final_test_read=False,sources=sources())
    write(out,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();install()
    root=a.root.resolve();plan=read(root/'plan.json')
    assert plan['seeds']==[0] and plan['sources']==sources()
    cache=root/'prevalidation_50';cache.mkdir(exist_ok=True)
    folders=list((root/'experts').glob('*/seed_0/stage_*'))
    assert len(folders)==10
    # Long idle policies first. Every expert has already completed its updates.
    folders.sort(key=lambda f:0 if f.parent.parent.name=='a2c' and f.name in ('stage_02','stage_05') else 1)
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(candidate,(str(f),str(cache))) for f in folders]):
            results.append(future.result())
            write(cache/'progress.json',dict(completed=len(results),total=10))
            print('PREVALIDATION',len(results),results[-1]['algorithm'],results[-1]['stage'],flush=True)
    selected={}
    for stage in range(1,6):
        options=sorted([r for r in results if r['stage']==stage],key=lambda r:r['algorithm'])
        r=max(options,key=lambda r:old.validation_score(r['validation']))
        selected[str(stage)]=dict(algorithm=r['algorithm'],training_seed=r['seed'],checkpoint=r['checkpoint'],
            validation=r['validation'],passes_gate=r['validation']['min_success_rate']>=plan['gate'],
            selection_used_final_test=False)
    failed=[int(s) for s,r in selected.items() if not r['passes_gate']]
    write(cache/'selection.json',selected)
    folder=root/'consolidated/seed_0'
    # If main orchestration has started selection, leave it entirely in control.
    # Otherwise atomically publish the SAME failed gate, avoiding a second run
    # of identical validation. No successful consolidation is skipped.
    pending_final=not (root/'experts/a2c/seed_0/stage_05/summary.json').exists()
    published=False
    if failed and pending_final and not folder.exists():
        temporary=root/'prevalidated_gate_seed_0'
        assert not temporary.exists()
        temporary.mkdir()
        write(temporary/'teachers.json',selected)
        write(temporary/'completion.json',dict(status='blocked_by_teacher_quality',failed_stages=failed,
            gate=plan['gate'],training_started=False,validation_count=50,validation_seed=3510000,
            same_frozen_checkpoints=True,prevalidation_overlapped_pending_test=True))
        folder.parent.mkdir(exist_ok=True)
        # Both absolute move paths are verified inside this experiment workspace.
        assert temporary.resolve().is_relative_to(root) and folder.resolve().is_relative_to(root)
        try:
            temporary.rename(folder)
            published=True
        except FileExistsError:
            pass
    write(cache/'completion.json',dict(complete=True,gate_published=published,failed_stages=failed,
        same_protocol=True,final_test_read=False,independent_new_experiments=0))
    print('PREVALIDATION COMPLETE published=',published,flush=True)


if __name__=='__main__':main()
