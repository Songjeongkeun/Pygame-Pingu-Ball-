"""동일한 별도 교사 검증을 진행 중인 최종 시험과 병렬로 수행한다."""
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
    # 오래 기다리는 정책부터 검사한다. 모든 전문가는 이미 가중치 갱신을 마친 상태다.
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
    # 주 실행기가 모델 선택을 시작했으면 해당 과정에 개입하지 않는다.
    # 그렇지 않으면 동일한 검증 실패 판정을 원자적으로 기록해 같은 검증을
    # 두 번 실행하지 않는다. 통합에 성공할 수 있는 경우를 건너뛰지는 않는다.
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
        # 이동할 두 절대 경로가 모두 현재 실험 작업 폴더 안에 있는지 확인한다.
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
