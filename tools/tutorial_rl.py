"""정리된 튜토리얼 1~5 실험을 검증·평가·재생한다. 기존 모델을 재학습하거나 덮어쓰지 않는다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'experiments/tutorial_1_5'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def models():
    return read(ARCHIVE/'models.json')['models']


def check_model(entry):
    """원래 설정의 소스 해시와 선택 모델의 해시를 그대로 검사한다."""
    checkpoint=ARCHIVE/entry['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError(f'Model hash mismatch: {entry["id"]}')
    config=read(checkpoint.with_name('config.json'))
    for name,digest in config['sources'].items():
        source=ARCHIVE/entry['runtime']/Path(name.replace('\\','/'))
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Training source mismatch: {entry["id"]}: {name}')
    return config


def verify():
    manifest=read(ARCHIVE/'archive_manifest.json')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ARCHIVE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Archive hash mismatch: {name}')
    for entry in models():
        check_model(entry)
        with zipfile.ZipFile(ARCHIVE/entry['checkpoint']) as saved:
            if saved.testzip() is not None:raise ValueError(f'Invalid model archive: {entry["id"]}')
    print(f'검증 완료: 파일 {len(manifest["files"])}개, 모델 {len(models())}개, 학습 소스 해시 일치')


def select(args, parser):
    choices=[m for m in models() if m['experiment']==args.experiment
             and m['stage']==args.stage and m['algorithm']==args.algorithm]
    if args.experiment=='recovery':choices=[m for m in choices if m['variant']==args.variant]
    seed=args.training_seed if args.training_seed is not None else (0 if args.experiment=='baseline' else 2)
    choices=[m for m in choices if m['training_seed']==seed]
    if len(choices)!=1:parser.error('해당 실험·맵·알고리즘·학습 시드 조합의 모델이 없습니다. list로 확인하세요.')
    return choices[0]


def launch(action, experiment, entry=None, **kwargs):
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'_worker',
             '--action',action,'--experiment',experiment]
    if entry:command+=['--id',entry['id']]
    for key,value in kwargs.items():
        if value is not None:command+=['--'+key,str(value)]
    subprocess.run(command,check=True,cwd=ROOT)


def worker(args):
    entries=[m for m in models() if m['experiment']==args.experiment
             and (args.id is None or m['id']==args.id)]
    if not entries:raise ValueError('Unknown archived model')
    runtime=ARCHIVE/entries[0]['runtime']
    sys.path.insert(0,str(runtime))
    os.chdir(runtime)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    import numpy as np
    import torch
    from rl.milestone_train import install
    from rl.milestone_models import ALGORITHMS
    from rl.milestone_env import MilestoneEnv
    from rl.five_models import action_probs
    from pingu_ball.engine import World
    torch.set_num_threads(1)
    install()
    results=[]
    for entry in entries:
        config=check_model(entry)
        checkpoint=ARCHIVE/entry['checkpoint']
        if args.action=='play':
            module='rl.milestone_watch' if args.experiment=='baseline' else 'rl.recovery_watch'
            command=[sys.executable,'-B','-m',module,'--checkpoint',str(checkpoint),
                     '--stage',str(entry['stage']),'--seed',str(args.seed)]
            if args.greedy:command+=['--greedy']
            subprocess.run(command,check=True,cwd=runtime)
            return
        model=ALGORITHMS[entry['algorithm']].load(checkpoint,device='cpu')
        if args.action=='evaluate':
            if args.experiment=='baseline':
                from rl import five_train
                result=five_train.evaluate(model,[entry['stage']],args.episodes,args.seed,'milestone')
                summary={k:v for k,v in result['maps'][0].items() if k!='raw'}
                summary['mode']='sampled'
            else:
                from rl.recovery_train import evaluate
                result=evaluate(model,entry['stage'],entry['repeat'],args.episodes,args.seed)
                summary={k:v for k,v in result.items() if k!='raw'}
            results.append(dict(model=entry['id'],evaluation_seed=args.seed,**summary))
        else:
            # 모델의 관측 규격과 실제 원래 물리 궤적을 작은 입력 표본으로 대조한다.
            env=MilestoneEnv(stage=entry['stage'],repeat=entry['repeat'])
            raw=env.unwrapped
            observation,_=env.reset(seed=7)
            native=World(raw.level,auto_reset=False)
            error=0.
            def on_tick(world,tick,direction):
                nonlocal error
                native.step(direction,min(native.physics.dt,180.-native.time))
                error=max(error,*(abs(getattr(world.ball,k)-getattr(native.ball,k))
                                  for k in ('x','y','vx','vy')))
            rng=np.random.default_rng(7)
            decisions=0
            try:
                for _ in range(20):
                    probability=action_probs(model,observation[None])[0]
                    if not np.all(np.isfinite(probability)):raise ValueError('Invalid action probabilities')
                    raw.step(int(rng.choice(3,p=probability/probability.sum())),on_tick=on_tick)
                    decisions+=1
                    observation=env.observation(raw.observe())
                    if raw.done:break
            finally:env.close()
            if error>1e-7:raise ValueError(f'Native physics mismatch: {entry["id"]}: {error}')
            results.append(dict(model=entry['id'],loaded=True,decisions=decisions,max_native_error=error))
    payload=dict(experiment=args.experiment,action=args.action,results=results)
    if args.output:
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='선택 모델과 기록된 독립 시험 성공률')
    sub.add_parser('verify',help='전체 파일·모델·학습 당시 소스의 해시 검사')
    smoke=sub.add_parser('smoke',help='28개 모델 로드 및 원래 물리 궤적 검사')
    smoke.add_argument('--output-dir',type=Path)
    for action in ('play','evaluate'):
        p=sub.add_parser(action,help='기존 모델 재생' if action=='play' else '기존 모델 추가 평가')
        p.add_argument('--experiment',choices=('baseline','recovery'),default='baseline')
        p.add_argument('--stage',type=int,choices=range(1,6),required=True)
        p.add_argument('--algorithm',choices=('a2c','ppo'),default='ppo')
        p.add_argument('--variant',choices=('stable','long','hold'),default='hold')
        p.add_argument('--training-seed',type=int)
        p.add_argument('--seed',type=int,default=20261101)
        if action=='play':p.add_argument('--greedy',action='store_true')
        else:
            p.add_argument('--episodes',type=int,default=5)
            p.add_argument('--output',type=Path)
    p=sub.add_parser('_worker',help=argparse.SUPPRESS)
    p.add_argument('--action',choices=('play','evaluate','smoke'),required=True)
    p.add_argument('--experiment',choices=('baseline','recovery'),required=True)
    p.add_argument('--id')
    p.add_argument('--seed',type=int,default=20261101)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--output')
    args=parser.parse_args()
    if args.command=='_worker':return worker(args)
    if args.command=='list':
        for entry in models():print(f'{entry["id"]}: {entry["successes"]}/{entry["episodes"]}')
    elif args.command=='verify':verify()
    elif args.command=='smoke':
        for experiment in ('baseline','recovery'):
            output=str((args.output_dir/f'{experiment}.json').resolve()) if args.output_dir else None
            launch('smoke',experiment,output=output)
    else:
        if args.command=='evaluate' and args.episodes<=0:parser.error('episodes는 양수여야 합니다.')
        entry=select(args,parser)
        options=dict(seed=args.seed)
        if args.command=='evaluate':
            options.update(episodes=args.episodes,output=str(args.output.resolve()) if args.output else None)
        elif args.greedy:
            # 내부 작업자의 불리언 인수는 별도 값 없이 전달한다.
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'_worker','--action','play',
                     '--experiment',args.experiment,'--id',entry['id'],'--seed',str(args.seed),'--greedy']
            return subprocess.run(command,check=True,cwd=ROOT)
        launch(args.command,args.experiment,entry,**options)


if __name__=='__main__':main()
