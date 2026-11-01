"""팀원 공통 PPO + 재사용 가능한 개인 보상의 독립 실험 (v3).

- common: 팀원의 공통 보상만 (+100/-25/마일스톤+10).
- combined: common + 2단계 점프 정렬 0.5/0.5 + 4단계 이동 발판 12/12를
  발판 타입/마일스톤 정보로 일반화 (스테이지 번호, 고정 좌표, +18, 오른쪽 진행 보상 없음).
- --init-model: 기존 공통환경 best.zip에서 이어 학습(조건이 다르면 즉시 오류).
- 결과는 results_v3만 사용. v1/v2 최종모델과 실험기록을 덮어쓰지 않음.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.milestone_models import MilestonePPO, MilestoneExtractor
from rl.five_models import action_probs
from tutorial_2.tutorial_test.environment_v3 import RewardEnv

HERE = Path(__file__).resolve().parent
CODE_VERSION = 'portable_reward_v3_20261101'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def source_hashes():
    filenames = ('rl/milestone_env.py', 'rl/five_env.py', 'rl/map_input_env.py',
                 'rl/all_tutorial_env.py', 'rl/milestone_models.py', 'rl/five_models.py',
                 'pingu_ball/engine.py', 'pingu_ball/special_lab.py', 'pingu_ball/specials.py',
                 'tutorial_2/tutorial_test/portable_rewards_v3.py', 'tutorial_2/tutorial_test/environment_v3.py',
                 'tutorial_2/tutorial_test/train_ppo_v3.py')
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in filenames}


def make_vector(stage, reward, count=4, repeat=6):
    return DummyVecEnv([lambda s=stage, r=reward: RewardEnv(stage=s, reward=r, repeat=repeat)
                        for _ in range(count)])


def output_dir(root, stage, reward, seed, init_model=None):
    tag = 'from_common' if init_model else 'scratch'
    return Path(root) / f'stage_{stage:02d}' / f'{reward}_{tag}' / f'seed_{seed}'


def entropy(fraction):
    p = min(1., max(0., float(fraction)))
    if p <= .2:
        return .02
    if p <= .8:
        return .02 + (.005 - .02) * (p - .2) / .6
    return .005 + (.001 - .005) * (p - .8) / .2


def make_new_model(vec, seed):
    model = MilestonePPO('MlpPolicy', vec, seed=seed, device='cpu', verbose=0,
                         gamma=1., learning_rate=.0003, gae_lambda=.98,
                         ent_coef=.02, vf_coef=.5, max_grad_norm=.5,
                         n_steps=128, batch_size=256, n_epochs=5,
                         clip_range=.2, target_kl=.02,
                         policy_kwargs=dict(features_extractor_class=MilestoneExtractor,
                                            net_arch=dict(pi=[128, 128], vf=[128, 128])))
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def evaluate(model, stage, reward, *, trials, seed, mode='stochastic'):
    if trials <= 0 or mode not in ('stochastic', 'greedy'):
        raise ValueError('잘못된 평가 옵션')
    old_training = model.policy.training
    model.policy.set_training_mode(False)
    vec = make_vector(stage, reward)
    remaining = [trials // 4 + int(i < trials % 4) for i in range(4)]
    rngs = [np.random.default_rng(seed + stage * 100003 + i * 104729) for i in range(4)]
    outcomes = []
    try:
        obs = vec.reset()
        while sum(remaining):
            probs = action_probs(model, obs)
            acts = []
            for i, (p, rng) in enumerate(zip(probs, rngs)):
                if remaining[i] == 0:
                    acts.append(1)
                elif mode == 'greedy':
                    acts.append(int(p.argmax()))
                else:
                    p = p / p.sum()
                    acts.append(int(rng.choice(3, p=p)))
            obs, _, dones, infos = vec.step(np.asarray(acts, dtype=np.int64))
            for i, done in enumerate(dones):
                if done and remaining[i] > 0:
                    outcomes.append(dict(infos[i]))
                    remaining[i] -= 1
    finally:
        vec.close()
        model.policy.set_training_mode(old_training)
    wins = [r for r in outcomes if r['success']]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return dict(stage=stage, reward=reward, mode=mode, seed=seed,
                trials=trials, wins=len(wins), success_pct=100.*len(wins)/trials,
                mean_success_seconds=mean([r['seconds'] for r in wins]),
                mean_episode_seconds=mean([r['seconds'] for r in outcomes]),
                mean_milestones=mean([len(r['milestones']) for r in outcomes]),
                mean_personal_reward=mean([r.get('personal_total', 0.) for r in outcomes]),
                mean_reward=mean([r['episode_return'] for r in outcomes]),
                mean_personal_counts={k: float(np.mean([r.get('personal_counts', {}).get(k, 0)
                                                      for r in outcomes]))
                                      for k in ('spring_height', 'spring_horizontal',
                                                'moving_first', 'moving_high')},
                failures=dict(Counter(str(r['reason']) for r in outcomes if not r['success'])))


class Monitor(BaseCallback):
    def __init__(self, folder, stage, reward, start_step, additional_steps,
                 eval_every, eval_trials, include_initial):
        super().__init__()
        self.folder, self.stage, self.reward = Path(folder), stage, reward
        self.start_step, self.additional_steps = start_step, additional_steps
        self.eval_every, self.eval_trials = eval_every, eval_trials
        self.next_eval = start_step + eval_every
        self.best_score = None
        self.best_step = None
        self.last_evaluated_step = None
        self.episodes = 0
        self.include_initial = include_initial

    def _on_step(self):
        for done in self.locals['dones']:
            self.episodes += int(done)
        if self.model.num_timesteps >= self.next_eval:
            self.checkpoint()
            self.next_eval = (self.model.num_timesteps // self.eval_every + 1) * self.eval_every
        return True

    def checkpoint(self):
        ev = evaluate(self.model, self.stage, self.reward, trials=self.eval_trials,
                      seed=11000, mode='stochastic')
        seconds = ev['mean_success_seconds']
        score = (ev['wins'], ev['mean_milestones'] or 0.,
                 -(seconds if seconds is not None else 1e9))
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_step = score, self.model.num_timesteps
            self.model.save(str(self.folder / 'best'))
            write_json(self.folder/'best_validation.json',
                       {'step': self.best_step, 'evaluation': ev,
                        'selected_from_initial': self.best_step == self.start_step and self.include_initial})
        self.model.save(str(self.folder/'last'))
        row = dict(step=self.model.num_timesteps, wins=ev['wins'], trials=ev['trials'],
                   success_pct=ev['success_pct'], milestones=ev['mean_milestones'],
                   personal=ev['mean_personal_reward'], best_step=self.best_step)
        with (self.folder/'validations.csv').open('a', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        self.last_evaluated_step = self.model.num_timesteps
        print(f'[검증] {self.stage}단계 {self.reward} step={self.model.num_timesteps} '
              f'{ev["wins"]}/{ev["trials"]} ({ev["success_pct"]:.1f}%) '
              f'마일스톤={ev["mean_milestones"]:.2f} 개인보상={ev["mean_personal_reward"]:.3f}',flush=True)
        return ev


def model_from_checkpoint(filename, vec, stage, seed):
    filename = Path(filename).resolve()
    if not filename.exists():
        raise FileNotFoundError(filename)
    model = MilestonePPO.load(str(filename), env=vec, device='cpu')
    assert model.observation_space.shape == vec.observation_space.shape, '관측 차원 불일치'
    assert model.action_space.n == 3, '행동 공간 불일치'
    # 지정한 모델이 stage 2의 common 모델인지 stage 4의 common 모델인지는 config로 확인.
    config = filename.parent/'config.json'
    if config.exists():
        old = json.loads(config.read_text(encoding='utf-8'))
        if old.get('stage') != stage:
            raise ValueError('초기 모델의 튜토리얼 단계가 현재 학습 단계와 다릅니다.')
        if old.get('reward') != 'common':
            raise ValueError('공정한 초기화 실험을 위해 common 모델만 초기 모델로 허용합니다.')
    else:
        raise ValueError(f'학습 출처 확인용 config.json이 없습니다: {config}')
    # 이전 공통 모델의 학습 당시 게임 및 팀 환경 소스가 변경되었는지도 확인.
    for name, previous_digest in old.get('source_hashes', {}).items():
        if name.startswith(('rl/', 'pingu_ball/')):
            current = PROJECT_ROOT / name
            if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != previous_digest:
                raise RuntimeError('기존 공통 모델과 현재 게임/공통 환경 코드가 달라 비교 불가: ' + name)
    # 동일 초기 정책을 이용하는 대조군/실험군에서 난수를 다시 같은 시드로 설정.
    model.set_random_seed(seed)
    return model


def train(args):
    if args.steps < 1 or args.eval_every < 1 or args.eval_trials < 1 or args.test_trials < 1:
        raise ValueError('스텝과 평가 횟수는 양수여야 합니다.')
    # 학습 시작 전 공통/개인 보상 환경이 동일 물리·관측을 유지하는지 먼저 확인.
    parity(argparse.Namespace(stage=args.stage, seed=args.seed, actions=200))
    filename = Path(args.init_model).resolve() if args.init_model else None
    folder = output_dir(args.root, args.stage, args.reward, args.seed, filename)
    if folder.exists():
        raise FileExistsError('기존 실험 보호: '+str(folder))
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    vec = make_vector(args.stage, args.reward)
    try:
        model = model_from_checkpoint(filename, vec, args.stage, args.seed) if filename else make_new_model(vec,args.seed)
        # torch seed / SB3 setup: 저장본은 load 후 그대로 유지, 새로운 학습 시드는 명시.
        start_step = model.num_timesteps
        model.set_logger(configure(str(folder), ['csv']))
        config = dict(version=CODE_VERSION, stage=args.stage, reward=args.reward, seed=args.seed,
                      init_model=str(filename) if filename else None,
                      init_sha256=hashlib.sha256(filename.read_bytes()).hexdigest() if filename else None,
                      init_steps=start_step, additional_steps=args.steps,
                      eval_trials=args.eval_trials, eval_every=args.eval_every,
                      algorithm='PPO, team common observations+action+physics',
                      settings='4 envs, n_steps=128, epochs=5, batch=256, lr=.0003, gamma=1, gae=.98, clip=.2, KL=.02',
                      reward_scale=.01, repeat=6, reward_rules=CODE_VERSION,
                      source_hashes=source_hashes())
        write_json(folder/'config.json', config)
        model.save(str(folder/'initial'))
        monitor = Monitor(folder,args.stage,args.reward,start_step,args.steps,
                          args.eval_every,args.eval_trials,bool(filename))
        # warmstart 모델 포함: 초기 common 정책의 품질을 별도 검증하고 best 후보로 유지.
        if filename:
            monitor.model = model
            print(f'[초기화] 공통 환경 모델 {filename} ({start_step} steps)에서 시작')
            monitor.checkpoint()
        model.after_update = lambda m: setattr(m,'ent_coef',
                 entropy((m.num_timesteps-start_step)/args.steps))
        print(f'[학습] stage={args.stage} reward={args.reward} seed={args.seed} '
              f'추가학습={args.steps} 초기스텝={start_step} 결과={folder}',flush=True)
        model.learn(total_timesteps=args.steps, reset_num_timesteps=(filename is None),
                    callback=monitor, log_interval=None)
        if monitor.last_evaluated_step != model.num_timesteps:
            monitor.model = model
            monitor.checkpoint()
        model.save(str(folder/'last'))
        selected = MilestonePPO.load(str(folder/'best.zip'),device='cpu')
        final = evaluate(selected,args.stage,args.reward,trials=args.test_trials,
                         seed=910000+args.seed,mode='stochastic')
        write_json(folder/'final_evaluation.json',final)
        write_json(folder/'summary.json',dict(config=config,final=final,best_step=monitor.best_step,
                       improvement_proven=bool(filename and monitor.best_step>start_step),
                       selected_initial=bool(filename and monitor.best_step==start_step),
                       source_unchanged=(source_hashes()==config['source_hashes'])))
        print(f'[완료] stage={args.stage} {args.reward} best={monitor.best_step} '
              f'독립평가={final["wins"]}/{final["trials"]} ({final["success_pct"]:.1f}%) '
              f'초기모델선택={bool(filename and monitor.best_step==start_step)}',flush=True)
    finally:
        vec.close()


def smoke(args):
    env = RewardEnv(stage=args.stage,reward=args.reward)
    try:
        obs, info = env.reset(seed=args.seed)
        print(f'[SMOKE] obs={obs.shape}, actions={env.action_space.n}, stage={args.stage}, reward={args.reward}')
        rng=np.random.default_rng(args.seed)
        for i in range(args.actions):
            obs,r,done,truncated,info=env.step(int(rng.integers(3)))
            if done or truncated:
                print(f'[DONE] i={i} success={info["success"]} '
                      f'milestones={len(info["milestones"])} personal={info["personal_total"]:.2f}')
                env.reset()
        print('[SMOKE OK] 학습은 진행하지 않음')
    finally:
        env.close()


def parity(args):
    """같은 행동열에서는 관측·도착·기본 보상이 반드시 같아야 함."""
    common, combined = RewardEnv(stage=args.stage,reward='common'), RewardEnv(stage=args.stage,reward='combined')
    rng=np.random.default_rng(args.seed)
    count=0
    try:
        a, _=common.reset(seed=args.seed)
        b, _=combined.reset(seed=args.seed)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
        for i in range(args.actions):
            action=int(rng.integers(3))
            a,r1,d1,t1,x=common.step(action)
            b,r2,d2,t2,y=combined.step(action)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
            assert (d1,t1,x['success'],x['milestones'])==(d2,t2,y['success'],y['milestones'])
            assert abs(r2-r1-y['personal_reward']/100.)<1e-5,(r1,r2,y['personal_reward'])
            count += 1
            if d1 or t1:
                a,_=common.reset()
                b,_=combined.reset()
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
        print(f'[PARITY PASS] 동일 행동 {count}회: 관측/물리/종료/공통 마일스톤 일치; 보상 차이는 개인 보상뿐')
    finally:
        common.close();combined.close()


def saved_eval(args):
    folder=Path(args.folder)
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config.get('version')!=CODE_VERSION:
        raise RuntimeError('v3 학습 파일만 평가 가능')
    if config['source_hashes']!=source_hashes():
        raise RuntimeError('학습 당시 코드가 변경되어 재평가 중단')
    model=MilestonePPO.load(str(folder/'best.zip'),device='cpu')
    out=evaluate(model,config['stage'],config['reward'],trials=args.trials,seed=args.eval_seed,
                 mode=args.mode)
    write_json(folder/f'eval_{args.mode}_{args.trials}_{args.eval_seed}.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description='팀원 공통환경 + 일반화 개인 보상 실험 v3')
    subs=p.add_subparsers(dest='command',required=True)
    for name in ('smoke','parity','train','evaluate'):
        q=subs.add_parser(name)
        if name in ('smoke','parity','train'):
            q.add_argument('--stage',type=int,choices=range(1,6),required=True)
            q.add_argument('--seed',type=int,default=42)
        if name in ('smoke','train'):
            q.add_argument('--reward',choices=('common','combined'),default='combined')
        if name in ('smoke','parity'):
            q.add_argument('--actions',type=int,default=1000)
        if name=='train':
            q.add_argument('--steps',type=int,default=100000)
            q.add_argument('--eval-every',type=int,default=20000)
            q.add_argument('--eval-trials',type=int,default=60)
            q.add_argument('--test-trials',type=int,default=200)
            q.add_argument('--init-model',type=Path,default=None,
                           help='기존 공통 보상 best.zip을 기준으로 추가학습')
            q.add_argument('--root',type=Path,default=HERE/'results_v3')
        if name=='evaluate':
            q.add_argument('--folder',type=Path,required=True)
            q.add_argument('--trials',type=int,default=500)
            q.add_argument('--eval-seed',type=int,default=60000)
            q.add_argument('--mode',choices=('stochastic','greedy'),default='stochastic')
    args=p.parse_args()
    {'smoke':smoke,'parity':parity,'train':train,'evaluate':saved_eval}[args.command](args)


if __name__=='__main__':
    main()
