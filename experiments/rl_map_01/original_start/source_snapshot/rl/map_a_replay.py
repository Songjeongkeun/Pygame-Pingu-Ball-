"""선택된 정책을 원래 시작점에서 재생해 실제 좌표와 물리 대조를 기록한다."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from pingu_ball.engine import World
from .five_models import action_probs
from .five_train import read,write
from .map_a_env import MapAEnv,ACTIONS,audit_reward
from .map_a_models import load_policy
from .map_a_train import sources
from .map_a_report import plot_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seeds',default='0,1,2')
    args = parser.parse_args();root = args.root.resolve()
    assert read(root/'completion.json')['complete']
    plan = read(root/'plan.json');assert plan['sources']==sources()
    torch.set_num_threads(1)
    results = []
    for seed in [int(x) for x in args.seeds.split(',')]:
        checkpoint = root/f'seed_{seed}'/'best.zip'
        model = load_policy(checkpoint);model.policy.set_training_mode(False)
        env = MapAEnv();obs,_ = env.reset();raw = env.unwrapped
        native = World(raw.level,auto_reset=False)
        trajectory = [dict(t=0.,x=raw.world.ball.x,y=raw.world.ball.y)]
        choices,max_error = [],0.
        def tick(world,number,direction):
            nonlocal max_error
            native.step(direction,min(native.physics.dt,180.-native.time))
            for name in ('x','y','vx','vy'):
                max_error = max(max_error,abs(getattr(world.ball,name)-getattr(native.ball,name)))
            assert max_error<1e-8 and world.events==native.events
            if number%12==0 or world.events:
                trajectory.append(dict(t=world.time,x=world.ball.x,y=world.ball.y))
        while not raw.done:
            probabilities = action_probs(model,obs[None])[0]
            action = int(probabilities.argmax())
            direction,frames = ACTIONS[action]
            choices.append(dict(t=raw.world.time,action=action,direction=direction,ticks=frames,
                                probabilities=probabilities.tolist()))
            raw.step(action,on_tick=tick);obs = env.observation(raw.observe())
        info = raw.info();audit_reward(info)
        assert max_error==0.
        trajectory.append(dict(t=raw.world.time,x=raw.world.ball.x,y=raw.world.ball.y))
        data = dict(seed=seed,protocol='greedy_one_episode_diagnostic',result=info,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            trajectory=trajectory,actions=choices,max_native_error=max_error,sources=sources())
        write(root/f'replay_seed_{seed}.json',data);results.append(data);env.close()
        print(f'시드{seed} 재생: 성공={info["success"]}, 구간={len(info["milestones"])}/16, 시간={info["seconds"]:.2f}초',flush=True)
    plot_map(root,results)
    print(root/'policy_paths.png',flush=True)


if __name__=='__main__':main()
