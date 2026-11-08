"""단계별 9행동 정책을 실제 화면 또는 GIF로 재생한다."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import torch
from .five_train import read,write
from .five_models import action_probs
from .stabilized_train import sources
from .stabilized_env import StabilizedEnv
from .stabilized_load import load_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',choices=['A','B','C'],default='C')
    parser.add_argument('--stage',type=int,choices=[2,4],required=True)
    parser.add_argument('--training-seed',type=int,default=0)
    parser.add_argument('--seed',type=int,default=20261101)
    parser.add_argument('--latest',action='store_true')
    parser.add_argument('--greedy',action='store_true')
    parser.add_argument('--gif',type=Path)
    args=parser.parse_args()
    variant='A' if args.stage==2 and args.variant=='B' else args.variant
    folder=args.root/variant/f'stage_{args.stage:02d}'/f'seed_{args.training_seed}'
    assert read(folder/'config.json')['sources']==sources()
    torch.set_num_threads(1)
    model=load_policy(folder/('latest.zip' if args.latest else 'best.zip'),device='cpu')
    model.policy.set_training_mode(False)
    env=StabilizedEnv(stage=args.stage,variant=variant);obs,_=env.reset()
    if args.gif:
        import os
        os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    import pygame
    from PIL import Image
    from pingu_ball.app import App
    rng=np.random.default_rng(args.seed);frames=[];ticks=[]
    with tempfile.TemporaryDirectory() as tmp:
        app=App(Path(tmp)/'save.json',headless=bool(args.gif));app.sound_on=False
        app.start_special(args.stage);clock=pygame.time.Clock()
        def tick(world,n,direction):
            app.world=world
            if n%12==0 or world.failed or world.cleared:
                app.render()
                if args.gif:
                    image=Image.frombytes('RGB',app.canvas.get_size(),pygame.image.tobytes(app.canvas,'RGB'))
                    image.thumbnail((1000,520));frames.append(image);ticks.append(n)
            if not args.gif:clock.tick(120)
        try:
            while not env.unwrapped.done:
                if any(e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE) for e in pygame.event.get()):break
                p=action_probs(model,obs[None])[0]
                action=int(p.argmax()) if args.greedy else int(rng.choice(9,p=p/p.sum()))
                env.unwrapped.step(action,on_tick=tick)
                obs=env.observation(env.unwrapped.observe())
        finally:pygame.quit()
    if args.gif and frames:
        args.gif.parent.mkdir(parents=True,exist_ok=True)
        durations=[max(10,round((b-a)/120*100)*10) for a,b in zip([0]+ticks[:-1],ticks)]
        # 같은 배경에 공통 팔레트를 사용해 프레임 간 색 흔들림과 파일 크기를 줄인다.
        palette=frames[0].quantize(colors=256)
        indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
        indexed[0].save(args.gif,save_all=True,append_images=indexed[1:],duration=durations,loop=0,optimize=True)
        frames[-1].save(args.gif.with_suffix('.png'))
        write(args.gif.with_suffix('.json'),dict(variant=variant,stage=args.stage,training_seed=args.training_seed,
              sampling_seed=args.seed,greedy=args.greedy,result=env.unwrapped.info()))
    print(env.unwrapped.info(),flush=True);env.close()


if __name__=='__main__':main()
