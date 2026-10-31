"""Show an expert or consolidated policy acting on current native game state."""
import argparse
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .five_env import FiveEnv
from .five_models import ALGORITHMS, action_probs
from .five_train import read, write, sources


def watch(checkpoint, stage, gif=None, sampled=False, seed=20261029):
    checkpoint = Path(checkpoint)
    config = read(checkpoint.parent/'config.json')
    assert config['sources'] == sources()
    if stage not in config.get('stages', [config.get('stage')]):
        raise ValueError('Checkpoint was not trained for this stage')
    torch.set_num_threads(1)
    model = ALGORITHMS[config['algorithm']].load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    env = FiveEnv(stage=stage, reward_mode=config['reward_mode'])
    obs, _ = env.reset()
    if gif:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import pygame
    from PIL import Image
    from pingu_ball.app import App
    rng = np.random.default_rng(seed)
    frames, ticks, actions = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        app = App(Path(tmp)/'progress.json', headless=bool(gif))
        app.sound_on = False
        app.start_special(stage)
        clock = pygame.time.Clock()
        def on_tick(world, tick, direction):
            app.world = world
            if tick % 12 == 0 or world.cleared or world.failed:
                app.render()
                if gif:
                    frame = Image.frombytes('RGB', app.canvas.get_size(), pygame.image.tobytes(app.canvas, 'RGB'))
                    frame.thumbnail((1000, 520))
                    frames.append(frame)
                    ticks.append(tick)
            if not gif:
                clock.tick(120)
        try:
            while not env.unwrapped.done:
                if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                       for e in pygame.event.get()):
                    break
                p = action_probs(model, obs[None])[0]
                action = int(rng.choice(3, p=p/p.sum())) if sampled else int(p.argmax())
                actions.append(action)
                env.unwrapped.step(action, on_tick=on_tick)
                obs = env.observation(env.unwrapped.observe())
        finally:
            pygame.quit()
    result = dict(checkpoint=str(checkpoint.resolve()), stage=stage, sampled=sampled, seed=seed,
                  live_state_feedback=True, playback_schedule=False, actions=actions, result=env.unwrapped.info())
    if gif and frames:
        gif = Path(gif)
        gif.parent.mkdir(parents=True, exist_ok=True)
        palette = frames[0].quantize(colors=128)
        indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        durations = [max(10, round((b-a)/120*100)*10) for a, b in zip([0]+ticks[:-1], ticks)]
        indexed[0].save(gif, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True)
        frames[-1].save(gif.with_suffix('.png'))
        write(gif.with_suffix('.json'), result)
    env.close()
    print(result['result'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--gif', type=Path)
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    watch(args.checkpoint, args.stage, args.gif, args.sampled, args.seed)
