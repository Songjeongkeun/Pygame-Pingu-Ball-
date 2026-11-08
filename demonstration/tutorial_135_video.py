\
\
\
\
\
\
\
\
\

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / 'experiments' / 'tutorial_1_5'
RUNTIME = ARCHIVE / 'baseline' / 'runtime'


class DemoQuit(Exception):
    pass


def find_checkpoint(stage):
    catalog = json.loads((ARCHIVE / 'models.json').read_text(encoding='utf-8'))['models']
    matches = [m for m in catalog if m['experiment'] == 'baseline'
               and m['algorithm'] == 'ppo' and m['stage'] == stage
               and m['training_seed'] == 0]
    if len(matches) != 1:
        raise RuntimeError(f'No unique archived PPO model for stage {stage}')
    entry = matches[0]
    path = ARCHIVE / entry['checkpoint']
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint was not downloaded from Git: {path}')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry['sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {path}')
    return entry, path.resolve()


def check_sources(entry, checkpoint):

    config = json.loads(checkpoint.with_name('config.json').read_text(encoding='utf-8'))
    for name, digest in config['sources'].items():
        source = (ARCHIVE / entry['runtime'] / Path(name.replace('\\', '/')))
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Archived source missing or changed: {source}')
    if config.get('algorithm') != 'ppo' or entry['stage'] not in config.get('stages', [config.get('stage')]):
        raise RuntimeError('Checkpoint configuration is incompatible with requested stage')
    return config


def main():
    parser = argparse.ArgumentParser(description='튜토리얼 1·3·5 PPO 실시간 1배속 AI 시연')
    parser.add_argument('--stage', type=int, choices=(1, 3, 5), required=True)
    parser.add_argument('--fps', type=int, choices=(30, 60), default=60)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--mode', choices=('stochastic', 'greedy'), default='stochastic',
                        help='기존 독립 평가와 같은 방식은 stochastic (100%% 확률 표본 선택)')
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be >=1')

    entry, checkpoint = find_checkpoint(args.stage)
    config = check_sources(entry, checkpoint)
    if not RUNTIME.is_dir():
        raise FileNotFoundError(f'Archived runtime missing: {RUNTIME}')



    os.chdir(RUNTIME)
    sys.path.insert(0, str(RUNTIME))
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import numpy as np
    import torch
    import pygame
    from rl.milestone_env import MilestoneEnv
    from rl.milestone_models import ALGORITHMS
    from rl.five_models import action_probs
    from pingu_ball.app import App

    torch.set_num_threads(1)
    model = ALGORITHMS['ppo'].load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    repeat = int(entry.get('repeat', 6))
    if repeat != 6:
        raise RuntimeError(f'Unexpected archived action repeat: {repeat}')
    env = MilestoneEnv(stage=args.stage, reward_mode='milestone', repeat=repeat)
    rng = np.random.default_rng(args.seed)

    class Demo:
        def __init__(self):
            self.tmp = tempfile.TemporaryDirectory(prefix='pingu-ai-demo-')
            self.app = App(save_path=Path(self.tmp.name) / 'progress.json')
            self.app.start_special(args.stage)
            self.app.world = env.unwrapped.world
            self.app.scene = 'play'
            self.app.sound_on = True
            self.app.sync_music()
            self.clock = pygame.time.Clock()
            self.render_every = 120 // args.fps
            self.ticks = 0
            self.episode = 1
            self.decisions = 0
            self.action = None
            self.probabilities = None
            self.last_info = None
            self.paused = False
            f = self.app.font_path
            self.font = pygame.font.Font(f, 20)
            self.small = pygame.font.Font(f, 16)
            self.big = pygame.font.Font(f, 30)
            self.label_font = pygame.font.Font(f, 12)
            self.label_font.set_bold(True)

        def events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise DemoQuit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            while self.paused:
                pygame.time.wait(35)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise DemoQuit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        self.paused = False

        def panel(self, rect, alpha=210):
            surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surface, (6, 33, 72, alpha), surface.get_rect(), border_radius=12)
            self.app.window.blit(surface, rect.topleft)

        def text(self, message, x, y, *, font=None, center=False, color=(255, 255, 255), max_width=None):
            surf = (font or self.font).render(str(message), True, color)
            if max_width and surf.get_width() > max_width:
                factor = max_width / surf.get_width()
                surf = pygame.transform.smoothscale(surf, (int(max_width), max(1, round(surf.get_height()*factor))))
            rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
            self.app.window.blit(surf, rect)

        def ball_ai(self):
            world = env.unwrapped.world
            v = self.app.map_view
            lx = v.x + world.ball.x * v.width / world.physics.width
            ly = v.y + world.ball.y * v.height / world.physics.height
            x = round(self.app.offset[0] + lx * self.app.scale)
            y = round(self.app.offset[1] + ly * self.app.scale - max(12, 16*self.app.scale))
            image = self.label_font.render('AI', True, (255, 255, 255))
            rect = image.get_rect(center=(x, y)).inflate(6, 3)
            self.panel(rect, 220)
            pygame.draw.rect(self.app.window, (119, 218, 255), rect, width=1, border_radius=7)
            self.app.window.blit(image, image.get_rect(center=(x, y)))

        def draw(self, *, intro=False, finished=False):


            original_flip = pygame.display.flip
            try:
                pygame.display.flip = lambda: None
                self.app.render()
            finally:
                pygame.display.flip = original_flip

            self.ball_ai()
            w, h = self.app.window.get_size()
            top = pygame.Rect(12, 12, min(470, w-24), 63)
            self.panel(top)
            self.text(f'TUTORIAL {args.stage:02d}  |  PPO AI 실제 플레이', top.x+12, top.y+7,
                      max_width=top.width-24)
            self.text(f'출발점부터 재생  |  물리 120Hz  |  {args.fps}FPS (1배속)',
                      top.x+12, top.y+36, font=self.small, color=(191,229,254),
                      max_width=top.width-24)

            bottom = pygame.Rect(12, h-67, w-24, 55)
            self.panel(bottom)
            action_desc = '시작 준비' if self.action is None else {-1:'왼쪽', 0:'대기', 1:'오른쪽'}[self.action-1]
            self.text(f'에피소드 {self.episode}  |  결정 {self.decisions}  |  현재 행동: {action_desc} (6프레임)',
                      bottom.x+12, bottom.y+5, font=self.small, max_width=bottom.width-24)
            raw = env.unwrapped
            completed = len(raw.tracker.completed)
            total = len(raw.specs)
            score = raw.episode_return
            mode_name = '확률 선택' if args.mode == 'stochastic' else '최대확률 선택'
            self.text(f'누적 보상 {score:+.1f}점  |  구간 완료 {completed}/{total}  |  {mode_name}  |  ESC 종료 / SPACE 일시정지',
                      bottom.x+12, bottom.y+30, font=self.small,
                      color=(191,229,254), max_width=bottom.width-24)

            if intro or finished:
                bw = min(w-40, 510)
                box = pygame.Rect((w-bw)//2, (h-132)//2, bw, 132)
                self.panel(box, 235)
                pygame.draw.rect(self.app.window, (117,218,254), box, 2, border_radius=12)
                if intro:
                    title = f'튜토리얼 {args.stage}단계 학습된 AI 실제 플레이'
                else:
                    title = (f'튜토리얼 {args.stage}단계 완주!' if raw.world.cleared
                             else f'튜토리얼 {args.stage}단계 이번 도전 실패')
                self.text(title, box.centerx, box.y+36, font=self.big, center=True,
                          max_width=box.width-32)
                self.text('PPO BEST 모델 · 출발점부터 실시간 재생', box.centerx, box.y+78,
                          center=True, max_width=box.width-24)
                self.text('SPACE 일시정지 / ESC 종료', box.centerx, box.y+107,
                          font=self.small, center=True, color=(191,229,254))
            pygame.display.flip()

        def on_tick(self, world, tick, direction):
            self.ticks += 1

            for kind, _ in world.events:
                if kind in self.app.sounds:
                    self.app.sound(kind)
            if self.ticks % self.render_every == 0 or world.cleared or world.failed:
                self.events()
                self.draw()
                self.clock.tick(args.fps)

        def intro(self):
            self.events()
            self.draw(intro=True)
            for _ in range(12):
                pygame.time.wait(100)
                self.events()

        def ending(self):
            self.draw(finished=True)
            print('[VIDEO] 마지막 화면 유지 중. ESC를 누르면 종료돼.', flush=True)
            while True:
                self.events()
                pygame.time.wait(35)

        def close(self):
            pygame.quit()
            self.tmp.cleanup()

    demo = None
    try:
        demo = Demo()
        print(f'[MODEL] stage={args.stage} checkpoint={checkpoint} trained_seed=0 '
              f'policy={args.mode} actions=3 repeat=6; no learning', flush=True)
        for episode in range(1, args.episodes+1):
            demo.episode = episode
            demo.decisions = 0
            demo.action = None
            demo.probabilities = None
            demo.ticks = 0
            observation, _ = env.reset()
            demo.app.world = env.unwrapped.world
            demo.intro()
            raw = env.unwrapped
            while not raw.done:
                demo.events()
                probabilities = action_probs(model, observation[None])[0]
                probabilities = probabilities/probabilities.sum()
                action = (int(rng.choice(3, p=probabilities)) if args.mode == 'stochastic'
                          else int(probabilities.argmax()))
                demo.action = action
                demo.probabilities = probabilities
                demo.decisions += 1
                raw.step(action, on_tick=demo.on_tick)
                observation = env.observation(raw.observe())
            info = raw.info()
            print(f'[EP {episode}] success={raw.world.cleared} failed={raw.world.failed} '
                  f'time={raw.world.time:.2f}s milestones={len(raw.tracker.completed)}/{len(raw.specs)} '
                  f'score={raw.episode_return:+.1f}', flush=True)
            if raw.world.cleared or episode == args.episodes:
                demo.ending()
            else:
                demo.draw(finished=True)
                for _ in range(10):
                    pygame.time.wait(100)
                    demo.events()
    except DemoQuit:
        print('[EXIT] 사용자 종료', flush=True)
    finally:
        if demo is not None:
            demo.close()
        env.close()


if __name__ == '__main__':
    main()
