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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUTORIAL = ROOT / 'tutorial_2_4' / 'tutorial2'
if str(TUTORIAL) not in sys.path:
    sys.path.insert(0, str(TUTORIAL))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame

from tutorial2_ppo_setting.environment import Stage2Env, ACTIONS
from tutorial2_ppo_setting.network import action_sample
from tutorial2_ppo_train import load_model, seed_all
from pingu_ball.app import App


class DemoQuit(Exception):
    pass


class RealTimeDemo:
    def __init__(self, env: Stage2Env, fps: int = 60):
        self.env = env
        self.app = App()
        self.app.start_special(2)
        self.app.world = env.world
        self.app.scene = 'play'

        self.app.sound_on = True
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.render_every = 120 // fps
        self.tick_count = 0
        self.current_action = None
        self.decision = 0
        self.episode = 1
        self.reward = 0.0
        self.info = None
        self.paused = False
        font_path = self.app.font_path
        self.font = pygame.font.Font(font_path, 20)
        self.small = pygame.font.Font(font_path, 16)
        self.big = pygame.font.Font(font_path, 30)
        self.label_font = pygame.font.Font(font_path, 12)
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

    def text(self, message, x, y, *, font=None, center=False, color=(255, 255, 255)):
        surf = (font or self.font).render(str(message), True, color)
        rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
        self.app.window.blit(surf, rect)

    def ball_ai(self):

        w = self.env.world
        v = self.app.map_view
        logical_x = v.x + w.ball.x * v.width / w.physics.width
        logical_y = v.y + w.ball.y * v.height / w.physics.height
        x = round(self.app.offset[0] + logical_x * self.app.scale)
        y = round(self.app.offset[1] + logical_y * self.app.scale - max(12, 16 * self.app.scale))
        label = self.label_font.render('AI', True, (255, 255, 255))
        bg = label.get_rect(center=(x, y)).inflate(6, 3)
        self.panel(bg, 220)
        pygame.draw.rect(self.app.window, (119, 218, 255), bg, 1, border_radius=7)
        self.app.window.blit(label, label.get_rect(center=(x, y)))

    def draw(self, *, intro=False, finished=False):





        flip = pygame.display.flip
        try:
            pygame.display.flip = lambda: None
            self.app.render()
        finally:
            pygame.display.flip = flip
        self.ball_ai()
        w, h = self.app.window.get_size()
        top = pygame.Rect(12, 12, min(470, w-24), 63)
        self.panel(top)
        self.text('TUTORIAL 02  |  PPO AI 실제 플레이', top.x+12, top.y+7)
        self.text(f'원래 출발점  |  물리 120Hz  |  화면 {self.fps}FPS (1배속)',
                  top.x+12, top.y+36, font=self.small, color=(191, 229, 254))
        bottom = pygame.Rect(12, h-67, w-24, 55)
        self.panel(bottom)
        if self.current_action is None:
            action_desc = '시작 준비'
        else:
            direction, frames = ACTIONS[self.current_action]
            direction_name = {-1: '왼쪽', 0: '대기', 1: '오른쪽'}[direction]
            action_desc = f'{direction_name} · {frames}프레임'
        self.text(f'에피소드 {self.episode}  |  결정 {self.decision}  |  현재 행동: {action_desc}',
                  bottom.x+12, bottom.y+5, font=self.small)
        if self.info:
            i = self.info
            self.text(f'누적 보상 {i["reward_total"]:+.2f}  |  점프 블록 {i["spring1"]}/{i["spring2"]}/{i["spring3"]}  |  구간 {i["milestone"]}/3',
                      bottom.x+12, bottom.y+30, font=self.small, color=(191, 229, 254))
        else:
            self.text('학습된 last.pt 체크포인트로 재생 (재학습 없음)',
                      bottom.x+12, bottom.y+30, font=self.small, color=(191, 229, 254))
        if intro or finished:
            ww = min(w-40, 510)
            box = pygame.Rect((w-ww)//2, (h-132)//2, ww, 132)
            self.panel(box, 235)
            pygame.draw.rect(self.app.window, (117, 218, 254), box, 2, border_radius=12)
            title = ('튜토리얼 2단계 학습된 AI 실제 플레이' if intro
                     else '튜토리얼 2단계 완주!' if self.env.world.cleared
                     else '이번 도전 실패')
            title_image = self.big.render(title, True, (255, 255, 255))
            allowed_width = max(1, box.width - 32)
            if title_image.get_width() > allowed_width:
                ratio = allowed_width / title_image.get_width()
                title_image = pygame.transform.smoothscale(
                    title_image,
                    (allowed_width, max(1, round(title_image.get_height() * ratio)))
                )
            self.app.window.blit(
                title_image,
                title_image.get_rect(center=(box.centerx, box.y + 36))
            )
            self.text('PPO 모델 · 출발점부터 실시간 재생', box.centerx, box.y+78, center=True)
            self.text('SPACE 일시정지 / ESC 종료', box.centerx, box.y+107,
                      font=self.small, center=True, color=(191, 229, 254))
        pygame.display.flip()

    def tick(self):
        self.tick_count += 1
        if self.tick_count % self.render_every == 0 or self.env.world.failed or self.env.world.cleared:
            self.events()
            self.draw()
            self.clock.tick(self.fps)

    def intro(self):
        self.events()
        self.draw(intro=True)

        for _ in range(13):
            pygame.time.wait(100)
            self.events()

    def finish(self):
        self.events()
        self.draw(finished=True)

        print('시연 종료. 화면에서 ESC를 눌러 닫아줘.')
        while True:
            self.events()
            pygame.time.wait(35)

    def close(self):
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='튜토리얼 2 PPO: 물리 프레임 단위 1배속 실시간 시연')
    parser.add_argument('--model', type=Path, default=TUTORIAL/'result_tutorial2'/'last.pt')
    parser.add_argument('--seed', type=int, default=60000)
    parser.add_argument('--mode', choices=['mixed', 'greedy', 'stochastic'], default='mixed')
    parser.add_argument('--fps', type=int, choices=[30, 60], default=60)
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes는 1 이상이어야 해')

    seed_all(args.seed)
    ck, model = load_model(args.model)
    env = Stage2Env()
    demo = RealTimeDemo(env, fps=args.fps)




    original_step = env.world.step
    def step_and_render(*a, **kw):
        result = original_step(*a, **kw)



        for kind, _payload in env.world.events:
            if kind in demo.app.sounds:
                demo.app.sound(kind)
        demo.tick()
        return result
    env.world.step = step_and_render

    print(f'[MODEL] {args.model} | trained_steps={ck.get("steps")} | mode={args.mode} | {args.fps} FPS')
    try:
        for ep in range(1, args.episodes+1):
            demo.episode = ep
            demo.decision = 0
            demo.info = None
            demo.current_action = None
            demo.tick_count = 0
            obs = env.reset()
            demo.intro()
            done = False
            while not done:
                demo.events()
                action, _, _ = action_sample(model, obs, mode=args.mode, greedy_mix=.05)
                demo.current_action = action
                demo.decision += 1
                obs, reward, done, info = env.step(action)
                demo.reward = reward
                demo.info = info
            print(f'[EP {ep}] success={info["success"]} reason={info["reason"]} time={info["seconds"]:.2f}s reward={info["reward_total"]:.2f}')
            if info['success']:
                demo.finish()
            else:
                demo.draw(finished=True)
                for _ in range(20):
                    pygame.time.wait(100)
                    demo.events()
        demo.finish()
    except DemoQuit:
        print('[EXIT] 사용자 종료')
    finally:
        demo.close()

if __name__ == '__main__':
    main()
