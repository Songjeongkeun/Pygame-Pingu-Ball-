"""Pygame UI with a 1600 by 832 logical canvas and proportional resizing."""
import argparse
from array import array
import math
import os
from pathlib import Path
import sys
import tempfile

import pygame
from .config import PHYSICS
from .engine import World
from .levels import make_levels
from .storage import Progress
from .skin import WinterSkin

ROOT = Path(__file__).resolve().parent.parent
INK = (23, 48, 70)
MUTED = (91, 116, 137)
WHITE = (247, 252, 255)
BLUE = (39, 119, 184)
GOLD = (239, 173, 55)
CORAL = (212, 90, 101)
TEAL = (39, 166, 172)
GRADES = {6: 'B', 7: 'A', 8: 'S', 9: 'SS', 10: 'SSS'}


def default_save_path():
    # One-file builds unpack assets into a temporary directory, but saves persist.
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else ROOT
    return root/'save_data'/'progress.json'


class App:
    def __init__(self, save_path=None, headless=False):
        self.headless = headless
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        self.window = pygame.display.set_mode((1000, 520), pygame.RESIZABLE)
        pygame.display.set_caption('Pingu Ball')
        self.canvas = pygame.Surface((PHYSICS.width, PHYSICS.height)).convert()
        self.clock = pygame.time.Clock()
        self.levels = make_levels()
        self.progress = Progress(save_path or default_save_path())
        self.scene = 'menu'
        self.world = None
        self.running = True
        self.buttons = []
        self.held_keys = set()
        self.hover = (0, 0)
        self.offset = (0, 0)
        self.scale = .625
        self.wall_time = 0.
        self.accumulator = 0.
        self.toast = ''
        self.toast_until = 0.
        self.new_record = False
        self.help = False
        self.sound_on = True
        self.sounds = {}
        self._font_cache = {}
        self.font_path = self.find_font()
        self.background = pygame.image.load(str(ROOT/'assets'/'antarctica.png')).convert()
        self.background = pygame.transform.smoothscale(self.background, self.canvas.get_size())
        self.skin = WinterSkin(ROOT)
        self.map_canvas = None
        self.previews = {}
        for level in self.levels:
            if level.preview:
                image = pygame.image.load(str(ROOT/level.preview)).convert()
                self.previews[level.index] = pygame.transform.smoothscale(image, (124, 78))
        self.prepare_sounds()

    @staticmethod
    def find_font():
        for path in ['C:/Windows/Fonts/malgun.ttf',
                     '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                     '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
            if Path(path).exists():
                return path
        return pygame.font.match_font('malgungothic,applesdgothicneo,notosanscjkkr,dejavusans')

    def text(self, text, x, y, size=22, color=INK, bold=False, center=False):
        key = (size, bold)
        if key not in self._font_cache:
            font = pygame.font.Font(self.font_path, size)
            font.set_bold(bold)
            self._font_cache[key] = font
        image = self._font_cache[key].render(str(text), True, color)
        rect = image.get_rect(center=(x, y)) if center else image.get_rect(topleft=(x, y))
        self.canvas.blit(image, rect)
        return rect

    def panel(self, rect, color=(249, 253, 255, 238), radius=20):
        rect = pygame.Rect(rect)
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surface, color, surface.get_rect(), border_radius=radius)
        self.canvas.blit(surface, rect)

    def button(self, label, rect, callback, primary=False):
        rect = pygame.Rect(rect)
        hovered = rect.collidepoint(self.hover)
        color = (31, 96, 146) if primary and hovered else BLUE if primary else (224, 240, 249) if hovered else WHITE
        pygame.draw.rect(self.canvas, color, rect, border_radius=12)
        self.text(label, *rect.center, 22, WHITE if primary else INK, True, True)
        self.buttons.append((rect, callback))

    def prepare_sounds(self):
        if not pygame.mixer.get_init():
            return
        for name, hz, length in [('fish', 880, .10), ('fail', 160, .16), ('clear', 660, .24), ('switch', 440, .08)]:
            frames = int(44100*length)
            samples = array('h', (int(6000*math.sin(2*math.pi*hz*i/44100)*(1-i/frames)**2) for i in range(frames)))
            self.sounds[name] = pygame.mixer.Sound(buffer=samples)

    def sound(self, name):
        if self.sound_on and name in self.sounds:
            self.sounds[name].play()

    def draw_pingu(self, x, y, size):
        self.canvas.blit(self.skin.image('ball', (size, size)), (round(x-size/2), round(y-size/2)))

    def fish(self, x, y, size=32, bone=False, collected=False):
        image = self.skin.image('bone' if bone else 'fish', (size, size))
        if collected:
            image = image.copy(); image.set_alpha(90)
        self.canvas.blit(image, (round(x-size/2), round(y-size/2)))

    def igloo(self, rect):
        x, y, w, h = rect
        self.canvas.blit(self.skin.image('goal', (w, h)), (round(x), round(y)))

    def tile(self, block, time, switches):
        if block.id.startswith('spike-base-'):
            return
        image = self.skin.platform(block, time, switches)
        if image is None:
            return
        x, y, width, height = block.rect_at(time)
        self.canvas.blit(image, (round(x), round(y)))
        if block.kind == 'switch':
            self.text(block.group, x+width-12, y+height-16, 14, WHITE, True, True)
        if block.break_at is not None:
            remaining = max(0., (block.break_at-time)/PHYSICS.break_delay)
            pygame.draw.line(self.canvas, CORAL, (x,y-4), (x+width*remaining,y-4), 3)

    def backdrop(self):
        self.canvas.blit(self.background, (0, 0))

    def draw_menu(self):
        self.backdrop()
        self.text('Pingu Ball', 64, 64, 68, INK, True)
        self.text('자동으로 튀는 핑구 공을 이글루까지 안내하세요.', 68, 160, 25, MUTED)
        self.draw_pingu(1480, 168, 124)
        self.button('플레이 방법', (1120, 36, 190, 48), lambda: setattr(self, 'help', True))
        self.button('종료', (1330, 36, 190, 48), self.quit)
        completed = sum(str(i) in self.progress.records for i in range(1, 8))
        self.text(f'스테이지 선택  /  {completed}개 완료', 64, 232, 26, INK, True)
        for index, level in enumerate(self.levels):
            col, row = index % 4, index // 4
            rect = pygame.Rect(64+col*372, 294+row*180, 350, 156)
            unlocked = index <= self.progress.unlocked
            self.panel(rect)
            self.text('튜토리얼' if index == 0 else f'STAGE {index}', rect.x+22, rect.y+16, 18, BLUE if unlocked else MUTED, True)
            self.text(level.title, rect.x+22, rect.y+49, 23, INK if unlocked else MUTED, True)
            if index in self.previews:
                self.canvas.blit(self.previews[index], (rect.right-138, rect.y+15))
            record = self.progress.records.get(str(index))
            if record:
                detail = 'CLEAR' if index == 0 else f"{GRADES[record['score']]} · {record['score']}점 · {record['time']:.2f}초"
            else:
                detail = '플레이 →' if unlocked else '앞 스테이지 클리어 필요'
            self.text(detail, rect.x+22, rect.y+111, 20, TEAL if record else MUTED)
            if unlocked:
                self.buttons.append((rect, lambda n=index: self.start(n)))
        self.text('← → 이동    ·    입장 즉시 자동 반동    ·    R 재시작', 64, 726, 22)
        self.button('소리 켜짐' if self.sound_on else '소리 꺼짐', (1320, 710, 200, 48), self.toggle_sound)
        if self.progress.error:
            self.text(self.progress.error, 64, 782, 20, CORAL)

    def start(self, index):
        self.world = World(self.levels[index])
        self.map_canvas = pygame.Surface((self.world.physics.width, self.world.physics.height), pygame.SRCALPHA).convert_alpha()
        self.scene = 'play'
        self.accumulator = 0.
        self.held_keys.clear()
        self.toast = ''
        self.new_record = False

    def quit(self):
        self.running = False

    def toggle_sound(self):
        self.sound_on = not self.sound_on

    def restart(self):
        if self.world:
            self.world.reset()
            self.scene = 'play'
            self.accumulator = 0.
            self.held_keys.clear()

    def menu(self):
        self.scene = 'menu'
        self.accumulator = 0.
        self.held_keys.clear()
        self.help = False

    def toggle_pause(self):
        if self.world and self.scene == 'play':
            self.world.paused = not self.world.paused
            self.accumulator = 0.
            self.held_keys.clear()

    def draw_play(self):
        world = self.world
        self.backdrop()
        screen = self.canvas
        self.canvas = self.map_canvas
        try:
            self.draw_map(world)
        finally:
            self.canvas = screen
        scale = min(1568/world.physics.width, 704/world.physics.height)
        size = (round(world.physics.width*scale), round(world.physics.height*scale))
        self.map_view = pygame.Rect((1600-size[0])//2, 72, *size)
        self.canvas.blit(pygame.transform.smoothscale(self.map_canvas, size), self.map_view)
        self.draw_play_hud(world)

    def draw_map(self, world):
        self.canvas.fill((0, 0, 0, 0))
        groups = {}
        for block in world.platforms:
            if block.kind == 'moving':
                prefix = block.id.rsplit('-', 1)[0]
                groups.setdefault(prefix, []).append(block)
        for blocks in groups.values():
            left = min(q.x for q in blocks)
            right = max(q.x+q.w+q.travel for q in blocks)
            y = blocks[0].y-12
            pygame.draw.line(self.canvas, (86, 185, 213), (left,y), (right,y), 2)
            for x in (left, right):
                pygame.draw.line(self.canvas, (86, 185, 213), (x,y-5), (x,y+5), 2)
        for block in world.platforms:
            self.tile(block, world.time, world.switches)
        for spike in world.level.spikes:
            self.canvas.blit(self.skin.image('spike', (spike.w, spike.h+(16 if spike.teeth == 6 else 0))),
                             (round(spike.x), round(spike.y)))
        for index, fish in enumerate(world.level.collectibles):
            if index not in world.collected:
                self.fish(fish.x, fish.y, int(fish.size), fish.value < 0)
        self.igloo(world.level.goal)
        self.draw_pingu(world.ball.x, world.ball.y, 28)

    def draw_play_hud(self, world):
        # Top row belongs to UI; the map never scrolls.
        self.panel((0, 0, 1600, 64), (249, 253, 255, 245), 0)
        self.text(('튜토리얼' if world.level.index == 0 else f'STAGE {world.level.index}')+' / '+world.level.title, 24, 18, 24, INK, True)
        total = sum(fish.value > 0 for fish in world.level.collectibles)
        if total:
            self.fish(530, 32)
            self.text(f'{world.plus}/{total}', 554, 18, 24)
        if any(fish.value < 0 for fish in world.level.collectibles):
            self.fish(640, 32, bone=True)
            self.text(str(world.minus), 664, 18, 24)
        self.text(f'{world.time:.2f}초', 750, 20, 22, MUTED)
        self.button('일시정지', (1000, 10, 170, 44), self.toggle_pause)
        self.button('재시작 R', (1180, 10, 170, 44), self.restart)
        self.button('스테이지', (1360, 10, 216, 44), self.menu)
        self.panel((16, 778, 1568, 42), radius=10)
        caption = '수평 발사 중 · 방향키를 누르면 낙하합니다.' if world.ball.launch else world.level.subtitle
        self.text(caption, 32, 785, 22)
        if self.toast and self.wall_time < self.toast_until:
            self.text(self.toast, 800, 146, 24, CORAL, True, True)
        if world.paused:
            self.draw_pause()
        if self.scene == 'result':
            self.draw_result()

    def overlay(self, rect):
        shade = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        shade.fill((19, 43, 63, 128))
        self.canvas.blit(shade, (0, 0))
        self.buttons = []
        self.panel(rect, (249, 253, 255, 250), 24)

    def draw_pause(self):
        self.overlay((500, 240, 600, 340))
        self.text('일시정지', 800, 300, 40, INK, True, True)
        self.text('공과 모든 블록의 시간이 멈췄습니다.', 800, 360, 22, MUTED, False, True)
        self.button('계속하기', (540, 416, 520, 54), self.toggle_pause, True)
        self.button('다시 시작', (540, 492, 250, 48), self.restart)
        self.button('스테이지 선택', (810, 492, 250, 48), self.menu)

    def draw_result(self):
        world = self.world
        self.overlay((480, 164, 640, 500))
        tutorial = world.level.index == 0
        self.text('튜토리얼 완료!' if tutorial else '집에 도착했어요!', 800, 224, 36, INK, True, True)
        self.text('CLEAR' if tutorial else world.grade, 800, 320, 68, BLUE, True, True)
        if not tutorial:
            self.text(f'{world.score}점 · {world.time:.2f}초', 800, 396, 28, INK, True, True)
            self.text('최고 기록 갱신' if self.new_record else '기존 최고 기록 유지', 800, 446, 22, TEAL, True, True)
        if self.progress.error:
            self.text(self.progress.error, 800, 484, 18, CORAL, False, True)
        self.button('다시 도전', (520, 516, 270, 54), self.restart)
        self.button('여행 완료' if world.level.index == 7 else '다음 스테이지', (810, 516, 270, 54), self.next_stage, True)
        self.button('스테이지 선택', (620, 594, 360, 44), self.menu)

    def next_stage(self):
        if self.world.level.index == 7:
            self.scene = 'ending'
        else:
            self.start(self.world.level.index+1)

    def draw_ending(self):
        self.backdrop()
        self.panel((350, 150, 900, 540))
        self.draw_pingu(800, 280, 140)
        self.text('남극 여행을 마쳤어요!', 800, 416, 44, INK, True, True)
        self.text('모든 스테이지에서 더 빠른 기록에 도전해 보세요.', 800, 490, 26, MUTED, False, True)
        self.button('스테이지 선택', (550, 566, 500, 60), self.menu, True)

    def draw_help(self):
        self.overlay((340, 116, 920, 600))
        self.text('플레이 방법', 390, 150, 36, INK, True)
        rows = [('← → / R', '좌우 이동 / 처음부터 재시작 · 입장 즉시 자동 반동'),
                ('생선 / 생선 가시', '+1점 / -1점 · 모두 선택 사항'),
                ('금 간 얼음', '첫 착지 0.5초 후 파괴'),
                ('점프 화살표 ↑', '일반 발판보다 더 높이 튕깁니다.'),
                ('이동 ↔ / 시간 발판', '표시 구간 왕복 이동 / 주기적 등장과 소멸'),
                ('빨간 스위치', '연결 다리 전환 · 완전히 벗어난 뒤 재작동'),
                ('좌우 발사 화살표', '중심 높이에서 발사 · 방향키 또는 벽 충돌로 해제'),
                ('이글루', '생선 획득 여부와 관계없이 도착하면 클리어')]
        for index, (label, body) in enumerate(rows):
            self.text(label, 390, 226+index*48, 22, BLUE, True)
            self.text(body, 610, 226+index*48, 21)
        self.button('알겠어요', (1000, 640, 200, 50), lambda: setattr(self, 'help', False), True)

    def update(self, seconds, direction):
        self.wall_time += seconds
        if self.scene != 'play' or not self.world or self.world.paused or self.help:
            self.accumulator = 0.
            return
        self.accumulator += seconds
        while self.accumulator+1e-12 >= PHYSICS.dt:
            self.world.step(direction)
            self.accumulator -= PHYSICS.dt
            for kind, payload in self.world.events:
                if kind in self.sounds:
                    self.sound(kind)
                if kind == 'fail':
                    self.toast = payload
                    self.toast_until = self.wall_time+1.7
                    self.held_keys.clear()
                    self.accumulator = 0.
                    return
            if self.world.cleared:
                score = None if self.world.level.index == 0 else self.world.score
                self.new_record = self.progress.complete(self.world.level.index, score, self.world.time)
                self.scene = 'result'
                self.accumulator = 0.
                break

    def render(self):
        self.buttons = []
        if self.scene == 'menu':
            self.draw_menu()
        elif self.scene == 'ending':
            self.draw_ending()
        else:
            self.draw_play()
        if self.help:
            self.draw_help()
        width, height = self.window.get_size()
        self.scale = min(width/PHYSICS.width, height/PHYSICS.height)
        size = (max(1, round(PHYSICS.width*self.scale)), max(1, round(PHYSICS.height*self.scale)))
        self.offset = ((width-size[0])//2, (height-size[1])//2)
        self.window.fill((15, 35, 52))
        self.window.blit(pygame.transform.smoothscale(self.canvas, size), self.offset)
        pygame.display.flip()

    def process_events(self):
        mx, my = pygame.mouse.get_pos()
        self.hover = ((mx-self.offset[0])/self.scale, (my-self.offset[1])/self.scale)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.scene in ('play', 'result') and not self.help:
                    self.restart()
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.scene == 'play' and not self.world.paused:
                    self.held_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.held_keys.discard(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                at = ((event.pos[0]-self.offset[0])/self.scale, (event.pos[1]-self.offset[1])/self.scale)
                for rect, callback in reversed(self.buttons):
                    if rect.collidepoint(at):
                        callback()
                        break
            elif event.type == pygame.WINDOWFOCUSLOST and self.world and self.scene == 'play':
                self.world.paused = True
                self.accumulator = 0.
                self.held_keys.clear()

    def run(self, frames=0):
        self.render()
        count = 0
        while self.running:
            seconds = 1/60 if self.headless else self.clock.tick(60)/1000
            self.process_events()
            direction = int(pygame.K_RIGHT in self.held_keys)-int(pygame.K_LEFT in self.held_keys)
            self.update(seconds, direction)
            self.render()
            count += 1
            if frames and count >= frames:
                break


def main():
    parser = argparse.ArgumentParser(description='Pingu Ball')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--screenshot')
    parser.add_argument('--stage', type=int, choices=range(8))
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--save-file')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pingu-') as temporary:
        save = args.save_file or (Path(temporary)/'progress.json' if args.headless else None)
        app = App(save, args.headless)
        try:
            if args.stage is not None:
                app.start(args.stage)
            if args.headless and not args.frames:
                app.render()
            else:
                app.run(args.frames)
            if args.screenshot:
                path = Path(args.screenshot)
                path.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(app.canvas, str(path))
        finally:
            pygame.quit()
