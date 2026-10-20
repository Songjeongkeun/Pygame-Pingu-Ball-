"""Pygame UI with a 1600 by 832 logical canvas and proportional resizing."""
import argparse
from array import array
from collections import OrderedDict
import math
import os
from pathlib import Path
import sys
import tempfile

import pygame
from .config import PHYSICS, REGULAR_STAGE_COUNT
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
DOUBLE_TAP_MS = 300


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
        self.scene = 'title'
        self.world = None
        self.running = True
        self.buttons = []
        self.held_keys = set()
        self.last_tap = None
        self.last_key_tap = None
        self.pending_power_direction = None
        self.active_finger = None
        self.hover = (0, 0)
        self.offset = (0, 0)
        self.scale = .625
        self.wall_time = 0.
        self.accumulator = 0.
        self.toast = ''
        self.toast_until = 0.
        self.new_record = False
        self.help = False
        self.lab_id = None
        self.sound_on = True
        self.sounds = {}
        self.music_ready = False
        self.music_playing = False
        self._font_cache = {}
        self._text_cache = OrderedDict()
        self._panel_cache = OrderedDict()
        self.font_path = self.find_font()
        self.background = pygame.image.load(str(ROOT/'assets'/'antarctica.png')).convert()
        self.background = pygame.transform.smoothscale(self.background, self.canvas.get_size())
        title = pygame.image.load(str(ROOT/'assets'/'title_screen.png')).convert()
        title_scale = max(self.canvas.get_width()/title.get_width(), self.canvas.get_height()/title.get_height())
        self.title_image = pygame.transform.smoothscale(title, (round(title.get_width()*title_scale), round(title.get_height()*title_scale)))
        self.title_rect = self.title_image.get_rect(center=self.canvas.get_rect().center)
        # Match the start button painted into the supplied artwork.
        self.title_start = pygame.Rect(round(self.title_rect.x+734*title_scale),
                                       round(self.title_rect.y+470*title_scale),
                                       round(580*title_scale), round(166*title_scale))
        self.skin = WinterSkin(ROOT)
        self.map_canvas = None
        self._scaled_map = None
        self._scaled_window = None
        self.previews = {}
        for level in self.levels:
            if level.index>0 and level.preview:
                image = pygame.image.load(str(ROOT/level.preview)).convert()
                self.previews[level.index] = pygame.transform.smoothscale(image, (124, 78))
        self.prepare_sounds()
        self.sync_music()

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
        render_key=(str(text),size,bold,tuple(color))
        image=self._text_cache.get(render_key)
        if image is None:
            image=self._font_cache[key].render(str(text),True,color)
            self._text_cache[render_key]=image
            if len(self._text_cache)>128: self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(render_key)
        rect = image.get_rect(center=(x, y)) if center else image.get_rect(topleft=(x, y))
        self.canvas.blit(image, rect)
        return rect

    def panel(self, rect, color=(249, 253, 255, 238), radius=20):
        rect = pygame.Rect(rect)
        key=(rect.size,tuple(color),radius)
        surface=self._panel_cache.get(key)
        if surface is None:
            surface=pygame.Surface(rect.size,pygame.SRCALPHA)
            pygame.draw.rect(surface,color,surface.get_rect(),border_radius=radius)
            self._panel_cache[key]=surface
            if len(self._panel_cache)>16: self._panel_cache.popitem(last=False)
        else:
            self._panel_cache.move_to_end(key)
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
        for name, hz, length in [('fish', 880, .10), ('switch', 440, .08),
                                 ('powerup', 1040, .12), ('power_use', 720, .12)]:
            frames = int(44100*length)
            samples = array('h', (int(6000*math.sin(2*math.pi*hz*i/44100)*(1-i/frames)**2) for i in range(frames)))
            self.sounds[name] = pygame.mixer.Sound(buffer=samples)
        self.sounds['fail'] = pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'fail_noot.mp3'))
        self.sounds['bounce'] = pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'bounce_seal.mp3'))
        self.sounds['clear'] = pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'goal_arrival.mp3'))
        pygame.mixer.music.load(str(ROOT/'assets'/'audio'/'title_opening.mp3'))
        pygame.mixer.music.set_volume(.5)
        self.music_ready = True

    def sync_music(self):
        if not self.music_ready or not pygame.mixer.get_init():
            return
        wanted = self.running and self.sound_on and self.scene in ('title', 'menu', 'normal_menu', 'lab_menu')
        if wanted:
            if not self.music_playing or not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(loops=-1)
                self.music_playing = True
        elif self.music_playing:
            pygame.mixer.music.stop()
            self.music_playing = False

    def sound(self, name):
        if self.sound_on and name in self.sounds:
            if name in ('fail', 'clear'):
                for other in ('bounce', 'fail', 'clear'):
                    self.sounds[other].stop()
            elif name == 'bounce':
                self.sounds[name].stop()
            self.sounds[name].play()

    def draw_pingu(self, x, y, size, charged=False):
        name='ball_'+charged if isinstance(charged,str) else 'ball_charged' if charged else 'ball'
        self.canvas.blit(self.skin.image(name, (size, size)),
                         (round(x-size/2), round(y-size/2)))

    def fish(self, x, y, size=32, bone=False, collected=False):
        image = self.skin.collectible(bone,size)
        if collected:
            image = image.copy(); image.set_alpha(90)
        self.canvas.blit(image, (round(x-image.get_width()/2), round(y-image.get_height()/2)))

    def igloo(self, rect):
        x, y, w, h = rect
        self.canvas.blit(self.skin.image('goal', (w, h)), (round(x), round(y)))

    def tile(self, block, time, switches, occupied_tiles=None):
        if block.id.startswith('spike-base-'):
            return
        image = self.skin.platform(block, time, switches)
        if image is None:
            return
        x, y, width, height = block.rect_at(time)
        if self.world and block.kind=='normal' and (width,height)==(64,64):
            above=((x,y-64) in occupied_tiles if occupied_tiles is not None else
                   any(q.id!=block.id and q.active(time,switches)
                       and q.rect_at(time)[:2]==(x,y-64) and (q.w,q.h)==(64,64)
                       for q in self.world.platforms))
            if above:
                image=self.skin.column(128).subsurface((0,64,64,64))
        self.canvas.blit(image, (round(x), round(y)))
        if block.kind == 'switch':
            label = {'gate-left':'A','gate-right':'B'}.get(block.group,block.group)
            self.text(label, x+width-12, y+height-16, 14, WHITE, True, True)

    def backdrop(self):
        self.canvas.blit(self.background, (0, 0))

    def draw_title(self):
        self.canvas.blit(self.title_image, self.title_rect)
        self.buttons.append((self.title_start, self.menu))
        self.button('소리 켜짐' if self.sound_on else '소리 꺼짐', (1320, 710, 200, 48), self.toggle_sound)

    def draw_menu(self):
        self.backdrop()
        self.text('Pingu Ball', 64, 64, 68, INK, True)
        self.text('자동으로 튀는 핑구 공을 이글루까지 안내하세요.', 68, 160, 25, MUTED)
        self.draw_pingu(1480, 168, 124)
        self.button('메인 화면', (920, 36, 180, 48), self.show_title)
        self.button('플레이 방법', (1120, 36, 190, 48), lambda: setattr(self, 'help', True))
        self.button('종료', (1330, 36, 190, 48), self.quit)
        self.text('스테이지 선택',64,232,30,INK,True)
        for x,title,detail,count,callback in [
                (64,'튜토리얼 스테이지','블록과 아이템을 차례대로 배워 보세요.','새 튜토리얼 10개 · 모두 선택 가능',self.show_lab_menu),
                (822,'일반 스테이지','다양한 맵에서 이글루까지 도전해요.',f'문서 스테이지 {REGULAR_STAGE_COUNT}개',self.show_normal_menu)]:
            self.panel((x,300,714,316))
            self.text(title,x+32,332,38,BLUE,True)
            self.text(detail,x+32,402,25,INK)
            self.text(count,x+32,450,22,MUTED)
            self.button('들어가기',(x+32,522,650,58),callback,True)
        self.button('소리 켜짐' if self.sound_on else '소리 꺼짐',(1320,710,200,48),self.toggle_sound)
        if self.progress.error:self.text(self.progress.error,64,782,20,CORAL)

    def show_normal_menu(self):
        self.lab_id = None
        self.scene = 'normal_menu'
        self.sync_music()
        self.accumulator = 0.
        self.held_keys.clear()
        self.clear_taps()
        self.help = False

    def draw_normal_menu(self):
        self.backdrop()
        self.text('일반 스테이지',64,48,48,INK,True)
        self.text(f'새 일반 스테이지 {REGULAR_STAGE_COUNT}개',68,116,24,MUTED)
        self.button('스테이지 선택',(1280,48,250,50),self.menu)
        completed = sum(str(i) in self.progress.records for i in range(1, REGULAR_STAGE_COUNT+1))
        self.text(f'{completed} / {REGULAR_STAGE_COUNT}개 완료 · 차례대로 도전하세요.',64,170,24,INK,True)
        for index, level in enumerate(self.levels[1:],1):
            col, row = (index-1) % 4, (index-1) // 4
            rect = pygame.Rect(64+col*372, 220+row*160, 350, 146)
            unlocked = index <= self.progress.unlocked
            self.panel(rect)
            self.text(f'STAGE {index}', rect.x+22, rect.y+16, 18, BLUE if unlocked else MUTED, True)
            self.text(level.title, rect.x+22, rect.y+49, 23, INK if unlocked else MUTED, True)
            if index in self.previews:
                self.canvas.blit(self.previews[index], (rect.right-138, rect.y+15))
            record = self.progress.records.get(str(index))
            if record:
                detail = f"{GRADES[record['score']]} · {record['score']}점 · {record['time']:.2f}초"
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
        self.lab_id = None
        self.world = World(self.levels[index])
        self.map_canvas = pygame.Surface((self.world.physics.width, self.world.physics.height), pygame.SRCALPHA).convert_alpha()
        self.map_view = self.map_viewport()
        self.scene = 'play'
        self.sync_music()
        self.accumulator = 0.
        self.held_keys.clear()
        self.clear_taps()
        self.toast = ''
        self.new_record = False

    def show_lab_menu(self):
        self.scene = 'lab_menu'
        self.sync_music()
        self.lab_id = None
        self.held_keys.clear()
        self.clear_taps()
        self.help = False

    def start_special(self, type_id):
        from .special_lab import make_lab, resolve_lab_type
        type_id = resolve_lab_type(type_id)
        self.lab_id = type_id
        self.world = World(make_lab(type_id))
        self.map_canvas = pygame.Surface((self.world.physics.width,self.world.physics.height),pygame.SRCALPHA).convert_alpha()
        self.map_view = self.map_viewport()
        self.scene = 'play'
        self.sync_music()
        self.accumulator = 0.
        self.held_keys.clear()
        self.clear_taps()
        self.toast = ''
        self.help = False

    def draw_lab_menu(self):
        from .special_lab import HINTS, TITLES, LAB_TYPES, LAB_NUMBERS
        self.backdrop()
        self.text('튜토리얼 스테이지',64,48,48,INK,True)
        self.text('블록과 아이템을 배우는 10개 튜토리얼 · R 재시작 · ← → 이동',68,116,24,MUTED)
        for n,code in enumerate(LAB_TYPES):
            col,row=n%4,n//4
            rect=pygame.Rect(64+col*372,188+row*164,350,144)
            self.panel(rect)
            self.text(f'{LAB_NUMBERS[code]:02d}',rect.x+18,rect.y+12,18,BLUE,True)
            self.text(TITLES[code],rect.x+18,rect.y+43,24,INK,True)
            hint=HINTS[code]
            font=pygame.font.Font(self.font_path,17)
            shown=hint
            while shown and font.size(shown+'…')[0]>rect.w-36: shown=shown[:-1]
            self.text(shown+('…' if shown!=hint else ''),rect.x+18,rect.y+99,17,MUTED)
            self.buttons.append((rect,lambda c=code:self.start_special(c)))
        self.button('스테이지 선택',(1280,48,250,50),self.menu)
        self.text('점프와 대시는 마지막 튜토리얼에서 함께 배웁니다.',64,728,22,MUTED)

    def quit(self):
        self.running = False
        self.sync_music()

    def toggle_sound(self):
        self.sound_on = not self.sound_on
        if not self.sound_on and pygame.mixer.get_init():
            pygame.mixer.stop()
        self.sync_music()

    def restart(self):
        if self.world:
            self.world.reset()
            self.scene = 'play'
            self.accumulator = 0.
            self.held_keys.clear()
            self.clear_taps()

    def menu(self):
        if self.lab_id is not None:
            self.show_lab_menu()
            return
        if (self.scene in ('play','result','ending') and self.world and self.world.level.index>0):
            self.show_normal_menu()
            return
        if self.scene in ('play','result') and self.world and self.world.level.index==0:
            self.show_lab_menu()
            return
        self.scene = 'menu'
        self.sync_music()
        self.accumulator = 0.
        self.held_keys.clear()
        self.help = False
        self.clear_taps()

    def show_title(self):
        self.scene = 'title'
        self.sync_music()
        self.accumulator = 0.
        self.held_keys.clear()
        self.help = False
        self.clear_taps()

    def toggle_pause(self):
        if self.world and self.scene == 'play':
            self.world.paused = not self.world.paused
            self.accumulator = 0.
            self.held_keys.clear()
            self.clear_taps()

    def clear_taps(self):
        self.last_tap = None
        self.last_key_tap = None
        self.pending_power_direction = None
        self.active_finger = None

    def arrow_down(self, key, timestamp, repeat=False):
        # Only fresh presses count. The key must be released between taps.
        if repeat or key in self.held_keys:
            return
        self.held_keys.add(key)
        if (self.scene != 'play' or not self.world or self.world.paused or self.help
                or self.world.cleared or self.world.failed):
            self.last_key_tap = None
            return
        kind = self.world.ball.power_up
        valid = ((kind == 'forward' and key in (pygame.K_LEFT, pygame.K_RIGHT))
                 or (kind == 'up' and key == pygame.K_UP))
        if not valid or {pygame.K_LEFT, pygame.K_RIGHT} <= self.held_keys:
            self.last_key_tap = None
            return
        previous = self.last_key_tap
        self.last_key_tap = (key, timestamp)
        if previous and previous[0] == key and 0 <= timestamp-previous[1] <= DOUBLE_TAP_MS:
            self.pending_power_direction = -1 if key == pygame.K_LEFT else 1 if key == pygame.K_RIGHT else 0
            self.last_key_tap = None
            self.last_tap = None

    def map_viewport(self):
        p = self.world.physics
        scale = min(1568/p.width, 704/p.height)
        size = (round(p.width*scale), round(p.height*scale))
        return pygame.Rect((1600-size[0])//2, 72, *size)

    def pointer_down(self, position, timestamp):
        at = ((position[0]-self.offset[0])/self.scale, (position[1]-self.offset[1])/self.scale)
        for rect, callback in reversed(self.buttons):
            if rect.collidepoint(at):
                self.clear_taps()
                callback()
                return
        if (self.scene != 'play' or self.world.paused or self.help
                or not self.map_view.collidepoint(at) or self.world.ball.power_up is None):
            self.last_tap = None
            return
        previous = self.last_tap
        self.last_tap = (timestamp, at)
        if previous and 0 <= timestamp-previous[0] <= DOUBLE_TAP_MS and math.dist(at, previous[1]) <= 64:
            ball_x = self.map_view.x+self.world.ball.x*self.map_view.width/self.world.physics.width
            self.pending_power_direction = 1 if at[0] >= ball_x else -1
            self.last_tap = None

    def draw_play(self):
        world = self.world
        self.backdrop()
        screen = self.canvas
        self.canvas = self.map_canvas
        try:
            self.draw_map(world)
        finally:
            self.canvas = screen
        self.map_view = self.map_viewport()
        if self._scaled_map is None or self._scaled_map.get_size()!=self.map_view.size:
            self._scaled_map=pygame.Surface(self.map_view.size,pygame.SRCALPHA).convert_alpha()
        pygame.transform.smoothscale(self.map_canvas,self.map_view.size,self._scaled_map)
        self.canvas.blit(self._scaled_map,self.map_view)
        self.draw_play_hud(world)

    def draw_map(self, world):
        self.canvas.fill((0, 0, 0, 0))
        from .special_draw import draw_patrols, draw_movement_markers
        draw_movement_markers(self.canvas,world)
        draw_patrols(self.canvas,world,self.skin)
        groups = {}
        for block in world.platforms:
            if block.kind == 'moving':
                prefix = block.id.rsplit('-', 1)[0]
                groups.setdefault(prefix, []).append(block)
        if world.level.show_movement_rails:
            for blocks in groups.values():
                left = min(q.x for q in blocks)
                right = max(q.x+q.w+q.travel for q in blocks)
                y = blocks[0].y-12
                pygame.draw.line(self.canvas, (86, 185, 213), (left,y), (right,y), 2)
                for x in (left, right):
                    pygame.draw.line(self.canvas, (86, 185, 213), (x,y-5), (x,y+5), 2)
        # Only a column's exposed upper tile has snow. Horizontal sprite
        # edges remain native, preserving the approved natural seams.
        positions={(q.x,q.y):q for q in world.platforms if q.kind=='normal' and (q.w,q.h)==(64,64)
                   and not hasattr(q,'device')}
        platform_rects={q.id:q.rect_at(world.time) for q in world.platforms}
        occupied_tiles={platform_rects[q.id][:2] for q in world.platforms
                        if (q.w,q.h)==(64,64) and q.active(world.time,world.switches)}
        drawn=set()
        for block in sorted(world.platforms,key=lambda q:q.y):
            if block.id in drawn: continue
            column=[block]
            if (block.x,block.y) in positions:
                while (block.x,block.y+64*len(column)) in positions:
                    column.append(positions[block.x,block.y+64*len(column)])
            if len(column)>1:
                self.canvas.blit(self.skin.column(64*len(column)),(round(block.x),round(block.y)))
                drawn.update(q.id for q in column)
            else:
                self.tile(block, world.time, world.switches,occupied_tiles)
        for d in world.special.devices.values():
            if d.type_id==508:
                x=d.x+d.w/2
                for y in range(round(d.y+32),round(d.y+d.options.get('travel',128.)+32),13):
                    pygame.draw.line(self.canvas,(86,185,213),(x,y),(x,y+6),2)
        for spike in world.level.spikes:
            self.canvas.blit(self.skin.image('spike', (spike.w, spike.h+(16 if spike.teeth == 6 else 0)+4)),
                             (round(spike.x), round(spike.y)))
        from .special_draw import draw_runtime
        draw_runtime(self.canvas,world,self.skin,platform_rects=platform_rects.values())
        for index, fish in enumerate(world.level.collectibles):
            if index not in world.collected:
                self.fish(fish.x, fish.y, int(fish.size), fish.value < 0)
        for index, item in enumerate(world.level.powerups):
            if index not in world.collected_powerups:
                self.canvas.blit(self.skin.image('power_'+item.kind, (item.size, item.size)),
                                 (round(item.x-item.size/2), round(item.y-item.size/2)))
        gx,gy,gw,gh=world.level.goal
        self.igloo((gx,gy+4,gw,gh))
        self.draw_pingu(world.ball.x, world.ball.y, 28, charged=world.ball.power_up or False)

    def draw_play_hud(self, world):
        # Top row belongs to UI; the map never scrolls.
        self.panel((0, 0, 1600, 64), (249, 253, 255, 245), 0)
        label = '튜토리얼' if self.lab_id is not None or world.level.index == 0 else f'STAGE {world.level.index}'
        self.text(label+' / '+world.level.title, 24, 18, 24, INK, True)
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
        self.button('튜토리얼 목록' if self.lab_id is not None else '스테이지', (1360, 10, 216, 44), self.menu)
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
        if self.lab_id is not None:
            self.text('튜토리얼 CLEAR',800,260,40,BLUE,True,True)
            self.text(f'{world.level.title} · {world.time:.2f}초',800,350,26,INK,True,True)
            self.button('다시 시험',(520,516,270,54),self.restart)
            from .special_lab import ENABLED_LAB_TYPES
            self.button('다음 튜토리얼' if self.lab_id!=ENABLED_LAB_TYPES[-1] else '튜토리얼 목록',
                        (810,516,270,54),self.next_stage,True)
            self.button('튜토리얼 목록',(620,594,360,44),self.show_lab_menu)
            return
        tutorial = world.level.index == 0
        self.text('튜토리얼 완료!' if tutorial else '집에 도착했어요!', 800, 224, 36, INK, True, True)
        self.text('CLEAR' if tutorial else world.grade, 800, 320, 68, BLUE, True, True)
        if not tutorial:
            self.text(f'{world.score}점 · {world.time:.2f}초', 800, 396, 28, INK, True, True)
            self.text('최고 기록 갱신' if self.new_record else '기존 최고 기록 유지', 800, 446, 22, TEAL, True, True)
        if self.progress.error:
            self.text(self.progress.error, 800, 484, 18, CORAL, False, True)
        self.button('다시 도전', (520, 516, 270, 54), self.restart)
        self.button('특수블록 배우기' if tutorial else '여행 완료' if world.level.index == REGULAR_STAGE_COUNT else '다음 스테이지', (810, 516, 270, 54), self.next_stage, True)
        self.button('스테이지 선택', (620, 594, 360, 44), self.menu)

    def next_stage(self):
        if self.lab_id is not None:
            from .special_lab import ENABLED_LAB_TYPES
            codes=ENABLED_LAB_TYPES
            if self.lab_id in codes and codes.index(self.lab_id)+1<len(codes):
                self.start_special(codes[codes.index(self.lab_id)+1])
            else:
                self.show_lab_menu()
            return
        if self.world.level.index == 0:
            self.show_lab_menu()
        elif self.world.level.index == REGULAR_STAGE_COUNT:
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
        self.overlay((340, 80, 920, 680))
        self.text('플레이 방법', 390, 112, 36, INK, True)
        rows = [('← → / R', '좌우 이동 / 처음부터 재시작 · 입장 즉시 자동 반동'),
                ('생선 / 생선 가시', '+1점 / -1점 · 모두 선택 사항'),
                ('금 간 얼음', '밟는 즉시 사라짐 · 착지 반동은 유지'),
                ('점프 화살표 ↑', '일반 발판보다 더 높이 튕깁니다.'),
                ('이동 ↔ / 시간 발판', '표시 구간 왕복 이동 / 주기적 등장과 소멸'),
                ('파랑 » / 노랑 ↑↑', '대시 / 점프 · 함께 보유하면 대시부터 사용'),
                ('아이템 사용', '←←·→→ / ↑↑ 또는 화면 더블 클릭·터치'),
                ('빨간 스위치', '연결 다리 전환 · 완전히 벗어난 뒤 재작동'),
                ('좌우 발사 화살표', '중심 높이에서 발사 · 방향키 또는 벽 충돌로 해제'),
                ('이글루', '생선 획득 여부와 관계없이 도착하면 클리어')]
        for index, (label, body) in enumerate(rows):
            self.text(label, 390, 180+index*46, 22, BLUE, True)
            self.text(body, 610, 180+index*46, 21)
        self.button('알겠어요', (1000, 680, 200, 50), lambda: setattr(self, 'help', False), True)

    def update(self, seconds, direction):
        self.wall_time += seconds
        if self.scene != 'play' or not self.world or self.world.paused or self.help:
            self.accumulator = 0.
            self.clear_taps()
            return
        self.accumulator += seconds
        while self.accumulator+1e-12 >= PHYSICS.dt:
            power_direction = self.pending_power_direction
            self.pending_power_direction = None
            self.world.step(direction, use_power=power_direction is not None,
                            power_direction=power_direction or 0)
            if power_direction is not None:
                self.clear_taps()
            self.accumulator -= PHYSICS.dt
            for kind, payload in self.world.events:
                if kind in self.sounds:
                    self.sound(kind)
                if kind == 'fail':
                    self.toast = payload
                    self.toast_until = self.wall_time+1.7
                    self.held_keys.clear()
                    self.clear_taps()
                    self.accumulator = 0.
                    return
            if self.world.cleared:
                score = None if self.world.level.index == 0 else self.world.score
                if self.lab_id is not None:
                    from .special_lab import LAB_NUMBERS
                    self.new_record=self.progress.complete_tutorial(LAB_NUMBERS[self.lab_id],self.world.time)
                else:
                    self.new_record=self.progress.complete(self.world.level.index,score,self.world.time)
                self.scene = 'result'
                self.accumulator = 0.
                break

    def render(self):
        self.sync_music()
        self.buttons = []
        if self.scene == 'title':
            self.draw_title()
        elif self.scene == 'menu':
            self.draw_menu()
        elif self.scene == 'normal_menu':
            self.draw_normal_menu()
        elif self.scene == 'lab_menu':
            self.draw_lab_menu()
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
        if size==self.canvas.get_size():
            self.window.blit(self.canvas,self.offset)
        else:
            if self._scaled_window is None or self._scaled_window.get_size()!=size:
                self._scaled_window=pygame.Surface(size).convert()
            pygame.transform.smoothscale(self.canvas,size,self._scaled_window)
            self.window.blit(self._scaled_window,self.offset)
        pygame.display.flip()

    def process_events(self):
        mx, my = pygame.mouse.get_pos()
        self.hover = ((mx-self.offset[0])/self.scale, (my-self.offset[1])/self.scale)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                if self.scene == 'title' and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.menu()
                elif self.scene == 'menu' and event.key == pygame.K_ESCAPE and not self.help:
                    self.show_title()
                elif self.scene in ('lab_menu','normal_menu') and event.key == pygame.K_ESCAPE:
                    self.menu()
                elif event.key == pygame.K_r and self.scene in ('play', 'result') and not self.help:
                    self.restart()
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN) and self.scene == 'play' and not self.world.paused and not self.help:
                    self.arrow_down(event.key, getattr(event,'timestamp',pygame.time.get_ticks()),
                                    repeat=getattr(event,'repeat',False))
            elif event.type == pygame.KEYUP:
                self.held_keys.discard(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not getattr(event, 'touch', False):
                self.pointer_down(event.pos, getattr(event, 'timestamp', pygame.time.get_ticks()))
            elif event.type == pygame.FINGERDOWN and self.active_finger is None:
                self.active_finger = event.finger_id
                width, height = self.window.get_size()
                self.pointer_down((event.x*width, event.y*height),
                                  getattr(event, 'timestamp', pygame.time.get_ticks()))
            elif event.type == pygame.FINGERUP and event.finger_id == self.active_finger:
                self.active_finger = None
            elif event.type == pygame.WINDOWFOCUSLOST and self.world and self.scene == 'play':
                self.world.paused = True
                self.accumulator = 0.
                self.held_keys.clear()
                self.clear_taps()

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
    from .special_lab import LAB_NUMBERS, ENABLED_LAB_TYPES
    parser.add_argument('--stage', type=int, choices=range(1,REGULAR_STAGE_COUNT+1))
    parser.add_argument('--special', type=int, choices=tuple(LAB_NUMBERS[c] for c in ENABLED_LAB_TYPES)+ENABLED_LAB_TYPES,
                        metavar='NUMBER',help='Open a document tutorial, numbered 1-10')
    parser.add_argument('--lab',action='store_true',help='Open the ten document tutorials')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--save-file')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pingu-') as temporary:
        save = args.save_file or (Path(temporary)/'progress.json' if args.headless else None)
        app = App(save, args.headless)
        try:
            if args.special is not None:
                app.start_special(args.special)
            elif args.lab:
                app.show_lab_menu()
            elif args.stage is not None:
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
