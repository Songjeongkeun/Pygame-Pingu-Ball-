"""Pingu Ball's native Pygame interface. 1200x600 logical, responsive letterbox."""
import argparse
from array import array
import math
import os
from pathlib import Path
import random
import tempfile

import pygame
from .config import PHYSICS
from .engine import World
from .levels import make_levels, waypoint_direction, condition_met
from .storage import Progress

ROOT=Path(__file__).resolve().parent.parent
INK=(23,48,70)
MUTED=(91,116,137)
WHITE=(247,252,255)
BLUE=(39,119,184)
GOLD=(239,173,55)
CORAL=(212,90,101)
TEAL=(39,166,172)


class App:
    def __init__(self, save_path=None, headless=False):
        self.headless=headless
        self.demo=False
        self.demo_point=0
        if headless:
            os.environ.setdefault('SDL_VIDEODRIVER','dummy')
            os.environ.setdefault('SDL_AUDIODRIVER','dummy')
        pygame.mixer.pre_init(44100,-16,1,512)
        pygame.init()
        self.window=pygame.display.set_mode((1200,600),pygame.RESIZABLE)
        pygame.display.set_caption('Pingu Ball')
        self.canvas=pygame.Surface((1200,600)).convert()
        self.clock=pygame.time.Clock()
        self.levels=make_levels()
        self.progress=Progress(save_path or ROOT/'save_data'/'progress.json')
        self.scene='menu'
        self.world=None
        self.running=True
        self.buttons=[]
        self.hover=(0,0)
        self.offset=(0,0)
        self.scale=1.
        self.wall_time=0.
        self.accumulator=0.
        self.toast=''
        self.toast_until=0.
        self.new_record=False
        self.help=False
        self.sound_on=True
        self.sounds={}
        self.particles=[]
        self.trail=[]
        self._font_cache={}
        self.font_path=self.find_font()
        self.background=pygame.image.load(str(ROOT/'assets'/'antarctica.png')).convert()
        self.background=pygame.transform.smoothscale(self.background,(1200,600))
        self.pingu=None
        sprite=ROOT/'assets'/'pingu.png'
        if sprite.exists():
            image=pygame.image.load(str(sprite)).convert_alpha()
            self.pingu=image.subsurface(image.get_bounding_rect()).copy()
        self.sprites={}
        self.prepare_sounds()
        self.snow=[(random.Random(i).randrange(1200),random.Random(i+100).randrange(600),1+i%2) for i in range(40)]

    def find_font(self):
        for p in ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
                  'C:/Windows/Fonts/malgun.ttf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
            if Path(p).exists():return p
        return pygame.font.match_font('malgungothic,applesdgothicneo,notosanscjkkr,dejavusans')

    def font(self,size,bold=False):
        key=(size,bold)
        if key not in self._font_cache:
            f=pygame.font.Font(self.font_path,size);f.set_bold(bold)
            self._font_cache[key]=f
        return self._font_cache[key]

    def text(self,text,x,y,size=16,color=INK,bold=False,center=False):
        image=self.font(size,bold).render(str(text),True,color)
        rect=image.get_rect(center=(x,y)) if center else image.get_rect(topleft=(x,y))
        self.canvas.blit(image,rect)
        return rect

    def panel(self,rect,color=(255,255,255,220),radius=20):
        rect=pygame.Rect(rect)
        surf=pygame.Surface(rect.size,pygame.SRCALPHA)
        pygame.draw.rect(surf,color,surf.get_rect(),border_radius=radius)
        self.canvas.blit(surf,rect)

    def button(self,label,rect,callback,primary=False,enabled=True):
        rect=pygame.Rect(rect)
        hovered=enabled and rect.collidepoint(self.hover)
        color=(31,96,146) if primary and hovered else BLUE if primary else (224,240,249) if hovered else (244,249,253)
        if not enabled:color=(223,232,239)
        pygame.draw.rect(self.canvas,color,rect,border_radius=12)
        self.text(label,*rect.center,16,WHITE if primary else INK if enabled else MUTED,True,True)
        if enabled:self.buttons.append((rect,callback))

    def prepare_sounds(self):
        if not pygame.mixer.get_init():return
        for name,hz,length in [('fish',880,.10),('fail',160,.16),('clear',660,.24),('switch',440,.08)]:
            frames=int(44100*length)
            samples=array('h',(int(6000*math.sin(2*math.pi*hz*i/44100)*(1-i/frames)**2) for i in range(frames)))
            try:self.sounds[name]=pygame.mixer.Sound(buffer=samples)
            except pygame.error:pass

    def sound(self,name):
        if self.sound_on and name in self.sounds:self.sounds[name].play()

    def draw_pingu(self,x,y,size):
        if self.pingu:
            if size not in self.sprites:
                self.sprites[size]=pygame.transform.smoothscale(self.pingu,(size,size))
            self.canvas.blit(self.sprites[size],(round(x-size/2),round(y-size/2)))
        else:
            pygame.draw.circle(self.canvas,(25,30,44),(round(x),round(y)),max(1,size//2))
            pygame.draw.circle(self.canvas,(242,245,243),(round(x-size*.20),round(y-size*.13)),max(1,size//9))
            pygame.draw.circle(self.canvas,(242,245,243),(round(x+size*.20),round(y-size*.13)),max(1,size//9))
            pygame.draw.ellipse(self.canvas,(242,62,61),(x-size*.22,y,size*.44,size*.22))

    def fish(self,x,y,size=14,bone=False,collected=False):
        color=(174,112,171) if bone else GOLD
        if collected:color=(169,187,196)
        if bone:
            pygame.draw.line(self.canvas,color,(x-size*.4,y),(x+size*.25,y),2)
            for offset in [-.2,0,.2]:
                pygame.draw.line(self.canvas,color,(x+size*offset,y-size*.24),(x+size*offset+.7,y+size*.24),2)
            pygame.draw.polygon(self.canvas,color,[(x+size*.23,y),(x+size*.5,y-size*.3),(x+size*.5,y+size*.3)])
            pygame.draw.circle(self.canvas,color,(round(x-size*.38),round(y)),max(2,int(size*.16)))
        else:
            pygame.draw.polygon(self.canvas,color,[(x+size*.12,y),(x+size*.5,y-size*.3),(x+size*.5,y+size*.3)])
            pygame.draw.ellipse(self.canvas,color,(x-size*.5,y-size*.3,size*.8,size*.6))
            pygame.draw.circle(self.canvas,INK,(round(x-size*.28),round(y-size*.05)),max(1,int(size*.055)))
            pygame.draw.line(self.canvas,(255,222,141),(x-size*.25,y-size*.17),(x+size*.1,y-size*.17),1)

    def igloo(self,x,y,size=20):
        rect=pygame.Rect(round(x),round(y),size,size)
        old_clip=self.canvas.get_clip()
        self.canvas.set_clip(rect)
        pygame.draw.ellipse(self.canvas,(197,220,233),(x-2,y+size*.6,size+4,size*.5))
        pygame.draw.ellipse(self.canvas,(245,251,253),(x,y,size,size*1.5))
        pygame.draw.rect(self.canvas,(231,244,250),(x,y+size*.55,size,size*.45))
        pygame.draw.line(self.canvas,(164,196,214),(x+2,y+size*.4),(x+size-2,y+size*.4),1)
        pygame.draw.line(self.canvas,(164,196,214),(x+2,y+size*.65),(x+size-2,y+size*.65),1)
        pygame.draw.arc(self.canvas,(164,196,214),(x,y,size,size*1.1),0,math.pi,1)
        pygame.draw.ellipse(self.canvas,(42,90,121),(x+size*.35,y+size*.48,size*.34,size*.65))
        pygame.draw.rect(self.canvas,(42,90,121),(x+size*.35,y+size*.7,size*.34,size*.3))
        self.canvas.set_clip(old_clip)

    def tile(self,q,time,switches):
        x,y,w,h=q.rect_at(time)
        active=q.active(time,switches)
        if not active:
            if q.kind=='timed':
                ghost=pygame.Surface((int(w),20),pygame.SRCALPHA)
                for sx in range(0,int(w),20):pygame.draw.rect(ghost,(45,161,176,70),(sx+1,1,18,18),1,border_radius=3)
                self.canvas.blit(ghost,(x,y))
            return
        if q.kind=='moving':
            pygame.draw.line(self.canvas,(109,161,189),(q.x+10,y+29),(q.x+q.travel+10,y+29),1)
            for tx in range(int(q.x),int(q.x+q.travel)+1,10):pygame.draw.circle(self.canvas,(109,161,189),(tx+10,int(y+29)),1)
        palette={'normal':((146,194,220),(88,149,188)),
                 'breakable':((166,201,220),(115,150,175)),
                 'spring':((228,150,171),(173,83,120)),
                 'moving':((92,160,214),(36,109,163)),
                 'timed':((91,207,205),(34,151,160)),
                 'linked':((114,190,157),(58,139,104)),
                 'switch':((150,192,214),(90,141,175)),
                 'launcher':((130,135,213),(78,81,170))}
        fill,edge=palette[q.kind]
        if q.kind=='timed' and (time+q.phase)%q.period > q.active_for-.5 and int(time*12)%2:
            fill=(204,242,240)
        for sx in range(0,round(w),20):
            tile=pygame.Rect(round(x+sx),round(y),20,20)
            pygame.draw.rect(self.canvas,edge,tile,border_radius=3)
            pygame.draw.rect(self.canvas,fill,tile.inflate(-2,-2),border_radius=3)
            pygame.draw.line(self.canvas,(220,241,250),(x+sx+3,y+4),(x+sx+16,y+4),2)
            pygame.draw.lines(self.canvas,(234,247,252),False,[(x+sx+1,y+3),(x+sx+5,y+1),(x+sx+11,y+3),(x+sx+18,y+1)],2)
            if q.kind=='breakable':
                remaining=1. if q.break_at is None else max(0.,(q.break_at-time)/PHYSICS.break_delay)
                crack=(118,139,160) if remaining>.3 else CORAL
                pygame.draw.lines(self.canvas,crack,False,[(x+sx+12,y+4),(x+sx+8,y+9),(x+sx+13,y+13),(x+sx+7,y+18)],1)
            if q.kind=='timed':
                pygame.draw.circle(self.canvas,WHITE,(round(x+sx+10),round(y+11)),5,1)
                pygame.draw.line(self.canvas,WHITE,(x+sx+10,y+7),(x+sx+10,y+11),1)
            if q.kind in ('moving','launcher'):
                sign=q.arrow if q.kind=='launcher' else 1
                pygame.draw.line(self.canvas,WHITE,(x+sx+5,y+11),(x+sx+15,y+11),2)
                pygame.draw.lines(self.canvas,WHITE,False,[(x+sx+11 if sign>0 else x+sx+9,y+7),(x+sx+15 if sign>0 else x+sx+5,y+11),(x+sx+11 if sign>0 else x+sx+9,y+15)],2)
            if q.kind=='spring':
                pygame.draw.lines(self.canvas,WHITE,False,[(x+sx+5,y+13),(x+sx+10,y+8),(x+sx+15,y+13)],2)
        if q.kind=='switch':
            on=switches.get(q.group,False)
            pygame.draw.rect(self.canvas,(102,180,139) if on else CORAL,(x+3,y-4,14,5),border_radius=2)
            self.text(q.group,x+w/2,y+12,10,WHITE,True,True)
        if q.kind=='breakable' and q.break_at is not None:
            width=max(0.,min(w,(q.break_at-time)/PHYSICS.break_delay*w))
            pygame.draw.line(self.canvas,CORAL,(x,y-4),(x+width,y-4),2)

    def backdrop(self,darken=False):
        self.canvas.blit(self.background,(0,0))
        veil=pygame.Surface((1200,600),pygame.SRCALPHA)
        veil.fill((231,245,252,55) if not darken else (19,51,73,75))
        self.canvas.blit(veil,(0,0))
        for x,y,r in self.snow:
            pygame.draw.circle(self.canvas,(239,250,255),(int((x+self.wall_time*5)%1200),int((y+self.wall_time*(6+r))%600)),r)

    def draw_menu(self):
        self.backdrop()
        self.text('ANTARCTIC ADVENTURE',60,42,13,MUTED,True)
        self.text('Pingu Ball',56,66,52,INK,True)
        self.text('작은 바운스로, 집까지.',60,130,19,MUTED)
        self.draw_pingu(1080,136,82)
        self.button('플레이 방법',(924,36,110,36),lambda:setattr(self,'help',True))
        self.button('종료',(1046,36,92,36),self.quit)
        completed=sum(str(i) in self.progress.records for i in range(1,8))
        self.text(f'스테이지 선택   /   {completed} OF 7 COMPLETE',60,190,15,INK,True)
        self.text('클리어하면 다음 스테이지가 열립니다. 생선은 선택 사항입니다.',60,216,13,MUTED)
        for i,level in enumerate(self.levels):
            col,row=i%4,i//4
            rect=pygame.Rect(60+col*275,255+row*137,255,119)
            unlocked=i<=self.progress.unlocked
            hover=rect.collidepoint(self.hover) and unlocked
            self.panel(rect,(255,255,255,244 if hover else 215),16)
            record=self.progress.records.get(str(i))
            tag='INTRO' if i==0 else f'{i:02d}'
            self.text(tag,rect.x+18,rect.y+13,13,BLUE if unlocked else MUTED,True)
            self.text(level.title,rect.x+18,rect.y+37,19,INK if unlocked else MUTED,True)
            if record:
                grade='CLEAR' if i==0 else {6:'B',7:'A',8:'S',9:'SS',10:'SSS'}[record['score']]
                self.text(f"{grade}  ·  {record['time']:.2f}s",rect.x+18,rect.y+79,13,TEAL,True)
                for j in range(3 if i else 1):self.fish(rect.right-25-j*20,rect.y+91,14,collected=i!=0 and record['score']<8+j)
            else:self.text('PLAY  →' if unlocked else '앞 스테이지 클리어 필요',rect.x+18,rect.y+80,12,BLUE if unlocked else MUTED,True)
            if unlocked:self.buttons.append((rect,lambda n=i:self.start(n)))
        self.text('← → 이동    ·    자동 반동    ·    R 다시 시작',60,559,13,INK)
        self.button('소리 켜짐' if self.sound_on else '소리 꺼짐',(1010,550,128,32),self.toggle_sound)
        if self.progress.error:self.text(self.progress.error,410,560,12,CORAL)

    def start(self,index):
        self.world=World(self.levels[index]);self.scene='play';self.accumulator=0.;self.trail=[];self.particles=[]
        self.toast='';self.new_record=False
        self.demo_point=0

    def quit(self):self.running=False
    def toggle_sound(self):self.sound_on=not self.sound_on

    def restart(self):
        if self.world:
            self.world.reset();self.scene='play';self.accumulator=0.;self.trail=[];self.particles=[]

    def menu(self):
        self.scene='menu';self.accumulator=0.;self.help=False

    def toggle_pause(self):
        if self.world and self.scene=='play':
            self.world.paused=not self.world.paused;self.accumulator=0.

    def draw_play(self):
        w=self.world
        self.backdrop()
        # Guides show the two valid route families without drawing a solution.
        self.text('선택 도전  /  생선 '+('1개' if w.level.index==0 else '3개'),445,282,12,MUTED)
        self.text('HOME',1128,449,10,BLUE,True,True)
        for q in w.platforms:self.tile(q,w.time,w.switches)
        for spike in w.level.spikes:
            for tri in spike.triangles:
                pygame.draw.polygon(self.canvas,(101,151,181),tri)
                inset=[(x+1,y-1) if j!=1 else (x,y+3) for j,(x,y) in enumerate(tri)]
                pygame.draw.polygon(self.canvas,(232,247,255),inset)
        for i,fish in enumerate(w.level.collectibles):
            if i not in w.collected:
                # Only the halo animates; the collectible and its hitbox do not move.
                glow=pygame.Surface((32,32),pygame.SRCALPHA)
                pygame.draw.circle(glow,(239,173,55,30) if fish.value>0 else (158,101,169,28),(16,16),13)
                self.canvas.blit(glow,(fish.x-16,fish.y-16))
                self.fish(fish.x,fish.y,14,fish.value<0)
        self.igloo(*w.level.goal[:2],20)
        for j,(x,y) in enumerate(self.trail[-12:]):
            pygame.draw.circle(self.canvas,(192,223,239),(round(x),round(y)),1)
        self.draw_pingu(w.ball.x,w.ball.y,9)
        if not w.started:
            pygame.draw.circle(self.canvas,BLUE,(round(w.ball.x),round(w.ball.y)),12,1)
            self.text('PINGU',w.ball.x, w.ball.y-24,10,BLUE,True,True)
        self.panel((20,16,1160,66),(249,253,255,235),18)
        self.text('Pingu Ball',38,27,23,INK,True)
        self.text(('INTRO' if w.level.index==0 else f'STAGE {w.level.index:02d}')+'  /  '+w.level.title,225,27,17,INK,True)
        self.text(f'{w.time:05.2f}s',225,53,12,MUTED)
        for j in range(1 if w.level.index==0 else 3):
            self.fish(560+j*27,40,20,collected=j>=w.plus)
        self.fish(655,40,20,True,collected=w.minus==0)
        self.text(f'{w.plus}/{1 if w.level.index==0 else 3}   ·   생선 가시 {w.minus}',547,58,11,MUTED)
        self.button('일시정지',(778,30,106,34),self.toggle_pause)
        self.button('재시작 R',(894,30,106,34),self.restart)
        self.button('스테이지',(1010,30,142,34),self.menu)
        self.panel((20,547,1160,36),(249,253,255,220),12)
        self.text(w.level.subtitle,36,556,13,INK)
        self.text('← → 이동   ·   R 재시작',958,557,12,MUTED)
        if self.toast and self.wall_time<self.toast_until:
            self.panel((435,102,330,40),(248,251,253,240),12)
            self.text(self.toast,600,122,15,CORAL,True,True)
        if not w.started and not w.paused:
            self.panel((420,140,360,84),(249,253,255,235),16)
            self.text('집으로 돌아갈 준비가 됐나요?',600,167,19,INK,True,True)
            self.text('← 또는 → 를 누르면 시작합니다',600,196,14,MUTED,False,True)
        # Special block states are always visible and use the same color as tiles.
        states=[]
        for q in w.platforms:
            if q.kind=='timed':states.append('시간 발판 '+('ON' if q.active(w.time,w.switches) else 'OFF'))
        if any(q.kind=='moving' for q in w.platforms):states.append('왕복 100px/s')
        for key,value in w.switches.items():states.append(f'스위치 {key} '+('ON' if value else 'OFF'))
        if w.ball.launch:states.append('수평 발사 · 방향키로 해제')
        if states:self.text('  /  '.join(dict.fromkeys(states)),32,97,12,INK)
        if w.paused:self.draw_pause()
        if self.scene=='result':self.draw_result()

    def overlay(self):
        shade=pygame.Surface((1200,600),pygame.SRCALPHA);shade.fill((19,43,63,128));self.canvas.blit(shade,(0,0));self.buttons=[]

    def draw_pause(self):
        self.overlay();self.panel((395,155,410,290),(249,253,255,250),24)
        self.text('잠깐 쉬어 가요',600,204,28,INK,True,True)
        self.text('공과 모든 블록의 시간이 멈췄습니다.',600,244,14,MUTED,False,True)
        self.button('계속하기',(440,282,320,46),self.toggle_pause,True)
        self.button('다시 시작',(440,340,154,42),self.restart)
        self.button('스테이지 선택',(606,340,154,42),self.menu)

    def draw_result(self):
        w=self.world
        self.overlay();self.panel((365,110,470,382),(249,253,255,250),24)
        self.text('HOME, SWEET HOME',600,145,13,TEAL,True,True)
        self.text('튜토리얼 완료!' if w.level.index==0 else '집에 도착했어요!',600,179,27,INK,True,True)
        self.text('CLEAR' if w.level.index==0 else w.grade,600,239,52,BLUE,True,True)
        self.text(f"{w.time:.2f}초" if w.level.index==0 else f'{w.score}점   ·   {w.time:.2f}초',600,288,18,INK,True,True)
        self.text('최고 기록 갱신' if self.new_record else '기존 최고 기록 유지',600,319,13,TEAL if self.new_record else MUTED,True,True)
        if self.progress.error:self.text(self.progress.error,600,345,12,CORAL,False,True)
        self.button('다시 도전',(400,367,190,44),self.restart)
        self.button('여행 완료' if w.level.index==7 else '다음 스테이지',(610,367,190,44),self.next_stage,True)
        self.button('스테이지 선택',(475,430,250,36),self.menu)

    def next_stage(self):
        if self.world.level.index==7:self.scene='ending'
        else:self.start(self.world.level.index+1)

    def draw_ending(self):
        self.backdrop();self.panel((290,100,620,400),(249,253,255,235),28)
        self.draw_pingu(600,181,95)
        self.text('남극 여행을 마쳤어요!',600,278,34,INK,True,True)
        self.text('이제 모든 스테이지에서 SSS 기록에 도전해 보세요.',600,326,17,MUTED,False,True)
        self.button('스테이지 다시 둘러보기',(420,383,360,48),self.menu,True)

    def draw_help(self):
        self.overlay();self.panel((285,75,630,450),(249,253,255,250),24)
        self.text('Pingu Ball 플레이 방법',322,105,26,INK,True)
        rows=[('← →','좌우 이동 · 공중에서도 가속과 감속'),('R','처음부터 다시 시작'),
              ('생선 / 생선 가시','각각 +1점 / −1점. 모두 선택 사항'),
              ('금 간 얼음','착지 0.5초 후 파괴'),('분홍 점프대','90px 반동 · 일반 발판은 60px'),
              ('파랑 / 청록','왕복 이동 발판 / 주기적으로 사라지는 발판'),
              ('빨간 스위치','A 다리 전환 · 완전히 벗어난 뒤 재작동'),
              ('보라 화살표','수평 발사 · 좌우 입력이나 벽 충돌로 해제')]
        for i,(label,body) in enumerate(rows):
            self.text(label,322,157+i*35,14,BLUE,True)
            self.text(body,475,157+i*35,13,INK)
        self.button('알겠어요',(740,462,135,38),lambda:setattr(self,'help',False),True)

    def update(self,seconds,direction):
        self.wall_time+=seconds
        if self.scene!='play' or not self.world or self.world.paused or self.help:
            self.accumulator=0.;return
        self.accumulator+=min(seconds,.1)
        while self.accumulator>=PHYSICS.dt:
            if self.demo:
                path=self.world.level.fish_waypoints
                target,condition=path[min(self.demo_point,len(path)-1)]
                ready=condition_met(self.world,condition)
                if condition.startswith('land:'):
                    ready=ready and abs(self.world.ball.x-target)<2 and self.world.time-self.world.last_landing_time<PHYSICS.dt*1.1
                if ready and self.demo_point<len(path)-1:self.demo_point+=1
                direction=waypoint_direction(self.world,path[self.demo_point][0])
            self.world.step(direction)
            self.accumulator-=PHYSICS.dt
            b=self.world.ball
            self.trail.append((b.x,b.y));self.trail=self.trail[-12:]
            for kind,payload in self.world.events:
                if kind=='fish':self.sound('fish')
                elif kind=='fail':
                    self.sound('fail');self.toast=payload;self.toast_until=self.wall_time+1.7;self.trail=[]
                elif kind=='switch':self.sound('switch')
            if self.world.cleared:
                self.sound('clear')
                score=None if self.world.level.index==0 else self.world.score
                self.new_record=self.progress.complete(self.world.level.index,score,self.world.time)
                self.scene='result';self.accumulator=0.;break

    def render(self):
        self.buttons=[]
        if self.scene=='menu':self.draw_menu()
        elif self.scene=='ending':self.draw_ending()
        else:self.draw_play()
        if self.help:self.draw_help()
        ww,wh=self.window.get_size();self.scale=min(ww/1200,wh/600)
        size=(max(1,round(1200*self.scale)),max(1,round(600*self.scale)))
        self.offset=((ww-size[0])//2,(wh-size[1])//2)
        self.window.fill((15,35,52))
        self.window.blit(pygame.transform.smoothscale(self.canvas,size),self.offset)
        pygame.display.flip()

    def process_events(self):
        mx,my=pygame.mouse.get_pos();self.hover=((mx-self.offset[0])/self.scale,(my-self.offset[1])/self.scale)
        for e in pygame.event.get():
            if e.type==pygame.QUIT:self.quit()
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_r and self.scene in ('play','result') and not self.help:self.restart()
            elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                at=((e.pos[0]-self.offset[0])/self.scale,(e.pos[1]-self.offset[1])/self.scale)
                for rect,callback in reversed(self.buttons):
                    if rect.collidepoint(at):callback();break
            elif e.type==pygame.WINDOWFOCUSLOST and self.world and self.scene=='play':self.world.paused=True

    def run(self,frames=0,demo=False):
        count=0;self.demo=demo
        self.render()
        while self.running:
            seconds=1/60 if self.headless else self.clock.tick(60)/1000
            self.process_events()
            keys=pygame.key.get_pressed()
            direction=int(keys[pygame.K_RIGHT])-int(keys[pygame.K_LEFT])
            self.update(seconds,direction)
            self.render();count+=1
            if frames and count>=frames:break


def main():
    parser=argparse.ArgumentParser(description='Pingu Ball — Pygame Antarctic platformer')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--screenshot')
    parser.add_argument('--stage',type=int,choices=range(8))
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--save-file')
    args=parser.parse_args()
    save=args.save_file
    if args.headless and not save:save=Path(tempfile.mkdtemp(prefix='pingu-smoke-'))/'progress.json'
    app=App(save,args.headless)
    try:
        if args.stage is not None:app.start(args.stage)
        if args.headless and not args.frames:
            app.render()
        else:app.run(args.frames,args.demo)
        if args.screenshot:
            path=Path(args.screenshot);path.parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(app.canvas,str(path));print(f'Screenshot: {path.resolve()}')
    finally:pygame.quit()
