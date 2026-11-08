"""승인된 강화학습 맵 목록을 그리고 실제 버튼 영역을 등록한다."""
import pygame

from .config import RL_MAP_COUNT
from .rl_maps import ROOT, rl_map_layouts

BLUE = (8,64,122)
WHITE = (249,254,255)


def _panel(canvas, rect, selected=False, snow=True):
    """그림자와 눈 장식까지 지정한 사각형 안에 담는다. 정적 화면 준비 때만 호출한다."""
    rect=pygame.Rect(rect)
    w,h=(rect.w-8)*2,(rect.h-9)*2
    image=pygame.Surface((w+16,h+18),pygame.SRCALPHA)
    pygame.draw.rect(image,(5,55,101,45),(8,12,w,h),border_radius=32)
    pygame.draw.rect(image,(11,82,152),(0,0,w,h),border_radius=32)
    pygame.draw.rect(image,(48,209,247),(4,4,w-8,h-8),border_radius=29)
    face=pygame.Surface((w-16,h-16),pygame.SRCALPHA)
    for y in range(h-16):
        t=y/max(1,h-17)
        a=(184,241,255) if selected else (224,248,255)
        b=(112,216,252) if selected else (177,230,249)
        pygame.draw.line(face,tuple(round(a[i]*(1-t)+b[i]*t) for i in range(3)),(0,y),(w-17,y))
    mask=pygame.Surface(face.get_size(),pygame.SRCALPHA)
    pygame.draw.rect(mask,(255,255,255),mask.get_rect(),border_radius=25)
    face.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
    image.blit(face,(8,8))
    pygame.draw.rect(image,(220,255,255),(10,10,w-20,h-20),3,border_radius=25)
    if selected:
        pygame.draw.rect(image,(255,225,108),(1,1,w-2,h-2),4,border_radius=32)
    if snow:
        pygame.draw.rect(image,(247,253,255),(10,5,w-20,15),border_radius=9)
        for x,drop in ((24,22),(int(w*.27),17),(int(w*.57),23),(w-34,27)):
            pygame.draw.ellipse(image,(250,254,255),(x-16,3,40,drop+15))
        pygame.draw.line(image,(255,255,255),(28,7),(w-28,7),4)
    for x in (10,w-24):
        points=[(x,h-30),(x+13,h-50),(x+21,h-24),(x+7,h-10)]
        pygame.draw.polygon(image,(113,224,255),points)
        pygame.draw.lines(image,(232,255,255),True,points,2)
    canvas.blit(pygame.transform.smoothscale(image,rect.size),rect)


def _thumbnail(image, size):
    image=pygame.transform.smoothscale(image,size).convert_alpha()
    mask=pygame.Surface(size,pygame.SRCALPHA)
    pygame.draw.rect(mask,(255,255,255),mask.get_rect(),border_radius=8)
    image.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
    return image


class RLMenuUI:
    def __init__(self, app):
        self.cards=[pygame.Rect(38+(i%4)*388,219+(i//4)*187,369,174)
                    for i in range(RL_MAP_COUNT)]
        self.play_buttons=[pygame.Rect(r.x+203,r.y+114,149,37) for r in self.cards]
        self.tabs=[pygame.Rect(42+i*261,126,245,54) for i in range(3)]
        self.back=pygame.Rect(1296,24,260,76)
        self.counter=pygame.Rect(1220,140,315,27)
        self.base=pygame.Surface(app.canvas.get_size()).convert()
        original=app.canvas
        app.canvas=self.base
        try:
            self._prepare(app)
            # 글자까지 포함한 버튼 그림을 캐시해 누름 효과를 같은 비율로 적용한다.
            self.play_image=pygame.Surface((149,37),pygame.SRCALPHA)
            app.canvas=self.play_image
            rect=self.play_image.get_rect()
            pygame.draw.rect(app.canvas,(135,79,5),rect,border_radius=18)
            pygame.draw.rect(app.canvas,(255,179,11),rect.inflate(-4,-4),border_radius=18)
            pygame.draw.rect(app.canvas,(255,217,83),rect.inflate(-10,-10),border_radius=18)
            pygame.draw.arc(app.canvas,(255,248,191),rect.inflate(-5,-5),.05,3.08,3)
            field=rect.inflate(-22,-12)
            field.w-=25
            app.fitted_text('플레이',field,21,(92,48,5),outline=(255,246,194),natural=True)
            pygame.draw.polygon(app.canvas,(101,52,2),[(122,9),(122,27),(135,18)])
        finally:
            app.canvas=original

    def _prepare(self, app):
        app.canvas.blit(app.background,(0,0))
        app.fitted_text('강화학습 맵',(42,30,430,75),52,WHITE,outline=(8,57,119),natural=True)
        _panel(app.canvas,self.back)
        app.fitted_text('← 스테이지 선택',(1308,44,236,35),26,BLUE,natural=True)
        for i,(rect,title) in enumerate(zip(self.tabs,('튜토리얼','일반 스테이지','강화학습 맵'))):
            _panel(app.canvas,rect,selected=i==2,snow=False)
            app.fitted_text(title,rect.inflate(-12,-12),24,BLUE,natural=True)
        _panel(app.canvas,(1210,133,335,43),snow=False)
        app.fitted_text('파괴 · 점프 · 좌우 이동 · 상하 이동 · 시간 블록을 조합한 코스',
                       (42,184,1050,27),18,(15,76,120),False,natural=True)
        for i,(rect,data) in enumerate(zip(self.cards,rl_map_layouts()),1):
            _panel(app.canvas,rect)
            tag=pygame.Rect(rect.x+14,rect.y+13,77,24)
            pygame.draw.rect(app.canvas,(16,129,207),tag,border_radius=12)
            app.fitted_text(f'RL {i:02d}',tag,17,WHITE,natural=True)
            app.fitted_text(data['title'],(rect.x+101,rect.y+12,251,26),22,BLUE,natural=True)
            thumb_rect=pygame.Rect(rect.x+14,rect.y+49,176,91)
            image=pygame.image.load(str(ROOT/f'assets/rl_maps/map_{i:02d}.png'))
            app.canvas.blit(_thumbnail(image,thumb_rect.size),thumb_rect)
            pygame.draw.rect(app.canvas,(112,192,232),thumb_rect,1,border_radius=8)
            app.fitted_text(f'{data["tier_count"]}층 · {data["section_count"]}개 구간',
                           (rect.x+204,rect.y+49,150,24),18,BLUE,natural=True)
            for j,kind in enumerate(('breakable','spring','moving','moving_vertical','timed')):
                app.canvas.blit(app.skin.image(kind,(23,23)),(rect.x+204+j*28,rect.y+81))

    def draw(self, app, press_seconds):
        app.canvas.blit(self.base,(0,0))
        completed=len(app.progress.rl_records)
        app.fitted_text(f'{completed} / {RL_MAP_COUNT} 완료  ·  모두 선택 가능',
                       self.counter,19,BLUE,natural=True)
        for i,(card,button) in enumerate(zip(self.cards,self.play_buttons),1):
            image=self.play_image
            target=button
            if app.play_press and app.play_press[0]==i:
                amount=(1-min(1,app.play_press[1]/press_seconds))**2
                size=(round(button.w*(1-.04*amount)),round(button.h*(1-.10*amount)))
                image=pygame.transform.smoothscale(image,size)
                shade=round(255-24*amount)
                image.fill((shade,shade,shade,255),special_flags=pygame.BLEND_RGBA_MULT)
                target=image.get_rect(center=(button.centerx,button.centery+round(3*amount)))
            app.canvas.blit(image,target)
            elapsed=app.progress.rl_records.get(str(i))
            text='최고 기록  —' if elapsed is None else f'최고 기록  {elapsed:.2f}초'
            app.fitted_text(text,(card.x+15,card.y+144,173,15),13,(53,104,136),False,natural=True)
            app.art_button(card,lambda n=i:app.press_rl_play(n))
        for rect,callback in zip(self.tabs,(app.show_lab_menu,app.show_normal_menu,app.show_rl_menu)):
            app.art_button(rect,callback)
        app.art_button(self.back,app.menu)
        app.draw_list_sound('rl_menu')
