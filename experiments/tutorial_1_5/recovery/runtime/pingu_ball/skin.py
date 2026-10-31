"""겨울 테마 그림과 크기 조절이 가능한 양방향 이동 발판 기호를 제공한다."""
from pathlib import Path
import pygame


FILES = {'normal': 'block_basic', 'breakable': 'block_cracked',
         'spring': 'block_jump', 'timed': 'block_timed_active',
         'timed_off': 'block_timed_inactive', 'timed_warning': 'block_timed_warning',
         'switch': 'block_switch_idle', 'switch_on': 'block_switch_pressed',
         'linked': 'block_linked_active', 'linked_off': 'block_linked_inactive',
         'launcher_left': 'block_launcher_left',
         'launcher_right': 'block_launcher_right', 'spike': 'block_spikes',
         'goal': 'goal_igloo', 'ball': 'ball', 'fish': 'item_fish', 'bone': 'item_fish_bone'}
FILES.update({'sun':'hazard_sun', 'icicle':'hazard_icicle',
              'sea_lion':'hazard_sea_lion', 'moving_vertical':'block_moving_vertical'})


class WinterSkin:
    def __init__(self, root):
        folder = Path(root)/'assets'/'assets_v2'/'winter_cartoon'
        self.originals = {}
        self.cache = {}
        for name, file in FILES.items():
            image = pygame.image.load(str(folder/(file+'.png'))).convert_alpha()
            # 기본 타일 아래의 희미한 그림자가 아니라 단단한 몸체를 기준으로 삼아
            # 기본·금 간 블록 그림을 같은 64px 정사각형 안에 맞춘다.
            alpha=128 if name in ('normal','breakable','moving_vertical') else 1
            self.originals[name] = image.subsurface(image.get_bounding_rect(min_alpha=alpha)).copy()
        charged = pygame.image.load(str(Path(root)/'assets'/'ball_charged.png')).convert_alpha()
        self.originals['ball_charged'] = charged.subsurface(charged.get_bounding_rect(min_alpha=32)).copy()
        angry_path = Path(root)/'assets'/'ball_angry.png'
        angry = pygame.image.load(str(angry_path)).convert_alpha() if angry_path.exists() else self.originals['ball']
        self.originals['angry'] = angry.subsurface(angry.get_bounding_rect(min_alpha=32)).copy()
        for kind,file in [('forward','item_dash.png'),('up','item_jump.png')]:
            image=pygame.image.load(str(folder/file)).convert_alpha()
            image=image.subsurface(image.get_bounding_rect(min_alpha=32)).copy()
            side=max(image.get_size())
            icon=pygame.Surface((side,side),pygame.SRCALPHA)
            icon.blit(image,image.get_rect(center=(side//2,side//2)))
            self.originals['power_'+kind]=icon
        for kind in ('forward','up'):
            self.originals['ball_'+kind]=self.power_ball_art(kind)
        image = pygame.transform.smoothscale(self.originals['normal'], (64, 64))
        # 가로 이동 발판에는 청록색 양방향 화살표 하나를 사용한다.
        # 한 방향 아이콘이 아래에 남지 않도록 기본 참조 블록 위에 그린다.
        # 발사 블록의 불꽃은 방향에 따른 별도 그림으로 유지한다.
        points = [(6,37), (21,23), (21,31), (43,31), (43,23),
                  (58,37), (43,51), (43,43), (21,43), (21,51)]
        pygame.draw.polygon(image, (9, 52, 90), points)
        pygame.draw.lines(image, (9, 52, 90), True, points, 5)
        pygame.draw.polygon(image, (42, 215, 245), points)
        pygame.draw.lines(image, (204, 253, 255), True, points, 2)
        pygame.draw.line(image, (146, 247, 255), (23,34), (41,34), 2)
        self.originals['moving'] = image

    def power_ball_art(self,kind):
        """승인된 외곽선과 발광 효과를 실제 크기의 8배로 한 번 렌더링한다."""
        factor=8;center=(20*factor,20*factor)
        image=pygame.Surface((40*factor,40*factor),pygame.SRCALPHA)
        rim=(25,221,250) if kind=='forward' else (255,204,69)
        base=(3,15,30) if kind=='forward' else (119,47,24)
        for radius,alpha in ((20,12),(19,19),(18,35)):
            layer=pygame.Surface(image.get_size(),pygame.SRCALPHA)
            pygame.draw.circle(layer,(*rim,alpha),center,radius*factor)
            image.blit(layer,(0,0))
        pygame.draw.circle(image,(5,34,61,245),center,round(16.7*factor))
        body=pygame.transform.smoothscale(self.originals['ball'],(112,112)).copy()
        for y in range(112):
            for x in range(112):
                color=body.get_at((x,y))
                is_beak=color.r>color.g*1.5 and color.r>color.b*1.5
                if color.a and max(color.r,color.g,color.b)<145 and not is_beak:
                    value=(color.r+color.g+color.b)/3/145
                    body.set_at((x,y),(*(min(255,round(c*(.72+.5*value))) for c in base),color.a))
        body=pygame.transform.smoothscale(body,(28*factor,28*factor))
        image.blit(body,(6*factor,6*factor))
        pygame.draw.circle(image,(*rim,255),center,round(15.2*factor),round(2.6*factor))
        light=(195,255,255) if kind=='forward' else (255,246,192)
        pygame.draw.arc(image,(*light,255),pygame.Rect(5*factor,5*factor,30*factor,30*factor),
                        .55,2.5,round(.65*factor))
        return image

    def ball(self,size=28,charged=False):
        """공 몸체의 크기는 유지하고, 능력 보유 상태의 그림만 충돌 영역 바깥으로 확장한다."""
        name='ball_'+charged if isinstance(charged,str) else 'ball_charged' if charged else 'ball'
        diameter=round(size*40/28) if charged in ('forward','up') else size
        return self.image(name,(diameter,diameter))

    def column(self, height):
        """쌓인 지형에서는 노출된 맨 위 눈 덮개 아래로 원래 돌 무늬를 반복한다."""
        key=('column',round(height))
        if key not in self.cache:
            tile=self.image('normal')
            height=round(height)
            image=pygame.Surface((64,height),pygame.SRCALPHA)
            image.blit(tile,(0,0),pygame.Rect(0,0,64,22))
            for y in range(22,height-4):
                offset=(y-22)%32
                for x in range(64):
                    color=tile.get_at((x,22+offset))
                    if y>=54 and offset<6:
                        before=tile.get_at((x,54+offset));ratio=(offset+1)/6
                        color=tuple(round(before[i]*(1-ratio)+color[i]*ratio) for i in range(4))
                    image.set_at((x,y),color)
            image.blit(tile,(0,height-4),pygame.Rect(0,60,64,4))
            self.cache[key]=image
        return self.cache[key]

    def image(self, name, size=(64, 64)):
        size = tuple(map(round, size))
        key = name, size
        if key not in self.cache:
            self.cache[key] = pygame.transform.smoothscale(self.originals[name], size)
        return self.cache[key]

    def collectible(self, bone=False, width=32, height=None):
        """맵에서는 정확한 크기를 사용하고, 상태 표시 아이콘에서는 비율을 유지한다."""
        name='bone' if bone else 'fish'
        source=self.originals[name]
        if height is None:height=round(width*source.get_height()/source.get_width())
        return self.image(name,(width,height))

    def platform(self, block, time, switches):
        if hasattr(block,'device'):
            from .special_draw import tile_image
            return tile_image(self,block,time,switches)
        return self.base_platform(block,time,switches)

    def base_platform(self, block, time, switches):
        """일반 지형과 같은 형태의 특수 장치가 공통으로 사용할 참조 그림을 반환한다."""
        if block.destroyed:
            return None
        if not block.active(time, switches):
            name={'timed':'timed_off','linked':'linked_off'}.get(block.kind)
            return self.image(name,(block.w,block.h)) if name else None
        name = block.kind
        if name == 'launcher':
            name = 'launcher_left' if block.arrow < 0 else 'launcher_right'
        elif name == 'switch' and not block.armed:
            name = 'switch_on'
        elif name == 'timed' and (time+block.phase) % block.period >= block.active_for-block.warning:
            name = 'timed_warning'
        return self.image(name, (block.w, block.h))
