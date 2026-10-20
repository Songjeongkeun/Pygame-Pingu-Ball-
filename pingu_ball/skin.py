"""Winter sprites and scalable, bidirectional moving-platform symbols."""
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
            # Use the solid body, not the faint shadow beneath the basic tile,
            # to fit both basic and cracked artwork into the same 64px square.
            alpha=128 if name in ('normal','breakable') else 1
            self.originals[name] = image.subsurface(image.get_bounding_rect(min_alpha=alpha)).copy()
        charged = pygame.image.load(str(Path(root)/'assets'/'ball_charged.png')).convert_alpha()
        self.originals['ball_charged'] = charged.subsurface(charged.get_bounding_rect(min_alpha=32)).copy()
        angry_path = Path(root)/'assets'/'ball_angry.png'
        angry = pygame.image.load(str(angry_path)).convert_alpha() if angry_path.exists() else self.originals['ball']
        self.originals['angry'] = angry.subsurface(angry.get_bounding_rect(min_alpha=32)).copy()
        for kind, color in [('forward', (16, 190, 236)), ('up', (245, 177, 35))]:
            icon = pygame.Surface((64, 64), pygame.SRCALPHA)
            points=[(32,2),(54,12),(62,36),(48,59),(19,61),(2,39),(9,14)]
            pygame.draw.polygon(icon,color,points)
            pygame.draw.lines(icon,(221,252,255),True,points,3)
            pygame.draw.lines(icon,(245,253,255),False,[(9,14),(22,9),(36,9)],4)
            if kind == 'forward':
                for x in (20, 33):
                    pygame.draw.lines(icon,(240,253,255),False,[(x,19),(x+13,32),(x,45)],9)
                    pygame.draw.lines(icon,(9,31,55),False,[(x,19),(x+13,32),(x,45)],5)
            else:
                for x in (22,42):
                    pygame.draw.line(icon,(245,253,255),(x,47),(x,20),10)
                    pygame.draw.polygon(icon,(245,253,255),[(x-10,29),(x,15),(x+10,29)])
                    pygame.draw.line(icon,(29,31,38),(x,46),(x,25),5)
                    pygame.draw.polygon(icon,(29,31,38),[(x-6,27),(x,19),(x+6,27)])
            self.originals['power_'+kind] = icon
        for kind,color,rim in [('forward',(3,15,30),(25,221,250)),('up',(91,46,25),(255,204,69))]:
            ball=pygame.transform.smoothscale(self.originals['ball'],(64,64)).copy()
            # Recolor only the dark head; red beak and white eyes stay intact.
            for y in range(64):
                for x in range(64):
                    c=ball.get_at((x,y))
                    if c.a and max(c.r,c.g,c.b)<130:
                        shade=(c.r+c.g+c.b)/3/130
                        ball.set_at((x,y),(*[round(v*(.6+.4*shade)) for v in color],c.a))
            pygame.draw.circle(ball,rim,(32,32),30,2)
            self.originals['ball_'+kind]=ball
        image = pygame.transform.smoothscale(self.originals['normal'], (64, 64))
        # One cyan double-headed arrow for every horizontal moving deck.
        # Build it on the plain reference block, with neither one-way icon
        # underneath. Launcher flames remain separate, directional sprites.
        points = [(6,37), (21,23), (21,31), (43,31), (43,23),
                  (58,37), (43,51), (43,43), (21,43), (21,51)]
        pygame.draw.polygon(image, (9, 52, 90), points)
        pygame.draw.lines(image, (9, 52, 90), True, points, 5)
        pygame.draw.polygon(image, (42, 215, 245), points)
        pygame.draw.lines(image, (204, 253, 255), True, points, 2)
        pygame.draw.line(image, (146, 247, 255), (23,34), (41,34), 2)
        self.originals['moving'] = image

    def column(self, height):
        """Repeat native stone below a single exposed cap for stacked terrain."""
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

    def platform(self, block, time, switches):
        if hasattr(block,'device'):
            from .special_draw import tile_image
            return tile_image(self,block,time,switches)
        return self.base_platform(block,time,switches)

    def base_platform(self, block, time, switches):
        """Reference sprites shared by terrain and equivalent special devices."""
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
