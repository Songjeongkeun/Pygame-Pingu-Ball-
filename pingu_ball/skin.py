"""Winter sprites and scalable, bidirectional moving-platform symbols."""
from pathlib import Path
import pygame


FILES = {'normal': 'block_basic', 'breakable': 'block_cracked',
         'spring': 'block_jump', 'timed': 'block_timed_active',
         'timed_off': 'block_timed_inactive', 'timed_warning': 'block_timed_warning',
         'switch': 'block_switch_idle', 'switch_on': 'block_switch_pressed',
         'linked': 'block_linked_active', 'launcher_left': 'block_launcher_left',
         'launcher_right': 'block_launcher_right', 'spike': 'block_spikes',
         'goal': 'goal_igloo', 'ball': 'ball', 'fish': 'item_fish', 'bone': 'item_fish_bone'}


class WinterSkin:
    def __init__(self, root):
        folder = Path(root)/'assets'/'assets_v2'/'winter_cartoon'
        self.originals = {}
        self.cache = {}
        for name, file in FILES.items():
            image = pygame.image.load(str(folder/(file+'.png'))).convert_alpha()
            # The sprites contain transparent padding; visible edges are physics edges.
            self.originals[name] = image.subsurface(image.get_bounding_rect()).copy()
        image = pygame.transform.smoothscale(self.originals['normal'], (64, 64))
        points = [(7,36), (21,23), (21,30), (43,30), (43,23),
                  (57,36), (43,49), (43,42), (21,42), (21,49)]
        pygame.draw.polygon(image, (51, 222, 250), points)
        pygame.draw.lines(image, (205, 253, 255), True, points, 2)
        self.originals['moving'] = image

    def image(self, name, size=(64, 64)):
        size = tuple(map(round, size))
        key = name, size
        if key not in self.cache:
            self.cache[key] = pygame.transform.smoothscale(self.originals[name], size)
        return self.cache[key]

    def platform(self, block, time, switches):
        if not block.active(time, switches):
            return self.image('timed_off', (block.w, block.h)) if block.kind == 'timed' else None
        name = block.kind
        if name == 'launcher':
            name = 'launcher_left' if block.arrow < 0 else 'launcher_right'
        elif name == 'switch' and not block.armed:
            name = 'switch_on'
        elif name == 'timed' and (time+block.phase) % block.period >= block.active_for-block.warning:
            if int(time*12) % 2:
                name = 'timed_warning'
        return self.image(name, (block.w, block.h))
