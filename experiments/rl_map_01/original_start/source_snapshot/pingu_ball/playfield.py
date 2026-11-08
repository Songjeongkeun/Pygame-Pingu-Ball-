"""플레이 가능한 월드를 상단 표시 영역 아래의 전체 화면 너비에 맞춘다."""
from dataclasses import replace
import math

VIEW_LEFT = 16
VIEW_TOP = 144
VIEW_WIDTH = 1568
VIEW_HEIGHT = 672


def expand_playfield(level):
    """양옆에 빈 공중 공간을 추가하고, 객체 사이의 거리를 유지하며 모두 함께 옮긴다.

    문서 원본 맵은 수정하지 않는다. 화면에 표시하는 월드와 실제 물리 영역을
    함께 넓혀 화면 안쪽에 보이지 않는 벽이 생기지 않게 한다.
    """
    width = max(level.width, math.ceil(level.height*VIEW_WIDTH/VIEW_HEIGHT))
    shift = (width-level.width)/2
    devices = []
    for device in level.specials:
        options = dict(device.options)
        if 'path' in options:
            x, y, w, h = options['path']
            options['path'] = (x+shift, y, w, h)
        devices.append(replace(device, x=device.x+shift, options=options))
    return replace(level, width=width,
        platforms=[replace(q, x=q.x+shift) for q in level.platforms],
        collectibles=[replace(q, x=q.x+shift) for q in level.collectibles],
        spikes=[replace(q, x=q.x+shift) for q in level.spikes],
        powerups=[replace(q, x=q.x+shift) for q in level.powerups],
        specials=devices, spawn=(level.spawn[0]+shift, level.spawn[1]),
        goal=(level.goal[0]+shift, *level.goal[1:]),
        movement_markers=[(x+shift, y) for x, y in level.movement_markers],
        no_fish_waypoints=[(x+shift, condition) for x, condition in level.no_fish_waypoints],
        fish_waypoints=[(x+shift, condition) for x, condition in level.fish_waypoints])
