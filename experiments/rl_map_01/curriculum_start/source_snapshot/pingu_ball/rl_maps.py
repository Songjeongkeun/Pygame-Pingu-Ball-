"""승인된 2~3층 강화학습용 맵을 기존 물리 엔진의 레벨로 구성한다."""
import json
from pathlib import Path

from .config import RL_MAP_COUNT, RL_MAPS_ID
from .engine import Level, Platform
from .specials import Device

ROOT = Path(__file__).resolve().parents[1]


def rl_map_layouts():
    data = json.loads((ROOT/'assets/rl_maps/layouts.json').read_text(encoding='utf-8'))
    maps = data['maps']
    if (data.get('map_set') != RL_MAPS_ID or len(maps) != RL_MAP_COUNT
            or [m['number'] for m in maps] != list(range(1, RL_MAP_COUNT+1))):
        raise ValueError('Invalid reinforcement learning map data')
    return maps


def _level(data):
    return Level(data['number'], data['title'], data['description'],
        [Platform(**p) for p in data['platforms']], [], [],
        tuple(data['spawn']), tuple(data['goal']),
        width=data['width'], height=data['height'], time_limit=data['time_limit'],
        specials=[Device(**d) for d in data['devices']], show_movement_rails=False,
        preview=f'assets/rl_maps/map_{data["number"]:02d}.png')


def make_rl_level(number):
    if type(number) is not int or not 1 <= number <= RL_MAP_COUNT:
        raise ValueError('RL map number must be between 1 and 10')
    return _level(rl_map_layouts()[number-1])


def make_rl_levels():
    return [_level(data) for data in rl_map_layouts()]
