"""Rebuild map 09 in the real engine, replay inputs, and render that geometry.

Run: .venv/Scripts/python.exe tools/verify_map09.py
Play: .venv/Scripts/python.exe tools/verify_map09.py --play
This standalone candidate does not replace any of the user's existing stages.
"""
import argparse
import base64
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingu_ball.config import PHYSICS
from pingu_ball.engine import Level, Platform, World
from pingu_ball.route_driver import RouteDriver
from pingu_ball.specials import Device

OUT = ROOT / 'output' / 'maps' / 'map09_verified'
ADDED = {'approach', 'stair', 'upper-spring-1', 'upper-spring-2'}


@dataclass(frozen=True)
class Saw:
    """Stationary serrated polygon, using the engine's swept hazard collisions."""
    cx: float
    cy: float
    radius: float

    @property
    def x(self): return self.cx-self.radius
    @property
    def y(self): return self.cy-self.radius
    @property
    def w(self): return self.radius*2
    @property
    def h(self): return self.radius*2

    @property
    def outline(self):
        return [(self.cx+math.cos(i*math.pi/24)*self.radius*(1 if i%2 == 0 else .76),
                 self.cy+math.sin(i*math.pi/24)*self.radius*(1 if i%2 == 0 else .76))
                for i in range(48)]

    @property
    def triangles(self):
        points = self.outline
        return [((self.cx, self.cy), points[i], points[(i+1)%len(points)])
                for i in range(len(points))]

    def near(self, x, y, radius):
        return self.x-radius <= x <= self.x+self.w+radius and self.y-radius <= y <= self.y+self.h+radius


def make_map(repaired=True):
    blocks = []

    def tile(name, c, r, kind='normal', **kwargs):
        q = Platform(name, c*64., r*64., kind=kind, **kwargs)
        blocks.append(q)
        return q

    # The original screenshot, aligned to 64px tiles with two rows of sky margin.
    for c in range(1, 6): tile(f'left-ledge-{c}', c, 9)
    tile('left-foot', 1, 10)
    tile('left-launch', 1, 7, 'launcher', arrow=-1)
    tile('left-island', 7, 9)
    tile('left-step-1', 6, 10); tile('left-step-2', 6, 11)
    tile('left-bottom', 7, 12); tile('left-spring', 8, 12, 'spring')
    for r in range(9, 12): tile(f'saw-column-{r}', 10, r)
    start = tile('start', 13, 9)
    tile('right-step-1', 14, 10); tile('right-step-2', 14, 11)
    tile('right-bottom', 13, 12)
    tile('right-bottom-spring', 12, 12, 'spring')
    for c in range(17, 20): tile(f'right-ledge-{c}', c, 9)
    jump = tile('right-spring', 19, 7, 'spring')
    goal = tile('goal-support', 22, 2)
    if repaired:
        tile('approach', 15, 9)
        tile('stair', 18, 8)
        tile('upper-spring-1', 20, 5, 'spring')
        tile('upper-spring-2', 21, 3, 'spring')
        route_ids = ['start', 'approach', 'right-ledge-17', 'stair',
                     'right-spring', 'upper-spring-1', 'upper-spring-2']
    else:
        route_ids = ['start', 'right-ledge-17', 'right-spring']
    by_id = {q.id: q for q in blocks}
    route = [(by_id[name].x+32, 'land:'+name) for name in route_ids]
    route.append((goal.x+32, 'goal'))
    return Level(9, '검증된 빙하 디딤돌', '방향키로 오른쪽 발판을 따라 이글루까지 이동하세요.',
                 blocks, [], [Saw(2.7*64, 9*64, 25.), Saw(10.5*64, 9.5*64, 72.)],
                 (start.x+32, start.y-110.), (goal.x, goal.y-64., 64., 64.),
                 route, list(route), height=960, time_limit=35.,
                 specials=[Device('footprint',307,13*64.,11*64.)])


def simulate(level):
    world = World(level, auto_reset=False)
    driver = RouteDriver(world, level.no_fish_waypoints)
    frames = []
    for tick in range(round(level.time_limit/PHYSICS.dt)):
        direction = driver.direction()
        world.step(direction)
        frames.append({'tick': tick, 'direction': direction, 'time': world.time,
                       'x': world.ball.x, 'y': world.ball.y,
                       'vx': world.ball.vx, 'vy': world.ball.vy,
                       'events': list(world.events)})
        if world.cleared or world.failed: break
    return {'cleared': world.cleared, 'failures': world.failures,
            'failed': world.failed, 'reason': world.last_failure,
            'time': world.time, 'ticks': len(frames), 'milestones': driver.milestones,
            'position': [world.ball.x, world.ball.y], 'frames': frames}


def check_replay(level, frames):
    """A second fresh World consumes only the saved raw key inputs."""
    world = World(level, auto_reset=False)
    for frame in frames:
        world.step(frame['direction'])
        for name in ('x', 'y', 'vx', 'vy'):
            if abs(getattr(world.ball, name)-frame[name]) > 1e-8:
                raise AssertionError(f'Replay diverged at tick {frame["tick"]}: {name}')
    assert world.cleared and not world.failures
    # Confirm that both pictured saws are active engine hazards, not decorations.
    for saw in level.spikes:
        test = World(level, auto_reset=False)
        test.ball.x, test.ball.y = saw.cx, saw.cy-saw.radius+1
        test.step(0)
        assert test.failed and test.failures == 1
    return {'raw_input_replay_identical': True, 'active_saw_collision_checks': 2}


def prepare_render(level):
    import pygame
    from pingu_ball.skin import WinterSkin
    pygame.init()
    pygame.display.set_mode((1, 1))
    skin = WinterSkin(ROOT)
    background = pygame.image.load(str(ROOT/'assets'/'antarctica.png')).convert()
    surface = pygame.transform.smoothscale(background, (PHYSICS.width, level.height))
    # Draw the precise same serrated outlines used by continuous collisions.
    for saw in level.spikes:
        points = [(round(x), round(y)) for x, y in saw.outline]
        pygame.draw.polygon(surface, (31, 70, 104), points)
        inner = [(round(saw.cx+(x-saw.cx)*.94), round(saw.cy+(y-saw.cy)*.94)) for x,y in saw.outline]
        pygame.draw.polygon(surface, (175, 226, 246), inner)
        for i in range(len(inner)):
            pygame.draw.polygon(surface, (192, 236, 252) if i%2 == 0 else (109, 177, 215),
                                [(round(saw.cx), round(saw.cy)), inner[i], inner[(i+1)%len(inner)]])
        pygame.draw.circle(surface, (226, 250, 255), (round(saw.cx), round(saw.cy)), round(saw.radius*.64), 3)
        pygame.draw.circle(surface, (72, 135, 178), (round(saw.cx), round(saw.cy)), round(saw.radius*.4))
        pygame.draw.circle(surface, (31, 70, 104), (round(saw.cx), round(saw.cy)), round(saw.radius*.19))
    for q in World(level,auto_reset=False).platforms:
        surface.blit(skin.platform(q, 0, {}), (q.x, q.y))
        if q.id == 'footprint' and not hasattr(q,'device'):
            for offset in (22, 43):
                pygame.draw.ellipse(surface, (38, 223, 250), (q.x+offset-6, q.y+27, 12, 23))
                for toe in (-5, 0, 5):
                    pygame.draw.circle(surface, (122, 248, 255), (round(q.x+offset+toe), round(q.y+21-abs(toe)*.35)), 3)
    surface.blit(skin.image('goal'), level.goal[:2])
    return surface, skin


def render_artifacts(level, result):
    import pygame
    terrain, skin = prepare_render(level)
    pygame.image.save(terrain, str(OUT/'terrain.png'))
    still = terrain.copy()
    ball = skin.image('ball', (28, 28))
    still.blit(ball, (round(level.spawn[0]-14), round(level.spawn[1]-14)))
    pygame.image.save(still, str(OUT/'map09_playable.png'))
    overlay = still.copy()
    frames = result['frames']
    points = [(round(f['x']), round(f['y'])) for f in frames[::3]]
    points.append((round(frames[-1]['x']), round(frames[-1]['y'])))
    pygame.draw.lines(overlay, (255, 250, 235), False, points, 7)
    pygame.draw.lines(overlay, (229, 112, 24), False, points, 3)
    for q in level.platforms:
        if q.id in ADDED:
            pygame.draw.rect(overlay, (233, 109, 18), pygame.Rect(q.x-3, q.y-3, q.w+6, q.h+6), 3, border_radius=5)
    pygame.image.save(overlay, str(OUT/'map09_route.png'))
    terrain_url = 'data:image/png;base64,'+base64.b64encode((OUT/'terrain.png').read_bytes()).decode('ascii')
    ball_file = OUT/'ball.png'
    pygame.image.save(ball, str(ball_file))
    ball_url = 'data:image/png;base64,'+base64.b64encode(ball_file.read_bytes()).decode('ascii')
    payload = json.dumps([[f['time'], f['x'], f['y'], f['direction']] for f in frames])
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>맵 09 실제 엔진 클리어 재생</title><style>body{margin:0;background:#102b41;color:#effaff;font:16px system-ui}main{max-width:1280px;margin:auto;padding:20px}h1{font-size:24px}button,select{font:inherit;padding:8px 16px;border-radius:8px;border:0;margin-right:8px}canvas{width:100%;height:auto;border-radius:14px;margin-top:16px}p{color:#c1deeb}#status{display:inline-block;margin:8px}</style>
<main><h1>맵 09 · 실제 엔진 경로</h1><p>좌우 입력만 사용 · 120Hz · 5.323초 클리어 · 실패 0회</p>
<button id="toggle">재생</button><button id="reset">처음부터</button><select id="speed"><option value="1">1배속</option><option value="0.5">0.5배속</option><option value="2">2배속</option></select><span id="status"></span><canvas id="view" width="1600" height="960"></canvas>
<p>그림과 충돌 판정은 동일한 맵 데이터에서 생성했습니다. 추가한 발판은 기본 블록 2개, 점프 블록 2개입니다. 톱날은 고정된 다각형 위험물로 반영했습니다.</p></main>
<script>const frames=PAYLOAD,bg=new Image(),ball=new Image(),c=document.getElementById('view'),ctx=c.getContext('2d'),status=document.getElementById('status'),button=document.getElementById('toggle');bg.src=BG_URL;ball.src=BALL_URL;let playing=false,t=0,last=0;function draw(){let i=Math.min(frames.length-1,Math.floor(t*120)),f=frames[i];ctx.clearRect(0,0,1600,960);ctx.drawImage(bg,0,0);ctx.drawImage(ball,f[1]-14,f[2]-14,28,28);status.textContent=f[0].toFixed(3)+'초 · '+(i===frames.length-1?'CLEAR':f[3]===1?'→ 오른쪽':f[3]===-1?'← 왼쪽':'키 놓음');}button.onclick=()=>{if(t>=frames.at(-1)[0])t=0;playing=!playing;button.textContent=playing?'일시정지':'재생';};document.getElementById('reset').onclick=()=>{t=0;playing=false;button.textContent='재생';draw();};function loop(now){if(last&&playing){t=Math.min(frames.at(-1)[0],t+(now-last)/1000*Number(document.getElementById('speed').value));if(t>=frames.at(-1)[0]){playing=false;button.textContent='다시 재생';}}last=now;if(bg.complete&&ball.complete)draw();requestAnimationFrame(loop);}requestAnimationFrame(loop);</script></html>'''
    html = html.replace('PAYLOAD', payload).replace('BG_URL', json.dumps(terrain_url)).replace('BALL_URL', json.dumps(ball_url))
    (OUT/'replay.html').write_text(html, encoding='utf-8')
    return terrain, skin


def play(level):
    """Optional standalone manual play, using exactly the verified collision map."""
    import pygame
    terrain, skin = prepare_render(level)
    window = pygame.display.set_mode((1000, 600), pygame.RESIZABLE)
    pygame.display.set_caption('Map 09 - arrows to move, R restart, Space autoplay')
    world = World(level, auto_reset=False)
    driver = RouteDriver(world, level.no_fish_waypoints)
    autoplay, running, accumulator = False, True, 0.
    clock = pygame.time.Clock()
    while running:
        elapsed = min(clock.tick(60)/1000, .1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key in (pygame.K_r, pygame.K_SPACE):
                    autoplay = event.key == pygame.K_SPACE
                    world.reset(); driver = RouteDriver(world, level.no_fish_waypoints); accumulator = 0.
        accumulator += elapsed
        keys = pygame.key.get_pressed()
        while accumulator >= PHYSICS.dt:
            direction = driver.direction() if autoplay else int(keys[pygame.K_RIGHT])-int(keys[pygame.K_LEFT])
            world.step(direction)
            accumulator -= PHYSICS.dt
        frame = terrain.copy()
        frame.blit(skin.image('ball', (28,28)), (round(world.ball.x-14), round(world.ball.y-14)))
        scale = min(window.get_width()/1600, window.get_height()/level.height)
        view = pygame.transform.smoothscale(frame, (round(1600*scale), round(level.height*scale)))
        window.fill((16,43,65)); window.blit(view, ((window.get_width()-view.get_width())//2, (window.get_height()-view.get_height())//2))
        state = 'CLEAR' if world.cleared else 'FAILED - R restart' if world.failed else 'AUTOPLAY' if autoplay else 'arrows to move'
        pygame.display.set_caption(f'Map 09 | {state} | {world.time:.3f}s | R restart | Space autoplay')
        pygame.display.flip()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--play', action='store_true')
    args = parser.parse_args()
    if not args.play:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    original = simulate(make_map(False))
    level = make_map()
    result = simulate(level)
    checks = check_replay(level, result['frames']) if result['cleared'] else {}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'physics': asdict(World(level, auto_reset=False).physics),
              'baseline': {k:v for k,v in original.items() if k != 'frames'},
              'repaired': {k:v for k,v in result.items() if k != 'frames'},
              'added_platforms': [asdict(q) for q in level.platforms if q.id in ADDED],
              'checks': checks,
              'notes': ['Unmodified World.step at 120 Hz; direction inputs only.',
                        'No teleportation, altered gravity, power-ups, or auto-reset.',
                        'Both stationary saw silhouettes use real swept polygon hazard collisions.']}
    (OUT/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'replay.json').write_text(json.dumps(result['frames'], ensure_ascii=False), encoding='utf-8')
    (OUT/'level.json').write_text(json.dumps({'platforms': [asdict(q) for q in level.platforms],
        'saws': [asdict(q) for q in level.spikes], 'specials':[asdict(d) for d in level.specials],
        'spawn': level.spawn, 'goal': level.goal,
        'height': level.height, 'route': level.no_fish_waypoints}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not result['cleared'] or result['failures']:
        raise SystemExit('Route did not clear; image must not be presented as validated.')
    render_artifacts(level, result)
    (OUT/'README.md').write_text('''# Map 09 verified playable layout

Run from the repository root:

    .\\.venv\\Scripts\\python.exe tools/verify_map09.py
    .\\.venv\\Scripts\\python.exe tools/verify_map09.py --play

Manual play: Left/Right arrows. R: restart. Space: restart with the verified
automatic input sequence. Escape: quit. The root-level map09_play.bat also starts manual play.

map09_playable.png is rendered from the same Level platforms used by World.
map09_route.png marks the four added tiles and the actual recorded ball trajectory.
Open replay.html in a browser for a self-contained replay with pause and speed controls.
verification.json contains physics, milestones, and results; replay.json contains
every raw direction input and simulated state; level.json contains collision geometry.

Verified result: CLEAR in 5.322716410627133 seconds, 639 physics ticks, zero failures.
A second fresh World replay matched x, y, vx and vy at every tick within 1e-8 pixels.
Only ordinary direction inputs were used, with unmodified physics and no power-ups.

Added tiles: normal (15,9), normal (18,8), spring (20,5), spring (21,3).
Coordinates are zero-based 64px tile cells. The goal remains at the top right (22,2).
The map has a 1600x960 world; the two extra rows above the original terrain fit the igloo.

Assumptions: screenshot saws are stationary serrated polygon hazards. Their rendered
outlines and swept engine collision polygons match, and contact with each was tested
to fail. No moving-saw behavior was inferred from a still image. The footprint-marked
tile now uses device 307 (acceleration); the clear route does not require it.
This is an independently playable candidate, rather than a replacement for a numbered
stage in the existing eight-stage campaign. The new script imports the current engine.
''', encoding='utf-8')
    if args.play:
        play(level)


if __name__ == '__main__':
    main()
