"""Display-independent simulation. All coordinates are logical pixels."""
from dataclasses import dataclass, field, replace
import copy
import math
from .config import PHYSICS, Physics
from .geometry import (EPS, circle_rect_overlap, circle_rect_separation,
                       sweep_circle_rect, sweep_point_rect,
                       sweep_circle_polygon, circle_polygon_overlap)


@dataclass
class Platform:
    id: str
    x: float
    y: float
    w: float = 64.
    h: float = 64.
    kind: str = 'normal'
    travel: float = 0.
    period: float = 2.8
    active_for: float = 2.
    phase: float = 0.
    warning: float = .5
    group: str = ''
    arrow: int = 1
    break_at: float | None = None
    destroyed: bool = False
    armed: bool = True

    def rect_at(self, time, physics=PHYSICS):
        x = self.x
        if self.kind == 'moving' and self.travel:
            distance = (time*physics.platform_speed+self.phase) % (2*self.travel)
            x += distance if distance <= self.travel else 2*self.travel-distance
        return x, self.y, self.w, self.h

    def active(self, time, switches):
        if self.destroyed:
            return False
        if self.kind == 'timed':
            return (time+self.phase) % self.period < self.active_for
        if self.kind == 'linked':
            return switches.get(self.group, False)
        return True


@dataclass(frozen=True)
class Collectible:
    x: float
    y: float
    value: int = 1
    size: float = 32.

    @property
    def rect(self):
        return self.x-self.size/2, self.y-self.size/2, self.size, self.size


@dataclass(frozen=True)
class Spike:
    x: float
    y: float
    w: float = 128.
    h: float = 64.
    teeth: int = 2

    def near(self, x, y, radius):
        return (self.x-radius <= x <= self.x+self.w+radius
                and self.y-radius <= y <= self.y+self.h+radius)

    @property
    def triangles(self):
        # The collision triangles are also the rendered spike silhouette.
        return [((self.x+i*self.w/self.teeth, self.y+self.h),
                 (self.x+(i+.5)*self.w/self.teeth, self.y),
                 (self.x+(i+1)*self.w/self.teeth, self.y+self.h)) for i in range(self.teeth)]


@dataclass
class Level:
    index: int
    title: str
    subtitle: str
    platforms: list[Platform]
    collectibles: list[Collectible]
    spikes: list[Spike]
    spawn: tuple[float, float]
    goal: tuple[float, float, float, float]
    no_fish_waypoints: list[tuple[float, str]] = field(default_factory=list)
    fish_waypoints: list[tuple[float, str]] = field(default_factory=list)
    height: int = 832
    preview: str = ''
    time_limit: float = 60.


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.
    vy: float = 0.
    launch: bool = False
    launch_source: str | None = None


def advance_x(v, target, acceleration, time):
    if abs(v-target) < EPS:
        return target*time, target
    sign = 1 if target > v else -1
    ramp = min(time, abs(target-v)/acceleration)
    dx = v*ramp + .5*sign*acceleration*ramp*ramp + target*(time-ramp)
    end = v+sign*acceleration*ramp
    return dx, target if ramp < time or abs(end-target) < EPS else end


def crossing_time(position, velocity, acceleration, boundary, duration, near):
    """Exact flat-face impact time for constant acceleration."""
    if abs(acceleration) < EPS:
        roots = [(boundary-position)/velocity] if abs(velocity) > EPS else []
    else:
        disc = velocity*velocity+2*acceleration*(boundary-position)
        roots = [] if disc < -EPS else [(-velocity+s*math.sqrt(max(0., disc)))/acceleration for s in (-1, 1)]
    valid = [max(0., t) for t in roots if -EPS <= t <= duration+EPS]
    return min(valid, key=lambda t: abs(t-near)) if valid else near


class World:
    def __init__(self, level: Level, physics: Physics = PHYSICS, auto_reset=True):
        self.level = level
        self.physics = replace(physics, height=level.height)
        self.auto_reset = auto_reset
        self.failures = 0
        self.reset()

    def reset(self):
        self.platforms = copy.deepcopy(self.level.platforms)
        self.ball = Ball(*self.level.spawn)
        self.time = 0.
        self.failed = False
        self.paused = False
        self.cleared = False
        self.collected = set()
        self.switches = {}
        self.last_landing = None
        self.last_landing_time = -1.
        self.last_failure = ''
        self.events = []
        self._crush_time = 0.

    @property
    def plus(self):
        return sum(self.level.collectibles[i].value > 0 for i in self.collected)

    @property
    def minus(self):
        return sum(self.level.collectibles[i].value < 0 for i in self.collected)

    @property
    def score(self):
        return 7 + self.plus - self.minus

    @property
    def grade(self):
        return None if self.level.index == 0 else {6:'B', 7:'A', 8:'S', 9:'SS', 10:'SSS'}[self.score]

    def fail(self, reason):
        self.failures += 1
        if self.auto_reset:
            self.reset()
        else:
            self.failed = True
        self.last_failure = reason
        self.events.append(('fail', reason))

    def step(self, direction=0, dt=None):
        self.events = []
        if self.cleared or self.failed or self.paused:
            return
        direction = max(-1, min(1, direction))
        dt = self.physics.dt if dt is None else dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        # CCD on short parabolic pieces; no discrete position-only hazard checks.
        count = max(4, math.ceil((abs(self.ball.vx)+abs(self.ball.vy))*dt/2))
        for _ in range(count):
            if not self._piece(direction, dt/count):
                return

    def _side_motion(self, direction, duration):
        """Constrain only motion into touching walls, until their side is cleared."""
        b, p = self.ball, self.physics
        target = direction*p.max_speed
        interval = duration
        walls = [((-40., -40., 40., p.height+100.), 0.),
                 ((p.width, -40., 40., p.height+100.), 0.)]
        for q in self.platforms:
            if q.id != b.launch_source and q.active(self.time, self.switches):
                rect = q.rect_at(self.time, p)
                probe = min(duration, 1e-6)
                speed = (q.rect_at(self.time+probe, p)[0]-rect[0])/probe
                walls.append((rect, speed))
        for (x, y, w, h), speed in walls:
            if not y-1e-7 <= b.y <= y+h+1e-7:
                continue
            if (b.y <= y+1e-7 and b.vy < 0) or (b.y >= y+h-1e-7 and b.vy >= 0):
                continue
            if abs(b.x-(x-p.radius)) <= 1e-5:
                side = -1
            elif abs(b.x-(x+w+p.radius)) <= 1e-5:
                side = 1
            else:
                side = 0
            if not side:
                continue
            if b.launch and (b.vx-speed)*side < 0:
                b.launch = False
            if side < 0:
                b.vx, target = min(b.vx, speed), min(target, speed)
            else:
                b.vx, target = max(b.vx, speed), max(target, speed)
            gravity = 0. if b.launch else p.gravity
            # Release at the side's vertical endpoint, even within this piece.
            for boundary, outward in ((y, -1), (y+h, 1)):
                if gravity:
                    disc = b.vy*b.vy+2*gravity*(boundary-b.y)
                    roots = [] if disc < 0 else [(-b.vy+s*math.sqrt(disc))/gravity for s in (-1, 1)]
                else:
                    roots = [(boundary-b.y)/b.vy] if b.vy else []
                for t in roots:
                    if 1e-10 < t < interval and (b.vy+gravity*t)*outward > 0:
                        interval = t
        return interval, target

    def _land(self, q):
        """Only a descending impact on a flat top may trigger block effects."""
        b, p = self.ball, self.physics
        b.y = q.y-p.radius
        self.last_landing = q.id
        self.last_landing_time = self.time
        self.events.append(('land', q.id))
        if q.kind == 'breakable' and q.break_at is None:
            q.break_at = self.time+p.break_delay
        if q.kind == 'switch' and q.armed:
            self.switches[q.group] = not self.switches.get(q.group, False)
            q.armed = False
            self.events.append(('switch', q.group))
        if q.kind == 'launcher':
            b.y = q.y+q.h/2
            b.launch = True; b.vy = 0.; b.vx = q.arrow*p.launch_speed
            b.launch_source = q.id
        else:
            b.launch = False
            b.vy = -p.pad_speed if q.kind == 'spring' else -p.bounce_speed
            b.y -= 1e-6

    def _piece(self, direction, duration, _depth=0):
        b, p = self.ball, self.physics
        if b.y-p.radius > p.height:
            self.fail('얼음 바다로 떨어졌어요'); return False
        for spike in self.level.spikes:
            if spike.near(b.x, b.y, p.radius) and any(circle_polygon_overlap(b.x,b.y,p.radius,t) for t in spike.triangles):
                self.fail('가시에 닿았어요'); return False
        end_time = self.time+duration
        old_active = {q.id:q.active(self.time, self.switches) for q in self.platforms}
        for q in self.platforms:
            if q.break_at is not None and self.time >= q.break_at-EPS:
                q.destroyed = True
            if q.kind == 'switch' and not q.armed:
                x, _, w, _ = q.rect_at(self.time, p)
                if b.x+p.radius < x or b.x-p.radius > x+w:
                    q.armed = True
            if q.id == b.launch_source:
                x,_,w,_=q.rect_at(self.time,p)
                if b.x+p.radius < x or b.x-p.radius > x+w:
                    b.launch_source=None
        if b.launch and direction:
            b.launch = False
            self.events.append(('release', None))
        remaining = duration
        for _collision in range(10):
            if remaining <= 1e-10:
                break
            interval, target = self._side_motion(direction, remaining)
            gravity = 0. if b.launch else p.gravity
            accel = p.acceleration if direction else p.deceleration
            dx, next_vx = (b.vx*interval, b.vx) if b.launch else advance_x(b.vx, target, accel, interval)
            dy = b.vy*interval+.5*gravity*interval**2
            start, delta = (b.x, b.y), (dx, dy)
            # event tuple: fraction, priority (failure first), kind, payload
            candidates = []
            solids = []
            for q in self.platforms:
                if q.id == b.launch_source:
                    # Center-height launches pass through their source until fully outside.
                    continue
                if q.active(self.time, self.switches):
                    rect = q.rect_at(self.time, p)
                    move_x = q.rect_at(self.time+interval, p)[0]-rect[0]
                    hit = sweep_circle_rect(start, (dx-move_x, dy), rect, p.radius)
                    if hit:
                        fraction, normal = hit
                        if normal in ((0., -1.), (0., 1.)):
                            boundary = rect[1]-p.radius if normal[1] < 0 else rect[1]+rect[3]+p.radius
                            hit_time = crossing_time(b.y, b.vy, gravity, boundary, interval, fraction*interval)
                            hit_dx = (b.vx*hit_time if b.launch else advance_x(b.vx, target, accel, hit_time)[0])
                            at_x = b.x+hit_dx-(q.rect_at(self.time+hit_time, p)[0]-rect[0])
                            if rect[0]-EPS <= at_x <= rect[0]+rect[2]+EPS:
                                fraction = min(1., hit_time/interval)
                        solids.append((fraction, 3, 'solid', (q, normal)))
            # World walls: no lower wall, so falling out fails.
            for rect in [(-40., -40., 40., p.height+100.), (p.width, -40., 40., p.height+100.), (0., -40., p.width, 40.)]:
                hit = sweep_circle_rect(start, delta, rect, p.radius)
                if hit:
                    solids.append((hit[0], 3, 'wall', hit[1]))
            for spike in self.level.spikes:
                if (max(start[0], start[0]+dx)+p.radius < spike.x
                        or min(start[0], start[0]+dx)-p.radius > spike.x+spike.w
                        or max(start[1], start[1]+dy)+p.radius < spike.y
                        or min(start[1], start[1]+dy)-p.radius > spike.y+spike.h):
                    continue
                for triangle in spike.triangles:
                    hit = sweep_circle_polygon(start, delta, triangle, p.radius)
                    if hit:
                        candidates.append((hit[0], 0, 'fail', '가시에 닿았어요'))
            if dy > 0 and b.y <= p.height+p.radius < b.y+dy:
                candidates.append(((p.height+p.radius-b.y)/dy, 0, 'fail', '얼음 바다로 떨어졌어요'))
            for i, fish in enumerate(self.level.collectibles):
                if i not in self.collected:
                    fraction = sweep_point_rect(start, delta, fish.rect)
                    if fraction is not None:
                        candidates.append((fraction, 1, 'fish', i))
            fraction = sweep_point_rect(start, delta, self.level.goal)
            if fraction is not None:
                candidates.append((fraction, 2, 'goal', None))
            candidates.extend(solids)
            if not candidates:
                b.x += dx; b.y += dy; b.vx = next_vx; b.vy += gravity*interval
                self.time += interval
                remaining -= interval
                continue
            fraction = min(q[0] for q in candidates)
            simultaneous = [q for q in candidates if abs(q[0]-fraction) <= 1e-7]
            elapsed = max(0., min(interval, fraction*interval))
            traveled, vx = (b.vx*elapsed, b.vx) if b.launch else advance_x(b.vx, target, accel, elapsed)
            b.x += traveled; b.y += b.vy*elapsed+.5*gravity*elapsed**2
            b.vx = vx; b.vy += gravity*elapsed
            self.time += elapsed
            remaining -= elapsed
            if any(q[2] == 'fail' for q in simultaneous):
                self.fail(next(q[3] for q in simultaneous if q[2] == 'fail'))
                return False
            for event in simultaneous:
                if event[2] == 'fish':
                    self.collected.add(event[3]); self.events.append(('fish', event[3]))
            if any(q[2] == 'goal' for q in simultaneous):
                self.cleared = True; b.vx = b.vy = 0.; b.launch = False
                self.events.append(('clear', None))
                return False
            hits = [q for q in simultaneous if q[2] in ('solid', 'wall')]
            if hits:
                tops = []
                for event in hits:
                    if event[2] != 'solid':
                        continue
                    q, normal = event[3]
                    x, y, w, _ = q.rect_at(self.time, p)
                    if (normal == (0., -1.) and b.vy > EPS and start[1]+p.radius <= y+1e-6
                            and x-EPS <= b.x <= x+w+EPS):
                        tops.append(q)
                if tops:
                    # Choose one landing effect at seams, but resolve all side contacts.
                    tops.sort(key=lambda q: 0 if q.kind in ('spring', 'switch', 'launcher') else 1)
                    self._land(tops[0])
                for _, _, kind, payload in hits:
                    q, normal = payload if kind == 'solid' else (None, payload)
                    if normal == (0., -1.):
                        continue
                    if normal == (0., 1.):
                        b.y = q.y+q.h+p.radius+1e-6 if q else p.radius+1e-6
                        if b.vy < 0: b.vy = 0.
                        b.launch = False
                    elif normal[0]:
                        # Outside the flat top, a corner is a horizontal barrier.
                        # Separate to the side so gravity cannot cause zero-time re-hits.
                        side = -1 if normal[0] < 0 else 1
                        if q:
                            x, _, w, _ = q.rect_at(self.time, p)
                            b.x = x-p.radius-1e-6 if side < 0 else x+w+p.radius+1e-6
                        else:
                            b.x = p.radius+1e-6 if side > 0 else p.width-p.radius-1e-6
                        if b.vx*side < 0: b.vx = 0.
                        b.launch = False
            # Collected sensor events vanish, so remaining time can continue.
        if remaining > 1e-10:
            # Keep sweeping the remainder; never advance the clock without motion.
            if _depth >= 8:
                self.fail('블록 사이에 끼었어요'); return False
            return (self._piece(direction, remaining/2, _depth+1)
                    and self._piece(direction, remaining/2, _depth+1))
        self.time = end_time
        # Activation/side pushes: resolve gentle overlap; only unresolved crush fails.
        for _ in range(4):
            overlaps = [q for q in self.platforms if q.id != b.launch_source and q.active(self.time, self.switches)
                        and circle_rect_overlap(b.x, b.y, p.radius, q.rect_at(self.time,p), 1e-5)]
            if not overlaps:
                break
            for q in overlaps:
                x,y,w,h=q.rect_at(self.time,p)
                if not circle_rect_overlap(b.x, b.y, p.radius, (x,y,w,h), 1e-5):
                    continue
                depth, normal = circle_rect_separation(b.x, b.y, p.radius, (x,y,w,h))
                b.x += normal[0]*(depth+1e-6)
                b.y += normal[1]*(depth+1e-6)
                if normal[0] and b.vx*normal[0] < 0: b.vx=0.
                if normal[0] == 0 and b.vy*normal[1] < 0: b.vy=0.
        still_overlaps = any(q.id != b.launch_source and q.active(self.time,self.switches) and circle_rect_overlap(b.x,b.y,p.radius,q.rect_at(self.time,p),1e-4) for q in self.platforms)
        if still_overlaps:
            self._crush_time += duration
            if any(q.kind=='timed' and not old_active[q.id] and q.active(self.time,self.switches) for q in self.platforms) or self._crush_time >= .05:
                self.fail('블록 사이에 끼었어요');return False
        else:
            self._crush_time=0.
        # Hazard check after any collision correction, before success is allowed.
        if b.y-p.radius > p.height:
            self.fail('얼음 바다로 떨어졌어요');return False
        for spike in self.level.spikes:
            if spike.near(b.x, b.y, p.radius) and any(circle_polygon_overlap(b.x,b.y,p.radius,t) for t in spike.triangles):
                self.fail('가시에 닿았어요');return False
        return True
