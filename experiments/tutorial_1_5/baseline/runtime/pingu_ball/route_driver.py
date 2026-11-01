"""Ordinary left/right input witnesses, including repeated moving-platform hops."""
import math
from .levels import waypoint_direction, condition_met


class RouteDriver:
    def __init__(self, world, path):
        self.world, self.path = world, path
        self.point = 0
        self.intercept = None
        self.planned_landing = None
        self.milestones = []
        self.by_id = {q.id: q for q in world.platforms}

    def flight(self, source_y, target_y, spring=False):
        p = self.world.physics
        speed = p.pad_speed if spring else p.bounce_speed
        discriminant = speed*speed+2*p.gravity*(target_y-source_y)
        return None if discriminant < 0 else (speed+math.sqrt(discriminant))/p.gravity

    def future_tiles(self, prefix, flight):
        w = self.world
        return [(q, q.rect_at(w.time+flight, w.physics)[0]+q.w/2)
                for q in w.platforms if q.id.startswith(prefix+'-')]

    def advance(self):
        self.milestones.append((self.point, round(self.world.time, 4), self.path[self.point][1]))
        self.point = min(self.point+1, len(self.path)-1)

    def direction(self):
        w, p, b = self.world, self.world.physics, self.world.ball
        target, condition = self.path[self.point]
        parts = condition.split(':'); kind = parts[0]
        fresh = w.time-w.last_landing_time < p.dt*1.1
        ready = False
        if kind in ('land', 'touch'):
            ready = fresh and (w.last_landing or '').startswith(parts[1])
            if kind == 'land': ready = ready and abs(b.x-target) < 2
        elif kind == 'release':
            if b.launch:
                self.advance()
                return -1 if b.vx > 0 else 1
            ready = True
        elif kind in ('flyleft', 'flyright'):
            sign = -1 if kind == 'flyleft' else 1
            ready = b.launch and sign*(b.x-float(parts[1])) >= 0
            if ready:
                self.advance()
                return -sign
            return 0
        elif kind == 'board':
            _, prefix, source_id, sign, *bounds = parts
            sign = int(sign); source = self.by_id[source_id]
            group = [q for q in w.platforms if q.id.startswith(prefix+'-')]
            flight = self.flight(source.y, group[0].y, source.kind == 'spring')
            if fresh and w.last_landing == source_id and abs(b.x-target) < 2 and flight:
                candidates = []
                for q, x in self.future_tiles(prefix, flight):
                    reach = p.max_speed*flight-12.
                    moving_right = q.rect_at(w.time+flight+.01, p)[0] > q.rect_at(w.time+flight, p)[0]
                    if moving_right != (sign > 0): continue
                    if bounds:
                        allowed = float(bounds[0]) <= x <= float(bounds[1])
                    else:
                        allowed = 48 <= sign*(x-b.x) <= reach
                    if allowed and abs(x-b.x) <= reach: candidates.append(x)
                if candidates:
                    self.intercept = max(candidates, key=lambda x: sign*x)
                    ready = True
        elif kind == 'ride':
            _, prefix, exit_id, sign = parts
            sign = int(sign); exit_block = self.by_id[exit_id]
            if fresh and (w.last_landing or '').startswith(prefix+'-'):
                flight = self.flight(self.by_id[w.last_landing].y, exit_block.y)
                reach = p.max_speed*flight-12. if flight else 0.
                ready = 0 < sign*(exit_block.x+32-b.x) <= reach
                if not ready and self.planned_landing != w.last_landing_time:
                    tiles = self.future_tiles(prefix, p.flight_time)
                    if p.platform_speed >= p.max_speed*.9:
                        # Keep moving across a fast ferry; stopping at each tile centre
                        # loses the platform. Predict its full span through reversals.
                        left = min(x-q.w/2 for q, x in tiles)+16.
                        right = max(x+q.w/2 for q, x in tiles)-16.
                        self.intercept = max(left, min(right, b.x+sign*(p.horizontal_range+64.)))
                    else:
                        choices = [x for _, x in tiles if abs(x-b.x) <= p.horizontal_range-12]
                        if choices: self.intercept = max(choices, key=lambda x: sign*x)
                    self.planned_landing = w.last_landing_time
            target = b.x if self.intercept is None else self.intercept
        else:
            ready = condition_met(w, condition)
        if ready:
            self.advance()
            target, condition = self.path[self.point]
            if condition.startswith('ride:'): target = self.intercept
            if condition.startswith(('flyleft:', 'flyright:')): return 0
        return waypoint_direction(w, target)
