"""Untrained transfer adapter: existing Pingu Ball mixed-map physics -> Stage-2 PPO input.

WHAT IT IS:
    1. Loads the exact released mixed RL map with make_rl_level and expand_playfield.
    2. Uses the exact World physics, 3 left/idle/right actions, 6 physics ticks/action.
    3. Encodes the same GLOBALS / FIELDS / LOCAL_COUNT schema as rl.milestone_env.
    4. Dynamically keeps the CAPACITY nearest real objects if a map contains more.
    5. Zeros all 4 tutorial-specific milestone slots (no fabricated route).
    6. Sets the stage-id one-hot to stage 2, the policy's trained stage.

WHAT IT IS NOT:
    This is not a native tutorial observation on an unseen map. No prior trained
    observations had 80+ objects / this layout. The nearest-object truncation,
    zero milestone fields, and stage-id placeholder are an OUT-OF-DISTRIBUTION
    adapter. Report successes as an experimental transfer result, never as an
    apples-to-apples stage2 score. No reward shaping/teacher/training occurs.
"""
from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pingu_ball.engine import World
from pingu_ball.playfield import expand_playfield
from pingu_ball.rl_maps import make_rl_level
from rl.all_tutorial_env import (DEVICE_KIND, ENTITY_DIM, KINDS, distance_feature,
                                  merge_rectangles)
from rl.milestone_env import (CAPACITY, FIELDS, GLOBALS, LOCAL_COUNT, OBS_SIZE,
                              MAX_MILESTONES, MILESTONE_FIELDS)
from .adapter_utils import ranked_indices


class MixedMapTransferEnv(gym.Env):
    """Evaluate a *frozen* tutorial-2 policy on a never-trained mixed map."""

    metadata = {'render_modes': []}

    def __init__(self, map_number: int = 1, *, limit: float = 180., repeat: int = 6):
        super().__init__()
        if type(map_number) is not int or not (1 <= map_number <= 10):
            raise ValueError('map_number must be in 1..10')
        if not math.isfinite(limit) or not 0 < limit <= 180.:
            raise ValueError('limit must be >0 and <=180 seconds')
        if repeat != 6:
            raise ValueError('Frozen stage2 PPO requires exactly 6 physics ticks/action')
        self.map_number = map_number
        self.limit = float(limit)
        self.repeat = repeat
        self.level = expand_playfield(make_rl_level(map_number))
        self.terrain = merge_rectangles(self.level.platforms)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)
        if GLOBALS != 29 + MAX_MILESTONES * MILESTONE_FIELDS:
            raise RuntimeError('Team milestone global schema changed; stop rather than silently adapt')
        if FIELDS != ENTITY_DIM + 6:
            raise RuntimeError('Team entity schema changed; stop rather than silently adapt')
        self.world: World
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world = World(self.level, auto_reset=False)
        self.done = False
        self.previous_action = 1  # same as rl.all_tutorial_env reset
        self.decisions = 0
        self.ticks = 0
        self.touched: set[int] = set()
        self.landing_index = None
        self.max_x = self.world.ball.x
        self.min_goal_distance = self.goal_distance()
        self.start_goal_distance = self.min_goal_distance
        self._support_lookup = self._build_support_lookup()
        self._static_hazard_bounds = {}
        for d in self.world.special.devices.values():
            if d.type_id == 201:
                polys = self.world.special.polygons(d, 0.)
                points = [pt for poly in polys for pt in poly]
                xs, ys = zip(*points)
                self._static_hazard_bounds[d.id] = (
                    min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self.last_reward = 0.
        return self.observe(), self.info()

    def _build_support_lookup(self):
        lookup: dict[str, int] = {}
        for p in self.level.platforms:
            for i, (x, y, width, height, kind) in enumerate(self.terrain):
                if (kind == p.kind and x <= p.x + 1e-6 and y <= p.y + 1e-6 and
                        p.x+p.w <= x+width+1e-6 and p.y+p.h <= y+height+1e-6):
                    lookup[p.id] = i
                    break
        for i, d in enumerate(self.world.special.devices.values(), len(self.terrain)):
            if d.id in self.world.special.platforms:
                lookup[d.id] = i
        return lookup

    def goal_distance(self):
        b = self.world.ball
        gx, gy, gw, gh = self.level.goal
        return math.hypot(b.x-gx-gw/2, b.y-gy-gh/2)

    def _entity(self, rect, kind, *, velocity=(0., 0.), active=True,
                travel=0., arrow=0., phase=0., until=0.,
                available=True, destroyed=False, source=False):
        """Same 31 field order and normalization as rl.all_tutorial_env.entity()."""
        if kind not in KINDS:
            raise ValueError(f'Unsupported native map object kind {kind!r}')
        x, y, width, height = rect
        b = self.world.ball
        values = [distance_feature(x-b.x), distance_feature(x+width-b.x),
                  distance_feature(y-b.y), distance_feature(y+height-b.y),
                  width/512, height/256,
                  velocity[0]/640, velocity[1]/640,
                  float(active), travel/512, arrow,
                  math.sin(phase*math.tau), math.cos(phase*math.tau),
                  until/10, float(available), float(destroyed), float(source), 1.]
        values += [float(kind == name) for name in KINDS]
        assert len(values) == ENTITY_DIM
        return values

    def _records(self):
        """All REAL world entities before truncation; no authored route or labels."""
        w = self.world
        b = w.ball
        records = []
        for i, (x, y, width, height, kind) in enumerate(self.terrain):
            rect = (x, y, width, height)
            records.append(dict(index=i, rect=rect, kind=kind, endpoints=None,
                                values=self._entity(rect, kind)))
        offset = len(records)
        for j, d in enumerate(w.special.devices.values()):
            if d.type_id not in DEVICE_KIND:
                raise ValueError(f'Unknown game device: {d.id} (type {d.type_id})')
            q = w.special.platforms.get(d.id)
            opts = d.options
            rect = w.special.rect(d, w.time)
            nxt = w.special.rect(d, w.time+.001)
            velocity = ((nxt[0]-rect[0])/.001, (nxt[1]-rect[1])/.001)
            active = q.active(w.time, w.switches) if q else w.special.enabled(d)
            phase = 0.
            until = 0.
            travel = opts.get('travel', 0.)
            arrow = opts.get('direction', 0.)
            if d.type_id == 201:
                rect = self._static_hazard_bounds[d.id]
            elif d.type_id == 302:
                period = opts.get('period', 2.8)
                on = opts.get('active_for', 2.)
                age = (w.time + opts.get('phase', 0.)) % period
                phase = age / period
                until = on-age if age < on else period-age
            elif d.type_id in (501, 508):
                travel = opts.get('travel', 128.)
                speed = opts.get('speed', w.physics.platform_speed)
                phase = ((w.time*speed + opts.get('phase', 0.)) % (2*travel)) / (2*travel)
            endpoints = None
            if d.type_id in (501, 508):
                endpoints = ((d.x, d.y),
                             (d.x + (travel if d.type_id == 501 else 0),
                              d.y + (travel if d.type_id == 508 else 0)))
            values = self._entity(rect, DEVICE_KIND[d.type_id], velocity=velocity,
                                  active=active, travel=travel, arrow=arrow,
                                  phase=phase, until=until,
                                  destroyed=q.destroyed if q else w.special.state[d.id]['destroyed'],
                                  source=b.launch_source == d.id)
            records.append(dict(index=offset+j, rect=rect, kind=DEVICE_KIND[d.type_id],
                                endpoints=endpoints, values=values))
        for j, item in enumerate(self.level.powerups):
            available = j not in w.collected_powerups
            kind = 'jump_item' if item.kind == 'up' else 'dash_item'
            records.append(dict(index=len(records), rect=item.rect, kind=kind,
                                endpoints=None,
                                values=self._entity(item.rect, kind,
                                                    active=available, available=available)))
        return records

    def observe(self):
        """Always returns original MilestonePPO OBS_SIZE without hiding mismatch."""
        w, b = self.world, self.world.ball
        gx, gy, gw, gh = self.level.goal
        globals24 = [b.x/self.level.width, b.y/self.level.height,
                     b.vx/640, b.vy/640,
                     distance_feature(gx+gw/2-b.x), distance_feature(gy+gh/2-b.y),
                     float(b.launch), float(b.power_up=='up'), float(b.power_up=='forward'),
                     distance_feature((b.forward_target_x or b.x)-b.x),
                     float(b.forward_target_x is not None),
                     b.forward_direction, max(0., b.boost_until-w.time), b.boost_direction,
                     max(0., 1-w.time/self.limit),
                     self.previous_action/2, w._crush_time,
                     w.special.speed_scale(), self.level.width/3000,
                     self.level.height/1500,
                     0., 0., 0., w.physics.radius/64]
        assert len(globals24) == 24
        # The *frozen stage2 model* knows only one-hot index 2; a sixth class
        # cannot be introduced without changing its input size and weights.
        global29 = np.zeros(29, dtype=np.float32)
        global29[:24] = globals24
        global29[24 + 2 - 1] = 1.  # trained stage=2, not an assertion about the mixed-map ID
        # No artificial source/target milestone pairs. Tutorial-only 4x11=44
        # slots are all zero. The policy is therefore out-of-distribution.
        global_input = np.zeros(GLOBALS, dtype=np.float32)
        global_input[:29] = global29

        records = self._records()
        ranked = ranked_indices(records, b.x, b.y)
        entities = np.zeros((CAPACITY, FIELDS), dtype=np.float32)
        local = np.zeros((LOCAL_COUNT, FIELDS), dtype=np.float32)

        def row(rec):
            values = np.zeros(FIELDS, dtype=np.float32)
            values[:ENTITY_DIM] = rec['values']
            endpoints = rec['endpoints']
            if endpoints is not None:
                (x1, y1), (x2, y2) = endpoints
                values[ENTITY_DIM:ENTITY_DIM+4] = [distance_feature(x1-b.x),
                    distance_feature(y1-b.y), distance_feature(x2-b.x),
                    distance_feature(y2-b.y)]
            values[-2] = float(rec['index'] in self.touched)
            values[-1] = float(rec['index'] == self.landing_index)
            return values

        for j, i in enumerate(ranked[:CAPACITY]):
            entities[j] = row(records[i])
        for j, i in enumerate(ranked[:LOCAL_COUNT]):
            rec = records[i]
            vector = row(rec)
            x, y, width, height = rec['rect']
            # native MapInputEnv/FiveRawEnv uses linear local positions
            vector[:6] = [(x-b.x)/256, (x+width-b.x)/256,
                          (y-b.y)/256, (y+height-b.y)/256,
                          width/64, height/64]
            local[j] = vector
        merged = np.concatenate([global_input, entities.reshape(-1), local.reshape(-1)])
        if merged.shape != (OBS_SIZE,):
            raise RuntimeError(f'Mixed map observation {merged.shape} vs trained {(OBS_SIZE,)}')
        return np.clip(merged, -8., 8.).astype(np.float32, copy=False)

    def info(self):
        w, b = self.world, self.world.ball
        failed = bool(w.failed)
        success = bool(w.cleared)
        timeout = bool(self.done and not failed and not success)
        return {'map': self.map_number, 'map_title': self.level.title,
                'success': success, 'is_success': success,
                'reason': ('success' if success else w.last_failure if failed
                           else 'time_limit' if timeout else ''),
                'seconds': round(float(w.time), 5),
                'decisions': self.decisions, 'ticks': self.ticks,
                'max_x': round(float(self.max_x), 3),
                'end_x': round(float(b.x), 3), 'end_y': round(float(b.y), 3),
                'start_goal_distance': round(float(self.start_goal_distance), 3),
                'min_goal_distance': round(float(self.min_goal_distance), 3),
                'final_goal_distance': round(float(self.goal_distance()), 3),
                'full_entities': len(self.terrain) + len(w.special.devices) + len(self.level.powerups),
                'visible_full_capacity': CAPACITY,
                'visible_local_capacity': LOCAL_COUNT,
                'adapter': 'nearest_native_entities_zero_milestones_stage2_onehot',
                'reward_mode': 'terminal_only_eval_not_training'}

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended; call reset()')
        if not self.action_space.contains(action):
            raise ValueError(f'Action out of [0,1,2]: {action!r}')
        action = int(action)
        world = self.world
        for _ in range(self.repeat):
            dt = min(world.physics.dt, self.limit-world.time)
            if dt < 1e-10 or world.failed or world.cleared:
                break
            world.step(action-1, dt)
            self.ticks += 1
            self.max_x = max(self.max_x, world.ball.x)
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            if not world.failed:
                for event, platform_id in world.events:
                    if event == 'land' and platform_id in self._support_lookup:
                        ix = self._support_lookup[platform_id]
                        self.touched.add(ix)
                        self.landing_index = ix
        self.previous_action = action
        self.decisions += 1
        self.done = bool(world.failed or world.cleared or world.time >= self.limit-1e-9)
        terminal = (100. if world.cleared else -25. if self.done else 0.)
        reward = terminal / 100.
        self.last_reward += reward
        return self.observe(), reward, self.done, False, self.info()
