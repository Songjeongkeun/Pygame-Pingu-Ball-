"""Pygame에 의존하지 않고 종료 상태에서 자동 재시작하지 않는 학습용 인터페이스."""
import math
from .engine import World
from .levels import make_level


class Environment:
    """왼쪽·무입력·오른쪽 행동과 선택적인 일회성 아이템 사용을 제공한다."""

    def __init__(self, stage=0, time_limit=None):
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError('time_limit must be finite and positive')
        self.stage = stage
        self._time_limit_override = time_limit
        self.world = World(make_level(stage), auto_reset=False)
        self.time_limit = self.world.level.time_limit if time_limit is None else time_limit
        self.previous_action = 0
        self.done = False

    def reset(self, stage=None):
        next_stage = self.stage if stage is None else stage
        world = World(make_level(next_stage), auto_reset=False)
        self.stage, self.world = next_stage, world
        self.time_limit = world.level.time_limit if self._time_limit_override is None else self._time_limit_override
        self.previous_action = 0
        self.done = False
        return self.observe(), {'stage': self.stage}

    def observe(self):
        world = self.world
        ball = world.ball
        platforms = []
        for block in world.platforms:
            rect = block.rect_at(world.time)
            moving_phase = (world.time*world.physics.platform_speed+block.phase) % (2*block.travel) if block.travel else 0.
            platforms.append({'id': block.id, 'kind': block.kind, 'rect': rect,
                              'group': block.group, 'arrow': block.arrow,
                              'travel_bounds': (block.x, block.x+block.travel),
                              'period': block.period, 'active_for': block.active_for,
                              'active': block.active(world.time, world.switches),
                              'destroyed': block.destroyed, 'armed': block.armed,
                              'break_remaining': None if block.break_at is None else max(0., block.break_at-world.time),
                              'direction': (1 if moving_phase < block.travel else -1) if block.travel else 0,
                              'cycle': (world.time+block.phase) % block.period if block.kind == 'timed' else None})
        return {'stage': self.stage, 'goal': world.level.goal,
                'world_size': (world.physics.width, world.physics.height),
                'collectibles': [{'index': index, 'rect': fish.rect, 'value': fish.value,
                                  'collected': index in world.collected}
                                 for index, fish in enumerate(world.level.collectibles)],
                'spikes': [{'rect': (spike.x, spike.y, spike.w, spike.h),
                            'triangles': spike.triangles} for spike in world.level.spikes],
                'ball': (ball.x, ball.y, ball.vx, ball.vy), 'launch': ball.launch,
                'launch_source': ball.launch_source, 'previous_action': self.previous_action,
                'time': world.time, 'time_remaining': max(0., self.time_limit-world.time),
                'collected': sorted(world.collected), 'switches': dict(world.switches),
                'power_up': ball.power_up,
                'held_powerups': list(ball.held_powerups),
                'powerups': [{'index': index, 'kind': item.kind, 'rect': item.rect,
                              'collected': index in world.collected_powerups}
                             for index, item in enumerate(world.level.powerups)],
                'boost_direction': ball.boost_direction,
                'boost_remaining': max(0., ball.boost_until-world.time),
                'forward_jump_active': ball.forward_target_x is not None,
                'forward_jump_remaining': abs(ball.forward_target_x-ball.x) if ball.forward_target_x is not None else 0.,
                'specials': world.special.observe(),
                'projectiles': [{'id':shot.id,'center':shot.center(world.time),
                                 'velocity':(shot.vx,shot.vy),'radius':shot.radius,
                                 'expires':shot.born+shot.lifetime} for shot in world.special.projectiles],
                'platforms': platforms, 'crush_time': world._crush_time,
                'cleared': world.cleared, 'failed': world.failed}

    def step(self, action, *, use_power=False, power_direction=0):
        if self.done:
            raise RuntimeError('Call reset() after a terminal or truncated step')
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError('action must be -1, 0 or 1')
        if type(use_power) is not bool or type(power_direction) is not int or power_direction not in (-1, 0, 1):
            raise ValueError('use_power must be bool and power_direction must be -1, 0 or 1')
        before = self.world.plus-self.world.minus
        duration = min(self.world.physics.dt, self.time_limit-self.world.time)
        self.world.step(action, duration, use_power=use_power, power_direction=power_direction)
        self.previous_action = action
        terminated = self.world.cleared or self.world.failed
        truncated = not terminated and self.world.time >= self.time_limit-1e-10
        reward = self.world.plus-self.world.minus-before
        reward += 7 if self.world.cleared else -7 if self.world.failed else 0
        self.done = terminated or truncated
        reason = 'success' if self.world.cleared else self.world.last_failure if self.world.failed else 'time_limit' if truncated else ''
        info = {'stage': self.stage, 'reason': reason,
                'score': self.world.score if self.world.cleared and self.stage else None}
        return self.observe(), reward, terminated, truncated, info
