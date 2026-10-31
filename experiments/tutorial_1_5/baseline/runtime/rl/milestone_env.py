"""User's +100/-25/+10 objective, original tutorials 1..5, real land events."""
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .five_env import FiveRawEnv, GLOBALS as BASE_GLOBALS, CAPACITY, FIELDS, LOCAL_COUNT
from .all_tutorial_env import distance_feature

MAX_MILESTONES = 4
MILESTONE_FIELDS = 11  # two rectangles (8), valid, armed, completed
GLOBALS = BASE_GLOBALS + MAX_MILESTONES*MILESTONE_FIELDS
OBS_SIZE = GLOBALS + (CAPACITY+LOCAL_COUNT)*FIELDS


@dataclass(frozen=True)
class Milestone:
    id: str
    source: tuple
    target: tuple


def definitions(stage):
    group = lambda name, n: tuple(f'{name}-{i}' for i in range(n))
    if stage == 1:
        return tuple(Milestone(f'break-crossing-{i}', (f'break-{i}',),
                     (f'break-{i+1}',) if i < 3 else group('finish', 3)) for i in range(4))
    if stage == 2:
        return (Milestone('spring-to-middle', ('spring-1-0',), group('middle', 3)),
                Milestone('spring-to-upper', ('spring-2-0',), group('upper', 2)),
                Milestone('spring-to-finish', ('spring-3-0',), group('finish', 4)))
    if stage == 3:
        return (Milestone('ferry-to-finish', group('ferry', 3), group('finish', 3)),)
    if stage == 4:
        return (Milestone('lift-to-finish', ('lift',), group('finish', 8)),)
    if stage == 5:
        return tuple(Milestone(f'clock-crossing-{i}', (f'clock-{i}',),
                     (f'clock-{i+1}',) if i < 2 else group('finish', 4)) for i in range(3))
    raise ValueError('Only tutorials 1..5')


class MilestoneTracker:
    def __init__(self, specs):
        self.specs = specs
        self.armed, self.completed = set(), set()
        self.last_events = []
        for spec in specs:
            if set(spec.source) & set(spec.target):
                raise ValueError('A connected support cannot reward moving within itself')

    def land(self, platform_id, tick):
        awarded = []
        for spec in self.specs:
            if (spec.id in self.armed and spec.id not in self.completed
                    and platform_id in spec.target):
                self.completed.add(spec.id)
                self.last_events.append(dict(id=spec.id, target=platform_id, tick=tick))
                awarded.append(spec.id)
            if platform_id in spec.source:
                self.armed.add(spec.id)
        return awarded


def terminal_points(cleared, failed, timeout):
    # Failure + timeout is one terminal event; success at the deadline is success.
    return -25. if failed else 100. if cleared else -25. if timeout else 0.


class MilestoneRawEnv(FiveRawEnv):
    def __init__(self, stage=1, reward_mode='milestone', limit=180., repeat=6):
        if reward_mode != 'milestone':
            raise ValueError('Use milestone reward mode')
        self.specs = definitions(stage)
        self.tracker = MilestoneTracker(self.specs)
        self.reward_events = []
        super().__init__(stage, reward_mode='terminal', limit=limit, repeat=repeat)
        self.observation_space = spaces.Dict({
            'global': spaces.Box(-8., 8., (GLOBALS,), dtype=np.float32),
            'entities': spaces.Box(-8., 8., (CAPACITY, FIELDS), dtype=np.float32),
            'local': spaces.Box(-8., 8., (LOCAL_COUNT, FIELDS), dtype=np.float32)})
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.tracker = MilestoneTracker(self.specs)
        self.reward_events = []
        super().reset(seed=seed, options=options)
        self.platform_by_id = {p.id: p for p in self.world.platforms}
        self.platform_by_id.update(self.world.special.platforms)
        for spec in self.specs:
            assert all(i in self.platform_by_id for i in spec.source+spec.target)
        return self.observe(), self.info()

    def bounds(self, ids):
        # During base-class construction platform_by_id may refer to an old reset.
        platforms = {p.id:p for p in self.world.platforms}
        platforms.update(self.world.special.platforms)
        rects = [self.world.special.rect(self.world.special.devices[i], self.world.time)
                 if i in self.world.special.devices else platforms[i].rect_at(self.world.time, self.world.physics)
                 for i in ids]
        left, top = min(r[0] for r in rects), min(r[1] for r in rects)
        return left, top, max(r[0]+r[2] for r in rects)-left, max(r[1]+r[3] for r in rects)-top

    def observe(self):
        obs = super().observe()
        ball = self.world.ball
        extra = np.zeros((MAX_MILESTONES, MILESTONE_FIELDS), dtype=np.float32)
        for i, spec in enumerate(self.specs):
            for j, ids in enumerate((spec.source, spec.target)):
                x,y,w,h = self.bounds(ids)
                extra[i, j*4:j*4+4] = [distance_feature(x-ball.x), distance_feature(y-ball.y),
                                      distance_feature(x+w-ball.x), distance_feature(y+h-ball.y)]
            extra[i, 8:] = (1., float(spec.id in self.tracker.armed), float(spec.id in self.tracker.completed))
        obs['global'][20] = len(self.tracker.completed)/MAX_MILESTONES
        obs['global'] = np.concatenate([obs['global'], extra.reshape(-1)])
        return obs

    def info(self, tick_delta=0):
        info = super().info(tick_delta)
        info.update(milestones=sorted(self.tracker.completed), armed=sorted(self.tracker.armed),
                    milestone_events=list(self.tracker.last_events),
                    reward_events=list(self.reward_events), learning_return=self.episode_return/100.)
        return info

    def step(self, action, on_tick=None):
        if self.done:
            raise RuntimeError('Reset before stepping a terminal episode')
        if not self.action_space.contains(action):
            raise ValueError(action)
        before, bonus = self.ticks, 0.
        self.reward_events = []
        for _ in range(self.repeat):
            dt = min(self.world.physics.dt, self.limit-self.world.time)
            if dt < 1e-10 or self.world.failed or self.world.cleared:
                break
            self.world.step(int(action)-1, dt)
            self.ticks += 1
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            for event, value in self.world.events:
                if event == 'land' and not self.world.failed:
                    if value in self.support_lookup:
                        self.landing_index = self.support_lookup[value]
                        self.touched.add(self.landing_index)
                    for milestone in self.tracker.land(value, self.ticks):
                        bonus += 10.
                        self.reward_events.append(dict(kind='milestone', id=milestone, points=10.))
            if on_tick:
                on_tick(self.world, self.ticks, int(action)-1)
        self.previous_action, self.decisions = int(action), self.decisions+1
        timeout = self.world.time >= self.limit-1e-9
        self.done = bool(self.world.cleared or self.world.failed or timeout)
        base = terminal_points(self.world.cleared, self.world.failed, timeout)
        if self.done:
            self.reward_events.append(dict(kind='success' if base>0 else 'failure', points=base))
        self.raw_base_return += base
        self.bonus_return += bonus
        self.episode_return += base+bonus  # original user points in all reports
        self.visited[self.cell()] = 1  # diagnostics only
        return self.observe(), (base+bonus)/100., self.done, False, self.info(self.ticks-before)


class MilestoneEnv(gym.ObservationWrapper):
    def __init__(self, **options):
        super().__init__(MilestoneRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['global'], obs['entities'].reshape(-1), obs['local'].reshape(-1)]).astype(np.float32)
