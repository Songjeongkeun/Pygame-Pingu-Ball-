"""Tutorials 1..5 only: native physics, three controls, observable contact reward."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .all_tutorial_env import AllTutorialEnv, ENTITY_DIM, maximum_entities, distance_feature
from .map_input_env import MapInputEnv, LOCAL_COUNT

CAPACITY = maximum_entities()
FIELDS = ENTITY_DIM + 6  # endpoints (4), visited support, last landing
GLOBALS = 29  # existing 24 globals (no visitation grid) + map one-hot
OBS_SIZE = GLOBALS + (CAPACITY + LOCAL_COUNT) * FIELDS


class FiveRawEnv(MapInputEnv):
    def __init__(self, stage=1, reward_mode="contact", limit=40., repeat=12):
        if stage not in range(1, 6):
            raise ValueError("Only tutorial stages 1..5 are allowed")
        if reward_mode not in ("contact", "terminal"):
            raise ValueError(reward_mode)
        self.reward_mode = reward_mode
        self.touched = set()
        self.landing_index = None
        self.support_lookup = {}
        super().__init__(stage=stage, variant="local", limit=limit, repeat=repeat, novelty=0.)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "global": spaces.Box(-8., 8., (GLOBALS,), dtype=np.float32),
            "entities": spaces.Box(-8., 8., (CAPACITY, FIELDS), dtype=np.float32),
            "local": spaces.Box(-8., 8., (LOCAL_COUNT, FIELDS), dtype=np.float32)})
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.touched = set()
        self.landing_index = None
        self.support_lookup = {}
        super().reset(seed=seed, options=options)
        for p in self.level.platforms:
            for i, (x, y, width, height, kind) in enumerate(self.terrain):
                if (p.kind == kind and x <= p.x and y <= p.y
                        and p.x + p.w <= x + width and p.y + p.h <= y + height):
                    self.support_lookup[p.id] = i
                    break
        for i, device in enumerate(self.world.special.devices.values(), len(self.terrain)):
            if device.id in self.world.special.platforms:
                self.support_lookup[device.id] = i
        # Do not pay for the starting support; no authored route/waypoints.
        bx, by = self.level.spawn
        below = [(r[1] - by, i) for i, r in enumerate(self.terrain)
                 if r[0] <= bx <= r[0] + r[2] and r[1] >= by]
        if below:
            self.touched.add(min(below)[1])
        return self.observe(), self.info()

    def observe(self):
        base = AllTutorialEnv.observe(self)
        records = self.entity_records()
        ball = self.world.ball
        entities = np.zeros((CAPACITY, FIELDS), dtype=np.float32)
        entities[:, :ENTITY_DIM] = base['entities']
        for i, row in enumerate(records):
            if row['endpoints'] is not None:
                (x1, y1), (x2, y2) = row['endpoints']
                entities[i, ENTITY_DIM:ENTITY_DIM+4] = [distance_feature(x1-ball.x),
                    distance_feature(y1-ball.y), distance_feature(x2-ball.x), distance_feature(y2-ball.y)]
            entities[i, -2] = float(i in self.touched)
            entities[i, -1] = float(i == self.landing_index)
        local = np.zeros((LOCAL_COUNT, FIELDS), dtype=np.float32)
        for j, i in enumerate(self.local_indices(records)):
            local[j] = entities[i]
            x, y, width, height = records[i]['rect']
            local[j, :6] = [(x-ball.x)/256, (x+width-ball.x)/256,
                           (y-ball.y)/256, (y+height-ball.y)/256, width/64, height/64]
        global_values = np.zeros(GLOBALS, dtype=np.float32)
        global_values[:24] = base['global'][:24]
        global_values[15] = self.previous_action / 2
        global_values[20] = self.bonus_credit / .1
        global_values[24 + self.stage - 1] = 1.
        return {
            'global': np.clip(global_values, -8, 8),
            'entities': np.clip(entities, -8, 8), 'local': np.clip(local, -8, 8)}

    def contact_credit(self, indices, failed=False):
        new = set(indices) - self.touched
        self.touched.update(indices)
        if failed or self.reward_mode == 'terminal':
            return 0.
        return min(.01 * len(new), max(0., .1 - self.bonus_credit))

    def step(self, action, on_tick=None):
        if self.done:
            raise RuntimeError('Reset before stepping a finished episode')
        if not self.action_space.contains(action):
            raise ValueError(action)
        before = self.ticks
        contacts = []
        for _ in range(self.repeat):
            dt = min(self.world.physics.dt, self.limit-self.world.time)
            if dt < 1e-10 or self.world.failed or self.world.cleared:
                break
            self.world.step(int(action)-1, dt)
            self.ticks += 1
            self.min_goal_distance = min(self.min_goal_distance, self.goal_distance())
            for event, value in self.world.events:
                if event == 'land' and value in self.support_lookup:
                    self.landing_index = self.support_lookup[value]
                    contacts.append(self.landing_index)
            if on_tick:
                on_tick(self.world, self.ticks, int(action)-1)
        self.previous_action = int(action)
        self.decisions += 1
        self.done = bool(self.world.cleared or self.world.failed or self.world.time >= self.limit-1e-9)
        base = 1. if self.world.cleared else -1. if self.done else 0.
        bonus = self.contact_credit(contacts, self.world.failed)
        self.bonus_credit += bonus
        self.bonus_return += bonus
        self.raw_base_return += base
        self.episode_return += base + bonus
        cell = self.cell()
        self.visited[cell] = 1  # diagnostic only, no grid-based reward
        info = self.info(self.ticks-before)
        info['landed_supports'] = len(self.touched)
        return self.observe(), base+bonus, self.done, False, info


class FiveEnv(gym.ObservationWrapper):
    def __init__(self, **options):
        super().__init__(FiveRawEnv(**options))
        self.observation_space = spaces.Box(-8., 8., (OBS_SIZE,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['global'], obs['entities'].reshape(-1),
                               obs['local'].reshape(-1)]).astype(np.float32, copy=False)
