"""게임 코드 수정 없이 개인 보상의 조건·중복 지급을 검증한다."""
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from tutorial_2.tutorial_test.personal_rewards import PersonalRewardTracker


@dataclass
class Spec:
    id: str
    source: tuple
    target: tuple


class FakeEnv:
    def __init__(self, stage):
        self.stage = stage
        self.world = SimpleNamespace(
            time=0.0, failed=False, cleared=False, events=[],
            physics=SimpleNamespace(radius=14.0, bounce_height=96.0),
            ball=SimpleNamespace(x=0., y=0.),
            special=SimpleNamespace(devices={}),
        )
        if stage == 2:
            self.specs = [Spec('spring-one', ('spring-a',), ('target-a',))]
            self.platform_by_id = {'spring-a': SimpleNamespace(kind='spring'),
                                   'target-a': SimpleNamespace(kind='normal')}
            self.rects = {'spring-a': (0., 200., 64., 64.),
                          'target-a': (160., 100., 64., 64.)}
        else:
            self.specs = [Spec('lift-finish', ('elevator',), ('target',))]
            self.world.special.devices['elevator'] = SimpleNamespace(
                type_id=508, y=0., options={'travel': 100.})
            self.platform_by_id = {'elevator': SimpleNamespace(kind='normal'),
                                   'target': SimpleNamespace(kind='normal')}
            self.rects = {'elevator': (0., 100., 64., 64.),
                          'target': (160., 0., 64., 64.)}

    def bounds(self, ids):
        assert len(ids) == 1
        return self.rects[ids[0]]


class TestPersonalRewards(unittest.TestCase):
    def test_spring_requires_actual_landing(self):
        env = FakeEnv(2)
        tracker = PersonalRewardTracker(env)
        env.world.ball.x = 180.
        env.world.ball.y = 86.
        self.assertEqual(tracker.on_tick(env.world), [])
        self.assertEqual(tracker.total, 0.)

    def test_spring_approach_bonus_only_once(self):
        env = FakeEnv(2)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'spring-a')]
        tracker.on_tick(env.world)
        env.world.events = []
        env.world.time = .2
        env.world.ball.x = 100.
        env.world.ball.y = 86.
        r = tracker.on_tick(env.world)
        self.assertEqual([x['kind'] for x in r], ['target_height'])
        env.world.ball.x = 165.
        r = tracker.on_tick(env.world)
        self.assertEqual([x['kind'] for x in r], ['target_horizontal'])
        for i in range(5):
            self.assertEqual(tracker.on_tick(env.world), [])
        self.assertEqual(tracker.total, 1.)

    def test_spring_timeout_cancels_bonuses(self):
        env = FakeEnv(2)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'spring-a')]
        tracker.on_tick(env.world)
        env.world.events = []
        env.world.time = 3.1
        env.world.ball.x = 170.
        env.world.ball.y = 86.
        self.assertEqual(tracker.on_tick(env.world), [])
        self.assertEqual(tracker.total, 0.)

    def test_spring_other_landing_cancels_window(self):
        env = FakeEnv(2)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'spring-a')]
        tracker.on_tick(env.world)
        env.world.events = [('land', 'other-platform')]
        tracker.on_tick(env.world)
        env.world.events = []
        env.world.ball.x = 165.
        env.world.ball.y = 86.
        self.assertEqual(tracker.on_tick(env.world), [])

    def test_spring_failed_tick_has_no_payment(self):
        env = FakeEnv(2)
        tracker = PersonalRewardTracker(env)
        env.world.failed = True
        env.world.events = [('land', 'spring-a')]
        self.assertEqual(tracker.on_tick(env.world), [])

    def test_lift_actual_landing_and_relative_height(self):
        env = FakeEnv(4)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'elevator')]
        got = tracker.on_tick(env.world)
        self.assertEqual([e['kind'] for e in got], ['lift_first'])
        self.assertEqual(tracker.total, 12.)
        env.world.events = []
        env.world.time = .5
        env.rects['elevator'] = (0., 40., 64., 64.)
        env.world.ball.x = 32.
        env.world.ball.y = 0.
        got = tracker.on_tick(env.world)
        self.assertEqual([e['kind'] for e in got], ['lift_high'])
        self.assertEqual(tracker.total, 24.)
        self.assertEqual(tracker.on_tick(env.world), [])

    def test_lift_no_credit_without_contact(self):
        env = FakeEnv(4)
        tracker = PersonalRewardTracker(env)
        env.rects['elevator'] = (0., 0., 64., 64.)
        env.world.ball.x = 20.
        env.world.ball.y = -10.
        self.assertEqual(tracker.on_tick(env.world), [])
        self.assertEqual(tracker.total, 0.)

    def test_lift_high_credit_requires_recent_contact(self):
        env = FakeEnv(4)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'elevator')]
        tracker.on_tick(env.world)
        env.world.events = []
        env.world.time = 2.
        env.rects['elevator'] = (0., 0., 64., 64.)
        env.world.ball.x = 32.
        env.world.ball.y = -10.
        self.assertEqual(tracker.on_tick(env.world), [])
        self.assertEqual(tracker.total, 12.)

    def test_reset_clears_episode_rewards(self):
        env = FakeEnv(4)
        tracker = PersonalRewardTracker(env)
        env.world.events = [('land', 'elevator')]
        tracker.on_tick(env.world)
        tracker.reset()
        self.assertEqual(tracker.total, 0.)
        self.assertEqual(len(tracker.lift_first_paid), 0)


if __name__ == '__main__':
    unittest.main()
