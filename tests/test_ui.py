"""Headless integration tests with actual Pygame events and surfaces."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pathlib import Path
import tempfile
import unittest
import pygame
from pingu_ball.app import App
from pingu_ball.storage import Progress


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'progress.json'
        self.app = App(self.path, headless=True)
        pygame.event.clear()
        self.app.render()

    def tearDown(self):
        pygame.quit()
        self.temp.cleanup()

    def click(self, x, y):
        app = self.app
        at = (round(x*app.scale+app.offset[0]), round(y*app.scale+app.offset[1]))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': at}))
        app.process_events()
        app.render()

    def key(self, key, pressed=True):
        event = pygame.KEYDOWN if pressed else pygame.KEYUP
        pygame.event.post(pygame.event.Event(event, {'key': key}))
        self.app.process_events()

    def test_locked_cards_and_automatic_tutorial_start(self):
        self.click(500, 350)
        self.assertEqual(self.app.scene, 'menu')
        self.click(200, 350)
        self.assertEqual(self.app.scene, 'play')
        self.assertEqual(self.app.world.level.index, 0)
        self.app.update(1/60, 0)
        self.assertGreater(self.app.world.time, 0)
        self.assertGreater(self.app.world.ball.y, 594.)

    def test_pause_resume_and_r_reset_clear_input(self):
        app = self.app
        app.start(0)
        app.update(.05, 1)
        app.render()
        self.click(1080, 32)
        self.assertTrue(app.world.paused)
        before = (app.world.time, app.world.ball.x, app.world.ball.y)
        app.update(.1, 1)
        self.assertEqual(before, (app.world.time, app.world.ball.x, app.world.ball.y))
        self.click(800, 440)
        self.assertFalse(app.world.paused)
        self.key(pygame.K_RIGHT)
        self.assertIn(pygame.K_RIGHT, app.held_keys)
        self.key(pygame.K_r)
        self.assertEqual(app.world.time, 0.)
        self.assertFalse(app.held_keys)
        app.update(.05, 0)
        self.assertGreater(app.world.time, 0.)

    def test_both_direction_keys_cancel(self):
        self.app.start(0)
        self.key(pygame.K_LEFT)
        self.key(pygame.K_RIGHT)
        direction = int(pygame.K_RIGHT in self.app.held_keys)-int(pygame.K_LEFT in self.app.held_keys)
        self.app.update(.05, direction)
        self.assertEqual(self.app.world.ball.vx, 0.)
        self.assertEqual(self.app.world.ball.x, 96.)

    def test_failure_discards_buffer_and_preserves_records(self):
        app = self.app
        app.progress.complete(0, None, 5.)
        app.start(1)
        self.key(pygame.K_RIGHT)
        app.world.ball.y = 900.
        app.update(.05, 1)
        self.assertFalse(app.held_keys)
        self.assertEqual(app.world.time, 0.)
        self.assertEqual(app.progress.unlocked, 1)
        self.assertNotIn('1', Progress(self.path).records)
        app.update(1/60, 0)
        self.assertGreater(app.world.time, 0.)

    def test_result_saves_and_next_stage_button(self):
        app = self.app
        app.start(1)
        app.world.cleared = True
        app.world.time = 8.
        app.world.collected = {0, 1, 2}
        app.update(.02, 0)
        app.render()
        self.assertEqual(app.scene, 'result')
        self.assertEqual(Progress(self.path).records['1']['score'], 10)
        self.click(930, 540)
        self.assertEqual(app.scene, 'play')
        self.assertEqual(app.world.level.index, 2)

    def test_help_resize_and_letterboxed_mouse_coordinates(self):
        app = self.app
        self.click(1210, 60)
        self.assertTrue(app.help)
        self.click(1100, 665)
        self.assertFalse(app.help)
        app.window = pygame.display.set_mode((800, 700), pygame.RESIZABLE)
        app.render()
        self.assertAlmostEqual(app.scale, .5)
        self.assertEqual(app.offset, (0, 142))
        self.click(200, 350)
        self.assertEqual(app.scene, 'play')
        self.assertEqual(app.canvas.get_size(), (1600, 832))

    def test_final_stage_ending_screen(self):
        app = self.app
        app.start(7)
        app.world.cleared = True
        app.update(.02, 0)
        app.render()
        self.click(930, 540)
        self.assertEqual(app.scene, 'ending')

    def test_focus_loss_pauses_and_clears_keys(self):
        app = self.app
        app.start(0)
        self.key(pygame.K_RIGHT)
        pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        app.process_events()
        self.assertTrue(app.world.paused)
        self.assertFalse(app.held_keys)

    def test_update_at_30_60_120_fps_and_slow_frame(self):
        app = self.app
        states = []
        for fps in (30, 60, 120):
            app.start(6)
            for _ in range(fps):
                app.update(1/fps, 0)
            world = app.world
            states.append((world.time, world.ball.y, world.ball.vy))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])
        app.start(6)
        app.update(.5, 0)
        self.assertAlmostEqual(app.world.time, .5)


if __name__ == '__main__':
    unittest.main()
