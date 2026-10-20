"""Headless integration tests with actual Pygame events and surfaces."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pygame
from pingu_ball.app import App, ROOT, default_save_path
from pingu_ball.storage import Progress


class SavePathTests(unittest.TestCase):
    def test_source_build_uses_project_save_directory(self):
        with patch('pingu_ball.app.sys.frozen', False, create=True):
            self.assertEqual(default_save_path(), ROOT/'save_data'/'progress.json')

    def test_executable_uses_persistent_directory_next_to_exe(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary)/'PinguBall.exe'
            with patch('pingu_ball.app.sys.frozen', True, create=True), \
                 patch('pingu_ball.app.sys.executable', str(executable)):
                path = default_save_path()
                self.assertEqual(path, executable.resolve().parent/'save_data'/'progress.json')
                Progress(path).complete(0, None, 1.)
                self.assertEqual(Progress(default_save_path()).records['0'], {'cleared': True})


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

    def click(self, x, y, timestamp=None):
        app = self.app
        at = (round(x*app.scale+app.offset[0]), round(y*app.scale+app.offset[1]))
        data = {'button': 1, 'pos': at}
        if timestamp is not None:
            data['timestamp'] = timestamp
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, data))
        app.process_events()
        app.render()

    def key(self, key, pressed=True):
        event = pygame.KEYDOWN if pressed else pygame.KEYUP
        pygame.event.post(pygame.event.Event(event, {'key': key}))
        self.app.process_events()

    def test_locked_cards_and_automatic_tutorial_start(self):
        self.click(*self.app.title_start.center)
        self.click(1150,550)
        self.assertEqual(self.app.scene,'normal_menu')
        self.click(572, 280)
        self.assertEqual(self.app.scene,'normal_menu')
        self.key(pygame.K_ESCAPE);self.app.render()
        self.click(400,550)
        self.assertEqual(self.app.scene,'lab_menu')
        self.click(1100,70)
        self.assertEqual(self.app.scene,'lab_menu')
        self.click(200,240)
        self.assertEqual(self.app.scene, 'play')
        self.assertEqual(self.app.world.level.index, 0)
        initial_y=self.app.world.ball.y
        self.app.update(1/60, 0)
        self.assertGreater(self.app.world.time, 0)
        self.assertGreater(self.app.world.ball.y, initial_y)

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
        app.world.ball.y = app.world.physics.height+20.
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
        app.update(.02, 0)
        app.render()
        self.assertEqual(app.scene, 'result')
        self.assertEqual(Progress(self.path).records['1']['score'], 7)
        self.click(930, 540)
        self.assertEqual(app.scene, 'play')
        self.assertEqual(app.world.level.index, 2)

    def test_help_resize_and_letterboxed_mouse_coordinates(self):
        app = self.app
        self.click(*app.title_start.center)
        self.click(1210, 60)
        self.assertTrue(app.help)
        self.click(1100, 705)
        self.assertFalse(app.help)
        app.window = pygame.display.set_mode((800, 700), pygame.RESIZABLE)
        app.render()
        self.assertAlmostEqual(app.scale, .5)
        self.assertEqual(app.offset, (0, 142))
        self.click(400,550)
        self.click(1100,70)
        self.assertEqual(app.scene,'lab_menu')
        self.click(200,240)
        self.assertEqual(app.scene, 'play')
        self.assertEqual(app.canvas.get_size(), (1600, 832))

    def test_normal_and_tutorial_lists_keep_their_own_return_paths(self):
        app=self.app
        self.click(*app.title_start.center)
        self.click(1150,550)
        app.progress.unlocked=12;app.render()
        # All twelve document maps are separate from the tutorial list.
        for index in range(1,13):
            self.click(200+(index-1)%4*372,280+(index-1)//4*160)
            self.assertEqual(app.world.level.index,index)
            app.menu();app.render();self.assertEqual(app.scene,'normal_menu')
        self.click(200,705);self.assertEqual(app.scene,'normal_menu')
        self.key(pygame.K_ESCAPE);app.render();self.assertEqual(app.scene,'menu')
        self.click(400,550);self.assertEqual(app.scene,'lab_menu')
        self.click(200,240);self.assertIsNotNone(app.lab_id)
        app.menu();app.render();self.assertEqual(app.scene,'lab_menu')
        self.click(1100,70);self.assertEqual(app.scene,'lab_menu')
        self.click(200,240);self.assertEqual(app.lab_id,301)
        app.world.cleared=True;app.update(.02,0);app.next_stage()
        self.assertEqual(app.scene,'play')
        self.assertEqual(app.lab_id,401)

    def test_title_artwork_start_button_after_resize_and_return(self):
        app = self.app
        self.assertEqual(app.scene, 'title')
        self.assertIsNone(app.world)
        self.click(200, 350)
        self.assertEqual(app.scene, 'title')
        app.window = pygame.display.set_mode((800, 700), pygame.RESIZABLE)
        app.render()
        self.click(*app.title_start.center)
        self.assertEqual(app.scene, 'menu')
        self.assertIsNone(app.world)
        self.click(1000, 60)
        self.assertEqual(app.scene, 'title')
        self.key(pygame.K_RETURN)
        app.render()
        self.assertEqual(app.scene, 'menu')
        self.key(pygame.K_ESCAPE)
        self.assertEqual(app.scene, 'title')

    def test_pickup_changes_sprite_and_double_click_uses_clicked_side(self):
        app = self.app
        for sign in (-1, 1):
            app.start(7)
            item = app.world.level.powerups[0]
            app.world.ball.x, app.world.ball.y = item.x, item.y
            app.update(1/120, 0)
            app.window = pygame.display.set_mode((800, 700), pygame.RESIZABLE)
            app.render()
            self.assertEqual(app.world.ball.power_up, 'forward')
            b = app.world.ball
            rect = pygame.Rect(round(b.x-14), round(b.y-14), 28, 28)
            self.assertEqual(app.map_canvas.subsurface(rect).get_at((14, 14)),
                             app.skin.image('ball_forward', (28, 28)).get_at((14, 14)))
            x = app.map_view.left+30 if sign < 0 else app.map_view.right-30
            self.click(x, 300, 100)
            self.assertIsNone(app.pending_power_direction)
            self.click(x, 300, 220)
            app.update(1/120, 0)
            app.render()
            self.assertIsNone(app.world.ball.power_up)
            self.assertGreater(app.world.ball.vx*sign, 0.)
            b = app.world.ball
            rect = pygame.Rect(round(b.x-14), round(b.y-14), 28, 28)
            self.assertEqual(app.map_canvas.subsurface(rect).get_at((14, 14)),
                             app.skin.image('ball', (28, 28)).get_at((14, 14)))

    def test_slow_or_distant_clicks_and_pause_do_not_use_charge(self):
        app = self.app
        app.start(1)
        app.world.ball.power_up = 'up'
        app.render()
        self.click(800, 300, 100)
        self.click(800, 300, 500)
        self.click(1100, 300, 600)
        self.assertIsNone(app.pending_power_direction)
        self.click(1080, 32, 650)
        self.assertTrue(app.world.paused)
        self.assertIsNone(app.last_tap)
        self.click(800, 440, 700)
        self.assertFalse(app.world.paused)
        self.click(800, 300, 750)
        self.assertIsNone(app.pending_power_direction)
        app.restart()
        self.assertIsNone(app.last_tap)
        self.assertIsNone(app.world.ball.power_up)

    def test_double_touch_ignores_emulated_mouse_and_second_finger(self):
        app = self.app
        app.start(1)
        app.world.ball.power_up = 'up'
        app.render()
        def finger(event_type, finger_id, timestamp):
            pygame.event.post(pygame.event.Event(event_type,
                {'finger_id': finger_id, 'x': .7, 'y': .5, 'timestamp': timestamp}))
        finger(pygame.FINGERDOWN, 1, 100)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
            {'button': 1, 'pos': (700, 260), 'touch': True, 'timestamp': 100}))
        finger(pygame.FINGERDOWN, 2, 110)
        app.process_events()
        self.assertIsNone(app.pending_power_direction)
        finger(pygame.FINGERUP, 1, 150)
        finger(pygame.FINGERDOWN, 1, 220)
        finger(pygame.FINGERUP, 1, 260)
        app.process_events()
        app.update(1/120, 0)
        self.assertIsNone(app.world.ball.power_up)
        self.assertAlmostEqual(app.world.ball.vx, 0.)
        self.assertLess(app.world.ball.vy, -500.)

    def test_final_stage_ending_screen(self):
        app = self.app
        app.start(12)
        app.world.cleared = True
        app.update(.02, 0)
        app.render()
        self.click(930, 540)
        self.assertEqual(app.scene, 'ending')

    def test_all_reference_maps_fit_between_hud_and_footer(self):
        app = self.app
        self.assertEqual(set(app.previews), set(range(1, 13)))
        for stage in range(1, 13):
            app.start(stage)
            app.render()
            self.assertGreaterEqual(app.map_view.top, 64)
            self.assertLessEqual(app.map_view.bottom, 778)
            self.assertEqual(app.map_canvas.get_size(), (app.world.level.width, app.world.level.height))
            # Empty world pixels stay transparent: there is one continuous background.
            self.assertEqual(app.map_canvas.get_at((800, 0)).a, 0)
        arrow = app.skin.image('moving')
        plain = app.skin.image('normal')
        for x in (10, 54):
            self.assertNotEqual(arrow.get_at((x, 36)), plain.get_at((x, 36)))

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
