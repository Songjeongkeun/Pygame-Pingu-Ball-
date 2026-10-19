"""Decode the supplied clip and exercise playback through actual failure events."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pygame
from pingu_ball.app import App, ROOT


class FailureAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = App(Path(cls.temp.name)/'progress.json', headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.temp.cleanup()

    def setUp(self):
        pygame.mixer.stop()
        self.app.sound_on = True
        self.app.start(0)

    def fail_by_falling(self):
        world = self.app.world
        world.ball.y = world.physics.height+world.physics.radius+1
        self.app.update(1/60, 0)

    def test_supplied_mp3_is_the_loaded_failure_sound(self):
        expected = pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'fail_noot.mp3'))
        actual = self.app.sounds['fail']
        self.assertEqual(actual.get_raw(), expected.get_raw())
        self.assertGreater(actual.get_length(), .5)

    def test_one_failure_plays_the_clip_once_and_restart_does_not_replay(self):
        with patch.object(self.app, 'sound', wraps=self.app.sound) as playback:
            self.fail_by_falling()
            self.assertEqual(self.app.world.failures, 1)
            playback.assert_called_once_with('fail')
            self.assertEqual(self.app.sounds['fail'].get_num_channels(), 1)
            self.app.update(1/60, 0)
            self.app.restart()
            playback.assert_called_once_with('fail')

    def test_muting_suppresses_failure_voice_and_stops_current_playback(self):
        self.fail_by_falling()
        self.assertTrue(pygame.mixer.get_busy())
        self.app.toggle_sound()
        self.assertFalse(pygame.mixer.get_busy())
        self.fail_by_falling()
        self.assertEqual(self.app.sounds['fail'].get_num_channels(), 0)

    def test_rapid_failures_do_not_overlap_multiple_copies(self):
        self.fail_by_falling()
        self.fail_by_falling()
        self.assertEqual(self.app.world.failures, 2)
        self.assertEqual(self.app.sounds['fail'].get_num_channels(), 1)


if __name__ == '__main__':
    unittest.main()
