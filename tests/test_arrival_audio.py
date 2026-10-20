"""Arrival audio through the actual igloo collision and result transition."""
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
from pingu_ball.engine import Level,World


class ArrivalAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.app=App(Path(cls.temp.name)/'progress.json',headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.temp.cleanup()

    def setUp(self):
        self.app.start(0)
        self.app.sound_on=True
        pygame.mixer.stop()
        level=Level(0,'Arrival test','',[],[],[],(100.,400.),(120.,380.,64.,64.))
        self.app.world=World(level)

    def test_supplied_file_is_loaded_as_the_clear_sound(self):
        expected=pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'goal_arrival.mp3'))
        self.assertEqual(self.app.sounds['clear'].get_raw(),expected.get_raw())
        self.assertGreater(expected.get_length(),.1)

    def test_igloo_entry_plays_once_and_result_does_not_replay(self):
        with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
            self.app.update(.2,1)
            self.assertTrue(self.app.world.cleared)
            self.assertEqual(self.app.scene,'result')
            self.assertEqual(self.app.sounds['clear'].get_num_channels(),1)
            self.app.update(.3,1)
            self.app.render(); self.app.render()
            playback.assert_called_once_with('clear')

    def test_muted_igloo_entry_still_clears_without_audio(self):
        self.app.toggle_sound()
        self.app.update(.2,1)
        self.assertTrue(self.app.world.cleared)
        self.assertEqual(self.app.sounds['clear'].get_num_channels(),0)

    def test_arrival_stops_prior_bounce_and_failure_audio(self):
        self.app.sound('fail')
        self.app.sound('bounce')
        self.assertEqual(self.app.sounds['fail'].get_num_channels(),1)
        self.app.update(.2,1)
        self.assertEqual(self.app.sounds['clear'].get_num_channels(),1)
        self.assertEqual(self.app.sounds['fail'].get_num_channels(),0)
        self.assertEqual(self.app.sounds['bounce'].get_num_channels(),0)

    def test_manual_restart_and_free_flight_do_not_play_arrival(self):
        self.app.restart()
        with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
            self.app.update(.02,0)
            playback.assert_not_called()
        self.assertFalse(self.app.world.cleared)

    def test_later_failure_stops_old_arrival_audio(self):
        self.app.update(.2,1)
        self.app.restart()
        self.app.world.ball.y=900.
        self.app.update(.02,0)
        self.assertEqual(self.app.sounds['clear'].get_num_channels(),0)
        self.assertEqual(self.app.sounds['fail'].get_num_channels(),1)


if __name__=='__main__': unittest.main()
