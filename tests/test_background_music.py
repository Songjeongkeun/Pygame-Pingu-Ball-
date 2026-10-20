"""Real music-stream playback across title, menus, mute, and game transitions."""
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


class BackgroundMusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = App(Path(cls.temp.name)/'progress.json',headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.temp.cleanup()

    def setUp(self):
        self.app.running = True
        self.app.sound_on = True
        self.app.start(0)
        pygame.mixer.stop()

    def test_supplied_music_decodes_and_title_starts_an_infinite_loop(self):
        clip = pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'title_opening.mp3'))
        self.assertGreater(clip.get_length(),1.)
        with patch.object(pygame.mixer.music,'play',wraps=pygame.mixer.music.play) as playback:
            self.app.show_title()
            playback.assert_called_once_with(loops=-1)
        self.assertTrue(pygame.mixer.music.get_busy())

    def test_rendering_and_menu_navigation_do_not_restart_music_every_frame(self):
        app = self.app
        with patch.object(pygame.mixer.music,'play',wraps=pygame.mixer.music.play) as playback:
            app.show_title()
            app.render(); app.render()
            app.menu(); app.render()
            app.show_normal_menu();app.render()
            app.show_lab_menu(); app.render()
            playback.assert_called_once_with(loops=-1)
        self.assertTrue(pygame.mixer.music.get_busy())

    def test_game_and_special_room_stop_music_and_returning_menu_restarts(self):
        app = self.app
        app.show_title(); app.start(0)
        self.assertFalse(pygame.mixer.music.get_busy())
        app.menu(); self.assertTrue(pygame.mixer.music.get_busy())
        app.start_special(401); self.assertFalse(pygame.mixer.music.get_busy())
        app.menu(); self.assertTrue(pygame.mixer.music.get_busy())

    def test_muting_stops_both_music_and_effects_and_unmute_works(self):
        app = self.app
        app.show_title(); app.sound('fail')
        self.assertTrue(pygame.mixer.get_busy())
        app.toggle_sound()
        self.assertFalse(pygame.mixer.music.get_busy())
        self.assertFalse(pygame.mixer.get_busy())
        app.render(); self.assertFalse(pygame.mixer.music.get_busy())
        app.toggle_sound(); self.assertTrue(pygame.mixer.music.get_busy())
        app.start(0); app.toggle_sound(); app.toggle_sound()
        self.assertFalse(pygame.mixer.music.get_busy())

    def test_mute_remains_in_effect_when_returning_to_title(self):
        app = self.app
        app.toggle_sound(); app.show_title()
        self.assertFalse(pygame.mixer.music.get_busy())

    def test_quit_stops_music_immediately(self):
        self.app.show_title(); self.app.quit()
        self.assertFalse(pygame.mixer.music.get_busy())

    def test_no_audio_device_is_safe(self):
        with patch.object(pygame.mixer,'get_init',return_value=None):
            self.app.sync_music()
            self.app.toggle_sound()


if __name__=='__main__': unittest.main()
