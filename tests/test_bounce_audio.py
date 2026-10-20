"""Use real collision events to play the supplied bounce clip once per rebound."""
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
from pingu_ball.engine import Level,Platform,World
from pingu_ball.specials import Device


class BounceAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = App(Path(cls.temp.name)/'progress.json',headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.temp.cleanup()

    def setUp(self):
        self.app.start(0)
        self.app.sound_on = True
        pygame.mixer.stop()

    def fixture(self,kind='normal',specials=(),blocks=None):
        if blocks is None:
            blocks=[Platform('floor',0.,640.,1600.,kind=kind)]
        level=Level(0,'Audio test','',blocks,[],[],(160.,625.),(1500.,700.,64.,64.),specials=list(specials))
        self.app.world=World(level)
        self.app.world.ball.vy=100.

    def land_again(self):
        self.app.world.ball.x,self.app.world.ball.y=160.,625.
        self.app.world.ball.vx,self.app.world.ball.vy=0.,100.
        self.app.update(.02,0)

    def test_supplied_clip_is_loaded_for_bounce(self):
        expected=pygame.mixer.Sound(str(ROOT/'assets'/'audio'/'bounce_seal.mp3'))
        self.assertEqual(self.app.sounds['bounce'].get_raw(),expected.get_raw())
        self.assertGreater(expected.get_length(),.1)

    def test_normal_and_spring_contacts_play_once_without_replaying_in_air(self):
        for kind in ('normal','spring'):
            with self.subTest(kind=kind):
                self.fixture(kind)
                with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
                    self.app.update(.02,0)
                    self.app.update(.1,0)
                    playback.assert_called_once_with('bounce')
                    self.assertLess(self.app.world.ball.vy,0)

    def test_muted_contact_still_bounces_without_playback(self):
        self.fixture()
        self.app.toggle_sound()
        self.app.update(.02,0)
        self.assertEqual(self.app.world.last_landing,'floor')
        self.assertLess(self.app.world.ball.vy,0)
        self.assertEqual(self.app.sounds['bounce'].get_num_channels(),0)

    def test_repeated_contacts_keep_only_one_copy_of_the_clip(self):
        self.fixture()
        self.land_again(); self.land_again()
        self.assertEqual(self.app.sounds['bounce'].get_num_channels(),1)

    def test_failure_stops_bounce_clip_and_plays_failure_voice(self):
        self.fixture(); self.land_again()
        self.app.world.ball.y=900.
        self.app.update(.02,0)
        self.assertEqual(self.app.world.failures,1)
        self.assertEqual(self.app.sounds['bounce'].get_num_channels(),0)
        self.assertEqual(self.app.sounds['fail'].get_num_channels(),1)

    def test_wall_side_contact_and_free_fall_do_not_make_bounce_sounds(self):
        self.fixture(blocks=[Platform('wall',256.,100.,64.,500.)])
        self.app.world.ball.x,self.app.world.ball.y=240.,300.
        with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
            self.app.update(.04,1)
            playback.assert_not_called()
        self.fixture(blocks=[])
        self.app.world.ball.y=200.
        with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
            self.app.update(.05,0)
            playback.assert_not_called()

    def test_launchers_are_not_rebound_events(self):
        for type_id in (402,405,406):
            with self.subTest(type_id=type_id):
                self.fixture(blocks=[],specials=[Device('launcher',type_id,128.,640.)])
                with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
                    self.app.update(.02,0)
                    self.assertTrue(self.app.world.ball.launch)
                    playback.assert_not_called()

    def test_enemy_stomp_and_blue_laser_rebound_make_bounce_sound(self):
        for type_id in (606,502):
            with self.subTest(type_id=type_id):
                options={'color':'blue'} if type_id==502 else {}
                self.fixture(blocks=[],specials=[Device('obstacle',type_id,256.,400.,options=options)])
                self.app.world.ball.x=288. if type_id==606 else 350.
                self.app.world.ball.y=370. if type_id==606 else 400.
                self.app.world.ball.vy=300.
                with patch.object(self.app,'sound',wraps=self.app.sound) as playback:
                    self.app.update(.08,0)
                    playback.assert_called_once_with('bounce')
                    self.assertEqual(self.app.world.failures,0)


if __name__=='__main__': unittest.main()
