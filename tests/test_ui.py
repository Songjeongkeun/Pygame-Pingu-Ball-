"""Headless Pygame integration tests using actual mouse/key events and surfaces."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import tempfile
import unittest
import pygame
from pingu_ball.app import App
from pingu_ball.storage import Progress


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'progress.json'
        self.app=App(self.path,headless=True)
        pygame.event.clear()
        self.app.render()

    def tearDown(self):
        pygame.quit();self.temp.cleanup()

    def click(self,x,y):
        a=self.app
        at=(round(x*a.scale+a.offset[0]),round(y*a.scale+a.offset[1]))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,{'button':1,'pos':at}))
        a.process_events();a.render()

    def test_locked_cards_and_tutorial_mouse_start(self):
        self.click(400,300)
        self.assertEqual(self.app.scene,'menu')
        self.click(100,300)
        self.assertEqual(self.app.scene,'play');self.assertEqual(self.app.world.level.index,0)
        self.assertFalse(self.app.world.started)
        self.app.update(1/60,1)
        self.assertTrue(self.app.world.started)

    def test_pause_resume_and_r_reset(self):
        a=self.app;a.start(0);a.update(.05,1);a.render()
        self.click(830,47);self.assertTrue(a.world.paused)
        snapshot=(a.world.time,a.world.ball.x,a.world.ball.y)
        a.update(.1,1)
        self.assertEqual(snapshot,(a.world.time,a.world.ball.x,a.world.ball.y))
        self.click(600,304);self.assertFalse(a.world.paused)
        a.update(.05,1)
        self.assertGreater(a.world.time,snapshot[0])
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_r}))
        a.process_events()
        self.assertEqual(a.world.time,0.);self.assertFalse(a.world.started)

    def test_result_saves_and_next_stage_button(self):
        a=self.app;a.start(1);a.world.cleared=True;a.world.started=True;a.world.time=8.;a.world.collected={0,1,2}
        a.update(.02,0);a.render()
        self.assertEqual(a.scene,'result')
        self.assertEqual(Progress(self.path).records['1']['score'],10)
        self.click(700,390)
        self.assertEqual(a.scene,'play');self.assertEqual(a.world.level.index,2)

    def test_help_and_resize_keep_mouse_coordinates_correct(self):
        a=self.app
        self.click(980,54);self.assertTrue(a.help)
        self.click(800,480);self.assertFalse(a.help)
        a.window=pygame.display.set_mode((900,700),pygame.RESIZABLE);a.render()
        self.assertAlmostEqual(a.scale,.75)
        self.click(100,300)
        self.assertEqual(a.scene,'play')
        self.assertEqual(a.canvas.get_size(),(1200,600))

    def test_final_stage_has_ending_screen(self):
        a=self.app;a.start(7);a.world.cleared=True;a.world.started=True
        a.update(.02,0);a.render()
        self.click(700,390)
        self.assertEqual(a.scene,'ending')


if __name__=='__main__':unittest.main()
