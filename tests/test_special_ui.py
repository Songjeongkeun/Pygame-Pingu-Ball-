"""Real Pygame integration for device rooms, menu access, and save isolation."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import tempfile
import unittest
import pygame
from pingu_ball.app import App
from pingu_ball.specials import CATALOG


class SpecialInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'progress.json'
        cls.app=App(cls.path,headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit(); cls.temp.cleanup()

    def test_all_37_rooms_render_and_advance_actual_engine(self):
        for code in CATALOG:
            with self.subTest(code=code):
                self.app.start_special(code)
                self.app.update(.2,0)
                self.app.render()
                self.assertEqual(self.app.lab_id,code)
                self.assertTrue(self.app.world.special.devices)
                self.assertGreater(self.app.world.time,0)

    def test_laboratory_results_do_not_write_tutorial_progress(self):
        app=self.app
        before=dict(app.progress.records)
        app.start_special(401)
        app.world.cleared=True
        app.world.events=[]
        app.update(1/60,0)
        app.render()
        self.assertEqual(app.scene,'result')
        self.assertEqual(app.progress.records,before)
        self.assertFalse(self.path.exists())
        app.next_stage()
        self.assertIsNotNone(app.lab_id)

    def test_lab_back_navigation_and_existing_stage_start(self):
        app=self.app
        app.start_special(503)
        app.menu(); app.render()
        self.assertEqual(app.scene,'lab_menu')
        self.assertIsNone(app.lab_id)
        app.menu(); self.assertEqual(app.scene,'menu')
        app.start(0); self.assertIsNone(app.lab_id)

    def test_catalog_all_pages_are_bounded_and_clickable(self):
        app=self.app
        for page in range(4):
            app.lab_page=page; app.show_lab_menu(); app.render()
            expected=min(12,37-page*12)
            self.assertEqual(len(app.buttons),expected+3)
            self.assertEqual(app.lab_page,page)
        app.lab_page=999; app.render(); self.assertEqual(app.lab_page,3)


if __name__=='__main__': unittest.main()
