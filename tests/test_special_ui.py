"""Integration tests for the ten document tutorials and their replacement UI."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch
import pygame
from pingu_ball.app import App, main
from pingu_ball.special_lab import LAB_TYPES, make_lab, resolve_lab_type
from pingu_ball.levels import waypoint_direction
from tools.verify_tutorials import route


class SpecialInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'progress.json'
        self.app=App(self.path,headless=True)

    def tearDown(self):
        pygame.quit()
        self.temp.cleanup()

    def test_ten_document_rooms_render_and_advance(self):
        for number,code in enumerate(LAB_TYPES,1):
            with self.subTest(number=number):
                self.app.start_special(number)
                self.app.update(.2,0)
                self.app.render()
                self.assertEqual(self.app.lab_id,code)
                self.assertTrue(self.app.world.level.title.startswith(f'{number} '))
                self.assertGreater(self.app.world.time,0)
                self.assertEqual(self.app.world.failures,0)
                self.assertTrue(Path(self.app.world.level.preview).is_file())

    def test_clear_records_new_tutorial_without_overwriting_regular_records(self):
        app=self.app
        app.start_special(2)
        before=dict(app.progress.records)
        app.world.cleared=True
        app.update(1/60,0)
        app.render()
        self.assertEqual(app.scene,'result')
        self.assertEqual(app.progress.records,before)
        self.assertEqual(set(app.progress.tutorial_records),{'2'})
        app.next_stage()
        self.assertEqual(app.lab_id,LAB_TYPES[2])
        app.start_special(10)
        app.next_stage()
        self.assertEqual(app.scene,'lab_menu')

    def test_menu_has_exactly_ten_enabled_cards_without_disabled_rooms(self):
        app=self.app
        app.show_lab_menu()
        with patch.object(app,'text',wraps=app.text) as rendered:
            app.render()
        numbers=[int(call.args[0]) for call in rendered.call_args_list
                 if isinstance(call.args[0],str) and call.args[0].isdigit()]
        self.assertEqual(numbers,list(range(1,11)))
        self.assertEqual(len(app.buttons),11)
        for i in range(10):
            callbacks=[cb for rect,cb in app.buttons if rect.collidepoint(200+i%4*372,240+i//4*164)]
            self.assertEqual(len(callbacks),1)

    def test_old_rooms_and_image_map_cli_are_removed(self):
        for code in (11,37,403,416,606):
            with self.assertRaises(ValueError): resolve_lab_type(code)
        for args in (['--special','37'],['--special','416'],['--stage','0'],
                     ['--image-map','1'],['--image-maps']):
            with self.subTest(args=args),patch('sys.argv',['main.py','--headless',*args]),patch('sys.stderr',StringIO()):
                with self.assertRaises(SystemExit) as error: main()
                self.assertEqual(error.exception.code,2)
        self.assertFalse(hasattr(self.app,'start_collection'))
        self.assertFalse(Path('assets/maps/collection_layouts.json').exists())

    def test_navigation_between_new_tutorials_and_regular_stages(self):
        app=self.app
        app.start_special(9)
        app.menu()
        self.assertEqual(app.scene,'lab_menu')
        self.assertIsNone(app.lab_id)
        app.menu()
        self.assertEqual(app.scene,'menu')
        app.start(1)
        app.menu()
        self.assertEqual(app.scene,'normal_menu')

    def test_breakable_contact_removes_sprite_and_restart_restores_it(self):
        app=self.app;app.start_special(1)
        app.render()
        point=(352,416)
        self.assertGreater(app.map_canvas.get_at(point).a,0)
        for _ in range(600):
            app.update(1/120,waypoint_direction(app.world,352))
            if app.world.special.platforms['break-0'].destroyed: break
        app.render()
        self.assertTrue(app.world.special.platforms['break-0'].destroyed)
        self.assertEqual(app.map_canvas.get_at(point).a,0)
        app.restart();app.render()
        self.assertGreater(app.map_canvas.get_at(point).a,0)

    def test_all_new_tutorials_have_real_input_witnesses(self):
        for number in range(1,11):
            with self.subTest(number=number):
                replay=route(number)
                self.assertTrue(replay.world.cleared)
                self.assertEqual(replay.world.failures,0)
                self.assertTrue(replay.inputs)

    def test_document_sizes_no_fish_and_apex_spawns(self):
        for number in range(1,11):
            level=make_lab(number)
            self.assertFalse(level.collectibles)
            self.assertTrue(all((q.w,q.h)==(64,64) for q in level.platforms))
            floor=min(q.y for q in level.platforms if q.x<=level.spawn[0]<=q.x+q.w)
            self.assertAlmostEqual(level.spawn[1],floor-110)
        level=make_lab(1)
        self.assertEqual(len(level.specials),4)
        self.assertEqual(level.spawn,(96,274))


if __name__=='__main__': unittest.main()
