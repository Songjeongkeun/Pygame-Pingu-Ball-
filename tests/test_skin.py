"""Reference artwork and state changes in terrain and special-device rendering."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import unittest
import pygame
from pingu_ball.engine import Level,Platform,World
from pingu_ball.skin import WinterSkin
from pingu_ball.specials import Device
from pingu_ball.special_draw import draw_runtime

ROOT=Path(__file__).resolve().parent.parent


class SkinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init();pygame.display.set_mode((1,1))
        cls.skin=WinterSkin(ROOT)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def assert_reference(self,image,filename):
        source=pygame.image.load(str(ROOT/'assets/assets_v2/winter_cartoon'/filename)).convert_alpha()
        alpha=128 if filename in ('block_basic.png','block_cracked.png') else 1
        expected=pygame.transform.smoothscale(source.subsurface(source.get_bounding_rect(min_alpha=alpha)),image.get_size())
        self.assertEqual(image.get_alpha(),255)
        self.assertEqual(pygame.image.tobytes(image,'RGBA'),pygame.image.tobytes(expected,'RGBA'))

    def world(self,device):
        return World(Level(1,'art','',[],[],[],(32.,32.),(1500.,700.,64.,64.),specials=[device]),auto_reset=False)

    def test_reference_sprites_match_for_terrain_and_special_devices(self):
        cases=[('normal',None,'block_basic.png',0,{}),
               ('breakable',301,'block_cracked.png',0,{}),
               ('spring',401,'block_jump.png',0,{}),
               ('timed',302,'block_timed_active.png',0,{}),
               ('timed',302,'block_timed_warning.png',1.6,{}),
               ('timed',302,'block_timed_inactive.png',2.1,{}),
               ('linked',416,'block_linked_active.png',0,{'bridge':True}),
               ('linked',416,'block_linked_inactive.png',0,{'bridge':False}),
               ('switch',415,'block_switch_idle.png',0,{}),
               ('switch',415,'block_switch_pressed.png',0,{},False)]
        for kind,code,filename,time,switches,*armed in cases:
            with self.subTest(filename=filename):
                terrain=Platform('tile',320.,320.,kind=kind,group='bridge',armed=not armed)
                self.assert_reference(self.skin.platform(terrain,time,switches),filename)
                if code is not None:
                    w=self.world(Device('device',code,320.,320.,group='bridge'))
                    block=w.special.platforms['device'];block.armed=not armed
                    self.assert_reference(self.skin.platform(block,time,switches),filename)

    def test_fish_and_bone_keep_the_supplied_artwork_aspect_ratio(self):
        for bone,filename in [(False,'item_fish.png'),(True,'item_fish_bone.png')]:
            source=pygame.image.load(str(ROOT/'assets/assets_v2/winter_cartoon'/filename)).convert_alpha()
            source=source.subsurface(source.get_bounding_rect(min_alpha=1))
            expected_size=(32,round(32*source.get_height()/source.get_width()))
            image=self.skin.collectible(bone)
            self.assertEqual(image.get_size(),expected_size)
            self.assertLess(image.get_height(),image.get_width())
            self.assert_reference(image,filename)

    def test_moving_platform_has_the_same_two_arrow_heads_in_both_directions(self):
        expected=self.skin.image('moving')
        for direction in (-1,1):
            terrain=Platform('tile',320.,320.,kind='moving',travel=128.,arrow=direction)
            w=self.world(Device('device',501,320.,320.,options={'direction':direction}))
            for time in (0.,1.,3.):
                for block in (terrain,w.special.platforms['device']):
                    self.assertEqual(pygame.image.tobytes(self.skin.platform(block,time,{}),'RGBA'),
                                     pygame.image.tobytes(expected,'RGBA'))
        for x in (10,54):
            self.assertNotEqual(expected.get_at((x,37)),self.skin.image('normal').get_at((x,37)))

    def test_basic_and_breakable_visible_bodies_both_fill_64_by_64_pixels(self):
        w=self.world(Device('device',301,320.,320.))
        normal=Platform('basic',0.,0.,w=64.,h=64.)
        for block in (normal,w.special.platforms['device']):
            with self.subTest(kind=block.kind):
                self.assertEqual((block.w,block.h),(64.,64.))
                image=self.skin.platform(block,0,{})
                self.assertEqual(image.get_size(),(64,64))
                self.assertEqual(image.get_bounding_rect(min_alpha=128),pygame.Rect(0,0,64,64))

    def test_launcher_art_stays_directional(self):
        for direction,name in [(-1,'left'),(1,'right')]:
            terrain=Platform('tile',320.,320.,kind='launcher',arrow=direction)
            w=self.world(Device('device',402,320.,320.,options={'direction':direction}))
            for block in (terrain,w.special.platforms['device']):
                self.assert_reference(self.skin.platform(block,0,{}),f'block_launcher_{name}.png')

    def test_inactive_linked_ghost_is_visible_but_has_no_collision(self):
        w=self.world(Device('device',416,320.,320.,group='bridge'))
        block=w.special.platforms['device']
        self.assertFalse(block.active(0,w.switches))
        ghost=self.skin.platform(block,0,w.switches)
        self.assert_reference(ghost,'block_linked_inactive.png')
        self.assertGreater(ghost.get_at((32,32)).a,0)

    def test_special_spikes_use_reference_art_instead_of_flat_triangles(self):
        w=self.world(Device('spike',201,0.,0.))
        surface=pygame.Surface((64,64),pygame.SRCALPHA)
        draw_runtime(surface,w,self.skin)
        self.assert_reference(surface,'block_spikes.png')

    def test_small_spike_art_stays_at_the_bottom_of_its_collision_tile(self):
        w=self.world(Device('spike',202,0.,0.))
        surface=pygame.Surface((64,64),pygame.SRCALPHA)
        draw_runtime(surface,w,self.skin)
        self.assertEqual(surface.get_bounding_rect(),pygame.Rect(24,48,16,16))

    def test_spike_foot_overlaps_support_snow_cap_to_close_the_seam(self):
        w=self.world(Device('spike',201,0.,0.))
        w.platforms.append(Platform('base',0.,64.))
        surface=pygame.Surface((64,86),pygame.SRCALPHA)
        draw_runtime(surface,w,self.skin)
        self.assert_reference(surface.subsurface((0,22,64,64)),'block_spikes.png')
        self.assertGreater(surface.get_at((32,66)).a,0)


if __name__=='__main__':unittest.main()
