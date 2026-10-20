"""Physical arrow-key double taps after real pickups, through Pygame events."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pathlib import Path
import tempfile
import unittest
import pygame
from pingu_ball.app import App
from pingu_ball.config import PHYSICS as P


class KeyboardPowerUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = App(Path(cls.temp.name)/'progress.json',headless=True)

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        cls.temp.cleanup()

    def setUp(self):
        self.app.help = False
        self.app.start(7)
        pygame.event.clear()

    def equip(self,kind='forward'):
        world = self.app.world
        item = next(item for item in world.level.powerups if item.kind == kind)
        world.ball.x,world.ball.y = item.x,item.y
        self.app.update(P.dt,0)
        self.assertEqual(world.ball.power_up,kind)

    def key(self,key,time,pressed=True,repeat=False):
        event = pygame.KEYDOWN if pressed else pygame.KEYUP
        pygame.event.post(pygame.event.Event(event,{'key':key,'timestamp':time,'repeat':repeat}))
        self.app.process_events()

    def tap(self,key,time):
        self.key(key,time)
        self.key(key,time+20,pressed=False)

    def test_actual_pickup_then_double_left_or_right_uses_one_directional_jump(self):
        for key,sign in ((pygame.K_LEFT,-1),(pygame.K_RIGHT,1)):
            with self.subTest(sign=sign):
                self.app.start(7)
                self.equip()
                self.tap(key,100)
                self.assertIsNone(self.app.pending_power_direction)
                self.key(key,220)
                self.app.update(P.dt,sign)
                self.assertIsNone(self.app.world.ball.power_up)
                self.assertAlmostEqual(self.app.world.ball.vx,sign*P.forward_jump_speed)
                self.assertLess(self.app.world.ball.vy,0.)
                self.assertEqual(sum(kind=='power_use' for kind,_ in self.app.world.events),1)
                self.assertIsNone(self.app.last_key_tap)

    def test_up_item_uses_double_up_and_keeps_upward_jump_physics(self):
        self.equip('up')
        self.tap(pygame.K_UP,100)
        self.key(pygame.K_UP,220)
        self.app.update(P.dt,0)
        self.assertIsNone(self.app.world.ball.power_up)
        self.assertAlmostEqual(self.app.world.ball.vx,0.)
        self.assertAlmostEqual(self.app.world.ball.vy,-P.power_up_speed+P.gravity*P.dt)

    def test_real_pickups_allow_dash_then_jump_with_their_arrow_inputs(self):
        self.equip('up')
        self.equip('forward')
        self.assertEqual(self.app.world.ball.held_powerups,('forward','up'))
        self.tap(pygame.K_RIGHT,100)
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertEqual(self.app.world.ball.held_powerups,('up',))
        self.assertIsNotNone(self.app.world.ball.forward_target_x)
        self.key(pygame.K_RIGHT,240,pressed=False)
        self.tap(pygame.K_UP,400)
        self.key(pygame.K_UP,520)
        self.app.update(P.dt,0)
        self.assertEqual(self.app.world.ball.held_powerups,())
        self.assertIsNone(self.app.world.ball.forward_target_x)
        self.assertAlmostEqual(self.app.world.ball.vy,-P.power_up_speed+P.gravity*P.dt)

    def test_keys_keep_normal_movement_without_an_item(self):
        self.tap(pygame.K_RIGHT,100)
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertIsNone(self.app.pending_power_direction)
        self.assertGreater(self.app.world.ball.vx,0.)
        self.assertFalse(any(kind=='power_use' for kind,_ in self.app.world.events))

    def test_held_key_and_os_repeat_do_not_count_as_second_press(self):
        self.equip()
        self.key(pygame.K_RIGHT,100)
        self.key(pygame.K_RIGHT,150,repeat=True)
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertEqual(self.app.world.ball.power_up,'forward')
        self.assertIsNone(self.app.pending_power_direction)

    def test_slow_taps_or_alternating_keys_do_not_consume(self):
        self.equip()
        self.tap(pygame.K_LEFT,100)
        self.tap(pygame.K_LEFT,450)
        self.tap(pygame.K_RIGHT,500)
        self.tap(pygame.K_LEFT,550)
        self.app.update(P.dt,0)
        self.assertEqual(self.app.world.ball.power_up,'forward')

    def test_opposite_held_keys_do_not_trigger_jump(self):
        self.equip()
        self.key(pygame.K_LEFT,100)
        self.tap(pygame.K_RIGHT,150)
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,0)
        self.assertEqual(self.app.world.ball.power_up,'forward')
        self.assertAlmostEqual(self.app.world.ball.vx,0.)

    def test_input_before_pickup_cannot_be_reused_as_first_tap(self):
        self.tap(pygame.K_RIGHT,100)
        self.equip()
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertEqual(self.app.world.ball.power_up,'forward')

    def test_pause_focus_loss_and_restart_clear_first_tap(self):
        for transition in ('pause','focus','restart'):
            with self.subTest(transition=transition):
                self.app.start(7)
                self.equip()
                self.tap(pygame.K_RIGHT,100)
                if transition=='pause':
                    self.app.toggle_pause(); self.app.toggle_pause()
                elif transition=='focus':
                    pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
                    self.app.process_events(); self.app.toggle_pause()
                else:
                    self.app.restart(); self.equip()
                self.key(pygame.K_RIGHT,220)
                self.app.update(P.dt,1)
                self.assertEqual(self.app.world.ball.power_up,'forward')

    def test_failure_clears_first_tap_before_new_pickup(self):
        self.equip()
        self.tap(pygame.K_RIGHT,100)
        world = self.app.world
        world.ball.y = world.physics.height+world.physics.radius+1
        self.app.update(P.dt,0)
        self.assertEqual(world.failures,1)
        self.equip()
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertEqual(world.ball.power_up,'forward')

    def test_wrong_arrow_does_not_consume_other_item_type(self):
        self.equip('up')
        self.tap(pygame.K_LEFT,100); self.key(pygame.K_LEFT,220)
        self.app.update(P.dt,-1)
        self.assertEqual(self.app.world.ball.power_up,'up')

    def test_single_mouse_and_single_key_tap_do_not_form_double_tap(self):
        self.equip()
        self.app.render()
        view = self.app.map_view
        point = (round((view.centerx)*self.app.scale+self.app.offset[0]),
                 round((view.centery)*self.app.scale+self.app.offset[1]))
        self.app.pointer_down(point,100)
        self.key(pygame.K_RIGHT,220)
        self.app.update(P.dt,1)
        self.assertEqual(self.app.world.ball.power_up,'forward')


if __name__=='__main__': unittest.main()
