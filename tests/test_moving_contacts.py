"""Moving faces and rounded corners must not turn ordinary contact into death."""
import unittest
from pingu_ball.engine import World, Level, Collectible, PowerUp
from pingu_ball.specials import Device


class MovingContactTests(unittest.TestCase):
    def world(self,type_id,phase,side,offset):
        level=Level(1,'Contact','',[],[Collectible(100,100)],[],(100,100),
                    (1400,100,64,64),powerups=[PowerUp(100,100,'up')],
                    specials=[Device('mover',type_id,400,320,
                                     options={'travel':64,'phase':phase,'speed':96})])
        world=World(level)
        x,y,w,_=world.platforms[0].rect_at(0,world.physics)
        world.ball.x=x-world.physics.radius-.001 if side<0 else x+w+world.physics.radius+.001
        world.ball.y=y+offset
        world.ball.collect_power_up('up')
        world.collected={0};world.collected_powerups={0}
        return world

    def test_contact_at_faces_corners_and_turnarounds_preserves_attempt(self):
        for type_id in (501,508):
            for phase in (0.,63.9,64.,127.9):
                for side in (-1,1):
                    for offset in (-20.,0.,32.,64.,84.):
                        with self.subTest(type_id=type_id,phase=phase,side=side,offset=offset):
                            world=self.world(type_id,phase,side,offset)
                            for _ in range(60):
                                world.step(0)
                                self.assertEqual(world.failures,0,world.last_failure)
                            self.assertAlmostEqual(world.time,.5,places=9)
                            self.assertEqual(world.collected,{0})
                            self.assertEqual(world.collected_powerups,{0})
                            self.assertEqual(world.ball.power_up,'up')

    def test_horizontal_corner_push_preserves_gravity_without_false_landing(self):
        for phase,side in ((0.,1),(64.,-1)):
            with self.subTest(phase=phase,side=side):
                world=self.world(501,phase,side,-20.)
                initial_y=world.ball.y
                for _ in range(36):world.step(0)
                self.assertEqual(world.failures,0)
                self.assertIsNone(world.last_landing)
                self.assertAlmostEqual(world.ball.y,initial_y+.5*world.physics.gravity*.3**2,places=5)
                self.assertAlmostEqual(world.ball.vy,world.physics.gravity*.3,places=5)


if __name__=='__main__':unittest.main()
