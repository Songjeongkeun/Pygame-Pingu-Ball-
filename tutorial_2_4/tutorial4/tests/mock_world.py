import sys
import types
from unittest.mock import patch

class Platform:
    def __init__(self,ident,x,y):
        self.id=ident;self.x=x;self.y=y;self.w=64.;self.h=64.
    def rect_at(self,t,p):
        return self.x,self.y,self.w,self.h

class Lift(Platform):
    def rect_at(self,t,p):
        return self.x, 384. if t<1 else 300., self.w,self.h

class FakeWorld:
    def __init__(self,level,auto_reset=False):
        self.level=level
        self.platforms=level.platforms
        self.physics=types.SimpleNamespace(dt=1/120,radius=14.,max_speed=213.33,bounce_speed=427.)
        self.lift=Lift('lift',384.,256.)
        self.special=types.SimpleNamespace(platforms={'lift':self.lift})
        self.reset()
    def reset(self):
        self.ball=types.SimpleNamespace(x=96.,y=338.,vx=0.,vy=0.)
        self.time=0.;self.cleared=False;self.failed=False;self.last_failure=''
        self.events=[]
        self.script={}
        self.tick=0
    def step(self,direction):
        self.tick+=1
        self.time+=1/120
        self.ball.x+=direction*1.5
        self.ball.vx=direction*180
        self.events=[]
        for ev in self.script.get(self.tick,[]):
            self.events.append(ev)
            if ev[0]=='clear':self.cleared=True
            if ev[0]=='fail':self.failed=True;self.last_failure='fake fall'


class FakeLevel:
    spawn=(96.,338.)
    platforms=tuple([Platform(f'start-{i}',64+64*i,448.) for i in range(4)]+[
        Platform(f'finish-{i}',512+64*i,256.) for i in range(8)])
    goal=(960.,192.,64.,64.)


def mocked_engine():
    original = types.ModuleType('pingu_ball')
    engine = types.ModuleType('pingu_ball.engine')
    engine.World = FakeWorld
    labs = types.ModuleType('pingu_ball.special_lab')
    labs.make_lab=lambda stage: FakeLevel() if stage==4 else None
    play=types.ModuleType('pingu_ball.playfield')
    play.expand_playfield=lambda l:l
    return patch.dict(sys.modules,{'pingu_ball':original,'pingu_ball.engine':engine,
                                  'pingu_ball.special_lab':labs,'pingu_ball.playfield':play})

