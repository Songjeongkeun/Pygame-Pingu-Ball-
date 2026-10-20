"""Replay stage 3 from its real spawn using only movement inputs."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.levels import make_level, waypoint_direction


class Replay:
    def __init__(self, level=None):
        self.world=World(level or make_level(3),auto_reset=False)
        self.inputs=[]
        self.milestones=[]

    def tick(self,direction=0):
        if self.world.failed or self.world.cleared: return
        self.inputs.append(direction)
        self.world.step(direction)

    def until(self,predicate,target=None,seconds=20,label=''):
        w=self.world
        for _ in range(round(seconds/w.physics.dt)):
            if predicate():
                self.milestones.append({'label':label,'time':round(w.time,4),'ball':[round(w.ball.x,2),round(w.ball.y,2)]})
                return
            if w.failed or w.cleared: break
            x=target() if callable(target) else target
            self.tick(waypoint_direction(w,x) if x is not None else 0)
        raise RuntimeError(f'{label}: t={w.time:.3f}, ball=({w.ball.x:.2f},{w.ball.y:.2f}), landing={w.last_landing}, failure={w.last_failure}')

    def land(self,prefix,target):
        w=self.world;since=w.time
        self.until(lambda:w.last_landing_time>since and w.time-w.last_landing_time<w.physics.dt*1.1
                   and (w.last_landing or '').startswith(prefix) and abs(w.ball.x-target)<2,
                   target,label=prefix)


def route(level=None):
    r=Replay(level);w=r.world
    r.land('normal-5',416)
    r.land('normal-10',608)
    r.land('normal-12',736)
    ferry=[q for q in w.platforms if q.kind=='moving']
    flight=(w.physics.bounce_speed+math.sqrt(w.physics.bounce_speed**2+2*w.physics.gravity*192))/w.physics.gravity
    board_x=792
    def can_board():
        if w.time-w.last_landing_time>w.physics.dt*1.1 or w.last_landing!='normal-12': return False
        rects=[q.rect_at(w.time+flight,w.physics) for q in ferry]
        left=min(q[0] for q in rects);right=max(q[0]+q[2] for q in rects)
        return left+20<=board_x<=right-20
    r.until(can_board,736,seconds=35,label='wait-for-ferry')
    r.until(lambda:(w.last_landing or '').startswith('ferry'),board_x,label='board-ferry')
    # Exit on the safe left side of the spike staircase before steering underneath.
    r.until(lambda:abs(w.ball.x-686)<1 and abs(w.ball.vx)<8,686,label='safe-drop-position')
    r.until(lambda:w.ball.y>720,686,label='below-spike-staircase')
    r.land('normal-46',736)
    r.land('normal-49',950)
    for prefix,x in (('spring-45',1120),('spring-44',1248),('spring-43',1376)):
        r.land(prefix,x)
    since=w.last_landing_time
    r.until(lambda:w.time-since>=.37,None,label='clear-goal-side')
    r.until(lambda:w.cleared,w.level.goal[0]+32,label='goal')
    return r


def main():
    parser=argparse.ArgumentParser()
    parser.parse_args()
    try:
        replay=route();w=replay.world
        report={'stage':3,'cleared':w.cleared,'failures':w.failures,'time':round(w.time,4),
                'from_spawn':True,'milestones':replay.milestones}
        folder=ROOT/'output/stage3-validation';folder.mkdir(parents=True,exist_ok=True)
        (folder/'inputs.json').write_text(json.dumps(replay.inputs),encoding='utf-8')
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True))
        return 0
    except RuntimeError as error:
        print(str(error).encode('ascii','backslashreplace').decode())
        return 1


if __name__=='__main__': raise SystemExit(main())
