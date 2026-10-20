"""Replay ordinary inputs and item presses through the ten real tutorial maps."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.special_lab import make_lab
from pingu_ball.levels import waypoint_direction


class Witness:
    def __init__(self,number):
        self.world=World(make_lab(number),auto_reset=False)
        self.inputs=[]

    def tick(self,direction=0,use=False):
        w=self.world
        if w.failed or w.cleared: return
        self.inputs.append([direction,int(use)])
        w.step(direction,use_power=use,power_direction=direction if use else 0)

    def until(self,predicate,target=None,seconds=12):
        w=self.world
        for _ in range(round(seconds/w.physics.dt)):
            if predicate() or w.failed or w.cleared: return
            x=target() if callable(target) else target
            self.tick(waypoint_direction(w,x) if x is not None else 0)
        raise RuntimeError(f'timed out at {w.time:.2f}, {w.ball.x:.1f},{w.ball.y:.1f}, landing={w.last_landing}')

    def land(self,prefix,x):
        w=self.world;since=w.time
        self.until(lambda:w.last_landing_time>since and (w.last_landing or '').startswith(prefix),x)

    def finish(self):
        w=self.world;goal=w.level.goal
        self.until(lambda:w.cleared,goal[0]+32)


def route(number):
    r=Witness(number);w=r.world
    if number==1:
        r.land('start-3',288)
        for i,x in enumerate((352,480,608,736)): r.land(f'break-{i}',x)
        r.finish()
    elif number==2:
        for name,x in [('spring-1',480),('spring-2',800),('spring-3',1056)]: r.land(name,x)
        r.finish()
    elif number==3:
        ferry=w.special.platforms['ferry-0']
        r.until(lambda:(w.last_landing or '').startswith('ferry'),lambda:ferry.rect_at(w.time+.2)[0]+96)
        r.until(lambda:ferry.rect_at(w.time)[0]>1180,lambda:ferry.rect_at(w.time+.35)[0]+96,seconds=18)
        r.finish()
    elif number==4:
        r.land('start-3',288)
        r.until(lambda:w.last_landing=='lift',416)
        r.until(lambda:w.time-w.last_landing_time<w.physics.dt*1.1
                and w.last_landing=='lift'
                and w.special.platforms['lift'].rect_at(w.time)[1]<=310,416)
        r.finish()
    elif number==5:
        for name,x in [('start-3',224),('clock-0',416),('clock-1',608),('clock-2',800)]: r.land(name,x)
        r.finish()
    elif number==6:
        r.land('upper-0',912)
        # Drop past the safe left edge of the upper deck, then steer under it.
        r.until(lambda:w.ball.y>375,850)
        r.land('launch',1120)
        r.until(lambda:w.cleared,None)
    elif number==7:
        for name,x in [('start-3',416),('middle-1',672),('spring',992)]: r.land(name,x)
        r.finish()
    elif number==8:
        for name,x in [('start-5',480),('middle-0',672),('middle-4',928),('finish-0',1120)]: r.land(name,x)
        r.finish()
    elif number==9:
        for name,x in [('start-2',352),('patrol-a-0',544),('patrol-a-2',672),('single',864),('patrol-b-0',1056),('patrol-b-2',1184)]: r.land(name,x)
        r.finish()
    else:
        r.land('start-3',288)
        if w.ball.power_up!='up': raise RuntimeError('jump pickup not collected')
        r.tick(1,True)
        r.land('dash-deck-0',480)
        r.land('dash-deck-2',608)
        if w.ball.power_up!='forward': raise RuntimeError('dash pickup not collected')
        launch_time=w.time
        r.until(lambda:w.time-launch_time>=.35,960)
        r.tick(1,True)
        r.finish()
    return r


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=int,choices=range(1,11))
    args=parser.parse_args()
    folder=ROOT/'output/tutorial-validation';folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in ([args.stage] if args.stage else range(1,11)):
        try:
            r=route(n);w=r.world
            result={'stage':n,'cleared':w.cleared,'failed':w.failed,'time':round(w.time,3),
                    'reason':w.last_failure,'position':[round(w.ball.x,2),round(w.ball.y,2)]}
            if w.cleared:
                (folder/f'route_{n:02d}.json').write_text(json.dumps(r.inputs),encoding='utf-8')
        except RuntimeError as error:
            result={'stage':n,'cleared':False,'error':str(error)}
        results.append(result);print(json.dumps(result,ensure_ascii=True),flush=True)
    (folder/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if all(r['cleared'] for r in results) else 1


if __name__=='__main__': raise SystemExit(main())
