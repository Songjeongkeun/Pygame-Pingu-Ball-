"""Replay stage 12's upper ferry and spring from the real stage spawn."""
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.levels import make_level
from tools.verify_campaign_routes import Witness


def upper_route():
    r=Witness(12);w=r.world
    r.land('normal-41',1440)
    r.land('spring-46',1376)
    r.land('normal-41',1440)
    ferry=next(q for q in w.platforms if q.id=='ferry-30')
    start=next(q for q in w.platforms if q.id=='normal-41')
    apex=start.y-w.physics.radius-w.physics.bounce_height
    rise=apex-(ferry.y-w.physics.radius)
    speed=w.physics.power_up_speed;gravity=w.physics.gravity
    flight=(speed+math.sqrt(speed*speed-2*gravity*rise))/gravity
    board_x=1696

    def can_board():
        x,_,width,_=ferry.rect_at(w.time+w.physics.flight_time/2+flight,w.physics)
        return (w.time-w.last_landing_time<w.physics.dt*1.1
                and w.last_landing=='normal-41' and x+16<board_x<x+width-16)

    r.until(can_board,1440,seconds=30,label='wait-for-ferry')
    r.until(lambda:w.ball.vy>=-8 and w.ball.y<apex+2,1520,label='launch-apex')
    r.tick(1,True)
    r.land('ferry-',board_x)
    r.until(lambda:w.time-w.last_landing_time<w.physics.dt*1.1
            and (w.last_landing or '').startswith('ferry-') and w.ball.x>2070,
            lambda:ferry.rect_at(w.time,w.physics)[0]+70,seconds=20,label='ride-right')
    r.land('spring-32',2208)
    r.land('normal-29',2080)
    return r


def main():
    replay=upper_route();world=World(make_level(12),auto_reset=False)
    for direction,use,power_direction in replay.inputs:
        world.step(direction,use_power=use,power_direction=power_direction)
        if world.failed: break
    passed=world.last_landing=='normal-29' and world.failures==0
    report=dict(stage=12,scope='upper_ferry_spring_and_basic_platform',from_spawn=True,
                passed=passed,failures=world.failures,time=round(world.time,4),
                full_clear_verified=False,milestones=replay.milestones)
    folder=ROOT/'output/campaign-routes';folder.mkdir(parents=True,exist_ok=True)
    (folder/'inputs_12_upper.json').write_text(json.dumps(replay.inputs),encoding='utf-8')
    (folder/'stage12_upper_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
