"""Replay stage 8's left ferry and both spring landings, not a full clear."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.levels import make_level
from tools.verify_campaign_routes import Witness


def left_route():
    r=Witness(8);w=r.world
    r.land('normal-9',1184)
    r.land('normal-7',1056)
    r.land('normal-6',864)
    ferry=next(q for q in w.platforms if q.kind=='moving')
    def ready_to_board():
        x,_,width,_=ferry.rect_at(w.time+w.physics.flight_time,w.physics)
        return (w.time-w.last_landing_time<w.physics.dt*1.1 and w.last_landing=='normal-6'
                and x+10<736<x+width-10)
    r.until(ready_to_board,864,seconds=30,label='wait-for-ferry')
    r.until(lambda:w.last_landing=='ferry-3',736,label='board-ferry')
    r.until(lambda:w.time-w.last_landing_time<w.physics.dt*1.1
            and w.last_landing=='ferry-3' and w.ball.x<340,
            lambda:ferry.rect_at(w.time,w.physics)[0]+32,seconds=20,label='ride-left')
    r.land('normal-2',160)
    r.land('spring-4',120)
    r.land('spring-38',272)
    return r


def main():
    replay=left_route();world=World(make_level(8),auto_reset=False)
    for direction,use,power_direction in replay.inputs:
        world.step(direction,use_power=use,power_direction=power_direction)
        if world.failed: break
    passed=world.last_landing=='spring-38' and not world.failures
    report=dict(stage=8,scope='left_ferry_and_two_springs',from_spawn=True,
                passed=passed,failures=world.failures,time=round(world.time,4),
                full_clear_verified=False,milestones=replay.milestones)
    folder=ROOT/'output/campaign-routes';folder.mkdir(parents=True,exist_ok=True)
    (folder/'inputs_08_left.json').write_text(json.dumps(replay.inputs),encoding='utf-8')
    (folder/'stage8_left_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
