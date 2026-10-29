"""End-to-end campaign witnesses using only real movement/item inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.levels import make_level
from tools.verify_stage3 import Replay

REVIEW_NOTES={
    8: ['왼쪽 이동 발판·점프 블록·파괴 블록 다리 7개는 tools/verify_stage8.py로 실제 시작점부터 검증했다.',
        '오른쪽 상하 이동 발판의 범위는 y=192~896px이다. 이글루까지의 전체 완주 경로는 미확인이다.'],
}


class Witness(Replay):
    def __init__(self,number):
        super().__init__(make_level(number))

    def tick(self,direction=0,use=False,power_direction=None):
        if self.world.failed or self.world.cleared: return
        power_direction=direction if power_direction is None else power_direction
        self.inputs.append([direction,bool(use),power_direction if use else 0])
        self.world.step(direction,use_power=bool(use),power_direction=power_direction if use else 0)

    def land(self,prefix,target):
        w=self.world;since=w.time
        self.until(lambda:w.last_landing_time>since and w.time-w.last_landing_time<w.physics.dt*1.1
                   and (w.last_landing or '').startswith(prefix),target,label=prefix)


def route(number):
    if number==3:
        from tools.verify_stage3 import route as existing
        return existing()
    if number==4:
        from tools.verify_stage4 import route as existing
        return existing()
    r=Witness(number);w=r.world
    if number==1:
        r.land('normal-7',193)
        r.land('normal-10',384)
        r.land('normal-11',416)
        r.land('normal-16',608)
        r.land('normal-17',672)
        r.land('normal-20',800)
        r.land('normal-21',864)
        r.until(lambda:w.ball.launch,992,label='upper-launcher')
        r.until(lambda:w.ball.x<192,None,label='fly-left')
        r.until(lambda:w.last_landing=='launcher_right-30',160,label='lower-launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==2:
        r.land('normal-2',288)
        for name,x in [('normal-8',544),('breakable-9',608),('normal-10',736),
                       ('breakable-12',864),('normal-13',928),('normal-14',992),
                       ('normal-15',1056),('breakable-16',1248),('normal-17',1440),
                       ('spring-19',1504),('spring-20',1376)]: r.land(name,x)
        r.until(lambda:w.ball.launch,1184,label='launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==5:
        r.land('normal-37',1504)
        # Wait on the safe start deck for the later timed staircase's phase.
        for _ in range(3): r.land('normal-37',1504)
        for name in ('launcher_left-43','launcher_left-44','launcher_left-45','launcher_left-46'):
            r.until(lambda:w.last_landing==name,1440,label=name)
            r.tick(1)
        for name,x in [('breakable-66',1376),('normal-36',1248),('normal-35',1184),
                       ('spring-33',1056),('spring-32',1184),('spring-31',1056),
                       ('spring-72',1184),('spring-73',1056),
                       ('launcher_left-47',1184)]: r.land(name,x)
        r.until(lambda:w.ball.x<340,None,label='fly-left')
        r.tick(1)
        r.until(lambda:w.ball.y>450,224,label='below-bridge')
        for name,x in [('timed-67',288),('timed-68',416),
                       ('timed_off-69',544),('timed_off-70',672)]: r.land(name,x)
        r.until(lambda:w.cleared,800,label='goal')
    elif number==6:
        r.land('normal-1',96)
        for _ in range(2): r.land('normal-1',96)
        r.land('breakable-9',160)
        r.land('breakable-10',352)
        r.land('breakable-11',160)
        r.until(lambda:w.ball.launch,352,label='lower-launcher')
        r.until(lambda:w.ball.x>1390,None,label='fly-right')
        r.tick(-1)
        r.land('normal-24',1440)
        r.land('spring-28',1504)
        ferry=[q for q in w.platforms if q.kind=='moving']
        flight=(w.physics.pad_speed+math.sqrt(w.physics.pad_speed**2-2*w.physics.gravity*128))/w.physics.gravity
        def ready_to_board():
            now=[q.rect_at(w.time,w.physics) for q in ferry]
            future=[q.rect_at(w.time+flight,w.physics) for q in ferry]
            # Rise beside the approaching deck, then land on its top.
            return (w.last_landing=='spring-28' and w.time-w.last_landing_time<w.physics.dt*1.1
                    and max(q[0]+q[2] for q in now)<1486
                    and min(q[0] for q in future)+20<1504<max(q[0]+q[2] for q in future)-20)
        r.until(ready_to_board,1504,seconds=40,label='wait-for-ferry')
        r.until(lambda:(w.last_landing or '').startswith('ferry'),1504,label='spring-to-ferry')
        r.until(lambda:w.time-w.last_landing_time<w.physics.dt*1.1
                and (w.last_landing or '').startswith('ferry') and w.ball.x<700,
                lambda:min(q.rect_at(w.time,w.physics)[0] for q in ferry)+32,
                seconds=25,label='ride-left')
        r.land('spring-27',544)
        r.land('normal-20',672)
        r.until(lambda:w.last_landing=='launcher_right-25' and w.ball.launch,800,label='upper-launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==7:
        for name,x in [('spring-3',224),('spring-7',352),('spring-11',480),
                       ('spring-15',608),('spring-34',736),('spring-41',864)]: r.land(name,x)
        ferry=[q for q in w.platforms if q.kind=='moving']
        flight=(w.physics.pad_speed+math.sqrt(w.physics.pad_speed**2+2*w.physics.gravity*64))/w.physics.gravity
        def ready_to_board():
            rects=[q.rect_at(w.time+flight,w.physics) for q in ferry]
            return (w.time-w.last_landing_time<w.physics.dt*1.1 and w.last_landing=='spring-41'
                    and min(q[0] for q in rects)+20<1080<max(q[0]+q[2] for q in rects)-20)
        r.until(ready_to_board,864,label='wait-for-ferry')
        r.until(lambda:(w.last_landing or '').startswith('ferry'),1080,label='board-ferry')
        r.until(lambda:w.time-w.last_landing_time<w.physics.dt*1.1
                and (w.last_landing or '').startswith('ferry') and w.ball.x>1250,
                lambda:max(q.rect_at(w.time,w.physics)[0]+q.w for q in ferry)-16,
                label='ride-right')
        r.tick(1,True)
        r.until(lambda:w.cleared,1440,label='goal')
    elif number==8:
        for name,x in [('normal-9',1184),('normal-11',1376),('normal-12',1440),
                       ('breakable-13',1632),('normal-14',1824),('normal-15',1888),
                       ('normal-40',2016)]: r.land(name,x)
    elif number==9:
        r.land('normal-0',96)
        r.land('normal-2',240)
        r.until(lambda:w.ball.x>=298,300,label='jump-timing')
        r.tick(1,True)
        r.land('normal-9',608)
        for name,x in [('spring-10',736),('breakable-12',864),('breakable-13',992),
                       ('breakable-14',1120),('breakable-15',1248),('spring-11',1376)]: r.land(name,x)
        since=w.last_landing_time
        r.until(lambda:w.time-since>=.37,None,label='launcher-timing')
        r.until(lambda:w.ball.launch,1312,label='launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==10:
        r.land('normal-5',224)
        for name,x in [('normal-11',416),('normal-13',544),('normal-15',672),
                       ('normal-18',864),('normal-20',992),('normal-22',1120),
                       ('normal-37',1248),('normal-34',1056),('normal-33',928),
                       ('normal-32',800),('normal-30',608),('normal-29',450),
                       ('normal-45',160)]: r.land(name,x)
        r.until(lambda:w.ball.x<50,32,label='launcher-approach')
        r.until(lambda:w.last_landing=='launcher_right-43',None,label='lower-launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==11:
        r.land('normal-55',1440)
        # Align the sea lion's phase before clearing the two breakable columns.
        r.land('normal-55',1440)
        for name,x in [('normal-51',1312),('normal-50',1248),('breakable-41',1184),
                       ('breakable-42',1184),('normal-40',1056),('normal-39',992),
                       ('breakable-36',864),('breakable-37',864),
                       ('normal-35',800),('normal-34',736),('spring-33',672)]: r.land(name,x)
        r.until(lambda:w.ball.y<562 and w.ball.vy<0,672,label='above-center-wall')
        for name,x in [('normal-29',608),('normal-28',544),('normal-27',480),('spring-26',416)]: r.land(name,x)
        # The collected jump item reaches the top of the original breakable column.
        r.until(lambda:w.ball.vy>=-8 and w.ball.y<485,416,label='upper-spring-apex')
        r.tick(-1,True)
        for n in (4,5,6,7,8): r.land(f'breakable-{n}',352)
        r.land('breakable-24',352)
        r.land('spring-23',288)
        r.land('spring-26',416)
        # The second jump pickup returns to the top of the left sun wall.
        # Drop through its gap onto the launcher; the broken columns now let
        # its horizontal flight reach the igloo underneath the starting ledge.
        r.until(lambda:w.ball.vy>=-8 and w.ball.y<485,416,label='return-spring-apex')
        r.tick(-1,True)
        r.land('normal-3',288)
        r.until(lambda:w.last_landing=='launcher_right-25',160,label='left-launcher')
        r.until(lambda:w.cleared,None,label='goal')
    elif number==12:
        r.land('normal-41',1440)
        # The aligned central column changes the initial landing height.
        # Wait on the safe ledge for the sun to clear the launcher's exit.
        for _ in range(3): r.land('normal-41',1440)
        r.until(lambda:w.ball.launch,1632,label='right-launcher')
        r.until(lambda:w.ball.x>=2110,None,label='fly-right')
        r.tick(1)
        r.until(lambda:w.last_landing=='launcher_left-66',2144,label='left-launcher')
        r.until(lambda:w.ball.x<1730,None,label='fly-left')
        r.until(lambda:w.cleared,1696,label='goal')
    else:
        raise RuntimeError(f'Route {number} not authored yet')
    return r


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=int,nargs='*',choices=range(1,13))
    args=parser.parse_args()
    folder=ROOT/'output/campaign-routes';folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in (args.stage or range(1,13)):
        try:
            witness=route(n)
            inputs=[frame if isinstance(frame,list) else [frame,False,0] for frame in witness.inputs]
            # Recreate the stage from scratch and replay the recorded controls.
            world=World(make_level(n),auto_reset=False)
            for direction,use,power_direction in inputs:
                world.step(direction,use_power=use,power_direction=power_direction)
                if world.failed or world.cleared: break
            result=dict(stage=n,cleared=world.cleared,failures=world.failures,
                        time=round(world.time,4),from_spawn=True,ticks=len(inputs),
                        position=[world.ball.x,world.ball.y],reason=world.last_failure)
            if world.cleared and not world.failures:
                (folder/f'inputs_{n:02d}.json').write_text(json.dumps(inputs),encoding='utf-8')
                result['milestones']=witness.milestones
        except RuntimeError as error:
            result=dict(stage=n,cleared=False,status='unverified',error=str(error))
            if n in REVIEW_NOTES: result['review_notes']=REVIEW_NOTES[n]
        results.append(result)
        print(json.dumps(result,ensure_ascii=True),flush=True)
    (folder/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    files=['pingu_ball/campaign.py','pingu_ball/engine.py','pingu_ball/config.py',
           'pingu_ball/specials.py','assets/stages/layouts.json','tools/verify_campaign_routes.py']
    (folder/'source_hashes.json').write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in files},indent=2),encoding='utf-8')
    return 0 if all(r['cleared'] for r in results) else 1


if __name__=='__main__': raise SystemExit(main())
