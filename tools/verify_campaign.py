"""Validate campaign loading, native device state, start safety and goal sensors.

This is a smoke check, not an end-to-end route witness for all twelve maps.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pingu_ball.engine import World
from pingu_ball.levels import make_levels


def main():
    results=[]
    for level in make_levels()[1:]:
        world=World(level,auto_reset=False)
        for _ in range(120): world.step(0)
        spawn_safe=not world.failed and world.time>.99
        gx,gy,_,_=level.goal
        supported=any(q.kind=='normal' and q.x==gx and q.y==gy+64 for q in level.platforms)
        finite=all(math.isfinite(v) for v in (world.ball.x,world.ball.y,world.ball.vx,world.ball.vy))
        moved=all(q.rect_at(0)!=q.rect_at(.2) for q in world.platforms if q.kind=='moving' or getattr(q,'device',None) and q.device.type_id==508)
        world.reset();world.ball.x,world.ball.y=gx+32,gy+32;world.step(0)
        full_route=False
        if level.index==3:
            from tools.verify_stage3 import route
            witness=route().world
            full_route=witness.cleared and not witness.failures
        elif level.index==4:
            from tools.verify_stage4 import route
            witness=route().world
            full_route=witness.cleared and not witness.failures
        elif level.index in (5,6,11):
            from tools.verify_campaign_routes import route
            witness=route(level.index).world
            full_route=witness.cleared and not witness.failures
        item={'stage':level.index,'spawn_safe':spawn_safe,'normal_goal_support':supported,
              'goal_sensor_clears':world.cleared,'finite':finite,'moving_devices_run':moved,
              'platforms':len(world.platforms),'devices':len(level.specials),'powerups':len(level.powerups),
              'full_route_verified':full_route}
        item['passed']=all(item[k] for k in ('spawn_safe','normal_goal_support','goal_sensor_clears','finite','moving_devices_run'))
        results.append(item);print(json.dumps(item),flush=True)
    baseline=json.loads((ROOT/'tests/fixtures/tutorial_baseline.json').read_text(encoding='utf-8'))
    tutorials_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in baseline.items())
    source=json.loads((ROOT/'assets/stages/layouts.json').read_text(encoding='utf-8'))
    document=ROOT/source['source']
    if not document.exists(): document=ROOT.parent/source['source']
    document_unchanged=hashlib.sha256(document.read_bytes()).hexdigest()==source['source_sha256']
    report={'stages':results,'tutorials_unchanged':tutorials_unchanged,'document_unchanged':document_unchanged}
    destination=ROOT/'output/stage-import/verification.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return 0 if all(q['passed'] for q in results) and tutorials_unchanged and document_unchanged else 1


if __name__=='__main__': raise SystemExit(main())
