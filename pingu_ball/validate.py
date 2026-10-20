"""Run reproducible route witnesses through the real game simulation."""
import argparse
import json
from pathlib import Path
from .engine import World
from .levels import make_levels
from .route_driver import RouteDriver


def run_route(level, collect_all, seconds=None):
    seconds = level.time_limit if seconds is None else seconds
    world=World(level, auto_reset=False)
    path=level.fish_waypoints if collect_all else level.no_fish_waypoints
    if not path:
        return {'stage':level.index, 'passed':False, 'cleared':False,
                'reason':'no_route_witness', 'route':'all_fish' if collect_all else 'no_fish'}
    point=0
    trace=[]
    driver=RouteDriver(world, path)
    for tick in range(int(seconds/world.physics.dt)):
        world.step(driver.direction())
        point=driver.point
        if tick%120==0:
            trace.append([round(world.time,2),round(world.ball.x,1),round(world.ball.y,1),point,world.plus,world.minus])
        if world.failures or world.cleared:break
    expected=sum(f.value > 0 for f in level.collectibles)
    success=world.cleared and world.minus==0 and (world.plus==expected if collect_all else world.plus==0)
    return {'stage':level.index,'route':'all_fish' if collect_all else 'no_fish',
            'passed':success,'cleared':world.cleared,'plus':world.plus,'minus':world.minus,
            'time':round(world.time,3),'failures':world.failures,'reason':world.last_failure,
            'position':[round(world.ball.x,2),round(world.ball.y,2)],'waypoint':point,'trace':trace,
            'condition':path[point], 'milestones':driver.milestones}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='')
    parser.add_argument('--stage',type=int)
    args=parser.parse_args()
    levels=make_levels()
    if args.stage is not None:levels=[levels[args.stage]]
    results=[run_route(level,all_fish) for level in levels for all_fish in (False,True)]
    for r in results:
        print(json.dumps(r,ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__=='__main__':main()
