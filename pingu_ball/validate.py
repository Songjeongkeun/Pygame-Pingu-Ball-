"""Run reproducible route witnesses through the real game simulation."""
import argparse
import json
from pathlib import Path
from .engine import World
from .levels import make_levels, waypoint_direction, condition_met


def run_route(level, collect_all, seconds=45):
    world=World(level)
    path=level.fish_waypoints if collect_all else level.no_fish_waypoints
    point=0
    trace=[]
    for tick in range(int(seconds/world.physics.dt)):
        target,condition=path[min(point,len(path)-1)]
        ready=condition_met(world,condition)
        if condition.startswith('land:'):
            ready = ready and abs(world.ball.x-target)<2 and world.time-world.last_landing_time < world.physics.dt*1.1
        if ready:
            point+=1
            if point>=len(path):break
            target,condition=path[point]
        direction=waypoint_direction(world,target)
        world.step(direction)
        if tick%120==0:
            trace.append([round(world.time,2),round(world.ball.x,1),round(world.ball.y,1),point,world.plus,world.minus])
        if world.failures or world.cleared:break
    expected=1 if level.index==0 else 3
    success=world.cleared and world.minus==0 and (world.plus==expected if collect_all else world.plus==0)
    return {'stage':level.index,'route':'all_fish' if collect_all else 'no_fish',
            'passed':success,'cleared':world.cleared,'plus':world.plus,'minus':world.minus,
            'time':round(world.time,3),'failures':world.failures,'reason':world.last_failure,
            'position':[round(world.ball.x,2),round(world.ball.y,2)],'waypoint':point,'trace':trace}


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
