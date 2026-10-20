"""Replay stage 4's spring staircase, ferry and breakable exit from spawn."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pingu_ball.levels import make_level
from tools.verify_stage3 import Replay


def route(level=None):
    replay = Replay(level or make_level(4))
    world = replay.world
    for prefix, x in (('normal-42',1504), ('spring-36',1376),
                      ('spring-35',1504), ('normal-13',1376),
                      ('normal-12',1312), ('normal-11',1184)):
        replay.land(prefix, x)
    ferry = [block for block in world.platforms if block.kind == 'moving']
    replay.until(lambda: (world.last_landing or '').startswith('ferry'),
                 1040, label='board-ferry')

    def ready_to_leave():
        left = min(block.rect_at(world.time,world.physics)[0] for block in ferry)
        return (world.time-world.last_landing_time < world.physics.dt*1.1
                and (world.last_landing or '').startswith('ferry') and left < 510)

    # The ball follows the deck using its own input; no platform carry is used.
    replay.until(ready_to_leave,
                 lambda: min(block.rect_at(world.time,world.physics)[0] for block in ferry)+16,
                 label='ride-left')
    replay.land('normal-8',288)
    replay.land('breakable-44',224)
    replay.until(lambda: world.cleared,480,label='goal')
    return replay


def main():
    replay = route()
    world = replay.world
    report = dict(stage=4, cleared=world.cleared, failures=world.failures,
                  time=round(world.time,4), from_spawn=True,
                  milestones=replay.milestones)
    folder = ROOT/'output/stage4-validation'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'inputs.json').write_text(json.dumps(replay.inputs),encoding='utf-8')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
    return 0 if world.cleared and not world.failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
