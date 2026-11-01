"""Twelve document campaign stages plus the existing first-tutorial API alias."""
from .config import REGULAR_STAGE_COUNT


def make_levels():
    return [make_level(i) for i in range(REGULAR_STAGE_COUNT + 1)]


def make_level(index):
    if type(index) is not int or not 0 <= index <= REGULAR_STAGE_COUNT:
        raise ValueError(f'stage must be between 0 and {REGULAR_STAGE_COUNT}')
    if index == 0:
        from .special_lab import make_lab
        return make_lab(1)
    from .campaign import make_campaign_level
    return make_campaign_level(index)


def waypoint_direction(world, target):
    """Supply ordinary left/right input for repeatable route validation."""
    b = world.ball
    distance = target-b.x
    stop = b.vx*abs(b.vx)/(2*world.physics.deceleration)
    if abs(distance) < 1.5 and abs(b.vx) < 8:
        return 0
    if (distance > 0 and b.vx > 0 and distance <= stop+.7) or (distance < 0 and b.vx < 0 and -distance <= -stop+.7):
        return 0
    return 1 if distance > 0 else -1


def condition_met(world, condition):
    if condition == 'goal':
        return world.cleared
    if condition.startswith('fish'):
        return int(condition[4:]) in world.collected
    if condition.startswith('land:'):
        return world.last_landing is not None and world.last_landing.startswith(condition[5:])
    if condition.startswith('switch:'):
        return world.switches.get(condition[7:], False)
    if condition.startswith('below:'):
        return world.ball.y > float(condition[6:])
    return False
