"""Fit the playable world to the HUD's full-width display area."""
from dataclasses import replace
import math

VIEW_LEFT = 16
VIEW_TOP = 144
VIEW_WIDTH = 1568
VIEW_HEIGHT = 672


def expand_playfield(level):
    """Add air at both sides; translate every object without changing distances.

    Source document levels remain untouched. Native physics dimensions grow
    with the rendered world, so there are no invisible walls inside the view.
    """
    width = max(level.width, math.ceil(level.height*VIEW_WIDTH/VIEW_HEIGHT))
    shift = (width-level.width)/2
    devices = []
    for device in level.specials:
        options = dict(device.options)
        if 'path' in options:
            x, y, w, h = options['path']
            options['path'] = (x+shift, y, w, h)
        devices.append(replace(device, x=device.x+shift, options=options))
    return replace(level, width=width,
        platforms=[replace(q, x=q.x+shift) for q in level.platforms],
        collectibles=[replace(q, x=q.x+shift) for q in level.collectibles],
        spikes=[replace(q, x=q.x+shift) for q in level.spikes],
        powerups=[replace(q, x=q.x+shift) for q in level.powerups],
        specials=devices, spawn=(level.spawn[0]+shift, level.spawn[1]),
        goal=(level.goal[0]+shift, *level.goal[1:]),
        movement_markers=[(x+shift, y) for x, y in level.movement_markers],
        no_fish_waypoints=[(x+shift, condition) for x, condition in level.no_fish_waypoints],
        fish_waypoints=[(x+shift, condition) for x, condition in level.fish_waypoints])
