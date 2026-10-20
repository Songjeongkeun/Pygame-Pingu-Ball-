"""Derive non-solid endpoint guides from the platforms' actual movement."""


def movement_endpoints(level):
    cells=set()
    for d in level.specials:
        if d.type_id not in (501,508):continue
        travel=d.options.get('travel',128.)
        for distance in (0.,travel):
            cells.add((d.x+(distance if d.type_id==501 else 0.),
                       d.y+(distance if d.type_id==508 else 0.),d.w,d.h))
    for q in level.platforms:
        if q.kind=='moving':
            for distance in (0.,q.travel):cells.add((q.x+distance,q.y,q.w,q.h))
    return tuple(sorted(cells))
