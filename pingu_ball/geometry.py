"""Continuous circle collision against rectangles and convex spike triangles.

Rounded corners are tested as circles, rather than an expanded square AABB.
The simulation splits parabolic motion into short pieces and refines flat-face
impact times against the actual acceleration equation.
"""
import math

EPS = 1e-8


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def circle_rect_overlap(x, y, r, rect, tolerance=0.0):
    left, top, w, h = rect
    qx = max(left, min(x, left + w))
    qy = max(top, min(y, top + h))
    return (x-qx)**2 + (y-qy)**2 < (r-tolerance)**2


def circle_rect_separation(x, y, r, rect):
    """Minimum outward translation, including rounded corners and inside starts."""
    left, top, w, h = rect
    qx = max(left, min(x, left+w))
    qy = max(top, min(y, top+h))
    dx, dy = x-qx, y-qy
    distance = math.hypot(dx, dy)
    if distance > EPS:
        return max(0., r-distance), (dx/distance, dy/distance)
    distance, normal = min(((x-left, (-1., 0.)), (left+w-x, (1., 0.)),
                            (y-top, (0., -1.)), (top+h-y, (0., 1.))))
    return r+distance, normal


def point_rect(x, y, rect):
    left, top, w, h = rect
    return left <= x <= left+w and top <= y <= top+h


def ray_circle(p, d, center, r):
    q = (p[0]-center[0], p[1]-center[1])
    a = dot(d, d)
    if a < EPS:
        return None
    b = 2 * dot(q, d)
    c = dot(q, q) - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    t = (-b-math.sqrt(max(0, disc))) / (2*a)
    if -EPS <= t <= 1+EPS:
        t = max(0.0, min(1.0, t))
        n = ((q[0]+d[0]*t)/r, (q[1]+d[1]*t)/r)
        if dot(d, n) < -EPS:
            return t, n
    return None


def sweep_circle_rect(p, d, rect, r):
    l, top, w, h = rect
    right, bottom = l+w, top+h
    candidates = []
    for boundary, axis, normal, lo, hi in [
        (l-r, 0, (-1., 0.), top, bottom),
        (right+r, 0, (1., 0.), top, bottom),
        (top-r, 1, (0., -1.), l, right),
        (bottom+r, 1, (0., 1.), l, right),
    ]:
        if dot(d, normal) >= -EPS or abs(d[axis]) < EPS:
            continue
        t = (boundary-p[axis]) / d[axis]
        other = p[1-axis] + d[1-axis]*t
        if -EPS <= t <= 1+EPS and lo-EPS <= other <= hi+EPS:
            candidates.append((max(0., min(1., t)), normal))
    for corner in [(l, top), (right, top), (right, bottom), (l, bottom)]:
        hit = ray_circle(p, d, corner, r)
        if hit:
            candidates.append(hit)
    return min(candidates, key=lambda hit: hit[0]) if candidates else None


def sweep_point_rect(p, d, rect):
    if point_rect(*p, rect):
        return 0.0
    l, top, w, h = rect
    enter, leave = 0., 1.
    for value, delta, low, high in [(p[0], d[0], l, l+w), (p[1], d[1], top, top+h)]:
        if abs(delta) < EPS:
            if not low <= value <= high:
                return None
        else:
            a, b = sorted(((low-value)/delta, (high-value)/delta))
            enter, leave = max(enter, a), min(leave, b)
            if enter > leave + EPS:
                return None
    return enter if 0 <= enter <= 1 else None


def sweep_circle_polygon(p, d, vertices, r):
    # Vertices must be clockwise in screen coordinates (positive signed area).
    candidates = []
    for i, a in enumerate(vertices):
        b = vertices[(i+1) % len(vertices)]
        ex, ey = b[0]-a[0], b[1]-a[1]
        length = math.hypot(ex, ey)
        n = (ey/length, -ex/length)
        initial = (p[0]-a[0])*n[0] + (p[1]-a[1])*n[1]
        speed = dot(d, n)
        if speed < -EPS:
            t = (r-initial)/speed
            if -EPS <= t <= 1+EPS:
                q = (p[0]+d[0]*t-r*n[0], p[1]+d[1]*t-r*n[1])
                along = ((q[0]-a[0])*ex+(q[1]-a[1])*ey)/(length*length)
                if -EPS <= along <= 1+EPS:
                    candidates.append((max(0., min(1., t)), n))
        hit = ray_circle(p, d, a, r)
        if hit:
            candidates.append(hit)
    return min(candidates, key=lambda hit: hit[0]) if candidates else None


def circle_polygon_overlap(x, y, r, vertices):
    inside = True
    nearest = float('inf')
    for i, a in enumerate(vertices):
        b = vertices[(i+1) % len(vertices)]
        dx, dy = b[0]-a[0], b[1]-a[1]
        cross = dx*(y-a[1])-dy*(x-a[0])
        inside &= cross >= 0
        t = max(0., min(1., ((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
        nearest = min(nearest, (x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2)
    return inside or nearest <= r*r
