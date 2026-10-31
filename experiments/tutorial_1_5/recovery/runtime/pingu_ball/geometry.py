"""원과 사각형·볼록한 가시 삼각형 사이의 연속 충돌을 계산한다.

모서리는 사각형 경계를 넓혀 처리하는 대신 둥근 원으로 검사한다.
시뮬레이션은 포물선 움직임을 짧은 구간으로 나누고, 실제 가속도 식을 이용해
평평한 면에 부딪히는 시간을 정밀하게 계산한다.
"""
import math

EPS = 1e-8


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def circle_rect_overlap(x, y, r, rect, tolerance=0.0):
    left, top, w, h = rect
    reach=abs(r-tolerance)
    if x+reach<left-EPS or x-reach>left+w+EPS or y+reach<top-EPS or y-reach>top+h+EPS:
        return False
    qx = max(left, min(x, left + w))
    qy = max(top, min(y, top + h))
    return (x-qx)**2 + (y-qy)**2 < (r-tolerance)**2


def circle_rect_separation(x, y, r, rect):
    """둥근 모서리와 내부에서 시작한 경우를 포함해, 밖으로 밀어낼 최소 이동량을 구한다."""
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


def swept_bounds_overlap(p, d, bounds, radius):
    """스치거나 허용 오차만큼 늦은 연속 충돌을 놓치지 않도록 보수적으로 후보를 거른다."""
    left,top,right,bottom=bounds
    end_x,end_y=p[0]+d[0],p[1]+d[1]
    margin=radius+EPS*(1+abs(d[0])+abs(d[1])+right-left+bottom-top)
    return not (max(p[0],end_x)+margin<left or min(p[0],end_x)-margin>right
                or max(p[1],end_y)+margin<top or min(p[1],end_y)-margin>bottom)


def sweep_circle_rect(p, d, rect, r):
    l, top, w, h = rect
    right, bottom = l+w, top+h
    if not swept_bounds_overlap(p,d,(l,top,right,bottom),r):
        return None
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
    # 꼭짓점은 화면 좌표에서 시계 방향이어야 하며, 부호 있는 넓이는 양수여야 한다.
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
