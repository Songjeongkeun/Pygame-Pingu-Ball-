"""The 37 numbered devices in the supplied special-block specification.

No Pygame dependency: sensors and moving hazards feed the same continuous
collision solver as ordinary terrain. Per-device options are level data.
"""
from dataclasses import dataclass, field
import math

from .engine import Platform
from .geometry import (circle_rect_overlap, circle_polygon_overlap, sweep_circle_polygon,
                       sweep_circle_rect, swept_bounds_overlap)


CATALOG = {
    301: ('파괴 블록', 'platform'), 401: ('고점프', 'platform'),
    302: ('점멸', 'platform'), 403: ('낙하', 'platform'),
    307: ('가속', 'movement'), 501: ('이동 블록', 'movement'),
    508: ('엘리베이터', 'movement'), 507: ('벽 미끄럼', 'movement'),
    511: ('자석', 'movement'), 512: ('시프트', 'movement'),
    304: ('별 스위치', 'switch'), 404: ('전원', 'switch'),
    305: ('장벽', 'switch'), 415: ('투명 스위치', 'switch'),
    416: ('투명 블록', 'switch'), 510: ('전기 블록', 'switch'),
    306: ('빨강·파랑', 'switch'),
    402: ('좌우 슈터', 'transport'), 405: ('4방향 슈터', 'transport'),
    406: ('8방향 슈터', 'transport'), 407: ('블랙홀', 'transport'),
    408: ('화이트홀', 'transport'), 410: ('포털', 'transport'),
    303: ('스파크', 'hazard'), 201: ('가시', 'hazard'),
    202: ('소형 가시', 'hazard'), 211: ('투명 가시', 'hazard'),
    221: ('밝은 가시', 'hazard'), 502: ('직선 레이저', 'hazard'),
    503: ('가로 회전 레이저', 'hazard'), 504: ('세로 회전 레이저', 'hazard'),
    601: ('소행성', 'hazard'), 603: ('대포', 'hazard'),
    602: ('레드바운스', 'hazard'), 604: ('단검', 'hazard'),
    605: ('벽 단검', 'hazard'), 606: ('붉은 적', 'hazard'),
    701: ('태양', 'hazard'), 702: ('고드름', 'hazard'),
    703: ('바다사자', 'hazard'),
}
SOLID = {301,401,302,403,307,501,508,507,511,512,304,404,305,415,416,510,306,402,405,406}


@dataclass
class Device:
    id: str
    type_id: int
    x: float
    y: float
    w: float = 64.
    h: float = 64.
    group: str = ''
    target: str = ''
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type_id not in CATALOG:
            raise ValueError(f'Unknown special type: {self.type_id}')
        if not self.id or not all(math.isfinite(v) for v in (self.x,self.y,self.w,self.h)) or min(self.w,self.h) <= 0:
            raise ValueError('Devices need a unique nonempty id and finite positive dimensions')
        positive = ('period','active_for','speed','length','thickness','duration','radius','lifetime','delay')
        for key in positive:
            if key in self.options and (not math.isfinite(self.options[key]) or self.options[key] <= 0):
                raise ValueError(f'{key} must be finite and positive')


def triangle_wave(distance, travel):
    return travel-abs(distance%(2*travel)-travel) if travel else 0.


def overlap_normal(point,polygon):
    """Outward normal of the nearest edge, including already-overlapping starts."""
    px,py=point
    inside=True
    candidates=[]
    for i,a in enumerate(polygon):
        b=polygon[(i+1)%len(polygon)]
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        inside &= dx*(py-a[1])-dy*(px-a[0]) >= 0
        along=max(0.,min(1.,((px-a[0])*dx+(py-a[1])*dy)/(length*length)))
        qx,qy=a[0]+along*dx,a[1]+along*dy
        distance=math.hypot(px-qx,py-qy)
        candidates.append((distance,(dy/length,-dx/length),
                           ((px-qx)/distance,(py-qy)/distance) if distance>1e-8 else (dy/length,-dx/length)))
    _,outward,from_edge=min(candidates,key=lambda item:item[0])
    return outward if inside else from_edge


class DevicePlatform(Platform):
    def __init__(self, device, runtime):
        o = device.options
        kind = {301:'breakable',401:'spring',302:'timed',501:'moving',415:'switch',416:'linked',402:'launcher'}.get(device.type_id,'normal')
        super().__init__(device.id, device.x, device.y, device.w, device.h, kind=kind,
                         travel=o.get('travel',128.) if device.type_id == 501 else 0.,
                         period=o.get('period',2.8), active_for=o.get('active_for',2.),
                         phase=o.get('phase',0.), group=device.group, arrow=o.get('direction',1))
        self.device, self.runtime = device, runtime
        self.offset = [0.,0.]
        self.fall_at = None

    def rect_at(self, time, physics=None):
        p = self.runtime.world.physics
        x,y,w,h = super().rect_at(time,p)
        d,o = self.device,self.device.options
        if d.type_id == 508:
            travel = o.get('travel',128.)
            y += triangle_wave(time*o.get('speed',p.platform_speed)+o.get('phase',0.),travel)
        elif d.type_id == 501:
            x = d.x + triangle_wave(time*o.get('speed',p.platform_speed)+o.get('phase',0.),o.get('travel',128.))
        if self.fall_at is not None:
            y += .5*o.get('gravity',p.gravity)*max(0.,time-self.fall_at)**2
        return x+self.offset[0],y+self.offset[1],w,h

    def active(self, time, switches):
        if self.device.type_id == 510: return not self.destroyed
        if not self.runtime.enabled(self.device): return False
        if self.device.type_id == 305:
            return not switches.get(self.group,False) and not self.destroyed
        return super().active(time,switches)


@dataclass
class Projectile:
    id: str
    x: float
    y: float
    vx: float
    vy: float
    born: float
    lifetime: float
    radius: float

    def center(self,time):
        return self.x+self.vx*(time-self.born), self.y+self.vy*(time-self.born)


class SpecialRuntime:
    def __init__(self, world):
        self.world = world
        self.devices = {d.id:d for d in world.level.specials}
        if len(self.devices) != len(world.level.specials) or set(self.devices)&{q.id for q in world.platforms}:
            raise ValueError('Special and platform ids must be unique')
        self.state = {d.id:{'armed':True,'triggered':None,'shot':0,'destroyed':False,
                            'last_effect':-100.,'next_shot':d.options.get('phase',0.)} for d in self.devices.values()}
        self.platforms = {d.id:DevicePlatform(d,self) for d in self.devices.values() if d.type_id in SOLID}
        world.platforms.extend(self.platforms.values())
        for d in self.devices.values():
            if d.type_id == 404: world.switches['power:'+d.group] = d.options.get('on',False)
            if d.type_id == 415: world.switches.setdefault(d.group,d.options.get('on',False))
        self.projectiles = []
        self.teleport_until = -1.
        self.speed_until = -1.
        self.speed_factor = 1.
        self.last_sync = 0.

    def enabled(self,d):
        bus = d.options.get('power_group')
        if bus is None and d.type_id not in (404,415,416,305,304,510):
            bus = d.group if 'power:'+d.group in self.world.switches else None
        return not self.state[d.id]['destroyed'] and (bus is None or self.world.switches.get('power:'+bus,False))

    def speed_scale(self):
        return self.speed_factor if self.world.time < self.speed_until else 1.

    def subdivisions(self,dt):
        # Bound moving/rotating shapes to <=2px between temporal collision samples.
        speed=0.
        for d in self.devices.values():
            o=d.options
            if d.type_id in (503,504):
                speed=max(speed,abs(math.radians(o.get('angular_speed',90.)))*o.get('length',192.))
            elif d.type_id == 702:
                age=self.icicle_age(d,self.world.time)
                speed=max(speed,o.get('gravity',self.world.physics.gravity)*max(0.,age-o.get('delay',.5)))
            elif d.type_id in (602,604,605,606,703):
                velocity=o.get('velocity',(96.,64.))
                speed=max(speed,math.hypot(*velocity),o.get('speed',96.))
        return max(math.ceil(dt*240),math.ceil(speed*dt/2))

    def rect(self,d,time):
        if d.id in self.platforms: return self.platforms[d.id].rect_at(time,self.world.physics)
        x,y,w,h = d.x,d.y,d.w,d.h
        o = d.options
        if d.type_id == 702:
            age=self.icicle_age(d,time)
            y += .5*o.get('gravity',self.world.physics.gravity)*max(0.,age-o.get('delay',.5))**2
        elif d.type_id == 703:
            cx,cy,_=self.patrol_pose(d,time)
            w=h=o.get('size',56.)
            x,y=cx-w/2,cy-h/2
        elif d.type_id == 602:
            x += triangle_wave(time*o.get('speed',96.),o.get('travel',128.))
            y -= abs(math.sin(time*math.pi/o.get('period',1.2)))*o.get('bounce_height',48.)
        elif d.type_id == 604:
            bounds = o.get('bounds',(0.,0.,self.world.physics.width-w,self.world.physics.height-h))
            speed = o.get('velocity',(96.,64.))
            x = bounds[0]+triangle_wave(x-bounds[0]+speed[0]*time,bounds[2]-bounds[0])
            y = bounds[1]+triangle_wave(y-bounds[1]+speed[1]*time,bounds[3]-bounds[1])
        elif d.type_id == 605:
            left,top,width,height = o.get('path',(d.x,d.y,128.,128.))
            distance = (time*o.get('speed',96.)+o.get('phase',0.))%(2*(width+height))
            if distance < width: x,y = left+distance,top
            elif distance < width+height: x,y = left+width,top+distance-width
            elif distance < 2*width+height: x,y = left+2*width+height-distance,top+height
            else: x,y = left,top+2*(width+height)-distance
        elif d.type_id == 606:
            x += triangle_wave(time*o.get('speed',32.),o.get('travel',0.))
        return x,y,w,h

    @staticmethod
    def icicle_age(d,time):
        return (time+d.options.get('phase',0.))%d.options.get('period',4.)

    @staticmethod
    def patrol_pose(d,time):
        """Center follows the joined block outline; angle is its tangent."""
        o=d.options
        left,top,width,height=o.get('path',(d.x,d.y,192.,64.))
        distance=(time*o.get('speed',64.)+o.get('phase',0.))%(2*(width+height))
        if distance<width: return left+distance,top,0
        if distance<width+height: return left+width,top+distance-width,90
        if distance<2*width+height: return left+2*width+height-distance,top+height,180
        return left,top+2*(width+height)-distance,270

    def sync(self,time):
        w = self.world
        dt = max(0.,time-self.last_sync)
        self.last_sync = time
        self.projectiles[:] = [p for p in self.projectiles if time < p.born+p.lifetime]
        for d in self.devices.values():
            if not self.enabled(d):
                if d.type_id == 603: self.state[d.id]['next_shot'] = time+d.options.get('period',1.5)
                continue
            state,o = self.state[d.id],d.options
            if d.type_id == 603:
                while state['next_shot'] <= time+1e-10:
                    self.fire(d,state['next_shot']); state['next_shot'] += o.get('period',1.5)
            if d.type_id == 511 and state['triggered'] is not None:
                mx,my,mw,mh=self.rect(d,time)
                center = (mx+mw/2,my+mh/2)
                for q in w.platforms:
                    if q.id == d.id or not self.targeted(d,q): continue
                    x,y,_,_ = q.rect_at(time,w.physics)
                    dx,dy = center[0]-x-q.w/2,center[1]-y-q.h/2
                    distance = math.hypot(dx,dy)
                    if 1e-6 < distance <= o.get('radius',256.):
                        sign = state.get('polarity',1)
                        step = min(o.get('speed',64.)*dt, max(0.,distance-64.)) if sign > 0 else o.get('speed',64.)*dt
                        if hasattr(q,'offset'):
                            q.offset[0] += dx/distance*step*sign; q.offset[1] += dy/distance*step*sign
                        else:
                            q.x += dx/distance*step*sign; q.y += dy/distance*step*sign

    def targeted(self,d,q):
        return q.id == d.target if d.target else bool(d.group and q.group == d.group)

    def fire(self,d,time):
        o = d.options
        angle = math.radians(o.get('angle',180. if d.type_id == 603 else 0.))
        speed = o.get('speed',192.)
        self.projectiles.append(Projectile(d.id+':'+str(time), d.x+d.w/2,d.y+d.h/2,
                                math.cos(angle)*speed,math.sin(angle)*speed,time,
                                o.get('lifetime',6.),o.get('radius',10.)))

    def land(self,q):
        if q.id not in self.devices: return
        d = self.devices[q.id]
        s,o,b,w = self.state[d.id],d.options,self.world.ball,self.world
        if d.type_id == 403 and q.fall_at is None: q.fall_at = w.time+o.get('delay',.1)
        elif d.type_id == 307:
            self.speed_factor = o.get('factor',1.5); self.speed_until = w.time+o.get('duration',2.)
        elif d.type_id in (304,404,511,512) and s['armed']:
            s['armed'] = False
            if d.type_id == 304 and w.plus >= o.get('required',1): w.switches[d.group] = True
            elif d.type_id == 404:
                key = 'power:'+d.group; w.switches[key] = not w.switches.get(key,False)
                if w.switches[key]:
                    for connected in self.devices.values():
                        if connected.type_id == 603 and (connected.group == d.group or connected.options.get('power_group') == d.group):
                            self.state[connected.id]['next_shot'] = w.time
            elif d.type_id == 511:
                s['triggered'] = w.time; s['polarity'] = -s.get('polarity',-o.get('polarity',1))
            elif d.type_id == 512:
                s['shifted'] = not s.get('shifted',False)
                dx,dy = o.get('offset',(128.,0.))
                for other in w.platforms:
                    if other.id == q.id or not self.targeted(d,other): continue
                    sign = 1 if s['shifted'] else -1
                    if hasattr(other,'offset'): other.offset[0] += sign*dx; other.offset[1] += sign*dy
                    else: other.x += sign*dx; other.y += sign*dy
            w.events.append(('device',d.id))
        elif d.type_id in (405,406):
            count = 4 if d.type_id == 405 else 8
            index = o.get('direction',o.get('initial_direction',0)+s['shot'])%count
            angle = -math.pi/2+index*math.tau/count
            speed = o.get('speed',w.physics.launch_speed)
            x,y,width,height=q.rect_at(w.time,w.physics)
            b.x,b.y = x+width/2,y+height/2
            b.vx,b.vy = math.cos(angle)*speed,math.sin(angle)*speed
            b.launch,b.launch_source = True,q.id
            s['shot'] += 1
        if d.type_id == 306 and s['triggered'] is None: s['triggered'] = w.time

    def wall_slide(self,q):
        if q.id in self.devices and self.devices[q.id].type_id == 507:
            self.world.ball.vy = min(self.world.ball.vy,self.devices[q.id].options.get('slide_speed',48.))

    def rearm(self):
        b,p = self.world.ball,self.world.physics
        for name,q in self.platforms.items():
            x,_,width,_ = q.rect_at(self.world.time,p)
            if b.x+p.radius < x or b.x-p.radius > x+width: self.state[name]['armed'] = True
            if q.device.type_id == 507:
                x,y,width,height = q.rect_at(self.world.time,p)
                touching = abs(b.x-(x-p.radius)) < .01 or abs(b.x-(x+width+p.radius)) < .01
                if touching and y <= b.y <= y+height:
                    b.vy = min(b.vy,q.device.options.get('slide_speed',48.))

    def polygons(self,d,time):
        """Convex polygons used by both rendering and swept collisions."""
        x,y,w,h = self.rect(d,time); o,t = d.options,d.type_id
        if t == 702:
            if self.icicle_age(d,time)<o.get('delay',.5): return []
            return [[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]]
        if t == 703:
            # Only the exposed half is hazardous; the other half is hidden
            # behind the opaque terrain, using the same tangent as rendering.
            angle=self.patrol_pose(d,time)[2]
            if angle==0: h/=2
            elif angle==90: x+=w/2;w/=2
            elif angle==180: y+=h/2;h/=2
            else: w/=2
            return [[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]]
        if t in (201,202,211,221):
            if t == 202: x += w*3/8; y += h*3/4; w /= 4; h /= 4
            elif t==201 and self.world.level.index==0 and any(
                    abs(q.rect_at(time,self.world.physics)[1]-(y+h))<.01
                    and q.rect_at(time,self.world.physics)[0]<x+w
                    and q.rect_at(time,self.world.physics)[0]+q.w>x for q in self.world.platforms):
                # Tutorial art overlaps the support snow by 22px. The exposed
                # teeth start at the same visible tip, not at an invisible
                # position above it; the hidden rocky foot stays in terrain.
                y+=22;h-=22
            n = o.get('teeth',3)
            return [[(x+i*w/n,y+h),(x+(i+.5)*w/n,y),(x+(i+1)*w/n,y+h)] for i in range(n)]
        if t in (502,503,504):
            angle = o.get('angle',90. if t == 504 else 0.)
            if t != 502: angle += o.get('angular_speed',90.)*time
            angle = math.radians(angle)
            cx,cy = x+w/2,y+h/2
            length,thick = o.get('length',192.),o.get('thickness',8.)
            ex,ey = cx+math.cos(angle)*length,cy+math.sin(angle)*length
            nx,ny = -math.sin(angle)*thick/2,math.cos(angle)*thick/2
            return [[(cx-nx,cy-ny),(ex-nx,ey-ny),(ex+nx,ey+ny),(cx+nx,cy+ny)]]
        if t == 303:
            if (time+o.get('phase',0.))%o.get('period',2.) >= o.get('active_for',1.): return []
            points=self.spark_outline(d,time)
            center=(x+w/2,y+h/2)
            return [[center,points[i],points[(i+1)%len(points)]] for i in range(len(points))]
        if t in (601,602,604,605):
            cx,cy = x+w/2,y+h/2
            if t == 601:
                vx,vy = o.get('velocity',(-128.,64.))
                cx += vx*time; cy += vy*time
            # A diamond is the actual sharp hit shape, not the icon's bounding box.
            return [[(cx,y if t != 601 else cy-h/2),(cx+w/2,cy),(cx,cy+h/2),(cx-w/2,cy)]]
        return [[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]]

    def spark_outline(self,d,time):
        x,y,w,h=self.rect(d,time)
        return [(x+w/2+math.cos(-math.pi/2+i*math.pi/8)*w*(.5 if i%2==0 else .23),
                 y+h/2+math.sin(-math.pi/2+i*math.pi/8)*h*(.5 if i%2==0 else .23)) for i in range(16)]

    def candidates(self,start,delta,duration):
        w,p = self.world,self.world.physics
        time = w.time
        result = []
        for d in self.devices.values():
            if not self.enabled(d): continue
            t,o,s = d.type_id,d.options,self.state[d.id]
            # Large image maps contain long spike/spark grids. Stationary
            # shapes outside the complete swept bounds cannot be contacts.
            if t in (303,201,202,211,221,306,701):
                end=(start[0]+delta[0],start[1]+delta[1])
                margin=p.radius+1.
                if (max(start[0],end[0])+margin<d.x or min(start[0],end[0])-margin>d.x+d.w
                        or max(start[1],end[1])+margin<d.y or min(start[1],end[1])-margin>d.y+d.h):
                    continue
            if t in (407,408,410):
                if time < self.teleport_until: continue
                rect = self.rect(d,time)
                hit = sweep_circle_rect(start,delta,rect,p.radius)
                fraction = 0. if circle_rect_overlap(*start,p.radius,rect) else hit[0] if hit else None
                partner = self.partner(d)
                if fraction is not None and partner and self.enabled(partner): result.append((fraction,1,'special',('teleport',d.id,None)))
                continue
            if t == 306:
                x,y,width,height = self.rect(d,time)
                near = circle_rect_overlap(*start,p.radius+1.,(x,y,width,height))
                if near and s['triggered'] is None: s['triggered'] = time
                if s['triggered'] is None: continue
                delay = o.get('delay',.5 if o.get('color','red') == 'blue' else .85)
                if time-s['triggered'] < delay: continue
                effect = 'repel' if o.get('color','red') == 'blue' else 'kill'
            elif t in (303,201,202,211,221,502,503,504,601,602,604,605,606,701,702,703):
                effect = 'repel' if t in (502,503,504) and o.get('color','red') == 'blue' else 'kill'
            else: continue
            if effect == 'repel' and time-s['last_effect'] < .15: continue
            hits = []
            # Relative swept translation plus midpoint/end samples for rotation.
            times = (time,time+duration/2,time+duration) if t in (503,504,703) else (time,)
            for sample_time in times:
                polys = self.polygons(d,sample_time)
                now_rect,next_rect = self.rect(d,time),self.rect(d,time+duration)
                mx,my = next_rect[0]-now_rect[0],next_rect[1]-now_rect[1]
                if t==702 and self.icicle_age(d,time+duration)<self.icicle_age(d,time):
                    # Respawning is not an upward-moving collision sweep.
                    mx=my=0.
                if t == 601:
                    velocity = o.get('velocity',(-128.,64.)); mx,my = velocity[0]*duration,velocity[1]*duration
                for poly in polys:
                    relative = (delta[0]-mx,delta[1]-my)
                    radius = p.radius+(.1 if t == 306 else 0.)
                    xs,ys=zip(*poly)
                    if not swept_bounds_overlap(start,relative,(min(xs),min(ys),max(xs),max(ys)),radius):
                        continue
                    hit = sweep_circle_polygon(start,relative,poly,radius)
                    if circle_polygon_overlap(*start,radius,poly): hits.append((0.,overlap_normal(start,poly)))
                    elif hit: hits.append(hit)
            if hits:
                fraction,normal = min(hits,key=lambda hit:hit[0])
                if t == 606:
                    rect = self.rect(d,time)
                    if w.ball.vy > 0 and start[1]+p.radius <= rect[1]+1e-5 and normal[1] < -.5:
                        result.append((fraction,1,'special',('stomp',d.id,normal))); continue
                if effect == 'kill': result.append((fraction,0,'fail',CATALOG[t][0]+'에 닿았어요'))
                else: result.append((fraction,1,'special',('repel',d.id,normal)))
        for shot in self.projectiles:
            cx,cy = shot.center(time)
            radius = shot.radius
            poly = [(cx-radius,cy-radius),(cx+radius,cy-radius),(cx+radius,cy+radius),(cx-radius,cy+radius)]
            hit = sweep_circle_polygon(start,(delta[0]-shot.vx*duration,delta[1]-shot.vy*duration),poly,p.radius)
            if circle_polygon_overlap(*start,p.radius,poly): hit = (0.,(0.,-1.))
            if hit: result.append((hit[0],0,'fail','대포 발사체에 닿았어요'))
        return result

    def partner(self,d):
        if d.target: return self.devices.get(d.target)
        return next((q for q in self.devices.values() if q.id != d.id and q.group and q.group == d.group and q.type_id in (407,408,410)),None)

    def resolve(self,payload):
        effect,name,normal = payload
        d,s,w = self.devices[name],self.state[name],self.world
        b,p = w.ball,w.physics
        if effect == 'teleport':
            other = self.partner(d)
            if other is None or not self.enabled(other): return
            angle = math.radians(other.options.get('exit_angle',-90.))
            x,y,width,height = self.rect(other,w.time)
            ux,uy=math.cos(angle),math.sin(angle)
            distance=min((width/2+p.radius+2)/abs(ux) if abs(ux)>1e-8 else math.inf,
                         (height/2+p.radius+2)/abs(uy) if abs(uy)>1e-8 else math.inf)
            exit_x = x+width/2+ux*distance
            exit_y = y+height/2+uy*distance
            blocked = any(q.active(w.time,w.switches) and circle_rect_overlap(exit_x,exit_y,p.radius,q.rect_at(w.time,p)) for q in w.platforms)
            self.teleport_until = w.time+.35
            if blocked or not (p.radius <= exit_x <= p.width-p.radius and p.radius <= exit_y <= p.height-p.radius):
                w.events.append(('teleport_blocked',name)); return
            if b.forward_target_x is not None:
                b.forward_target_x+=exit_x-b.x
            b.x,b.y = exit_x,exit_y
            w.events.append(('teleport',(name,other.id)))
        elif effect == 'stomp':
            s['destroyed'] = True
            b.forward_target_x = None
            b.y = self.rect(d,w.time)[1]-p.radius-1e-4
            b.vy = -p.bounce_speed; b.launch = False
            w.events.append(('stomp',name))
            w.events.append(('bounce',name))
        else:
            s['last_effect'] = w.time
            nx,ny = normal or (0.,-1.)
            speed = d.options.get('repel_speed',p.pad_speed)
            b.vx,b.vy = nx*speed,ny*speed
            b.x += nx*.01; b.y += ny*.01
            b.launch = False
            b.forward_target_x = None
            # Retain the horizontal impulse for a short, explicit period.
            if abs(nx) > .1:
                b.boost_direction = 1 if nx > 0 else -1
                b.boost_until = w.time+.15
            w.events.append(('repel',name))
            w.events.append(('bounce',name))

    def observe(self):
        return [{'id':d.id,'type_id':d.type_id,'name':CATALOG[d.type_id][0],
                 'rect':self.rect(d,self.world.time),'enabled':self.enabled(d),
                 'group':d.group,'target':d.target,'options':dict(d.options),
                 'state':dict(self.state[d.id]),
                 'polygons':self.polygons(d,self.world.time) if d.type_id not in SOLID else []}
                for d in self.devices.values()]
