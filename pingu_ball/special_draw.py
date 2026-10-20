"""Winter-styled device symbols and collision-matched hazard rendering."""
import math
import pygame
from .specials import SOLID

CYAN=(47,222,250)
GOLD=(255,184,35)
RED=(248,75,78)
WHITE=(225,250,255)
INK=(20,61,93)


def arrow(surface,start,end,color=GOLD,width=5):
    pygame.draw.line(surface,INK,start,end,width+3)
    pygame.draw.line(surface,color,start,end,width)
    angle=math.atan2(end[1]-start[1],end[0]-start[0])
    points=[end,(end[0]-13*math.cos(angle-.6),end[1]-13*math.sin(angle-.6)),
                (end[0]-13*math.cos(angle+.6),end[1]-13*math.sin(angle+.6))]
    pygame.draw.polygon(surface,INK,points)
    pygame.draw.polygon(surface,color,points)


def tile_image(skin,block,time,switches):
    d,runtime=block.device,block.runtime
    t,s,o=d.type_id,runtime.state[d.id],d.options
    if block.destroyed: return None
    if t in (301,401,302,501,415,416,402):
        return skin.base_platform(block,time,switches)
    active=block.active(time,switches)
    if not active:
        if t==305:
            image=pygame.Surface((64,64),pygame.SRCALPHA)
            pygame.draw.rect(image,CYAN,(3,3,58,58),3,border_radius=6)
            return pygame.transform.smoothscale(image,(round(block.w),round(block.h)))
        return None
    image=skin.image('normal').copy()
    if t==403: arrow(image,(32,22),(32,50),GOLD)
    elif t==307:
        pygame.draw.arc(image,GOLD,(12,19,40,35),math.pi,math.tau,5)
        arrow(image,(31,42),(45,27),GOLD,4)
    elif t==508:
        return skin.image('moving_vertical',(block.w,block.h))
    elif t==507:
        pygame.draw.rect(image,CYAN,(6,18,7,39),border_radius=2)
        for y in (24,36,48): pygame.draw.lines(image,WHITE,False,[(17,y-4),(23,y),(17,y+4)],2)
    elif t==511:
        pygame.draw.arc(image,RED,(16,21,32,31),math.pi,math.tau,10)
        pygame.draw.line(image,CYAN,(17,35),(17,48),9); pygame.draw.line(image,CYAN,(47,35),(47,48),9)
    elif t==512:
        arrow(image,(15,28),(49,28),CYAN,4); arrow(image,(49,44),(15,44),(167,108,250),4)
    elif t==304:
        on=switches.get(d.group,False)
        points=[(32+math.cos(-math.pi/2+i*math.pi/5)*(20 if i%2==0 else 9),
                 36+math.sin(-math.pi/2+i*math.pi/5)*(20 if i%2==0 else 9)) for i in range(10)]
        pygame.draw.polygon(image,GOLD if on else (117,146,158),points)
    elif t in (404,510):
        on=switches.get('power:'+d.group,False)
        pygame.draw.polygon(image,GOLD if on else CYAN,[(34,17),(20,38),(31,38),(27,53),(46,30),(35,30)])
        if t==510: image.fill((120,210,255,60),special_flags=pygame.BLEND_RGBA_ADD)
    elif t==305:
        pygame.draw.rect(image,INK,(13,18,38,41),border_radius=4)
        for x in (18,28,38,48): pygame.draw.line(image,CYAN,(x,18),(x,58),4)
        pygame.draw.circle(image,RED,(32,37),10); pygame.draw.line(image,WHITE,(32,32),(32,43),3)
    elif t==306:
        color=CYAN if o.get('color','red')=='blue' else RED
        pygame.draw.circle(image,color,(32,36),17)
        pygame.draw.line(image,WHITE,(32,25),(32,38),4); pygame.draw.circle(image,WHITE,(32,46),2)
    elif t in (405,406):
        count=4 if t==405 else 8
        for i in range(count):
            angle=-math.pi/2+i*math.tau/count
            arrow(image,(32+math.cos(angle)*8,36+math.sin(angle)*8),(32+math.cos(angle)*23,36+math.sin(angle)*23),GOLD,3)
        pygame.draw.circle(image,INK,(32,36),9); pygame.draw.circle(image,RED,(32,36),5)
    return pygame.transform.smoothscale(image,(round(block.w),round(block.h)))


def draw_movement_markers(surface, world):
    """Range endpoints are guides, never solid blocks or hazards."""
    color = (25, 63, 86, 220)
    for x, y in world.level.movement_markers:
        x, y = round(x), round(y)
        for i in range(3, 61, 11):
            for a, b in (((x+i,y+3),(x+min(i+6,61),y+3)),
                         ((x+i,y+61),(x+min(i+6,61),y+61)),
                         ((x+3,y+i),(x+3,y+min(i+6,61))),
                         ((x+61,y+i),(x+61,y+min(i+6,61)))):
                pygame.draw.line(surface, color, a, b, 2)
        pygame.draw.circle(surface, color, (x+32,y+32), 10, 2)


def draw_patrols(surface,world,skin):
    """Draw enemies behind opaque terrain so only the outside portion shows."""
    for d in world.special.devices.values():
        if d.type_id!=703 or not world.special.enabled(d): continue
        cx,cy,angle=world.special.patrol_pose(d,world.time)
        size=d.options.get('size',56.)
        image=pygame.transform.rotate(skin.image('sea_lion',(size,size)),-angle)
        surface.blit(image,image.get_rect(center=(round(cx),round(cy))))


def draw_runtime(surface,world,skin,platform_rects=None):
    runtime=world.special
    if platform_rects is None:
        platform_rects=[q.rect_at(world.time,world.physics) for q in world.platforms]
    for d in runtime.devices.values():
        t=d.type_id
        if t in SOLID or t==703 or runtime.state[d.id]['destroyed']: continue
        x,y,w,h=runtime.rect(d,world.time)
        enabled=runtime.enabled(d)
        if t in (701,702) and enabled:
            image=skin.image('sun' if t==701 else 'icicle',(w,h))
            if t==702 and runtime.icicle_age(d,world.time)<d.options.get('delay',.5):
                image=image.copy();image.set_alpha(90)
            surface.blit(image,(round(x),round(y)))
        elif t in (201,202,221) and enabled:
            if t==202:
                x+=w*3/8;y+=h*3/4;w/=4;h/=4
            # Tuck the rocky foot into the snow cap below it. The PNG's
            # rounded edges otherwise leave a visible seam at touching tiles.
            support=any(abs(top-(y+h))<.01 and left<x+w and left+width>x
                        for left,top,width,_ in platform_rects)
            overlap=22. if support and t!=202 else min(4.,h/4) if support else 0.
            image=skin.image('spike',(w,h))
            if t==221:
                image=image.copy()
                image.fill((35,35,35,0),special_flags=pygame.BLEND_RGB_ADD)
            surface.blit(image,(round(x),round(y+overlap)))
        elif t in (407,408,410):
            color=(70,151,255) if t==407 else (180,250,255) if t==408 else (153,99,246)
            if not enabled: color=(80,106,124)
            center=(round(x+w/2),round(y+h/2))
            pygame.draw.ellipse(surface,INK,(x,y,w,h))
            for scale in (.95,.7,.45):
                rect=pygame.Rect(0,0,round(w*scale),round(h*scale)); rect.center=center
                pygame.draw.ellipse(surface,color,rect,3)
            angle=world.time*3
            end=(center[0]+math.cos(angle)*w*.37,center[1]+math.sin(angle)*h*.37)
            pygame.draw.line(surface,WHITE,center,end,3)
        elif t==603:
            surface.blit(skin.image('normal',(w,h)),(x,y))
            angle=math.radians(d.options.get('angle',180.))
            center=(x+w/2,y+h/2)
            tip=(center[0]+math.cos(angle)*w*.48,center[1]+math.sin(angle)*h*.48)
            pygame.draw.line(surface,INK,center,tip,19); pygame.draw.line(surface,CYAN if enabled else (100,125,140),center,tip,11)
        else:
            polygons=runtime.polygons(d,world.time) if enabled else []
            if not polygons:
                pygame.draw.rect(surface,(95,138,160),(x+8,y+8,w-16,h-16),2,border_radius=5)
            if t==303 and polygons:
                points=runtime.spark_outline(d,world.time)
                pygame.draw.polygon(surface,(255,124,35),points)
                pygame.draw.lines(surface,(255,225,135),True,points,2)
                pygame.draw.circle(surface,(255,241,173),(round(x+w/2),round(y+h/2)),max(2,round(min(w,h)*.1)))
                continue
            layer=pygame.Surface(surface.get_size(),pygame.SRCALPHA) if t==211 else surface
            for poly in polygons:
                color=CYAN if d.options.get('color','red')=='blue' else RED if t in (303,502,503,504,602,606) else (146,220,248)
                if t==211: color=(*CYAN,90)
                pygame.draw.polygon(layer,color,poly)
                pygame.draw.lines(layer,WHITE if t!=211 else (*WHITE,150),True,poly,2)
                if t in (502,503,504):
                    pygame.draw.circle(layer,INK,(round(x+w/2),round(y+h/2)),9)
            if layer is not surface: surface.blit(layer,(0,0))
            if t==606 and enabled:
                surface.blit(skin.image('angry',(w,h)),(x,y))
            if t==221:
                for poly in polygons: pygame.draw.lines(surface,(245,253,255),True,poly,4)
    for shot in runtime.projectiles:
        cx,cy=shot.center(world.time); radius=shot.radius
        pygame.draw.rect(surface,RED,(cx-radius,cy-radius,radius*2,radius*2))
        pygame.draw.line(surface,WHITE,(cx-radius,cy),(cx+radius,cy),2)
