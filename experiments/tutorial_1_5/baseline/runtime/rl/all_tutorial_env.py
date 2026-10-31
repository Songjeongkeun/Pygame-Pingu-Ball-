"""Route-free, original-spawn Gymnasium tasks for all ten native tutorials.

No map changes, previous models, authored waypoints, teleports, or teacher input.
State contains every entity; no nearest-K truncation hides remote hazards.
"""
import math
from functools import lru_cache
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pingu_ball.engine import World
from pingu_ball.playfield import expand_playfield
from pingu_ball.special_lab import make_lab


KINDS=("normal","spring","breakable","timed","moving_x","moving_y","launcher",
       "spike","sun","icicle","sea_lion","jump_item","dash_item")
DEVICE_KIND={301:"breakable",401:"spring",302:"timed",501:"moving_x",508:"moving_y",
             402:"launcher",201:"spike",701:"sun",702:"icicle",703:"sea_lion"}
ENTITY_DIM=31
GRID=(32,16)
GLOBAL_DIM=24+GRID[0]*GRID[1]


def distance_feature(value):
    return math.copysign(math.log1p(abs(value)/64)/4,value)


def merge_rectangles(platforms):
    """Only exact rectangular unions of static normal terrain; no hole filling."""
    fixed=[];normal=[]
    for p in platforms:
        row=(p.x,p.y,p.w,p.h,p.kind)
        (normal if p.kind=="normal" and not p.travel else fixed).append(row)
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(normal):
            for j in range(i+1,len(normal)):
                b=normal[j];rect=None
                if a[1]==b[1] and a[3]==b[3] and (a[0]+a[2]==b[0] or b[0]+b[2]==a[0]):
                    rect=(min(a[0],b[0]),a[1],a[2]+b[2],a[3],"normal")
                elif a[0]==b[0] and a[2]==b[2] and (a[1]+a[3]==b[1] or b[1]+b[3]==a[1]):
                    rect=(a[0],min(a[1],b[1]),a[2],a[3]+b[3],"normal")
                if rect:
                    normal[i]=rect;normal.pop(j);changed=True;break
            if changed:break
    return tuple(sorted(normal+fixed,key=lambda r:(r[1],r[0],r[4])))


@lru_cache(maxsize=1)
def maximum_entities():
    return max(len(merge_rectangles(make_lab(s).platforms))+len(make_lab(s).specials)
               +len(make_lab(s).powerups) for s in range(1,11))


class AllTutorialEnv(gym.Env):
    metadata={"render_modes":[]}

    def __init__(self,stage=1,limit=40.,repeat=12,novelty=.03,reward_scale=.1):
        if stage not in range(1,11) or limit<=0 or repeat<1:
            raise ValueError("stage=1..10, positive limit and repeat required")
        self.stage,self.limit,self.repeat=stage,limit,repeat
        self.novelty,self.reward_scale=novelty,reward_scale
        self.level=expand_playfield(make_lab(stage))
        self.terrain=merge_rectangles(self.level.platforms)
        self.n_entities=len(self.terrain)+len(self.level.specials)+len(self.level.powerups)
        self.max_entities=maximum_entities()
        # First nine tasks do not contain usable items, so no redundant actions.
        self.action_space=spaces.Discrete(6 if self.level.powerups else 3)
        self.observation_space=spaces.Dict({
            "global":spaces.Box(-8.,8.,(GLOBAL_DIM,),dtype=np.float32),
            "entities":spaces.Box(-8.,8.,(self.max_entities,ENTITY_DIM),dtype=np.float32)})
        self.reset()

    def cell(self):
        x=int(np.clip(self.world.ball.x/self.level.width*GRID[0],0,GRID[0]-1))
        y=int(np.clip(self.world.ball.y/self.level.height*GRID[1],0,GRID[1]-1))
        return y*GRID[0]+x

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.world=World(self.level,auto_reset=False)
        self.done=False;self.previous_action=1;self.ticks=self.decisions=0
        self.raw_base_return=self.shaping_return=self.bonus_return=self.episode_return=0.
        self.bonus_credit=0.;self.visited=np.zeros(GRID[0]*GRID[1],np.float32)
        self.visited[self.cell()]=1.
        self.pickups=self.power_uses=0
        self.min_goal_distance=self.goal_distance()
        self.static_hazard_bounds={}
        for d in self.world.special.devices.values():
            if d.type_id==201:
                polys=self.world.special.polygons(d,0.)
                pts=[p for polygon in polys for p in polygon]
                xs,ys=zip(*pts)
                self.static_hazard_bounds[d.id]=(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
        return self.observe(),self.info()

    def goal_distance(self):
        b=self.world.ball;gx,gy,gw,gh=self.level.goal
        return math.hypot(b.x-gx-gw/2,b.y-gy-gh/2)

    def potential(self):
        return 0. if self.done else -2*self.goal_distance()/math.hypot(self.level.width,self.level.height)

    def decode_action(self,action):
        if not self.action_space.contains(action):raise ValueError(action)
        if int(action)<3:return int(action)-1,False,0
        b=self.world.ball
        if int(action)==3:return 0,b.power_up=="up",0
        direction=-1 if int(action)==4 else 1
        return direction,b.power_up=="forward",direction

    def entity(self,rect,kind,*,velocity=(0.,0.),active=True,travel=0.,arrow=0.,phase=0.,
               until=0.,available=True,destroyed=False,source=False):
        x,y,width,height=rect;b=self.world.ball
        flags=[float(kind==name) for name in KINDS]
        values=[distance_feature(x-b.x),distance_feature(x+width-b.x),
                distance_feature(y-b.y),distance_feature(y+height-b.y),width/512,height/256,
                velocity[0]/640,velocity[1]/640,float(active),travel/512,arrow,
                math.sin(phase*math.tau),math.cos(phase*math.tau),until/10,
                float(available),float(destroyed),float(source),1.]
        return values+flags

    def observe(self):
        w,b=self.world,self.world.ball;gx,gy,gw,gh=self.level.goal
        global_values=[b.x/self.level.width,b.y/self.level.height,b.vx/640,b.vy/640,
            distance_feature(gx+gw/2-b.x),distance_feature(gy+gh/2-b.y),
            float(b.launch),float(b.power_up=="up"),float(b.power_up=="forward"),
            distance_feature((b.forward_target_x or b.x)-b.x),float(b.forward_target_x is not None),
            b.forward_direction,max(0.,b.boost_until-w.time),b.boost_direction,
            max(0.,1-w.time/self.limit),self.previous_action/5,w._crush_time,
            w.special.speed_scale(),self.level.width/3000,self.level.height/1500,
            self.bonus_credit,self.pickups/2,self.power_uses/2,w.physics.radius/64]
        entities=[]
        for x,y,width,height,kind in self.terrain:
            entities.append(self.entity((x,y,width,height),kind))
        for d in w.special.devices.values():
            if d.type_id not in DEVICE_KIND:
                raise ValueError(f"Unsupported device {d.type_id}; observation must be extended")
            q=w.special.platforms.get(d.id);o=d.options
            rect=w.special.rect(d,w.time)
            nxt=w.special.rect(d,w.time+.001)
            velocity=((nxt[0]-rect[0])/.001,(nxt[1]-rect[1])/.001)
            active=q.active(w.time,w.switches) if q else w.special.enabled(d)
            phase=until=0.;travel=o.get("travel",0.);arrow=o.get("direction",0.)
            if d.type_id==201:rect=self.static_hazard_bounds[d.id]
            elif d.type_id==302:
                period=o.get("period",2.8);on=o.get("active_for",2.)
                age=(w.time+o.get("phase",0.))%period;phase=age/period
                until=on-age if age<on else period-age
            elif d.type_id in (501,508):
                travel=o.get("travel",128.);speed=o.get("speed",w.physics.platform_speed)
                phase=((w.time*speed+o.get("phase",0.))%(2*travel))/(2*travel)
            elif d.type_id==702:
                age=w.special.icicle_age(d,w.time);delay=o.get("delay",.5);period=o.get("period",4.)
                velocity=(0.,o.get("gravity",w.physics.gravity)*max(0.,age-delay))
                phase=age/period;active=active and age>=delay
                until=delay-age if age<delay else period-age
            elif d.type_id==703:
                left,top,width,height=o.get("path",(d.x,d.y,192.,64.))
                perimeter=2*(width+height);travel=perimeter
                phase=((w.time*o.get("speed",64.)+o.get("phase",0.))%perimeter)/perimeter
                arrow=w.special.patrol_pose(d,w.time)[2]/180
                polys=w.special.polygons(d,w.time)
                xs,ys=zip(*(p for poly in polys for p in poly))
                rect=(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
            entities.append(self.entity(rect,DEVICE_KIND[d.type_id],velocity=velocity,active=active,
                travel=travel,arrow=arrow,phase=phase,until=until,
                destroyed=q.destroyed if q else w.special.state[d.id]["destroyed"],
                source=b.launch_source==d.id))
        for i,item in enumerate(self.level.powerups):
            available=i not in w.collected_powerups
            entities.append(self.entity(item.rect,"jump_item" if item.kind=="up" else "dash_item",
                                        active=available,available=available))
        padded=np.zeros((self.max_entities,ENTITY_DIM),dtype=np.float32)
        padded[:len(entities)]=np.asarray(entities,dtype=np.float32)
        return {"global":np.clip(np.asarray(global_values+list(self.visited),np.float32),-8,8),
                "entities":np.clip(padded,-8,8)}

    def info(self,tick_delta=0):
        w=self.world
        return dict(stage=self.stage,success=bool(w.cleared),is_success=bool(w.cleared),
            reason="success" if w.cleared else w.last_failure if w.failed else "training_time_limit" if self.done else "",
            seconds=w.time,ticks=self.ticks,tick_delta=tick_delta,decisions=self.decisions,
            base_return=self.raw_base_return,shaping_return=self.shaping_return,bonus_return=self.bonus_return,
            episode_return=self.episode_return,goal_distance=self.goal_distance(),min_goal_distance=self.min_goal_distance,
            visited_cells=int(self.visited.sum()),pickups=self.pickups,power_uses=self.power_uses)

    def step(self,action,on_tick=None):
        if self.done:raise RuntimeError("Reset before stepping a terminal episode")
        direction,use,power_direction=self.decode_action(action)
        phi=self.potential();before=self.ticks;previous_pickups=len(self.world.collected_powerups)
        power_uses=0
        for i in range(self.repeat):
            dt=min(self.world.physics.dt,self.limit-self.world.time)
            if dt<1e-10 or self.world.failed or self.world.cleared:break
            self.world.step(direction,dt,use_power=use and i==0,power_direction=power_direction)
            power_uses+=sum(e[0]=="power_use" for e in self.world.events)
            self.ticks+=1
            self.min_goal_distance=min(self.min_goal_distance,self.goal_distance())
            if on_tick:on_tick(self.world,self.ticks,direction)
        self.previous_action=int(action);self.decisions+=1
        self.done=bool(self.world.cleared or self.world.failed or self.world.time>=self.limit-1e-9)
        base=10. if self.world.cleared else -1. if self.done else 0.
        shaping=self.potential()-phi # finite horizon gamma=1
        cell=self.cell();new=not self.visited[cell];self.visited[cell]=1
        pickups=len(self.world.collected_powerups)-previous_pickups
        credit=(self.novelty*new+.1*pickups) if not self.world.failed else 0.
        bonus=min(credit,max(0.,1-self.bonus_credit));self.bonus_credit+=bonus
        self.pickups+=pickups;self.power_uses+=power_uses
        reward=(base+shaping+bonus)*self.reward_scale
        self.raw_base_return+=base;self.shaping_return+=shaping;self.bonus_return+=bonus;self.episode_return+=reward
        return self.observe(),float(reward),self.done,False,self.info(self.ticks-before)


class FlatTutorialEnv(gym.ObservationWrapper):
    """Lossless packing for SB3's standard n-step replay buffer."""
    def __init__(self,**options):
        super().__init__(AllTutorialEnv(**options))
        size=GLOBAL_DIM+maximum_entities()*ENTITY_DIM
        self.observation_space=spaces.Box(-8.,8.,(size,),dtype=np.float32)

    def observation(self,obs):
        return np.concatenate([obs["global"],obs["entities"].reshape(-1)])
