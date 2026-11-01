"""Matched tutorial-only input ablation. No routes, map edits or teacher actions."""
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .all_tutorial_env import (AllTutorialEnv, GLOBAL_DIM, ENTITY_DIM,
                               maximum_entities, distance_feature)

VARIANTS=("existing","local")
LOCAL_COUNT=12
MOTION_DIM=4
LOCAL_DIM=ENTITY_DIM+MOTION_DIM


class MapInputEnv(AllTutorialEnv):
    def __init__(self,stage=1,variant="existing",limit=40.,repeat=12,novelty=.03):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant
        super().__init__(stage,limit,repeat,novelty,reward_scale=.1)
        # Identical six input requests on all ten tasks and in both variants.
        # Jump/dash requests only use the corresponding currently held item.
        self.action_space=spaces.Discrete(6)
        if variant=="local":
            self.observation_space=spaces.Dict({
                **self.observation_space.spaces,
                "motion":spaces.Box(-8.,8.,(self.max_entities,MOTION_DIM),dtype=np.float32),
                "local":spaces.Box(-8.,8.,(LOCAL_COUNT,LOCAL_DIM),dtype=np.float32)})

    def entity_records(self):
        """Raw world geometry, in exactly the same order as the full observation."""
        w=self.world;records=[]
        for i,(x,y,width,height,kind) in enumerate(self.terrain):
            records.append(dict(id=f"terrain-{i}",kind=kind,rect=(x,y,width,height),
                                active=True,endpoints=None))
        for d in w.special.devices.values():
            q=w.special.platforms.get(d.id)
            rect=w.special.rect(d,w.time)
            if d.type_id==201:rect=self.static_hazard_bounds[d.id]
            elif d.type_id==703:
                points=[p for polygon in w.special.polygons(d,w.time) for p in polygon]
                xs,ys=zip(*points);rect=(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
            active=q.active(w.time,w.switches) if q else w.special.enabled(d)
            if d.type_id==702:active=active and w.special.icicle_age(d,w.time)>=d.options.get('delay',.5)
            endpoints=None
            if d.type_id in (501,508):
                travel=d.options.get('travel',128.)
                endpoints=((d.x,d.y),(d.x+(travel if d.type_id==501 else 0),
                                        d.y+(travel if d.type_id==508 else 0)))
            records.append(dict(id=d.id,kind=d.type_id,rect=rect,active=bool(active),endpoints=endpoints))
        for i,item in enumerate(self.level.powerups):
            records.append(dict(id=f'item-{i}',kind=item.kind,rect=item.rect,
                                active=i not in w.collected_powerups,endpoints=None))
        return records

    def local_indices(self,records=None):
        b=self.world.ball;records=self.entity_records() if records is None else records
        def separation(i):
            x,y,width,height=records[i]['rect']
            dx=max(x-b.x,0,b.x-x-width);dy=max(y-b.y,0,b.y-y-height)
            return math.hypot(dx,dy),i
        return sorted(range(len(records)),key=separation)[:LOCAL_COUNT]

    def observe(self):
        obs=super().observe()
        if self.variant=="existing":return obs
        records=self.entity_records();b=self.world.ball
        motion=np.zeros((self.max_entities,MOTION_DIM),np.float32)
        for i,row in enumerate(records):
            if row['endpoints'] is not None:
                (x1,y1),(x2,y2)=row['endpoints']
                motion[i]=[distance_feature(x1-b.x),distance_feature(y1-b.y),
                           distance_feature(x2-b.x),distance_feature(y2-b.y)]
        local=np.zeros((LOCAL_COUNT,LOCAL_DIM),np.float32)
        for j,i in enumerate(self.local_indices(records)):
            local[j,:ENTITY_DIM]=obs['entities'][i]
            x,y,width,height=records[i]['rect']
            # Linear, sub-tile coordinates for precise nearby landing relations.
            local[j,:6]=[(x-b.x)/256,(x+width-b.x)/256,
                          (y-b.y)/256,(y+height-b.y)/256,width/64,height/64]
            local[j,ENTITY_DIM:]=motion[i]
        obs['motion']=np.clip(motion,-8,8)
        obs['local']=np.clip(local,-8,8)
        return obs

    def potential(self):
        # No straight-line goal-distance shaping in either comparison arm.
        return 0.

    def step(self,action,on_tick=None):
        if self.done:raise RuntimeError("Reset before stepping a terminal episode")
        direction,use,power_direction=self.decode_action(action)
        before=self.ticks;previous_pickups=len(self.world.collected_powerups);power_uses=0
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
        cell=self.cell();new=not self.visited[cell];self.visited[cell]=1
        credit=self.novelty*new if not self.world.failed else 0.
        bonus=min(credit,max(0.,1-self.bonus_credit));self.bonus_credit+=bonus
        self.pickups+=len(self.world.collected_powerups)-previous_pickups;self.power_uses+=power_uses
        reward=(base+bonus)*self.reward_scale
        self.raw_base_return+=base;self.bonus_return+=bonus;self.episode_return+=reward
        return self.observe(),float(reward),self.done,False,self.info(self.ticks-before)


class FlatMapInputEnv(gym.ObservationWrapper):
    def __init__(self,**options):
        super().__init__(MapInputEnv(**options))
        size=GLOBAL_DIM+maximum_entities()*ENTITY_DIM
        if self.unwrapped.variant=='local':size+=maximum_entities()*MOTION_DIM+LOCAL_COUNT*LOCAL_DIM
        self.observation_space=spaces.Box(-8.,8.,(size,),dtype=np.float32)

    def observation(self,obs):
        parts=[obs['global'],obs['entities'].reshape(-1)]
        if self.unwrapped.variant=='local':parts.extend((obs['motion'].reshape(-1),obs['local'].reshape(-1)))
        return np.concatenate(parts).astype(np.float32,copy=False)
