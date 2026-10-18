"""gdm-like commands from scene metadata; no geometric oracle fields enter actor features."""
import math
import random
from dataclasses import replace
from .commands import Conditions, RouteHint, NavCommand, CommandAdapter, policy_features
from . import schema

def route_projection(polyline, xy):
    """Nearest polyline segment -> route s, signed lateral error, yaw, remaining length."""
    best=None;offset=0.;total=sum(math.dist(a,b) for a,b in zip(polyline,polyline[1:]))
    for a,b in zip(polyline,polyline[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length == 0: continue
        u=max(0.,min(1.,((xy[0]-a[0])*dx+(xy[1]-a[1])*dy)/(length*length)))
        px,py=a[0]+u*dx,a[1]+u*dy
        d=math.hypot(xy[0]-px,xy[1]-py)
        if best is None or d < best[0]:
            best=(d,offset+u*length,(-dy*(xy[0]-px)+dx*(xy[1]-py))/length,math.atan2(dy,dx))
        offset+=length
    if best is None: raise ValueError('route needs a nonzero segment')
    return best[1],best[2],best[3],total-best[1]

def route_hint(tile, xy):
    s,offset,yaw,remaining=route_projection(tile.route_polyline,xy)
    segment=len(tile.segments)-1
    for i,(_,start,end) in enumerate(tile.segments):
        if start <= s <= end: segment=i;break
    kind,start,end=tile.segments[segment]
    following=tile.segments[min(segment+1,len(tile.segments)-1)][0]
    return RouteHint(segment,kind,yaw,max(0.,end-s),following),s,offset,remaining

class CommandGenerator:
    def __init__(self, tiles, cfg, command_cfg, *, seed):
        self.tiles=list(tiles);self.cfg=cfg;self.command_cfg=command_cfg
        self.rng=[random.Random(seed+i) for i in range(len(tiles))]
        self.adapters=[None]*len(tiles);self.commands=[None]*len(tiles)
        self.task=['']*len(tiles);self.target=[None]*len(tiles)
        self.epoch=[0]*len(tiles);self.seq=[0]*len(tiles);self.elapsed=[0.]*len(tiles)
        self.previous_toggle=[0]*len(tiles)
        self.previous_resample=[0]*len(tiles)

    def _sample_conditions(self,i):
        r=self.rng[i]
        if r.random()<self.cfg['commands']['default_conditions_fraction']:
            return Conditions(**self.command_cfg['defaults'])
        return Conditions(*(r.uniform(*self.command_cfg[k+'_range']) for k in schema.CONDITION_KEYS))

    def reset(self,env_ids,*,now_ms,yaws,tiles=None):
        if tiles is not None:
            for i,tile in zip(env_ids,tiles):self.tiles[int(i)]=tile
        target=self.cfg['commands']['task_probabilities'];mix=self.cfg['scene_mix']
        for j,i in enumerate(env_ids):
            i=int(i);r=self.rng[i];kind=self.tiles[i].kind
            if kind=='legacy':task='legacy'
            elif kind=='nav_turn_room':task='turn'
            elif kind=='nav_stair_landing':
                task='stairs' if r.random()<target['stairs']/mix[kind] else 'stop_start'
            else:
                pass_fraction=target['passage']/(mix['nav_corridor']+mix['nav_door'])
                task='passage' if r.random()<pass_fraction else 'stop_start'
            self.task[i]=task;self.elapsed[i]=0.;self.previous_toggle[i]=0;self.previous_resample[i]=0
            self.epoch[i]+=1;self.seq[i]+=1
            c=self._sample_conditions(i);self.target[i]=c
            mode='LEGACY' if task=='legacy' else 'TURN' if task=='turn' else 'NAV'
            self.adapters[i]=CommandAdapter(self.command_cfg,initial_yaw=float(yaws[j]))
            self.commands[i]=NavCommand(self.seq[i],now_ms,mode,r.random()<self.cfg['commands']['go_probability'],
                r.uniform(*self.command_cfg['v_cap_range']),-r.uniform(self.command_cfg['turn_cap_range'][0],c.turn_cap),c,
                self.epoch[i],self.epoch[i],self.seq[i])

    def tick(self,env_id,*,now_ms,dt,xy,stable,hold=False,fsm_go=True,phase='APPROACH',dwell_progress=0.,legacy=(0.,0.,0.)):
        i=int(env_id);r=self.rng[i];self.elapsed[i]+=dt
        cmd=self.commands[i]
        if cmd is None:raise RuntimeError('reset generator before tick')
        if r.random()<self.cfg['commands']['condition_change_probability']*dt:
            self.target[i]=self._sample_conditions(i)
        values=[]
        for key in schema.CONDITION_KEYS:
            old=getattr(cmd.c,key);goal=getattr(self.target[i],key)
            step=self.cfg['commands']['condition_rate_per_s'][key]*dt
            new=old+max(-step,min(step,goal-old))
            if key=='margin' and new<old and not stable:new=old
            values.append(new)
        hint,s,offset,remaining=route_hint(self.tiles[i],xy)
        landing_id=next((j for j,(k,_,_) in enumerate(self.tiles[i].segments) if k=='LANDING'),None)
        go=cmd.requested_go
        interval=int(self.elapsed[i]/self.cfg['commands']['resample_s'])
        mode=cmd.requested_mode
        if interval != self.previous_resample[i]:
            go=r.random()<self.cfg['commands']['go_probability']
            if self.task[i]=='stop_start':mode='TURN' if mode=='NAV' else 'NAV'
            cmd=replace(cmd,v_cap=r.uniform(*self.command_cfg['v_cap_range']),
                        turn_rate=-r.uniform(self.command_cfg['turn_cap_range'][0],cmd.c.turn_cap),
                        manoeuvre_id=cmd.manoeuvre_id+1,requested_mode=mode)
        self.previous_resample[i]=interval
        toggle=int(self.elapsed[i]/self.cfg['commands']['stop_start_toggle_s'])
        if self.task[i]=='stop_start' and toggle != self.previous_toggle[i]:go=not go
        self.previous_toggle[i]=toggle
        self.seq[i]+=1
        cmd=replace(cmd,seq=self.seq[i],heartbeat=self.seq[i],timestamp_ms=now_ms,c=Conditions(*values),
                    requested_go=go,route_hint=hint,landing_id=landing_id)
        self.commands[i]=cmd
        adapter=self.adapters[i];adapter.accept(cmd,now_ms=now_ms,stationary=stable)
        brake=None
        if phase=='PRE_BRAKE':
            brake=min(cmd.v_cap,math.sqrt(2*self.cfg['commands']['brake_deceleration_mps2']*hint.dist_to_next_m))
        applied=adapter.tick(now_ms=now_ms,stable=stable,hold=hold,fsm_go=fsm_go,pre_brake_v=brake,legacy=legacy)
        return cmd,applied,policy_features(applied,adapter.c,phase,dwell_progress),{'route_s':s,'offset':offset,'remaining':remaining}
