"""Compact E1 per-approach metrics; UPDATE21 swing definitions reused verbatim."""
import numpy as np
import math
def segments(c):
    lo=np.flatnonzero(c[:-1]&~c[1:])+1
    td=np.flatnonzero(~c[:-1]&c[1:])+1
    return [(int(k),int(td[np.searchsorted(td,k)])) if np.searchsorted(td,k)<len(td) else (int(k),None) for k in lo]

def attempt(F,c,t,a,b,cut):
    """First near-deck-origin airborne episode reaching the near edge.
    Returned-near / censored attempts stay visible; no replacement by recovery.
    First-air sample is operational LO; preceding loaded sample brackets it.
    """
    entry=np.flatnonzero(F[:cut,0]>=a)
    if not len(entry):return {'status':'foot_not_reached'}
    first=int(entry[0])
    for k,j in segments(c[:cut]):
        stop=j if j is not None else cut-1
        if not (a-1<=F[k-1,0]<a and F[k-1,2]>=-.03):continue
        if F[k:stop+1,0].max()<a:continue
        e={'lo_i':k,'td_i':j,'entry_i':first,'late_after_entry':bool(t[k]-t[first]>.12),
           'lo_margin_cm':float(100*(a-F[k,0])),
           'lo_previous_margin_cm':float(100*(a-F[k-1,0])),
           'apex_cm':float(100*F[k:stop+1,2].max()),
           'apex_bottom_cm':float(100*(F[k:stop+1,2].max()-.03)),
           'lo_t':float(t[k]), 'duration_s':float(t[stop]-t[k])}
        if j is None:return dict(e,status='no_touchdown')
        if F[j,0]<a:return dict(e,status='returned_near')
        over=(F[k:j+1,0]>a)&(F[k:j+1,0]<b)
        margin=float(100*(F[j,0]-b))
        e.update(status='complete',td_margin_cm=margin,length_cm=float(100*(F[j,0]-F[k,0])),
                 td_z_cm=float(100*F[j,2]),td_t=float(t[j]),
                 short=bool(margin<=3+1e-9),inside=bool(F[j,0]<b),
                 dip=bool(np.any(F[k:j+1,2][over]<0)),
                 min_slot_z_cm=float(100*F[k:j+1,2][over].min()) if over.any() else None)
        assert abs(e['length_cm']-e['lo_margin_cm']-100*(b-a)-margin)<1e-8
        return e
    return {'status':'no_near_origin_swing','entry_i':first,'entry_in_contact':bool(c[first])}

def summarize_episode(rows,near,far,deck,foot_names,episode,env_id,termination,condition):
    if not rows:return None
    a=np.asarray(rows,dtype=float);t=a[:,0];x=a[:,1];height=a[:,2];p=a[:,3];roll=a[:,4]
    feet=a[:,5:17].reshape(-1,4,3).copy();feet[:,:,2]-=deck
    contact=a[:,17:21]>0;front=[i for i,n in enumerate(foot_names) if n.upper().startswith('F')]
    if len(front)!=2:raise ValueError('expected FR/FL foot names')
    approached=np.any(x>=near-1.5)
    if not approached:return None
    crossed=np.flatnonzero((x>=far+.8)&(height-deck>.3)&(np.maximum(np.abs(p),np.abs(roll))<.5))
    cut=int(crossed[0])+1 if len(crossed) else len(a)
    # Stop first-approach metrics at failure/stall; never count failure dwell.
    stop_reason=None
    progress=np.maximum.accumulate(x)
    for i in range(cut):
        if x[i]<near-1.5:continue
        if height[i]-deck<.3 or max(abs(p[i]),abs(roll[i]))>.5:
            cut=i+1;stop_reason="fall";break
        j=int(np.searchsorted(t,t[i]+2.))
        if j<len(t) and progress[j]-progress[i]<.05:
            cut=i+1;stop_reason="stall";break
    crossed=crossed[crossed<cut]
    feet=feet[:cut];contact=contact[:cut];t=t[:cut];x=x[:cut];p=p[:cut]
    flat=(x>=near-2.5)&(x<near-1.3)
    base=float(np.degrees(p[flat]).mean()) if flat.any() else None
    changes={}
    for name,lo,hi in [('pre_1p0_0p7',-1.,-.7),('pre_0p7_0p4',-.7,-.4),('edge_0p2_0',-.2,0.)]:
        vals=np.degrees(p[(x-near>=lo)&(x-near<hi)])
        changes[name]={'n':len(vals),'mean_absolute_deg':float(vals.mean()) if len(vals) else None,'mean_delta_deg':float(vals.mean()-base) if len(vals) and base is not None else None,'min_delta_deg':float(vals.min()-base) if len(vals) and base is not None else None}
    over=(feet[:,:,0]>near)&(feet[:,:,0]<far);drop=bool(np.any(over&(feet[:,:,2]<0)))
    swings=[dict(leg=foot_names[i],group='front' if i in front else 'hind',**attempt(feet[:,i],contact[:,i],t,near,far,cut)) for i in range(4)]
    touchdowns=[]
    for i in front:
        edge=np.flatnonzero(feet[:,i,0]>=near);limit=int(edge[0]) if len(edge) else cut
        td=np.flatnonzero(~contact[:-1,i]&contact[1:,i])+1
        td=td[(td<limit)&(feet[td,i,0]>=near-1)&(feet[td,i,0]<near)]
        if len(td):k=int(td[-1]);touchdowns.append({'leg':foot_names[i],'t':float(t[k]),'margin_m':float(near-feet[k,i,0])})
    return {'env_id':env_id,'episode_id':episode,'condition':condition,'width_m':far-near,'near':near,'far':far,
        'reached':bool(np.any(feet[:,front,0]>=near)),'crossed':bool(len(crossed)),'clean':bool(len(crossed)) and not drop,
        'stop_reason':stop_reason,'stall':stop_reason=='stall',
        'foot_drop_front':bool(np.any((over&(feet[:,:,2]<0))[:,front])),
        'foot_drop_hind':bool(np.any((over&(feet[:,:,2]<0))[:,[i for i in range(4) if i not in front]])),
        'foot_drop':drop,'dip_ge15cm':bool(np.any(over&(feet[:,:,2]<-.15))),
        'flat_pitch_deg':base,'pitch':changes,'crossing_swings':swings,
        'last_front_touchdown':max(touchdowns,key=lambda v:v['t']) if touchdowns else None,
        'last_recorded_base_x':float(a[-1,1]),'metric_cut_base_x':float(x[-1]),
        'termination':termination,'censored':termination=='window_end' and not bool(len(crossed)),
        'fall':bool(np.any((height[:cut]-deck<.3)|(np.maximum(np.abs(a[:cut,3]),np.abs(a[:cut,4]))>.5))),
        'first_switch_step':int(a[a[:,21]>=0,21][0]) if np.any(a[:,21]>=0) else None}

class EpisodeMetrics:
    def __init__(self,n,foot_names,condition):
        self.rows=[[] for _ in range(n)];self.episodes=[0]*n;self.records=[]
        self.names=foot_names;self.condition=condition;self.geometry=[None]*n
    def add(self,packed,near,far,deck,done,timeout):
        for e,row in enumerate(packed):
            if self.geometry[e] is None:self.geometry[e]=(float(near[e]),float(far[e]),float(deck[e]))
            self.rows[e].append(row)
            if done[e]:self.close(e,'timeout' if timeout[e] else 'terminated')
    def close(self,e,reason):
        if self.geometry[e] is not None:
            rec=summarize_episode(self.rows[e],*self.geometry[e],self.names,self.episodes[e],e,reason,self.condition)
            if rec:self.records.append(rec)
        self.rows[e]=[];self.geometry[e]=None;self.episodes[e]+=1
    def finish(self):
        for e in range(len(self.rows)):self.close(e,'window_end')
        return self.records
