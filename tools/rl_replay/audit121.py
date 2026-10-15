"""Privileged evaluator. Never passed to mapper or teacher."""
import numpy as np
from heightmap116 import R

class Audit:
    def __init__(self,n):
        self.cells=[{} for _ in range(n)];self.sums={};self.counts={}
        self.provenance=[];self.confirmed=[];self.stamps=[]
    def add(self,name,values):
        values=np.asarray(values,dtype=float)
        self.sums[name]=self.sums.get(name,0.)+float(values.sum())
        self.counts[name]=self.counts.get(name,0)+values.size
    def reset(self,ids):
        for e in ids:self.cells[e]={}
    def deliver(self,ids,labels,stamp):
        for e in ids:
            cells=self.cells[e]
            for a in labels[e]:
                xy=np.where(a['gap'][...,None],a['projected'][...,:2],a['points'][...,:2])
                usable=a['hit']&np.isfinite(xy).all(-1)
                keys=np.floor(xy[usable]/R.resolution+1e-7).astype(int)
                flags=np.stack([a['gap'],~a['gap'],a['ir_observable'],a['bright']&a['ir_observable']&a['gap'],a['dark']&a['ir_observable']&~a['gap']&a['horizontal'],a['surviving']],-1)[usable]
                unique,inverse=np.unique(keys,axis=0,return_inverse=True);merged=np.zeros((len(unique),6),bool)
                np.logical_or.at(merged,inverse,flags)
                for key,flag in zip(unique,merged):
                    key=tuple(key);old=cells.get(key)
                    if old is not None and stamp-old[0]<=1.:flag=flag|old[1]
                    cells[key]=(stamp,flag)
            self.cells[e]={key:value for key,value in cells.items() if stamp-value[0]<=1.}
    def sample(self,maps,packed,poses,gt,enabled,stamp,family):
        self.provenance.append(maps.provenance.copy());self.confirmed.append(maps.confirmed.copy());self.stamps.append(stamp)
        h,v=packed[:,:187],packed[:,187:]>0
        predicted=maps.confirmed|(v&(h>=4.999))
        gap=gt[:,374:561]>0;known=gt[:,187:374]>0
        self.add('gap_recall',predicted[enabled[:,None]&known&gap])
        self.add('false_gap_stairs',predicted[enabled[:,None]&known&~gap] if family!='gap' else [])
        for e in np.flatnonzero(enabled):
            c,s=np.cos(poses[e,2]),np.sin(poses[e,2]);xy=np.c_[poses[e,0]+c*maps.x-s*maps.y,poses[e,1]+s*maps.x+c*maps.y]
            for j,key in enumerate(np.floor(xy/R.resolution+1e-7).astype(int)):
                record=self.cells[e].get(tuple(key))
                if record is None or stamp-record[0]>1.:continue
                g,ng,observable,bright,dark,filled=record[1];mixed=g and ng
                if filled:self.add('false_fill_boundary_mixed',[mixed])
                if mixed or not known[e,j]:continue
                if gap[e,j] and observable:self.add('gap_recall_observable',[predicted[e,j]])
                if gap[e,j] and bright:self.add('bright_gap_miss',[not predicted[e,j]])
                if family=='gap' and not gap[e,j] and dark:self.add('false_gap_dark_flat',[predicted[e,j]])
                if gap[e,j] and filled:
                    self.add('false_fill_correction',[predicted[e,j]])
                    self.add('false_fill_inferred_correction',[maps.confirmed[e,j]])
                    self.add('false_fill_unknown',[not v[e,j]])
    def result(self):
        names=['gap_recall','gap_recall_observable','false_gap_stairs','false_gap_dark_flat','bright_gap_miss','false_fill_correction','false_fill_inferred_correction','false_fill_unknown','false_fill_boundary_mixed']
        return {k:dict(n=int(self.counts.get(k,0)),numerator=self.sums.get(k,0.),mean=self.sums[k]/self.counts[k] if self.counts.get(k,0) else None) for k in names}
    def save(self,path):
        np.savez_compressed(path,stamp=np.array(self.stamps),provenance=np.array(self.provenance),confirmed=np.array(self.confirmed))
