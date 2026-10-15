"""UPDATE130 L0S: LiDAR point-structure gap judgement (pure numpy, CPU).

Per capture, valid non-self returns are binned in yaw-aligned lanes (5 cm wide, 2 cm bins forward).
Horizontal support segments are chained per lane. A return lying below the support extrapolated from
the preceding segment, and not on a horizontal tread at most STEP_MAX below, proves that its ray passed
below that support height from u_cross to the hit: no support at that height there (free space).
No-return beams are never used as evidence (UPDATE130/130B review). Gap vs stairs is not decided by the
step height between planes: a stair shadow ends on a horizontal tread within STEP_MAX, a gap does not
(vertical wall, deep floor, or nothing). Thresholds fixed a priori, not tuned on rollout results.
"""
import numpy as np
from heightmap116 import rotation

S=dict(lane_w=.05,lane_half=.55,bin_u=.02,u_max=1.6,surf_spread=.035,chain_dz=.02,min_bins=3,max_slope=.75,
       ref_reach=.8,slope_min_len=.2,mark_depth=.30,sup_ttl=2.,sup_res=.02,block_depth=.02,use_base_ref=False,min_bin_points=2,clus_gap=.04,clus_spread=.025,use_pca=False,dual_grid=True,dual_offset=.025,nrm_win=5,nrm_min=8,nrm_ext=.015,nrm_ratio=3.,plane_rms=.012,nz_level=.85,nz_vert=.4,memref_back=.04,step_max=.25,pending_depth=.03,gap_depth=.08,pending_min_points=2,
       retract_band=.02,retract_points=3,retract_margin=.01,d_deep=.5,hind_pass_x=-.45,gap_max_age=10.,pending_ttl=2.)
BASE_END=.30  # pseudo-segment of base_ref ends at the body front (x=0.30, front hip ~0.31)
import os
S['d_deep']=float(os.environ.get('U130_DDEEP',S['d_deep']))
S['dual_grid']=os.environ.get('U130_DUAL','1') in ('1','2')
S['dual_or']=os.environ.get('U130_DUAL','1')=='2'  # 2 = v2.5-OR: GAP if centre is GAP in either grid  # 0 = v2.4 behaviour (single grid, centre or >=2 sub-points)
NL=int(round(2*S['lane_half']/S['lane_w']));NB=int(round(S['u_max']/S['bin_u']))

def self_envelope(points,body_pos,body_quat):
    local=(points-body_pos)@rotation(body_quat)
    return (np.abs(local[...,0])<.43)&(np.abs(local[...,1])<.27)&(local[...,2]>-.50)&(local[...,2]<.20)

from scipy import ndimage
def point_classes(P,V):
    """Per beam on the (sensor,row,col) grid: 1 horizontal, 2 vertical, 0 neither, from PCA over the 5x5 grid
    neighbourhood (valid only). Planarity: smallest-eigenvalue rms <= plane_rms; >= 8 valid neighbours."""
    cls=np.zeros(V.shape,np.int8)
    for k in range(V.shape[0]):
        w=V[k].astype(float);p=np.where(V[k][...,None],P[k],0.)
        f=lambda a:ndimage.uniform_filter(a,size=S['nrm_win'],mode='constant')
        n=f(w);ok=V[k]&(n*S['nrm_win']**2>=S['nrm_min'])
        nn=np.maximum(n,1e-9);m=np.stack([f(w*p[...,i]) for i in range(3)],-1)/nn[...,None]
        C=np.empty(V[k].shape+(3,3))
        for i in range(3):
            for j in range(i,3):C[...,i,j]=C[...,j,i]=f(w*p[...,i]*p[...,j])/nn-m[...,i]*m[...,j]
        ev,evec=np.linalg.eigh(np.where(ok[...,None,None],C,np.eye(3)))
        nz=np.abs(evec[...,2,0]);rms=np.sqrt(np.maximum(ev[...,0],0))
        # v2.4: the normal is trusted only when the window's in-plane extent clearly exceeds the noise; in the
        # near field the 5x5 beam window spans ~1-2 cm and 1 cm range noise would read flat ground as vertical.
        ext=np.sqrt(np.maximum(ev[...,1],0))
        planar=ok&(rms<=S['plane_rms'])&(ext>=S['nrm_ext'])&(ext>=S['nrm_ratio']*rms)
        cls[k][planar&(nz>=S['nz_level'])]=1;cls[k][planar&(nz<=S['nz_vert'])]=2
    return cls

def bin_stats(lane,b,z):
    """Per (lane,bin) over all lanes at once: median, robust spread p90-p10, top z, count within chain_dz of top."""
    cell=lane*NB+b;order=np.lexsort((z,cell));cs=cell[order];zs=z[order]
    uc_,st,cnt=np.unique(cs,return_index=True,return_counts=True)
    q=lambda f:zs[st+np.floor(f*(cnt-1)).astype(int)]
    med=.5*(zs[st+(cnt-1)//2]+zs[st+cnt//2]);spread=q(.9)-q(.1);top=zs[st+cnt-1]
    # count of points >= top-chain_dz: searchsorted within each run
    lo=np.array([np.searchsorted(zs[a:a+n],t-S['chain_dz']) for a,n,t in zip(st,cnt,top)]) if len(st) else np.zeros(0,int)
    topn=cnt-lo
    # v2.3: low cluster per bin (points within chain_dz of the bin minimum) and its median
    lo2=np.array([np.searchsorted(zs[a:a+n],zs[a]+S['chain_dz'],side='right') for a,n in zip(st,cnt)]) if len(st) else np.zeros(0,int)
    low=np.array([np.median(zs[a:a+k]) for a,k in zip(st,lo2)]) if len(st) else np.zeros(0)
    MED=np.full((NL,NB),np.nan);SPR=np.full((NL,NB),np.nan);TOP=np.full((NL,NB),np.nan);TOPN=np.zeros((NL,NB),int);CNT=np.zeros((NL,NB),int);LOW=np.full((NL,NB),np.nan);LOWN=np.zeros((NL,NB),int)
    l_,b_=np.divmod(uc_,NB);MED[l_,b_]=med;SPR[l_,b_]=spread;TOP[l_,b_]=top;TOPN[l_,b_]=topn;CNT[l_,b_]=cnt;LOW[l_,b_]=low;LOWN[l_,b_]=lo2
    return MED,SPR,TOP,TOPN,CNT,LOW,LOWN

def level_clusters(lane,b,z):
    """v2.3: per point, True if it belongs to a level cluster inside its (lane,bin): points sorted by z split at
    gaps > clus_gap; a cluster with >= 2 points and spread <= clus_spread is a level patch (e.g. two treads of a
    step edge running along the lane). Wall faces give continuous z, not separated level clusters."""
    cell=lane*NB+b;o=np.lexsort((z,cell));cs=cell[o];zs=z[o]
    brk=np.r_[True,(cs[1:]!=cs[:-1])|(np.diff(zs)>S['clus_gap'])];cid=np.cumsum(brk)-1
    n=np.bincount(cid);lo=np.minimum.reduceat(zs,np.flatnonzero(brk));hi=np.maximum.reduceat(zs,np.flatnonzero(brk))
    good=(n>=2)&(hi-lo<=S['clus_spread'])
    # a cell that is a single cluster is handled by the bin spread test; only split cells matter here
    split=np.zeros(len(cs),bool)
    if len(cs):
        per=np.zeros(cs.max()-cs.min()+1,int);np.add.at(per,np.unique(np.c_[cs,cid],axis=0)[:,0]-cs.min(),1);split=per[cs-cs.min()]>=2
    out=np.zeros(len(z),bool);out[o]=good[cid]&split
    return out

def lane_planes(med,spr,top,topn,cnt):
    """Chained level segments of one lane from its bin stats. Returns per-bin seg id, fits, level bins, top z, level-top."""
    # v2.2: a single return says nothing about level vs vertical; such bins are neither (min_bin_points).
    has=np.isfinite(med)&(cnt>=S['min_bin_points']);surfbin=has&(spr<=S['surf_spread']);levtop=has&((topn>=2)|surfbin)
    seg=np.full(NB,-1);segs=[];cur=[]
    def close():
        if len(cur)>=S['min_bins']:
            uu=(np.array(cur)+.5)*S['bin_u'];zz=med[cur]
            # Slope only from segments >= 20 cm; short ones are taken level (noise would dominate extrapolation).
            k,a=np.polyfit(uu,zz,1) if np.ptp(uu)>=S['slope_min_len'] else (0.,float(np.median(zz)))
            if abs(k)<=S['max_slope']:
                seg[cur]=len(segs);segs.append((a,k,uu.min(),uu.max(),float(np.median(zz))))
    for q in np.flatnonzero(has):
        if surfbin[q] and cur and q-cur[-1]<=2 and abs(med[q]-med[cur[-1]])<=S['chain_dz']:cur.append(q);continue
        close();cur=[q] if surfbin[q] else []
    close()
    return seg,segs,surfbin,np.where(has,top,np.nan),levtop

def analyze(origins,dirs,ranges,valid,body_pos,body_quat,base_ref=None,support_mem=None,stamp=None,self_hit=None):
    """One capture of one robot.
    base_ref: causal support height under the robot (mapper reference), a level pseudo-segment ending at the
    body front when a lane has no earlier segment. support_mem: StructMemory whose recent support cells stop
    a free-space walk (a beam may pass under an unseen-from-here plate). Returns world-xy GAP bins (support z,
    proven depth), pending points (shallow unexplained), and support points (retraction, support memory)."""
    pts=origins[:,None,None,:]+dirs*ranges[...,None]
    CLS=point_classes(pts,valid) if S['use_pca'] else np.zeros(valid.shape,np.int8)  # v2.4: off by default
    keep=valid&~self_envelope(pts,body_pos,body_quat)
    # v2.2: returns on the robot's own links (URDF + joint state self-filter, as a real driver would) are dropped;
    # the torso envelope alone misses swinging feet in front of the body.
    if self_hit is not None:keep&=~self_hit
    pts=pts[keep];d=dirs[keep];cls=CLS[keep]
    w,x,y,zq=body_quat/np.linalg.norm(body_quat);yaw=np.arctan2(2*(w*zq+x*y),1-2*(y*y+zq*zq));c,s=np.cos(yaw),np.sin(yaw)
    rx=pts[:,0]-body_pos[0];ry=pts[:,1]-body_pos[1]
    u=c*rx+s*ry;v=-s*rx+c*ry;z=pts[:,2];du=c*d[:,0]+s*d[:,1];dv=-s*d[:,0]+c*d[:,1];dz=d[:,2]
    roi=(u>=0)&(u<S['u_max'])&(np.abs(v)<S['lane_half'])
    u,v,z,du,dv,dz,pts,cls=u[roi],v[roi],z[roi],du[roi],dv[roi],dz[roi],pts[roi],cls[roi]
    if len(u)==0:  # v2.4 (UPDATE130C P1): an empty capture is no evidence
        return dict(gap_xy=np.zeros((0,2)),gap_z=np.zeros(0),gap_depth=np.zeros(0),pending_xy=np.zeros((0,2)),support_xyz=np.zeros((0,3)),n_points=0,n_evidence=0,evid_dbg=np.zeros((0,5)))
    lane=np.floor((v+S['lane_half'])/S['lane_w']).astype(int).clip(0,NL-1);b=np.floor(u/S['bin_u']).astype(int).clip(0,NB-1)
    REF=np.full((NL,NB),np.nan);TOPZ=np.full((NL,NB),np.nan);LEV=np.zeros((NL,NB),bool)
    pend=np.zeros(len(u),bool);support=np.zeros(len(u),bool);evid=np.zeros(len(u),bool);zref=np.full(len(u),np.nan)
    uc=(np.arange(NB)+.5)*S['bin_u'];MED,SPR,TOPA,TOPN,CNTA,LOWA,LOWNA=bin_stats(lane,b,z);INCL=level_clusters(lane,b,z)
    MEMREF=None
    if support_mem is not None and support_mem.sup:
        gu=np.broadcast_to(uc[None,:]-S['memref_back'],(NL,NB));gv=np.broadcast_to(((np.arange(NL)+.5)*S['lane_w']-S['lane_half'])[:,None],(NL,NB))
        MEMREF=support_mem.sup_lookup(body_pos[0]+c*gu-s*gv,body_pos[1]+s*gu+c*gv,stamp)
    # Pass 1, per lane: level segments, reference support, point classes.
    for l in np.unique(lane):
        m=np.flatnonzero(lane==l)
        seg,segs,surfbin,topz,levtop=lane_planes(MED[l],SPR[l],TOPA[l],TOPN[l],CNTA[l]);TOPZ[l]=topz;LEV[l]=levtop
        if S['use_base_ref'] and base_ref is not None:
            segs=[(float(base_ref),0.,0.,BASE_END,float(base_ref))]+segs;seg=np.where(seg>=0,seg+1,-1)
        # Reference per bin: latest segment (other than the bin's own) whose far end is at or before the bin.
        if segs:
            ends=np.array([g[3] for g in segs]);A=np.array([g[0] for g in segs]);Kk=np.array([g[1] for g in segs])
            last=np.searchsorted(ends,uc+1e-9,side='right')-1           # segments appended in increasing u
            own=(seg>=0)&(last==seg);last=np.where(own,last-1,last)
            ok=last>=0;li=last.clip(0)
            ok&=uc-ends[li]<=S['ref_reach']
            REF[l]=np.where(ok,A[li]+Kk[li]*uc,np.nan)
        # v2.2: where this capture has no earlier segment, the reference is the support remembered just
        # behind the bin (4 cm, same lane) from earlier captures, not the robot-centred map median.
        if MEMREF is not None:REF[l]=np.where(np.isfinite(REF[l]),REF[l],MEMREF[l])
        pb=b[m];zr=REF[l,pb];depth=zr-z[m];dep=np.nan_to_num(depth,nan=-1.)
        # Level points (level bin, robust spread): within STEP_MAX of the reference they are support (tread,
        # incl. short fragments); deeper they are evidence only if this lane later rises back by >= STEP_MAX
        # (gap floor), never for a stair run / drop-off. Vertical points: non-level bin, below its level top.
        # v2.2: support and deep-level evidence only from chained segments (>=3 level bins); isolated level
        # bins (e.g. noisy riser fragments at long range) are neither support nor evidence.
        level=(seg[pb]>=0) if segs else np.zeros(len(m),bool)
        tread=level&(dep<=S['step_max'])
        support[m]=tread;zref[m]=zr
        deeplevel=level&(dep>S['step_max'])
        if deeplevel.any():
            ok=levtop&np.isfinite(topz);ul=uc[ok];zl=topz[ok]
            for k in np.flatnonzero(deeplevel):
                if not ((ul>u[m[k]])&(ul-u[m[k]]<=S['ref_reach'])&(zl>=z[m[k]]+S['step_max'])).any():deeplevel[k]=False
        # v2.3: a bin whose low cluster continues at the same height over >= 3 bins along the lane is a tread
        # edge running along the lane (lateral step), not a wall: its low points are not vertical evidence.
        lowrun=np.zeros(NB,bool);lz=LOWA[l];ok=np.isfinite(lz)&(LOWNA[l]>=2);idx=np.flatnonzero(ok);run=[]
        for q in idx:
            if run and q-run[-1]<=2 and abs(lz[q]-lz[run[-1]])<=S['chain_dz']:run.append(q);continue
            if len(run)>=S['min_bins']:lowrun[run]=True
            run=[q]
        if len(run)>=S['min_bins']:lowrun[run]=True
        onlow=lowrun[pb]&(np.abs(z[m]-lz[pb])<=S['clus_spread'])
        vertical=~surfbin[pb]&(CNTA[l,pb]>=S['min_bin_points'])&~INCL[m]&((cls[m]==2) if S['use_pca'] else True)&~onlow&~(levtop[pb]&(z[m]>=topz[pb]-S['chain_dz']))
        cand=np.isfinite(depth)&(depth>=S['pending_depth'])&(vertical|deeplevel)
        gapc=cand&(depth>=S['gap_depth'])&(dz[m]<-1e-3)
        pend[m[cand&~gapc]]=True;evid[m[gapc]]=True
    # Pass 2: walk each evidence beam back from where it dropped below the support (u_cross) toward its
    # return, in 1 cm steps across lanes; every step is free space at support height until the beam reaches
    # a bin with a level top within STEP_MAX of that support (observed now) or a recently supported world
    # cell, or goes deeper than MARK_DEPTH below the support.
    out=dict(gap_xy=np.zeros((0,2)),gap_z=np.zeros(0),gap_depth=np.zeros(0))
    e=np.flatnonzero(evid)
    if len(e):
        dep=zref[e]-z[e];t_cross=dep/(-dz[e])                      # beam length from crossing to return
        K=int(np.ceil(S['u_max']/.01));tt=np.arange(K)*.01
        L3=np.sqrt(du[e]**2+dv[e]**2+dz[e]**2)
        tq=(t_cross[:,None]-tt[None,:])                              # distance back from the return
        okq=tq>=0
        su=u[e,None]-du[e,None]*tq/L3[:,None];sv=v[e,None]-dv[e,None]*tq/L3[:,None]
        sd=dep[:,None]-(-dz[e,None])*tq/L3[:,None]                  # depth of the beam below support there
        ql=np.floor((sv+S['lane_half'])/S['lane_w']).astype(int);qb=np.floor(su/S['bin_u']).astype(int)
        inside=okq&(ql>=0)&(ql<NL)&(qb>=0)&(qb<NB)
        qlc=ql.clip(0,NL-1);qbc=qb.clip(0,NB-1)
        blocked=LEV[qlc,qbc]&(TOPZ[qlc,qbc]>=zref[e,None]-S['step_max'])
        wx=body_pos[0]+c*su-s*sv;wy=body_pos[1]+s*su+c*sv
        if support_mem is not None and support_mem.sup:
            memz=support_mem.sup_lookup(wx,wy,stamp)
            blocked|=np.isfinite(memz)&(memz>=zref[e,None]-S['step_max'])
        blocked&=sd>=S['block_depth']   # at the crossing itself the beam grazes the support edge
        blocked|=sd>S['mark_depth']
        blocked&=okq;blocked|=~inside&okq
        stop=np.where(blocked.any(1),blocked.argmax(1),K)
        take=okq&(np.arange(K)[None,:]<stop[:,None])
        out['gap_xy']=np.stack((wx[take],wy[take]),-1);out['gap_z']=np.broadcast_to(zref[e,None],take.shape)[take]
        out['gap_depth']=np.broadcast_to(dep[:,None],take.shape)[take]
    out['pending_xy']=pts[pend,:2];out['support_xyz']=np.c_[pts[support,:2],z[support]]
    out['n_points']=int(len(u));out['n_evidence']=int(len(e))
    out['evid_dbg']=np.c_[u[e],v[e],z[e],zref[e],cls[e]] if len(e) else np.zeros((0,5))
    return out

class StructMemory:
    """World GAP memory {cell:(z_support,depth,t_first,t_last)} on 5 cm cells; v2.5 adds a second 5 cm grid offset by
    2.5 cm (gap2) and the teacher cell is GAP only if its centre is GAP in both (staggered AND = 2.5 cm boundary).
    pending {cell:t} and support {2 cm cell:(z,t)} stay single-grid."""
    def __init__(self,res=.05):
        self.res=res;self.gap={};self.gap2={};self.pending={};self.sup={};self.counts=dict(gap_marks=0,pending_marks=0,retracted=0,expired=0)
        self.grids=[(self.gap,0.),(self.gap2,S['dual_offset'])] if S['dual_grid'] else [(self.gap,0.)]
    def reset(self):self.gap.clear();self.gap2.clear();self.pending.clear();self.sup.clear()
    def sup_lookup(self,wx,wy,stamp):
        keys=np.array(list(self.sup.keys()),dtype=np.int64);vals=np.array(list(self.sup.values()))
        fresh=stamp-vals[:,1]<=S['sup_ttl'];keys=keys[fresh];vz=vals[fresh,0]
        out=np.full(wx.shape,np.nan)
        if not len(keys):return out
        code=keys[:,0]*1000003+keys[:,1];o=np.argsort(code);code=code[o];vz=vz[o]
        qi=np.floor(wx/S['sup_res']+1e-7).astype(np.int64);qj=np.floor(wy/S['sup_res']+1e-7).astype(np.int64);qc=qi*1000003+qj
        pos=np.searchsorted(code,qc).clip(0,len(code)-1);hit=code[pos]==qc
        out[hit]=vz[pos[hit]];return out
    def sup_z(self,k,stamp):
        g=self.sup.get(k);return g[0] if g and (stamp is None or stamp-g[1]<=S['sup_ttl']) else np.nan
    def cells(self,xy,off=0.):return np.floor((np.asarray(xy)-off)/self.res+1e-7).astype(int)
    def update(self,res,stamp):
        nows=[]
        for G,off in self.grids:
            now=set();nows.append(now);gc=self.cells(res['gap_xy'],off);gz=res['gap_z'];gd=res['gap_depth']
            if len(gc):
                # v2.4: one mark per cell per capture: median support z (no max bias), deepest proof
                code=gc[:,0].astype(np.int64)*1000003+gc[:,1];o=np.lexsort((gz,code));code=code[o];gc=gc[o];gz=gz[o];gd=gd[o]
                st_=np.flatnonzero(np.r_[True,code[1:]!=code[:-1]]);n_=np.diff(np.r_[st_,len(code)])
                zmed=.5*(gz[st_+(n_-1)//2]+gz[st_+n_//2]);dmax=np.maximum.reduceat(gd,st_)
                gc=gc[st_];gz=zmed;gd=dmax
            for (i,j),zs,dp in zip(gc.tolist(),gz,gd):
                k=(i,j);old=G.get(k);now.add(k)
                G[k]=(float(zs),max(dp,old[1]) if old else float(dp),old[2] if old else stamp,stamp)  # latest support z
                if off==0.:self.counts['gap_marks']+=1
        now=nows[0]
        pc=self.cells(res['pending_xy'])
        if len(pc):
            keys,cnt=np.unique(pc,axis=0,return_counts=True)
            for k in map(tuple,keys[cnt>=S['pending_min_points']].tolist()):
                if k not in self.gap:self.pending[k]=stamp;self.counts['pending_marks']+=1
        sx=res['support_xyz']
        if len(sx):
            # Support memory (highest recent support per cell): stops later free-space walks under it.
            # v2.2: per-cell median of this capture's support returns (max was biased up by noise), latest wins.
            sc=np.floor(sx[:,:2]/S['sup_res']+1e-7).astype(np.int64);code=sc[:,0]*1000003+sc[:,1]
            o=np.lexsort((sx[:,2],code));cs=code[o];zs=sx[o,2];first=np.flatnonzero(np.r_[True,cs[1:]!=cs[:-1]])
            cnt=np.diff(np.r_[first,len(cs)]);med=.5*(zs[first+(cnt-1)//2]+zs[first+cnt//2])
            for k,zz in zip(map(tuple,sc[o[first]].tolist()),med):self.sup[k]=(float(zz),stamp)
        # Retraction (v2.4, UPDATE130C P1): returns in the cell interior (>= retract_margin from its edges) only.
        # (a) design rule: >= retract_points returns within +-retract_band of the stored support height, over
        #     >= 2 quadrants; (b) a solid lower tread: >= 2*retract_points returns between stored-STEP_MAX and
        #     stored-retract_band covering >= 3 quadrants. Returns above stored+retract_band never retract.
        #     Pending cells: either rule with their own capture's support (no stored height: any tread level).
        if len(sx):
            for (G,off),now in zip(self.grids,nows):self._retract(G,off,now,sx,pend=(off==0.))
    def _retract(self,G,off,now,sx,pend):
        if not (G or (pend and self.pending)):return
        f=(sx[:,:2]-off)/self.res;cc=np.floor(f+1e-7).astype(np.int64);fr=f-cc;m_=S['retract_margin']/self.res
        inner=((fr>m_)&(fr<1-m_)).all(1);quad=(fr[:,0]>=.5).astype(int)*2+(fr[:,1]>=.5).astype(int)
        band={};low={}
        for (i,j),qq,zz,ok in zip(cc.tolist(),quad.tolist(),sx[:,2],inner):
            if not ok:continue
            k=(i,j);g=G.get(k)
            if g is not None:
                if abs(zz-g[0])<=S['retract_band']:band.setdefault(k,[]).append(qq)
                elif g[0]-S['step_max']<=zz<g[0]-S['retract_band']:low.setdefault(k,[]).append(qq)
            elif pend and k in self.pending:band.setdefault(k,[]).append(qq)
        for k in set(band)|set(low):
            b_=band.get(k,[]);l_=low.get(k,[])
            ok=(len(b_)>=S['retract_points'] and len(set(b_))>=2) or (len(l_)>=2*S['retract_points'] and len(set(l_))>=3)
            if not ok:continue
            if k in G and k not in now:
                del G[k]
                if off==0.:self.counts['retracted']+=1
            if pend and k in self.pending:del self.pending[k]
    def expire(self,pose,stamp):
        c,s=np.cos(pose[2]),np.sin(pose[2])
        for G,off in self.grids:
            for k in list(G):
                g=G[k];cx=(k[0]+.5)*self.res+off-pose[0];cy=(k[1]+.5)*self.res+off-pose[1]
                if c*cx+s*cy<S['hind_pass_x'] or stamp-g[3]>S['gap_max_age']:
                    del G[k]
                    if off==0.:self.counts['expired']+=1
        for k in [k for k,t in self.pending.items() if stamp-t>S['pending_ttl']]:del self.pending[k]
        for k in [k for k,g in self.sup.items() if stamp-g[1]>S['sup_ttl']]:del self.sup[k]
    def overlay(self,packed_row,pose,scanner_z,stamp,source_row):
        """Teacher 187 cells: GAP if centre or >=2 of 4 sub-points (+-2.5 cm) are GAP; else pending centre -> unknown."""
        self.expire(pose,stamp)
        if not self.gap and not self.gap2 and not self.pending:return
        y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij');bx=x.ravel();by=y.ravel()
        c,s=np.cos(pose[2]),np.sin(pose[2])
        def look(dx,dy):
            ox=bx+dx;oy=by+dy;ij=self.cells(np.c_[pose[0]+c*ox-s*oy,pose[1]+s*ox+c*oy]).tolist()
            return [self.gap.get(tuple(k)) for k in ij],[tuple(k) in self.pending for k in ij]
        cg,cp=look(0,0)
        if S['dual_grid']:
            # v2.5: centre point GAP in both staggered grids (same point the teacher's scanner samples)
            ox=bx;oy=by;ij2=self.cells(np.c_[pose[0]+c*ox-s*oy,pose[1]+s*ox+c*oy],S['dual_offset']).tolist()
            c2=[self.gap2.get(tuple(k)) for k in ij2];subs=[[None]*187]*4
        else:
            c2=cg;subs=[look(dx,dy)[0] for dx,dy in ((-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025))]
        for q in range(187):
            hits=[g for g in [sb[q] for sb in subs] if g]
            if ((cg[q] or c2[q]) if S.get('dual_or') else (cg[q] and c2[q])) or len(hits)>=2:
                g=cg[q] or c2[q] or max(hits,key=lambda t:t[1])
                packed_row[q]=5*np.clip(scanner_z-(g[0]-S['d_deep'])-.5,-1,1);packed_row[187+q]=1;source_row[q]=4
            elif cp[q]:
                packed_row[q]=0.;packed_row[187+q]=0;source_row[q]=5
