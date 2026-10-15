"""UPDATE130 CPU fixtures for structure130 (no Isaac, no GPU).
Real LiDAR beam pattern (lidar126.mounted, 1 sensor) cast against box terrain (infinite in y),
range noise 1 cm, 1% drop. Robot trunk at x=0, z=0.42, level, heading +x; terrain moved toward it.
"""
import os,sys
os.environ.setdefault('U127_SENSOR_COUNT','1')
import numpy as np
from lidar126 import mounted
import importlib;st=importlib.import_module(os.environ.get('T130_MOD','structure130'))

def cast(boxes,o,d):
    """Slab intersection in x/z with boxes (x0,x1,z0,z1); returns range (inf if none)."""
    best=np.full(d.shape[:-1],np.inf)
    for x0,x1,z0,z1 in boxes:
        with np.errstate(divide='ignore',invalid='ignore'):
            tx0=(x0-o[0])/d[...,0];tx1=(x1-o[0])/d[...,0];tz0=(z0-o[2])/d[...,2];tz1=(z1-o[2])/d[...,2]
        tn=np.maximum(np.minimum(tx0,tx1),np.minimum(tz0,tz1));tf=np.minimum(np.maximum(tx0,tx1),np.maximum(tz0,tz1))
        hit=(tf>=tn)&(tf>0)&np.isfinite(tn)
        best=np.where(hit&(tn>0)&(tn<best),tn,best)
    return best

def scene(kind,xe):
    G=[(-5,xe,-1,0.)]
    if kind=='flat':return [(-5,5,-1,0.)],None
    if kind.startswith('gap') and kind[3:].replace('.','').isdigit():
        w=float(kind[3:])/100;return G+[(xe+w,5,-1,0.),(-5,5,-2,-.65)],(xe,xe+w)
    if kind=='subway':  # no wall: platform, 7.6 cm gap, 2 cm thick car floor plate +3 cm, track 1.2 m below
        w=.076;return G+[(xe+w,5,.01,.03),(-5,5,-2,-1.2)],(xe,xe+w)
    if kind=='subway_low':  # car floor 5 cm below platform
        w=.076;return G+[(xe+w,5,-.07,-.05),(-5,5,-2,-1.2)],(xe,xe+w)
    if kind=='stairs_down':return G+[(xe+.3*k,xe+.3*(k+1),-1,-.15*(k+1)) for k in range(4)]+[(xe+1.2,5,-1,-.6)],None
    if kind=='stairs_up':return G+[(xe+.3*k,xe+.3*(k+1),-1,.15*(k+1)) for k in range(4)]+[(xe+1.2,5,-1,.6)],None
    if kind.startswith('sd'):  # sdRR_TT: descending stairs riser RR cm, tread TT cm, robot already on the flight
        rr,tt=[float(x)/100 for x in kind[2:].split('_')]
        return [(-5,xe-tt,-1,rr),(xe-tt,xe,-1,0.)]+[(xe+tt*k,xe+tt*(k+1),-1,-rr*(k+1)) for k in range(6)]+[(xe+6*tt,5,-1,-6*rr)],None
    if kind=='groove':return G+[(xe+.10,5,-1,0.),(-5,5,-2,-.05)],None   # 10 cm wide, 5 cm deep: not a gap
    if kind=='curb':return G+[(xe,5,-1,-.05)],None                       # 5 cm step down
    if kind=='gap_step':w=.10;return G+[(xe+w,5,-1,-.15),(-5,5,-2,-.65)],(xe,xe+w)  # gap then 15 cm lower
    raise KeyError(kind)

PITCH=float(os.environ.get('T130_PITCH','0'))
def capture(boxes,rng):
    a=np.deg2rad(PITCH)/2;p=np.array([[0.,0.,.42]]);q=np.array([[np.cos(a),0,np.sin(a),0]])
    o,d=mounted(p,q);o=o[0];d=d[0]
    r=np.stack([cast(boxes,o[s],d[s]) for s in range(len(o))])
    valid=np.isfinite(r)&(r<=10)
    r=np.where(valid,r+rng.normal(0,.01,r.shape),10.);valid&=rng.random(r.shape)>=.01
    return o,d,np.where(valid,r,10.),valid,np.array([0.,0.,.42]),q[0]

def run(kind,positions=(1.2,1.0,.85,.7,.6,.5),seed=0):
    """Robot approaches: terrain edge at body-frame xe; memory in a world frame that moves with the terrain."""
    rng=np.random.default_rng(seed);mem=st.StructMemory();rows=[]
    for t,xe in enumerate(positions):
        boxes,gap=scene(kind,xe)
        o,d,r,v,bp,bq=capture(boxes,rng);bp=bp+np.array([positions[0]-xe,0.,0.]);o=o+np.array([positions[0]-xe,0.,0.])
        res=st.analyze(o,d,r,v,bp,bq,base_ref=0.,support_mem=mem,stamp=t*.1)  # world frame: terrain fixed, robot moves
        mem.update(res,t*.1)
        gx=np.array([(k[0]+.5)*mem.res for k in mem.gap]) if mem.gap else np.zeros(0)
        lane_ok=np.array([abs((k[1]+.5)*mem.res)<.5 for k in mem.gap]) if mem.gap else np.zeros(0,bool)
        gx=gx[lane_ok]
        if gap:
            g0,g1=gap[0]+positions[0]-xe,gap[1]+positions[0]-xe
            inside=((gx>=g0-.05)&(gx<=g1+.05)).sum();outside=len(gx)-inside
        else:inside=0;outside=len(gx)
        rows.append((xe,inside,outside,len(mem.pending),res['n_points']))
    return rows

if __name__=='__main__':
    kinds=sys.argv[1:] or ['flat','gap6.5','gap13','gap30','subway','subway_low','gap_step','stairs_down','stairs_up','groove','curb']
    for k in kinds:
        print(f'== {k}: edge xe | GAP cells in gap(+-5cm) | GAP cells elsewhere | pending cells | points')
        for r in run(k):print('   %.2f  %4d  %4d  %4d  %5d'%r)
