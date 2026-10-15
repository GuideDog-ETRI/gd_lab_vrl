"""UPDATE130 v2.4 CPU regression with pass/fail asserts (no GPU). Run: python test_structure130_assert.py"""
import os,sys
os.environ.setdefault('U127_SENSOR_COUNT','1')
import numpy as np
import importlib
st=importlib.import_module(os.environ.get('T130_MOD','structure130'))
import test_structure130 as T

fails=[]
def check(cond,msg):
    print(('PASS ' if cond else 'FAIL ')+msg)
    if not cond:fails.append(msg)

# 1. scenes: no GAP outside the true gap (+-5 cm) at any approach step; gaps detected by a given distance
DETECT={'gap13':.6,'gap30':1.0,'gap6.5':.5}
for pitch in (0.,8.,-8.):
    T.PITCH=pitch
    for k in ['flat','gap6.5','gap13','gap30','subway','subway_low','gap_step','stairs_down','stairs_up','sd08_25','sd10_25','sd12_28','sd15_30','groove','curb']:
        rows=T.run(k)
        out=max(r[2] for r in rows)
        check(out==0,f'pitch {pitch:+.0f} {k}: no GAP outside gap (max {out})')
        if k in DETECT and pitch==0:
            got=[r for r in rows if r[0]<=DETECT[k]+1e-9 and r[1]>0]
            check(len(got)>0,f'pitch 0 {k}: detected by edge distance {DETECT[k]} m')

# 2. empty inputs never raise and add nothing
T.PITCH=0.
o,d,r,v,bp,bq=T.capture(T.scene('flat',.7)[0],np.random.default_rng(0))
for name,vv,sh in [('all invalid',np.zeros_like(v),None),('all self-hit',v,v.copy()),('no valid in ROI',v&False,None)]:
    try:
        res=st.analyze(o,d,r,vv,bp,bq,support_mem=st.StructMemory(),stamp=0.,self_hit=sh)
        mem=st.StructMemory();mem.update(res,0.)
        check(len(res['gap_xy'])==0 and len(res['pending_xy'])==0 and len(res['support_xyz'])==0,f'empty input ({name}): empty outputs')
    except Exception as ex:check(False,f'empty input ({name}): raised {type(ex).__name__}: {ex}')

# 3. retraction rules on a stored GAP cell at support z=0 (cell (10,10), 5 cm)
def mem_with_gap():
    m=st.StructMemory();m.gap[(10,10)]=(0.,.3,0.,0.);return m
def pts(zs,frs):
    return np.array([[(10+fx)*.05,(10+fy)*.05,z] for z,(fx,fy) in zip(zs,frs)])
E=dict(gap_xy=np.zeros((0,2)),gap_z=np.zeros(0),gap_depth=np.zeros(0),pending_xy=np.zeros((0,2)))
q2=[(.3,.3),(.7,.3),(.3,.7)]                 # interior, 3 quadrants
m=mem_with_gap();m.update(dict(E,support_xyz=pts([0.,.005,-.01],q2)),.1);check((10,10) not in m.gap,'retract: 3 returns at support height over >=2 quadrants')
m=mem_with_gap();m.update(dict(E,support_xyz=pts([.5,.5,.5,.5,.5,.5],q2*2)),.1);check((10,10) in m.gap,'no retract: returns 50 cm above support')
m=mem_with_gap();m.update(dict(E,support_xyz=pts([0.,0.,0.],[(.1,.3),(.1,.7),(.15,.5)])),.1);check((10,10) in m.gap,'no retract: returns only in the 1 cm edge margin / one side')
m=mem_with_gap();m.update(dict(E,support_xyz=pts([0.,0.],q2[:2])),.1);check((10,10) in m.gap,'no retract: only 2 returns')
m=mem_with_gap();m.update(dict(E,support_xyz=pts([-.10]*6,q2*2)),.1);check((10,10) not in m.gap,'retract: solid lower tread (-10 cm, 6 returns, 3 quadrants)')
m=mem_with_gap();m.update(dict(E,support_xyz=pts([-.40]*6,q2*2)),.1);check((10,10) in m.gap,'no retract: returns 40 cm below (gap floor)')
m=mem_with_gap();m.update(dict(E,gap_xy=np.array([[10.5*.05,10.5*.05]]),gap_z=np.array([0.]),gap_depth=np.array([.2]),support_xyz=pts([0.,0.,0.],q2)),.1)
check((10,10) in m.gap,'no retract when the same capture marks the cell GAP')

# 4. v2.5 overlay: staggered dual grid, centre AND
if st.S.get('dual_grid'):
    pose=np.array([0.,0.,0.]);Q=5*17+11   # teacher cell row 5 (y=0), col 11 (x=+0.3): in front, not expired
    qx,qy=.3,0.
    orm=bool(st.S.get('dual_or'))
    for name,inA,inB,expect in [('A only',True,False,orm),('B only',False,True,orm),('both',True,True,True)]:
        m=st.StructMemory()
        if inA:m.gap[tuple(m.cells(np.array([[qx,qy]]))[0])]=(0.,.3,0.,0.)
        if inB:m.gap2[tuple(m.cells(np.array([[qx,qy]]),st.S['dual_offset'])[0])]=(0.,.3,0.,0.)
        row=np.zeros(374,np.float32);src=np.zeros(187,np.uint8);m.overlay(row,pose,.42,0.,src)
        check((src[Q]==4)==expect and (row[187+Q]==1)==expect,f'overlay dual {"OR" if orm else "AND"}: {name} -> GAP {expect}')
        if expect and inA:check(abs(row[Q]-5*np.clip(.42-(0.-st.S['d_deep'])-.5,-1,1))<1e-5,'overlay encoding h=5*clip(scanner-(z_sup-D)-0.5)')
print('\nRESULT:',('ALL PASS' if not fails else f'{len(fails)} FAIL'))
sys.exit(1 if fails else 0)
