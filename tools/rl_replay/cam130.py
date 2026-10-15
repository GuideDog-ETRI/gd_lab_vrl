"""UPDATE130 camera path: the four vendor_new depth cameras as the point source for the same mapper/structure130.

Same packet as lidar126.Lidar.capture (origins, unit dirs, ranges, valid, body pose, self_hit), so L0S/L0D run unchanged.
Depth and IR come from the existing UPDATE121 sensor model (sensor121.IsaacSensor: ray cast on the Isaac terrain,
optical-axis depth, stereo fill-in patches over gaps, flying pixels, spikes, holes, dropout, 1 cm noise, 0.15-5 m clip).
Added here: occlusion by the robot's own legs (URDF collision links, live pose; the trunk is excluded because the
cameras are flush-mounted on it and look out of it). Sensor data only: the mapper never sees terrain labels.
"""
import os,time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from heightmap116 import rotation
from sensor121 import IsaacSensor
from lidar126 import RobotMesh

# UPDATE136 diagnosis: U136_CAM_OFF=spike,flying,dropout,noise_sigma,deep_hole zeroes those sensor121 rates (default: none).
from heightmap116 import R as _R
for _k in filter(None,os.environ.get('U136_CAM_OFF','').split(',')):
    assert hasattr(_R,_k),_k;object.__setattr__(_R,_k,0.)  # Rules is a frozen dataclass
CAM_FILTER=os.environ.get('U136_CAM_FILTER','0')=='1'
from camfilter136 import reject_outliers
CAM_W=int(os.environ.get('U130_CAM_W','160'));CAM_H=int(os.environ.get('U130_CAM_H','90'))

def qmul(a,b):
    w1,x1,y1,z1=np.moveaxis(a,-1,0);w2,x2,y2,z2=np.moveaxis(b,-1,0)
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],-1)

def leg_urdf(src,dst):
    tree=ET.parse(src)
    for link in tree.getroot().findall('link'):
        if link.get('name')=='trunk':
            for c in link.findall('collision'):link.remove(c)
    tree.write(dst);return dst

class CamSensor:
    def __init__(self,base,geometry):
        from gd_lab.core.camera_contract import load_camera_contract
        self.base=base;self.robot=base.scene['robot'];self.trunk=self.robot.body_names.index('trunk')
        c=load_camera_contract('vendor_new');self.contract=c
        self.P=np.array(c.positions,float);self.Q=np.array(c.quaternions_opengl,float)
        fx=CAM_W*c.focal_m/c.sensor_size_m[0];fy=CAM_H*c.focal_m/c.sensor_size_m[1]
        self.K=np.array([[fx,0,CAM_W/2],[0,fy,CAM_H/2],[0,0,1.]])
        self.model=IsaacSensor(base,geometry,42)
        urdf=Path(__file__).resolve().parents[2]/'assets/robots/rbq10/urdf/rbq10_simple.urdf'
        tmp=Path(os.environ.get('TMPDIR','/tmp'))/'rbq10_legs_only_u130.urdf'
        self.legs=RobotMesh(self.robot,base.device,leg_urdf(urdf,tmp))
        self.counts={};self.times=[];self.steps=[]
    def add(self,k,v):self.counts[k]=self.counts.get(k,0)+int(v)
    def capture(self,episodes,step):
        begin=time.perf_counter();data=self.robot.data
        bp=data.root_pos_w.cpu().numpy().copy();bq=data.root_quat_w.cpu().numpy().copy()
        p=data.body_pos_w[:,self.trunk].cpu().numpy();q=data.body_quat_w[:,self.trunk].cpu().numpy()
        n=len(p);ncam=len(self.P)
        positions=p[:,None,:]+np.einsum('nij,cj->nci',rotation(q),self.P)
        quats=qmul(np.repeat(q[:,None,:],ncam,1),np.broadcast_to(self.Q,(n,ncam,4)))
        ks=np.broadcast_to(self.K,(n,ncam,3,3)).copy()
        gen=self.model.capture(positions,quats,ks,episodes,int(step),(CAM_H,CAM_W))
        v,u=np.mgrid[:CAM_H,:CAM_W]
        local=np.stack(((u-self.K[0,2])/self.K[0,0],-(v-self.K[1,2])/self.K[1,1],-np.ones((CAM_H,CAM_W))),-1)
        rays=np.einsum('hwj,ncij->nchwi',local,rotation(quats))  # = heightmap116.rays(k,q,h,w) per camera
        length=np.linalg.norm(rays,axis=-1);unit=rays/length[...,None]
        depth=np.stack([g[0] for g in gen]);valid=np.stack([g[1] for g in gen])
        clean=np.stack([np.where(np.stack([l['hit'] for l in g[4]]),np.linalg.norm(np.stack([l['points'] for l in g[4]])-positions[e][:,None,None,:],axis=-1),np.inf) for e,g in enumerate(gen)])
        ranges=np.where(valid,depth*length,10.)
        starts=np.broadcast_to(positions[:,:,None,None,:],unit.shape)
        leg,_=self.legs.distances(starts.reshape(n,-1,3),unit.reshape(n,-1,3));leg=leg.reshape(valid.shape)
        blocked=np.isfinite(leg)&(leg<clean)
        for e,episode in enumerate(episodes):
            rng=np.random.default_rng(np.random.SeedSequence([42,e,int(episode),int(step),130]))
            b=blocked[e];ranges[e][b]=leg[e][b]+rng.normal(0,.01,int(b.sum()))
        valid=np.where(blocked,(ranges>=.15*length)&(ranges<5.*length),valid)  # 0.15-5 m clip is on optical depth
        if CAM_FILTER:  # UPDATE136 B: camera-side outlier rejection (camfilter136), rejected pixels -> unknown
            kept=reject_outliers(ranges/length,valid);self.add('filter_rejected',(valid&~kept).sum());self.add('filter_rejected_leg',(valid&~kept&blocked).sum());valid=kept
        ranges[~valid]=10.
        assert np.isfinite(ranges).all() and np.isfinite(unit).all()
        for g in gen:
            for k,val in g[3].items():self.add(k,val)
        self.add('beams',valid.size);self.add('valid_returns',valid.sum());self.add('leg_occluded',blocked.sum());self.add('leg_returns',(blocked&valid).sum())
        self.steps.append(int(step));self.times.append(time.perf_counter()-begin)
        return dict(origins=positions,dirs=unit,ranges=ranges,valid=valid,body_pos=bp,body_quat=bq,self_hit=blocked&valid)
    def report(self):
        return dict(sensor='vendor_new depth cameras x4',resolution=[CAM_H,CAM_W],counts=self.counts,capture_steps=self.steps,
                    capture_seconds=sum(self.times),capture_ms_mean=1000*np.mean(self.times) if self.times else None,
                    noise_model='sensor121.IsaacSensor (UPDATE121 realistic depth+IR) + leg occlusion (URDF, trunk excluded)',camera_filter='camfilter136' if CAM_FILTER else None,camera_noise_off=os.environ.get('U136_CAM_OFF',''),
                    positions_trunk_m=self.P.tolist(),quaternions_opengl_wxyz=self.Q.tolist(),K=self.K.tolist())
