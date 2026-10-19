"""GPU0 sensor microbenchmark only. No Isaac launch or learning.

Synthetic closed primitive scene, real URDF collision geometry with nominal
standing FK (THIGH=.76, KNEE=-1.45); same processing for every active primitive.
Includes snapshot adapter, vendor_new extrinsics/K, validation, query and output.
Excludes physics/rendering/supervisor/map/history, explicitly NOT a rollout test.
"""
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from gd_lab.gast_nav.producer168 import SensorPrimitives,produce_blocks168,BOX,CYLINDER
from gd_lab.gast_nav.producer168_adapter import load_self_collisions168,primitive_snapshot168,vendor_cameras168,d4xx_blocks168

ROOT=Path(__file__).resolve().parents[2]/'logs/gast/nav_v1/update168_producer'
URDF=str(Path(__file__).resolve().parents[2]/'assets/robots/rbq10/urdf/rbq10_simple.urdf')


def nominal_bodies():
    transforms={'trunk':(np.array([0.,0.,.55]),np.eye(3))}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link')
        if parent not in transforms:raise ValueError('URDF topology')
        origin=joint.find('origin');xyz=np.array(list(map(float,origin.get('xyz','0 0 0').split())))
        rpy=list(map(float,origin.get('rpy','0 0 0').split()))
        rr,pp,yy=rpy;cr,sr=math.cos(rr),math.sin(rr);cp,sp=math.cos(pp),math.sin(pp);cy,sy=math.cos(yy),math.sin(yy)
        origin_rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        angle=.76 if joint.get('name').endswith('THIGH') else (-1.45 if joint.get('name').endswith('KNEE') else 0.)
        axis=np.array(list(map(float,joint.find('axis').get('xyz').split()))) if joint.find('axis') is not None else np.array([1.,0.,0.])
        cross=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
        rot=np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
        pp,pr=transforms[parent];transforms[child]=(pp+pr@xyz,pr@origin_rotation@rot)
    names=list(transforms);pos=[];quat=[]
    for name in names:
        p,r=transforms[name]
        # Only collision-bearing bodies (and trunk) are needed; all have a
        # positive-trace nominal joint rotation at this posture.
        if name not in {s.link for s in load_self_collisions168(URDF)}:continue
        w=math.sqrt(1+np.trace(r))/2
        quat.append([w,(r[2,1]-r[1,2])/(4*w),(r[0,2]-r[2,0])/(4*w),(r[1,0]-r[0,1])/(4*w)]);pos.append(p)
    names=[name for name in names if name in {s.link for s in load_self_collisions168(URDF)}]
    return names,pos,quat


def setup(n,total):
    specs=load_self_collisions168(URDF);names,pos,quat=nominal_bodies();p=total-len(specs);device='cuda'
    centres=[[0.,0.,-.02],[4.,0.,1.],[-4.,0.,1.],[0.,4.,1.],[0.,-4.,1.]]
    half=[[10.,10.,.02],[.1,4.,1.],[.1,4.,1.],[4.,.1,1.],[4.,.1,1.]];kind=[BOX]*5
    for i in range(p-5):
        a=2*math.pi*i/(p-5);centres.append([2.*math.cos(a),2.*math.sin(a),.5]);half.append([.025,.025,.5]);kind.append(CYLINDER)
    tensor=lambda v:torch.tensor(v,device=device,dtype=torch.float32)
    g=SensorPrimitives(tensor(centres)[None].expand(n,-1,-1).clone(),torch.eye(3,device=device)[None,None].expand(n,p,3,3).clone(),tensor(half)[None].expand(n,-1,-1).clone(),torch.tensor(kind,device=device)[None].expand(n,-1).clone(),torch.ones((n,p),device=device,dtype=torch.bool),torch.ones(n,device=device,dtype=torch.bool))
    pos=tensor(pos)[None].expand(n,-1,-1).clone();quat=tensor(quat)[None].expand(n,-1,-1).clone()
    sample=torch.zeros((n,4),device=device);sample[:,2]=.55;ids=torch.ones(n,device=device,dtype=torch.long);blocks=d4xx_blocks168(device=device)
    def capture(aperture,backend='triton'):
        snap=primitive_snapshot168(g,pos,quat,names,specs,sample,scene_capture_id=ids,body_capture_id=ids,camera_capture_id=ids)
        k,r,t=vendor_cameras168(pos[:,0],quat[:,0],sample)
        return produce_blocks168(snap,k,r,t,blocks,backend=backend,self_trunk_aperture=aperture,context='training_sim')
    return capture


def main():
    torch.set_num_threads(1);results=[]
    for total in (32,64):
        capture=setup(4096,total)
        for aperture in (False,True):
            for _ in range(3):q=capture(aperture)
            torch.cuda.synchronize();del q;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter();q=capture(aperture);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            item=dict(envs=4096,blocks_per_env=240,primitives_per_env=total,self_colliders=17,self_trunk_aperture=aperture,repetitions=20,median_capture_ms=float(np.median(times)),p95_capture_ms=float(np.percentile(times,95)),max_capture_ms=max(times),estimated_10_captures_iter_s=float(np.median(times)/100),valid_fraction=float(q.volume_valid.float().mean()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,cutoff_min_m=float(q.cutoff_m.min()),cutoff_max_m=float(q.cutoff_m.max()))
            results.append(item);print(json.dumps(item),flush=True);del q
        del capture;torch.cuda.empty_cache()
    payload=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,scope='synthetic primitive snapshot sensor ONLY; adapter+K/extrinsics+validation+query+FreeFrusta; not rollout',timing_includes_compilation=False,results=results)
    (ROOT/'performance168.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
