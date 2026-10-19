import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from gd_lab.gast_nav.producer168 import SensorPrimitives,produce_blocks168,BOX,CYLINDER,SPHERE
from gd_lab.gast_nav.producer168_adapter import load_self_collisions168,vendor_cameras168,primitive_snapshot168,d4xx_blocks168

ROOT=Path(__file__).resolve().parents[2]/'logs/gast/nav_v1/update168_producer'
ROOT.mkdir(parents=True,exist_ok=True)
URDF=str(Path(__file__).resolve().parents[2]/'assets/robots/rbq10/urdf/rbq10_simple.urdf')


def scene(centres,half,kind=None,rotation=None,trunk=None,dtype=torch.float64):
    centres=torch.tensor(centres,dtype=dtype)
    if centres.ndim==2:centres=centres[None]
    n,p,_=centres.shape
    half=torch.tensor(half,dtype=dtype).expand(n,p,3).clone()
    rotation=torch.eye(3,dtype=dtype).expand(n,p,3,3).clone() if rotation is None else torch.tensor(rotation,dtype=dtype).expand(n,p,3,3).clone()
    kind=torch.zeros((n,p),dtype=torch.long) if kind is None else torch.tensor(kind,dtype=torch.long).expand(n,p).clone()
    return SensorPrimitives(centres,rotation,half,kind,torch.ones((n,p),dtype=torch.bool),torch.ones(n,dtype=torch.bool),None if trunk is None else torch.tensor(trunk,dtype=torch.bool).expand(n,p))


def query(g,blocks=None,**kwargs):
    n=len(g.centres);dtype=g.centres.dtype;dev=g.centres.device
    k=torch.tensor([[[40.,0,40],[0,40,22.5],[0,0,1]]],dtype=dtype,device=dev)
    r=torch.eye(3,dtype=dtype,device=dev).expand(n,1,3,3);t=torch.zeros((n,1,3),dtype=dtype,device=dev)
    blocks=torch.tensor([[0,36,18,44,26]],device=dev) if blocks is None else blocks.to(dev)
    return produce_blocks168(g,k,r,t,blocks,**kwargs)


def oracle_rays(centres,half,rotations,kinds,u,v):
    """Independent NumPy double per-ray exact surface oracle (not SAT/proxy)."""
    d=np.stack((u,v,np.ones_like(u)),axis=-1);best=np.full(len(u),np.inf)
    for centre,h,rot,kind in zip(centres,half,rotations,kinds):
        o=-np.asarray(centre)@rot;dl=d@rot
        candidates=[]
        if kind==BOX:
            entry=np.full(len(u),-np.inf);exit=np.full(len(u),np.inf);ok=np.ones(len(u),bool)
            for j in range(3):
                nonzero=np.abs(dl[:,j])>1e-14
                inv=np.divide(1.,dl[:,j],out=np.zeros(len(u)),where=nonzero)
                a=(-h[j]-o[j])*inv;b=(h[j]-o[j])*inv
                entry=np.maximum(entry,np.where(nonzero,np.minimum(a,b),-np.inf));exit=np.minimum(exit,np.where(nonzero,np.maximum(a,b),np.inf))
                ok &= nonzero|(abs(o[j])<=h[j])
            candidates=[np.where(ok&(exit>=entry)&(entry>=0),entry,np.inf),np.where(ok&(exit>=entry)&(exit>=0),exit,np.inf)]
        else:
            dims=3 if kind==SPHERE else 2
            a=(dl[:,:dims]**2).sum(-1);b=(dl[:,:dims]*o[:dims]).sum(-1);c=(o[:dims]**2).sum()-h[0]**2
            disc=b*b-a*c
            for sign in (-1,1):
                t=(-b+sign*np.sqrt(np.maximum(disc,0)))/np.maximum(a,1e-30)
                ok=(disc>=0)&(a>1e-14)&(t>=0)
                if kind==CYLINDER:ok &= abs(o[2]+t*dl[:,2])<=h[2]
                candidates.append(np.where(ok,t,np.inf))
            if kind==CYLINDER:
                for sign in (-1,1):
                    t=np.divide(sign*h[2]-o[2],dl[:,2],out=np.full(len(u),np.inf),where=abs(dl[:,2])>1e-14)
                    with np.errstate(invalid='ignore'):
                        xy=o[:2]+t[:,None]*dl[:,:2]
                    candidates.append(np.where((t>=0)&((xy**2).sum(-1)<=h[0]**2),t,np.inf))
        best=np.minimum(best,np.min(candidates,axis=0))
    return best


def test_aperture_default_near_and_other_links():
    g=scene([[0,0,0],[0,0,3]],[[.2,.2,.3],[5,5,.1]],trunk=[True,False])
    assert not query(g).volume_valid.any()
    q=query(g,self_trunk_aperture=True,context='training_sim')
    assert q.volume_valid.all() and q.cutoff_m.item()>=.3
    assert abs(q.depth_lower_m.item()-2.9)<.0001
    assert torch.equal(q.cutoff_m,q.frusta.offsets[:,:,4])
    with pytest.raises(ValueError):query(g,self_trunk_aperture=True)
    near=query(g,self_trunk_aperture=True,context='training_sim',near_m=.5)
    assert near.cutoff_m.item()==.5
    for kind in (BOX,CYLINDER,SPHERE):
        blocked=scene([[0,0,0],[0,0,.08],[0,0,3]],[[.2,.2,.3],[.03,.03,.03],[5,5,.1]],kind=[BOX,kind,BOX],trunk=[True,False,False])
        assert not query(blocked,self_trunk_aperture=True,context='training_sim').volume_valid.any()


def test_trunk_encounter_outside_origin_and_wide_block_exit():
    outside=scene([[0,0,1],[0,0,3]],[[.2,.2,.1],[5,5,.1]],trunk=[True,False])
    q=query(outside,self_trunk_aperture=True,context='training_sim')
    assert q.volume_valid.all() and q.depth_lower_m.item()<.9
    # Oblique box: whole-block last exit, not centre-ray exit.
    a=.7;c,s=math.cos(a),math.sin(a);rot=[[c,0,s],[0,1,0],[-s,0,c]]
    g=scene([[0,0,0],[0,0,3]],[[.3,.2,.2],[5,5,.1]],rotation=[rot,np.eye(3).tolist()],trunk=[True,False])
    q=query(g,self_trunk_aperture=True,context='training_sim')
    u,v=np.meshgrid(np.linspace(-.1125,.0875,65),np.linspace(-.125,.075,65))
    exits=oracle_rays(g.centres[0,:1].numpy(),g.half[0,:1].numpy(),g.rotations[0,:1].numpy(),[BOX],u.ravel(),v.ravel())
    assert exits.max()<=q.cutoff_m.item()
    assert q.volume_valid.all()


def test_invalid_and_no_return():
    g=scene([[20,0,2]],[[.1,.1,.1]])
    assert not query(g).volume_valid.any()
    g=scene([[0,0,3]],[[5,5,.1]],kind=[7])
    assert not query(g).volume_valid.any()
    g=scene([[0,0,3]],[[5,5,.1]])
    g.complete[:]=False
    assert not query(g).volume_valid.any()
    g=scene([[0,0,.12],[0,0,3]],[[.1,.1,.01],[5,5,.1]])
    assert not query(g).volume_valid.any() # before near cannot carve behind it


def test_wall_hidden_person():
    a=query(scene([[0,0,1.5],[0,0,2]],[[4,4,.05],[.025,.025,.3]],kind=[BOX,CYLINDER]))
    b=query(scene([[0,0,1.5]],[[4,4,.05]]))
    assert torch.equal(a.depth_lower_m,b.depth_lower_m) and a.depth_lower_m.item()<1.45


def fixture_sweep():
    records=[]
    for block in ([0,36,18,44,26],[0,0,0,8,8],[0,72,37,80,45]):
        x0=(block[1]-.5-40)/40;x1=(block[3]-.5-40)/40
        y0=(block[2]-.5-22.5)/40;y1=(block[4]-.5-22.5)/40
        for distance in (.2,.5,1.,2.,4.):
            for fx in (-.03,.01,.19,.47,.79,.99,1.03):
                for fy in (.03,.46,.97):
                    for kind in (BOX,CYLINDER,SPHERE):
                        u=x0+(x1-x0)*fx;v=y0+(y1-y0)*fy
                        half=[.025,.015,.025] if kind==BOX else ([.025,.025,.015] if kind==CYLINDER else [.025]*3)
                        a=.31;rot=[[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]]
                        if kind==CYLINDER:rot=[[1,0,0],[0,0,-1],[0,1,0]] # vertical in optical y
                        records.append((block,[[u*distance,v*distance,distance],[0,0,4.8]],[half,[20,20,.1]],[kind,BOX],[rot,np.eye(3).tolist()]))
    return records


def test_independent_subpixel_distance_fov_sweep():
    records=fixture_sweep();valid=0;miss=0;covered=0;aliased=0;max_slack=0.;by_shape={}
    # Distinct blocks handled as separate groups; 315 env/group, 945 total.
    for start in range(0,len(records),315):
        rows=records[start:start+315];g=scene([r[1] for r in rows],[r[2] for r in rows],[r[3] for r in rows],[r[4] for r in rows])
        block=rows[0][0];q=query(g,blocks=torch.tensor([block]))
        uu,vv=np.meshgrid(np.linspace((block[1]-.5-40)/40,(block[3]-.5-40)/40,81),np.linspace((block[2]-.5-22.5)/40,(block[4]-.5-22.5)/40,81))
        for i,row in enumerate(rows):
            truth=oracle_rays(g.centres[i].numpy(),g.half[i].numpy(),g.rotations[i].numpy(),row[3],uu.ravel(),vv.ravel()).min()
            if truth<4.69:
                covered+=1
                corners=oracle_rays(g.centres[i,:1].numpy(),g.half[i,:1].numpy(),g.rotations[i,:1].numpy(),row[3][:1],np.array([uu[0,0],uu[0,-1],uu[-1,0],uu[-1,-1],(uu[0,0]+uu[0,-1])/2]),np.array([vv[0,0],vv[0,-1],vv[-1,0],vv[-1,-1],(vv[0,0]+vv[-1,0])/2]))
                aliased+=int(np.isinf(corners).all())
            if q.volume_valid[i,0]:
                valid+=1;bound=q.depth_lower_m[i,0].item();miss+=int(bound>truth+1e-8);max_slack=max(max_slack,float(truth-bound));by_shape[str(row[3][0])]=by_shape.get(str(row[3][0]),0)+1
    result=dict(cases=len(records),valid=valid,valid_fraction=valid/len(records),dense_oracle_rays_per_case=81**2,foreground_hit_cases=covered,foreground_missed_by_all_five_witness_rays=aliased,missed_first_hit=miss,max_conservative_slack_m=max_slack,valid_by_kind=by_shape)
    (ROOT/'fixture168_results.json').write_text(json.dumps(result,indent=2)+'\n')
    assert miss==0 and valid>len(records)*.7 and covered>len(records)*.8


def test_adapter_all_self_colliders_and_sync():
    specs=load_self_collisions168(URDF)
    assert len(specs)==17 and sum(s.link=='trunk' for s in specs)==1
    assert sum(s.kind==SPHERE for s in specs)==4
    names=sorted(set(s.link for s in specs));n=2
    pos=torch.zeros((n,len(names),3),dtype=torch.float64);quat=torch.zeros((n,len(names),4),dtype=torch.float64);quat[:,:,0]=1
    g=scene([[[0,0,-.1]],[[0,0,-.1]]],[[10,10,.1]])
    pose=torch.zeros((n,4),dtype=torch.float64);ids=torch.tensor([4,9])
    joined=primitive_snapshot168(g,pos,quat,names,specs,pose,scene_capture_id=ids,body_capture_id=ids,camera_capture_id=ids)
    assert joined.centres.shape==(2,18,3) and joined.is_trunk.sum()==2
    with pytest.raises(ValueError):primitive_snapshot168(g,pos,quat,names,specs,pose,scene_capture_id=ids,body_capture_id=ids+1,camera_capture_id=ids)
    k,r,t=vendor_cameras168(pos[:,0],quat[:,0],pose)
    assert k.shape==(4,3,3) and r.shape==(2,4,3,3)
    assert d4xx_blocks168().shape==(240,5)
    assert torch.allclose(torch.linalg.det(r),torch.ones((2,4),dtype=r.dtype))


def test_torch_cuda_parity():
    if not torch.cuda.is_available():pytest.skip('GPU only in performance/parity run')
    rows=fixture_sweep();g=scene([r[1] for r in rows],[r[2] for r in rows],[r[3] for r in rows],[r[4] for r in rows],dtype=torch.float32)
    gpu=SensorPrimitives(**{key:(value.cuda() if value is not None else None) for key,value in vars(g).items()})
    a=query(g);b=query(gpu,backend='triton')
    assert torch.equal(a.volume_valid,b.volume_valid.cpu())
    assert torch.allclose(a.depth_lower_m,b.depth_lower_m.cpu(),atol=2e-6,rtol=1e-6)
    for trunk in (False,True):
        h=scene([[0,0,0],[0,0,3]],[[.2,.2,.3],[5,5,.1]],trunk=[True,False],dtype=torch.float32)
        hg=SensorPrimitives(**{key:(value.cuda() if value is not None else None) for key,value in vars(h).items()})
        a=query(h,self_trunk_aperture=trunk,context='training_sim');b=query(hg,self_trunk_aperture=trunk,context='training_sim',backend='triton')
        assert torch.equal(a.volume_valid,b.volume_valid.cpu()) and torch.allclose(a.cutoff_m,b.cutoff_m.cpu())


def test_no_frusta_outside_image():
    g=scene([[0,0,3]],[[5,5,.1]])
    with pytest.raises(ValueError):query(g,blocks=torch.tensor([[0,72,37,81,45]]))


def test_producer_to_map_measurement_bridge():
    from gd_lab.gast_nav.free_evidence167 import FreeCapture,FreeMeasurement,FreeEvidenceConfig,volume_free_proof
    from gd_lab.gast_nav.map_types import Grid
    g=scene([[3,0,.4]],[[.1,5,5]],dtype=torch.float32)
    k=torch.tensor([[[40.,0,40],[0,40,22.5],[0,0,1]]])
    r=torch.tensor([[[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]]])
    t=torch.tensor([[[0.,0.,.4]]]);blocks=torch.tensor([[0,0,0,80,45]])
    q=produce_blocks168(g,k,r,t,blocks)
    assert q.volume_valid.all()
    grid=Grid(1,1,.1,1.,0.)
    support=torch.zeros(1,1,1);pose=torch.zeros(1,4);time=torch.tensor([1000]);epoch=torch.tensor([0])
    cap=FreeCapture(q.frusta,pose,support,time,epoch)
    m=FreeMeasurement((cap,),pose,time,epoch,torch.zeros(1,1),torch.zeros(1),torch.zeros(1),FreeEvidenceConfig())
    proof=volume_free_proof(support,m,grid=grid)
    assert proof.free[:,:2].all() # B1+B2 positive, with default error erosion
    # Unknown observed support cannot be filled by this producer's scene geometry.
    unknown=volume_free_proof(torch.full_like(support,torch.nan),m,grid=grid)
    assert not unknown.free.any()


def test_independent_arbitrary_camera_and_primitive_rotations():
    rng=np.random.default_rng(168);n=96
    def rotations(count):
        result=[]
        for _ in range(count):
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(q)<0:q[:,0]*=-1
            result.append(q)
        return np.array(result)
    camera_r=rotations(n);camera_t=rng.normal(size=(n,3))
    primitive_r=rotations(n*2).reshape(n,2,3,3)
    cc=np.zeros((n,2,3));cc[:,0,2]=rng.uniform(.3,4.,n);cc[:,0,:2]=rng.uniform(-.08,.08,(n,2))*cc[:,0,2,None];cc[:,1,2]=4.8
    h=np.empty((n,2,3));h[:,0]=rng.uniform(.015,.04,(n,3));h[:,1]=[20,20,.1]
    primitive_r[:,1]=np.eye(3)
    centres=np.einsum('nij,npj->npi',camera_r,cc)+camera_t[:,None]
    axes=np.einsum('nij,npjk->npik',camera_r,primitive_r)
    g=scene(centres,h,rotation=axes)
    k=torch.tensor([[[40.,0,40],[0,40,22.5],[0,0,1.]]],dtype=torch.float64)
    q=produce_blocks168(g,k,torch.tensor(camera_r[:,None]),torch.tensor(camera_t[:,None]),torch.tensor([[0,36,18,44,26]]))
    u,v=np.meshgrid(np.linspace(-.1125,.0875,81),np.linspace(-.125,.075,81))
    assert q.volume_valid.all()
    for i in range(n):
        truth=oracle_rays(cc[i],h[i],primitive_r[i],[BOX,BOX],u.ravel(),v.ravel()).min()
        assert q.depth_lower_m[i,0].item()<=truth+1e-8


def test_vendor_new_all_four_cameras_self_colliders_gpu_parity():
    if not torch.cuda.is_available():pytest.skip('GPU only in performance/parity run')
    from producer168_bench import setup
    capture=setup(16,32)
    for aperture in (False,True):
        a=capture(aperture,backend='torch');b=capture(aperture,backend='triton')
        assert torch.equal(a.volume_valid,b.volume_valid)
        assert torch.allclose(a.depth_lower_m,b.depth_lower_m,atol=2e-6,rtol=1e-6)
        assert torch.equal(a.cutoff_m,b.cutoff_m)
