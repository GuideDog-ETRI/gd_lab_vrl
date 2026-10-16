"""UPDATE126 range LiDAR, actual Isaac/Warp terrain + moving URDF collision meshes.
Sensor data only: no terrain family, gap slot, deck label or IR enters mapper.
"""
import os,time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from heightmap116 import rotation, R, ElevationMap
from heightmap119z import BatchMap as H0
import structure130

MOUNT=np.array([.33,0.,-.06])
THETA=np.deg2rad(50.)
# Columns: OpenGL right, up, backward in trunk coordinates.
R_GL=np.array([[0,np.sin(THETA),-np.cos(THETA)],[-1,0,0],[0,np.cos(THETA),np.sin(THETA)]])
Q_GL=Rotation.from_matrix(R_GL).as_quat()[[3,0,1,2]]
# ROS optical right/down/forward = GL right/-up/-backward.
R_ROS=R_GL@np.diag([1.,-1.,-1.])
Q_ROS=Rotation.from_matrix(R_ROS).as_quat()[[3,0,1,2]]

def local_rays():
    elev,az=np.meshgrid(np.deg2rad(20.25+np.arange(120)*.5),np.deg2rad(-22.25+np.arange(90)*.5),indexing='ij')
    body=np.stack([np.cos(elev)*np.cos(az),np.cos(elev)*np.sin(az),-np.sin(elev)],-1)
    return body@R_GL

SENSOR_COUNT=int(os.environ.get('U127_SENSOR_COUNT','2'))
assert SENSOR_COUNT in (1,2)
MOUNTS=np.array([[.33,.08,-.06],[.33,-.08,-.06]]) if SENSOR_COUNT==2 else MOUNT[None].copy()
YAWS=np.array([18.,-18.]) if SENSOR_COUNT==2 else np.array([0.])
ROT_YAW=Rotation.from_euler('z',YAWS,degrees=True).as_matrix()
R_GLS=ROT_YAW@R_GL
R_ROSS=R_GLS@np.diag([1.,-1.,-1.])
Q_GLS=Rotation.from_matrix(R_GLS).as_quat()[:,[3,0,1,2]]
Q_ROSS=Rotation.from_matrix(R_ROSS).as_quat()[:,[3,0,1,2]]

def mounted(pos,quat):
    rb=rotation(quat)
    orig=pos[:,None,:]+np.einsum('nij,sj->nsi',rb,MOUNTS)
    gl=np.einsum('nij,sjk->nsik',rb,R_GLS)
    rays=np.einsum('hwj,nsij->nshwi',local_rays(),gl)
    return orig,rays

def cast(mesh,starts,dirs,device,max_dist=10.):
    import torch
    from isaaclab.utils.warp import raycast_mesh
    a=torch.as_tensor(np.ascontiguousarray(starts),device=device,dtype=torch.float32)
    d=torch.as_tensor(np.ascontiguousarray(dirs),device=device,dtype=torch.float32)
    _,distance,_,_=raycast_mesh(a,d,mesh,max_dist=max_dist,return_distance=True)
    return distance.cpu().numpy()

class RobotMesh:
    """Triangle versions of all URDF collision boxes/spheres, live link poses.
    Uses each env's robot separately so neighboring replicated envs cannot occlude.
    """
    def __init__(self,robot,device,urdf):
        import trimesh
        from isaaclab.utils.warp import convert_to_warp_mesh
        self.robot=robot;self.device=device;self.items=[];self.faces=[];self.leg_faces=[];offset=0
        for link in ET.parse(urdf).getroot().findall('link'):
            name=link.get('name')
            for c in link.findall('collision'):
                assert name in robot.body_names, 'Missing collision body '+name
                g=c.find('geometry');box=g.find('box');sphere=g.find('sphere')
                if box is not None:m=trimesh.creation.box(extents=np.fromstring(box.get('size'),sep=' '))
                elif sphere is not None:m=trimesh.creation.icosphere(subdivisions=2,radius=float(sphere.get('radius')))
                else:raise ValueError('Unsupported actual URDF collision '+name)
                o=c.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ')
                verts=np.asarray(m.vertices)@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz
                self.items.append((robot.body_names.index(name),verts,name));self.faces.append(np.asarray(m.faces)+offset)
                self.leg_faces.extend([name!='trunk']*len(m.faces));offset+=len(verts)
        self.faces=np.concatenate(self.faces).astype(np.int32)
        self.convert=convert_to_warp_mesh
    def distances(self,starts,dirs):
        pos=self.robot.data.body_pos_w.cpu().numpy();rot=rotation(self.robot.data.body_quat_w.cpu().numpy())
        out=[];legs=[]
        # Individual mesh refit is unnecessary: fresh bounded meshes keep capture pose exact.
        from isaaclab.utils.warp import raycast_mesh
        import torch
        for e in range(len(starts)):
            verts=np.concatenate([v@rot[e,i].T+pos[e,i] for i,v,_ in self.items])
            mesh=self.convert(verts,self.faces,self.device)
            # Ignore numerical self-intersection at the flush-mounted trunk surface only.
            a=starts[e:e+1]+dirs[e:e+1]*1e-4
            _,dist,_,face=raycast_mesh(torch.as_tensor(a,device=self.device,dtype=torch.float32),torch.as_tensor(dirs[e:e+1],device=self.device,dtype=torch.float32),mesh,max_dist=10.,return_distance=True,return_face_id=True)
            dist=dist.cpu().numpy()[0]+1e-4;face=face.cpu().numpy()[0]
            leg=np.asarray(self.leg_faces)[np.clip(face,0,len(self.leg_faces)-1)]&np.isfinite(dist)
            out.append(dist);legs.append(leg)
        return np.array(out),np.array(legs)

class Lidar:
    def __init__(self,base):
        self.base=base;scanner=base.scene['height_scanner'];self.mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
        self.robot=base.scene['robot'];self.trunk=self.robot.body_names.index('trunk')
        urdf=Path(__file__).resolve().parents[2]/'assets/robots/rbq10/urdf/rbq10_simple.urdf'
        self.body=RobotMesh(self.robot,base.device,urdf);self.counts={};self.times=[];self.steps=[]
    def add(self,k,v):self.counts[k]=self.counts.get(k,0)+int(v)
    def capture(self,episodes,step):
        begin=time.perf_counter();data=self.robot.data
        bp=data.root_pos_w.cpu().numpy().copy();bq=data.root_quat_w.cpu().numpy().copy()
        p=data.body_pos_w[:,self.trunk].cpu().numpy();q=data.body_quat_w[:,self.trunk].cpu().numpy()
        origins,dirs=mounted(p,q);shape=dirs.shape[:-1];starts=np.broadcast_to(origins[:,:,None,None,:],dirs.shape)
        flatdirs=dirs.reshape(len(p),-1,3);flatstarts=starts.reshape(len(p),-1,3)
        terrain=cast(self.mesh,flatstarts,flatdirs,self.base.device).reshape(shape)
        body,leg=self.body.distances(flatstarts,flatdirs);body=body.reshape(shape);leg=leg.reshape(shape)
        blocked=np.isfinite(body)&(body<terrain);distance=np.minimum(body,terrain)
        hit=np.isfinite(distance)&(distance>=.1)&(distance<=10.)
        valid=hit.copy();noisy=np.where(hit,distance,10.).copy()
        for e,episode in enumerate(episodes):
            rng=np.random.default_rng(np.random.SeedSequence([42,e,int(episode),int(step),126]))
            noisy[e]+=rng.normal(0,.01,noisy[e].shape)
            drop=rng.random(noisy[e].shape)<.01
            valid[e]&=~drop&(noisy[e]>=.1)&(noisy[e]<=10.)
            self.add('random_return_drop',drop.sum())
        noisy[~valid]=10.
        assert np.isfinite(noisy).all() and np.isfinite(dirs).all() and np.isfinite(origins).all()
        self.add('beams',hit.size);self.add('geometric_hits',hit.sum());self.add('valid_returns',valid.sum())
        self.add('body_occluded',blocked.sum());self.add('leg_occluded',(blocked&leg).sum())
        self.add('trunk_occluded',(blocked&~leg).sum());self.steps.append(int(step));self.times.append(time.perf_counter()-begin)
        # UPDATE130 v2.3: returns on the robot's own links (URDF + live joint state), the self-filter a real
        # driver applies; only the L0S structure path uses it, h0 reducers keep their torso envelope.
        return dict(origins=origins,dirs=dirs,ranges=noisy,valid=valid,body_pos=bp,body_quat=bq,self_hit=blocked&valid)
    def report(self):return dict(counts=self.counts,capture_steps=self.steps,capture_seconds=sum(self.times),capture_ms_mean=1000*np.mean(self.times),collision_model='All 17 URDF collision primitives: boxes and subdivision-2 foot spheres; per-env live link triangle raycast',q_gl_wxyz=Q_GLS.tolist(),q_ros_wxyz=Q_ROSS.tolist(),mounts_trunk_m=MOUNTS.tolist(),yaw_deg=YAWS.tolist(),sensors_per_packet=SENSOR_COUNT)

# UPDATE129 deep memory rule, fixed a priori: direct hit >=10 cm below local support is deep evidence;
# kept (deepest) until the cell is behind the hind feet (body x < -0.45 m; hind hip x = -0.31) or 10 s old.
DEEP_BELOW_M=.10;HIND_PASS_X=-.45;DEEP_MAX_AGE_S=10.
# Fixed a priori carving certificate, never retuned from rollout results.
CARVING=dict(below_support_m=.05,neighbor_radius_m=.35,min_points=6,min_xy_span_m=.10,max_rms_m=.02,max_slope=.75,max_prediction_sigma_m=.02,range_clearance_m=.03)

def traversed_minima(origin,dirs,ranges,valid,map_origin):
    """Exact 2D voxel traversal; minimum beam z within each traversed segment.
    No-return rays are excluded. Stop 3 sigma before a noisy hit; no hit extension.
    """
    d=dirs[valid].reshape(-1,3);ends=np.maximum(0,ranges[valid]-.03)
    out=np.full(R.size*R.size,np.inf)
    if not len(d):return out
    xymin=map_origin*R.resolution;xymax=(map_origin+R.size)*R.resolution
    safe=np.where(abs(d[:,:2])>1e-12,d[:,:2],1e-12)
    a=(xymin-origin[:2])/safe;b=(xymax-origin[:2])/safe
    start=np.maximum(0,np.minimum(a,b).max(1));end=np.minimum(ends,np.maximum(a,b).min(1))
    active=end>start+1e-7;t=start.copy()
    ij=np.floor((origin[:2]+d[:,:2]*(t[:,None]+1e-7))/R.resolution).astype(int)-map_origin
    direction=np.where(d[:,:2]>=0,1,-1)
    for _ in range(2*R.size+2):
        ids=np.flatnonzero(active)
        if not len(ids):break
        ii=ij[ids];dd=d[ids];tt=t[ids]
        boundary=(ii+map_origin+(direction[ids]>0))*R.resolution
        nextxy=(boundary-origin[:2])/safe[ids]
        nxt=np.minimum(nextxy.min(1),end[ids]);nxt=np.maximum(nxt,tt)
        z=np.minimum(origin[2]+dd[:,2]*tt,origin[2]+dd[:,2]*nxt)
        ok=(ii>=0).all(1)&(ii<R.size).all(1)&(nxt>tt+1e-8)
        np.minimum.at(out,ii[ok,1]*R.size+ii[ok,0],z[ok])
        crossed=nextxy<=nxt[:,None]+1e-8
        ij[ids]+=direction[ids]*crossed;t[ids]=nxt
        active[ids]=(nxt<end[ids]-1e-7)&(ij[ids]>=0).all(1)&(ij[ids]<R.size).all(1)
    return out

class Maps:
    def __init__(self,n,arm):
        self.direct=H0(n,{'L0D':'h0d','L0D2':'h0d2','L0P':'h0p','L0S':'h0d'}.get(arm,'h0'))
        # UPDATE130 L0S: L0D mapper + point-structure GAP/pending overlay (structure130), per-env world memory.
        self.struct=[structure130.StructMemory() for _ in range(n)];self.struct_times=[];self.struct_counts=dict(evidence=0,gap_bins=0,pending_points=0)
        # UPDATE130 raw capture dump for offline replay (evaluation diagnostics only; never fed back).
        self.dump_envs=[int(x) for x in os.environ.get('U130_DUMP','').split(',') if x];self.dump=[];self.maps=self.direct.maps;self.carves=[ElevationMap('h0') for _ in range(n)];self.arm=arm
        self.counts=dict(carved_updates=0,uncertain_candidates=0);self.depths=[];self.last_sources=None
        # UPDATE129 L0M: deep evidence kept per world cell until the hind feet pass.
        self.vis=[ElevationMap('h0') for _ in range(n)]  # UPDATE129 G/GV: last LiDAR hit time per world 5 cm cell
        self.deep=[{} for _ in range(n)];self.counts.update(deep_kept_updates=0,deep_overrides=0)
    def reset(self,ids):
        self.direct.reset(ids)
        for e in ids:self.carves[e].reset()
        for e in ids:self.deep[e].clear()
        for e in ids:self.vis[e].reset()
        for e in ids:self.struct[e].reset()
    def integrate(self,ids,packet,stamp,now):
        for e in ids:
            m=self.maps[e];c=self.carves[e]
            if stamp<=m.last_capture:continue
            origin=packet['origins'][e];dirs=packet['dirs'][e];ranges=packet['ranges'][e];valid=packet['valid'][e]
            m.precomputed_rays=dirs
            # Exact existing h0 reducer/filter/memory update, substituting unit LiDAR rays/ranges.
            m.integrate(ranges,valid,np.zeros(valid.shape),origin,None,None,packet['body_pos'][e],packet['body_quat'][e],stamp)
            m.precomputed_rays=None
            if self.arm=='L0S':
                if e in self.dump_envs:
                    self.dump.append(dict(env=e,stamp=stamp,now=now,origins=origin.astype(np.float32),ranges=ranges.astype(np.float32),valid=valid,self_hit=packet['self_hit'][e] if 'self_hit' in packet else None,dirs=dirs.astype(np.float16),body_pos=packet['body_pos'][e],body_quat=packet['body_quat'][e],base_ref=m.reference(packet['body_pos'][e][:2],packet['body_pos'][e][2])))
                t0=time.perf_counter();bp=packet['body_pos'][e]
                res=structure130.analyze(origin,dirs,ranges,valid,bp,packet['body_quat'][e],base_ref=m.reference(bp[:2],bp[2]),support_mem=self.struct[e],stamp=stamp,self_hit=packet['self_hit'][e] if 'self_hit' in packet else None)
                self.struct[e].update(res,stamp);self.struct_times.append(time.perf_counter()-t0)
                self.struct_counts['evidence']+=res['n_evidence'];self.struct_counts['gap_bins']+=len(res['gap_xy']);self.struct_counts['pending_points']+=len(res['pending_xy'])
                continue
            if self.arm in ('G','GV'):
                v=self.vis[e];v.recenter(packet['body_pos'][e,:2])
                pts=(origin[:,None,None,:]+dirs*ranges[...,None])[valid]
                ij,ins=v.indices(pts[:,:2]);v.last[ij[ins,1],ij[ins,0]]=stamp
            if self.arm=='L0M':
                ref=m.reference(packet['body_pos'][e,:2],packet['body_pos'][e,2])
                fresh=(m.last==stamp)&np.isfinite(m.z)
                for j,i in zip(*np.nonzero(fresh&(m.z<ref-DEEP_BELOW_M))):
                    key=(int(i+m.origin[0]),int(j+m.origin[1]));old=self.deep[e].get(key)
                    self.deep[e][key]=(min(float(m.z[j,i]),old[0]) if old else float(m.z[j,i]),stamp);self.counts['deep_kept_updates']+=1
            if self.arm in ('L0','L0M','L0D','L0D2','L0P','G','GV'):continue
            c.recenter(packet['body_pos'][e,:2]);points=origin[:,None,None,:]+dirs*ranges[...,None]
            ij,inside=m.indices(points[...,:2]);hit=valid&inside
            direct_cells=np.unique(ij[...,1][hit]*R.size+ij[...,0][hit])
            c.z.ravel()[direct_cells]=np.nan;c.last.ravel()[direct_cells]=-np.inf
            minimum=np.minimum.reduce([traversed_minima(origin[v],dirs[v],ranges[v],valid[v],m.origin) for v in range(len(origin))])
            minimum[direct_cells]=np.inf
            fresh=(stamp-m.finite<=R.ttl)&np.isfinite(m.z)&(m.var<.0025)
            minimum[fresh.ravel()]=np.inf
            candidates=np.flatnonzero(np.isfinite(minimum))
            yy,xx=np.nonzero(fresh);coords=np.c_[xx+m.origin[0]+.5,yy+m.origin[1]+.5]*R.resolution
            if len(coords)<6:self.counts['uncertain_candidates']+=len(candidates);continue
            tree=cKDTree(coords);cj,ci=np.divmod(candidates,R.size);query=np.c_[ci+m.origin[0]+.5,cj+m.origin[1]+.5]*R.resolution
            neighborhoods=tree.query_ball_point(query,CARVING['neighbor_radius_m'])
            for idx,xy,neighbors in zip(candidates,query,neighborhoods):
                if len(neighbors)<6:self.counts['uncertain_candidates']+=1;continue
                neighbors=np.asarray(neighbors);xyobs=coords[neighbors];zobs=m.z[yy[neighbors],xx[neighbors]]
                # All surrounding certified direct heights must form one well-supported plane.
                a=np.c_[xyobs-xy,np.ones(len(neighbors))]
                if np.min(np.ptp(xyobs,axis=0))<.10 or np.linalg.matrix_rank(a)<3:self.counts['uncertain_candidates']+=1;continue
                coef=np.linalg.lstsq(a,zobs,rcond=None)[0];rms=np.sqrt(np.mean((a@coef-zobs)**2))
                sigma=np.sqrt(np.linalg.pinv(a.T@a)[2,2]*max(rms*rms,1e-6))
                if rms>.02 or np.linalg.norm(coef[:2])>.75 or sigma>.02:self.counts['uncertain_candidates']+=1;continue
                z=minimum[idx];depth=coef[2]-z
                if depth<.05:continue
                j,i=divmod(int(idx),R.size)
                # Keep the strongest still-fresh free-space bound; direct hits always win.
                if stamp-c.last[j,i]<=R.ttl and np.isfinite(c.z[j,i]):z=min(z,c.z[j,i])
                c.z[j,i]=z;c.var[j,i]=0.;c.last[j,i]=stamp
                self.depths.append(float(depth));self.counts['carved_updates']+=1
    def visible(self,poses,stamp):
        """Teacher cells whose 10 cm footprint (4 sub-points) got a LiDAR hit within the 2 s TTL."""
        y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij');bx=x.ravel();by=y.ravel()
        out=np.zeros((len(poses),187),bool)
        for e,v in enumerate(self.vis):
            if v.origin is None:continue
            c,s_=np.cos(poses[e,2]),np.sin(poses[e,2])
            for dx,dy in ((-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)):
                ox=bx+dx;oy=by+dy;xy=np.stack((poses[e,0]+c*ox-s_*oy,poses[e,1]+s_*ox+c*oy),-1)
                ij,ins=v.indices(xy);i=ij[:,0].clip(0,R.size-1);j=ij[:,1].clip(0,R.size-1)
                out[e]|=ins&(stamp-v.last[j,i]<=R.ttl)
        return out
    def sample(self,poses,scanner,stamp):
        packed=self.direct.sample(poses,scanner,stamp);source=(packed[:,187:]>0).astype(np.uint8)
        if self.arm in ('L1','L2'):
            for e,c in enumerate(self.carves):
                if c.origin is None:continue
                alt,_=c.sample(poses[e],scanner[e],stamp);use=(source[e]==0)&(alt[187:]>0)
                packed[e,:187][use]=5.0 if self.arm=='L2' else alt[:187][use];packed[e,187:][use]=1;source[e,use]=2
        if self.arm=='L0M':
            y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij');bx=x.ravel();by=y.ravel()
            for e in range(len(poses)):
                d=self.deep[e]
                if not d:continue
                c,s_=np.cos(poses[e,2]),np.sin(poses[e,2]);wx=poses[e,0]+c*bx-s_*by;wy=poses[e,1]+s_*bx+c*by
                gi=np.floor(wx/R.resolution+1e-7).astype(int);gj=np.floor(wy/R.resolution+1e-7).astype(int)
                for cell in range(187):
                    hit=d.get((int(gi[cell]),int(gj[cell])))
                    if hit is None or bx[cell]<HIND_PASS_X or stamp-hit[1]>DEEP_MAX_AGE_S:continue
                    h=5*np.clip(scanner[e]-hit[0]-.5,-1,1)
                    if source[e,cell]==0 or packed[e,cell]<h-.25:
                        packed[e,cell]=h;packed[e,187+cell]=1;source[e,cell]=3;self.counts['deep_overrides']+=1
                # forget cells the hind feet have passed (body x < HIND_PASS_X)
                for key in [k for k in d if ((k[0]+.5)*R.resolution-poses[e,0])*np.cos(poses[e,2])+((k[1]+.5)*R.resolution-poses[e,1])*np.sin(poses[e,2])<HIND_PASS_X]:del d[key]
        if self.arm=='L0S':
            for e in range(len(poses)):self.struct[e].overlay(packed[e],poses[e],scanner[e],stamp,source[e])
        assert np.isfinite(packed).all()
        self.last_sources=source
        return packed
    def report(self):
        import hashlib
        st=dict(code_sha256=hashlib.sha256(open(structure130.__file__,'rb').read()).hexdigest(),rules=structure130.S,counts=self.struct_counts,memory={k:sum(m.counts[k] for m in self.struct) for k in self.struct[0].counts},
                ms_per_env_capture=dict(n=len(self.struct_times),mean=1000*float(np.mean(self.struct_times)),p95=1000*float(np.quantile(self.struct_times,.95)),max=1000*float(np.max(self.struct_times))) if self.struct_times else None)
        return dict(structure130=st,counts=self.counts,carving_depth_m=dict(n=len(self.depths),quantiles=dict(zip(['min','p05','p25','p50','p75','p95','max'],np.quantile(self.depths,[0,.05,.25,.5,.75,.95,1]).tolist())) if self.depths else None),rules=CARVING)

# UPDATE145 opt-in observation hooks; Maps(..., provenance=None) is default OFF.
from provenance145 import install_maps as _install_provenance
_install_provenance(Maps)
