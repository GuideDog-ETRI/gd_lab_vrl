"""One preflight suite, actual Isaac/Warp CPU triangle intersections (no mocks)."""
import sys,json,time,traceback
from pathlib import Path
from types import SimpleNamespace as NS
RDIR=Path(__file__).resolve().parent
sys.path[:0]=[str(RDIR/'src'),str(RDIR/'tools/rl_replay')]
import numpy as np,torch
from isaaclab.utils.warp import convert_to_warp_mesh
from lidar126 import mounted,cast,rotation,Q_GL,Q_ROS,R_GL,R_ROS,Q_GLS,Q_ROSS,R_GLS,YAWS,MOUNTS,Maps,RobotMesh,traversed_minima
from heightmap119z import BatchMap as H0
from gd_lab.core.camera_transport import CameraTransport,CameraTransportConfig,delivery_mask
O=RDIR/'outputs/update128';checks={};started=time.perf_counter()
def check(name,condition,evidence=None):
 checks[name]={'passed':bool(condition),'evidence':evidence}
 assert condition,name

def floor_mesh(gap=False):
 verts=[];faces=[]
 for a,b,z in ([(-20,.7,0),(.95,20,0),(.7,.95,-1)] if gap else [(-20,20,0)]):
  offset=len(verts);verts.extend([[a,-20,z],[b,-20,z],[b,20,z],[a,20,z]]);faces.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
 if gap:
  for x in [.7,.95]:
   offset=len(verts);verts.extend([[x,-20,-1],[x,20,-1],[x,20,0],[x,-20,0]]);faces.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
 return convert_to_warp_mesh(np.array(verts),np.array(faces),'cpu')
try:
 pos=np.array([[0.,0.,.56]]);quat=np.array([[1.,0.,0.,0.]])
 origin,dirs=mounted(pos,quat);starts=np.broadcast_to(origin[:,:,None,None],dirs.shape).reshape(1,-1,3)
 for v in range(2):
  ros=Q_ROSS[v];qgl=np.array([-ros[1],ros[0],ros[3],-ros[2]])
  check('ROS_OpenGL_conversion_'+str(v),np.allclose(rotation(qgl),R_GLS[v],atol=1e-12),{'q_ros_wxyz':ros.tolist(),'q_gl_wxyz':Q_GLS[v].tolist(),'mount':MOUNTS[v].tolist(),'yaw_deg':float(YAWS[v]),'conversion':'q_gl = q_ros * (0,1,0,0)'})
 check('unit_downward_angles',np.allclose(np.linalg.norm(dirs,axis=-1),1) and np.all(dirs[...,2]<0),{'elevation_center_degrees':[20.25,79.75],'azimuth_center_degrees':[-22.25,22.25],'beam_shape':[120,90],'spacing_degrees':.5})
 actual=cast(floor_mesh(),starts,dirs.reshape(1,-1,3),'cpu').reshape(1,2,120,90)
 expected=-origin[:,:,None,None,2]/dirs[...,2]
 valid=np.isfinite(actual)&(actual>=.1)&(actual<=10)
 error=float(np.max(abs(actual[valid]-expected[valid])))
 check('real_CPU_flat_raycast',valid.mean()>.8 and error<1e-4,{'backend':'isaaclab.utils.warp.raycast_mesh CPU','hit_n':int(valid.sum()),'beam_n':valid.size,'hit_fraction':float(valid.mean()),'range_quantiles_m':np.quantile(actual,[0,.05,.5,.95,1]).tolist(),'geometric_prediction_quantiles_m':np.quantile(expected,[0,.05,.5,.95,1]).tolist(),'max_abs_range_error_m':error})
 for v in range(2):
  av=actual[:,v];ev=expected[:,v];vv=valid[:,v]
  az=np.rad2deg(np.arctan2(dirs[0,v,...,1],dirs[0,v,...,0]))
  check('sensor_flat_'+str(v),vv.mean()>.8 and np.max(abs(av[vv]-ev[vv]))<1e-4 and np.allclose([az.min(),az.max()],YAWS[v]+np.array([-22.25,22.25])),dict(hit_n=int(vv.sum()),beam_n=vv.size,range_quantiles_m=np.quantile(av,[0,.05,.5,.95,1]).tolist(),max_abs_range_error_m=float(np.max(abs(av[vv]-ev[vv]))),azimuth_deg=[float(az.min()),float(az.max())]))
 # Actual URDF collision primitive triangle fixture: trunk at mount, front calf moved into FOV.
 import xml.etree.ElementTree as ET
 urdf=RDIR/'assets/robots/rbq10/urdf/rbq10_simple.urdf'
 names=[l.get('name') for l in ET.parse(urdf).getroot().findall('link') if l.findall('collision')]
 bodypos=np.full((1,len(names),3),20.);bodypos[0,names.index('trunk')]=pos[0];bodypos[0,names.index('FL_calf')]=[.65,.18,.30]
 bodyquat=np.zeros((1,len(names),4));bodyquat[...,0]=1
 robot=NS(body_names=names,data=NS(body_pos_w=torch.tensor(bodypos),body_quat_w=torch.tensor(bodyquat)))
 rm=RobotMesh(robot,'cpu',urdf);selfdist,isleg=rm.distances(starts,dirs.reshape(1,-1,3))
 blocked=np.isfinite(selfdist)&(selfdist<actual.reshape(1,-1))
 check('real_CPU_moving_body_occlusion',(blocked&isleg).any(),{'leg_occluded_beams':int((blocked&isleg).sum()),'body_occluded_beams':int(blocked.sum()),'collision_primitives':len(rm.items)})
 # Same raw range input reaches the untouched h0 update; verify exact equality.
 packet=dict(origins=origin,dirs=dirs,ranges=actual,valid=valid,body_pos=pos,body_quat=quat)
 a=Maps(1,'L0');reference=H0(1,'h0');a.integrate([0],packet,.1,.2)
 ref=reference.maps[0];ref.precomputed_rays=dirs[0];ref.integrate(actual[0],valid[0],np.zeros((2,120,90)),origin[0],None,None,pos[0],quat[0],.1)
 pose=np.array([[0.,0.,0.]]);scanner=np.array([.56])
 check('h0_mapper_exact_parity',np.array_equal(a.sample(pose,scanner,.2),reference.sample(pose,scanner,.2)))
 check('h0_ttl',not a.sample(pose,scanner,2.2)[:,187:].any())
 a.reset([0]);check('reset_unknown',not a.sample(pose,scanner,.2).any())
 # Flat must never carve; a geometric gap must generate direct or carved observations.
 b=Maps(1,'L1');b.integrate([0],packet,.1,.2)
 check('flat_no_carving',b.counts['carved_updates']==0,b.counts.copy())
 gaprange=cast(floor_mesh(True),starts,dirs.reshape(1,-1,3),'cpu').reshape(1,2,120,90)
 packet['ranges']=gaprange;packet['valid']=np.isfinite(gaprange)&(gaprange<=10)
 gapmap=Maps(1,'L1');gapmap.integrate([0],packet,.1,.2)
 finitecarve=np.isfinite(gapmap.carves[0].z)
 check('gap_carving_positive',finitecarve.any(),gapmap.report())
 # A subsequent direct floor hit supersedes free-space memory (including in reducer).
 packet['ranges']=actual;packet['valid']=valid;gapmap.integrate([0],packet,.2,.3)
 m=gapmap.maps[0];c=gapmap.carves[0];points=origin[0,:,None,None,:]+dirs[0]*actual[0,...,None];ij,inside=m.indices(points[...,:2]);hit=inside&valid[0]
 check('later_direct_hit_priority',not np.isfinite(c.z[ij[...,1][hit],ij[...,0][hit]]).any())
 # Uncertain support: a single direct point cannot certify carving.
 unknown=Maps(1,'L1');bad=packet.copy();bad['valid']=np.zeros_like(valid);bad['valid'][0,0,60,45]=True
 unknown.integrate([0],bad,.1,.2);check('uncertain_plane_no_carving',unknown.counts['carved_updates']==0)
 t=CameraTransport(CameraTransportConfig((100,100),(0,150),.05),.01,42);step=0;cap=[]
 for _ in range(20):step=t.next_capture(step);cap.append(step);t.capture(step,None);t.receive(step)
 check('transport_100ms',np.all(np.diff(cap)==10),{'capture_steps':cap,'config':t.config.manifest(.01)})
 mask=delivery_mask(10,torch.tensor([1,1,1]),torch.tensor([1,2,1]),torch.tensor([9,9,11]));check('transport_reset_reorder_gate',mask.tolist()==[True,False,False])
 check('finite_outputs',all(np.isfinite(x.sample(pose,scanner,.3)).all() for x in [a,b,gapmap,unknown]))
 # Execute the exact rollout delivery-loop AST with one synthetic packet and real mapper.
 import ast
 tree=ast.parse((RDIR/'tools/rl_replay/e1_lidar126.py').read_text())
 loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For) and ast.unparse(node.iter)=='transport.receive(step)')
 counter_init=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='counters' for t in node.targets))
 ct=CameraTransport(CameraTransportConfig((100,100),(0,0),0),.01,42)
 ep=torch.zeros(1,dtype=torch.long);last_packet=torch.full_like(ep,-1)
 ct.capture(10,(packet,ep.clone()));fake_maps=Maps(1,'L0')
 scope=dict(transport=ct,step=10,episodes=ep,last=last_packet,delivery_mask=delivery_mask,events=[],maps=fake_maps,dt=.01,time=time,timing={'mapping':[]})
 exec(compile(ast.Module(body=[counter_init,loop],type_ignores=[]),'actual_rollout_delivery_loop','exec'),scope)
 check('rollout_fake_packet_delivery_counter',scope['counters']['delivered_env_packets']==1 and scope['counters']['foot_lane_cells']==0 and scope['counters']['foot_lane_valid']==0 and last_packet.tolist()==[10] and len(scope['timing']['mapping'])==1 and ct.delivered==1,dict(counters=scope['counters'],last=last_packet.tolist(),delivered=ct.delivered,source='e1_lidar126.run actual AST delivery loop; real Maps.integrate'))
 check('all_rollout_counter_keys_are_strings',all(not isinstance(node.slice,ast.Tuple) for node in ast.walk(tree) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='counters'))
 # UPDATE128: same physical evidence, only adopted carving output changes.
 packet['ranges']=gaprange;packet['valid']=np.isfinite(gaprange)&(gaprange<=10)
 one=Maps(1,'L1');two=Maps(1,'L2')
 one.integrate([0],packet,.1,.2);two.integrate([0],packet,.1,.2)
 h1=one.sample(pose,scanner,.2);h2=two.sample(pose,scanner,.2);adopt=two.last_sources==2;direct=two.last_sources==1
 check('L2_same_carving_certificate',np.array_equal(one.carves[0].z,two.carves[0].z,equal_nan=True) and one.report()==two.report())
 check('L2_only_carving_value_changes',adopt.any() and np.array_equal(one.last_sources,two.last_sources) and np.array_equal(h1[:,187:],h2[:,187:]) and np.array_equal(h1[:,:187][~adopt],h2[:,:187][~adopt]) and np.all(h2[:,:187][adopt]==5),dict(adopted=int(adopt.sum()),direct=int(direct.sum())))
 # Execute the actual clean_terrain function AST, supplying a saturated deep-gap fixture.
 otree=ast.parse((RDIR/'src/gd_lab/gast/observations.py').read_text())
 clean=next(node for node in otree.body if isinstance(node,ast.FunctionDef) and node.name=='clean_terrain')
 gscope=dict(torch=torch,terrain_family_gate=lambda env,families:torch.zeros(1,dtype=torch.long))
 exec(compile(ast.Module(body=[clean],type_ignores=[]),'actual_clean_terrain','exec'),gscope)
 scan=NS(pos_w=torch.tensor([[0.,0.,.56]]),ray_hits_w=torch.tensor([[[0.,0.,-2.]]]*1).expand(1,187,3))
 class Scene(dict):pass
 scene=Scene(height_scanner=NS(data=scan));scene.env_origins=torch.zeros((1,3))
 truth=gscope['clean_terrain'](NS(scene=scene)).numpy()
 observed=h2[:,:187][adopt];expected=np.full(observed.shape,truth[0,0],dtype=observed.dtype)
 check('L2_GT_gap_encoding_bitwise',truth[0,374]==1 and truth[0,0]==5 and observed.tobytes()==expected.tobytes() and observed.astype(np.float32).tobytes()==np.full(observed.shape,truth[0,0],dtype=np.float32).tobytes(),dict(value=float(truth[0,0]),mapper_dtype=str(observed.dtype),teacher_dtype='float32',float32_hex=np.float32(truth[0,0]).tobytes().hex(),source='actual clean_terrain AST; deep gap z=-2m'))
 packet['ranges']=actual;packet['valid']=valid
 one.integrate([0],packet,.2,.3);two.integrate([0],packet,.2,.3)
 v1=one.sample(pose,scanner,.3);v2=two.sample(pose,scanner,.3)
 check('L2_later_direct_priority',np.array_equal(v1,v2) and not (two.last_sources==2).any())
 check('L2_TTL',not two.sample(pose,scanner,2.3)[:,187:].any())
 two.reset([0]);check('L2_reset_unknown',not two.sample(pose,scanner,.3).any())
 result=dict(status='PASS',checks=checks,wall_seconds=time.perf_counter()-started)
except BaseException as exc:
 result=dict(status='FAIL',checks=checks,error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
finally:
 (O/'cpu128.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
if result['status']!='PASS':sys.exit(1)
