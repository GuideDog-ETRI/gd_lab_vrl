"""UPDATE54 host-only packet contracts, independent of inference and GT inputs."""
import torch
from gd_lab.students.alignment import validate_teacher_camera_capture, camera_signature


def check_payload(frames, extra, metadata, step, epochs, camera_contract):
    n = len(frames)
    if frames.shape != (n,4,2,45,80) or not torch.isfinite(frames).all():
        raise RuntimeError('packet frames shape/finite contract')
    if metadata['capture_step'] != step or not torch.equal(metadata['episode'],epochs):
        raise RuntimeError('packet capture step/episode contract')
    if epochs.shape != (n,) or epochs.dtype != torch.long or (epochs<0).any():
        raise RuntimeError('packet episode shape/type contract')
    if metadata['camera_signature'] != camera_signature(camera_contract):
        raise RuntimeError('packet camera profile/calibration contract')
    for value,shape in zip(extra[:7],((n,203),(n,748),(n,374),(n,7),(n,),(n,12),(n,32))):
        if value.shape != shape or not torch.isfinite(value).all():
            raise RuntimeError('packet capture target shape/finite contract')
    if not torch.allclose(extra[4],torch.full_like(extra[4],step*.01),atol=1e-6,rtol=0):
        raise RuntimeError('packet target capture timestamp contract')
    if not torch.equal(extra[3],metadata['pose']):raise RuntimeError('packet target pose contract')
    if not torch.allclose(extra[3][:,3:].norm(dim=-1),torch.ones(n,device=frames.device),atol=1e-4,rtol=0):
        raise RuntimeError('packet quaternion contract')
    if metadata['scanner_z'].shape != (n,) or not torch.isfinite(metadata['scanner_z']).all():
        raise RuntimeError('packet height datum contract')


@torch.no_grad()
def capture_metadata(env, obs, extra, epochs, contract):
    # Applies each alignment.py check explicitly to the camera-visible 374
    # capture observation, NOT to the GAST teacher history/clean full grid.
    terrain = validate_teacher_camera_capture(env,obs,contract)
    if not torch.equal(extra[2],terrain):raise RuntimeError('packet terrain target contract')
    robot = env.scene['robot'].data
    w,x,y,z = robot.root_quat_w.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pose = torch.cat((robot.root_pos_w[:,:2],yaw[:,None],robot.root_quat_w),-1)
    if not torch.equal(extra[3],pose):raise RuntimeError('packet capture root pose contract')
    scan = env.scene['height_scanner'].data
    # The training warp requires the declared yaw-aligned grid. Reject a
    # scanner origin/order mismatch instead of silently repairing geometry.
    ys,xs = torch.meshgrid(torch.linspace(-.5,.5,11,device=pose.device),torch.linspace(-.8,.8,17,device=pose.device),indexing='ij')
    xy = torch.stack((xs,ys),-1).reshape(1,187,2)
    c,s = yaw[:,None].cos(),yaw[:,None].sin()
    expected = torch.stack((c*xy[...,0]-s*xy[...,1],s*xy[...,0]+c*xy[...,1]),-1)+pose[:,None,:2]
    hits = scan.ray_hits_w
    visible = terrain[:,187:]> .5
    if (visible & ~torch.isfinite(hits).all(-1)).any():raise RuntimeError('visible nonfinite scan hit')
    if not torch.allclose(hits[visible][:,:2],expected[visible],atol=1e-4,rtol=0):
        raise RuntimeError('capture scanner grid/pose contract')
    height = (scan.pos_w[:,2,None]-hits[...,2]-.5).clamp(-1,1)*5
    if not torch.allclose(terrain[:,:187][visible],height[visible],atol=1e-5,rtol=0):
        raise RuntimeError('capture scanner height datum contract')
    meta = {'capture_step':int(env.common_step_counter),'episode':epochs.detach().clone(),
            'camera_signature':camera_signature(contract),'pose':pose.detach().clone(),
            'scanner_z':scan.pos_w[:,2].detach().clone()}
    check_payload(env._vrl_camera_snapshot[0],extra,meta,meta['capture_step'],epochs,contract)
    return meta
