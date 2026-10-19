import math
import torch
from gd_lab.gast_nav.body_projection import projected_box_mask, CollisionFootprint
from gd_lab.core.camera_geometry import camera_rotation_matrix


def test_projected_box_uses_shape_and_exact_euclidean_margin():
    xy=torch.tensor([[[.419,0.],[.421,0.],[.41,.11],[.419,.119],[0.,.121]]])
    centre=torch.zeros(1,2);axes=torch.eye(3)[None,:2]
    got=projected_box_mask(xy,centre,axes,torch.tensor([.4,.1,.06]),.02)
    assert got.tolist()==[[[True,False,True,False,False]]]
    q=torch.tensor([[math.cos(math.pi/4),0.,0.,math.sin(math.pi/4)]])
    axes=camera_rotation_matrix(q)[:,:2]
    assert projected_box_mask(torch.tensor([[[0.,.419],[.419,0.]]]),centre,axes,torch.tensor([.4,.1,.06]),.02).tolist()==[[[True,False]]]


def test_urdf_collision_projection_tracks_each_live_leg():
    names=['trunk']+[f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf','foot')]
    model=CollisionFootprint(names,'cpu');assert len(model.parts)==17
    pos=torch.zeros(1,len(names),3);pos[:,1:]=10.
    quat=torch.zeros(1,len(names),4);quat[...,0]=1
    xy=torch.tensor([[[0.,0.],[.5,0.],[1.,0.],[1.051,0.]]]);pose=torch.zeros(1,4)
    assert model.mask(xy,pos,quat,pose,.02).tolist()==[[[True,False,False,False]]]
    pos[:,names.index('FL_foot')]=torch.tensor([1.,0.,0.])
    assert model.mask(xy,pos,quat,pose,.02).tolist()==[[[True,False,True,False]]]


def test_vendor_depth_resolution_preserves_calibration():
    from gd_lab.core.camera_contract import load_camera_contract
    from gd_lab.gast_nav.depth_camera_rays import camera_pixel_rays
    import pytest
    c=load_camera_contract('vendor_new')
    a,la=camera_pixel_rays(c,'cpu',1);b,lb=camera_pixel_rays(c,'cpu',2)
    assert a.shape==(c.width*c.height,3)
    assert b.shape==(4*c.width*c.height,3)
    assert torch.equal(a,b.reshape(c.height*2,c.width*2,3)[::2,::2].reshape(-1,3))
    assert torch.equal(la,lb.reshape(c.height*2,c.width*2)[::2,::2].reshape(-1))
    assert torch.allclose(a.norm(dim=-1),torch.ones_like(la))
    with pytest.raises(ValueError):camera_pixel_rays(c,'cpu',9)


def test_depth_clipping_never_promotes_occlusion_or_missing_return():
    from gd_lab.gast_nav.depth_camera_rays import clip_depth_returns
    hit=torch.tensor([.1,1.,float('inf'),1.])
    own=torch.tensor([float('inf'),float('inf'),float('inf'),.4])
    distance,masked=clip_depth_returns(hit,own,torch.tensor(.15),torch.tensor(5.))
    assert torch.allclose(distance,torch.tensor([0.,.85,4.85,.25]))
    assert masked.tolist()==[True,False,True,True]
