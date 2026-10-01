import torch
from gd_lab.teachers.bivt.raycast_visibility import ray_hits_box, ray_hits_spheres, raycast_visible_points

def test_box_parallel_inside_and_finite_segment():
    starts=torch.tensor([[0.,0.,-2.],[2.,0.,-2.],[0.,0.,0.],[0.,0.,-2.]])
    directions=torch.tensor([[0.,0.,1.]]).expand(4,-1)
    result=ray_hits_box(starts,directions,torch.tensor([4.,4.,4.,.5]),torch.zeros(4,3),
                        torch.eye(3).expand(4,-1,-1),torch.tensor([.5,.5,.5]))
    assert result.tolist()==[True,False,True,False]

def test_foot_sphere_occlusion():
    starts=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    directions=torch.tensor([[0.,0.,1.]]).expand(2,-1)
    feet=torch.tensor([[[0.,0.,.5]]]).expand(2,-1,-1)
    assert ray_hits_spheres(starts,directions,torch.ones(2),feet,.03).tolist()==[True,False]

def test_near_clip_avoids_camera_mount_self_occlusion():
    points=torch.tensor([[[0.,0.,1.]]])
    k=torch.tensor([[40.,0.,40.],[0.,40.,22.5],[0.,0.,1.]])
    def cast(starts,directions,max_dist): return 1./directions[:,2]
    box=(torch.zeros(1,3),torch.eye(3)[None],torch.tensor([.1,.1,.05]),
         torch.tensor([[[3.,3.,3.]]]),.03)
    assert raycast_visible_points(points,torch.zeros(1,3),torch.tensor([[1.,0.,0.,0.]]),k,
        (45,80),(.15,5.),cast,body_occluders=box,conservative=True).all()

def test_edge_disagreement_is_more_conservative():
    points=torch.tensor([[[0.,0.,1.]]])
    k=torch.tensor([[40.,0.,40.],[0.,40.,22.5],[0.,0.,1.]])
    def cast(starts,directions,max_dist):
        z=torch.where(directions[:,0]>.01,1.015,.985)
        return z/directions[:,2]
    args=(points,torch.zeros(1,3),torch.tensor([[1.,0.,0.,0.]]),k,(45,80),(.15,5.),cast)
    assert raycast_visible_points(*args).all()
    assert not raycast_visible_points(*args,conservative=True).any()
