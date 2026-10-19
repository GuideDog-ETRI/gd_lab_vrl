import math
import torch
from gd_lab.gast_nav.analytic_sweep177 import stopping_sweep177


def fixture(v=.6,w=0):
 c=torch.tensor([[[.4,.2,-.1],[-.3,-.2,-.4]]]);h=torch.tensor([[[.2,.1,.1],[.05,.05,.05]]])
 return c,h,torch.tensor([[v,0.]]),torch.tensor([float(w)])


def test_translation_distance_delay_tick_and_full_posture():
 c,h,v,w=fixture();s,d=stopping_sweep177(c,h,v,w)
 assert torch.allclose(d['stop_distance'],torch.tensor([.39]))
 assert s.qualified.all() and s.required.all()
 assert torch.allclose((s.centres+s.half)[0,:,0].max(),torch.tensor(1.03))
 # Zero command still contains every current collision part and gait margin.
 s,d=stopping_sweep177(c,h,v*0,w)
 assert ((s.centres-s.half)[0,:,0].min()<=-.39)


def test_turn_sweep_covers_dense_between_sample_rotations():
 c,h,v,w=fixture(0,-.6);s,d=stopping_sweep177(c,h,v,w)
 assert torch.allclose(d['stop_angle'],torch.tensor([-.21]))
 points=[]
 for a in torch.linspace(-.21-math.radians(3),math.radians(3),301):
  r=torch.stack((a.cos(),-a.sin(),a.sin(),a.cos())).reshape(2,2)
  for i in range(2):
   for x in (-1,1):
    for y in (-1,1):points.append(r@(c[0,i,:2]+torch.tensor([x,y])*(h[0,i,:2]+.04)))
 p=torch.stack(points)
 covered=((p[:,None]-s.centres[0,None,:,:2]).abs()<=s.half[0,None,:,:2]+1e-6).all(-1).any(-1)
 assert covered.all()


def test_large_current_posture_not_clipped_to_verified_radius():
 c,h,v,w=fixture(0,-1);c[...,0]+=2
 s,d=stopping_sweep177(c,h,v,w)
 assert d['radius']>2 and s.qualified.all()
