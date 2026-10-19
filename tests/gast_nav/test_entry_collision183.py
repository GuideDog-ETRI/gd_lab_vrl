import torch
from gd_lab.gast_nav.entry_collision183 import overlapping_parts183
from gd_lab.gast_nav.producer168 import SensorPrimitives

def shape(centre,half,kind=0):
 return SensorPrimitives(torch.tensor([[centre]],dtype=torch.float32),torch.eye(3)[None,None],torch.tensor([[half]],dtype=torch.float32),torch.tensor([[kind]]),torch.tensor([[True]]),torch.tensor([True]))

def test_sphere_contact_is_not_volume_overlap_and_penetration_rejected():
 floor=shape([0,0,-.5],[2,2,.5])
 assert not overlapping_parts183(shape([0,0,.02],[.02]*3,2),floor).item()
 assert overlapping_parts183(shape([0,0,.019],[.02]*3,2),floor).item()

def test_box_sat_and_sphere_corner_not_aabb_false_positive():
 box=shape([0,0,0],[1,1,1])
 assert not overlapping_parts183(shape([2,0,0],[1,1,1]),box).item()
 assert overlapping_parts183(shape([1.99,0,0],[1,1,1]),box).item()
 assert not overlapping_parts183(shape([1.8,1.8,0],[1,1,1],2),box).item()
 assert overlapping_parts183(shape([1.7,1.7,0],[1,1,1],2),box).item()
