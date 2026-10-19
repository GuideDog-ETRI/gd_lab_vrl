import torch
from gd_lab.gast_nav.head_support182 import retain_observed_flat182
from gd_lab.gast_nav.schema import SEGMENT_TYPES

def fixture():
 h=torch.full((1,1,7),-.5);observed=torch.tensor([[[True,False,False,True,True,True,True]]]);normal=torch.zeros(1,1,7,3);normal[...,2]=1
 grown=torch.full_like(h,torch.nan);grown[...,0]=-.5
 return h,observed,normal,torch.tensor([-.5]),grown

def test_disconnected_actual_floor_kept_but_blind_gap_not_filled():
 h,o,n,ref,g=fixture();out,mask=retain_observed_flat182(h,o,n,ref,SEGMENT_TYPES.index('PASSAGE'),g)
 assert torch.isnan(out[...,1:3]).all() and (out[...,3:]==-.5).all()
 assert torch.isnan(g[...,3:]).all()

def test_high_object_bad_normal_unseen_no_contact_rejected():
 h,o,n,ref,g=fixture();h[...,3]=-.33;n[...,4,:]=torch.nan;o[...,5]=False
 out,_=retain_observed_flat182(h,o,n,ref,SEGMENT_TYPES.index('FLAT'),g)
 assert torch.isnan(out[...,3:6]).all()
 out,_=retain_observed_flat182(h,o,n,ref*torch.nan,SEGMENT_TYPES.index('FLAT'),g)
 assert torch.isnan(out).all()

def test_stair_connectivity_unchanged_and_flat_threshold_inclusive():
 h,o,n,ref,g=fixture();h[...,3]=-.34
 out,_=retain_observed_flat182(h,o,n,ref,SEGMENT_TYPES.index('FLAT'),g)
 assert torch.isfinite(out[...,3]).all()
 for kind in ('STAIR_UP','STAIR_DOWN','LANDING'):
  out,_=retain_observed_flat182(h,o,n,ref,SEGMENT_TYPES.index(kind),g)
  assert torch.isnan(out[...,1:]).all()
