"""Process-local diagnostic patches. Never imported by runtime/training."""
from dataclasses import replace
import torch
MODES=('off','unknown','support','occupied','swept','health','all')
def select181(mode,known,diag):
 if mode not in MODES:raise ValueError(mode)
 out=dict(diag)
 out['raw_unknown']=diag['unknown'].clone();out['raw_missing_traversable']=diag['missing_traversable'].clone();out['raw_occupied']=diag['occupied'].clone()
 if mode=='unknown':known=~diag['unknown']
 elif mode=='support':known=~diag['missing_traversable']
 elif mode=='occupied':known=~diag['occupied']
 elif mode in ('off','swept','health'):known=torch.ones_like(known)
 if mode not in ('occupied','all'):out['occupied']=torch.zeros_like(diag['occupied'])
 return known,out

def install181(mode):
 if mode not in MODES:raise ValueError(mode)
 import gd_lab.gast_nav.occgrid_gate176 as gate
 from gd_lab.gast_nav.supervisor import Supervisor
 original=gate.certify_stopping_grid176
 def diagnose(*args,**kwargs):
  known,diag=original(*args,**kwargs)
  return select181(mode,known,diag)
 gate.certify_stopping_grid176=diagnose
 select=Supervisor.select
 def health(self,h,*,simulation):
  if mode not in ('health','all'):
   h=replace(h,terrain_age_ms=0.,obstacle_age_ms=0.,pose_age_ms=0.)
  return select(self,h,simulation=simulation)
 Supervisor.select=health
