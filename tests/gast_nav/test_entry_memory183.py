import torch
from gd_lab.gast_nav.entry_memory183 import copy_seeded_rows183
from gd_lab.gast_nav.temporal_map.memory import TemporalMap,load_settings
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176

def test_reset_preload_does_not_replace_other_episode_or_alias_storage():
 settings=load_settings();settings['world_cells']=[3,3]
 cfg=OccConfig176(x_min_m=-.05,x_max_m=.05,y_min_m=-.05,y_max_m=.05)
 old=TemporalMap(2,settings,output_cfg=cfg);new=TemporalMap(2,settings,output_cfg=cfg)
 old.state[0]=2;old.seen[0]=100;old.epoch[:]=4;old.profiles=['a'*64];old.profile[0]=0
 new.state[:]=1;new.seen[:]=50;new.epoch[:]=5;new.profiles=['b'*64];new.profile[:]=0
 copy_seeded_rows183(old,new,torch.tensor([False,True]))
 assert (old.state[0]==2).all() and (old.seen[0]==100).all() and old.epoch.tolist()==[4,5]
 assert (old.state[1]==1).all() and (old.seen[1]==50).all() and (old.profile[1]==1).all()
 new.state.zero_()
 assert (old.state[1]==1).all()
