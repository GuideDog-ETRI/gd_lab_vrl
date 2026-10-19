"""Reuse UPDATE157 exactly; no alternate P0-P3 definition or actor mutation."""
from tools.rl_replay.pose_error157 import PoseError,LEVELS
class PoseStream:
 def __init__(self,level,n,dt,seed,device):
  self.model=PoseError(LEVELS[level],n,dt,seed,device)
 def reset(self,ids):self.model.reset(ids)
 def update(self,gt_pose,speed,yaw_rate):
  noisy=self.model.update(gt_pose[:,[0,1,3]],speed,yaw_rate)
  self.model.records.clear() # original error sampling retained, unbounded report history disabled
  out=gt_pose.clone();out[:,[0,1,3]]=noisy
  return out
