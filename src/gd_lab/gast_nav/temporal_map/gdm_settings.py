"""Read gdm's actual coverage filters without importing/executing ROS launch code."""
import ast,math
from pathlib import Path
import torch,yaml

def literal_parameter(path,key):
    values=[]
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node,ast.Dict):
            for k,v in zip(node.keys,node.values):
                if isinstance(k,ast.Constant) and k.value==key:
                    try:values.append(ast.literal_eval(v))
                    except (ValueError,TypeError):raise ValueError(f'{key}: nonliteral launch parameter')
    if len(values)!=1:raise ValueError(f'{key}: expected one configured value, found {len(values)}')
    return values[0]

class GdmCoverage:
    def __init__(self,settings):
        g=settings['gdm'];self.trav=yaml.safe_load(Path(g['trav_rbq']).read_text())['/trav_map']['ros__parameters']
        self.core=yaml.safe_load(Path(g['trav_core']).read_text())['/trav_map']['ros__parameters']
        self.self_hits=yaml.safe_load(Path(g['sensor_configuration']).read_text()).get('self_hits',{})
        self.mid_rear_deg=float(literal_parameter(g['planning_launch'],'livox_user_sector_deg'))
        self.mid_min_range=float(literal_parameter(g['planning_launch'],'livox_min_range'))
        self.mid_max_range=float(literal_parameter(g['planning_launch'],'livox_max_range'))
        self.mid_vertical=settings['sensor_sim']['livox_vertical_fov_deg']
        self.ouster_min_range=float(literal_parameter(g['planning_launch'],'sensor_blind_radius'))
        self.ouster_exclude=self.trav['input_exclude_azimuth_deg'];self.rings=self.trav['ouster_ring_range']
        if self.ouster_min_range<.4:raise ValueError('Ouster near cut violates requested >=0.4 m convention')
    def keep_directions(self,sensor,directions,*,screen_door=False):
        az=torch.atan2(directions[:,1],directions[:,0])*180/math.pi
        if sensor=='mid360':
            el=torch.rad2deg(torch.atan2(directions[:,2],directions[:,:2].norm(dim=-1)))
            lo,hi=self.mid_vertical
            keep=(az.abs()<=(180-self.mid_rear_deg/2))&(el>=lo)&(el<=hi)
            return keep&(directions[:,0]>0) if screen_door else keep
        if sensor=='ouster':
            cw=torch.remainder(-az,360);start,end=self.ouster_exclude
            if start==end:return torch.ones_like(az,dtype=torch.bool)
            if start>end:raise ValueError('gdm configuration does not support wrapped exclusion sectors')
            return ~((cw>=start)&(cw<=end))
        if sensor=='d455':return torch.ones_like(az,dtype=torch.bool)  # native camera frustum
        raise ValueError('sensor')
