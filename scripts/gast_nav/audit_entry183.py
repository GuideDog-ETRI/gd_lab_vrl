import torch,json,sys
from pathlib import Path
from gd_lab.gast_nav.body_projection import CollisionFootprint
from gd_lab.core.camera_geometry import camera_rotation_matrix
run=Path(sys.argv[1]);d=torch.load(run/'temporal_probe_199.pt',map_location='cpu',weights_only=False)
p=d['body_pos_w'];q=d['body_quat_w'];names=d['body_names'];foot=CollisionFootprint(names,'cpu');rot=camera_rotation_matrix(q)
pts=[]
for v in d['contact_points'].values():
 valid=torch.isfinite(v[:,0]).all(-1);pts.append(torch.where(valid[...,None],v[:,0],torch.nan)[...,2].nanmedian(-1).values)
z=torch.stack(pts,1).nanmedian(1).values
items=[]
for i,kind,off,local,size in foot.parts:
 c=p[:,i]+torch.einsum('nij,j->ni',rot[:,i],off)
 low=c[:,2]-(size if kind=='sphere' else torch.einsum('nij,j->ni',(rot[:,i]@local).abs(),size)[:,2])
 diff=low-z
 items.append(dict(link=names[i],kind=kind,below_measured_contact=int((diff<0).sum()),min_offset_m=float(diff.nan_to_num(nan=100).min()),max_penetration_m=float((-diff).nan_to_num().clamp_min(0).max())))
print(json.dumps(items,indent=2));(run/'entry_floor_audit183.json').write_text(json.dumps(items,indent=2))
