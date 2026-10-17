import torch,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for p in root.glob('checkpoints/teachers/gast/rayir_top5_20261016/*/best_top5/*_top1.pt'):
 if p.name not in ('2769_top1.pt','3906_top1.pt'):continue
 c=torch.load(p,map_location='cpu',weights_only=False);e=(c.get('infos') or {}).get('gd_lab',{});print(p,hashlib.sha256(p.read_bytes()).hexdigest(),'iteration',c.get('iter'));print('extra',e.keys());print('warm',e.get('gast_warm_start'));print('levels',e.get('terrain_level_means_by_family'))
