import torch,json
from pathlib import Path
for p in Path('checkpoints/teachers/gast/rayir_top5_20261016').glob('*/best_top5/*_top1.pt'):
 if p.name not in ('2769_top1.pt','3906_top1.pt'):continue
 c=torch.load(p,map_location='cpu',weights_only=False);e=c['infos']['gd_lab'];o=c.get('optimizer_state_dict',{})
 print(p.name,'iter',c['iter'],'keys',list(c),'extra',list(e));print('PPO',len(o.get('state',{})),[{k:v for k,v in g.items() if k!='params'} for g in o.get('param_groups',[])]);print('CENet',len(e.get('cenet_optimizer_state_dict',{}).get('state',{})));print('metadata', {k:v for k,v in e.items() if k not in ('cenet_optimizer_state_dict','deploy_context')})
