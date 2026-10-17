import json
from pathlib import Path
from gd_lab.gast.resume149 import validate_resume
root=Path(__file__).resolve().parents[1]
specs=[('2769_from2000','2769','65da772b8b793caf5e210f7db778c17d006999010ce964b2fc2af2deadad20c9','Gd-GastRayIrV21-Rbq10-Dreamwaq-v0'),('3906_gap50_from_dwb38000','3906','f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98','Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0')]
rows=[]
for folder,name,sha,task in specs:
 p=root/f'checkpoints/teachers/gast/rayir_top5_20261016/{folder}/best_top5/{name}_top1.pt'
 q=validate_resume(p,sha,task);q['last_update_index_after_2000']=q['start_iteration']+1999;rows.append(q)
 for badsha,badtask in [('0'*64,task),(sha,'wrong-task')]:
  try: validate_resume(p,badsha,badtask)
  except ValueError: pass
  else: raise AssertionError('failed to reject invalid source')
print(json.dumps(rows,indent=2))
(root/'archive/update149_validation/sources149b.json').write_text(json.dumps(rows,indent=2))
