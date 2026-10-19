"""Serial local GPU0 diagnostics only; wait for the already-running sanity."""
import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'logs/gast/nav_v1/update181'
while Path('/proc/3836995').exists():time.sleep(5)
sanity=json.loads((root/'logs/gast/nav_v1/update181_off600_v1/result.json').read_text())
if sanity['status']!='completed':raise RuntimeError('sanity incomplete')
for mode in ('unknown','support','occupied','swept','health','all'):
 run=f'update181_{mode}200_v1';env={**os.environ,'GAST_RUN_ID':run}
 print('START',mode,flush=True)
 with (out/f'{mode}.log').open('w') as f:
  p=subprocess.run(['bash','scripts/gast_nav/diagnose_easy_c181.sh','--steps','200','--ablation',mode],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{mode} failed: {p.returncode}')
 print('DONE',mode,flush=True)
 subprocess.run(['python3','scripts/gast_nav/summarize181.py'],cwd=root,stdout=(out/'summary.log').open('w'),check=True)
