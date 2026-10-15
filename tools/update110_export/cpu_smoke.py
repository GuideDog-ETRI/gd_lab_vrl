"""Export CPU one-update fixtures, then run the UNMODIFIED deploy checker read-only."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2]
checker=Path('/home/user/gd_project/gd_rbq10_deploy_vrl/export/check_gast_bundle.py')
checker_sha=hashlib.sha256(checker.read_bytes()).hexdigest()
summary=[]
for mode in ['memory_head','all_but_quality']:
 for weight in ['equal3','prep4']:
    checkpoint=ROOT/'outputs/update31_cpu'/f'{mode}_{weight}.pt';out=checkpoint.with_suffix('')
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('export_bundle.py')),
      '--checkpoint',str(checkpoint),'--output-dir',str(out),'--expected-student-iteration','34626',
      '--expected-teacher-sha256','db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4'],check=True)
    # Pathlib absolute resource argument resolves into this copy, never deploy resources.
    subprocess.run([sys.executable,'-B',str(checker),str(out),'--expected-iteration','34626'],check=True)
    m=json.loads((out/'manifest.json').read_text())
    summary.append({'mode':mode,'weight':weight,'status':m['status'],'synthetic':m['synthetic_smoke_only_NOT_TRAINED'],
      'student_error':m['student_max_abs_error'],'actor_error':m['actor_max_abs_error'],'bundle':str(out)})
assert checker_sha==hashlib.sha256(checker.read_bytes()).hexdigest()
(ROOT/'outputs/update31_cpu/export_validation.json').write_text(json.dumps({'cases':summary,'checker_sha256':checker_sha,'checker_unchanged':True},indent=2)+'\n')
print('ALL FOUR CPU EXPORTS AND ORIGINAL DEPLOY CHECKER: PASS')
