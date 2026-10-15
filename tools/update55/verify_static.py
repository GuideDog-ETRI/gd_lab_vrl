"""Host-only AST/shell/hash audit. Does not execute any launcher."""
from pathlib import Path
import ast,hashlib,json,difflib,subprocess
root=Path(__file__).resolve().parents[2]
baseline=json.loads((root/'tools/update55/baseline_sha256.json').read_text())
changed=[p for p,h in baseline.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
assert changed==['src/gd_lab/gast/belief_aux.py'],changed
s=(root/changed[0]).read_text();prefix=s.split('class LookaheadAux(')[0].rstrip('\n')+'\n'
assert hashlib.sha256(prefix.encode()).hexdigest()==baseline[changed[0]]
paths=[p for folder in ('src','tools','scripts') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.md') and (str(p.relative_to(root)) not in baseline or str(p.relative_to(root)) in changed)]+[root/'run_v3_update55_90.sh']
for p in paths:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
(root/'tools/update55/changed_files.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
pairs=[('tools/rl_replay/v3_runner_update54.py','tools/rl_replay/v3_runner_update55.py'),('scripts/gast/train_student_live_update54.py','scripts/gast/train_student_live_update55.py'),('run_v3_update54_90.sh','run_v3_update55_90.sh'),('tools/update51/export_v3.py','tools/update55/export_v3.py')]
patch=''.join(''.join(difflib.unified_diff((root/a).read_text().splitlines(True),(root/b).read_text().splitlines(True),fromfile=a,tofile=b)) for a,b in pairs)
patch+=''.join(difflib.unified_diff(prefix.splitlines(True),s.splitlines(True),fromfile='UPDATE54/belief_aux.py',tofile='src/gd_lab/gast/belief_aux.py'))
(root/'tools/update55/review_delta.patch').write_text(patch)
report={'status':'PASS','modified_prior_files':changed,'update54_aux_original_prefix_unchanged':True,'python_ast_and_shell_syntax':'PASS','new_code_and_docs_count':len(paths)-1}
(root/'outputs/update55_cpu/static_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
