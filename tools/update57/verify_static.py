"""No git metadata: preserve old source hashes, list new files, AST and diff."""
import ast,difflib,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((root/'tools/update57/baseline_sha256.json').read_text())
assert all(sha(root/p)==h for p,h in baseline.items())
files=list((root/'tools/update57').glob('*.py'))+[root/p for p in ('src/gd_lab/gast/student_v3_update57.py','src/gd_lab/gast/v3_realtime.py','tools/rl_replay/v3_runner_update57.py','scripts/gast/train_student_live_update57.py')]
for p in files:ast.parse(p.read_text())
new={str(p.relative_to(root)):sha(p) for p in files}
(root/'tools/update57/changed_files.json').write_text(json.dumps(new,indent=2)+'\n')
pairs=[('tools/rl_replay/v3_runner_update55.py','tools/rl_replay/v3_runner_update57.py'),('scripts/gast/train_student_live_update55.py','scripts/gast/train_student_live_update57.py'),('tools/update55/export_v3.py','tools/update57/export_v3.py')]
diff=''.join(''.join(difflib.unified_diff((root/a).read_text().splitlines(True),(root/b).read_text().splitlines(True),fromfile=a,tofile=b)) for a,b in pairs)
(root/'tools/update57/review_delta.patch').write_text(diff)
result={'status':'PASS','baseline_files_unchanged':len(baseline),'python_ast':len(files),'new_files_sha256':new,'pilot_files_changed':0}
(root/'outputs/update57_cpu/static_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
