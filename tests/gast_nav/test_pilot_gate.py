import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('gate',Path(__file__).resolve().parents[2]/'scripts/gast_nav/check_pilot_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def test_full_pilot_gate_requires_all_ranks_and_integrity(tmp_path):
    assert gate.assess(tmp_path,expected=2)['status']=='incomplete'
    for r in range(3):
        rows=[dict(iteration=i+1,iteration_s=35.,nan=False,frozen_checksums_ok=True,ddp_fingerprint=str(i),
            peak_reserved_bytes=41*2**30,num_envs=1366 if r==0 else 1365,horizon=100,
            controller_fractions={'1':.5},rewards={'progress':0.},
            person_contact={'person_contact':0,'invalid':0,'episodes_started':100}) for i in range(2)]
        (tmp_path/f'rank{r}_metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
        (tmp_path/f'rank{r}_result.json').write_text(json.dumps(dict(status='completed',iteration=2,optimizer_steps=40)))
    assert gate.assess(tmp_path,expected=2)['status']=='pass'
    p=tmp_path/'rank1_metrics.jsonl';base=p.read_text()
    for key,value in [('ddp_fingerprint','different'),('iteration_s',45.),('nan',True),('peak_reserved_bytes',61*2**30)]:
        rows=[json.loads(s) for s in base.splitlines()]
        for v in rows:v[key]=value
        p.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        assert gate.assess(tmp_path,expected=2)['status']=='fail'
    p.write_text(base)
    (tmp_path/'rank2_result.json').unlink()
    assert gate.assess(tmp_path,expected=2)['status']=='incomplete'
