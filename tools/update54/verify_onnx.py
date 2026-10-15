"""Reuse UPDATE51 recursive parity verbatim, redirect outputs, instantiate separate aux.
The original verification program and its artifacts are never modified.
"""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
source=(root/'tools/update51/verify_cpu.py').read_text()
source=source.replace("out=root/'outputs/update51_cpu'", "out=root/'outputs/update54_cpu/onnx'")
source=source.replace('model=BeliefStudent(visual,size)',
    "model=BeliefStudent(visual,size)\n    from gd_lab.gast.belief_aux import BeliefAux\n    aux=BeliefAux(model.belief_dim,.1)\n    assert aux.head is not None")
exec(compile(source,str(root/'tools/update51/verify_cpu.py'),'exec'),{'__file__':str(root/'tools/update51/verify_cpu.py'),'__name__':'__main__'})
results={}
for size in ('small','large'):
    for tag in ('encode','tick'):
        name=f'{size}_{tag}.onnx'
        a=root/'outputs/update51_cpu'/name;b=root/'outputs/update54_cpu/onnx'/name
        results[name]={'sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'byte_identical_to_update51':a.read_bytes()==b.read_bytes()}
        assert results[name]['byte_identical_to_update51'],name
(root/'outputs/update54_cpu/onnx_unchanged.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
