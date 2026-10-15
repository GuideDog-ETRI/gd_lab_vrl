"""Two independent recurrent state streams: PyTorch vs ORT CPU."""
import importlib.util,json,os,shutil
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from gd_lab.gast.student_v3_update55 import make_student,capture_transform

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/update55_cpu/onnx';OUT.mkdir(exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1);torch.manual_seed(55)
spec=importlib.util.spec_from_file_location('export55',ROOT/'tools/update55/export_v3.py');exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
report={}
for asynchronous,predict_correct in ((False,False),(True,False),(True,True)):
  for size in ('small','large'):
    model=make_student(size=size,asynchronous=asynchronous,predict_correct=predict_correct).eval();tag=f'{size}_{"predict_correct" if predict_correct else ("async" if asynchronous else "default")}'
    if asynchronous:
        with torch.no_grad():model.belief.weight_ih[:,-13:].uniform_(-.03,.03)
    checkpoint=OUT/f'{tag}.pt';torch.save({'student_arch':model.architecture,'size':size,'model':model.state_dict(),
        'teacher_sha256':'db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4','manifest':{'cpu_fixture':True}},checkpoint)
    if (OUT/tag).exists():shutil.rmtree(OUT/tag)
    exporter.export(checkpoint,OUT/tag);checkpoint.unlink()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sessions=[ort.InferenceSession(str(OUT/tag/f'{name}.onnx'),sess_options=opts,providers=['CPUExecutionProvider']) for name in ('encode','tick')]
    max_error=0.
    def check(expected,actual):
        global max_error
        for x,y in zip(expected,actual):
            np.testing.assert_allclose(x.numpy(),y,atol=2e-5,rtol=2e-4)
            max_error=max(max_error,float(np.max(np.abs(x.numpy()-y))))
    with torch.no_grad():
      for n in (1,2):
        hv,hb=model.init_hidden(n,'cpu');ov,ob=hv.numpy().copy(),hb.numpy().copy()
        for step in range(80):
            reset=torch.zeros(n,dtype=torch.bool)
            if step in (0,37):reset[0]=True
            accepted=torch.full((n,),step%9==0,dtype=torch.bool)
            pose=torch.zeros(n,7);pose[:,0]=.02*np.sin(step);pose[:,2]=.04*np.cos(step);pose[:,3]=torch.cos(pose[:,2]/2);pose[:,6]=torch.sin(pose[:,2]/2)
            encode=(torch.rand(n,4,2,45,80),pose,torch.full((n,1),.01*(step%16)),accepted,reset,hv)
            expected=model.encode(*encode)
            inputs={k.name:v.numpy() for k,v in zip(sessions[0].get_inputs(),encode)};inputs['h_visual']=ov
            actual=sessions[0].run(None,inputs);check(expected,actual);hv=expected[0];ov=actual[0]
            age=torch.full((n,2),.01*(step%40));valid=age[:,0]<.3
            args=(torch.randn(n,230),torch.randn(n,12),torch.full((n,1),.01),age,valid,torch.randn(n,3)*.01,reset,hv,hb)
            if asynchronous:
                current=pose.clone();current[:,0]+=.1;current[:,2]+=.1;current[:,3]=torch.cos(current[:,2]/2);current[:,6]=torch.sin(current[:,2]/2)
                args+=(accepted,capture_transform(current,torch.full((n,),.5),pose,torch.full((n,),.45)))
            expected=model.tick(*args)
            inputs={k.name:v.numpy() for k,v in zip(sessions[1].get_inputs(),args)};inputs['h_visual']=ov;inputs['h_belief']=ob
            actual=sessions[1].run(None,inputs);check(expected,actual);hb=expected[1];ob=actual[1]
    names=[v.name for v in sessions[1].get_inputs()]
    assert all(not any(x in name for x in ('future','target','teacher','executed_action')) for name in names)
    if asynchronous:assert names[-2:]==['new_packet','capture_to_now'] and names[4]=='visual_valid'
    report[tag]={'max_abs_error':max_error,'ticks':80,'batches':[1,2],'tick_inputs':names}
report.update(status='PASS',device='cpu',provider='CPUExecutionProvider')
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
