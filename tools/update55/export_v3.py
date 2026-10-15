"""Offline CPU two-graph export of a trained v3 checkpoint. No simulator imports."""
import argparse,json,hashlib
from pathlib import Path
import torch,onnx
from gd_lab.gast.student_v3 import EncodeGraph,TickGraph
from gd_lab.gast.student_v3_update55 import make_student,AsyncTickGraph

def export(checkpoint,directory):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['student_arch'] in ('gast_belief_v3','gast_belief_v3_async55','gast_belief_v3_predict_correct55')
    asynchronous=saved['student_arch']!='gast_belief_v3'
    predict_correct=saved['student_arch']=='gast_belief_v3_predict_correct55'
    assert saved['teacher_sha256']=='db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4'
    model=make_student(size=saved['size'],asynchronous=asynchronous,predict_correct=predict_correct);model.load_state_dict(saved['model'],strict=True);model.eval()
    out=Path(directory);out.mkdir(parents=True,exist_ok=False)
    hv,hb=model.init_hidden(1,'cpu');yes=torch.ones(1,dtype=torch.bool);no=~yes;pose=torch.zeros(1,7);pose[:,3]=1
    specs=[('encode',EncodeGraph(model),(torch.zeros(1,4,2,45,80),pose,torch.zeros(1,1),yes,no,hv),['frames','relative_pose','packet_age','accepted','reset','h_visual'],['h_visual_out','raw_capture','spatial_capture','quality']),
           ('tick',TickGraph(model),(torch.zeros(1,230),torch.zeros(1,12),torch.full((1,1),.01),torch.ones(1,2),no,torch.zeros(1,3),no,hv,hb),['proprio','previous_action','dt','camera_age','camera_fresh','relative_motion','reset','h_visual','h_belief'],['latent_raw','h_belief_out'])]
    if asynchronous:
        name,_,args,inputs,outputs=specs[1]
        inputs[4]='visual_valid'
        specs[1]=(name,AsyncTickGraph(model),args+(no,torch.eye(4)[:3].unsqueeze(0)),inputs+['new_packet','capture_to_now'],outputs)
    graphs={}
    for name,graph,args,inputs,outputs in specs:
        path=out/f'{name}.onnx'
        torch.onnx.export(graph,args,str(path),opset_version=18,dynamo=False,input_names=inputs,output_names=outputs,dynamic_axes={k:{0:'batch'} for k in inputs+outputs})
        onnx.checker.check_model(str(path));graphs[name]={'inputs':dict(zip(inputs,[list(v.shape) for v in args])),'outputs':outputs,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={'architecture':model.architecture,'size':model.size,'graphs':graphs,'teacher_sha256':saved['teacher_sha256'],'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'training_manifest':saved['manifest'],'gate':'host quality * ready * (0 <= capture_age < .3); exactly once before actor','requires_runtime_change':True,'trained_checkpoint_parity':'run deployment sequence parity before use; CPU architecture parity is not trained checkpoint approval'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');a=p.parse_args();export(a.checkpoint,a.output)
