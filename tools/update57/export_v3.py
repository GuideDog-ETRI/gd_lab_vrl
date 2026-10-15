"""Offline CPU two-graph export of a trained v3 checkpoint. No simulator imports."""
import argparse,json,hashlib
from pathlib import Path
import torch,onnx
from gd_lab.gast.student_v3 import TickGraph
from gd_lab.gast.student_v3_update57 import ProductionEncode, CachedTick, AsyncCachedTick
from gd_lab.gast.student_v3_update55 import make_student,AsyncTickGraph

def export(checkpoint,directory,projection_cache=False):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['student_arch'] in ('gast_belief_v3','gast_belief_v3_async55','gast_belief_v3_predict_correct55')
    asynchronous=saved['student_arch']!='gast_belief_v3'
    predict_correct=saved['student_arch']=='gast_belief_v3_predict_correct55'
    assert saved['teacher_sha256']=='db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4'
    model=make_student(size=saved['size'],asynchronous=asynchronous,predict_correct=predict_correct);model.load_state_dict(saved['model'],strict=True);model.eval()
    out=Path(directory);out.mkdir(parents=True,exist_ok=False)
    hv,hb=model.init_hidden(1,'cpu');yes=torch.ones(1,dtype=torch.bool);no=~yes;pose=torch.zeros(1,7);pose[:,3]=1
    specs=[('encode',ProductionEncode(model,projection_cache),(torch.zeros(1,4,2,45,80),pose,torch.zeros(1,1),yes,no,hv),['frames','relative_pose','packet_age','accepted','reset','h_visual'],['h_visual_out','quality']+(['visual_projection'] if projection_cache else [])),
           ('tick',TickGraph(model),(torch.zeros(1,230),torch.zeros(1,12),torch.full((1,1),.01),torch.ones(1,2),no,torch.zeros(1,3),no,hv,hb),['proprio','previous_action','dt','camera_age','camera_fresh','relative_motion','reset','h_visual','h_belief'],['latent_raw','h_belief_out'])]
    if asynchronous:
        name,_,args,inputs,outputs=specs[1]
        inputs[4]='visual_valid'
        specs[1]=(name,AsyncTickGraph(model),args+(no,torch.eye(4)[:3].unsqueeze(0)),inputs+['new_packet','capture_to_now'],outputs)
    if projection_cache:
        name,_,args,inputs,outputs=specs[1]
        args=list(args);args[7]=model.visual_projection(hv[:,:5984]).detach()
        inputs[7]='visual_projection'
        specs[1]=(name,AsyncCachedTick(model) if asynchronous else CachedTick(model),tuple(args),inputs,outputs)
    graphs={}
    for name,graph,args,inputs,outputs in specs:
        path=out/f'{name}.onnx'
        torch.onnx.export(graph,args,str(path),opset_version=18,dynamo=False,input_names=inputs,output_names=outputs,dynamic_axes={k:{0:'batch'} for k in inputs+outputs})
        onnx.checker.check_model(str(path));audit_graph(path,name,projection_cache);graphs[name]={'inputs':dict(zip(inputs,[list(v.shape) for v in args])),'outputs':outputs,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={'projection_cache':projection_cache,'export_revision':57,'architecture':model.architecture,'size':model.size,'graphs':graphs,'teacher_sha256':saved['teacher_sha256'],'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'training_manifest':saved['manifest'],'gate':'host quality * ready * (0 <= capture_age < .3); exactly once before actor','requires_runtime_change':True,'cache_contract':'encode produces unmasked projection; tick masks AFTER projection; host invalidates ready/cache on reset, epoch or weight change; h_visual writer encode, h_belief writer tick; serialized reset barrier','trained_checkpoint_parity':'run deployment sequence parity before use; CPU architecture parity is not trained checkpoint approval'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
def audit_graph(path,kind,cache=False):
    graph=onnx.load(str(path)).graph
    names=[v.name for v in graph.output]+[v.name for v in graph.initializer]
    names += [n.name for n in graph.node] + [v for n in graph.node for v in (*n.input,*n.output)]
    forbidden=('raw_capture','spatial_capture','capture_head','spatial_head','hazard_head','lookahead','aux','future')
    assert not any(token in name.lower() for name in names for token in forbidden), names
    if kind=='encode':
        assert [o.name for o in graph.output]==['h_visual_out','quality']+(['visual_projection'] if cache else [])
    if kind=='tick' and cache:
        assert not any('visual_projection.0.' in name for name in names)
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--projection-cache',action='store_true');a=p.parse_args();export(a.checkpoint,a.output,a.projection_cache)
