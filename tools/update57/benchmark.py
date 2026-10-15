"""Local CPU reference only. Never connects to a Pilot machine.
Batch1 FP32, ORT CPU EP, intra/inter-op=1, sequential execution, EXTENDED.
Actor is the exact frozen GAST2000 235->512->256->128->12 MLP;
normalizer/CENet/camera/transport/DDS are excluded and reported as such.
"""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def session(path):
    o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
def feeds(s):
    result={}
    for i in s.get_inputs():
        shape=[1 if isinstance(d,str) else d for d in i.shape]
        result[i.name]=np.zeros(shape,np.bool_ if i.type=='tensor(bool)' else np.float32)
    if 'relative_pose' in result:result['relative_pose'][:,3]=1
    for name in ('accepted','camera_fresh','visual_valid'):
        if name in result:result[name][:]=True
    if 'dt' in result:result['dt'][:]=.01
    if 'frames' in result:result['frames'][:]=np.random.default_rng(57).random(result['frames'].shape,dtype=np.float32)
    return result
def measure(fn,count,warmup):
    for _ in range(warmup):fn()
    data=np.empty(count)
    for i in range(count):
        t=time.perf_counter_ns();fn();data[i]=(time.perf_counter_ns()-t)/1e6
    return {**dict(zip(('p50','p95','p99','p99.9'),map(float,np.percentile(data,[50,95,99,99.9])))),
            'max':float(data.max()),'mean':float(data.mean()),'n':count,'over_2ms':int((data>2).sum()),'over_10ms':int((data>10).sum())},data

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=('small','large','all'),default='small');p.add_argument('--samples',type=int,default=10000);p.add_argument('--warmup',type=int,default=200);p.add_argument('--output',default='outputs/update57_cpu/latency.json');a=p.parse_args()
    if a.samples<1000 or a.warmup<1:raise ValueError('at least 1000 samples and warmup required')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    out=ROOT/a.output;out.parent.mkdir(exist_ok=True,parents=True)
    checkpoint=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
    assert sha(checkpoint)=='db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4'
    actor=nn.Sequential(nn.Linear(235,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')})
    path=out.parent/'actor.onnx';torch.onnx.export(actor,(torch.zeros(1,235),),str(path),input_names=['actor_input'],output_names=['action'],opset_version=18,dynamo=False)
    actor_session=session(path);actor_feed=feeds(actor_session)
    report={'hardware':{'lscpu':subprocess.check_output(['lscpu'],text=True),'uname':platform.uname()._asdict(),'affinity':sorted(os.sched_getaffinity(0)),'ort':ort.__version__,'torch':torch.__version__,'board':Path('/sys/class/dmi/id/board_name').read_text().strip() if Path('/sys/class/dmi/id/board_name').exists() else 'unknown','loadavg_start':os.getloadavg()},
            'settings':{'provider':'CPUExecutionProvider','intra_op':1,'inter_op':1,'execution':'sequential','optimization':'EXTENDED','batch':1,'precision':'FP32','warmup_per_stage':a.warmup,'samples_per_stage':a.samples,'units':'ms','clock':'perf_counter_ns','thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')}},
            'scope':'Z890 local CPU reference, untrained student fixture, real GAST2000 actor weights. Python session.run overhead included. No Pilot/real-time/camera/DDS/normalizer/CENet/end-to-end claim; serial combined path, no concurrent encode contention.',
            'actor_checkpoint_sha256':sha(checkpoint),'actor_onnx_sha256':sha(path),'results':{}}
    samples={}
    for size in (('small','large') if a.size=='all' else (a.size,)):
      for cache in (False,True):
        tag=f'{size}_default'+('_cache' if cache else '')
        directory=ROOT/'outputs/update57_cpu/onnx'/tag
        enc=session(directory/'encode.onnx');tick=session(directory/'tick.onnx');ef=feeds(enc);tf=feeds(tick)
        def encode():
            result=enc.run(None,ef);ef['h_visual']=result[0]
            tf['visual_projection' if cache else 'h_visual']=result[2] if cache else result[0]
            return result
        def belief():
            result=tick.run(None,tf);tf['h_belief']=result[1];return result
        def act():return actor_session.run(None,actor_feed)
        def tick_actor():
            raw=belief()[0];actor_feed['actor_input'][:,-32:]=raw*.5
            return act()
        def combined():encode();return tick_actor()
        stages={}
        for name,fn in (('encode',encode),('tick',belief),('actor',act),('tick_actor',tick_actor),('encode_tick_actor',combined)):
            stages[name],samples[tag+'_'+name]=measure(fn,a.samples,a.warmup)
        report['results'][tag]={'graphs':{name:sha(directory/f'{name}.onnx') for name in ('encode','tick')},'stages':stages}
        print(tag,json.dumps(stages),flush=True)
        out.write_text(json.dumps(report,indent=2)+'\n')
    report['hardware']['loadavg_end']=os.getloadavg();report['tail_note']='10000 samples provide about 10 observations above p99.9; measured max is not WCET. Deployment requires target concurrent-load long runs under separate authorization.'
    out.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out.with_suffix('.samples.npz'),**samples)
if __name__=='__main__':main()
