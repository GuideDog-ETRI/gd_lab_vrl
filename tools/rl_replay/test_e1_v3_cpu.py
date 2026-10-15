"""CPU fixture, real frozen teacher actor, training-host/E1 parity. No Isaac imports."""
import json
import sys
import tempfile
from pathlib import Path
import torch
from torch import nn
from e1_v3_adapter import BeliefStudent, BeliefBank, E1V3Driver, load_checkpoint, reference, ROOT
from gd_lab.core.camera_transport import CameraPacket

def main():
    torch.set_num_threads(1)
    torch.manual_seed(62)
    actor = nn.Sequential(nn.Linear(235,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).eval()
    saved_teacher = torch.load(ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt',map_location='cpu',weights_only=False)
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in saved_teacher['model_state_dict'].items() if k.startswith('actor.')}, strict=True)
    with tempfile.TemporaryDirectory(prefix='update62_cpu_') as tmp:
        path=Path(tmp)/'v3_fixture.pt'
        torch.save({'student_arch':'gast_belief_v3','size':'small','teacher_sha256':reference['teacher_sha256'],
                    'manifest':reference,'updates':1,'synthetic_cpu_fixture':True,'model':BeliefStudent().state_dict()},path)
        student,_=load_checkpoint(path,allow_fixture=True)
        train_student,_=load_checkpoint(path,allow_fixture=True)
        try:load_checkpoint(path)
        except ValueError:pass
        else:raise AssertionError('fixture accepted for GPU evaluation')
    if len(sys.argv) > 1:
        student,_=load_checkpoint(sys.argv[1])
        train_student,_=load_checkpoint(sys.argv[1])
    train=BeliefBank(train_student,2,'cpu');e1=E1V3Driver(student,2,'cpu')
    frames=torch.rand(2,4,2,45,80);pose=torch.zeros(2,7);pose[:,3]=1
    epochs=torch.zeros(2,dtype=torch.long)
    # Arrival order: newer capture first, then out-of-order packet; old episode after reset.
    packets={5:[CameraPacket(0,5,(frames,epochs.clone(),pose.clone()))],
             10:[CameraPacket(9,10,(frames,epochs.clone(),pose.clone())),CameraPacket(7,10,(frames,epochs.clone(),pose.clone()))],
             43:[CameraPacket(38,43,(frames,epochs.clone(),pose.clone()))]}
    max_latent=max_action=0.
    with torch.inference_mode():
        for step in range(48):
            for packet in packets.get(step,[]):
                f,e,p=packet.payload
                ids,_,_=train.publish(f,p,packet.capture_step,packet.delivery_step,e,step)
                other,_,_=e1.deliver(packet,step)
                assert torch.equal(ids,other)
            proprio=torch.randn(2,230);base=torch.randn(2,203)
            pose[:,0]=step*.01
            raw=train.tick(proprio,pose,step)
            expected_latent=raw*train.effective(step)
            expected_action=actor(torch.cat((base,expected_latent),-1))
            action,latent,gate=e1.action(actor,base,proprio,pose,step)
            torch.testing.assert_close(latent,expected_latent,atol=0,rtol=0)
            torch.testing.assert_close(action,expected_action,atol=0,rtol=0)
            torch.testing.assert_close(e1.bank.hidden,train.hidden,atol=0,rtol=0)
            torch.testing.assert_close(e1.bank.belief,train.belief,atol=0,rtol=0)
            max_latent=max(max_latent,float((latent-expected_latent).abs().max()))
            max_action=max(max_action,float((action-expected_action).abs().max()))
            if step<5:assert gate.count_nonzero()==0
            if step==38:assert (gate>0).all() # capture 9 + 29 ticks
            if step==39:assert gate.count_nonzero()==0 # exactly 300ms
            done=torch.tensor([step==40,False]);executed=action.clamp(-.2,.2)
            train.previous_action=executed.clone();train.reset(done,step+1)
            e1.after_step(executed,done,step+1)
            if step==43:assert not train.ready[0] and train.ready[1]
        try:e1.deliver(CameraPacket(47,47,(frames[:,:3],train.episode,pose)),47)
        except ValueError:pass
        else:raise AssertionError('partial view set accepted')
    print(json.dumps({'status':'PASS','device':'cpu','ticks':48,'batch':2,'fixture_strict_load':True,
        'parity_checkpoint':sys.argv[1] if len(sys.argv)>1 else 'synthetic_cpu_fixture',
        'actor':'actual frozen 34624/B3 teacher actor','max_latent_error':max_latent,'max_action_error':max_action,
        'checks':['arrival delivery','out of order rejection','reset epoch rejection','300ms stale gate','4-view enforcement','executed clipped previous action','GPU fixture rejection']},indent=2))

if __name__=='__main__':main()
