"""CPU-only UPDATE54 regression; no simulation, optimizer step, or GPU."""
import copy, hashlib, json, os, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch import nn
from gd_lab.gast.belief_aux import BeliefAux, aligned_targets, row_losses
from gd_lab.gast.student import GastStudent
from gd_lab.gast.student_v3 import BeliefStudent
from gd_lab.gast.v3_state import BeliefBank
from gd_lab.gast.b3_objective import WindowLoss
from gd_lab.gast.b2_objective import action_rows
from gd_lab.gast.geometry import targets as capture_targets
from gd_lab.gast.v3_packet_contract import capture_metadata, check_payload
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses, camera_visible_points

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/update54_cpu';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(54)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

def pose(n=2):
    p=torch.zeros(n,7);p[:,3]=1;return p

def terrain(n=2):return torch.cat((torch.zeros(n,187),torch.ones(n,187)),-1)

def fixture(n=2):
    p=pose(n);contract=load_camera_contract('vendor_new')
    root=torch.zeros(n,3);root[:,2]=.5
    robot=NS(body_names=['trunk'],data=NS(root_pos_w=root,root_quat_w=p[:,3:].clone(),body_pos_w=root[:,None],body_quat_w=p[:,None,3:].clone()))
    positions,rotations=mounted_camera_world_poses(root,p[:,3:],contract)
    k=torch.tensor([[contract.fx,0,contract.width/2],[0,contract.fy,contract.height/2],[0,0,1.]]).expand(n,4,3,3).clone()
    y,x=torch.meshgrid(torch.linspace(-.5,.5,11),torch.linspace(-.8,.8,17),indexing='ij')
    hits=torch.stack((x,y,torch.zeros_like(x)),-1).flatten(0,1)[None].repeat(n,1,1)
    scan=NS(data=NS(pos_w=root.clone(),ray_hits_w=hits))
    t=terrain(n)
    env=NS(common_step_counter=7,cfg=NS(camera_profile='vendor_new'),scene={'robot':robot,'height_scanner':scan},
        _vrl_camera_snapshot=(torch.rand(n,4,2,45,80),torch.ones(n,4,45,80),positions,rotations,k),
        _vrl_camera_snapshot_steps=torch.full((n,),7),_vrl_teacher_terrain_capture_steps=torch.full((n,),7),
        _vrl_teacher_terrain_snapshot=t.clone(),_vrl_teacher_terrain_contract=contract.manifest())
    extra=(torch.zeros(n,203),torch.cat((t,torch.zeros(n,187),torch.ones(n,187)),-1),t,p,torch.full((n,),.07),torch.zeros(n,12),torch.zeros(n,32))
    epochs=torch.zeros(n,dtype=torch.long)
    return env,{'terrain':t},extra,epochs,contract

class Checks(unittest.TestCase):
    def test_zero_is_exact_output_loss_gradient_rng_both_sizes(self):
        for size in ('small','large'):
            model=BeliefStudent(size=size);other=copy.deepcopy(model)
            rng=torch.get_rng_state().clone();aux=BeliefAux(model.belief_dim,0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(aux.parameters(),[])
            enabled=BeliefAux(model.belief_dim,.1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            n=2;frames=torch.rand(n,4,2,45,80);hv,hb=model.init_hidden(n,'cpu');yes=torch.ones(n,dtype=torch.bool)
            groups=torch.tensor([1,2]);inputs=(torch.randn(n,230),torch.randn(n,12),torch.full((n,1),.01),torch.zeros(n,2),yes,torch.zeros(n,3),~yes)
            losses=[];outputs=[]
            for m in (model,other):
                enc=m.encode(frames,pose(),torch.zeros(n,1),yes,~yes,hv)
                raw,belief=m.tick(*inputs,enc[0],hb)
                w=WindowLoss('prep4');w.action.add(raw.square().mean(-1),groups)
                w.raw.add(enc[1].square().mean(-1),groups);w.spatial(enc[2],fixture()[2][1],terrain())
                w.preserve(raw.square().mean(-1),groups,yes)
                loss,_=w.value(raw.new_zeros(()));outputs.append((*enc,raw,belief));losses.append(loss)
            actual,_=aux.value(losses[1]);self.assertIs(actual,losses[1])
            self.assertTrue(torch.equal(actual,losses[0]))
            for a,b in zip(*outputs):self.assertTrue(torch.equal(a,b))
            losses[0].backward();actual.backward()
            for (name,a),(_,b) in zip(model.named_parameters(),other.named_parameters()):
                self.assertEqual(a.grad is None,b.grad is None,name)
                if a.grad is not None:self.assertTrue(torch.equal(a.grad,b.grad),name)
            self.assertFalse(any('aux' in k for k in model.state_dict()))

    def test_aux_only_gradient_excludes_actor_teacher_quality_and_heads(self):
        for size in ('small','large'):
            model=BeliefStudent(size=size);aux=BeliefAux(model.belief_dim,.1)
            bank=BeliefBank(model,2,'cpu');p=pose();frames=torch.rand(2,4,2,45,80)
            bank.publish(frames,p,0,0,bank.episode.clone(),0)
            raw=bank.tick(torch.randn(2,230),p,0)
            raw.retain_grad();bank.belief.retain_grad()
            actor=nn.Linear(235,12).requires_grad_(False)
            teacher=nn.Linear(2,32).requires_grad_(False)
            teacher_target=teacher(torch.ones(2,2));gate=bank.gate
            action_rows(actor,torch.zeros(2,203),raw,teacher_target,gate)
            gt=terrain().requires_grad_();y,m=aligned_targets(gt,p,torch.zeros(2),p,torch.zeros(2),torch.ones(2,dtype=torch.bool))
            aux.add(bank.belief,y,m,torch.tensor([1,2]));loss,stats=aux.value(raw.new_zeros(()));loss.backward()
            self.assertGreater(float(bank.belief.grad.abs().sum()),0)
            self.assertIsNone(raw.grad);self.assertIsNone(gt.grad)
            for prefix in ('belief','visual_projection','visual.cell_gru'):
                self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for k,v in model.named_parameters() if k.startswith(prefix)))
            for prefix in ('head','capture_head','visual.quality','visual.cnn','visual.geometry','visual.spatial_head'):
                self.assertTrue(all(v.grad is None for k,v in model.named_parameters() if k.startswith(prefix)),prefix)
            self.assertTrue(all(v.grad is None for module in (actor,teacher) for v in module.parameters()))
            self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in aux.parameters()))

    def test_alignment_translation_yaw_vertical_and_clipping(self):
        t=terrain(1);p=pose(1);z=torch.tensor([.5]);ok=torch.tensor([True])
        h=torch.arange(17).float()[None,None,:].expand(1,11,17)*.01
        t[:,:187]=h.flatten(1)*5
        now=p.clone();now[:,0]=.1
        y,m=aligned_targets(t,p,z,now,z+.2,ok)
        torch.testing.assert_close(y.reshape(1,11,17,5)[0,:,:16,0],h[0,:,1:]+.2,atol=1e-6,rtol=0)
        self.assertFalse(m.reshape(1,11,17,5)[0,:,-1].any())
        now=p.clone();now[:,2]=torch.pi
        y,m=aligned_targets(t,p,z,now,z,ok)
        torch.testing.assert_close(y[...,0].reshape(11,17),h[0].flip((0,1)),atol=1e-6,rtol=0)
        self.assertTrue(m[...,0].all())
        t[0,0]=5
        y,m=aligned_targets(t,p,z,p,z,ok);self.assertFalse(m[0,0,0]);self.assertTrue(y[0,0,1])
        now=p.clone();now[:,0]=.05;t[:,187+1]=0
        y,m=aligned_targets(t,p,z,now,z,ok);self.assertFalse(m[0,0,0]);self.assertFalse(m[0,0,2])
        y,m=aligned_targets(t,p,z,p,z,~ok);self.assertFalse(m.any())
        now[:,0]=float('nan');y,m=aligned_targets(t,p,z,now,z,ok);self.assertFalse(m.any());self.assertTrue(torch.isfinite(y).all())

    def test_masks_unknown_nan_edges_support_and_effective_weights(self):
        t=terrain();t[:,187+90]=0;t[:,90]=float('nan')
        t[:,:187].reshape(2,11,17)[:,:,9:]=-.5
        p=pose();y,m=aligned_targets(t,p,torch.zeros(2),p,torch.zeros(2),torch.ones(2,dtype=torch.bool))
        self.assertFalse(m[:,90,0].any());self.assertFalse(m[:,89,2:4].any())
        self.assertTrue(y[:,8,2].all());self.assertFalse(m.reshape(2,11,17,5)[:,:,-1,2:4].any())
        pred=torch.zeros(2,187,5,requires_grad=True);rows,_=row_losses(pred,y,m);rows.sum().backward()
        self.assertEqual(int(pred.grad[~m].count_nonzero()),0);self.assertGreater(int(pred.grad[...,1].count_nonzero()),0)
        a=BeliefAux(8,.2);a.add(torch.randn(2,8),y,m,torch.tensor([1,2]));loss,stats=a.value(torch.zeros(()))
        pre=stats['aux_positive_edge_effective_coefficient_minmax']['prep'];non=stats['aux_positive_edge_effective_coefficient_minmax']['nonprep']
        self.assertAlmostEqual(pre[0]/non[0],4.);self.assertEqual(stats['aux_nominal_prep_positive_edge_multiplier'],20)
        (OUT/'effective_weights.json').write_text(json.dumps(stats,indent=2)+'\n')
        none=BeliefAux(8,.2);none.add(torch.randn(2,8),y,torch.zeros_like(m),torch.tensor([1,2]));loss,_=none.value(torch.tensor(2.));self.assertEqual(float(loss),2.)
        for bad in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):BeliefAux(8,bad)

    def test_inherited_unknown_tokens_and_capture_masks(self):
        model=GastStudent();observed=[]
        hook=model.geometry.register_forward_pre_hook(lambda mod,args:observed.append(args[0].detach().clone()))
        for depth in (0.,1.,float('nan'),float('inf')):
            frames=torch.zeros(1,4,2,45,80);frames[:,:,0]=depth
            with torch.no_grad():model.encode_v3(frames,model.init_hidden(1,'cpu'),pose(1),torch.zeros(1,1))
            geom=observed[-1];self.assertEqual(geom.shape,(1,4,144,8));self.assertEqual(int(geom[...,6].count_nonzero()),0);self.assertTrue((geom[...,7]==1).all())
        hook.remove()
        clean=fixture()[2][1];unknown=terrain();unknown[:,187:]=0
        y,m=capture_targets(clean,unknown);self.assertFalse(m[...,[0,2,3,4,5]].any());self.assertTrue(m[...,1].all())
        # Actual depth visibility helper rejects missing/far returns.
        point=torch.tensor([[[0.,0.,1.]]]);cam=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]])
        k=torch.tensor([[[1.,0.,2.],[0.,1.,2.],[0.,0.,1.]]])
        for value in (0.,5.,float('inf'),float('nan')):
            self.assertFalse(camera_visible_points(point,cam,q,k,torch.full((1,5,5),value)).any())
        self.assertTrue(camera_visible_points(point,cam,q,k,torch.ones(1,5,5)).all())

    def test_capture_contract_negatives(self):
        env,obs,extra,epochs,c=fixture();meta=capture_metadata(env,obs,extra,epochs,c)
        mutations=[lambda e,o,x,ep: e._vrl_camera_snapshot_steps.fill_(6),
            lambda e,o,x,ep:e._vrl_teacher_terrain_capture_steps.fill_(6),
            lambda e,o,x,ep:e._vrl_camera_snapshot[4][:,:,0,0].mul_(.8),
            lambda e,o,x,ep:e._vrl_camera_snapshot[2].add_(.1),
            lambda e,o,x,ep:e._vrl_teacher_terrain_contract.update(profile='bad'),
            lambda e,o,x,ep:x[3][:,0].add_(.1),lambda e,o,x,ep:x[4].fill_(.06),
            lambda e,o,x,ep:e.scene['height_scanner'].data.pos_w[:,2].add_(.1),
            lambda e,o,x,ep:e.scene['height_scanner'].data.ray_hits_w[:,:,0].add_(.01)]
        for mutate in mutations:
            e,o,x,ep,c=fixture();mutate(e,o,x,ep)
            with self.assertRaises(RuntimeError):capture_metadata(e,o,x,ep,c)
        for key,value in (('capture_step',6),('episode',epochs+1),('camera_signature',('bad',)),('pose',pose()+.1)):
            bad=copy.deepcopy(meta);bad[key]=value
            with self.assertRaises(RuntimeError):check_payload(env._vrl_camera_snapshot[0],extra,bad,7,epochs,c)

    def test_delivery_episode_execution_target_separation(self):
        env,obs,extra,epochs,c=fixture();capture_metadata(env,obs,extra,epochs,c)
        model=BeliefStudent();bank=BeliefBank(model,2,'cpu');frames=env._vrl_camera_snapshot[0]
        bank.reset(torch.tensor([True,False]),8)
        ids,_,_=bank.publish(frames,extra[3],7,10,epochs,10);self.assertEqual(ids.tolist(),[1])
        self.assertEqual(bank.publish(frames,extra[3],7,11,epochs,11)[0].numel(),0)
        self.assertEqual(bank.publish(frames,extra[3],12,15,bank.episode,14)[0].numel(),0)
        raw=bank.tick(torch.randn(2,230),pose(),10)
        actor=nn.Linear(235,12).requires_grad_(False);base=torch.ones(2,203);target=torch.ones(2,32);gate=bank.effective(10)
        loss,act=action_rows(actor,base,raw,target,gate)
        expected=actor(torch.cat((base,target*gate),-1))
        self.assertTrue(torch.equal(loss,(act-expected).square().mean(-1)))
        stale=actor(torch.cat((extra[0],extra[6]*gate),-1))
        self.assertFalse(torch.equal(expected,stale))
        self.assertTrue(torch.equal(extra[0],torch.zeros_like(extra[0])))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    baseline=json.loads((ROOT/'tools/update54/baseline_sha256.json').read_text())
    changed=[name for name,sha in baseline.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    report={'status':'PASS' if result.wasSuccessful() and not changed else 'FAIL','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'device':'cpu','optimizer_steps':0,
            'update51_modified_files':changed}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if report['status']!='PASS':raise SystemExit(1)
