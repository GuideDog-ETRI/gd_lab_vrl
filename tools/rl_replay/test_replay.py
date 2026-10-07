"""CPU-only regressions: python -B -m unittest discover -s tools/rl_replay -p 'test_*.py'."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import importlib.util
import sys
import subprocess
from unittest.mock import patch
import torch
from rollout_log import RolloutLog, _round, json_text, policy_terrain_input
from runtime import fix_velocity, validate_window, reserve_recording, task_source
import server


class ReplayTests(unittest.TestCase):
    def test_task_registry_cpu(self):
        # GAST tasks.py imports Isaac configuration classes. Execute only its real
        # top-level registration calls, not those classes; BIVT imports normally.
        root=Path(__file__).resolve().parents[2]
        script = """
import sys, ast, importlib.util
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import gd_lab
from gd_lab.core import registry
import gymnasium
task=sys.argv[2]
assert Path(gd_lab.__file__).resolve().is_relative_to(Path(sys.argv[1]))
if task.startswith("Gd-Gast"):
    path=Path(importlib.util.find_spec("gd_lab.gast").origin).parent/"tasks.py"
    tree=ast.parse(path.read_text())
    calls=[n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
           and n.value.func.value.id=="registry" and n.value.func.attr=="register_task"]
    assert calls
    exec(compile(ast.Module(body=calls,type_ignores=[]),str(path),"exec"),{"registry":registry})
else:
    import gd_lab.teachers.bivt
assert task in gymnasium.registry, task
assert "isaaclab.app" not in sys.modules
"""
        for _,task in server.TASKS:
            with self.subTest(task=task):
                result=subprocess.run([sys.executable,"-I","-B","-c",script,str(task_source(root,task)),task],
                                      capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(task_source(d,server.TASKS[0][1]),Path(d)/"src")  # merged layout

    def test_gpu_compute_policy(self):
        # A desktop compositor's allocated memory alone must not block recording.
        for stdout,code,busy in (("",0,False),("123\n",0,True),("N/A",0,True),("",1,True)):
            with patch.object(server.subprocess,"run",return_value=NS(stdout=stdout,returncode=code)) as call:
                self.assertEqual(bool(server.gpu_busy()),busy)
                self.assertIn("--query-compute-apps=pid",call.call_args.args[0])

    def test_finite_json(self):
        self.assertEqual(json.loads(json_text(_round(torch.tensor([float("inf"), float("nan"), 2])))), [None, None, 2])

    def test_terrain_explicit(self):
        obs = {"terrain": torch.zeros(1,374), "gast_history": torch.ones(1,3000)}
        self.assertEqual(policy_terrain_input(obs,1,"cpu","terrain").sum(), 0)
        self.assertEqual(policy_terrain_input(obs,1,"cpu","gast_history").sum(), 374)
        with self.assertRaises(ValueError):
            policy_terrain_input(obs,1,"cpu","automatic")

    def test_command_phase(self):
        class Term:
            vel_command_b = torch.zeros(2,3)
            is_standing_env = torch.ones(2,dtype=torch.bool)
            is_heading_env = torch.ones(2,dtype=torch.bool)
            def _resample_command(self, ids): self.vel_command_b[:] = -1
            def _update_command(self): self.vel_command_b[:] = 5
        term=Term()
        fix_velocity(term,.8)
        for method, args in ((term._resample_command, ([0],)), (term._update_command, ())):
            method(*args)
            torch.testing.assert_close(term.vel_command_b, torch.tensor([[.8,0,0],[.8,0,0]]))
            self.assertFalse(term.is_standing_env.any())

    def test_invalid_windows(self):
        for args in ((0,4,1,None),(True,4,1,None),(8,float("inf"),1,None),(8,4,-1,None),(8,4,1,float("nan"))):
            with self.assertRaises(ValueError): validate_window(*args)

    def test_request_schema(self):
        with tempfile.TemporaryDirectory() as d, patch.object(server,"REPO",Path(d)):
            Path(d,"x.pt").touch()
            valid={"task":server.TASKS[0][1],"checkpoint":"x.pt"}
            self.assertEqual(server.validate_request({**valid,"vx":".8"})["vx"],.8)
            for req in ([],{**valid,"force":"false"},{**valid,"num_envs":4096},
                        {**valid,"name":"../x"},{**valid,"seconds":float("nan")}):
                with self.assertRaises((ValueError,TypeError)): server.validate_request(req)

    def test_live_offsets(self):
        with tempfile.TemporaryDirectory() as d, patch.object(server,"REPO",Path(d)), patch.object(server,"RECORDINGS",Path(d)):
            p=Path(d,"a.live.ndjson")
            p.write_bytes(b'{"meta":{}}\n{"t":0}\n{"t":')
            code,row=server.live_lines("path=a.live.ndjson")
            self.assertEqual(code,200);self.assertEqual(len(row["lines"]),2)
            self.assertEqual(server.live_lines(f"path=a.live.ndjson&offset={row['next']}")[1]["lines"],[])
            self.assertEqual(server.live_lines("path=a.live.ndjson&offset=-1")[0],400)
            self.assertEqual(server.live_lines("path=../a.live.ndjson")[0],404)

    def test_lock_and_collision(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,"out.json");fd=reserve_recording(p)
            try:
                with self.assertRaises(OSError): reserve_recording(Path(d,"other.json"))
            finally: os.close(fd)
            p.touch()
            with self.assertRaises(ValueError): reserve_recording(p)

    def test_finish_math_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            log=RolloutLog.__new__(RolloutLog)
            log.meta={"gamma":.9,"lam":.95,"term_names":["a","b"]}
            log.n=1;log.clouds=[];log.live=None;log._original_reset=None
            log.rec={k:[torch.tensor([x]) for x in values] for k,values in {
                "reward":[1.,2.],"value":[3.,4.],"done":[False,False],
                "terms":[[.4,.6],[.8,1.2]],"foot_contact":[[True],[False]]}.items()}
            out=Path(d,"out.json");log.finish(torch.tensor([5.]),out)
            e=json.loads(out.read_text())["envs"][0]
            self.assertEqual(e["return"],[6.85,6.5])
            self.assertEqual(e["advantage"],[3.7375,2.5])
            self.assertAlmostEqual(sum(e["term_returns"][0]),2.8)
            with self.assertRaises(FileExistsError):log.finish(torch.tensor([5.]),out)
            log.rec["done"][0]=torch.tensor([True])
            log.finish(torch.tensor([5.]),Path(d,"terminal.json"))
            self.assertEqual(json.loads(Path(d,"terminal.json").read_text())["envs"][0]["return"][0],1)

    def test_student_inference_no_training(self):
        import ast
        tree=ast.parse(Path(__file__).with_name("student_replay.py").read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ("backward","zero_grad","save","load_state_dict","add_scalar"):
            self.assertNotIn(name,calls)
        self.assertNotIn("attention.update",Path(__file__).with_name("student_replay.py").read_text())

    def test_student_drives_frozen_cpu(self):
        import student_replay
        transport_path=Path(__file__).resolve().parents[2]/"src/gd_lab/core/camera_transport.py"
        spec=importlib.util.spec_from_file_location("test_transport",transport_path)
        transport_module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"test_transport":transport_module}):
            spec.loader.exec_module(transport_module)
        class Term:
            vel_command_b=torch.zeros(1,3)
            def _resample_command(self, ids):pass
            def _update_command(self):pass
        term=Term()
        class Env:
            num_envs=1;step_dt=.01;device="cpu";common_step_counter=0
            cfg=NS(sim=NS(dt=.005),decimation=2)
            scene={"robot":NS(actuators={})}
            command_manager=NS(get_term=lambda key:term,get_command=lambda key:term.vel_command_b)
            @property
            def unwrapped(self):return self
            def reset(self):
                self.common_step_counter=0
                return self.get_observations(),{}
            def get_observations(self):return {"terrain":torch.zeros(1,374)}
            def step(self,action):
                self.common_step_counter+=1
                self._vrl_camera_snapshot=(torch.zeros(1,4,2,45,80),)
                return self.get_observations(),torch.zeros(1),torch.zeros(1,dtype=torch.bool),{}
        base=Env()
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0
            def init_hidden(self,n,device):return torch.zeros(n,4)
            def encode(self,frames,hidden,pose,age,available):
                assert not torch.is_grad_enabled()
                self.calls+=1
                return self.weight.expand(len(frames),32),hidden+1,None,None,None
        student=Student()
        teacher=NS(act_with_terrain_latent=lambda obs,latent:latent[:,:12],
                   act_inference=lambda obs:torch.full((1,12),99.),
                   terrain_latent=lambda obs:torch.zeros(1,32),evaluate=lambda obs:torch.zeros(1,1))
        attention=NS(latent=torch.zeros(1,32),stamp=torch.full((1,),-100.),ready=torch.zeros(1,dtype=torch.bool))
        attention.capture=lambda obs,action:(None,None,None,torch.zeros(1,7),torch.tensor([base.common_step_counter*.01]),None,None)
        attention.reset=lambda done:None
        actions=[]
        class Log:
            def __init__(self,*args,**kwargs):pass
            def before_step(self,obs,**kwargs):actions.append(kwargs["action"].clone())
            def after_step(self,*args):pass
            def full(self):return len(actions)>=4
            def finish(self,*args):pass
        args=NS(replay_vx=.8,replay_warmup_seconds=0,task="test",student_resume="unused",replay_seconds=.04,
            replay_live=None,replay_cloud_envs=0,camera_interval_ms=[10,10],camera_delay_ms=[0,0],camera_drop_prob=0,replay_out="unused")
        transport=transport_module.CameraTransport(transport_module.CameraTransportConfig((10,10),(0,0),0),.01,42)
        with tempfile.TemporaryDirectory() as d:
            teacher_path=Path(d,"teacher");teacher_path.touch()
            with patch.object(student_replay,"RolloutLog",Log),patch.dict(sys.modules,{"gd_lab.core.camera_transport":transport_module}):
                student_replay.run(base,teacher,student,attention,transport,args,
                    NS(algorithm=NS(gamma=.99,lam=.95)),NS(camera_profile="vendor_new"),teacher_path,0)
        self.assertGreater(student.calls,0)
        self.assertTrue(all((a!=99).all() for a in actions))
        self.assertTrue((actions[0]==0).all())
        self.assertTrue((actions[-1]==1).all())
        self.assertEqual(student.weight.item(),1)
        self.assertIsNone(student.weight.grad)

    def test_capture_coordinates_and_terminal_contacts(self):
        class RaycastVisibleTerrainDropout:
            last_step=torch.tensor([0])
        ray=RaycastVisibleTerrainDropout()
        data=NS(root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[1.,0,0,0]]),
            root_lin_vel_b=torch.zeros(1,3),body_pos_w=torch.zeros(1,1,3),
            joint_pos=torch.zeros(1,1),joint_vel=torch.zeros(1,1))
        robot=NS(body_names=["LF_foot"],joint_names=["joint"],data=data)
        contact=NS(body_names=["LF_foot"],data=NS(net_forces_w=torch.zeros(1,1,3)))
        scan=NS(data=NS(ray_hits_w=torch.zeros(1,187,3),pos_w=torch.tensor([[0.,0,1.]])))
        class Scene(dict):pass
        scene=Scene(robot=robot);scene.sensors={"height_scanner":scan,"contact_forces":contact}
        base=NS(scene=scene,reward_manager=NS(active_terms=["r"],_step_reward=torch.ones(1,1)),
            step_dt=.01,num_envs=1,common_step_counter=0,
            observation_manager=NS(_group_obs_term_cfgs={"terrain":[NS(func=ray)]}))
        base._reset_idx=lambda ids:contact.data.net_forces_w.zero_()
        obs={"terrain":torch.zeros(1,374)}
        args=dict(mean=torch.zeros(1,1),std=torch.zeros(1,1),action=torch.zeros(1,1),
                  value=torch.zeros(1),latent=torch.zeros(1,32),command=torch.zeros(1,3))
        with tempfile.TemporaryDirectory() as d:
            ckpt=Path(d,"model");ckpt.touch()
            log=RolloutLog(base,task="test",checkpoint=ckpt,gamma=.99,lam=.95,steps=2)
            log.before_step(obs,**args)
            contact.data.net_forces_w[:]=2
            base._reset_idx(torch.tensor([0]))
            log.after_step(torch.ones(1),torch.ones(1,dtype=torch.bool))
            self.assertFalse(log.rec["foot_contact"][0].any())
            self.assertEqual(log.rec["foot_contact_after"][0][0,0],1)
            base.common_step_counter=1;scan.data.pos_w[:,2]=2
            log.before_step(obs,**args)
            self.assertEqual(log.rec["terrain_scan_z"][1].item(),1)
            self.assertEqual(log.rec["terrain_capture_step"][1].item(),0)
            obs["gast_history"]=torch.ones(1,3000)
            for enabled in (False,True):
                history=RolloutLog(base,task="test",checkpoint=ckpt,gamma=.99,lam=.95,steps=1,
                                  terrain_source="gast_history",terrain_history=enabled)
                history.before_step(obs,**args)
                self.assertEqual("terrain_history" in history.rec,enabled)


if __name__=="__main__": unittest.main()
