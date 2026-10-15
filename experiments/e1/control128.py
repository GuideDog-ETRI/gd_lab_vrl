from pathlib import Path
import os,sys,subprocess,json,time,hashlib,fcntl,datetime
R=Path(__file__).resolve().parent;H=Path('/home/user/gd_project/notes');O=R/'outputs/update128'
P=H/'Z890_UPDATE128_PROGRESS.md';manifest=json.loads((O/'manifest128.json').read_text())
def update(state,detail):
 P.write_text('# UPDATE128 진행 상황\n\n- 시각: '+datetime.datetime.now().astimezone().isoformat()+'\n- 상태: '+state+'\n- '+detail+'\n- 작업 사본: '+str(R)+'\n- CPU 검사 통과. 코드·모델·프로토콜 SHA: outputs/update128/manifest128.json\n- UPDATE114 PID 접촉 없음. git/원격/kill/자동 재시작 없음. GPU 작업 1개만 순차 실행.\n')
 print(state,detail,flush=True)
def verify():
 for path,digest in manifest['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'동결 파일 변경: '+path
locks=[]
try:
 assert json.loads((O/'cpu128.json').read_text())['status']=='PASS'
 # Open existing shared advisory locks without changing their contents.
 for path in [R.parent/'update110_eval/outputs/update114_gpu.lock',H/'Z890_UPDATE70_FOLLOWUP.lock']:
  f=path.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for arm,seconds,name in [('L2',2,'smoke_L2x2'),('L2',30,'L2x2')]:
  verify();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name,used_gpu_memory','--format=csv,noheader'],text=True)
  (O/(name+'_gpu_before.txt')).write_text(gpu)
  assert all('gnome-remote-desktop' in line for line in gpu.splitlines() if line.strip()),'GPU 사용 중: '+gpu
  out=O/name;out.mkdir();update('RUNNING',f'{name}: seed42, 32 env × {seconds}초, vx 1.0. 이 조건은 한 번만 실행.')
  childenv=dict(os.environ,UPDATE116_GPU_LOCK_HELD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',U127_SENSOR_COUNT='1' if name=='L0_single' else '2')
  begin=time.monotonic()
  with (out/'run.log').open('x') as log:
   child=subprocess.Popen(['bash',str(R/'run_trial128.sh'),arm,'42','32',str(out/'result.json'),str(seconds)],cwd=R,env=childenv,stdout=log,stderr=subprocess.STDOUT)
   (O/'active128.json').write_text(json.dumps(dict(pid=child.pid,run=name,start=time.time())))
   code=child.wait()
  wall=time.monotonic()-begin;(out/'execution128.json').write_text(json.dumps(dict(returncode=code,wall_seconds=wall,simulated_seconds=seconds),indent=2))
  assert code==0,f'{name} 종료 코드 {code}'
  result=json.loads((out/'result.json').read_text())
  assert result['arm']==arm+('' if name=='L0_single' else 'x2') and result['seconds']==seconds and result['num_envs']==32 and result['seed']==42
  assert result['teacher_sha256'].startswith('db500e96') and result['student_sha256'] is None
  assert result['nan_checks_passed']
  if seconds==2:
   c=result['cell_counts'];gate=dict(flat_hit_n=c['flat_sensor_hits'],flat_beam_n=c['flat_sensor_beams'],flat_hit_fraction=c['flat_sensor_hits']/c['flat_sensor_beams'] if c['flat_sensor_beams'] else None,gap_direct=c['gap_direct'],gap_carving=c['gap_carving'],nan_checks_passed=result['nan_checks_passed'])
   gate['passed']=gate['flat_hit_fraction'] is not None and gate['flat_hit_fraction']>.8 and gate['gap_direct']+gate['gap_carving']>0 and gate['nan_checks_passed']
   (O/'semantic_gate128.json').write_text(json.dumps(gate,indent=2));assert gate['passed'],'smoke 의미 게이트 실패: '+str(gate)
  update('RUNNING',f'{name} 완료, wall {wall:.1f}초. '+('의미 게이트 통과.' if seconds==2 else '결과 JSON 검증 통과.'))
 verify()
 reference=json.loads((R.parent/'update127_lidar_x2/outputs/update127b/L1x2/result.json').read_text());result=json.loads((O/'L2x2/result.json').read_text())
 paired={key:reference[key]==result[key] for key in ['teacher_sha256','seed','num_envs','initial_state_sha256','initial_terrain','terrain_grid_sha256','initial_environment','transport_config','packet_schedule_sha256']}
 (O/'paired128.json').write_text(json.dumps(paired,indent=2));assert all(paired.values()),'초기 조건/전송 동등성 불일치'
 update('EVALUATED','smoke 1회, L2x2 본실행 1회 완료. 한국어 결과 집계 중.')
except BaseException as exc:
 (O/'failure128.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)),ensure_ascii=False,indent=2))
 update('STOPPED','실패로 중단: '+str(exc)+'. 지시대로 수정·재시작 없이 보고.')
 raise
finally:
 for f in locks:f.close()
