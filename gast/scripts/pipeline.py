"""Run authorized teacher -> student stages; stop on any failure, never restart."""
import argparse
import fcntl
import datetime
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Explicitly select full training; default is smoke only')
parser.add_argument('--teacher-envs', type=int, default=256)
parser.add_argument('--student-envs', type=int, default=16)
parser.add_argument('--teacher-iterations', type=int, default=20000)
parser.add_argument('--student-iterations', type=int, default=20000)
args = parser.parse_args()
if not args.train:
    args.teacher_envs, args.student_envs = 32, 4
    args.teacher_iterations, args.student_iterations = 3, 16
(root/'logs').mkdir(exist_ok=True)
lock = (root/'logs/pipeline.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
run = 'gast_arm4_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
directory = root/'logs'/run
directory.mkdir(parents=True)
state = {'run':run, 'settings':vars(args), 'started_at':datetime.datetime.now().isoformat(), 'status':'teacher'}

def save():
    path = root/'logs'/'current_gast_run.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)

def stage(name, command):
    state['status']=name
    state[name+'_command']=command
    save()
    with (directory/(name+'.console.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        state[name+'_pid']=process.pid
        save()
        code=process.wait()
    if code:
        raise RuntimeError(f'{name} exited with code {code}; see {directory}')

try:
    stage('teacher', ['./scripts/run.sh','teacher','--num_envs',str(args.teacher_envs),
        '--max_iterations',str(args.teacher_iterations),'--run_name',run] +
        ([] if args.train else ['agent.num_steps_per_env=16']))
    lines=(directory/'teacher.console.log').read_text().splitlines()
    completed=[json.loads(line.removeprefix('[GAST_COMPLETE] ')) for line in lines if line.startswith('[GAST_COMPLETE] ')]
    if len(completed)!=1 or completed[0]['completed_updates']!=args.teacher_iterations:
        raise RuntimeError('Teacher completion marker missing or wrong count')
    checkpoint=Path(completed[0]['checkpoint'])
    if not checkpoint.is_file(): raise RuntimeError('Final teacher checkpoint missing')
    state['teacher_result']=completed[0]
    stage('student', ['./scripts/run.sh','student','--num_envs',str(args.student_envs),
        '--iterations',str(args.student_iterations),'--teacher_checkpoint',str(checkpoint),
        '--bptt_steps','64' if args.train else '8','--lr','.0003','--save_interval','200',
        '--perception_run_name',run+'_student'] +
        ([] if args.train else ['--student_warmup','0','--student_ramp','1']))
    student=root/'logs/gast/arm4'/(run+'_student')/f'perception_{args.student_iterations}.pt'
    if not student.is_file(): raise RuntimeError('Final student checkpoint missing')
    if not args.train:
        stage('student_resume', ['./scripts/run.sh','student','--num_envs','4','--iterations','24',
            '--teacher_checkpoint',str(checkpoint),'--student_resume',str(student),'--bptt_steps','8',
            '--lr','.0003','--student_warmup','0','--student_ramp','1',
            '--perception_run_name',run+'_resume'])
        student=root/'logs/gast/arm4'/(run+'_resume')/'perception_24.pt'
        if not student.is_file(): raise RuntimeError('Resumed student checkpoint missing')
    state.update(status='complete',student_checkpoint=str(student),completed_at=datetime.datetime.now().isoformat())
    save()
except BaseException as exc:
    state.update(status='failed',error=str(exc))
    save()
    raise
