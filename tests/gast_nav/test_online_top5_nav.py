import json,os,time
from pathlib import Path
from gd_lab.gast_nav.online_top5_nav import OnlineTop5,rates,validate,DEFAULTS


def row(P=.5,T=.8,C=.02,U=.01,R=.1,L=.9):
 return dict(P=P,T=T,C=C,U=U,R=R,L=L)


def test_rates_from_counters():
 r=rates([5,10,3,4,1,100,.5,5,20,18],terminations=2,timeouts=18)
 assert r==dict(P=.5,T=.75,U=.01,R=.1,L=.9,C=.1)
 assert rates([0]*10,0,0)['P'] is None


def test_gates_spacing_baseline_and_score(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json')
 assert json.loads((tmp_path/'c.json').read_text())==DEFAULTS
 saved=[]
 for i in range(1,401):d=t.observe(i,row(),'A0' if i<=300 else 'A')
 assert 'before top5_start_iteration' not in d['reasons'] or i<400
 d=t.observe(401,row(),'A');assert d['eligible'] and abs(t.baseline-.9)<1e-9
 s=.35*.5+.15*.8+.2*.98+.1*.99+.1*.9+.1*.9;assert abs(d['score']-s)<1e-9
 assert t.consider(d,lambda p:(p.write_text('x'),saved.append(p)))
 d=t.observe(450,row(P=.9),'A');assert 'spacing' in d['reasons']
 for i in range(451,560):d=t.observe(i,row(C=.2),'A')
 assert 'termination rate' in d['reasons']


def test_criteria_edit_online_and_invalid_kept(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json');t.observe(1,row(),'A')
 time.sleep(.01);(tmp_path/'c.json').write_text(json.dumps(dict(DEFAULTS,w_P=.5,window=10)))
 c,msg=t.criteria_file.refresh();assert c['w_P']==.5 and c['window']==10 and 'reloaded' in msg
 time.sleep(.01);(tmp_path/'c.json').write_text('{"w_P": 7}')
 c,msg=t.criteria_file.refresh();assert c['w_P']==.5 and 'invalid' in msg
 for bad in ({'nope':1},{'window':0},{'w_P':-1}):
  try:validate(bad);raise AssertionError(bad)
  except ValueError:pass


def test_state_roundtrip(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json')
 for i in range(1,60):t.observe(i,row(),'A')
 s=t.state();u=OnlineTop5(tmp_path/'best',tmp_path/'c.json');u.load_state(s)
 assert u.baseline==t.baseline and len(u.history)==len(t.history)
