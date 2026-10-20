import json,time
from gd_lab.gast_nav.online_top5_nav import OnlineTop5,rates,validate,DEFAULTS


def counts(P=.5,T=.8,C=.02,U=.01,R=.1,L=.9,n=100.):
 return [P*n,n,T*n,n,U*n,n,R*n,n,n,L*n,C*n,(1-C)*n]


def test_rates_from_counters():
 r=rates([5,10,3,4,1,100,.5,5,20,18,2,18])
 assert r==dict(P=.5,T=.75,U=.01,R=.1,L=.9,C=.1)
 assert rates([0]*12)['P'] is None


def test_window_uses_ratio_of_sums_not_mean_of_ratios(tmp_path):
 # Episode-end waves: most iterations only terminations, one wave of time-outs.
 t=OnlineTop5(tmp_path/'b',tmp_path/'c.json')
 for i in range(1,51):
  c=counts();c[10],c[11]=(5,95) if i%20==0 else (2,0)
  d=t.observe(i,c,'A')
 # mean of per-iteration C would be ~0.9; summed ratio is 98/(98+190)
 assert abs(d['window']['C']-(2*48+5*2)/((2*48+5*2)+95*2))<1e-9


def test_gates_spacing_baseline_and_score(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json')
 assert json.loads((tmp_path/'c.json').read_text())==DEFAULTS
 for i in range(1,401):d=t.observe(i,counts(),'A0' if i<=300 else 'A')
 d=t.observe(401,counts(),'A');assert d['eligible'],d['reasons']
 assert abs(t.baseline-.9)<1e-9 and abs(t.unknown_baseline-.01)<1e-9
 s=.35*.5+.15*.8+.2*.98+.1*.99+.1*.9+.1*.9;assert abs(d['score']-s)<1e-9
 assert t.consider(d,lambda p:p.write_text('x'))
 d=t.observe(450,counts(P=.9),'A');assert 'spacing' in d['reasons']
 for i in range(451,560):d=t.observe(i,counts(C=.2),'A')
 assert 'termination rate' in d['reasons']
 for i in range(560,620):d=t.observe(i,counts(U=.2),'A')
 assert 'unknown entry rate' in d['reasons']


def test_criteria_edit_online_and_invalid_kept(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json');t.observe(1,counts(),'A')
 time.sleep(.01);(tmp_path/'c.json').write_text(json.dumps(dict(DEFAULTS,w_P=.5,window=10)))
 c,msg=t.criteria_file.refresh();assert c['w_P']==.5 and c['window']==10 and 'reloaded' in msg
 time.sleep(.01);(tmp_path/'c.json').write_text('{"w_P": 7}')
 c,msg=t.criteria_file.refresh();assert c['w_P']==.5 and 'invalid' in msg
 for bad in ({'nope':1},{'window':0},{'w_P':-1}):
  try:validate(bad);raise AssertionError(bad)
  except ValueError:pass


def test_state_roundtrip_and_old_rows_dropped(tmp_path):
 t=OnlineTop5(tmp_path/'best',tmp_path/'c.json')
 for i in range(1,80):t.observe(i,counts(),'A')
 s=t.state();s['history'].append({'iteration':80,'P':.5})
 u=OnlineTop5(tmp_path/'best',tmp_path/'c.json');u.load_state(s)
 assert u.baseline==t.baseline and len(u.history)==len(t.history)


def test_old_state_baselines_are_remeasured(tmp_path):
    top5 = OnlineTop5(tmp_path / "best", tmp_path / "criteria.json")
    top5.load_state({"baseline": 0.41, "unknown_baseline": 0.63, "history": [{"iteration": 1, "rates": {}}]})
    assert top5.baseline is None and top5.unknown_baseline is None and not top5.history
    fresh = OnlineTop5(tmp_path / "best2", tmp_path / "criteria2.json")
    fresh.baseline, fresh.unknown_baseline = 0.9, 0.6
    again = OnlineTop5(tmp_path / "best3", tmp_path / "criteria3.json")
    again.load_state(fresh.state())
    assert again.baseline == 0.9 and again.unknown_baseline == 0.6
