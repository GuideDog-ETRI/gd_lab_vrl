from types import SimpleNamespace
import pytest
from gd_lab.gast_nav.rollout_growth167 import RolloutGrowthProbe

def test_probe_preserves_results_errors_and_bounded_rows():
    def f(x):
        if x<0:raise ValueError('negative')
        return x+1
    obj=SimpleNamespace(f=f);p=RolloutGrowthProbe();p.wrap(obj,'f','f')
    for _ in range(100):
        assert obj.f(2)==3
        with pytest.raises(ValueError,match='negative'):obj.f(-1)
        rows=p.flush();assert rows['f']['calls']==2
        assert rows['f']['wall_s']>=0
        assert not p.rows
    p.close();assert obj.f is f;assert not p.originals

def test_retained_counts_are_cpu_cardinalities():
    from gd_lab.gast_nav.rollout_growth167 import retained_counts
    ns=SimpleNamespace
    env=ns(nav=ns(control=ns(fsm=[ns(consumed={1,2},queue=[])],supervisors=[ns(turn_states={1:None})]),pipeline=ns(mapper=ns(history=[None]*3))))
    alg=ns(capture_pool=ns(records=[None]),nav_normalization_obs=[None]*100)
    assert retained_counts(env,alg)==dict(map_frames=3,landing_consumed=2,landing_queue=0,turn_states=1,packed_records=1,normalization_snapshots=100)
