"""Validation parity against the previous deep-copy predicate, including malformed leaves."""
from dataclasses import asdict,fields,replace
import pytest
from gd_lab.gast_nav.commands import Conditions,RouteHint,Release,NavCommand,_finite,_finite_command_tree


def test_no_copy_finite_tree_matches_previous_predicate():
    cmd=NavCommand(1,100.,'NAV',True,.2,-.1,Conditions(0.,.1,.2),1,1,1,
        RouteHint(0,'FLAT',0.,1.,'LANDING'),0,Release(0,1,'AUTO',99.))
    cases=[cmd]
    for field in fields(cmd):
        for value in (float('nan'),float('inf'),-float('inf'),0,None,False,{'x':[1.,float('nan')]}):
            cases.append(replace(cmd,**{field.name:value}))
    for cls,key in ((cmd.c,'c'),(cmd.route_hint,'route_hint'),(cmd.landing_release,'landing_release')):
        for field in fields(cls):
            for value in (float('nan'),float('inf'),.123,None,{'nested':(1.,float('nan'))}):
                cases.append(replace(cmd,**{key:replace(cls,**{field.name:value})}))
    for case in cases:assert _finite_command_tree(case)==_finite(asdict(case))


def test_reused_route_and_skipped_unused_features_preserve_commands():
    import copy
    from test_env160 import tile,CFG,RUNTIME
    from gd_lab.gast_nav.command_generator import CommandGenerator,route_hint
    tiles=[tile(k) for k in ('nav_corridor','nav_turn_room','nav_stair_landing','legacy')]
    a=CommandGenerator(tiles,CFG,RUNTIME['command'],seed=193)
    a.reset(range(4),now_ms=0,yaws=[0.]*4);b=copy.deepcopy(a)
    for step in range(800):
        for i in range(4):
            xy=(step*.002,0.03)
            kw=dict(now_ms=step*10,dt=.01,xy=xy,stable=step%13<7,hold=step%31<3)
            left=a.tick(i,**kw)
            right=b.tick(i,**kw,route_info=route_hint(tiles[i],xy),return_features=False)
            assert left[:2]==right[:2] and left[3]==right[3]
            assert a.rng[i].getstate()==b.rng[i].getstate()
            assert a.adapters[i].__dict__==b.adapters[i].__dict__
