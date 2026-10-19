import copy
from dataclasses import replace
import unittest
from cpu_fixture179 import runtime
from gd_lab.gast_nav.commands import *
from gd_lab.gast_nav.landing_fsm import *
from gd_lab.gast_nav.supervisor import *

def command(**kwargs):
    cmd=NavCommand(1,1000.,'NAV',True,.6,-.5,Conditions(0.,.1,.6),7,1,1,
                   RouteHint(4,'LANDING',.2,1.,'STAIR_DOWN'),8)
    return replace(cmd,**kwargs)

def seen(**kwargs):
    return LandingObservation(**dict(dict(hint_matches=True,landing_detected=True,brake_point=True,entered=True,
         four_contacts=True,speed_mps=0.,yaw_rate_radps=0.,slipping=False,descent_visible=True),**kwargs))

class CommandTests(unittest.TestCase):
    def setUp(self): self.cfg=runtime()['command']; self.cfg['omega_max_verified']=.4
    def test_wire_roundtrip_crc_and_nonfinite(self):
        c=command(); self.assertEqual(decode(encode(c),cfg=self.cfg,now_ms=1000),c)
        with self.assertRaises(ValueError): decode(encode(c)[:-1],cfg=self.cfg,now_ms=1000)
        with self.assertRaises(ValueError): encode(replace(c,v_cap=float('nan')))
        a=NavAck(1,1000,'NAV','ACTIVE',True,'APPROACH',0.,'',c.c,c.c,7,1)
        self.assertEqual(decode(encode(a),cfg=self.cfg,now_ms=1000),a)
    def test_handshake_clamp_and_stop_priority(self):
        a=CommandAdapter(self.cfg,initial_yaw=0.)
        c=command(requested_mode='TURN',turn_rate=-.9)
        self.assertTrue(a.accept(c,now_ms=1000,stationary=True))
        self.assertFalse(a.tick(now_ms=1000,stable=False)['effective_go'])
        self.assertFalse(a.tick(now_ms=1000,stable=True)['effective_go'])
        self.assertEqual(a.handshake,'STABLE')
        self.assertEqual(a.tick(now_ms=1000,stable=True)['velocity'],(0.,0.,-.4))
        self.assertFalse(a.accept(c,now_ms=1000,stationary=True))
        self.assertEqual(a.active_mode,'TURN')
        a.accept(replace(c,seq=2,requested_go=False),now_ms=1000,stationary=True)
        self.assertEqual(a.tick(now_ms=1000,stable=True)['velocity'],(0.,0.,0.))
    def test_invalid_margin_and_stale_do_not_default(self):
        a=CommandAdapter(self.cfg,initial_yaw=.7)
        self.assertFalse(a.accept(command(c=Conditions(0,.05,.6)),now_ms=1000,stationary=False))
        self.assertIn('MARGIN',a.tick(now_ms=1000,stable=True)['hold_reason'])
        self.assertTrue(a.accept(command(),now_ms=1000,stationary=True))
        self.assertFalse(a.tick(now_ms=2000,stable=True)['effective_go'])
        self.assertEqual(a.c.margin,.1)
    def test_feature_width_and_prebrake(self):
        a=CommandAdapter(self.cfg,initial_yaw=0); a.accept(command(),now_ms=1000,stationary=True)
        a.tick(now_ms=1000,stable=True)
        out=a.tick(now_ms=1000,stable=True,pre_brake_v=.1)
        self.assertEqual(out['velocity'][0],.1)
        out['v_cap']=.6
        self.assertEqual(len(policy_features(out,a.c,'PRE_BRAKE',0)),16)

class LandingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=runtime()['landing']
        self.f=LandingFSM(self.cfg)
    def step(self,dt=.1,o=None,c=None,hold=False):
        return self.f.tick(now_ms=1000,dt=dt,command=c or command(),observation=o or seen(),hold=hold)
    def test_normal_auto_and_duplicate(self):
        out=self.step(2.5)
        self.assertEqual(out['phase'],'RELEASED'); self.assertTrue(out['acks'][0]['accepted'])
        r=Release(8,out['acks'][0]['seq'],'AUTO',1000)
        out=self.step(c=command(landing_release=r))
        self.assertEqual(out['acks'][0]['reason'],'DUPLICATE')
    def test_pause_slip_and_hold_timers(self):
        self.step(1.)
        self.step(1.,c=command(requested_go=False))
        self.assertEqual((self.f.dwell_timer,self.f.announce_timer),(1.,1.))
        self.step(0.,c=command(requested_go=False),o=seen(slipping=True))
        self.assertEqual((self.f.dwell_timer,self.f.announce_timer),(0.,1.))
        self.step(0.,hold=True)
        self.assertEqual((self.f.dwell_timer,self.f.announce_timer),(0.,0.))
    def test_no_descent_view_and_recovery(self):
        out=self.step(3.,o=seen(descent_visible=False))
        self.assertEqual(out['hold_reason'],'NO_DESCENT_VIEW'); self.assertFalse(out['acks'])
        self.assertEqual(self.step(2.5)['phase'],'RELEASED')
    def test_missing_hint_and_reset(self):
        out=self.step(c=command(route_hint=None)); self.assertFalse(out['effective_go'])
        self.step(1.)
        self.step(0.,c=command(route_epoch=2))
        self.assertEqual(self.f.dwell_timer,0.)
    def test_release_threshold_composes_config_values(self):
        self.f.cfg['announce_s']=1.0
        self.f.cfg['stable_after_announce_s']=.5
        self.assertEqual(self.step(1.)['phase'],'DWELL')
        self.assertEqual(self.step(.5)['phase'],'RELEASED')

class SupervisorTests(unittest.TestCase):
    def setUp(self): self.cfg=runtime()['supervisor']; self.s=Supervisor(self.cfg)
    def test_health_table(self):
        h=HealthInput(0,0,0)
        for x,health,controller in [(h,'OK','NAV_POLICY'),(replace(h,terrain_age_ms=301),'TERRAIN_STALE','HOLD_3906'),
             (replace(h,obstacle_age_ms=301),'TERRAIN_STALE','HOLD_3906'),(replace(h,pose_age_ms=151),'POSE_STALE','HOLD_3906'),
             (replace(h,risk=True),'SUPERVISOR_RISK','HOLD_3906'),(replace(h,nav_invalid=True),'NAV1_INVALID','HOLD_3906'),
             (replace(h,infer_failed=True),'INFER_FAIL','DAMP')]:
            d=self.s.select(x,simulation=False); self.assertEqual((d.health,d.controller),(health,controller))
        self.assertEqual(self.s.select(replace(h,infer_failed=True),simulation=True).fallback_action,'VISION_HOLD')
    def test_turn_unwrap_accumulation_and_mode_roundtrip(self):
        def sample(y,mode='TURN'): return self.s.observe_turn(route_epoch=1,manoeuvre_id=7,active_mode=mode,yaw=y,yaw_rate=0,dt=.01)
        self.assertFalse(sample(3.13)); self.assertFalse(sample(-3.13))
        sample(-3.13,'NAV')
        self.assertFalse(sample(-3.10))
        self.assertTrue(sample(-3.07))
    def test_horizon(self):
        h=predict_horizon(map_age_s=.2,compute_actuation_s=.1,stopping_s=.4,speed_mps=.6,cfg=self.cfg)
        self.assertAlmostEqual(h.future_s,.5); self.assertAlmostEqual(h.person_inflation_m,.56); self.assertFalse(h.fallback)

class BoundaryTests(unittest.TestCase):
    def test_slip_preserves_announcement_and_replayed_request_does_not_block_auto(self):
        cfg=runtime()['landing']
        f=LandingFSM(cfg)
        c=command(landing_release=Release(8,99,'AUTO',1000))
        f.tick(now_ms=1000,dt=1.,command=c,observation=seen())
        f.tick(now_ms=1000,dt=1.,command=c,observation=seen(slipping=True))
        self.assertEqual(f.announce_timer,1.)
        out=f.tick(now_ms=1000,dt=2.5,command=c,observation=seen())
        self.assertEqual(out['phase'],'RELEASED')
    def test_clamp_positive_turn_go0_and_hold(self):
        cfg=runtime()['command'];cfg['omega_max_verified']=.4
        a=CommandAdapter(cfg,initial_yaw=0)
        a.accept(command(requested_mode='TURN',turn_rate=.5),now_ms=1000,stationary=True)
        a.tick(now_ms=1000,stable=True)
        self.assertEqual(a.tick(now_ms=1000,stable=True)['velocity'],(0.,0.,0.))
        self.assertFalse(a.tick(now_ms=1000,stable=True,hold=True)['effective_go'])

class ExtraContractTests(unittest.TestCase):
    def test_release_stale_and_source_disabled(self):
        cfg=runtime()['landing'];f=LandingFSM(cfg)
        c=command(landing_release=Release(8,4,'GDM',1000))
        out=f.tick(now_ms=1000,dt=2.5,command=c,observation=seen())
        self.assertEqual(out['acks'][0]['reason'],'SOURCE_NOT_ENABLED')
        out=f.tick(now_ms=1000,dt=0,command=command(landing_release=Release(8,5,'AUTO',0)),observation=seen())
        self.assertEqual(out['acks'][0]['reason'],'STALE_RELEASE')
    def test_margin_decrease_stationary_accept_and_wire_range(self):
        cfg=runtime()['command'];a=CommandAdapter(cfg,initial_yaw=.1)
        c=command(c=Conditions(0,.05,.6))
        self.assertTrue(a.accept(c,now_ms=1000,stationary=True))
        with self.assertRaises(ValueError): decode(encode(command(v_cap=2.)),cfg=cfg,now_ms=1000)
    def test_rate_trigger_independent_of_accumulated_angle(self):
        s=Supervisor(runtime()['supervisor'])
        for i in range(2):
            self.assertFalse(s.observe_turn(route_epoch=0,manoeuvre_id=1,active_mode='TURN',yaw=0,yaw_rate=.2,dt=.1))
        self.assertTrue(s.observe_turn(route_epoch=0,manoeuvre_id=1,active_mode='TURN',yaw=0,yaw_rate=.2,dt=.1))
    def test_both_announcement_and_dwell_required(self):
        f=LandingFSM(runtime()['landing'])
        f.context=(1,'NAV',8,0);f.phase='DWELL';f.dwell_timer=3.;f.announce_timer=1.
        self.assertEqual(f.tick(now_ms=1000,dt=0,command=command(),observation=seen())['phase'],'DWELL')
