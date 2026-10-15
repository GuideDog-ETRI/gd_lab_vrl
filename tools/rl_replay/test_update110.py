import unittest
import torch
from e1_core import LatentSwitch
from test_e1 import transport_mod

class PacketTests(unittest.TestCase):
    def test_capture_hold_single_gate_freshness(self):
        s=LatentSwitch(4,'cpu');z=torch.ones(4,32);g=torch.full((4,1),.2)
        args=(z,z*g,g,torch.tensor([True,True,False,True]),torch.tensor([.1,.3,.1,-.1]),z*9,torch.ones(4),10)
        p,e=s.select('p_teacher_packet',*args,packet_teacher=z*3)
        self.assertTrue(torch.equal(p[0],z[0]*.6))
        self.assertTrue(torch.equal(p[1:],torch.zeros(3,32)))
        f,_=s.select('f_teacher_student_gate',*args)
        self.assertTrue(torch.allclose(f[0],z[0]*1.8))
        # A later current teacher must not replace the held capture payload.
        again,_=s.select('p_teacher_packet',*args,packet_teacher=z*3)
        self.assertTrue(torch.equal(p,again))
    def test_delayed_reset_and_reorder(self):
        t=transport_mod.CameraTransport(transport_mod.CameraTransportConfig((10,10),(20,20),0),.01,7)
        t.capture(1,(torch.tensor([0,0]),torch.ones(2,32)*3))
        self.assertEqual(t.receive(2),[])
        packet=t.receive(3)[0];captured,payload=packet.payload
        valid=transport_mod.delivery_mask(packet.capture_step,captured,torch.tensor([1,0]),torch.tensor([-1,-1]))
        self.assertEqual(valid.tolist(),[False,True])
        self.assertFalse(transport_mod.delivery_mask(1,captured,captured,torch.tensor([2,2])).any())
    def test_gain_freeze_preserves_initial_and_disables_resampling(self):
        from types import SimpleNamespace as N
        from gain_control import freeze_gain_resampling
        initial={'stiffness':123.,'damping':2.4}
        cfg=N(mode='reset',func=lambda env,ids,**kw:initial.update(stiffness=999.),params={})
        manager=N(get_term_cfg=lambda name:cfg,set_term_cfg=lambda name,value:setattr(manager,'active',value))
        b=N(scene={'robot':N(actuators={'hip':initial})},event_manager=manager)
        result=freeze_gain_resampling(b);manager.active.func(b,[0])
        self.assertEqual(initial['stiffness'],123.)
        self.assertIsNot(manager.active,cfg)
        self.assertEqual(result['mode'],'seeded_initial_gains_frozen_after_initial_reset')
    def test_drop(self):
        t=transport_mod.CameraTransport(transport_mod.CameraTransportConfig((10,10),(0,0),.999999),.01,7)
        t.capture(1,torch.ones(1,32));self.assertEqual(t.receive(1),[]);self.assertEqual(t.dropped,1)

if __name__=='__main__':unittest.main()
